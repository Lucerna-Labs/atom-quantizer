//! Atom Quantizer — adaptive Q2/Q4/Q8 weight codec research for LLM GGUFs.
//!
//! Applies the KV-cache codec technique (adaptive 2/4/8-bit per-block quant +
//! blind RF-repair + integrity tag) to the weight tensors of a GGUF model.
//! The strategy picker (Q2 → Q4 → Q8) uses a dual gate: KL divergence over
//! softmax(W·x), plus a cosine safety floor. One-dimensional tensors use a
//! histogram-KL proxy. This is an under-development research codec.
//!
//! In-RAM by default. `--out model.oq` streams the compressed weights to a
//! real container file on disk, which is then re-read to verify every tensor
//! decodes (integrity + reversibility check).
//!
//! Usage:
//!   atom-quantizer <model.gguf> [--limit N] [--key 0x..] [--out model.oq] [--asym]
//!   atom-quantizer compare <a.gguf> <b.gguf>
//!
//! Build with `cargo build --release --features cuda` to route the KL matvec
//! through cuBLAS (recommended for anything above ~200 M params). Default build
//! uses CPU matmul via `metrics::cpu_matmul_batch` — correct but slower.

#[cfg(feature = "cuda")]
use atom_quantizer::cuda_backend;
use atom_quantizer::encoded::{EncodedTensor, Payload, Transform};
use atom_quantizer::{a22, allocation, calibration};
use atom_quantizer::{gguf, metrics, oq, stacks, wq};

use wq::{cosine, encode_with, WqConfig};

mod assess;
mod build_functional;
mod packed_apply;

/// Tensors whose error compounds most (embeddings, output projection, norms,
/// biases) hold a higher fidelity bar: the KL ceiling is stricter for them.
fn is_protected(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("token_embd")
        || n.contains("output.weight")
        || n.contains("lm_head")
        || n.contains("norm")
        || n.contains("bias")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncodingMode {
    OptimizedQ2,
    Q2,
    Q4,
    Q8,
    Exact,
}

struct Candidate {
    mode: EncodingMode,
    tensor: EncodedTensor,
    fwd: f64,
    rev: f64,
    cos: f32,
    output_mse: Option<f64>,
    distortion: f64,
    bytes: u64,
    checksum: u64,
}

struct Reference {
    samples: Option<calibration::Samples>,
    ys: Vec<Vec<f32>>,
    ps: Vec<Vec<f32>>,
    calibrated: bool,
}

#[derive(Clone, Copy)]
struct FidelityLimits {
    kl: f64,
    cosine: f32,
    output_mse: f64,
}

impl FidelityLimits {
    fn new(protected: bool, max_output_mse: f64) -> Self {
        Self {
            kl: if protected { 0.25 } else { 1.0 },
            cosine: if protected { 0.995 } else { 0.99 },
            output_mse: max_output_mse * if protected { 0.25 } else { 1.0 },
        }
    }
}

struct Fidelity {
    fwd: f64,
    rev: f64,
    cosine: f32,
    output_mse: Option<f64>,
}

struct GateResults {
    kl: bool,
    cosine: bool,
    output_mse: bool,
}

impl GateResults {
    fn accepted(&self) -> bool {
        self.kl && self.cosine && self.output_mse
    }
}

impl Fidelity {
    fn gates(&self, calibrated: bool, limits: FidelityLimits) -> GateResults {
        GateResults {
            kl: self.fwd.is_finite() && self.rev.is_finite() && self.fwd.max(self.rev) <= limits.kl,
            // Finite weights of equal length yield a finite cosine; keep the
            // quantizer's existing comparison and f32 threshold precision.
            cosine: self.cosine >= limits.cosine,
            output_mse: !calibrated || self.output_mse.is_some_and(|m| m <= limits.output_mse),
        }
    }
}

#[derive(Clone)]
struct QuantizeOptions {
    calibration: Option<String>,
    budget_bytes: Option<u64>,
    max_output_mse: f64,
    optimize_q2: bool,
}
impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            calibration: None,
            budget_bytes: None,
            max_output_mse: 0.02,
            optimize_q2: true,
        }
    }
}

fn operator_outputs(
    weights: &[f32],
    samples: &calibration::Samples,
    rows: usize,
    cols: usize,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> Vec<Vec<f32>> {
    match samples {
        calibration::Samples::Embedding(ids) => ids
            .iter()
            .map(|&i| weights[i * cols..(i + 1) * cols].to_vec())
            .collect(),
        calibration::Samples::Linear(xs) => {
            #[cfg(feature = "cuda")]
            if let Some(gpu) = gpu {
                if let Ok(ys) = gpu.matmul_batch(weights, xs, rows, cols) {
                    return ys;
                }
            }
            metrics::cpu_matmul_batch(weights, xs, rows, cols)
        }
    }
}

fn make_reference(
    weights: &[f32],
    meta: &gguf::TensorInfo,
    index: usize,
    calibration: Option<&calibration::Calibration>,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> Result<Reference, String> {
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(format!("nonfinite source tensor {}", meta.name));
    }
    let cols = meta.row_len.max(1);
    let rows = weights.len() / cols;
    if rows <= 1 || cols <= 1 {
        return Ok(Reference {
            samples: None,
            ys: Vec::new(),
            ps: Vec::new(),
            calibrated: false,
        });
    }
    let samples = if let Some(cal) = calibration {
        let samples = cal
            .tensors
            .get(&meta.name)
            .ok_or_else(|| format!("missing real calibration for {}", meta.name))?;
        samples.validate_for(meta)?;
        samples.clone()
    } else {
        calibration::Samples::Linear(metrics::make_gaussian_inputs(
            cols,
            4,
            0xB4_5E_D0_C0u32.wrapping_add(index as u32),
        ))
    };
    let ys = operator_outputs(
        weights,
        &samples,
        rows,
        cols,
        #[cfg(feature = "cuda")]
        gpu,
    );
    let ps = metrics::softmax_batch(&ys);
    Ok(Reference {
        samples: Some(samples),
        ys,
        ps,
        calibrated: calibration.is_some(),
    })
}

fn measure_fidelity(
    source: &[f32],
    reconstructed: &[f32],
    meta: &gguf::TensorInfo,
    reference: &Reference,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> Result<Fidelity, String> {
    if source.len() != reconstructed.len()
        || source.is_empty()
        || reconstructed.iter().any(|v| !v.is_finite())
    {
        return Err(format!("invalid reconstructed weights for {}", meta.name));
    }
    let cols = meta.row_len.max(1);
    let rows = source.len() / cols;
    let (fwd, rev, output_mse) = if let Some(samples) = &reference.samples {
        let ys = operator_outputs(
            reconstructed,
            samples,
            rows,
            cols,
            #[cfg(feature = "cuda")]
            gpu,
        );
        let (fwd, rev) = metrics::bidirectional_kl_cached(&reference.ps, &ys);
        (
            fwd,
            rev,
            Some(metrics::relative_output_mse(&reference.ys, &ys)?),
        )
    } else {
        let (fwd, rev) = metrics::kl_histogram_bidirectional(source, reconstructed, 256);
        (fwd, rev, None)
    };
    Ok(Fidelity {
        fwd,
        rev,
        cosine: cosine(source, reconstructed),
        output_mse,
    })
}

#[allow(clippy::too_many_arguments)]
fn candidates_for_tensor(
    data: &[f32],
    meta: &gguf::TensorInfo,
    cfg: &WqConfig,
    reference: &Reference,
    options: &QuantizeOptions,
    all: bool,
    only: Option<EncodingMode>,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> Result<Vec<Candidate>, String> {
    let cols = meta.row_len.max(1);
    let rows = data.len() / cols;
    let is_2d = rows > 1 && cols > 1;
    let transform = if is_2d && cols >= 8 && cols.is_multiple_of(8) {
        Transform::RowHadamard8
    } else {
        Transform::None
    };
    let mut encoded_input = data.to_vec();
    if transform == Transform::RowHadamard8 {
        stacks::row_hadamard_8_apply(&mut encoded_input, rows, cols);
    }
    let row_norms = if is_2d {
        stacks::row_l2_norms(data, rows, cols)
    } else {
        Vec::new()
    };
    let importance = if reference.calibrated {
        reference
            .samples
            .as_ref()
            .and_then(|s| s.column_importance(transform))
    } else {
        None
    };
    let modes = [
        EncodingMode::OptimizedQ2,
        EncodingMode::Q2,
        EncodingMode::Q4,
        EncodingMode::Q8,
        EncodingMode::Exact,
    ];
    let mut accepted = Vec::new();
    let limits = FidelityLimits::new(cfg.protect, options.max_output_mse);
    let energy: f64 = data.iter().map(|v| (*v as f64).powi(2)).sum();
    for mode in modes {
        if only.is_some_and(|selected| selected != mode) {
            continue;
        }
        if mode == EncodingMode::OptimizedQ2 && !options.optimize_q2 {
            continue;
        }
        let mut tensor = if mode == EncodingMode::Exact {
            EncodedTensor::exact(data.to_vec(), meta.dims.clone())
        } else {
            let c = match mode {
                EncodingMode::OptimizedQ2 => {
                    if cfg.asym {
                        wq::encode_q2_optimized_asym(&encoded_input, cfg.key, importance.as_deref())
                    } else {
                        wq::encode_q2_optimized(&encoded_input, cfg.key, importance.as_deref())
                    }
                }
                EncodingMode::Q2 => encode_with(&encoded_input, wq::QuantStrategy::Q2, cfg),
                EncodingMode::Q4 => encode_with(&encoded_input, wq::QuantStrategy::Q4, cfg),
                EncodingMode::Q8 => encode_with(&encoded_input, wq::QuantStrategy::Q8, cfg),
                EncodingMode::Exact => unreachable!(),
            };
            EncodedTensor {
                payload: Payload::Blocks(c),
                dims: meta.dims.clone(),
                transform,
                row_norms: row_norms.clone(),
                output_type: atom_quantizer::encoded::storage_type_for_source(meta.ggml_type),
            }
        };
        tensor.output_type = atom_quantizer::encoded::storage_type_for_source(meta.ggml_type);
        let record = match oq::encode_record(&meta.name, meta.ggml_type, &tensor) {
            Ok(record) => record,
            Err(_) if mode != EncodingMode::Exact => continue,
            Err(e) => return Err(e),
        };
        let bytes = record.len() as u64;
        let checksum = oq::checksum(&record[..record.len() - 8]);
        let tensor = oq::decode_record(&record)?.tensor;
        let recon = match tensor.decode() {
            Ok(recon) => recon,
            Err(_) if mode != EncodingMode::Exact => continue,
            Err(e) => return Err(e),
        };
        let fidelity = measure_fidelity(
            data,
            &recon,
            meta,
            reference,
            #[cfg(feature = "cuda")]
            gpu,
        )?;
        if !fidelity.gates(reference.calibrated, limits).accepted() {
            continue;
        }
        let (fwd, rev, cos, output_mse) = (
            fidelity.fwd,
            fidelity.rev,
            fidelity.cosine,
            fidelity.output_mse,
        );
        let error: f64 = data
            .iter()
            .zip(&recon)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum();
        let weight_mse = if energy > 0.0 {
            error / energy
        } else if error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        let distortion = if reference.calibrated {
            output_mse.unwrap()
        } else {
            weight_mse
        };
        if !distortion.is_finite() {
            continue;
        }
        accepted.push(Candidate {
            mode,
            tensor,
            fwd,
            rev,
            cos,
            output_mse,
            distortion,
            bytes,
            checksum,
        });
        if !all {
            break;
        }
    }
    if accepted.is_empty() {
        return Err(format!("no acceptable candidate for {}", meta.name));
    }
    Ok(accepted)
}

#[derive(Clone, Copy)]
struct PlannedChoice {
    mode: EncodingMode,
    checksum: u64,
    bytes: u64,
}

#[allow(clippy::too_many_arguments)]
fn plan_global_allocation(
    g: &mut gguf::Gguf,
    source_path: &str,
    limit: usize,
    key: u64,
    asym: bool,
    options: &QuantizeOptions,
    calibration: &calibration::Calibration,
    header_bytes: u64,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> Result<std::collections::HashMap<String, PlannedChoice>, String> {
    let budget = options.budget_bytes.ok_or("missing global budget")?;
    let record_budget = budget
        .checked_sub(header_bytes)
        .ok_or("budget is smaller than model metadata")?;
    let metas = g.tensors.clone();
    let mut names = Vec::new();
    let mut descriptions = Vec::new();
    let mut groups = Vec::new();
    for (index, meta) in metas.iter().enumerate() {
        if limit > 0 && names.len() >= limit {
            break;
        }
        if !gguf::dequantizable(meta.ggml_type) {
            continue;
        }
        let w = gguf::read_tensor_f32(g, index)?;
        let reference = make_reference(
            &w,
            meta,
            index,
            Some(calibration),
            #[cfg(feature = "cuda")]
            gpu,
        )?;
        let cfg = WqConfig {
            protect: is_protected(&meta.name),
            key,
            asym,
        };
        let candidates = candidates_for_tensor(
            &w,
            meta,
            &cfg,
            &reference,
            options,
            true,
            None,
            #[cfg(feature = "cuda")]
            gpu,
        )?;
        groups.push(
            candidates
                .iter()
                .map(|c| allocation::Choice {
                    bytes: c.bytes,
                    distortion: c.distortion,
                })
                .collect(),
        );
        descriptions.push(
            candidates
                .iter()
                .map(|c| PlannedChoice {
                    mode: c.mode,
                    checksum: c.checksum,
                    bytes: c.bytes,
                })
                .collect::<Vec<_>>(),
        );
        names.push(meta.name.clone());
        if names.len().is_multiple_of(20) {
            eprintln!("budget pass: measured {} tensors", names.len());
        }
    }
    let chosen = allocation::allocate(&groups, record_budget)?;
    if calibration::file_sha256(source_path)? != calibration.source_sha256 {
        return Err("source GGUF changed during budget planning".into());
    }
    println!("budget plan: {} bytes including metadata / {} allowed; minimum feasible {} bytes; surrogate distortion {:.6}",
        chosen.bytes + header_bytes, budget, chosen.minimum_bytes + header_bytes, chosen.distortion);
    Ok(names
        .into_iter()
        .enumerate()
        .map(|(i, name)| (name, descriptions[i][chosen.choices[i]]))
        .collect())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--version") {
        println!("atom-quantizer {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    if args.get(1).map(String::as_str) == Some("build-functional") {
        match build_functional::run(&args[2..]) {
            Ok(report) => println!("{}", report),
            Err(error) => fail(&format!("functional build failed: {error}")),
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("apply-packed") {
        match packed_apply::run(&args[2..]) {
            Ok(report) => println!("{report}"),
            Err(error) => fail(&format!("packed operation failed: {error}")),
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("assess") {
        match assess::run(&args[2..]) {
            Ok(report) => {
                use std::io::Write;
                writeln!(std::io::stdout().lock(), "{}", report.json).unwrap_or_else(|error| {
                    fail(&format!("cannot write assessment report: {error}"))
                });
                if !report.accepted {
                    std::process::exit(2);
                }
            }
            Err(error) => fail(&error),
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("decode") {
        if args.len() != 5 || args[3] != "--out" {
            eprintln!("usage: atom-quantizer decode <model.oq> --out <new-model.gguf>");
            std::process::exit(2);
        }
        let is_a22 = match a22::is_archive(&args[2]) {
            Ok(value) => value,
            Err(error) => fail(&error),
        };
        if is_a22 {
            match a22::decode_to_gguf(&args[2], &args[4]) {
                Ok(stats) => println!(
                    "decoded {} committed A22-2 tensors natively to {}",
                    stats.tensors, args[4]
                ),
                Err(error) => fail(&format!("A22 decode failed: {error}")),
            }
            return;
        }
        match oq::decode_to_gguf(&args[2], &args[4]) {
            Ok(count) => println!("decoded {count} tensors from OQ03 to {}", args[4]),
            Err(e) => {
                eprintln!("decode failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.get(1).map(String::as_str) == Some("verify") {
        if args.len() != 3 {
            eprintln!("usage: atom-quantizer verify <model.oq>");
            std::process::exit(2);
        }
        let is_a22 = match a22::is_archive(&args[2]) {
            Ok(value) => value,
            Err(error) => fail(&error),
        };
        if is_a22 {
            match a22::verify_file(&args[2]) {
                Ok(stats) => println!("verified and natively decoded {} A22-2 tensors ({} archive bytes; {} decoded bytes)", stats.tensors, stats.artifact_bytes, stats.decoded_bytes),
                Err(error) => fail(&format!("A22 verification failed: {error}")),
            }
            return;
        }
        match oq::verify_file(&args[2]) {
            Ok(stats) => {
                println!(
                    "verified and decoded {} tensors ({} bytes)",
                    stats.count, stats.file_bytes
                );
                if !stats.full_record_integrity {
                    println!("legacy OQ02: only codes are checksummed; missing transforms/norms cannot be recovered");
                }
            }
            Err(e) => {
                eprintln!("verification failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    if args.len() < 2 || matches!(args.get(1).map(String::as_str), Some("--help" | "-h")) {
        eprintln!("Atom Quantizer — adaptive weight codec research\n");
        eprintln!("usage: atom-quantizer <model.gguf> [--limit N] [--key 0xHEX] [--out model.oq] [--asym]");
        eprintln!("       atom-quantizer compare <a.gguf> <b.gguf>");
        eprintln!("       atom-quantizer build-functional <source.gguf> --base-calibration BASE.acal --residual-calibration RESIDUAL.acal --out NEW.a22");
        eprintln!(
            "       atom-quantizer assess <source.gguf> <candidate.gguf> --calibration INPUT.acal"
        );
        eprintln!("       atom-quantizer decode <model.oq> --out <new-model.gguf>");
        eprintln!("       atom-quantizer verify <model.oq>");
        eprintln!("       atom-quantizer apply-packed <model.a22> --tensor NAME --input F32_FILE --batch N --out NEW_F32_FILE");
        eprintln!("       decode/verify also support committed A22-2 scalar and rank-two recipes");
        eprintln!("       --calibration INPUT.acal     source-bound operator samples");
        eprintln!("       --q2-scale mse|maxabs        scale fit (default: mse)");
        eprintln!(
            "       --max-output-mse VALUE      calibrated relative MSE ceiling (default: 0.02)"
        );
        eprintln!(
            "       --budget-bytes N            whole-artifact byte budget; requires calibration"
        );
        eprintln!("       --budget-mib N              same budget in MiB");
        eprintln!(
            "       --out-gguf NEW.gguf          export a complete dense model from saved records"
        );
        std::process::exit(if args.len() < 2 { 2 } else { 0 });
    }

    if args.get(1).map(String::as_str) == Some("compare") {
        if args.len() < 4 {
            eprintln!("usage: atom-quantizer compare <a.gguf> <b.gguf>");
            std::process::exit(2);
        }
        run_compare(&args[2], &args[3]);
        return;
    }

    // Autonomous stack search: iterate rounds of candidate stacks, report
    // aggregate cos + KL + effective bits/w per stack on the full model.
    if args.get(1).map(String::as_str) == Some("stack-search") {
        if args.len() < 3 {
            eprintln!("usage: atom-quantizer stack-search <model.gguf> [--round N] [--limit N]");
            std::process::exit(2);
        }
        let path = args[2].clone();
        let mut round: usize = 1;
        let mut limit = 0usize;
        let mut i = 3;
        while i < args.len() {
            match args[i].as_str() {
                "--round" => {
                    i += 1;
                    round = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
                }
                "--limit" => {
                    i += 1;
                    limit = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
                }
                other => eprintln!("(ignoring unknown arg {other})"),
            }
            i += 1;
        }
        run_stack_search(&path, round, limit);
        return;
    }

    let path = args[1].clone();
    let mut limit = 0usize;
    let mut key: u64 = WqConfig::default().key;
    let mut out_path: Option<String> = None;
    let mut out_gguf: Option<String> = None;
    let mut use_asym = false;
    let mut options = QuantizeOptions::default();
    let mut output_gate_requested = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--asym" => use_asym = true,
            "--calibration" => {
                i += 1;
                options.calibration = Some(
                    args.get(i)
                        .unwrap_or_else(|| fail("--calibration needs a path"))
                        .clone(),
                );
            }
            "--budget-bytes" => {
                if options.budget_bytes.is_some() {
                    fail("specify one byte budget");
                }
                i += 1;
                options.budget_bytes = Some(
                    args.get(i)
                        .and_then(|v| v.parse::<u64>().ok())
                        .filter(|n| *n > 0)
                        .unwrap_or_else(|| fail("--budget-bytes needs a positive integer")),
                );
            }
            "--budget-mib" => {
                if options.budget_bytes.is_some() {
                    fail("specify one byte budget");
                }
                i += 1;
                let mib = args
                    .get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v > 0.0 && *v < u64::MAX as f64 / 1048576.0)
                    .unwrap_or_else(|| fail("--budget-mib needs a finite positive size"));
                options.budget_bytes = Some((mib * 1048576.0).floor() as u64);
            }
            "--max-output-mse" => {
                output_gate_requested = true;
                i += 1;
                options.max_output_mse = args
                    .get(i)
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or_else(|| fail("--max-output-mse needs a finite positive value"));
            }
            "--q2-scale" => {
                i += 1;
                options.optimize_q2 = match args.get(i).map(String::as_str) {
                    Some("mse") => true,
                    Some("maxabs") => false,
                    _ => fail("--q2-scale must be mse or maxabs"),
                };
            }
            "--limit" => {
                i += 1;
                limit = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| fail("--limit needs an integer"));
            }
            "--key" => {
                i += 1;
                let value = args
                    .get(i)
                    .unwrap_or_else(|| fail("--key needs an integer"));
                key = if let Some(hex) = value
                    .strip_prefix("0x")
                    .or_else(|| value.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16)
                } else {
                    value.parse::<u64>()
                }
                .unwrap_or_else(|_| fail("invalid --key"));
            }
            "--out" => {
                i += 1;
                out_path = Some(
                    args.get(i)
                        .unwrap_or_else(|| fail("--out needs a path"))
                        .clone(),
                );
            }
            "--out-gguf" => {
                i += 1;
                out_gguf = Some(
                    args.get(i)
                        .unwrap_or_else(|| fail("--out-gguf needs a path"))
                        .clone(),
                );
            }
            other => fail(&format!("unknown quantizer argument {other}")),
        }
        i += 1;
    }

    if output_gate_requested && options.calibration.is_none() {
        fail("--max-output-mse requires --calibration");
    }
    run_quantize(&path, limit, key, out_path, out_gguf, use_asym, options);
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}

fn run_quantize(
    path: &str,
    limit: usize,
    key: u64,
    out_path: Option<String>,
    out_gguf: Option<String>,
    use_asym: bool,
    options: QuantizeOptions,
) {
    let mut destinations = std::collections::HashSet::new();
    for destination in out_path.iter().chain(out_gguf.iter()) {
        let p = std::path::Path::new(destination);
        if p.symlink_metadata().is_ok() {
            fail(&format!("destination already exists: {destination}"));
        }
        let parent = p
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new("."));
        let parent = parent
            .canonicalize()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let normalized = parent.join(p.file_name().unwrap_or_else(|| fail("invalid output path")));
        if !destinations.insert(normalized) {
            fail("--out and --out-gguf must use different paths");
        }
    }

    let mut g = match gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let total_params: u128 = g.tensors.iter().map(|t| t.n_elems as u128).sum();
    println!("┌─ Atom Quantizer ─────────────────────────────────────────────");
    println!("│ model      {path}");
    println!(
        "│ arch       {}",
        if g.arch.is_empty() { "?" } else { &g.arch }
    );
    println!("│ tensors    {}", g.tensors.len());
    println!("│ params     {:.3} B", total_params as f64 / 1e9);
    println!(
        "├─ {}",
        if options.budget_bytes.is_some() {
            "global byte-budget allocation across accepted candidates"
        } else {
            "most aggressive candidate that clears the acceptance gates"
        }
    );
    println!(
        "│ {:<34} {:>5} {:>11}  {:<3} {:>7}  {:>9} {:>9}  {:>11}",
        "tensor", "type", "elems", "→", "ratio", "KL_fwd", "KL_rev", "cos"
    );

    let metas: Vec<(String, u32, usize, usize)> = g
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type, t.n_elems, t.row_len))
        .collect();

    // Dual gate: KL ceiling AND cosine floor. Both must clear.
    //
    // Empirical: KL alone let Q2 through on tensors whose weight cosine was 0.81,
    // and the model hallucinated in LM Studio. The cosine floor is the safety net.
    let normal_limits = FidelityLimits::new(false, options.max_output_mse);
    let protected_limits = FidelityLimits::new(true, options.max_output_mse);
    let kl_ceiling_normal = normal_limits.kl;
    let kl_ceiling_protected = protected_limits.kl;
    let cosine_floor_normal = normal_limits.cosine;
    let cosine_floor_protected = protected_limits.cosine;
    let calibration = options
        .calibration
        .as_deref()
        .map(|p| calibration::Calibration::load(p, path))
        .transpose()
        .unwrap_or_else(|e| fail(&e));
    if options.budget_bytes.is_some() && calibration.is_none() {
        fail("global allocation requires --calibration with real operator inputs");
    }
    if let Some(cal) = &calibration {
        for meta in g
            .tensors
            .iter()
            .filter(|t| gguf::dequantizable(t.ggml_type))
            .take(if limit > 0 { limit } else { usize::MAX })
        {
            if meta.row_len > 1 && meta.n_elems / meta.row_len > 1 {
                cal.tensors
                    .get(&meta.name)
                    .unwrap_or_else(|| fail(&format!("missing real calibration for {}", meta.name)))
                    .validate_for(meta)
                    .unwrap_or_else(|e| fail(&e));
            }
        }
        println!(
            "│ calibration: {} supplied operator batches; output MSE ceiling {} (protected {})",
            cal.tensors.len(),
            options.max_output_mse,
            options.max_output_mse * 0.25
        );
    }

    #[cfg(feature = "cuda")]
    let gpu = cuda_backend::CudaMatvec::new().ok();
    #[cfg(feature = "cuda")]
    println!(
        "│ KL matvec  {}",
        if gpu.is_some() {
            "GPU (cuBLAS)"
        } else {
            "CPU fallback"
        }
    );
    #[cfg(not(feature = "cuda"))]
    let _gpu: Option<()> = None;
    #[cfg(not(feature = "cuda"))]
    println!("│ KL matvec  CPU (build with --features cuda for cuBLAS)");
    println!(
        "│ KL ceiling  normal={} protected={}",
        kl_ceiling_normal, kl_ceiling_protected
    );
    println!(
        "│ cos floor   normal={} protected={}",
        cosine_floor_normal, cosine_floor_protected
    );

    if out_path.is_some() && limit == 0 {
        if let Some(t) = g.tensors.iter().find(|t| !gguf::dequantizable(t.ggml_type)) {
            eprintln!(
                "error: --out needs every tensor dequantizable, but '{}' is {} (that K-quant isn't wired yet). Refusing to write a partial model.",
                t.name,
                gguf::type_name(t.ggml_type)
            );
            std::process::exit(1);
        }
    }
    let model_header = gguf::model_header(&mut g).unwrap_or_else(|e| fail(&e));
    let header_bytes = model_header.len() as u64 + 24;
    let planned = if options.budget_bytes.is_some() {
        plan_global_allocation(
            &mut g,
            path,
            limit,
            key,
            use_asym,
            &options,
            calibration.as_ref().unwrap(),
            header_bytes,
            #[cfg(feature = "cuda")]
            gpu.as_ref(),
        )
        .unwrap_or_else(|e| fail(&e))
    } else {
        std::collections::HashMap::new()
    };
    if out_gguf.is_some() {
        if limit > 0 && limit < g.tensors.len() {
            fail(
                "--out-gguf requires a complete model; use --out alone for a limited tensor sample",
            );
        }
        if let Some(meta) = g
            .tensors
            .iter()
            .find(|t| !gguf::writable_passthrough(t.ggml_type))
        {
            fail(&format!(
                "GGUF export cannot encode source dtype {} for {}",
                gguf::type_name(meta.ggml_type),
                meta.name
            ));
        }
    }
    let temporary_artifact = if out_path.is_none() {
        out_gguf.as_ref().map(|p| {
            let p = std::path::Path::new(p);
            p.with_file_name(format!(
                ".{}.{}.oq",
                p.file_name().unwrap().to_string_lossy(),
                std::process::id()
            ))
            .to_string_lossy()
            .into_owned()
        })
    } else {
        None
    };
    let container_path = out_path.as_ref().or(temporary_artifact.as_ref());
    let mut writer = container_path
        .map(|p| oq::Writer::create_with_header(p, model_header))
        .transpose()
        .unwrap_or_else(|e| fail(&e));

    let (mut orig_bytes, mut comp_bytes, mut count) = (0u128, header_bytes as u128, 0usize);
    let (mut kl_fwd_sum, mut kl_rev_sum) = (0f64, 0f64);
    let (mut n_q2, mut n_q4, mut n_q8, mut n_f32) = (0usize, 0usize, 0usize, 0usize);
    let mut max_kl: (f64, String) = (0.0, String::new());
    let mut all_verified = true;
    let mut ceiling_misses = 0usize;
    let mut skipped: Vec<(String, u32)> = Vec::new();

    for (idx, (name, ttype, n_elems, row_len)) in metas.iter().enumerate() {
        if !gguf::dequantizable(*ttype) {
            skipped.push((name.clone(), *ttype));
            continue;
        }
        if limit > 0 && count >= limit {
            break;
        }
        let w = match gguf::read_tensor_f32(&mut g, idx) {
            Ok(w) => w,
            Err(e) => {
                fail(&format!("cannot read tensor {name}: {e}"));
            }
        };
        if w.iter().any(|v| !v.is_finite()) {
            eprintln!("error: tensor {name} contains nonfinite source values");
            std::process::exit(1);
        }
        let protect = is_protected(name);
        let kl_ceiling = if protect {
            kl_ceiling_protected
        } else {
            kl_ceiling_normal
        };
        let cos_floor = if protect {
            cosine_floor_protected
        } else {
            cosine_floor_normal
        };
        let cfg = WqConfig {
            protect,
            key,
            asym: use_asym,
        };

        let _ = row_len;
        let reference = make_reference(
            &w,
            &g.tensors[idx],
            idx,
            calibration.as_ref(),
            #[cfg(feature = "cuda")]
            gpu.as_ref(),
        )
        .unwrap_or_else(|e| fail(&e));
        let plan = if options.budget_bytes.is_some() {
            Some(
                planned
                    .get(name)
                    .unwrap_or_else(|| fail("tensor absent from global allocation")),
            )
        } else {
            None
        };
        let candidate = candidates_for_tensor(
            &w,
            &g.tensors[idx],
            &cfg,
            &reference,
            &options,
            false,
            plan.map(|p| p.mode),
            #[cfg(feature = "cuda")]
            gpu.as_ref(),
        )
        .unwrap_or_else(|e| fail(&e))
        .remove(0);
        if let Some(plan) = plan {
            if candidate.checksum != plan.checksum || candidate.bytes != plan.bytes {
                fail(&format!(
                    "candidate changed between allocation and encoding for {name}"
                ));
            }
        }
        let (kl_fwd, kl_rev, cos, output_mse, record_bytes) = (
            candidate.fwd,
            candidate.rev,
            candidate.cos,
            candidate.output_mse,
            candidate.bytes,
        );
        let c = candidate.tensor;
        let worst_kl = kl_fwd.max(kl_rev);
        let kl_miss = worst_kl > kl_ceiling;
        let cos_miss = (cos as f64) < (cos_floor as f64);
        if kl_miss || cos_miss {
            ceiling_misses += 1;
        }
        all_verified &= c.validate().is_ok();

        if let Some(wr) = writer.as_mut() {
            if let Err(e) = wr.append_encoded(name, *ttype, &c) {
                eprintln!("error writing tensor {name}: {e}");
                std::process::exit(1);
            }
        }

        let ob = (w.len() * 4) as u128;
        let cb = record_bytes as u128;
        orig_bytes += ob;
        comp_bytes += cb;
        kl_fwd_sum += kl_fwd;
        kl_rev_sum += kl_rev;
        if worst_kl > max_kl.0 {
            max_kl = (worst_kl, name.clone());
        }
        count += 1;
        match c.strategy() {
            Some(wq::QuantStrategy::Q2) => n_q2 += 1,
            Some(wq::QuantStrategy::Q4) => n_q4 += 1,
            Some(wq::QuantStrategy::Q8) => n_q8 += 1,
            None => n_f32 += 1,
        }
        let short = shorten(name, 34);
        let mark = if protect { "*" } else { " " };
        let miss = if kl_miss || cos_miss { "!" } else { " " };
        println!(
            "│{mark}{short:<34} {:>5} {:>11}  {:<3} {:>6.1}x  {miss}{:>8.4} {:>9.4}  cos={:>7.4} output_mse={}",
            gguf::type_name(*ttype),
            n_elems,
            c.strategy_name(),
            ob as f64 / cb.max(1) as f64,
            kl_fwd,
            kl_rev,
            cos,
            output_mse.map(|m| format!("{m:.6}")).unwrap_or_else(|| "n/a".into()),
        );
    }

    println!("└──────────────────────────────────────────────────────────────");
    if count == 0 {
        println!("No dequantizable (F32/F16/BF16/Q8_0/Q4_K/Q6_K) tensors were processed.");
        if !skipped.is_empty() {
            let t = skipped[0].1;
            println!(
                "This model stores weights as {} — that K-quant isn't wired yet.",
                gguf::type_name(t)
            );
        }
        return;
    }
    let ratio = orig_bytes as f64 / comp_bytes.max(1) as f64;
    println!("processed {count} tensors  |  strategy mix: Q2={n_q2} Q4={n_q4} Q8={n_q8} F32={n_f32}  (* = protected)");
    println!(
        "dense f32 {:.1} MB  →  Atom Quantizer {:.1} MB   ({ratio:.2}× smaller)",
        orig_bytes as f64 / 1e6,
        comp_bytes as f64 / 1e6,
    );
    let mean_fwd = kl_fwd_sum / count as f64;
    let mean_rev = kl_rev_sum / count as f64;
    let asym = if mean_rev > 1e-12 {
        mean_fwd / mean_rev
    } else {
        f64::INFINITY
    };
    println!(
        "mean KL_fwd: {:.4}   mean KL_rev: {:.4}   fwd/rev asymmetry: {:.3}",
        mean_fwd, mean_rev, asym
    );
    println!(
        "currency: {}; histogram KL for 1-D; original cosine floors retained",
        if calibration.is_some() {
            "real operator output MSE plus bidirectional KL"
        } else {
            "Gaussian output probes (uncalibrated diagnostics)"
        }
    );
    println!(
        "worst-tensor max(KL_fwd, KL_rev): {:.4}  ({})",
        max_kl.0,
        shorten(&max_kl.1, 40)
    );
    if ceiling_misses > 0 {
        fail(&format!(
            "internal error: {ceiling_misses} accepted tensors failed the gates"
        ));
    }
    println!(
        "integrity: {}",
        if all_verified {
            "all tensor records verified \u{2713}"
        } else {
            "FAILED \u{2717}"
        }
    );
    if !skipped.is_empty() {
        println!(
            "skipped {} already-K-quantized tensors (dequant not wired): e.g. {}",
            skipped.len(),
            skipped[0].0
        );
    }

    if let Some(cal) = &calibration {
        if calibration::file_sha256(path).unwrap_or_else(|e| fail(&e)) != cal.source_sha256 {
            fail("source GGUF changed during encoding");
        }
    }
    if let Some(budget) = options.budget_bytes {
        if comp_bytes > budget as u128 {
            fail("encoded artifact exceeds the planned global budget");
        }
        println!("verified byte budget: {comp_bytes} / {budget}");
    }
    if let (Some(w), Some(p)) = (writer, container_path) {
        match w.finish() {
            Ok(n) => {
                let src = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                println!("├─ .oq writeback ──────────────────────────────────────────────");
                match oq::verify_file(p) {
                    Ok(st) => {
                        let vs_gguf = src as f64 / st.file_bytes.max(1) as f64;
                        println!("│ wrote {n} tensors → {p}");
                        println!("│ source GGUF on disk : {:>8.1} MB", src as f64 / 1e6);
                        println!(
                            "│ Atom Quantizer .oq         : {:>8.1} MB   ({vs_gguf:.2}× smaller than the GGUF file on disk)",
                            st.file_bytes as f64 / 1e6
                        );
                        println!(
                            "│ re-read verify      : {} ({} tensors)",
                            if st.all_verified {
                                "all integrity tags OK \u{2713}"
                            } else {
                                "FAILED \u{2717}"
                            },
                            st.count
                        );
                        if let Some((nm, ot, dec)) = &st.first {
                            println!(
                                "│ decoded off disk    : {} ({}, {dec} weights)",
                                shorten(nm, 30),
                                gguf::type_name(*ot)
                            );
                        }
                        if limit > 0 || !skipped.is_empty() {
                            println!("│ NOTE: partial model (--limit/skip) — not a complete loadable model.");
                        }
                    }
                    Err(e) => fail(&format!("artifact verification failed: {e}")),
                }
                println!("└──────────────────────────────────────────────────────────────");
            }
            Err(e) => fail(&format!("artifact publication failed: {e}")),
        }
    }

    if let Some(destination) = &out_gguf {
        let artifact = container_path.expect("GGUF export has a complete intermediate container");
        let result = oq::decode_to_gguf(artifact, destination);
        if let Some(temporary) = &temporary_artifact {
            let _ = std::fs::remove_file(temporary);
        }
        match result {
            Ok(n) => println!(
                "decoded {n} saved tensor records to {destination} using the stored GGUF metadata"
            ),
            Err(e) => fail(&format!("GGUF export failed: {e}")),
        }
    }
}

fn run_compare(pa: &str, pb: &str) {
    let mut ga = match gguf::open(pa) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error opening {pa}: {e}");
            std::process::exit(1);
        }
    };
    let mut gb = match gguf::open(pb) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error opening {pb}: {e}");
            std::process::exit(1);
        }
    };
    let bmap: std::collections::HashMap<String, (usize, u32, usize)> = gb
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.clone(), (i, t.ggml_type, t.n_elems)))
        .collect();
    let a_meta: Vec<(String, u32, usize)> = ga
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type, t.n_elems))
        .collect();

    println!("┌─ compare (cosine of a vs b, per matched tensor) ──────────────");
    println!("│ a  {pa}");
    println!("│ b  {pb}   ← reference");
    println!(
        "│ {:<34} {:>10}  {:<11} {:>7}",
        "tensor", "elems", "a→b type", "cosine"
    );

    let (mut cos_sum, mut count, mut worst) = (0f64, 0usize, (1.0f32, String::new()));
    let (mut q4_sum, mut q4_n) = (0f64, 0usize);
    let mut shown = 0;
    for (idx, (name, ta_type, na)) in a_meta.iter().enumerate() {
        let Some(&(bidx, tb_type, nb)) = bmap.get(name) else {
            continue;
        };
        if !gguf::dequantizable(*ta_type) || !gguf::dequantizable(tb_type) || *na != nb || *na == 0
        {
            continue;
        }
        let wa = match gguf::read_tensor_f32(&mut ga, idx) {
            Ok(w) => w,
            Err(_) => continue,
        };
        let wb = match gguf::read_tensor_f32(&mut gb, bidx) {
            Ok(w) => w,
            Err(_) => continue,
        };
        let cos = cosine(&wa, &wb);
        cos_sum += cos as f64;
        count += 1;
        if *ta_type == gguf::GGML_Q4_K {
            q4_sum += cos as f64;
            q4_n += 1;
        }
        if cos < worst.0 {
            worst = (cos, name.clone());
        }
        if shown < 14 {
            let types = format!("{}→{}", gguf::type_name(*ta_type), gguf::type_name(tb_type));
            println!(
                "│ {:<34} {:>10}  {:<11} {:>7.5}",
                shorten(name, 34),
                na,
                types,
                cos
            );
            shown += 1;
        }
    }
    println!("└──────────────────────────────────────────────────────────────");
    if count == 0 {
        println!("no comparable tensors (names/shapes/types didn't match or aren't dequantizable)");
        return;
    }
    println!(
        "matched {count} tensors  |  mean cosine (all): {:.5}",
        cos_sum / count as f64
    );
    if q4_n > 0 {
        println!(
            "of which {q4_n} are genuinely Q4_K in a  |  mean cosine (Q4_K-only): {:.5}",
            q4_sum / q4_n as f64
        );
    }
    println!(
        "worst tensor: {} at cos={:.5}",
        shorten(&worst.1, 40),
        worst.0
    );
}

fn shorten(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let keep = max - 1;
        let tail: String = s.chars().skip(s.chars().count() - keep).collect();
        format!("…{tail}")
    }
}

/// A named candidate stack for the autonomous search. Each stack is applied
/// via `stacks::apply` which honors the correct pipeline order:
/// PRE forward → EXTRACT (top-K aside) → QUANT → blind repair → outlier restore
/// → PRE inverse → optional row norms. Outlier restore MUST precede PRE inverse — see the
/// stacks module's tests for the invariant.
struct StackCandidate {
    name: &'static str,
    stack: stacks::Stack,
}

/// Curated stack candidates by search round. The rounds walk the composer
/// chain systematically: individual atoms first, then pair combinations,
/// then full-chain compositions.
fn candidates_for_round(round: usize) -> Vec<StackCandidate> {
    use stacks::{Extract, Post, Pre, Quantize, Stack};
    match round {
        1 => vec![
            StackCandidate {
                name: "baseline (sym-Q4)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "+preserve",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::PreserveRowNorm,
                },
            },
            StackCandidate {
                name: "+errfeed",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::ErrfeedPass,
                },
            },
            StackCandidate {
                name: "+wavelet",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::WaveletLift,
                },
            },
        ],
        2 => vec![
            StackCandidate {
                name: "baseline (sym-Q4)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC+sym-Q4",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowH8+sym-Q4",
                stack: Stack {
                    pre: Pre::RowHadamard8,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDCT8+sym-Q4",
                stack: Stack {
                    pre: Pre::DctRow8,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "haarMix8+sym-Q4",
                stack: Stack {
                    pre: Pre::HaarMix8,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
        ],
        3 => vec![
            StackCandidate {
                name: "sym-Q4 (baseline)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymUniform,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "asym-affine",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::AsymAffine,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "codebook-LM16",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::CodebookLloydMax16,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "sym-codebook",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SymmetricCodebook,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "superpose (S+A)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::SuperposeSymAsym,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "refract (b+t 30%)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "compose (nested)",
                stack: Stack {
                    pre: Pre::Null,
                    extract: Extract::Null,
                    quantize: Quantize::ComposeNestedScale,
                    post: Post::Null,
                },
            },
        ],
        4 => vec![
            StackCandidate {
                name: "rowDC + refract",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::Null,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDCT8 + refract",
                stack: Stack {
                    pre: Pre::DctRow8,
                    extract: Extract::Null,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC + codebook",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::Null,
                    quantize: Quantize::CodebookLloydMax16,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDCT8 + codebook",
                stack: Stack {
                    pre: Pre::DctRow8,
                    extract: Extract::Null,
                    quantize: Quantize::CodebookLloydMax16,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC + superpose",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::Null,
                    quantize: Quantize::SuperposeSymAsym,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC + compose",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::Null,
                    quantize: Quantize::ComposeNestedScale,
                    post: Post::Null,
                },
            },
        ],
        5 => vec![
            StackCandidate {
                name: "rowDC/top2/refract/preserve",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::OutlierTopK2,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::PreserveRowNorm,
                },
            },
            StackCandidate {
                name: "rowDC/top4/refract/preserve",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::OutlierTopK4,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::PreserveRowNorm,
                },
            },
            StackCandidate {
                name: "rowDC/top4/refract/-",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::OutlierTopK4,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC/top4/superpose/-",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::OutlierTopK4,
                    quantize: Quantize::SuperposeSymAsym,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDC/top4/codebook/-",
                stack: Stack {
                    pre: Pre::RowDcRemove,
                    extract: Extract::OutlierTopK4,
                    quantize: Quantize::CodebookLloydMax16,
                    post: Post::Null,
                },
            },
            StackCandidate {
                name: "rowDCT8/top4/refract/-",
                stack: Stack {
                    pre: Pre::DctRow8,
                    extract: Extract::OutlierTopK4,
                    quantize: Quantize::RefractBulkTail30,
                    post: Post::Null,
                },
            },
        ],
        _ => Vec::new(),
    }
}

#[derive(Default, Clone)]
struct StackAggregate {
    name: String,
    total_elems: u128,
    total_orig_bytes: u128,
    total_code_bytes: u128,
    total_side_bytes: u128,
    weighted_cos: f64,
    weighted_kl_fwd: f64,
    weighted_kl_rev: f64,
    cos_below_099: usize,
    tensors: usize,
}

fn run_stack_search(path: &str, round: usize, limit: usize) {
    let candidates = candidates_for_round(round);
    if candidates.is_empty() {
        eprintln!("no candidates defined for round {round}");
        std::process::exit(2);
    }
    let mut g = match gguf::open(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "┌─ stack-search round {round} ({} candidates) ────",
        candidates.len()
    );
    println!("│ model  {path}");
    println!("│ tensors {}", g.tensors.len());
    println!(
        "│ pipeline: PRE -> EXTRACT -> QUANT -> blind repair -> outliers -> PRE inverse -> row norms"
    );
    println!("│ Storage figures estimate payload only with f32 side data; no serialized research record exists.");
    println!("├─ candidates:");
    for c in &candidates {
        println!("│   {}", c.name);
    }

    const KL_K: usize = 4;
    #[cfg(feature = "cuda")]
    let gpu = cuda_backend::CudaMatvec::new().ok();
    #[cfg(feature = "cuda")]
    println!(
        "│ KL matvec: {}",
        if gpu.is_some() {
            "cuBLAS"
        } else {
            "CPU fallback"
        }
    );
    #[cfg(not(feature = "cuda"))]
    println!("│ KL matvec: CPU");

    let metas: Vec<(String, u32, usize, usize)> = g
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type, t.n_elems, t.row_len))
        .collect();
    let mut aggs: Vec<StackAggregate> = candidates
        .iter()
        .map(|c| StackAggregate {
            name: c.name.to_string(),
            ..Default::default()
        })
        .collect();

    let mut processed = 0usize;
    for (idx, (_name, ttype, n_elems, row_len)) in metas.iter().enumerate() {
        if !gguf::dequantizable(*ttype) {
            continue;
        }
        if limit != 0 && processed >= limit {
            break;
        }
        let Ok(w) = gguf::read_tensor_f32(&mut g, idx) else {
            continue;
        };
        if w.is_empty() {
            continue;
        }
        let in_len = (*row_len).max(1);
        let out_len = if *n_elems % in_len == 0 {
            *n_elems / in_len
        } else {
            1
        };
        let is_2d = out_len >= 2 && in_len >= 2;

        let xs = if is_2d {
            metrics::make_gaussian_inputs(in_len, KL_K, 0xB4_5E_D0_C0u32.wrapping_add(idx as u32))
        } else {
            Vec::new()
        };
        let ps_orig = if is_2d {
            #[cfg(feature = "cuda")]
            let ys = if let Some(ref g) = gpu {
                g.matmul_batch(&w, &xs, out_len, in_len)
                    .unwrap_or_else(|_| metrics::cpu_matmul_batch(&w, &xs, out_len, in_len))
            } else {
                metrics::cpu_matmul_batch(&w, &xs, out_len, in_len)
            };
            #[cfg(not(feature = "cuda"))]
            let ys = metrics::cpu_matmul_batch(&w, &xs, out_len, in_len);
            metrics::softmax_batch(&ys)
        } else {
            Vec::new()
        };

        for (ci, cand) in candidates.iter().enumerate() {
            let (recon, side_bytes) = stacks::apply(cand.stack, &w, out_len, in_len);
            let cos = cosine(&w, &recon) as f64;
            let (fwd, rev) = if is_2d {
                #[cfg(feature = "cuda")]
                let ys_q = if let Some(ref g) = gpu {
                    g.matmul_batch(&recon, &xs, out_len, in_len)
                        .unwrap_or_else(|_| metrics::cpu_matmul_batch(&recon, &xs, out_len, in_len))
                } else {
                    metrics::cpu_matmul_batch(&recon, &xs, out_len, in_len)
                };
                #[cfg(not(feature = "cuda"))]
                let ys_q = metrics::cpu_matmul_batch(&recon, &xs, out_len, in_len);
                metrics::bidirectional_kl_cached(&ps_orig, &ys_q)
            } else {
                metrics::kl_histogram_bidirectional(&w, &recon, 256)
            };
            let a = &mut aggs[ci];
            let elems = w.len() as u128;
            a.total_elems += elems;
            a.total_orig_bytes += (w.len() * 4) as u128;
            a.total_code_bytes += stacks::estimated_code_bytes(cand.stack, w.len()) as u128;
            a.total_side_bytes += side_bytes as u128;
            a.weighted_cos += cos * elems as f64;
            a.weighted_kl_fwd += fwd * elems as f64;
            a.weighted_kl_rev += rev * elems as f64;
            if cos < 0.99 {
                a.cos_below_099 += 1;
            }
            a.tensors += 1;
        }
        processed += 1;
        if processed.is_multiple_of(40) {
            eprintln!("│   ... processed {processed} tensors");
        }
    }

    println!("├─ results (weighted by element count):");
    println!(
        "│ {:<40} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "stack", "mean cos", "mean KLfwd", "est.bits/w", "est.ratio", "cos<0.99"
    );
    for a in &aggs {
        if a.tensors == 0 {
            continue;
        }
        let mean_cos = a.weighted_cos / a.total_elems.max(1) as f64;
        let mean_kl_fwd = a.weighted_kl_fwd / a.total_elems.max(1) as f64;
        let total_out = a.total_code_bytes + a.total_side_bytes;
        let bits_per_w = total_out as f64 * 8.0 / a.total_elems.max(1) as f64;
        let ratio = a.total_orig_bytes as f64 / total_out.max(1) as f64;
        println!(
            "│ {:<40} {:>10.5} {:>10.5} {:>9.2}b {:>9.2}x {:>10}",
            a.name, mean_cos, mean_kl_fwd, bits_per_w, ratio, a.cos_below_099
        );
    }
    let winner = aggs.iter().filter(|a| a.tensors != 0).max_by(|a, b| {
        let ca = a.weighted_cos / a.total_elems.max(1) as f64;
        let cb = b.weighted_cos / b.total_elems.max(1) as f64;
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(w) = winner {
        let mean_cos = w.weighted_cos / w.total_elems.max(1) as f64;
        let mean_kl_fwd = w.weighted_kl_fwd / w.total_elems.max(1) as f64;
        let total_out = w.total_code_bytes + w.total_side_bytes;
        let bits_per_w = total_out as f64 * 8.0 / w.total_elems.max(1) as f64;
        println!("├─ Round {round} winner: {}", w.name);
        println!(
            "│   mean cos = {:.5}   mean KL_fwd = {:.5}   bits/w = {:.2}",
            mean_cos, mean_kl_fwd, bits_per_w
        );
    }
    println!("└──────────────────────────────────────────────────────────────");
}
