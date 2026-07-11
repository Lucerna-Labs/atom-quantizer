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
use atom_quantizer::{gguf, metrics, oq, stacks, wq};

use wq::{cosine, encode_with, expand, strategy_name, CompressedTensor, WqConfig, CANDIDATES};

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

/// Dual-gate picker: BOTH KL and cosine must clear their respective bars.
///
/// KL over `softmax(W · x)` with random-Gaussian `x` (k=4) turned out to be an
/// insufficient currency on its own: on Qwen3.5-0.8B it accepted Q2 for every
/// big 2-D linear tensor with KL_fwd 0.0001, while the actual weight cosine
/// dropped to ~0.81 and inference in LM Studio produced hallucinations. The
/// KL metric was blind to per-weight drift that compounds through 24 layers of
/// structured (non-Gaussian) real activations.
///
/// Fix: keep the KL check (it's still a real signal for output-distribution
/// preservation) but add a cosine floor as a mandatory second gate. Both must
/// clear for a candidate strategy to be accepted. If either fails, walk to the
/// next tier. This matches the empirically-safe behavior of the v0.1 cosine
/// picker while keeping the KL numbers reported as diagnostic.
/// Apply the POST-repair stage of the composer chain to a wq reconstruction.
///
/// v0.2.3 wires the first station of the kit's POST-repair set: `preserve` —
/// rescale each row of the recon so its L2 norm matches the original's. Direct
/// cosine-repair primitive per THESIS §14.2. For 2-D linear tensors only.
///
/// Returns the repaired reconstruction; on 1-D or empty tensors returns the
/// input unchanged.
fn apply_post_preserve(orig: &[f32], recon: Vec<f32>, out_len: usize, in_len: usize) -> Vec<f32> {
    if out_len <= 1 || in_len <= 1 || recon.is_empty() {
        return recon;
    }
    let target_norms = stacks::row_l2_norms(orig, out_len, in_len);
    stacks::preserve_row_norm(recon, &target_norms, out_len, in_len)
}

/// v0.2.3 wires PRE = `rowH8` (row-wise 8-pt Walsh-Hadamard) into the codec's
/// per-tensor path. Stack-search Round 2 on Qwen3.5-0.8B ranked rowH8+sym-Q4
/// above baseline sym-Q4 at the SAME bit budget (5 bits/w): mean cos 0.99533
/// vs 0.99499 baseline, and 0 tensors below cos 0.99 vs 1 for baseline.
/// Hadamard is self-inverse (orthonormal 8×8) — same fn call reverses it.
fn apply_rowh8(data: &[f32], out_len: usize, in_len: usize) -> Vec<f32> {
    let mut v = data.to_vec();
    if out_len >= 1 && in_len >= 8 {
        stacks::row_hadamard_8_apply(&mut v, out_len, in_len);
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn compress_kl_adaptive(
    data: &[f32],
    cfg: &WqConfig,
    is_2d: bool,
    out_len: usize,
    in_len: usize,
    xs: &[Vec<f32>],
    ps_orig: &[Vec<f32>],
    kl_ceiling: f64,
    cosine_floor: f32,
    #[cfg(feature = "cuda")] gpu: Option<&cuda_backend::CudaMatvec>,
) -> (CompressedTensor, Vec<f32>, f64, f64, f32) {
    // PRE-transform stage: apply rowH8 (Walsh-Hadamard on each row's 8-elem
    // sub-blocks). Rotates into a basis where per-block max-abs quant produces
    // less directional drift. Only for 2-D tensors with in_len >= 8.
    // The wq codec then encodes in the Hadamard domain; we inverse-transform
    // the reconstruction back to weight space before measurement/writeback.
    let use_pre_rowh8 = is_2d && in_len >= 8 && in_len.is_multiple_of(8);
    let encode_input: Vec<f32> = if use_pre_rowh8 {
        apply_rowh8(data, out_len, in_len)
    } else {
        data.to_vec()
    };

    let mut last: Option<(CompressedTensor, Vec<f32>, f64, f64, f32)> = None;
    for &s in &CANDIDATES {
        let cand = encode_with(&encode_input, s, cfg);
        let recon_hadamard = expand(&cand);
        // PRE inverse: same fn (Hadamard is self-inverse) to return to weight space.
        let recon_raw = if use_pre_rowh8 {
            apply_rowh8(&recon_hadamard, out_len, in_len)
        } else {
            recon_hadamard
        };
        // POST-repair: apply `preserve` (row-norm restore) — completes the composer.
        let recon = apply_post_preserve(data, recon_raw, out_len, in_len);
        let (fwd, rev) = if is_2d {
            #[cfg(feature = "cuda")]
            let ys_q = if let Some(g) = gpu {
                g.matmul_batch(&recon, xs, out_len, in_len)
                    .unwrap_or_else(|_| metrics::cpu_matmul_batch(&recon, xs, out_len, in_len))
            } else {
                metrics::cpu_matmul_batch(&recon, xs, out_len, in_len)
            };
            #[cfg(not(feature = "cuda"))]
            let ys_q = metrics::cpu_matmul_batch(&recon, xs, out_len, in_len);
            metrics::bidirectional_kl_cached(ps_orig, &ys_q)
        } else {
            metrics::kl_histogram_bidirectional(data, &recon, 256)
        };
        let cos = cosine(data, &recon);
        let kl_ok = fwd.max(rev) <= kl_ceiling;
        let cos_ok = cos >= cosine_floor;
        if kl_ok && cos_ok {
            return (cand, recon, fwd, rev, cos);
        }
        last = Some((cand, recon, fwd, rev, cos));
    }
    last.unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Atom Quantizer — KL-driven adaptive weight quantization\n");
        eprintln!("usage: atom-quantizer <model.gguf> [--limit N] [--key 0xHEX] [--out model.oq] [--asym]");
        eprintln!("       atom-quantizer compare <a.gguf> <b.gguf>");
        std::process::exit(2);
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
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--asym" => use_asym = true,
            "--limit" => {
                i += 1;
                limit = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--key" => {
                i += 1;
                key =
                    args.get(i)
                        .and_then(|s| {
                            s.trim_start_matches("0x").parse::<u64>().ok().or_else(|| {
                                u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
                            })
                        })
                        .unwrap_or(key);
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            "--out-gguf" => {
                i += 1;
                out_gguf = args.get(i).cloned();
            }
            other => eprintln!("(ignoring unknown arg {other})"),
        }
        i += 1;
    }

    run_quantize(&path, limit, key, out_path, out_gguf, use_asym);
}

fn run_quantize(
    path: &str,
    limit: usize,
    key: u64,
    out_path: Option<String>,
    out_gguf: Option<String>,
    use_asym: bool,
) {
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
    println!("├─ per-tensor (adaptive: most aggressive strategy that clears BOTH KL ceiling AND cosine floor)");
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
    let kl_ceiling_normal: f64 = 1.0;
    let kl_ceiling_protected: f64 = 0.25;
    let cosine_floor_normal: f32 = 0.99;
    let cosine_floor_protected: f32 = 0.995;
    const KL_K_INPUTS: usize = 4;

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

    if out_path.is_some() {
        if let Some(t) = g.tensors.iter().find(|t| !gguf::dequantizable(t.ggml_type)) {
            eprintln!(
                "error: --out needs every tensor dequantizable, but '{}' is {} (that K-quant isn't wired yet). Refusing to write a partial model.",
                t.name,
                gguf::type_name(t.ggml_type)
            );
            std::process::exit(1);
        }
    }
    let mut writer = match out_path.as_ref() {
        Some(p) => match oq::Writer::create(p) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("error creating {p}: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    // Passthrough GGUF writer: byte-for-byte copy of source, then overwrite each
    // tensor's stored bytes with the Atom Quantizer reconstruction (cast to the source
    // dtype). LM Studio / llama.cpp will load this as if it were the source
    // model, but every tensor holds Atom Quantizer's decoded values — the functional
    // proof that the picker + blind repair preserve inference behavior.
    let (mut gguf_dest, mut gguf_unwritable_count) = if let Some(dp) = out_gguf.as_ref() {
        if let Err(e) = std::fs::copy(path, dp) {
            eprintln!("error copying source GGUF to {dp}: {e}");
            std::process::exit(1);
        }
        match std::fs::OpenOptions::new().read(true).write(true).open(dp) {
            Ok(f) => (Some(f), 0usize),
            Err(e) => {
                eprintln!("error opening {dp} for writeback: {e}");
                std::process::exit(1);
            }
        }
    } else {
        (None, 0usize)
    };
    let gguf_data_start = g.data_start;

    let (mut orig_bytes, mut comp_bytes, mut count) = (0u128, 0u128, 0usize);
    let (mut kl_fwd_sum, mut kl_rev_sum) = (0f64, 0f64);
    let (mut n_q2, mut n_q4, mut n_q8) = (0usize, 0usize, 0usize);
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
                eprintln!("│ ! {name}: {e}");
                continue;
            }
        };
        if w.is_empty() {
            continue;
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

        let in_len = (*row_len).max(1);
        let out_len = if *n_elems % in_len == 0 {
            *n_elems / in_len
        } else {
            1
        };
        let is_2d = out_len > 1 && in_len > 1;

        let xs = if is_2d {
            metrics::make_gaussian_inputs(
                in_len,
                KL_K_INPUTS,
                0xB4_5E_D0_C0u32.wrapping_add(idx as u32),
            )
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

        let (c, post_recon, kl_fwd, kl_rev, cos) = compress_kl_adaptive(
            &w,
            &cfg,
            is_2d,
            out_len,
            in_len,
            &xs,
            &ps_orig,
            kl_ceiling,
            cos_floor,
            #[cfg(feature = "cuda")]
            gpu.as_ref(),
        );
        let worst_kl = kl_fwd.max(kl_rev);
        let kl_miss = worst_kl > kl_ceiling;
        let cos_miss = (cos as f64) < (cos_floor as f64);
        if kl_miss || cos_miss {
            ceiling_misses += 1;
        }
        all_verified &= c.verify();

        if let Some(wr) = writer.as_mut() {
            if let Err(e) = wr.append(name, *ttype, &c) {
                eprintln!("error writing tensor {name}: {e}");
                std::process::exit(1);
            }
        }

        // Passthrough GGUF: overwrite this tensor's stored bytes with the
        // Atom Quantizer reconstruction (cast to the source dtype). K-quant sources
        // are left untouched — they can't be re-encoded without a K-quant
        // writer; the counter tracks how many were left.
        if let Some(dest) = gguf_dest.as_mut() {
            if gguf::writable_passthrough(*ttype) {
                // Write the POST-repaired reconstruction (with preserve applied)
                // — this is what the picker's KL/cos measurement was against.
                let ti = &g.tensors[idx];
                if let Err(e) =
                    gguf::write_tensor_reconstruction(dest, ti, gguf_data_start, &post_recon)
                {
                    eprintln!("error writing passthrough GGUF tensor {name}: {e}");
                    std::process::exit(1);
                }
            } else {
                gguf_unwritable_count += 1;
            }
        }

        let ob = (w.len() * 4) as u128;
        let cb = c.compressed_bytes() as u128;
        orig_bytes += ob;
        comp_bytes += cb;
        kl_fwd_sum += kl_fwd;
        kl_rev_sum += kl_rev;
        if worst_kl > max_kl.0 {
            max_kl = (worst_kl, name.clone());
        }
        count += 1;
        match c.strategy {
            wq::QuantStrategy::Q2 => n_q2 += 1,
            wq::QuantStrategy::Q4 => n_q4 += 1,
            wq::QuantStrategy::Q8 => n_q8 += 1,
        }
        let short = shorten(name, 34);
        let mark = if protect { "*" } else { " " };
        let miss = if kl_miss || cos_miss { "!" } else { " " };
        println!(
            "│{mark}{short:<34} {:>5} {:>11}  {:<3} {:>6.1}x  {miss}{:>8.4} {:>9.4}  cos={:>7.4}",
            gguf::type_name(*ttype),
            n_elems,
            strategy_name(c.strategy),
            c.ratio_vs_f32(),
            kl_fwd,
            kl_rev,
            cos,
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
    println!("processed {count} tensors  |  strategy mix: Q2={n_q2} Q4={n_q4} Q8={n_q8}  (* = protected)");
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
    println!("currency: softmax(W·x) with k=4 Gaussian inputs for 2-D; 256-bin histogram for 1-D");
    println!(
        "worst-tensor max(KL_fwd, KL_rev): {:.4}  ({})",
        max_kl.0,
        shorten(&max_kl.1, 40)
    );
    if ceiling_misses > 0 {
        println!("! {ceiling_misses} tensor(s) exceeded the KL ceiling even at Q8 (kept at Q8 anyway; marked '!')");
    }
    println!(
        "integrity: {}",
        if all_verified {
            "all codes verified \u{2713}"
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

    if let (Some(w), Some(p)) = (writer, out_path.as_ref()) {
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
                    Err(e) => eprintln!("│ verify failed: {e}"),
                }
                println!("└──────────────────────────────────────────────────────────────");
            }
            Err(e) => eprintln!("finish failed: {e}"),
        }
    }

    // Passthrough GGUF summary — LM Studio–loadable proof-of-decode.
    if let (Some(_), Some(dp)) = (gguf_dest.as_ref(), out_gguf.as_ref()) {
        drop(gguf_dest); // ensure the file is flushed / closed before size stat
        let src_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let dst_size = std::fs::metadata(dp).map(|m| m.len()).unwrap_or(0);
        println!("├─ GGUF passthrough ───────────────────────────────────────────");
        println!("│ wrote → {dp}");
        println!(
            "│ source size : {:>8.1} MB    passthrough size : {:>8.1} MB",
            src_size as f64 / 1e6,
            dst_size as f64 / 1e6
        );
        println!(
            "│ tensors overwritten with Atom Quantizer reconstruction: {} (F32/F16/BF16 in-place cast)",
            count - gguf_unwritable_count
        );
        if gguf_unwritable_count > 0 {
            println!(
                "│ tensors left as source K-quant bytes: {} (no K-quant writer in the codec)",
                gguf_unwritable_count
            );
        }
        println!("│ verification path: load in LM Studio, chat with the model —");
        println!("│ if output is coherent, Atom Quantizer's KL-driven picker + blind repair preserved inference.");
        println!("└──────────────────────────────────────────────────────────────");
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
/// PRE forward → EXTRACT (top-K aside) → QUANT → POST → outlier restore
/// → PRE inverse. Outlier restore MUST happen before PRE inverse — see the
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
        "│ pipeline: PRE fwd -> EXTRACT top-K aside -> QUANT -> POST -> outlier restore -> PRE inv"
    );
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
            a.total_code_bytes += stacks::nominal_code_bytes(w.len()) as u128;
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
        "stack", "mean cos", "mean KLfwd", "bits/w", "vs f32", "cos<0.99"
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
