//! atom-quantizer-q2 — companion binary that runs Composer 2 (Q2-targeting) on top of
//! Composer 1's dequantized output. Never touches main.rs, stacks.rs, or the
//! current picker.
//!
//! Pipeline per tensor:
//!   1. Read w from the GGUF.
//!   2. Apply the legacy cosine-only Composer 1 approximation → w1
//!   3. For each Q2 candidate stack: apply Composer 2 to w1 → w2
//!   4. Report stored-dtype cosine and estimated payload bits/weight
//!   5. Aggregate: mean cos, worst cos, cos<0.99 count per stack

mod q2_composer;

use atom_quantizer::{calibration, encoded, gguf, oq, wq};
use q2_composer::{apply_composer1, apply_q2_composer, q2_candidates, Q2Stack};
use std::path::Path;
use wq::{cosine, QuantStrategy};

fn shorten(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let keep = max - 1;
        let tail: String = s.chars().skip(s.chars().count() - keep).collect();
        format!("…{tail}")
    }
}

#[derive(Default, Clone)]
struct Q2Aggregate {
    name: String,
    total_elems: u128,
    weighted_cos_vs_orig: f64,
    weighted_cos_vs_w1: f64,
    total_bits: f64,
    cos_below_099: usize,
    worst_cos_vs_orig: f32,
    worst_name: String,
    tensors: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("atom-quantizer-q2 — Composer 2 (Q2-targeting) on top of Composer 1's output");
        eprintln!();
        eprintln!("usage: atom-quantizer-q2 <model.gguf> [--limit N] [--out-gguf NEW_PATH]");
        std::process::exit(2);
    }
    let path = args[1].clone();
    let mut limit = 0usize;
    let mut out_gguf: Option<String> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                limit = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| fail("--limit requires a nonnegative integer"));
            }
            "--out-gguf" => {
                i += 1;
                out_gguf = Some(
                    args.get(i)
                        .cloned()
                        .unwrap_or_else(|| fail("--out-gguf requires a new destination path")),
                );
            }
            other => fail(&format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if let Some(destination) = &out_gguf {
        if Path::new(destination).symlink_metadata().is_ok() {
            fail(&format!("destination already exists: {destination}"));
        }
    }
    let mut g = match gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let source_hash = out_gguf.as_ref().map(|_| {
        if limit > 0 && limit < g.tensors.len() {
            fail("--out-gguf requires every tensor to be measured; remove --limit for full-model export");
        }
        if g.tensors.iter().any(|t| !gguf::writable_passthrough(t.ggml_type) || t.n_elems == 0) {
            fail("GGUF export requires a complete nonempty F32/F16/BF16 model");
        }
        calibration::file_sha256(&path).unwrap_or_else(|e| fail(&e))
    });
    let candidates = q2_candidates();
    println!("┌─ atom-quantizer-q2 — Composer 2 stack search on top of Composer 1 ─────");
    println!("│ model      {path}");
    println!("│ tensors    {}", g.tensors.len());
    println!("│ pipeline: W  →  [Composer 1]  →  W1  →  [Composer 2 (each stack)]  →  W2");
    println!("│ measurement: cos(W, W2)  and  cos(W1, W2)  per tensor per stack");
    println!("├─ Composer 2 candidates ({} total):", candidates.len());
    for c in &candidates {
        println!("│   {}", c.name);
    }

    let metas: Vec<(String, u32, usize, usize)> = g
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.ggml_type, t.n_elems, t.row_len))
        .collect();

    let mut aggs: Vec<Q2Aggregate> = candidates
        .iter()
        .map(|c| Q2Aggregate {
            name: c.name.to_string(),
            worst_cos_vs_orig: 1.0,
            ..Default::default()
        })
        .collect();

    // Composer 1's strategy mix over tensors it saw.
    let (mut n_c1_q2, mut n_c1_q4, mut n_c1_q8) = (0usize, 0usize, 0usize);
    let mut processed = 0usize;

    for (idx, (name, ttype, n_elems, row_len)) in metas.iter().enumerate() {
        if !gguf::dequantizable(*ttype) {
            continue;
        }
        if limit > 0 && processed >= limit {
            break;
        }
        let w = gguf::read_tensor_f32(&mut g, idx).unwrap_or_else(|e| fail(&e));
        if w.iter().any(|v| !v.is_finite()) {
            fail(&format!("nonfinite source tensor: {name}"));
        }
        if w.is_empty() {
            continue;
        }
        let in_len = (*row_len).max(1);
        let out_len = if *n_elems % in_len == 0 {
            *n_elems / in_len
        } else {
            1
        };

        // Composer 1.
        let (w1, c1_strat) = apply_composer1(&w, out_len, in_len);
        match c1_strat {
            QuantStrategy::Q2 => n_c1_q2 += 1,
            QuantStrategy::Q4 => n_c1_q4 += 1,
            QuantStrategy::Q8 => n_c1_q8 += 1,
        }

        // Composer 2 — for each candidate stack, apply to w1 and measure.
        for (ci, cand) in candidates.iter().enumerate() {
            let w2 = stored_reconstruction(&w1, cand, out_len, in_len, *ttype)
                .unwrap_or_else(|e| fail(&e));
            let cos_orig = cosine(&w, &w2);
            let cos_w1 = cosine(&w1, &w2);
            let bits = q2_composer::estimated_bits_per_weight(w.len(), out_len, in_len, cand);
            let a = &mut aggs[ci];
            let elems = w.len() as u128;
            a.total_elems += elems;
            a.weighted_cos_vs_orig += cos_orig as f64 * elems as f64;
            a.weighted_cos_vs_w1 += cos_w1 as f64 * elems as f64;
            a.total_bits += bits * elems as f64;
            if (cos_orig as f64) < 0.99 {
                a.cos_below_099 += 1;
            }
            if cos_orig < a.worst_cos_vs_orig {
                a.worst_cos_vs_orig = cos_orig;
                a.worst_name = name.clone();
            }
            a.tensors += 1;
        }

        processed += 1;
        if processed.is_multiple_of(20) {
            eprintln!("│   … processed {processed} tensors");
        }
    }

    println!("├─ Composer 1 strategy mix (legacy cosine-only approximation):");
    println!("│   Q2={n_c1_q2}  Q4={n_c1_q4}  Q8={n_c1_q8}");
    println!("├─ Composer 2 results (weighted by element count):");
    println!(
        "│ {:<45} {:>12} {:>12} {:>10} {:>10}",
        "stack", "cos(W,W2)", "cos(W1,W2)", "est.bits/w", "cos<0.99"
    );
    for a in &aggs {
        if a.tensors == 0 {
            continue;
        }
        let mean_cos_orig = a.weighted_cos_vs_orig / a.total_elems.max(1) as f64;
        let mean_cos_w1 = a.weighted_cos_vs_w1 / a.total_elems.max(1) as f64;
        let mean_bits = a.total_bits / a.total_elems.max(1) as f64;
        println!(
            "│ {:<45} {:>12.5} {:>12.5} {:>9.2}b {:>10}",
            a.name, mean_cos_orig, mean_cos_w1, mean_bits, a.cos_below_099
        );
    }
    println!(
        "│ Payload sizes are estimates with f32 side data; no packed Composer 2 format exists."
    );
    // The overall ranking remains diagnostic; export requires a passing candidate.
    let winner = aggs.iter().filter(|a| a.tensors != 0).max_by(|a, b| {
        let ca = a.weighted_cos_vs_orig / a.total_elems.max(1) as f64;
        let cb = b.weighted_cos_vs_orig / b.total_elems.max(1) as f64;
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(w) = winner {
        let mean = w.weighted_cos_vs_orig / w.total_elems.max(1) as f64;
        let bits = w.total_bits / w.total_elems.max(1) as f64;
        println!("├─ Best by cos(W, W2): {}", w.name);
        println!(
            "│   mean cos = {:.5}   bits/w = {:.2}   worst cos = {:.5}  ({})",
            mean,
            bits,
            w.worst_cos_vs_orig,
            shorten(&w.worst_name, 40)
        );
    }
    // Any stack that clears cos ≥ 0.99?
    let clears: Vec<&Q2Aggregate> = aggs
        .iter()
        .filter(|a| a.tensors != 0 && a.cos_below_099 == 0)
        .collect();
    if clears.is_empty() {
        println!("├─ No candidate cleared cosine ≥ 0.99 on all {processed} tensors.");
        println!("│   Reporting how close each got — the ranking is above.");
    } else {
        println!(
            "├─ {} candidate(s) cleared cosine ≥ 0.99 on every tensor:",
            clears.len()
        );
        for a in &clears {
            let bits = a.total_bits / a.total_elems.max(1) as f64;
            let mean = a.weighted_cos_vs_orig / a.total_elems.max(1) as f64;
            println!("│   {}  (mean cos {:.5}, bits/w {:.2})", a.name, mean, bits);
        }
    }
    println!("└──────────────────────────────────────────────────────────────");

    // Export only the highest-scoring candidate that passed on the full model.
    if let Some(dp) = out_gguf.as_ref() {
        let selected = aggs
            .iter()
            .enumerate()
            .filter(|(_, a)| a.tensors == g.tensors.len() && a.tensors > 0 && a.cos_below_099 == 0)
            .max_by(|(_, a), (_, b)| a.weighted_cos_vs_orig.total_cmp(&b.weighted_cos_vs_orig))
            .map(|(i, _)| candidates[i])
            .unwrap_or_else(|| {
                fail("no candidate passed the cosine floor on every tensor; no output published")
            });
        println!("┌─ writing passthrough GGUF with Composer1→Composer2 output ──");
        println!("│ dest      {dp}");
        println!("│ Composer 2 stack: {}", selected.name);
        let hash = source_hash.unwrap_or_else(|| fail("missing source identity"));
        let written = export_selected(&path, dp, &selected, hash).unwrap_or_else(|e| fail(&e));
        let src_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let dst_size = std::fs::metadata(dp).map(|m| m.len()).unwrap_or(0);
        println!("│ wrote and verified all {written} tensors");
        println!(
            "│ source size : {:>8.1} MB    passthrough size : {:>8.1} MB",
            src_size as f64 / 1e6,
            dst_size as f64 / 1e6
        );
        println!("│ Export passed stored-weight cosine checks; model-level quality requires inference evaluation.");
        println!("└──────────────────────────────────────────────────────────────");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1)
}

fn stored_reconstruction(
    w1: &[f32],
    stack: &Q2Stack,
    rows: usize,
    cols: usize,
    dtype: u32,
) -> Result<Vec<f32>, String> {
    let mut values = apply_q2_composer(w1, stack, rows, cols);
    if gguf::writable_passthrough(dtype) {
        for v in &mut values {
            *v = gguf::round_to_storage(*v, dtype)?;
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("candidate produced nonfinite stored weights".into());
    }
    Ok(values)
}

fn export_selected(
    source: &str,
    destination: &str,
    stack: &Q2Stack,
    source_hash: [u8; 32],
) -> Result<u32, String> {
    let output = Path::new(destination);
    if output.symlink_metadata().is_ok() {
        return Err("destination already exists".into());
    }
    if calibration::file_sha256(source)? != source_hash {
        return Err("source changed after candidate measurement".into());
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let name = output
        .file_name()
        .ok_or("invalid output path")?
        .to_string_lossy();
    let intermediate =
        output.with_file_name(format!(".{name}.{}.{}.q2.oq", std::process::id(), stamp));
    let intermediate_path = intermediate.to_str().ok_or("non-UTF8 output path")?;
    let mut published_intermediate = false;
    let result = (|| {
        let mut source_model = gguf::open(source)?;
        let prefix = gguf::model_header(&mut source_model)?;
        let mut writer = oq::Writer::create_with_header(intermediate_path, prefix)?;
        let metas = source_model.tensors.clone();
        for (index, meta) in metas.iter().enumerate() {
            let original = gguf::read_tensor_f32(&mut source_model, index)?;
            let cols = meta.row_len.max(1);
            let rows = original.len() / cols;
            let (w1, _) = apply_composer1(&original, rows, cols);
            let values = stored_reconstruction(&w1, stack, rows, cols, meta.ggml_type)?;
            let score = cosine(&original, &values);
            if !score.is_finite() || score < 0.99 {
                return Err(format!(
                    "selected candidate failed export validation for {}",
                    meta.name
                ));
            }
            let mut tensor = encoded::EncodedTensor::exact(values, meta.dims.clone());
            tensor.output_type = meta.ggml_type;
            writer.append_encoded(&meta.name, meta.ggml_type, &tensor)?;
        }
        if calibration::file_sha256(source)? != source_hash {
            return Err("source changed during export".into());
        }
        writer.finish()?;
        published_intermediate = true;
        oq::decode_to_gguf(intermediate_path, destination)
    })();
    if published_intermediate {
        std::fs::remove_file(&intermediate).map_err(|e| e.to_string())?;
    }
    result
}
