//! atom-quantizer-q2 — companion binary that runs Composer 2 (Q2-targeting) on top of
//! Composer 1's dequantized output. Never touches main.rs, stacks.rs, or the
//! current picker.
//!
//! Pipeline per tensor:
//!   1. Read w from the GGUF.
//!   2. Apply Composer 1 (rowH8 → wq Q2/Q4/Q8 dual-gate → preserve) → w1
//!   3. For each Q2 candidate stack: apply Composer 2 to w1 → w2
//!   4. Report: cos(w, w2), cos(w1, w2), bits/w for w2's compressed form
//!   5. Aggregate: mean cos, worst cos, cos<0.99 count per stack

mod q2_composer;

use atom_quantizer::{gguf, wq};
use q2_composer::{
    apply_composer1, apply_q2_composer, q2_bits_per_w, q2_candidates, Q2Post, Q2Pre,
};
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
        eprintln!("usage: atom-quantizer-q2 <model.gguf> [--limit N]");
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
                limit = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--out-gguf" => {
                i += 1;
                out_gguf = args.get(i).cloned();
            }
            other => eprintln!("(ignoring unknown arg {other})"),
        }
        i += 1;
    }

    let mut g = match gguf::open(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

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

        // Composer 1.
        let (w1, c1_strat) = apply_composer1(&w, out_len, in_len);
        match c1_strat {
            QuantStrategy::Q2 => n_c1_q2 += 1,
            QuantStrategy::Q4 => n_c1_q4 += 1,
            QuantStrategy::Q8 => n_c1_q8 += 1,
        }

        // Composer 2 — for each candidate stack, apply to w1 and measure.
        for (ci, cand) in candidates.iter().enumerate() {
            let w2 = apply_q2_composer(&w1, cand, out_len, in_len);
            let cos_orig = cosine(&w, &w2);
            let cos_w1 = cosine(&w1, &w2);
            let bits = q2_bits_per_w(
                w.len(),
                out_len,
                in_len,
                cand.k_extract,
                cand.pre == Q2Pre::RowDc,
                matches!(
                    cand.quant,
                    q2_composer::Q2Quant::Asym
                        | q2_composer::Q2Quant::Superpose
                        | q2_composer::Q2Quant::Refract
                ),
            );
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
            // Prevent unused-variable warning on POST field.
            let _ = cand.post == Q2Post::Null;
        }

        processed += 1;
        if processed.is_multiple_of(20) {
            eprintln!("│   … processed {processed} tensors");
        }
    }

    println!("├─ Composer 1 strategy mix (v0.2.3 dual gate on original W):");
    println!("│   Q2={n_c1_q2}  Q4={n_c1_q4}  Q8={n_c1_q8}");
    println!("├─ Composer 2 results (weighted by element count):");
    println!(
        "│ {:<45} {:>12} {:>12} {:>10} {:>10}",
        "stack", "cos(W,W2)", "cos(W1,W2)", "bits/w", "cos<0.99"
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
    // Best-cosine winner.
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

    // ─────────────────── crush the winning stack to disk ───────────────────
    if let Some(dp) = out_gguf.as_ref() {
        let winner = q2_composer::Q2Stack {
            name: "rowDC + top4 + refractQ2 + preserve",
            pre: q2_composer::Q2Pre::RowDc,
            k_extract: 4,
            quant: q2_composer::Q2Quant::Refract,
            post: q2_composer::Q2Post::Preserve,
        };
        println!("┌─ writing passthrough GGUF with Composer1→Composer2 output ──");
        println!("│ dest      {dp}");
        println!("│ Composer 2 stack: {}", winner.name);
        if let Err(e) = std::fs::copy(&path, dp) {
            eprintln!("│ error copying source to dest: {e}");
            std::process::exit(1);
        }
        let mut dest = match std::fs::OpenOptions::new().read(true).write(true).open(dp) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("│ error opening dest for writeback: {e}");
                std::process::exit(1);
            }
        };
        // Re-open source GGUF for the second pass (previous `g` moved in for-loop).
        let mut g2 = match gguf::open(&path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("│ error re-opening source: {e}");
                std::process::exit(1);
            }
        };
        let data_start = g2.data_start;
        let metas2: Vec<(String, u32, usize, usize, u64)> = g2
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.ggml_type, t.n_elems, t.row_len, t.offset))
            .collect();
        let mut written = 0usize;
        let mut skipped = 0usize;
        for (idx, (name, ttype, n_elems, row_len, _off)) in metas2.iter().enumerate() {
            if !gguf::dequantizable(*ttype) {
                continue;
            }
            if !gguf::writable_passthrough(*ttype) {
                skipped += 1;
                continue;
            }
            let Ok(w) = gguf::read_tensor_f32(&mut g2, idx) else {
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
            let (w1, _c1_strat) = apply_composer1(&w, out_len, in_len);
            let w2 = apply_q2_composer(&w1, &winner, out_len, in_len);
            let ti = &g2.tensors[idx];
            if let Err(e) = gguf::write_tensor_reconstruction(&mut dest, ti, data_start, &w2) {
                eprintln!("│ writeback error at {name}: {e}");
                std::process::exit(1);
            }
            written += 1;
            if written.is_multiple_of(40) {
                eprintln!("│   … wrote {written} tensors");
            }
        }
        let src_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let dst_size = std::fs::metadata(dp).map(|m| m.len()).unwrap_or(0);
        println!("│ wrote {written} tensors  |  {skipped} left as source K-quant bytes");
        println!(
            "│ source size : {:>8.1} MB    passthrough size : {:>8.1} MB",
            src_size as f64 / 1e6,
            dst_size as f64 / 1e6
        );
        println!("│ every writable tensor overwritten with (Composer 1 → Composer 2 winner) reconstruction");
        println!("│ load in LM Studio, chat with the model — if coherent, Q2 composer preserved inference.");
        println!("└──────────────────────────────────────────────────────────────");
    }
}
