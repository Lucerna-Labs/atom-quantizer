//! Bounded study diagnostics. Imports the unchanged production modules.
//! Synthetic tensors establish codec/metric properties, not model quality.
//! Run from the repository root; see REPORT.md for commands.
#![allow(dead_code)]

#[path = "../../src/gguf.rs"]
mod gguf;
#[path = "../../src/metrics.rs"]
mod metrics;
#[path = "../../src/oq.rs"]
mod oq;
#[path = "../../src/stacks.rs"]
mod stacks;
#[path = "../../src/wq.rs"]
mod wq;

use std::io::{Cursor, Read};
use std::path::Path;

fn u32_read(r: &mut Cursor<&[u8]>) -> u32 {
    let mut b = [0; 4];
    r.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}

fn u64_read(r: &mut Cursor<&[u8]>) -> u64 {
    let mut b = [0; 8];
    r.read_exact(&mut b).unwrap();
    u64::from_le_bytes(b)
}

// Minimal reader for the known one-record OQ02 fixture. Reconstruction itself
// uses production wq::expand; file verification uses production oq::verify_file.
fn read_fixture_record(bytes: &[u8]) -> (wq::CompressedTensor, usize) {
    assert_eq!(&bytes[..4], b"OQ02");
    let mut r = Cursor::new(bytes);
    r.set_position(4);
    assert_eq!(u32_read(&mut r), 1);
    let name_len = u32_read(&mut r) as u64;
    r.set_position(r.position() + name_len);
    let _orig_type = u32_read(&mut r);
    let len = u64_read(&mut r) as usize;
    let mut strategy = [0];
    r.read_exact(&mut strategy).unwrap();
    let strategy = match strategy[0] {
        0 => wq::QuantStrategy::Q2,
        1 => wq::QuantStrategy::Q4,
        2 => wq::QuantStrategy::Q8,
        _ => panic!("invalid strategy"),
    };
    let repair_strength = f32::from_bits(u32_read(&mut r));
    let integrity = u64_read(&mut r);
    let key = u64_read(&mut r);
    let count = u32_read(&mut r) as usize;
    let scale_offset = r.position() as usize;
    let scales = (0..count)
        .map(|_| f32::from_bits(u32_read(&mut r)))
        .collect();
    let count = u32_read(&mut r) as usize;
    let mins = (0..count)
        .map(|_| f32::from_bits(u32_read(&mut r)))
        .collect();
    let count = u64_read(&mut r) as usize;
    let mut packed = vec![0; count];
    r.read_exact(&mut packed).unwrap();
    assert_eq!(r.position() as usize, bytes.len());
    (
        wq::CompressedTensor {
            strategy,
            len,
            scales,
            mins,
            packed,
            repair_strength,
            integrity,
            key,
        },
        scale_offset,
    )
}

fn write_fixture(path: &Path, w: &[f32], rows: usize, cols: usize) {
    let mut b = Vec::from(*b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    let name = b"probe.weight";
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name);
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&(cols as u64).to_le_bytes());
    b.extend_from_slice(&(rows as u64).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes()); // F32
    b.extend_from_slice(&0u64.to_le_bytes()); // tensor offset
    b.resize(b.len().div_ceil(32) * 32, 0);
    for &v in w {
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn relative_l2(a: &[f32], b: &[f32]) -> f64 {
    let e: f64 = a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).powi(2)).sum();
    let n: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum();
    (e / n.max(1e-30)).sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "usage: probe <codec-binary> <output-directory>");
    let dir = Path::new(&args[2]);
    std::fs::create_dir_all(dir).unwrap();
    let (rows, cols) = (4usize, 32usize);
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let a = (i as f32 * 0.37).sin() * 0.08;
            let b = (i as f32 * 0.17).cos() * 0.04;
            a + b + (i / cols) as f32 * 0.005
        })
        .collect();
    let original = dir.join("synthetic.gguf");
    let compressed = dir.join("synthetic.oq");
    let passthrough = dir.join("synthetic-passthrough.gguf");
    write_fixture(&original, &w, rows, cols);
    let output = std::process::Command::new(&args[1])
        .arg(&original)
        .arg("--out").arg(&compressed)
        .arg("--out-gguf").arg(&passthrough)
        .output().unwrap();
    std::fs::write(dir.join("cli.stdout.txt"), &output.stdout).unwrap();
    std::fs::write(dir.join("cli.stderr.txt"), &output.stderr).unwrap();
    assert!(output.status.success(), "fixture CLI failed");

    let bytes = std::fs::read(&compressed).unwrap();
    let (c, scale_offset) = read_fixture_record(&bytes);
    let decoded = wq::expand(&c);
    let mut gguf = gguf::open(passthrough.to_str().unwrap()).unwrap();
    let scored = gguf::read_tensor_f32(&mut gguf, 0).unwrap();
    let mut corrected = decoded.clone();
    stacks::row_hadamard_8_apply(&mut corrected, rows, cols);
    corrected = stacks::preserve_row_norm(
        corrected, &stacks::row_l2_norms(&w, rows, cols), rows, cols,
    );
    let max_diff = scored.iter().zip(&corrected)
        .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);

    let mut altered_bytes = bytes.clone();
    let altered_scale = c.scales[0] * 1.25;
    altered_bytes[scale_offset..scale_offset + 4]
        .copy_from_slice(&altered_scale.to_le_bytes());
    let altered = dir.join("synthetic-altered-scale.oq");
    std::fs::write(&altered, &altered_bytes).unwrap();
    let (altered_c, _) = read_fixture_record(&altered_bytes);
    let altered_verified = oq::verify_file(altered.to_str().unwrap()).unwrap().all_verified;

    let logits = vec![vec![0.25, 0.5, -0.25, -0.5]];
    let shifted = vec![vec![4.25, 4.5, 3.75, 3.5]];
    let (shift_kl_fwd, shift_kl_rev) = metrics::bidirectional_kl_from_ys(&logits, &shifted);

    let r = vec![0.9, 0.4];
    let o = vec![1.0, 0.0];
    let restored = stacks::preserve_row_norm(r.clone(), &[1.0], 1, 2);

    let mut outlier_w = vec![1.0f32; 32];
    outlier_w[7] = 99.0;
    let (stack_recon, _) = stacks::apply(stacks::Stack {
        pre: stacks::Pre::RowDcRemove,
        extract: stacks::Extract::OutlierTopK1,
        quantize: stacks::Quantize::SymUniform,
        post: stacks::Post::PreserveRowNorm,
    }, &outlier_w, 1, 32);
    let original_norm = stacks::row_l2_norms(&outlier_w, 1, 32)[0];
    let stack_norm = stacks::row_l2_norms(&stack_recon, 1, 32)[0];

    let (superpose_recon, superpose_side) = stacks::apply(stacks::Stack {
        pre: stacks::Pre::Null,
        extract: stacks::Extract::Null,
        quantize: stacks::Quantize::SuperposeSymAsym,
        post: stacks::Post::Null,
    }, &(0..32).map(|i| (i as f32 - 14.0) / 17.0).collect::<Vec<_>>(), 1, 32);
    let mut unique: Vec<u32> = superpose_recon.iter().map(|v| v.to_bits()).collect();
    unique.sort_unstable();
    unique.dedup();

    println!("{{");
    println!("  \"evidence_type\": \"synthetic diagnostics using unchanged production modules and CLI\",");
    println!("  \"model_quality_benchmark\": false,");
    println!("  \"elements\": {},", w.len());
    println!("  \"cli_strategy\": \"{}\",", wq::strategy_name(c.strategy));
    println!("  \"passthrough_cosine_vs_original\": {},", wq::cosine(&w, &scored));
    println!("  \"oq_decode_cosine_vs_original\": {},", wq::cosine(&w, &decoded));
    println!("  \"oq_decode_relative_l2_vs_scored\": {},", relative_l2(&scored, &decoded));
    println!("  \"missing_operations_reconstructed_max_abs_diff\": {},", max_diff);
    println!("  \"original_oq_verified\": {},", oq::verify_file(compressed.to_str().unwrap()).unwrap().all_verified);
    println!("  \"altered_scale_oq_verified\": {},", altered_verified);
    println!("  \"altered_scale_changes_decode\": {},", decoded != wq::expand(&altered_c));
    println!("  \"softmax_shift_kl_fwd\": {},", shift_kl_fwd);
    println!("  \"softmax_shift_kl_rev\": {},", shift_kl_rev);
    println!("  \"softmax_shift_raw_output_mse\": 16.0,");
    println!("  \"row_cosine_before_norm_restore\": {},", wq::cosine(&o, &r));
    println!("  \"row_cosine_after_norm_restore\": {},", wq::cosine(&o, &restored));
    println!("  \"rowdc_outlier_preserve_original_norm\": {},", original_norm);
    println!("  \"rowdc_outlier_preserve_final_norm\": {},", stack_norm);
    println!("  \"superpose_distinct_values_in_one_block\": {},", unique.len());
    println!("  \"superpose_reported_bits_per_weight\": {}", (stacks::nominal_code_bytes(32) + superpose_side) as f64 * 8.0 / 32.0);
    println!("}}");
}
