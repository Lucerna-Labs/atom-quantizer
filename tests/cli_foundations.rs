use atom_quantizer::{calibration, gguf, oq};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/cli-tests")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_atom-quantizer"))
        .args(args)
        .output()
        .unwrap()
}

fn succeeds(args: &[&str]) -> Output {
    let result = run(args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result
}

fn fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let source = dir.join("source.gguf");
    let mut b = Vec::from(*b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let name = b"probe.weight";
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name);
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&32u64.to_le_bytes());
    b.extend_from_slice(&4u64.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.resize(b.len().div_ceil(32) * 32, 0);
    for i in 0..128 {
        let v = (i as f32 * 0.37).sin() * 0.08 + (i as f32 * 0.17).cos() * 0.04;
        b.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&source, b).unwrap();
    let calibration_path = dir.join("samples.acal");
    let mut c = Vec::from(*b"AC01");
    c.extend_from_slice(&calibration::file_sha256(source.to_str().unwrap()).unwrap());
    c.extend_from_slice(&[0; 32]); // synthetic test provenance, not a real corpus
    c.extend_from_slice(&1u32.to_le_bytes());
    c.extend_from_slice(&(name.len() as u32).to_le_bytes());
    c.extend_from_slice(name);
    c.push(0);
    c.extend_from_slice(&32u32.to_le_bytes());
    c.extend_from_slice(&32u32.to_le_bytes());
    for r in 0..32 {
        for col in 0..32 {
            c.extend_from_slice(&(if r == col { 1.0f32 } else { 0.0 }).to_le_bytes());
        }
    }
    let hash = Sha256::digest(&c);
    c.extend_from_slice(&hash);
    std::fs::write(&calibration_path, c).unwrap();
    (source, calibration_path)
}

#[test]
fn cli_artifact_reproduces_scored_weights_and_honors_complete_byte_budget() {
    let dir = directory();
    let (source, cal) = fixture(&dir);
    let first = dir.join("first.oq");
    let direct = dir.join("direct.gguf");
    succeeds(&[
        source.to_str().unwrap(),
        "--calibration",
        cal.to_str().unwrap(),
        "--out",
        first.to_str().unwrap(),
        "--out-gguf",
        direct.to_str().unwrap(),
    ]);
    let budget = std::fs::metadata(&first).unwrap().len();
    let bounded = dir.join("bounded.oq");
    succeeds(&[
        source.to_str().unwrap(),
        "--calibration",
        cal.to_str().unwrap(),
        "--budget-bytes",
        &budget.to_string(),
        "--out",
        bounded.to_str().unwrap(),
    ]);
    assert!(std::fs::metadata(&bounded).unwrap().len() <= budget);
    succeeds(&["verify", bounded.to_str().unwrap()]);
    std::fs::remove_file(source).unwrap();
    let decoded = dir.join("decoded.gguf");
    succeeds(&[
        "decode",
        first.to_str().unwrap(),
        "--out",
        decoded.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(direct).unwrap(),
        std::fs::read(decoded).unwrap()
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn impossible_budget_and_wrong_source_do_not_produce_an_artifact() {
    let dir = directory();
    let (source, cal) = fixture(&dir);
    let output = dir.join("should-not-exist.oq");
    let result = run(&[
        source.to_str().unwrap(),
        "--calibration",
        cal.to_str().unwrap(),
        "--budget-bytes",
        "1",
        "--out",
        output.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(!output.exists());
    let mut changed = std::fs::read(&source).unwrap();
    *changed.last_mut().unwrap() ^= 1;
    std::fs::write(&source, changed).unwrap();
    let result = run(&[
        source.to_str().unwrap(),
        "--calibration",
        cal.to_str().unwrap(),
        "--out",
        output.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(!output.exists());
    assert!(String::from_utf8_lossy(&result.stderr).contains("SHA-256 mismatch"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn failed_lossy_candidates_use_exact_escape_and_corruption_cannot_be_exported() {
    let dir = directory();
    let (source, cal) = fixture(&dir);
    let output = dir.join("exact.oq");
    succeeds(&[
        source.to_str().unwrap(),
        "--calibration",
        cal.to_str().unwrap(),
        "--max-output-mse",
        "0.000000000001",
        "--out",
        output.to_str().unwrap(),
    ]);
    let mut reader = oq::Reader::open(output.to_str().unwrap()).unwrap();
    let record = reader.next_tensor().unwrap().unwrap();
    assert!(record.tensor.strategy().is_none());
    let mut src = gguf::open(source.to_str().unwrap()).unwrap();
    assert_eq!(
        record.tensor.decode().unwrap(),
        gguf::read_tensor_f32(&mut src, 0).unwrap()
    );
    let mut corrupt = std::fs::read(&output).unwrap();
    let n = corrupt.len();
    corrupt[n - 12] ^= 1;
    std::fs::write(&output, corrupt).unwrap();
    assert!(!run(&["verify", output.to_str().unwrap()]).status.success());
    let decoded = dir.join("corrupt.gguf");
    assert!(!run(&[
        "decode",
        output.to_str().unwrap(),
        "--out",
        decoded.to_str().unwrap()
    ])
    .status
    .success());
    assert!(!decoded.exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn fp16_and_bf16_exports_are_exactly_the_values_seen_by_the_gates() {
    for dtype in [gguf::GGML_F16, gguf::GGML_BF16] {
        let dir = directory();
        let (source, cal) = fixture(&dir);
        let mut g = gguf::open(source.to_str().unwrap()).unwrap();
        let values = gguf::read_tensor_f32(&mut g, 0).unwrap();
        let mut prefix = gguf::model_header(&mut g).unwrap();
        let type_offset = 24 + 8 + "probe.weight".len() + 4 + 16;
        prefix[type_offset..type_offset + 4].copy_from_slice(&dtype.to_le_bytes());
        let mut meta = g.tensors[0].clone();
        meta.ggml_type = dtype;
        let mut f = std::fs::File::create(&source).unwrap();
        f.write_all(&prefix).unwrap();
        gguf::write_tensor_reconstruction(&mut f, &meta, g.data_start, &values).unwrap();
        drop(f);
        let mut calibration_bytes = std::fs::read(&cal).unwrap();
        calibration_bytes[4..36]
            .copy_from_slice(&calibration::file_sha256(source.to_str().unwrap()).unwrap());
        let end = calibration_bytes.len() - 32;
        let hash = Sha256::digest(&calibration_bytes[..end]);
        calibration_bytes[end..].copy_from_slice(&hash);
        std::fs::write(&cal, calibration_bytes).unwrap();
        let output = dir.join("rounded.oq");
        succeeds(&[
            source.to_str().unwrap(),
            "--calibration",
            cal.to_str().unwrap(),
            "--out",
            output.to_str().unwrap(),
        ]);
        let mut r = oq::Reader::open(output.to_str().unwrap()).unwrap();
        let tensor = r.next_tensor().unwrap().unwrap().tensor;
        assert_eq!(tensor.output_type, dtype);
        let expected = tensor.decode().unwrap();
        let decoded = dir.join("rounded.gguf");
        succeeds(&[
            "decode",
            output.to_str().unwrap(),
            "--out",
            decoded.to_str().unwrap(),
        ]);
        let mut g = gguf::open(decoded.to_str().unwrap()).unwrap();
        assert_eq!(gguf::read_tensor_f32(&mut g, 0).unwrap(), expected);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn existing_outputs_and_failed_encodes_preserve_user_files() {
    let dir = directory();
    let (source, _) = fixture(&dir);
    let before = std::fs::read(&source).unwrap();
    assert!(
        !run(&[source.to_str().unwrap(), "--out", source.to_str().unwrap()])
            .status
            .success()
    );
    assert_eq!(std::fs::read(&source).unwrap(), before);
    let mut truncated = before;
    truncated.truncate(truncated.len() - 10);
    std::fs::write(&source, truncated).unwrap();
    let output = dir.join("incomplete.oq");
    assert!(
        !run(&[source.to_str().unwrap(), "--out", output.to_str().unwrap()])
            .status
            .success()
    );
    assert!(!output.exists());
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn limited_calibration_does_not_require_unprocessed_tensors() {
    let dir = directory();
    let (source, cal) = fixture(&dir);
    let original = std::fs::read(&source).unwrap();
    let values = &original[96..];
    let mut header = Vec::from(*b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&2u64.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    for (i, name) in ["probe.weight", "unprocessed.weight"].iter().enumerate() {
        header.extend_from_slice(&(name.len() as u64).to_le_bytes());
        header.extend_from_slice(name.as_bytes());
        header.extend_from_slice(&2u32.to_le_bytes());
        header.extend_from_slice(&32u64.to_le_bytes());
        header.extend_from_slice(&4u64.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&(i as u64 * 512).to_le_bytes());
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend_from_slice(values);
    header.extend_from_slice(values);
    std::fs::write(&source, header).unwrap();
    let mut c = std::fs::read(&cal).unwrap();
    c[4..36].copy_from_slice(&calibration::file_sha256(source.to_str().unwrap()).unwrap());
    let end = c.len() - 32;
    let hash = Sha256::digest(&c[..end]);
    c[end..].copy_from_slice(&hash);
    std::fs::write(&cal, c).unwrap();
    let output = dir.join("partial.oq");
    succeeds(&[
        source.to_str().unwrap(),
        "--limit",
        "1",
        "--calibration",
        cal.to_str().unwrap(),
        "--budget-bytes",
        "10000",
        "--out",
        output.to_str().unwrap(),
    ]);
    assert_eq!(oq::verify_file(output.to_str().unwrap()).unwrap().count, 1);
    let decoded = dir.join("partial.gguf");
    assert!(!run(&[
        "decode",
        output.to_str().unwrap(),
        "--out",
        decoded.to_str().unwrap()
    ])
    .status
    .success());
    assert!(!decoded.exists());
    std::fs::remove_dir_all(dir).unwrap();
}
