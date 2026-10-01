use atom_quantizer::calibration;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn directory() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/assess-tests")
        .join(format!(
            "{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn gguf(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = Vec::from(*b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    let mut payload = Vec::new();
    for (name, dims, values) in tensors {
        assert_eq!(dims.iter().product::<usize>(), values.len());
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        header.extend_from_slice(&(name.len() as u64).to_le_bytes());
        header.extend_from_slice(name.as_bytes());
        header.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for dim in dims {
            header.extend_from_slice(&(*dim as u64).to_le_bytes());
        }
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend(payload);
    std::fs::write(path, header).unwrap();
}

enum Samples<'a> {
    Linear(&'a str, Vec<Vec<f32>>),
    Embedding(&'a str, Vec<u32>),
}

fn calibration_file(path: &Path, source: &Path, entries: &[Samples<'_>]) {
    let mut data = Vec::from(*b"AC01");
    data.extend_from_slice(&calibration::file_sha256(source.to_str().unwrap()).unwrap());
    data.extend_from_slice(&[0; 32]); // Deliberately synthetic test provenance.
    data.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for entry in entries {
        let name = match entry {
            Samples::Linear(name, _) | Samples::Embedding(name, _) => name,
        };
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name.as_bytes());
        match entry {
            Samples::Linear(_, rows) => {
                data.push(0);
                data.extend_from_slice(&(rows.len() as u32).to_le_bytes());
                data.extend_from_slice(&(rows[0].len() as u32).to_le_bytes());
                for row in rows {
                    for value in row {
                        data.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
            Samples::Embedding(_, ids) => {
                data.push(1);
                data.extend_from_slice(&(ids.len() as u32).to_le_bytes());
                for id in ids {
                    data.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
    }
    data.extend_from_slice(&Sha256::digest(&data));
    std::fs::write(path, data).unwrap();
}

fn run(source: &Path, candidate: &Path, calibration: &Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_atom-quantizer"))
        .args([
            "assess",
            source.to_str().unwrap(),
            candidate.to_str().unwrap(),
            "--calibration",
            calibration.to_str().unwrap(),
        ])
        .args(extra)
        .output()
        .unwrap()
}

fn report(output: &Output, code: i32) -> String {
    assert_eq!(
        output.status.code(),
        Some(code),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout.clone()).unwrap()
}

#[test]
fn identity_assessment_and_json_escaping_cover_every_tensor() {
    let dir = directory();
    let source = dir.join(if cfg!(windows) {
        "source λ.gguf"
    } else {
        "source\".gguf"
    });
    let cal = dir.join("fit.acal");
    let name = "probe\"\\\n\u{1}λ.weight";
    gguf(
        &source,
        &[
            (name, vec![32, 4], vec![0.5; 128]),
            ("output_norm.weight", vec![32], vec![1.0; 32]),
        ],
    );
    calibration_file(&cal, &source, &[Samples::Linear(name, vec![vec![1.0; 32]])]);
    let text = report(&run(&source, &source, &cal, &[]), 0);
    let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed["source"].as_str(), source.to_str());
    assert!(text.contains("\"tensor_count\":2,\"passed_count\":2,\"failed_count\":0"));
    assert!(text.contains("\"name\":\"probe\\\"\\\\\\n\\u0001λ.weight\""));
    assert!(
        text.contains("\"observation\":\"histogram\",\"samples\":0,\"output_gate_required\":false")
    );
    assert!(text.contains("\"all_passed\":true"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn independent_kl_cosine_and_output_error_failures_remain_distinct() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    for case in ["output", "cosine", "kl"] {
        let original = vec![if case == "kl" { 10.0 } else { 0.5 }; 128];
        let mut changed = original.clone();
        let mut inputs = vec![0.0; 32];
        inputs[0] = 1.0;
        match case {
            "output" => changed.fill(0.75),
            "cosine" => changed[10] = 5.0,
            "kl" => {
                changed[..32].fill(10.3125);
                inputs.fill(100.0);
            }
            _ => unreachable!(),
        }
        gguf(&source, &[("probe.weight", vec![32, 4], original)]);
        gguf(&candidate, &[("probe.weight", vec![32, 4], changed)]);
        calibration_file(
            &cal,
            &source,
            &[Samples::Linear("probe.weight", vec![inputs])],
        );
        let text = report(&run(&source, &candidate, &cal, &[]), 2);
        let gates = match case {
            "output" => "\"gates\":{\"kl\":true,\"cosine\":true,\"output_mse\":false}",
            "cosine" => "\"gates\":{\"kl\":true,\"cosine\":false,\"output_mse\":true}",
            "kl" => "\"gates\":{\"kl\":false,\"cosine\":true,\"output_mse\":true}",
            _ => unreachable!(),
        };
        assert!(text.contains(gates), "{case}: {text}");
        assert!(text.contains("\"all_passed\":false"));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn protected_tensors_keep_the_stricter_output_error_ceiling() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    for (name, expected) in [("probe.weight", 0), ("blk.0.attn_output.weight", 2)] {
        gguf(&source, &[(name, vec![32, 4], vec![10.0; 128])]);
        gguf(&candidate, &[(name, vec![32, 4], vec![11.0; 128])]);
        calibration_file(&cal, &source, &[Samples::Linear(name, vec![vec![1.0; 32]])]);
        let text = report(&run(&source, &candidate, &cal, &[]), expected);
        assert!(text.contains(if expected == 0 {
            "\"protected\":false"
        } else {
            "\"protected\":true"
        }));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unbounded_output_error_is_reported_as_a_failed_gate_not_invalid_json() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    gguf(&source, &[("probe.weight", vec![32, 4], vec![0.0; 128])]);
    gguf(&candidate, &[("probe.weight", vec![32, 4], vec![1.0; 128])]);
    calibration_file(
        &cal,
        &source,
        &[Samples::Linear("probe.weight", vec![vec![1.0; 32]])],
    );
    let text = report(&run(&source, &candidate, &cal, &[]), 2);
    assert!(text.contains("\"output_mse\":null,\"output_mse_state\":\"unbounded\""));
    assert!(text.contains("\"gates\":{\"kl\":true,\"cosine\":false,\"output_mse\":false}"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn embedding_lookup_and_unused_output_head_observations_are_explicit() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    gguf(
        &source,
        &[("token_embd.weight", vec![32, 8], vec![0.5; 256])],
    );
    let mut values = vec![0.5; 256];
    values[7 * 32] = 10.0; // Outside both observed lookup rows.
    gguf(&candidate, &[("token_embd.weight", vec![32, 8], values)]);
    calibration_file(
        &cal,
        &source,
        &[
            Samples::Embedding("token_embd.weight", vec![0, 2]),
            Samples::Linear("token_embd.weight::output", vec![vec![1.0; 32]]),
        ],
    );
    let text = report(&run(&source, &candidate, &cal, &[]), 2);
    assert!(text.contains("\"observation\":\"embedding_lookup\",\"samples\":2"));
    assert!(text.contains("\"output_mse\":0,"));
    assert!(text.contains("\"unused_calibration_entries\":[\"token_embd.weight::output\"]"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn incomplete_or_ambiguous_models_and_nonfinite_values_produce_no_report() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    gguf(
        &source,
        &[
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("b.weight", vec![32, 4], vec![0.5; 128]),
        ],
    );
    calibration_file(
        &cal,
        &source,
        &[
            Samples::Linear("a.weight", vec![vec![1.0; 32]]),
            Samples::Linear("b.weight", vec![vec![1.0; 32]]),
        ],
    );
    let cases = vec![
        vec![("a.weight", vec![32, 4], vec![0.5; 128])],
        vec![
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("a.weight", vec![32, 4], vec![0.5; 128]),
        ],
        vec![
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("extra.weight", vec![32, 4], vec![0.5; 128]),
        ],
        vec![
            ("a.weight", vec![16, 8], vec![0.5; 128]),
            ("b.weight", vec![32, 4], vec![0.5; 128]),
        ],
        vec![
            ("a.weight", vec![32, 4], vec![f32::NAN; 128]),
            ("b.weight", vec![32, 4], vec![0.5; 128]),
        ],
    ];
    for tensors in cases {
        gguf(&candidate, &tensors);
        let result = run(&source, &candidate, &cal, &[]);
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
    }
    gguf(
        &candidate,
        &[
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("b.weight", vec![32, 4], vec![0.5; 128]),
            ("extra.weight", vec![32, 4], vec![0.5; 128]),
        ],
    );
    let extra = run(&source, &candidate, &cal, &[]);
    assert_eq!(extra.status.code(), Some(1));
    assert!(extra.stdout.is_empty());
    gguf(
        &source,
        &[
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("a.weight", vec![32, 4], vec![0.5; 128]),
        ],
    );
    gguf(
        &candidate,
        &[
            ("a.weight", vec![32, 4], vec![0.5; 128]),
            ("b.weight", vec![32, 4], vec![0.5; 128]),
        ],
    );
    let duplicate = run(&source, &candidate, &cal, &[]);
    assert_eq!(duplicate.status.code(), Some(1));
    assert!(duplicate.stdout.is_empty());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("duplicate source tensor"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn required_source_bound_calibration_and_strict_flags_are_enforced() {
    let dir = directory();
    let source = dir.join("source.gguf");
    let candidate = dir.join("candidate.gguf");
    let cal = dir.join("fit.acal");
    gguf(&source, &[("probe.weight", vec![32, 4], vec![0.5; 128])]);
    gguf(
        &candidate,
        &[("probe.weight", vec![32, 4], vec![0.75; 128])],
    );
    calibration_file(
        &cal,
        &source,
        &[Samples::Linear("probe.weight", vec![vec![1.0; 32]])],
    );
    for extra in [
        vec!["--limit", "1"],
        vec!["--max-output-mse", "NaN"],
        vec!["--max-output-mse", "0"],
        vec!["--calibration", cal.to_str().unwrap()],
    ] {
        let output = run(&source, &candidate, &cal, &extra);
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
    }
    let missing = Command::new(env!("CARGO_BIN_EXE_atom-quantizer"))
        .args([
            "assess",
            source.to_str().unwrap(),
            candidate.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(missing.status.code(), Some(1));
    assert!(missing.stdout.is_empty());
    calibration_file(
        &cal,
        &source,
        &[Samples::Linear("wrong.weight", vec![vec![1.0; 32]])],
    );
    let missing_operator = run(&source, &candidate, &cal, &[]);
    assert_eq!(missing_operator.status.code(), Some(1));
    assert!(missing_operator.stdout.is_empty());
    calibration_file(
        &cal,
        &candidate,
        &[Samples::Linear("probe.weight", vec![vec![1.0; 32]])],
    );
    let wrong_source = run(&source, &candidate, &cal, &[]);
    assert_eq!(wrong_source.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&wrong_source.stderr).contains("SHA-256 mismatch"));
    std::fs::remove_dir_all(dir).unwrap();
}
