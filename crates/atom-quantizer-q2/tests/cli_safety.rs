use atom_quantizer::{gguf, wq};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/q2-cli-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn model(&self, filename: &str, count: usize, dtype: u32, noisy: bool) -> PathBuf {
        let path = self.0.join(filename);
        let stride = if dtype == gguf::GGML_F32 { 512 } else { 256 };
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend((count as u64).to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        for index in 0..count {
            let name = format!("probe{index}.weight");
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(2u32.to_le_bytes());
            bytes.extend(32u64.to_le_bytes());
            bytes.extend(4u64.to_le_bytes());
            bytes.extend(dtype.to_le_bytes());
            bytes.extend((index as u64 * stride).to_le_bytes());
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        std::fs::write(&path, bytes).unwrap();
        let model = gguf::open(path.to_str().unwrap()).unwrap();
        let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        for (index, tensor) in model.tensors.iter().enumerate() {
            let values: Vec<f32> = (0..128)
                .map(|i| {
                    if noisy {
                        (wq::mix_u32((i + index * 128 + 13) as u32) as f64 / u32::MAX as f64 * 2.0
                            - 1.0) as f32
                    } else {
                        (index + 1) as f32 * if i % 2 == 0 { 1.0 } else { -1.0 }
                    }
                })
                .collect();
            gguf::write_tensor_reconstruction(&mut file, tensor, model.data_start, &values)
                .unwrap();
        }
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn run(source: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_atom-quantizer-q2"))
        .arg(source)
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn input_existing_output_and_aliases_are_preserved() {
    let f = Fixture::new();
    let source = f.model("source.gguf", 1, gguf::GGML_F32, false);
    let original = std::fs::read(&source).unwrap();
    let existing = f.0.join("existing.gguf");
    std::fs::write(&existing, b"existing user data").unwrap();
    let hardlink = f.0.join("alias.gguf");
    std::fs::hard_link(&source, &hardlink).unwrap();
    for destination in [&source, &existing, &hardlink] {
        let before = std::fs::read(destination).unwrap();
        assert!(
            !run(&source, &["--out-gguf", destination.to_str().unwrap()])
                .status
                .success()
        );
        assert_eq!(std::fs::read(destination).unwrap(), before);
    }
    #[cfg(unix)]
    {
        let link = f.0.join("dangling.gguf");
        std::os::unix::fs::symlink(f.0.join("absent.gguf"), &link).unwrap();
        assert!(!run(&source, &["--out-gguf", link.to_str().unwrap()])
            .status
            .success());
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    }
    assert_eq!(std::fs::read(source).unwrap(), original);
}

#[test]
fn partial_or_failing_search_never_publishes_a_model() {
    let f = Fixture::new();
    let source = f.model("source.gguf", 2, gguf::GGML_F32, false);
    let destination = f.0.join("partial.gguf");
    let result = run(
        &source,
        &["--limit", "1", "--out-gguf", destination.to_str().unwrap()],
    );
    assert!(!result.status.success());
    assert!(!destination.exists());
    let noisy = f.model("noisy.gguf", 2, gguf::GGML_F32, true);
    let destination = f.0.join("failed.gguf");
    let result = run(&noisy, &["--out-gguf", destination.to_str().unwrap()]);
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("no candidate passed"));
    assert!(!destination.exists());
}

#[test]
fn successful_export_uses_the_reported_candidate_and_stored_dtype() {
    for dtype in [gguf::GGML_F32, gguf::GGML_F16, gguf::GGML_BF16] {
        let f = Fixture::new();
        let source = f.model("source.gguf", 2, dtype, false);
        let destination = f.0.join("decoded.gguf");
        let original = std::fs::read(&source).unwrap();
        let result = run(&source, &["--out-gguf", destination.to_str().unwrap()]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let text = String::from_utf8(result.stdout).unwrap();
        let reported = text
            .lines()
            .find_map(|l| l.split_once("Best by cos(W, W2): ").map(|(_, v)| v))
            .unwrap();
        let exported = text
            .lines()
            .find_map(|l| l.split_once("Composer 2 stack: ").map(|(_, v)| v))
            .unwrap();
        assert_eq!(reported, exported);
        assert!(text.contains("wrote and verified all 2 tensors"));
        let mut src = gguf::open(source.to_str().unwrap()).unwrap();
        let mut dst = gguf::open(destination.to_str().unwrap()).unwrap();
        assert_eq!(src.tensors.len(), dst.tensors.len());
        for i in 0..src.tensors.len() {
            assert_eq!(dst.tensors[i].ggml_type, dtype);
            let a = gguf::read_tensor_f32(&mut src, i).unwrap();
            let b = gguf::read_tensor_f32(&mut dst, i).unwrap();
            assert!(wq::cosine(&a, &b) >= 0.99);
        }
        assert_eq!(std::fs::read(source).unwrap(), original);
        assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 2);
    }
}

#[test]
fn invalid_options_fail_instead_of_changing_scope() {
    let f = Fixture::new();
    let source = f.model("source.gguf", 1, gguf::GGML_F32, false);
    for args in [
        vec!["--limit"],
        vec!["--limit", "invalid"],
        vec!["--out-gguf"],
        vec!["--unknown"],
    ] {
        assert!(!run(&source, &args).status.success());
    }
}
