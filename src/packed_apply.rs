//! File boundary for one verified packed matrix operation.
use atom_quantizer::{a22, calibration};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let archive = args.first().ok_or("missing A22 archive")?;
    let mut options = BTreeMap::new();
    let mut rest = args[1..].iter();
    while let Some(flag) = rest.next() {
        if !["--tensor", "--input", "--batch", "--out"].contains(&flag.as_str()) {
            return Err(format!("unknown packed-operation option {flag}"));
        }
        let value = rest
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if options.insert(flag.as_str(), value.as_str()).is_some() {
            return Err(format!("repeated option {flag}"));
        }
    }
    if options.len() != 4 {
        return Err("require --tensor NAME --input F32_FILE --batch N --out NEW_F32_FILE".into());
    }
    Ok((archive, options))
}

pub fn run(args: &[String]) -> Result<Value, String> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let (archive, options) = arguments(args)?;
    let get = |key| {
        options
            .get(key)
            .copied()
            .ok_or_else(|| format!("missing {key}"))
    };
    let name = get("--tensor")?;
    let input = get("--input")?;
    let batch = get("--batch")?
        .parse::<usize>()
        .map_err(|_| "invalid batch count")?;
    if batch == 0 {
        return Err("batch count must be nonzero".into());
    }
    let destination = Path::new(get("--out")?);
    if destination.symlink_metadata().is_ok() {
        return Err("destination already exists".into());
    }
    let source_hash = calibration::file_sha256(archive)?;
    let matrix = a22::load_tensor(archive, name)?;
    if matrix.shape().len() != 2 {
        return Err("selected tensor is not a matrix".into());
    }
    let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
    let input_bytes = batch
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or("input extent overflow")?;
    if File::open(input)
        .and_then(|f| f.metadata())
        .map_err(|e| e.to_string())?
        .len()
        != input_bytes as u64
    {
        return Err("input file length differs from batch and matrix columns".into());
    }
    let bytes = std::fs::read(input).map_err(|e| e.to_string())?;
    if bytes.len() != input_bytes {
        return Err("input file changed size".into());
    }
    let input_hash: [u8; 32] = Sha256::digest(&bytes).into();
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    let output = matrix.matmul(&values, batch, a22::Kernel::Fused)?;
    if calibration::file_sha256(archive)? != source_hash
        || calibration::file_sha256(input)? != input_hash
    {
        return Err("packed-operation inputs changed during execution".into());
    }
    let filename = destination
        .file_name()
        .ok_or("invalid output path")?
        .to_string_lossy();
    let staged = destination.with_file_name(format!(
        ".{filename}.{}-{}.packed.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)
        .map_err(|e| e.to_string())?;
    let temporary = Temporary(staged);
    let mut writer = BufWriter::new(file);
    let mut hash = Sha256::new();
    for value in &output {
        let data = value.to_le_bytes();
        writer.write_all(&data).map_err(|e| e.to_string())?;
        hash.update(data);
    }
    writer.flush().map_err(|e| e.to_string())?;
    writer.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(writer);
    std::fs::hard_link(&temporary.0, destination).map_err(|e| e.to_string())?;
    Ok(
        json!({"status":"PUBLISHED", "scope":"selected_tensor", "backend":"native_cpu_packed",
        "kernel":"fused", "tensor":name, "rows":rows, "columns":cols, "batch":batch,
        "input_layout":"batch_by_columns_f32_le", "output_layout":"batch_by_rows_f32_le",
        "archive_sha256":hex(&source_hash), "input_sha256":hex(&input_hash),
        "output_sha256":format!("{:x}", hash.finalize()), "output_bytes":output.len()*4,
        "owned_tensor_bytes":matrix.owned_bytes(), "output":destination.to_string_lossy()}),
    )
}
