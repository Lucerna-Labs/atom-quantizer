//! Native verification/export of committed A22-2 scalar and rank-two recipes.
//! Encoder fitting remains separate. Unsupported decoder state is rejected.
mod header;
mod record;
mod tensor;

pub use tensor::{DenseTensor, Kernel, PackedTensor};

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use zip::{CompressionMethod, ZipArchive};

const MAX_RECORD: usize = 2 * 1024 * 1024 * 1024;
const MAX_PREFIX: usize = 256 * 1024 * 1024;
const MAX_MANIFEST: usize = 64 * 1024 * 1024;

fn number(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("invalid integer field {key}"))
}
fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("invalid string field {key}"))
}
fn shape(value: &Value, zero: bool) -> Result<Vec<usize>, String> {
    let items = value.as_array().ok_or("invalid shape")?;
    if items.len() > 8 || (!zero && items.is_empty()) {
        return Err("invalid shape rank".into());
    }
    items
        .iter()
        .map(|v| {
            let n = usize::try_from(v.as_u64().ok_or("invalid shape dimension")?)
                .map_err(|_| "dimension overflow")?;
            if !zero && n == 0 {
                return Err("empty tensor dimension".into());
            }
            Ok(n)
        })
        .collect()
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(value: &Value, key: &str) -> Result<String, String> {
    let result = text(value, key)?;
    if result.len() != 64
        || !result
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(format!("invalid digest {key}"));
    }
    Ok(result.into())
}

/// ZipArchive indexes names uniquely. Inspect the raw directory too so duplicate
/// central-directory entries cannot disappear behind that lookup table.
fn directory_names(path: &str, start: u64, length: u64) -> Result<BTreeSet<String>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(start))
        .map_err(|e| e.to_string())?;
    let mut names = BTreeSet::new();
    loop {
        let mut signature = [0u8; 4];
        file.read_exact(&mut signature).map_err(|e| e.to_string())?;
        if signature == *b"PK\x05\x06" || signature == *b"PK\x06\x06" {
            break;
        }
        if signature != *b"PK\x01\x02" || names.len() >= (1 << 20) + 2 {
            return Err("invalid ZIP central directory".into());
        }
        let mut header = [0u8; 42];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;
        let name_len = u16::from_le_bytes([header[24], header[25]]) as usize;
        let extra = u16::from_le_bytes([header[26], header[27]]) as u64;
        let comment = u16::from_le_bytes([header[28], header[29]]) as u64;
        let mut bytes = vec![0u8; name_len];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let name = String::from_utf8(bytes).map_err(|_| "invalid ZIP member UTF-8")?;
        if name.is_empty() || !names.insert(name) {
            return Err("empty or duplicate ZIP member".into());
        }
        let position = file.stream_position().map_err(|e| e.to_string())?;
        let end = position
            .checked_add(extra)
            .and_then(|n| n.checked_add(comment))
            .ok_or("ZIP directory extent overflow")?;
        if end > length {
            return Err("ZIP directory extends past file".into());
        }
        file.seek(SeekFrom::Start(end)).map_err(|e| e.to_string())?;
    }
    Ok(names)
}

fn member(zip: &mut ZipArchive<File>, name: &str, limit: usize) -> Result<Vec<u8>, String> {
    let entry = zip.by_name(name).map_err(|e| e.to_string())?;
    if entry.compression() != CompressionMethod::Stored || entry.size() > limit as u64 {
        return Err(format!(
            "unsupported compression or oversized ZIP member {name}"
        ));
    }
    let size = usize::try_from(entry.size()).map_err(|_| "ZIP member size overflow")?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|e| e.to_string())?;
    entry
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() != size {
        return Err("ZIP member length mismatch".into());
    }
    Ok(bytes)
}

struct Tensor {
    name: String,
    member: String,
    bytes: usize,
    record_hash: String,
    decoded_hash: String,
    extent: header::Extent,
}

struct Reader {
    zip: ZipArchive<File>,
    prefix: Vec<u8>,
    tensors: Vec<Tensor>,
    source_bytes: u64,
    artifact_bytes: u64,
}

impl Reader {
    fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let artifact_bytes = file.metadata().map_err(|e| e.to_string())?.len();
        let mut zip = ZipArchive::new(file).map_err(|e| e.to_string())?;
        if zip.offset() != 0 || zip.len() > (1 << 20) + 2 {
            return Err("unsupported ZIP layout/count".into());
        }
        let names = directory_names(path, zip.central_directory_start(), artifact_bytes)?;
        if names.len() != zip.len()
            || names.iter().map(String::as_str).collect::<BTreeSet<_>>()
                != zip.file_names().collect()
        {
            return Err("ZIP directory names/count differ".into());
        }
        if zip.has_overlapping_files().map_err(|e| e.to_string())? {
            return Err("overlapping ZIP members".into());
        }
        let manifest: Value =
            serde_json::from_slice(&member(&mut zip, "manifest.json", MAX_MANIFEST)?)
                .map_err(|e| e.to_string())?;
        if text(&manifest, "format")? != "A22-2" {
            return Err("native A22 requires committed format A22-2".into());
        }
        let prefix = member(&mut zip, "gguf-prefix.bin", MAX_PREFIX)?;
        if hash(&prefix) != digest(&manifest, "prefix_sha256")? {
            return Err("GGUF prefix checksum mismatch".into());
        }
        let header = header::parse(&prefix)?;
        let source_bytes = number(&manifest, "source_bytes")?;
        if source_bytes < header.end || source_bytes > header.padded_end {
            return Err("source size differs from stored GGUF extents".into());
        }
        digest(&manifest, "source_sha256")?;
        let entries = manifest
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or("missing tensor manifest")?;
        if entries.len() != header.tensors.len() {
            return Err("incomplete tensor manifest".into());
        }
        let mut expected =
            BTreeSet::from(["manifest.json".to_string(), "gguf-prefix.bin".to_string()]);
        let mut extents = header.tensors;
        let mut tensors = Vec::new();
        for entry in entries {
            let name = text(entry, "name")?.to_string();
            let extent = extents
                .remove(&name)
                .ok_or("unknown or duplicate manifest tensor")?;
            let member = text(entry, "entry")?.to_string();
            if !expected.insert(member.clone()) {
                return Err("duplicate or reserved tensor member".into());
            }
            if shape(entry.get("shape").ok_or("missing manifest shape")?, false)? != extent.shape
                || number(entry, "gguf_offset")? != extent.offset
            {
                return Err("manifest tensor differs from authoritative GGUF table".into());
            }
            let bytes =
                usize::try_from(number(entry, "bytes")?).map_err(|_| "record length overflow")?;
            if bytes == 0 || bytes > MAX_RECORD {
                return Err("invalid record member size".into());
            }
            tensors.push(Tensor {
                name,
                member,
                bytes,
                record_hash: digest(entry, "sha256")?,
                decoded_hash: digest(entry, "decoded_sha256")?,
                extent,
            });
        }
        if !extents.is_empty() || expected != names {
            return Err("ZIP member set differs from manifest".into());
        }
        Ok(Self {
            zip,
            prefix,
            tensors,
            source_bytes,
            artifact_bytes,
        })
    }

    fn prepared(&mut self, index: usize) -> Result<PackedTensor, String> {
        let tensor = self.tensors.get(index).ok_or("unknown tensor index")?;
        let bytes = member(&mut self.zip, &tensor.member, MAX_RECORD)?;
        if bytes.len() != tensor.bytes || hash(&bytes) != tensor.record_hash {
            return Err(format!(
                "record checksum/length differs for {}",
                tensor.name
            ));
        }
        let record = record::Record::parse(&bytes)?;
        drop(bytes);
        let prepared = record.prepare(&tensor.extent.shape)?;
        let mut checksum = Sha256::new();
        let mut count = 0u64;
        prepared.decode_chunks(|chunk| {
            count = count
                .checked_add(chunk.len() as u64)
                .ok_or("decoded length overflow")?;
            if count > tensor.extent.bytes {
                return Err("decoded tensor exceeds stored extent".into());
            }
            checksum.update(chunk);
            Ok(())
        })?;
        if count != tensor.extent.bytes
            || format!("{:x}", checksum.finalize()) != tensor.decoded_hash
        {
            return Err(format!("decoded commitment mismatch for {}", tensor.name));
        }
        Ok(prepared)
    }

    fn decode(&mut self, mut output: Option<&mut BufWriter<File>>) -> Result<(), String> {
        for tensor in &self.tensors {
            let bytes = member(&mut self.zip, &tensor.member, MAX_RECORD)?;
            if bytes.len() != tensor.bytes || hash(&bytes) != tensor.record_hash {
                return Err(format!(
                    "record checksum/length differs for {}",
                    tensor.name
                ));
            }
            let record = record::Record::parse(&bytes)?;
            if let Some(file) = output.as_mut() {
                file.seek(SeekFrom::Start(tensor.extent.offset))
                    .map_err(|e| e.to_string())?;
            }
            let mut checksum = Sha256::new();
            let mut decoded_bytes = 0u64;
            record.decode(&tensor.extent.shape, |chunk| {
                decoded_bytes = decoded_bytes
                    .checked_add(chunk.len() as u64)
                    .ok_or("decoded length overflow")?;
                if decoded_bytes > tensor.extent.bytes {
                    return Err("decoded tensor exceeds stored extent".into());
                }
                checksum.update(chunk);
                if let Some(file) = output.as_mut() {
                    file.write_all(chunk).map_err(|e| e.to_string())?;
                }
                Ok(())
            })?;
            let actual = format!("{:x}", checksum.finalize());
            if decoded_bytes != tensor.extent.bytes || actual != tensor.decoded_hash {
                return Err(format!(
                    "decoded commitment mismatch for {}: expected {}, got {}",
                    tensor.name, tensor.decoded_hash, actual
                ));
            }
        }
        Ok(())
    }
}

pub struct Stats {
    pub tensors: usize,
    pub artifact_bytes: u64,
    pub decoded_bytes: u64,
}

/// Verify and prepare only the selected tensor; other records are not decoded.
pub fn load_tensor(path: &str, name: &str) -> Result<PackedTensor, String> {
    let before = crate::calibration::file_sha256(path)?;
    let mut reader = Reader::open(path)?;
    let index = reader
        .tensors
        .iter()
        .position(|t| t.name == name)
        .ok_or("tensor not found")?;
    let tensor = reader.prepared(index)?;
    if crate::calibration::file_sha256(path)? != before {
        return Err("archive changed during tensor loading".into());
    }
    Ok(tensor)
}

pub struct StoredModel<T> {
    pub tensors: Vec<(String, T)>,
    pub prefix: Vec<u8>,
    pub artifact_sha256: [u8; 32],
}

fn load_model_as<T, F: FnMut(PackedTensor) -> Result<T, String>>(
    path: &str,
    mut convert: F,
) -> Result<StoredModel<T>, String> {
    let before = crate::calibration::file_sha256(path)?;
    let mut reader = Reader::open(path)?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(reader.tensors.len())
        .map_err(|e| e.to_string())?;
    for index in 0..reader.tensors.len() {
        let name = reader.tensors[index].name.clone();
        tensors.push((name, convert(reader.prepared(index)?)?));
    }
    if crate::calibration::file_sha256(path)? != before {
        return Err("archive changed during model loading".into());
    }
    Ok(StoredModel {
        tensors,
        prefix: reader.prefix,
        artifact_sha256: before,
    })
}

/// Load and verify every stored tensor, retaining packed representations.
pub fn load_packed_model(path: &str) -> Result<StoredModel<PackedTensor>, String> {
    load_model_as(path, Ok)
}

/// Matched verification/loading control that retains decoded F32 weights.
pub fn load_dense_model(path: &str) -> Result<StoredModel<DenseTensor>, String> {
    load_model_as(path, |tensor| tensor.to_dense())
}

fn functional_config(config: &Value, initial: bool) -> Result<(), String> {
    let fields = config
        .as_object()
        .ok_or("invalid functional configuration")?;
    if !initial
        && fields.len() == 1
        && config.get("family").and_then(Value::as_str) == Some("lossless")
    {
        return Ok(());
    }
    if fields.len() != 3
        || !fields.contains_key("bits")
        || !fields.contains_key("rank")
        || !fields.contains_key("rank_geometry")
        || (number(config, "bits")? != 6 && (initial || number(config, "bits")? != 8))
        || number(config, "rank")? != 2
        || text(config, "rank_geometry")? != "activation"
    {
        return Err("archive does not use the fixed functional-six profile".into());
    }
    Ok(())
}

/// Check the encoder provenance and stored recipe requested by build-functional.
/// This does not replace native reconstruction and independent fidelity checks.
pub fn validate_functional_profile(
    path: &str,
    source: &str,
    base: &str,
    residual: &str,
) -> Result<(), String> {
    let mut reader = Reader::open(path)?;
    let manifest: Value =
        serde_json::from_slice(&member(&mut reader.zip, "manifest.json", MAX_MANIFEST)?)
            .map_err(|e| e.to_string())?;
    if digest(&manifest, "source_sha256")? != source
        || digest(&manifest, "calibration_sha256")? != base
        || digest(&manifest, "residual_calibration_sha256")? != residual
    {
        return Err("functional archive source/calibration provenance differs".into());
    }
    functional_config(
        manifest
            .get("config")
            .ok_or("missing functional configuration")?,
        true,
    )?;
    let entries = manifest
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or("missing functional tensor table")?;
    for (tensor, entry) in reader.tensors.iter().zip(entries) {
        let config = entry.get("config").ok_or("missing tensor configuration")?;
        functional_config(config, false)?;
        let data = member(&mut reader.zip, &tensor.member, MAX_RECORD)?;
        if data.len() != tensor.bytes || hash(&data) != tensor.record_hash {
            return Err("functional record checksum differs".into());
        }
        record::Record::parse(&data)?.validate_functional_config(config)?;
    }
    Ok(())
}

pub fn is_archive(path: &str) -> Result<bool, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).map_err(|e| e.to_string())?;
    Ok(magic == *b"PK\x03\x04")
}

pub fn verify_file(path: &str) -> Result<Stats, String> {
    let before = crate::calibration::file_sha256(path)?;
    let mut reader = Reader::open(path)?;
    reader.decode(None)?;
    if crate::calibration::file_sha256(path)? != before {
        return Err("archive changed during verification".into());
    }
    Ok(Stats {
        tensors: reader.tensors.len(),
        artifact_bytes: reader.artifact_bytes,
        decoded_bytes: reader.source_bytes,
    })
}

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn open_staging(path: PathBuf) -> Result<(File, Temporary), String> {
    let file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    Ok((file, Temporary(path)))
}

pub fn decode_to_gguf(path: &str, destination: &str) -> Result<Stats, String> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let destination = Path::new(destination);
    if destination.symlink_metadata().is_ok() {
        return Err("destination already exists".into());
    }
    let before = crate::calibration::file_sha256(path)?;
    let mut reader = Reader::open(path)?;
    let filename = destination
        .file_name()
        .ok_or("invalid output path")?
        .to_string_lossy();
    let temporary_path = destination.with_file_name(format!(
        ".{filename}.{}-{}.a22.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let (file, temporary) = open_staging(temporary_path)?;
    let mut file = BufWriter::new(file);
    file.write_all(&reader.prefix).map_err(|e| e.to_string())?;
    reader.decode(Some(&mut file))?;
    file.flush().map_err(|e| e.to_string())?;
    file.get_ref()
        .set_len(reader.source_bytes)
        .map_err(|e| e.to_string())?;
    file.get_ref().sync_all().map_err(|e| e.to_string())?;
    drop(file);
    if crate::calibration::file_sha256(path)? != before {
        return Err("archive changed during export".into());
    }
    std::fs::hard_link(&temporary.0, destination).map_err(|e| e.to_string())?;
    Ok(Stats {
        tensors: reader.tensors.len(),
        artifact_bytes: reader.artifact_bytes,
        decoded_bytes: reader.source_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staging_collision_does_not_delete_the_existing_file() {
        let root = std::env::temp_dir().join(format!("a22-staging-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("existing");
        std::fs::write(&path, b"preserve").unwrap();
        assert!(open_staging(path.clone()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"preserve");
        std::fs::remove_dir_all(root).unwrap();
    }
}
