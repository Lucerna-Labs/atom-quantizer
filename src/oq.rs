//! OQ03: framed, checksummed records with complete decoder recipes.
//!
//! Header: magic, u32 count, u64 GGUF-prefix length, prefix, CRC64.
//! Record: u64 payload length, payload, CRC64(length + payload).
//! OQ02 remains readable as a legacy representation with code-only integrity.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write};

use crate::encoded::{EncodedTensor, Payload, Transform};
use crate::wq::{CompressedTensor, QuantStrategy};

const MAGIC: &[u8; 4] = b"OQ03";
const LEGACY_MAGIC: &[u8; 4] = b"OQ02";
const MAX_PREFIX: usize = 256 * 1024 * 1024;

const fn crc_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u64) << 56;
        let mut bit = 0;
        while bit < 8 {
            c = if c & (1 << 63) != 0 {
                (c << 1) ^ 0x42F0_E1EB_A9EA_3693
            } else {
                c << 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}
const CRC_TABLE: [u64; 256] = crc_table();

/// CRC-64/ECMA-182, for accidental corruption detection.
pub fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0u64, |c, &b| {
        (c << 8) ^ CRC_TABLE[((c >> 56) as u8 ^ b) as usize]
    })
}

fn strategy_code(s: QuantStrategy) -> u8 {
    match s {
        QuantStrategy::Q2 => 0,
        QuantStrategy::Q4 => 1,
        QuantStrategy::Q8 => 2,
    }
}
fn strategy_from_code(c: u8) -> Result<QuantStrategy, String> {
    match c {
        0 => Ok(QuantStrategy::Q2),
        1 => Ok(QuantStrategy::Q4),
        2 => Ok(QuantStrategy::Q8),
        _ => Err(format!("bad strategy code {c}")),
    }
}
fn rd_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64(r: &mut impl Read) -> Result<u64, String> {
    let mut b = [0; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}
fn rd_u8(r: &mut impl Read) -> Result<u8, String> {
    let mut b = [0];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b[0])
}
fn rd_f32(r: &mut impl Read) -> Result<f32, String> {
    Ok(f32::from_bits(rd_u32(r)?))
}
fn size(n: u64) -> Result<usize, String> {
    usize::try_from(n).map_err(|_| "length does not fit this platform".into())
}
fn read_bytes(r: &mut impl Read, n: usize) -> Result<Vec<u8>, String> {
    let mut b = Vec::new();
    b.try_reserve_exact(n).map_err(|e| e.to_string())?;
    b.resize(n, 0);
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}
fn floats(r: &mut impl Read, n: usize) -> Result<Vec<f32>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|e| e.to_string())?;
    for _ in 0..n {
        v.push(rd_f32(r)?);
    }
    Ok(v)
}

#[derive(Clone, Debug)]
pub struct OqTensor {
    pub name: String,
    pub orig_type: u32,
    pub tensor: EncodedTensor,
}

fn header_bytes(count: u32, prefix: &[u8]) -> Vec<u8> {
    let mut b = Vec::from(*MAGIC);
    b.extend_from_slice(&count.to_le_bytes());
    b.extend_from_slice(&(prefix.len() as u64).to_le_bytes());
    b.extend_from_slice(prefix);
    let crc = checksum(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    b
}

/// Encode precisely the bytes charged to a record, including its frame and CRC.
pub fn encode_record(name: &str, orig_type: u32, t: &EncodedTensor) -> Result<Vec<u8>, String> {
    t.validate()?;
    if t.output_type != crate::encoded::storage_type_for_source(orig_type) {
        return Err("record dtype differs from the reconstruction that was measured".into());
    }
    if name.len() > 65536 {
        return Err("tensor name too long".into());
    }
    let mut b = vec![0; 8]; // frame length, patched before checksumming
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&orig_type.to_le_bytes());
    b.push(t.dims.len() as u8);
    for &d in &t.dims {
        b.extend_from_slice(&(d as u64).to_le_bytes());
    }
    b.push(match t.transform {
        Transform::None => 0,
        Transform::RowHadamard8 => 1,
    });
    b.extend_from_slice(&(t.row_norms.len() as u64).to_le_bytes());
    for &n in &t.row_norms {
        b.extend_from_slice(&n.to_le_bytes());
    }
    match &t.payload {
        Payload::Blocks(c) => {
            b.push(0);
            b.extend_from_slice(&(c.len as u64).to_le_bytes());
            b.push(strategy_code(c.strategy));
            b.extend_from_slice(&c.repair_strength.to_le_bytes());
            b.extend_from_slice(&c.integrity.to_le_bytes());
            b.extend_from_slice(&c.key.to_le_bytes());
            b.extend_from_slice(&(c.scales.len() as u64).to_le_bytes());
            for &s in &c.scales {
                b.extend_from_slice(&s.to_le_bytes());
            }
            b.extend_from_slice(&(c.mins.len() as u64).to_le_bytes());
            for &m in &c.mins {
                b.extend_from_slice(&m.to_le_bytes());
            }
            b.extend_from_slice(&(c.packed.len() as u64).to_le_bytes());
            b.extend_from_slice(&c.packed);
        }
        Payload::F32(values) => {
            b.push(1);
            b.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in values {
                b.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    let payload = (b.len() - 8) as u64;
    b[..8].copy_from_slice(&payload.to_le_bytes());
    let crc = checksum(&b);
    b.extend_from_slice(&crc.to_le_bytes());
    Ok(b)
}

fn bounded_count(r: &mut Cursor<&[u8]>, width: usize) -> Result<usize, String> {
    let n = size(rd_u64(r)?)?;
    let available = r.get_ref().len().saturating_sub(r.position() as usize);
    if n > available / width {
        return Err("field extends past the record".into());
    }
    Ok(n)
}

/// Validate a complete frame before parsing or decoding any numerical metadata.
pub fn decode_record(bytes: &[u8]) -> Result<OqTensor, String> {
    if bytes.len() < 16 {
        return Err("truncated record frame".into());
    }
    let payload = size(u64::from_le_bytes(bytes[..8].try_into().unwrap()))?;
    if payload.checked_add(16) != Some(bytes.len()) {
        return Err("record length mismatch".into());
    }
    let end = bytes.len() - 8;
    let crc = u64::from_le_bytes(bytes[end..].try_into().unwrap());
    if checksum(&bytes[..end]) != crc {
        return Err("record checksum mismatch".into());
    }
    let mut r = Cursor::new(&bytes[8..end]);
    let n = rd_u32(&mut r)? as usize;
    if n > 65536 || n > payload {
        return Err("invalid tensor name length".into());
    }
    let name = String::from_utf8(read_bytes(&mut r, n)?).map_err(|e| e.to_string())?;
    let orig_type = rd_u32(&mut r)?;
    let count = rd_u8(&mut r)? as usize;
    if !(1..=8).contains(&count) {
        return Err("invalid dimension count".into());
    }
    let mut dims = Vec::with_capacity(count);
    for _ in 0..count {
        dims.push(size(rd_u64(&mut r)?)?);
    }
    let transform = match rd_u8(&mut r)? {
        0 => Transform::None,
        1 => Transform::RowHadamard8,
        v => return Err(format!("unknown inverse transform {v}")),
    };
    let count = bounded_count(&mut r, 4)?;
    let row_norms = floats(&mut r, count)?;
    let kind = rd_u8(&mut r)?;
    let tensor_payload = if kind == 1 {
        let count = bounded_count(&mut r, 4)?;
        Payload::F32(floats(&mut r, count)?)
    } else if kind == 0 {
        let len = size(rd_u64(&mut r)?)?;
        let strategy = strategy_from_code(rd_u8(&mut r)?)?;
        let repair_strength = rd_f32(&mut r)?;
        let integrity = rd_u64(&mut r)?;
        let key = rd_u64(&mut r)?;
        let count = bounded_count(&mut r, 4)?;
        let scales = floats(&mut r, count)?;
        let count = bounded_count(&mut r, 4)?;
        let mins = floats(&mut r, count)?;
        let count = bounded_count(&mut r, 1)?;
        let packed = read_bytes(&mut r, count)?;
        Payload::Blocks(CompressedTensor {
            strategy,
            len,
            scales,
            mins,
            packed,
            repair_strength,
            integrity,
            key,
        })
    } else {
        return Err(format!("unknown payload kind {kind}"));
    };
    if r.position() as usize != payload {
        return Err("unexpected record suffix".into());
    }
    let tensor = EncodedTensor {
        payload: tensor_payload,
        dims,
        transform,
        row_norms,
        output_type: crate::encoded::storage_type_for_source(orig_type),
    };
    tensor.validate()?;
    Ok(OqTensor {
        name,
        orig_type,
        tensor,
    })
}

/// The same serialization and decode path used for measurement and file reads.
pub fn roundtrip(t: &EncodedTensor) -> Result<EncodedTensor, String> {
    Ok(decode_record(&encode_record("", t.output_type, t)?)?.tensor)
}

pub struct Writer {
    f: Option<BufWriter<File>>,
    count: u32,
    prefix: Vec<u8>,
    names: HashSet<String>,
    temporary: std::path::PathBuf,
    destination: std::path::PathBuf,
}
impl Writer {
    pub fn create(path: &str) -> Result<Self, String> {
        Self::create_with_header(path, Vec::new())
    }
    pub fn create_with_header(path: &str, prefix: Vec<u8>) -> Result<Self, String> {
        if prefix.len() > MAX_PREFIX {
            return Err("GGUF prefix too large".into());
        }
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let destination = std::path::PathBuf::from(path);
        if destination.symlink_metadata().is_ok() {
            return Err(format!("destination already exists: {path}"));
        }
        let name = destination
            .file_name()
            .ok_or("invalid output path")?
            .to_string_lossy();
        let temporary = destination.with_file_name(format!(
            ".{name}.{}-{}.tmp",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut f = BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temporary)
                .map_err(|e| e.to_string())?,
        );
        f.write_all(&header_bytes(0, &prefix))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            f: Some(f),
            count: 0,
            prefix,
            names: HashSet::new(),
            temporary,
            destination,
        })
    }
    pub fn append(
        &mut self,
        name: &str,
        orig_type: u32,
        c: &CompressedTensor,
    ) -> Result<(), String> {
        let mut tensor = EncodedTensor::plain(c.clone());
        tensor.output_type = crate::encoded::storage_type_for_source(orig_type);
        self.append_encoded(name, orig_type, &tensor).map(|_| ())
    }
    pub fn append_encoded(
        &mut self,
        name: &str,
        orig_type: u32,
        t: &EncodedTensor,
    ) -> Result<usize, String> {
        if self.names.contains(name) {
            return Err(format!("duplicate tensor {name}"));
        }
        let b = encode_record(name, orig_type, t)?;
        self.f
            .as_mut()
            .ok_or("writer already finished")?
            .write_all(&b)
            .map_err(|e| e.to_string())?;
        self.count = self.count.checked_add(1).ok_or("tensor count overflow")?;
        self.names.insert(name.to_string());
        Ok(b.len())
    }
    pub fn finish(mut self) -> Result<u32, String> {
        let mut f = self
            .f
            .take()
            .ok_or("writer already finished")?
            .into_inner()
            .map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        f.write_all(&header_bytes(self.count, &self.prefix))
            .map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        drop(f);
        verify_file(self.temporary.to_str().ok_or("non-UTF8 output path")?)?;
        std::fs::hard_link(&self.temporary, &self.destination).map_err(|e| e.to_string())?;
        Ok(self.count)
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.f.take();
        let _ = std::fs::remove_file(&self.temporary);
    }
}

pub struct Reader {
    r: BufReader<File>,
    remaining: u32,
    pub count: u32,
    pub legacy: bool,
    pub model_header: Vec<u8>,
    file_bytes: u64,
    names: HashSet<String>,
}
impl Reader {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
        let mut r = BufReader::new(file);
        let mut magic = [0; 4];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        let legacy = if &magic == MAGIC {
            false
        } else if &magic == LEGACY_MAGIC {
            true
        } else {
            return Err("not an OQ02/OQ03 file".into());
        };
        let count = rd_u32(&mut r)?;
        if count > 1 << 20 {
            return Err("absurd tensor count".into());
        }
        let model_header = if legacy {
            Vec::new()
        } else {
            let n = size(rd_u64(&mut r)?)?;
            if n > MAX_PREFIX || n as u64 > file_bytes.saturating_sub(24) {
                return Err("invalid GGUF prefix length".into());
            }
            let prefix = read_bytes(&mut r, n)?;
            let expected = rd_u64(&mut r)?;
            let header = header_bytes(count, &prefix);
            if checksum(&header[..header.len() - 8]) != expected {
                return Err("container header checksum mismatch".into());
            }
            prefix
        };
        Ok(Self {
            r,
            remaining: count,
            count,
            legacy,
            model_header,
            file_bytes,
            names: HashSet::new(),
        })
    }
    pub fn next_tensor(&mut self) -> Result<Option<OqTensor>, String> {
        if self.remaining == 0 {
            if self.r.stream_position().map_err(|e| e.to_string())? != self.file_bytes {
                return Err("trailing bytes or uncounted records".into());
            }
            return Ok(None);
        }
        let rec = if self.legacy {
            read_legacy_record(&mut self.r, self.file_bytes)?
        } else {
            let n = rd_u64(&mut self.r)?;
            let pos = self.r.stream_position().map_err(|e| e.to_string())?;
            if n > self.file_bytes.saturating_sub(pos).saturating_sub(8) {
                return Err("truncated record payload".into());
            }
            let mut b = Vec::new();
            let len = size(n)?.checked_add(16).ok_or("record size overflow")?;
            b.try_reserve_exact(len).map_err(|e| e.to_string())?;
            b.extend_from_slice(&n.to_le_bytes());
            b.resize(len, 0);
            self.r.read_exact(&mut b[8..]).map_err(|e| e.to_string())?;
            decode_record(&b)?
        };
        if !self.names.insert(rec.name.clone()) {
            return Err(format!("duplicate tensor {}", rec.name));
        }
        self.remaining -= 1;
        Ok(Some(rec))
    }
}

fn read_legacy_record(r: &mut BufReader<File>, file_bytes: u64) -> Result<OqTensor, String> {
    let n = rd_u32(r)? as usize;
    if n > 65536 {
        return Err("invalid legacy tensor name length".into());
    }
    let name = String::from_utf8(read_bytes(r, n)?).map_err(|e| e.to_string())?;
    let orig_type = rd_u32(r)?;
    let len = size(rd_u64(r)?)?;
    let strategy = strategy_from_code(rd_u8(r)?)?;
    let repair_strength = rd_f32(r)?;
    let integrity = rd_u64(r)?;
    let key = rd_u64(r)?;
    let n = rd_u32(r)? as usize;
    if n != len.div_ceil(crate::wq::BLOCK) || n as u64 > file_bytes / 4 {
        return Err("invalid legacy scale count".into());
    }
    let scales = floats(r, n)?;
    let n = rd_u32(r)? as usize;
    if n != 0 && n != scales.len() {
        return Err("invalid legacy minimum count".into());
    }
    let mins = floats(r, n)?;
    let n = size(rd_u64(r)?)?;
    let expected = len
        .checked_mul(crate::wq::bits(strategy) as usize)
        .ok_or("legacy length overflow")?
        .div_ceil(8);
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    if n != expected || n as u64 > file_bytes.saturating_sub(pos) {
        return Err("invalid legacy packed length".into());
    }
    let packed = read_bytes(r, n)?;
    let tensor = EncodedTensor::plain(CompressedTensor {
        strategy,
        len,
        scales,
        mins,
        packed,
        repair_strength,
        integrity,
        key,
    });
    tensor.validate()?;
    Ok(OqTensor {
        name,
        orig_type,
        tensor,
    })
}

pub struct OqStats {
    pub count: u32,
    pub all_verified: bool,
    pub file_bytes: u64,
    pub full_record_integrity: bool,
    pub first: Option<(String, u32, usize)>,
}

/// Verify and actually decode every record. Errors are returned, never hidden.
pub fn verify_file(path: &str) -> Result<OqStats, String> {
    let mut r = Reader::open(path)?;
    let mut first = None;
    while let Some(rec) = r.next_tensor()? {
        let decoded = rec.tensor.decode()?;
        if first.is_none() {
            first = Some((rec.name, rec.orig_type, decoded.len()));
        }
    }
    Ok(OqStats {
        count: r.count,
        all_verified: true,
        file_bytes: r.file_bytes,
        full_record_integrity: !r.legacy,
        first,
    })
}

/// Rebuild a complete dense GGUF using only OQ03 records and stored metadata.
/// The original weights are never read. Source F32/F16/BF16 storage dtypes are
/// preserved; legacy files and partial containers cannot supply a whole model.
pub fn decode_to_gguf(path: &str, destination: &str) -> Result<u32, String> {
    let mut reader = Reader::open(path)?;
    if reader.legacy || reader.model_header.len() < 24 {
        return Err("model export requires OQ03 with the source GGUF metadata".into());
    }
    if &reader.model_header[..4] != b"GGUF" {
        return Err("invalid stored GGUF metadata".into());
    }
    let expected = u64::from_le_bytes(reader.model_header[8..16].try_into().unwrap());
    if expected != reader.count as u64 {
        return Err("partial container: tensor count differs from the stored model".into());
    }
    let output = std::path::Path::new(destination);
    if output.exists() {
        return Err(format!("destination already exists: {destination}"));
    }
    let filename = output
        .file_name()
        .ok_or("invalid output path")?
        .to_string_lossy();
    let temporary =
        output.with_file_name(format!(".{filename}.{}.oqdecode.tmp", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(&reader.model_header)
            .map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        let g = crate::gguf::open(temporary.to_str().ok_or("non-UTF8 output path")?)?;
        if g.data_start != reader.model_header.len() as u64 {
            return Err("stored GGUF prefix has the wrong length".into());
        }
        let metas: std::collections::HashMap<_, _> =
            g.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
        if metas.len() != g.tensors.len() {
            return Err("duplicate tensor in model metadata".into());
        }
        for t in &g.tensors {
            if !crate::gguf::writable_passthrough(t.ggml_type) {
                return Err(format!(
                    "GGUF export cannot encode source dtype {} for {}",
                    crate::gguf::type_name(t.ggml_type),
                    t.name
                ));
            }
        }
        while let Some(rec) = reader.next_tensor()? {
            let t = metas
                .get(rec.name.as_str())
                .ok_or("record missing from model metadata")?;
            if t.dims != rec.tensor.dims || t.ggml_type != rec.orig_type {
                return Err(format!(
                    "record and model metadata disagree for {}",
                    rec.name
                ));
            }
            crate::gguf::write_tensor_reconstruction(
                &mut file,
                t,
                g.data_start,
                &rec.tensor.decode()?,
            )?;
        }
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(reader.count)
    })();
    drop(file);
    match result {
        Ok(count) => {
            // Hard-link publication fails if a destination appeared meanwhile.
            // Both paths are in the same directory and therefore filesystem.
            if let Err(e) = std::fs::hard_link(&temporary, output) {
                let _ = std::fs::remove_file(&temporary);
                return Err(e.to_string());
            }
            std::fs::remove_file(&temporary).map_err(|e| e.to_string())?;
            Ok(count)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&temporary);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{stacks, wq};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn path() -> String {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/oq-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!(
            "{}-{}.oq",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ))
        .to_str()
        .unwrap()
        .to_string()
    }
    fn fixture() -> (Vec<f32>, EncodedTensor) {
        let w: Vec<f32> = (0..128).map(|i| (i as f32 * 0.37).sin() * 0.08).collect();
        let mut rotated = w.clone();
        stacks::row_hadamard_8_apply(&mut rotated, 4, 32);
        let t = EncodedTensor {
            payload: Payload::Blocks(wq::encode_with(
                &rotated,
                QuantStrategy::Q4,
                &wq::WqConfig::default(),
            )),
            dims: vec![32, 4],
            transform: Transform::RowHadamard8,
            row_norms: stacks::row_l2_norms(&w, 4, 32),
            output_type: crate::gguf::GGML_F32,
        };
        (w, t)
    }

    #[test]
    fn crc64_matches_ecma_check_vector() {
        assert_eq!(checksum(b"123456789"), 0x6C40_DF5F_0B49_7347);
    }

    #[test]
    fn complete_recipe_survives_disk_and_recovers_scored_values() {
        let (w, t) = fixture();
        let expected = t.decode().unwrap();
        let p = path();
        let mut writer = Writer::create(&p).unwrap();
        let bytes = writer.append_encoded("probe", 0, &t).unwrap();
        writer.finish().unwrap();
        assert_eq!(std::fs::metadata(&p).unwrap().len(), (24 + bytes) as u64);
        let mut r = Reader::open(&p).unwrap();
        let read = r.next_tensor().unwrap().unwrap();
        assert_eq!(read.tensor.decode().unwrap(), expected);
        assert!(wq::cosine(&w, &expected) > 0.98);
        assert!(r.next_tensor().unwrap().is_none());
        assert!(verify_file(&p).unwrap().full_record_integrity);
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn every_single_byte_mutation_including_metadata_is_detected() {
        let (_, mut t) = fixture();
        t.output_type = crate::gguf::GGML_BF16;
        let b = encode_record("probe", 30, &t).unwrap();
        for i in 0..b.len() {
            let mut corrupt = b.clone();
            corrupt[i] ^= 1;
            assert!(decode_record(&corrupt).is_err(), "undetected byte {i}");
        }
    }

    #[test]
    fn every_record_truncation_is_rejected() {
        let (_, t) = fixture();
        let b = encode_record("probe", 0, &t).unwrap();
        for i in 0..b.len() {
            assert!(decode_record(&b[..i]).is_err());
        }
    }

    #[test]
    fn file_count_and_trailing_bytes_are_checked() {
        let (_, t) = fixture();
        let p = path();
        let mut writer = Writer::create(&p).unwrap();
        writer.append_encoded("probe", 0, &t).unwrap();
        writer.finish().unwrap();
        let b = std::fs::read(&p).unwrap();
        let mut corrupt = b.clone();
        corrupt[4] ^= 1;
        std::fs::write(&p, &corrupt).unwrap();
        assert!(Reader::open(&p).is_err());
        let mut suffix = b;
        suffix.push(0);
        std::fs::write(&p, suffix).unwrap();
        assert!(verify_file(&p).is_err());
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn legacy_oq02_is_explicit_and_decodes_original_representation() {
        let c = wq::encode_with(&[0.03; 32], QuantStrategy::Q4, &wq::WqConfig::default());
        let mut b = Vec::from(*LEGACY_MAGIC);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.push(b'w');
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&(c.len as u64).to_le_bytes());
        b.push(1);
        b.extend_from_slice(&c.repair_strength.to_le_bytes());
        b.extend_from_slice(&c.integrity.to_le_bytes());
        b.extend_from_slice(&c.key.to_le_bytes());
        b.extend_from_slice(&(c.scales.len() as u32).to_le_bytes());
        for &s in &c.scales {
            b.extend_from_slice(&s.to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&(c.packed.len() as u64).to_le_bytes());
        b.extend_from_slice(&c.packed);
        let p = path();
        std::fs::write(&p, b).unwrap();
        let mut r = Reader::open(&p).unwrap();
        assert!(r.legacy);
        assert_eq!(
            r.next_tensor().unwrap().unwrap().tensor.decode().unwrap(),
            wq::expand(&c)
        );
        assert!(!verify_file(&p).unwrap().full_record_integrity);
        std::fs::remove_file(p).unwrap();
    }
}
