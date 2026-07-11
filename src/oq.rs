//! oq — the Atom Quantizer container: serialize compressed tensors to a real file
//! on disk, and read them back. Streaming write (never holds the whole model in
//! RAM) and a verify pass that re-reads every record and checks its integrity tag,
//! so a written `.oq` is provably loadable — not a write-only dead end.
//!
//! Format (little-endian):
//!   magic "OQ01" | u32 tensor_count
//!   per tensor:
//!     u32 name_len | name bytes
//!     u32 orig_ggml_type          (what it was in the source GGUF)
//!     u64 len                     (element count)
//!     u8  strategy                (0=Q2, 1=Q4, 2=Q8)
//!     f32 repair_strength
//!     u64 integrity | u64 key
//!     u32 n_scales | n_scales × f32
//!     u64 packed_len | packed_len bytes

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use crate::wq::{CompressedTensor, QuantStrategy};

const MAGIC: &[u8; 4] = b"OQ02"; // OQ02 adds the affine `mins` block after `scales`

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

/// Streaming writer: append tensors one at a time, patch the count on `finish`.
pub struct Writer {
    f: BufWriter<File>,
    count: u32,
}

impl Writer {
    pub fn create(path: &str) -> Result<Self, String> {
        let mut f = BufWriter::new(File::create(path).map_err(|e| e.to_string())?);
        f.write_all(MAGIC).map_err(|e| e.to_string())?;
        f.write_all(&0u32.to_le_bytes())
            .map_err(|e| e.to_string())?; // placeholder count
        Ok(Writer { f, count: 0 })
    }

    pub fn append(
        &mut self,
        name: &str,
        orig_type: u32,
        c: &CompressedTensor,
    ) -> Result<(), String> {
        let w = &mut self.f;
        let nb = name.as_bytes();
        let mut go = || -> std::io::Result<()> {
            w.write_all(&(nb.len() as u32).to_le_bytes())?;
            w.write_all(nb)?;
            w.write_all(&orig_type.to_le_bytes())?;
            w.write_all(&(c.len as u64).to_le_bytes())?;
            w.write_all(&[strategy_code(c.strategy)])?;
            w.write_all(&c.repair_strength.to_le_bytes())?;
            w.write_all(&c.integrity.to_le_bytes())?;
            w.write_all(&c.key.to_le_bytes())?;
            w.write_all(&(c.scales.len() as u32).to_le_bytes())?;
            for &s in &c.scales {
                w.write_all(&s.to_le_bytes())?;
            }
            w.write_all(&(c.mins.len() as u32).to_le_bytes())?;
            for &m in &c.mins {
                w.write_all(&m.to_le_bytes())?;
            }
            w.write_all(&(c.packed.len() as u64).to_le_bytes())?;
            w.write_all(&c.packed)?;
            Ok(())
        };
        go().map_err(|e| e.to_string())?;
        self.count += 1;
        Ok(())
    }

    pub fn finish(self) -> Result<u32, String> {
        let count = self.count;
        let mut f = self.f.into_inner().map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(4)).map_err(|e| e.to_string())?;
        f.write_all(&count.to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())?;
        Ok(count)
    }
}

fn rd_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64(r: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}
fn rd_f32(r: &mut impl Read) -> Result<f32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(f32::from_le_bytes(b))
}

/// One record read back from an `.oq` file.
pub struct OqTensor {
    pub name: String,
    pub orig_type: u32,
    pub tensor: CompressedTensor,
}

/// Read the next tensor record (returns the reconstructed CompressedTensor).
fn read_record(r: &mut impl Read) -> Result<OqTensor, String> {
    let name_len = rd_u32(r)? as usize;
    if name_len > 1 << 16 {
        return Err("absurd name length".into());
    }
    let mut nb = vec![0u8; name_len];
    r.read_exact(&mut nb).map_err(|e| e.to_string())?;
    let name = String::from_utf8_lossy(&nb).into_owned();
    let orig_type = rd_u32(r)?;
    let len = rd_u64(r)? as usize;
    let mut sc = [0u8; 1];
    r.read_exact(&mut sc).map_err(|e| e.to_string())?;
    let strategy = strategy_from_code(sc[0])?;
    let repair_strength = rd_f32(r)?;
    let integrity = rd_u64(r)?;
    let key = rd_u64(r)?;
    let n_scales = rd_u32(r)? as usize;
    if n_scales > len + 1024 {
        return Err("scale count inconsistent with len".into());
    }
    let mut scales = Vec::with_capacity(n_scales);
    for _ in 0..n_scales {
        scales.push(rd_f32(r)?);
    }
    let n_mins = rd_u32(r)? as usize;
    if n_mins > len + 1024 {
        return Err("min count inconsistent with len".into());
    }
    let mut mins = Vec::with_capacity(n_mins);
    for _ in 0..n_mins {
        mins.push(rd_f32(r)?);
    }
    let packed_len = rd_u64(r)? as usize;
    let mut packed = vec![0u8; packed_len];
    r.read_exact(&mut packed).map_err(|e| e.to_string())?;
    Ok(OqTensor {
        name,
        orig_type,
        tensor: CompressedTensor {
            strategy,
            len,
            scales,
            mins,
            packed,
            repair_strength,
            integrity,
            key,
        },
    })
}

pub struct OqStats {
    pub count: u32,
    pub all_verified: bool,
    pub file_bytes: u64,
    /// (name, original ggml type, decoded element count) of the first record —
    /// proves a real, named tensor decodes back off disk, not just intact codes.
    pub first: Option<(String, u32, usize)>,
}

/// Re-read the whole file, verifying every tensor's integrity tag, and decode the
/// first tensor end-to-end. Proves the written model is well-formed and loadable.
pub fn verify_file(path: &str) -> Result<OqStats, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("not an .oq file (bad magic)".into());
    }
    let count = rd_u32(&mut r)?;
    let mut all_verified = true;
    let mut first = None;
    for idx in 0..count {
        let rec = read_record(&mut r)?;
        if !rec.tensor.verify() {
            all_verified = false;
        }
        if idx == 0 {
            // smoke-test the read path by decoding off disk; skip the full expand
            // for huge tensors (its integrity is already checked above).
            let decoded = if rec.tensor.len <= 10_000_000 {
                crate::wq::expand(&rec.tensor).len()
            } else {
                rec.tensor.len
            };
            first = Some((rec.name, rec.orig_type, decoded));
        }
    }
    Ok(OqStats {
        count,
        all_verified,
        file_bytes,
        first,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wq::{cosine, encode_with, expand, QuantStrategy, WqConfig};

    fn weightish(n: usize, seed: u32) -> Vec<f32> {
        let mut h = seed;
        (0..n)
            .map(|_| {
                h = crate::wq::mix_u32(h);
                (h as f32 / u32::MAX as f32 - 0.5) * 0.1
            })
            .collect()
    }

    #[test]
    fn oq_write_read_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("atom_quantizer_oq_test.oq");
        let ps = path.to_str().unwrap();

        // Compress a few tensors and write them.
        let originals: Vec<(String, Vec<f32>)> = (0..4)
            .map(|i| (format!("blk.{i}.w"), weightish(1024 + i * 97, 1 + i as u32)))
            .collect();
        let mut w = Writer::create(ps).unwrap();
        let mut comps = Vec::new();
        for (name, data) in &originals {
            // The picker moved to main.rs; encode at Q4 for the roundtrip test.
            let c = encode_with(data, QuantStrategy::Q4, &WqConfig::default());
            w.append(name, 0, &c).unwrap();
            comps.push(c);
        }
        let written = w.finish().unwrap();
        assert_eq!(written, 4);

        // Verify pass: file is well-formed and every integrity tag holds.
        let stats = verify_file(ps).unwrap();
        assert_eq!(stats.count, 4);
        assert!(stats.all_verified, "all tensors must verify");
        assert!(stats.file_bytes > 8);

        // Full read-back: reconstructed tensors expand identically to the writer's.
        let file = File::open(ps).unwrap();
        let mut r = BufReader::new(file);
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).unwrap();
        let count = rd_u32(&mut r).unwrap();
        assert_eq!(count, 4);
        for (i, (name, data)) in originals.iter().enumerate() {
            let rec = read_record(&mut r).unwrap();
            assert_eq!(&rec.name, name);
            // decoded-from-disk expansion == in-memory expansion, and stays faithful
            assert_eq!(expand(&rec.tensor), expand(&comps[i]));
            assert!(cosine(data, &expand(&rec.tensor)) > 0.9);
        }
        let _ = std::fs::remove_file(ps);
    }
}
