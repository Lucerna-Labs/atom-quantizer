//! Real operator inputs bound to a particular GGUF and calibration corpus.
//!
//! AC01 stores a source SHA-256, corpus SHA-256, named linear input batches or
//! embedding indices, and a SHA-256 of the entire preceding calibration file.

use crate::{encoded::Transform, gguf::TensorInfo, stacks};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Cursor, Read};

#[derive(Clone, Debug)]
pub enum Samples {
    Linear(Vec<Vec<f32>>),
    Embedding(Vec<usize>),
}

impl Samples {
    pub fn validate_for(&self, tensor: &TensorInfo) -> Result<(), String> {
        let rows = tensor.n_elems / tensor.row_len.max(1);
        match self {
            Self::Linear(xs) if xs.iter().all(|x| x.len() == tensor.row_len) => Ok(()),
            Self::Embedding(ids)
                if tensor.name == "token_embd.weight" && ids.iter().all(|i| *i < rows) =>
            {
                Ok(())
            }
            _ => Err(format!(
                "calibration operator or shape mismatch for {}",
                tensor.name
            )),
        }
    }

    pub fn column_importance(&self, transform: Transform) -> Option<Vec<f32>> {
        let Self::Linear(xs) = self else {
            return None;
        };
        let n = xs[0].len();
        let mut importance = vec![0.0f64; n];
        for x in xs {
            let mut rotated = x.clone();
            if transform == Transform::RowHadamard8 {
                stacks::row_hadamard_8_apply(&mut rotated, 1, n);
            }
            for (h, v) in importance.iter_mut().zip(rotated) {
                *h += (v as f64).powi(2);
            }
        }
        Some(
            importance
                .into_iter()
                .map(|h| (h / xs.len() as f64) as f32)
                .collect(),
        )
    }
}

pub struct Calibration {
    pub source_sha256: [u8; 32],
    pub corpus_sha256: [u8; 32],
    pub tensors: HashMap<String, Samples>,
}

pub fn file_sha256(path: &str) -> Result<[u8; 32], String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hash.finalize().into())
}

fn u32_read(r: &mut Cursor<&[u8]>) -> Result<usize, String> {
    let mut b = [0; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b) as usize)
}
fn available(r: &Cursor<&[u8]>) -> usize {
    r.get_ref().len().saturating_sub(r.position() as usize)
}

impl Calibration {
    pub fn load(path: &str, source_gguf: &str) -> Result<Self, String> {
        let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
        if metadata.len() > 512 * 1024 * 1024 {
            return Err("calibration exceeds 512 MiB".into());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let result = Self::from_bytes(&bytes)?;
        if file_sha256(source_gguf)? != result.source_sha256 {
            return Err("calibration belongs to a different source GGUF (SHA-256 mismatch)".into());
        }
        Ok(result)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 104 || &bytes[..4] != b"AC01" {
            return Err("not an AC01 calibration file".into());
        }
        let end = bytes.len() - 32;
        let actual: [u8; 32] = Sha256::digest(&bytes[..end]).into();
        if actual.as_slice() != &bytes[end..] {
            return Err("calibration checksum mismatch".into());
        }
        let source_sha256 = bytes[4..36].try_into().unwrap();
        let corpus_sha256 = bytes[36..68].try_into().unwrap();
        let mut r = Cursor::new(&bytes[..end]);
        r.set_position(68);
        let count = u32_read(&mut r)?;
        if count == 0 || count > 1 << 20 {
            return Err("invalid calibration record count".into());
        }
        let mut tensors = HashMap::new();
        for _ in 0..count {
            let n = u32_read(&mut r)?;
            if n == 0 || n > 65536 || n > available(&r) {
                return Err("invalid calibration name length".into());
            }
            let mut name = vec![0; n];
            r.read_exact(&mut name).map_err(|e| e.to_string())?;
            let name = String::from_utf8(name).map_err(|e| e.to_string())?;
            let mut kind = [0];
            r.read_exact(&mut kind).map_err(|e| e.to_string())?;
            let k = u32_read(&mut r)?;
            if k == 0 || k > 1 << 20 {
                return Err("empty or excessive calibration batch".into());
            }
            let samples = match kind[0] {
                0 => {
                    let cols = u32_read(&mut r)?;
                    if cols == 0 || k.checked_mul(cols).is_none_or(|n| n > available(&r) / 4) {
                        return Err("invalid linear calibration shape".into());
                    }
                    let mut xs = Vec::with_capacity(k);
                    for _ in 0..k {
                        let mut x = Vec::with_capacity(cols);
                        for _ in 0..cols {
                            let v = f32::from_bits(u32_read(&mut r)? as u32);
                            if !v.is_finite() {
                                return Err("nonfinite activation sample".into());
                            }
                            x.push(v);
                        }
                        xs.push(x);
                    }
                    Samples::Linear(xs)
                }
                1 => {
                    if k > available(&r) / 4 {
                        return Err("truncated embedding samples".into());
                    }
                    let mut ids = Vec::with_capacity(k);
                    for _ in 0..k {
                        ids.push(u32_read(&mut r)?);
                    }
                    Samples::Embedding(ids)
                }
                _ => return Err("unknown calibration operator".into()),
            };
            if tensors.insert(name, samples).is_some() {
                return Err("duplicate calibration tensor".into());
            }
        }
        if r.position() as usize != end {
            return Err("trailing calibration data".into());
        }
        Ok(Self {
            source_sha256,
            corpus_sha256,
            tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_activation_geometry_is_rotated_with_weights() {
        let samples = Samples::Linear(vec![vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        let plain = samples.column_importance(Transform::None).unwrap();
        let rotated = samples.column_importance(Transform::RowHadamard8).unwrap();
        assert_eq!(plain[0], 4.0);
        assert!(rotated.iter().all(|v| (*v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn calibration_requires_integrity_and_nonempty_batches() {
        let mut b = Vec::from(*b"AC01");
        b.extend_from_slice(&[0; 64]);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        b.push(b'w');
        b.push(0);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b.extend_from_slice(&2.0f32.to_le_bytes());
        let h = Sha256::digest(&b);
        b.extend_from_slice(&h);
        assert!(Calibration::from_bytes(&b).is_ok());
        b[4] ^= 1;
        assert!(Calibration::from_bytes(&b).is_err());
    }
}
