#![allow(dead_code)]
//! gguf — a minimal, streaming GGUF reader (pure std, no crates).
//!
//! Parses the header, metadata (skipped, alignment captured), and tensor table,
//! then dequantizes a tensor's stored data to f32 on demand. Large models are
//! never fully loaded — each tensor is seek+read individually. Dequant is
//! implemented for the formats the local the target models use: F32, F16, Q8_0, Q6_K.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

// ── ggml tensor type ids ──
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q6_K: u32 = 14;
pub const GGML_BF16: u32 = 30;

pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        _ => "?",
    }
}

/// True if `read_tensor_f32` can dequantize this type.
pub fn dequantizable(t: u32) -> bool {
    matches!(
        t,
        GGML_F32 | GGML_F16 | GGML_BF16 | GGML_Q8_0 | GGML_Q4_K | GGML_Q6_K
    )
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    pub offset: u64,
    pub n_elems: usize,
    /// Innermost (row / in-feature) dim, or n_elems for 1-D tensors.
    /// GGUF stores dims in [innermost, ..., outermost] order.
    pub row_len: usize,
    pub dims: Vec<usize>,
}

pub struct Gguf {
    reader: BufReader<File>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
    pub arch: String,
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
fn rd_string(r: &mut impl Read) -> Result<String, String> {
    let len = rd_u64(r)? as usize;
    if len > 1 << 24 {
        return Err("absurd string length".into());
    }
    let mut b = vec![0u8; len];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Byte size of one metadata value of the given type (skipping arrays too).
fn skip_value(r: &mut impl Read, vtype: u32) -> Result<Option<u32>, String> {
    // Returns Some(u32) when the value is a u32 (used to capture general.alignment).
    match vtype {
        0 | 1 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(None)
        }
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(None)
        }
        2 | 3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(None)
        }
        4 => Ok(Some(rd_u32(r)?)), // UINT32
        5 | 6 => Ok(Some(rd_u32(r)?)),
        8 => {
            rd_string(r)?;
            Ok(None)
        }
        10..=12 => {
            rd_u64(r)?;
            Ok(None)
        }
        9 => {
            // ARRAY: elem_type, count, then count elements
            let elem_type = rd_u32(r)?;
            let count = rd_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
            Ok(None)
        }
        _ => Err(format!("unknown metadata value type {vtype}")),
    }
}

fn block_layout(t: u32) -> Option<(usize, usize)> {
    // (elements per block, bytes per block)
    match t {
        GGML_F32 => Some((1, 4)),
        GGML_F16 => Some((1, 2)),
        GGML_BF16 => Some((1, 2)),
        GGML_Q8_0 => Some((32, 34)),   // f16 scale + 32×i8
        GGML_Q4_K => Some((256, 144)), // super-block: d+dmin(f16)+scales[12]+qs[128]
        GGML_Q6_K => Some((256, 210)), // super-block: ql[128]+qh[64]+scales[16]+d(f16)
        _ => None,
    }
}

pub fn open(path: &str) -> Result<Gguf, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = rd_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let tensor_count = rd_u64(&mut r)? as usize;
    let kv_count = rd_u64(&mut r)? as usize;
    if tensor_count > 1 << 20 || kv_count > 1 << 20 {
        return Err("absurd header counts".into());
    }

    let mut alignment: u64 = 32;
    let mut arch = String::new();
    for _ in 0..kv_count {
        let key = rd_string(&mut r)?;
        let vtype = rd_u32(&mut r)?;
        if key == "general.architecture" && vtype == 8 {
            arch = rd_string(&mut r)?;
        } else {
            let v = skip_value(&mut r, vtype)?;
            if key == "general.alignment" {
                if let Some(a) = v {
                    alignment = a as u64;
                }
            }
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = rd_string(&mut r)?;
        let n_dims = rd_u32(&mut r)? as usize;
        if !(1..=8).contains(&n_dims) {
            return Err("tensor must have 1 through 8 dims".into());
        }
        let mut n_elems: usize = 1;
        let mut row_len: usize = 1;
        let mut dims = Vec::with_capacity(n_dims);
        for k in 0..n_dims {
            let d = rd_u64(&mut r)? as usize;
            if k == 0 {
                row_len = d; // innermost dim = row length in GGUF
            }
            n_elems = n_elems.checked_mul(d).ok_or("tensor dimensions overflow")?;
            dims.push(d);
        }
        let ggml_type = rd_u32(&mut r)?;
        let offset = rd_u64(&mut r)?;
        tensors.push(TensorInfo {
            name,
            ggml_type,
            offset,
            n_elems,
            row_len,
            dims,
        });
    }

    // Tensor data begins after the header, padded up to `alignment`.
    let pos = r.stream_position().map_err(|e| e.to_string())?;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("GGUF alignment must be a nonzero power of two".into());
    }
    let data_start = pos
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or("GGUF header size overflow")?;

    Ok(Gguf {
        reader: r,
        tensors,
        data_start,
        arch,
    })
}

/// Preserve model/tokenizer metadata and the source tensor table, without weights.
pub fn model_header(g: &mut Gguf) -> Result<Vec<u8>, String> {
    let len = usize::try_from(g.data_start).map_err(|_| "GGUF prefix too large")?;
    if len > 256 * 1024 * 1024 {
        return Err("GGUF prefix exceeds 256 MiB".into());
    }
    let mut bytes = vec![0; len];
    g.reader
        .seek(SeekFrom::Start(0))
        .map_err(|e| e.to_string())?;
    g.reader.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let mag = if exp == 0 {
        mant * 2f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0.0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 {
        -mag
    } else {
        mag
    }
}

/// Which source-tensor types the passthrough writer can round-trip. K-quant
/// tensors can be READ (dequant to f32) but not WRITTEN back — a proper GGUF
/// passthrough would require a K-quant packing implementation the codec does
/// not carry. F32/F16/BF16 all support in-place writeback of the reconstruction
/// at the same byte offset because the stored size is identical.
pub fn writable_passthrough(t: u32) -> bool {
    matches!(t, GGML_F32 | GGML_F16 | GGML_BF16)
}

/// Cast f32 → source-type bytes, in-place overwrite at the source's byte
/// offset. Caller opens `dest` for RW, and provides `data_start` from the
/// source `Gguf` (dest is a byte-for-byte copy of source, so offsets match).
///
/// Returns `Err` if the source type is not passthrough-writable (K-quants).
pub fn write_tensor_reconstruction(
    dest: &mut std::fs::File,
    ti: &TensorInfo,
    data_start: u64,
    recon: &[f32],
) -> Result<(), String> {
    use std::io::Write;
    let abs = data_start + ti.offset;
    dest.seek(SeekFrom::Start(abs)).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = match ti.ggml_type {
        GGML_F32 => {
            let mut v = Vec::with_capacity(recon.len() * 4);
            for &f in recon {
                v.extend_from_slice(&f.to_le_bytes());
            }
            v
        }
        GGML_F16 => {
            let mut v = Vec::with_capacity(recon.len() * 2);
            for &f in recon {
                v.extend_from_slice(&f32_to_f16_bits(f).to_le_bytes());
            }
            v
        }
        GGML_BF16 => {
            let mut v = Vec::with_capacity(recon.len() * 2);
            for &f in recon {
                v.extend_from_slice(&f32_to_bf16_bits(f).to_le_bytes());
            }
            v
        }
        _ => {
            return Err(format!(
                "passthrough writeback not supported for {}",
                type_name(ti.ggml_type)
            ))
        }
    };
    dest.write_all(&bytes).map_err(|e| e.to_string())
}

/// f32 → bf16 with round-to-nearest-even. bf16 = top 16 bits of f32 (sign + 8 exp
/// + 7 mantissa); the rounding term keeps parity with numpy / torch bf16 casts.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// f32 → f16 (IEEE 754 binary16) with round-to-nearest-even. Handles subnormals,
/// overflow to Inf, NaN preservation.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        let m: u16 = if mant != 0 { 0x0200 } else { 0 };
        return (sign << 15) | 0x7C00 | m;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return (sign << 15) | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign << 15;
        }
        let m = mant | 0x0080_0000;
        let shift = 14 - e;
        let rounding = (1u32 << (shift - 1)) - 1 + ((m >> shift) & 1);
        let m_shifted = ((m + rounding) >> shift) as u16;
        return (sign << 15) | m_shifted;
    }
    let rounded = mant + 0x0FFF + ((mant >> 13) & 1);
    let e = e + (rounded >> 23) as i32;
    if e >= 31 {
        return (sign << 15) | 0x7C00;
    }
    (sign << 15) | ((e as u16) << 10) | ((rounded >> 13) as u16 & 0x03FF)
}

/// Round exactly as dense GGUF export does, for fidelity measurement.
pub fn round_to_storage(value: f32, dtype: u32) -> Result<f32, String> {
    match dtype {
        GGML_F32 => Ok(value),
        GGML_F16 => Ok(f16_to_f32(f32_to_f16_bits(value))),
        GGML_BF16 => Ok(f32::from_bits((f32_to_bf16_bits(value) as u32) << 16)),
        _ => Err("unsupported decoded storage dtype".into()),
    }
}

/// Read + dequantize a tensor's stored data to f32.
pub fn read_tensor_f32(g: &mut Gguf, idx: usize) -> Result<Vec<f32>, String> {
    let ti = g.tensors.get(idx).ok_or("tensor index out of range")?;
    let (blk_elems, blk_bytes) = block_layout(ti.ggml_type)
        .ok_or_else(|| format!("dequant not implemented for {}", type_name(ti.ggml_type)))?;
    let n = ti.n_elems;
    let n_blocks = n.div_ceil(blk_elems);
    let byte_size = n_blocks * blk_bytes;
    let abs = g.data_start + ti.offset;
    let ttype = ti.ggml_type;

    g.reader
        .seek(SeekFrom::Start(abs))
        .map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; byte_size];
    g.reader.read_exact(&mut buf).map_err(|e| e.to_string())?;

    dequant_bytes(ttype, &buf, n)
}

/// Pure dequant of stored bytes → f32 (separated out so it's unit-testable
/// without a file). `n` is the logical element count (trailing block pad dropped).
pub(crate) fn dequant_bytes(ttype: u32, buf: &[u8], n: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(n);
    match ttype {
        GGML_F32 => {
            for c in buf.as_chunks::<4>().0 {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        GGML_F16 => {
            for c in buf.as_chunks::<2>().0 {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        GGML_BF16 => {
            // bf16 is the top 16 bits of f32 (sign + 8 exp + 7 mantissa).
            // Reconstruct f32 by shifting into the high half.
            for c in buf.as_chunks::<2>().0 {
                let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
                out.push(f32::from_bits(bits));
            }
        }
        GGML_Q8_0 => {
            for blk in buf.as_chunks::<34>().0 {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..34] {
                    out.push(d * (q as i8) as f32);
                }
            }
        }
        GGML_Q4_K => dequant_q4_k(buf, &mut out),
        GGML_Q6_K => dequant_q6_k(buf, &mut out),
        _ => return Err(format!("dequant not implemented for {}", type_name(ttype))),
    }
    out.truncate(n);
    Ok(out)
}

/// Q4_K 6-bit packed scale/min accessor, per llama.cpp `get_scale_min_k4`.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Q4_K super-block dequant, faithful to llama.cpp's `dequantize_row_q4_K`.
/// 256 weights in 144 bytes: d, dmin (f16), scales[12] (6-bit packed scales+mins),
/// qs[128] (4-bit). Asymmetric: weight = d·scale·q − dmin·min.
fn dequant_q4_k(buf: &[u8], out: &mut Vec<f32>) {
    for sb in buf.as_chunks::<144>().0 {
        let d = f16_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([sb[2], sb[3]]));
        let scales = &sb[4..16];
        let qs = &sb[16..144];
        let mut is = 0usize;
        for grp in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mm1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mm2 = dmin * m2 as f32;
            let q = &qs[grp * 32..grp * 32 + 32];
            for &qb in q {
                out.push(d1 * (qb & 0x0F) as f32 - mm1);
            }
            for &qb in q {
                out.push(d2 * (qb >> 4) as f32 - mm2);
            }
            is += 2;
        }
    }
}

/// Q6_K super-block dequant, faithful to llama.cpp's `dequantize_row_q6_K`.
/// Super-block = 256 weights in 210 bytes: ql[128] (low 4 bits), qh[64] (high 2
/// bits), scales[16] (i8), d (f16). Each weight = d · scale · (6-bit − 32).
fn dequant_q6_k(buf: &[u8], out: &mut Vec<f32>) {
    let n_sb = buf.len() / 210;
    out.resize(n_sb * 256, 0.0);
    for (sbi, sb) in buf.as_chunks::<210>().0.iter().enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([sb[208], sb[209]]));
        let base = sbi * 256;
        // two halves of 128 weights each
        for half in 0..2 {
            let (ql_o, qh_o, sc_o, y_o) = (half * 64, 128 + half * 32, 192 + half * 8, half * 128);
            for l in 0..32 {
                let is = l / 16; // scale sub-index within this half
                let ql_l = sb[ql_o + l];
                let ql_h = sb[ql_o + l + 32];
                let qh = sb[qh_o + l];
                let q1 = ((ql_l & 0x0F) as i32 | (((qh & 3) as i32) << 4)) - 32;
                let q2 = ((ql_h & 0x0F) as i32 | ((((qh >> 2) & 3) as i32) << 4)) - 32;
                let q3 = ((ql_l >> 4) as i32 | ((((qh >> 4) & 3) as i32) << 4)) - 32;
                let q4 = ((ql_h >> 4) as i32 | ((((qh >> 6) & 3) as i32) << 4)) - 32;
                let sc = |k: usize| sb[sc_o + is + k] as i8 as f32;
                out[base + y_o + l] = d * sc(0) * q1 as f32;
                out[base + y_o + l + 32] = d * sc(2) * q2 as f32;
                out[base + y_o + l + 64] = d * sc(4) * q3 as f32;
                out[base + y_o + l + 96] = d * sc(6) * q4 as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_storage_rounding_matches_ieee_ties_and_subnormals() {
        assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02);
        assert_eq!(f32_to_f16_bits(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16_bits(3.0 * 2f32.powi(-25)), 2);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7C00);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_bf16_bits(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(f32_to_bf16_bits(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert!(round_to_storage(f32::from_bits(0x7FFF_FFFF), GGML_BF16)
            .unwrap()
            .is_nan());
    }

    /// Build a Q6_K super-block (210 bytes) from raw fields.
    fn q6k_block(ql: &[u8; 128], qh: &[u8; 64], scales: &[i8; 16], d_f16: u16) -> Vec<u8> {
        let mut b = Vec::with_capacity(210);
        b.extend_from_slice(ql);
        b.extend_from_slice(qh);
        b.extend(scales.iter().map(|&s| s as u8));
        b.extend_from_slice(&d_f16.to_le_bytes());
        b
    }

    #[test]
    fn q6k_dequant_matches_hand_computed() {
        // d = 1.0 (f16 0x3C00), all scales = 1.
        let scales = [1i8; 16];
        // All-zero quants → 6-bit value 0 → (0 − 32) = −32 everywhere.
        let out = dequant_bytes(
            GGML_Q6_K,
            &q6k_block(&[0; 128], &[0; 64], &scales, 0x3C00),
            256,
        )
        .unwrap();
        assert_eq!(out.len(), 256);
        assert!(
            out.iter().all(|&v| v == -32.0),
            "all-zero block must be −32"
        );

        // Set ql[0] low nibble = 0xF and qh[0] low 2 bits = 3 → q1 = 15|(3<<4)=63 → 63−32=31.
        // ql[0]>>4 = 0 and (qh[0]>>4)&3 = 0 → q3 = 0−32 = −32.
        let mut ql = [0u8; 128];
        ql[0] = 0x0F;
        let mut qh = [0u8; 64];
        qh[0] = 0x03;
        let out = dequant_bytes(GGML_Q6_K, &q6k_block(&ql, &qh, &scales, 0x3C00), 256).unwrap();
        assert_eq!(out[0], 31.0, "q1 unpack: y[0] must be 31");
        assert_eq!(out[64], -32.0, "q3 unpack: y[64] must be −32");
        assert_eq!(out[32], -32.0, "q2 (ql[32]=0) must be −32");

        // Scale application: scales[0]=2, d=1 → y[0] scales to 2·31 = 62.
        let mut sc2 = [1i8; 16];
        sc2[0] = 2;
        let out = dequant_bytes(GGML_Q6_K, &q6k_block(&ql, &qh, &sc2, 0x3C00), 256).unwrap();
        assert_eq!(out[0], 62.0, "scale[0]=2 must double y[0]");
    }

    /// Build a Q4_K super-block (144 bytes) from raw fields.
    fn q4k_block(d_f16: u16, dmin_f16: u16, scales: &[u8; 12], qs: &[u8; 128]) -> Vec<u8> {
        let mut b = Vec::with_capacity(144);
        b.extend_from_slice(&d_f16.to_le_bytes());
        b.extend_from_slice(&dmin_f16.to_le_bytes());
        b.extend_from_slice(scales);
        b.extend_from_slice(qs);
        b
    }

    #[test]
    fn q4k_dequant_matches_hand_computed() {
        // d=1.0 (0x3C00), dmin=0 → no min term. scales[0]=1 (sc for j=0),
        // scales[1]=1 (sc for j=1). qs[0]=0x05 → low nibble 5, high nibble 0.
        let mut scales = [0u8; 12];
        scales[0] = 1;
        scales[1] = 1;
        let mut qs = [0u8; 128];
        qs[0] = 0x05;
        let out = dequant_bytes(GGML_Q4_K, &q4k_block(0x3C00, 0x0000, &scales, &qs), 256).unwrap();
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 5.0, "low-nibble: d·sc·q = 1·1·5 = 5");
        assert_eq!(out[32], 0.0, "high-nibble of qs[0] is 0");

        // Min subtraction: dmin=1.0, scales[4]=2 → m1=2, so y[0] = 1·1·5 − 1·2 = 3.
        let mut sc2 = [0u8; 12];
        sc2[0] = 1;
        sc2[4] = 2;
        let out = dequant_bytes(GGML_Q4_K, &q4k_block(0x3C00, 0x3C00, &sc2, &qs), 256).unwrap();
        assert_eq!(out[0], 3.0, "asymmetric min: d·sc·q − dmin·m = 5 − 2 = 3");
    }
}
