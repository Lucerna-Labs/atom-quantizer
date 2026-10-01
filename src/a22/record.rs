//! Native AR01/ACZ1 validation and streaming reconstruction of supported recipes.
use super::tensor::{Layout, Output, PackedTensor, Scalar};
use super::{number, shape, text, MAX_RECORD};
use flate2::{Decompress, FlushDecompress, Status};
use serde_json::Value;
use std::collections::BTreeMap;
use std::ops::Range;

struct Array {
    dtype: String,
    shape: Vec<usize>,
    range: Range<usize>,
}

pub(super) struct Record {
    raw: Vec<u8>,
    meta: Value,
    arrays: BTreeMap<String, Array>,
}

fn u32_at(bytes: &[u8]) -> Result<u32, String> {
    let b = bytes.get(..4).ok_or("truncated f32 field")?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(bytes: &[u8]) -> Result<u64, String> {
    let b = bytes.get(..8).ok_or("truncated u64 field")?;
    Ok(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn half(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = ((bits >> 10) & 31) as u32;
    let fraction = (bits & 1023) as u32;
    let value = if exponent == 0 {
        if fraction == 0 {
            sign
        } else {
            let top = 31 - fraction.leading_zeros();
            sign | ((top + 103) << 23) | ((fraction - (1 << top)) << (23 - top))
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (fraction << 13)
    } else {
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(value)
}

impl Record {
    pub(super) fn validate_functional_config(&self, config: &Value) -> Result<(), String> {
        let exact = config.get("family").and_then(Value::as_str) == Some("lossless");
        match text(&self.meta, "kind")? {
            "raw_f32" | "raw_bf16" if exact => Ok(()),
            "constant" if !exact => Ok(()),
            "scalar" if !exact => {
                if self.meta.get("config") != Some(config)
                    || number(&self.meta, "bits")? != number(config, "bits")?
                    || !self.arrays.contains_key("lowrank_u")
                    || !self.arrays.contains_key("lowrank_v")
                {
                    return Err("stored scalar recipe differs from functional profile".into());
                }
                Ok(())
            }
            _ => Err("record kind differs from declared functional profile".into()),
        }
    }

    pub(super) fn parse(data: &[u8]) -> Result<Self, String> {
        let raw = if data.starts_with(b"ACZ1") {
            let length = usize::try_from(u64_at(data.get(4..).ok_or("truncated ACZ1")?)?)
                .map_err(|_| "record size overflow")?;
            if length > MAX_RECORD {
                return Err("record exceeds native size bound".into());
            }
            let compressed = data.get(12..).ok_or("truncated compressed frame")?;
            let capacity = length.checked_add(1).ok_or("record size overflow")?;
            let mut output = Vec::new();
            output
                .try_reserve_exact(capacity)
                .map_err(|e| e.to_string())?;
            output.resize(capacity, 0);
            let mut decoder = Decompress::new(true);
            let status = decoder
                .decompress(compressed, &mut output, FlushDecompress::Finish)
                .map_err(|e| e.to_string())?;
            if status != Status::StreamEnd
                || decoder.total_in() != compressed.len() as u64
                || decoder.total_out() != length as u64
            {
                return Err("truncated, oversized or trailing compressed record".into());
            }
            output.truncate(length);
            output
        } else {
            if data.len() > MAX_RECORD {
                return Err("record exceeds native size bound".into());
            }
            data.to_vec()
        };
        if raw.len() < 40 || !raw.starts_with(b"AR01") {
            return Err("invalid AR01 record".into());
        }
        let end = raw.len() - 32;
        use sha2::{Digest, Sha256};
        if Sha256::digest(&raw[..end]).as_slice() != &raw[end..] {
            return Err("AR01 checksum mismatch".into());
        }
        let json_length = u32_at(&raw[4..])? as usize;
        let body_start = 8usize
            .checked_add(json_length)
            .ok_or("record metadata overflow")?;
        if body_start > end {
            return Err("truncated record metadata".into());
        }
        let value: Value =
            serde_json::from_slice(&raw[8..body_start]).map_err(|e| e.to_string())?;
        if number(&value, "version")? != 1 {
            return Err("unsupported AR version".into());
        }
        let meta = value
            .get("meta")
            .filter(|v| v.is_object())
            .ok_or("invalid record metadata")?
            .clone();
        let descriptions = value
            .get("arrays")
            .and_then(Value::as_array)
            .ok_or("invalid array table")?;
        if descriptions.len() > 64 {
            return Err("too many record arrays".into());
        }
        let mut arrays = BTreeMap::new();
        let mut offset = 0usize;
        for description in descriptions {
            let name = text(description, "name")?.to_string();
            if name.is_empty() || arrays.contains_key(&name) {
                return Err("empty or duplicate array name".into());
            }
            let dtype = text(description, "dtype")?.to_string();
            let width = match dtype.as_str() {
                "|u1" => 1,
                "<u2" | "<f2" => 2,
                "<f4" => 4,
                _ => return Err("unsupported native array dtype".into()),
            };
            let dims = shape(description.get("shape").ok_or("missing array shape")?, true)?;
            let count = dims
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or("array shape overflow")?;
            let size = count.checked_mul(width).ok_or("array size overflow")?;
            if number(description, "offset")? != offset as u64
                || number(description, "bytes")? != size as u64
            {
                return Err("array extent differs from descriptor".into());
            }
            let begin = body_start
                .checked_add(offset)
                .ok_or("array offset overflow")?;
            let finish = begin.checked_add(size).ok_or("array extent overflow")?;
            if finish > end {
                return Err("array extends past record".into());
            }
            arrays.insert(
                name,
                Array {
                    dtype,
                    shape: dims,
                    range: begin..finish,
                },
            );
            offset = offset.checked_add(size).ok_or("array offset overflow")?;
        }
        if body_start.checked_add(offset) != Some(end) {
            return Err("unexpected record suffix".into());
        }
        Ok(Self { raw, meta, arrays })
    }

    fn array(&self, name: &str, dtype: &str, dims: &[usize]) -> Result<&[u8], String> {
        let array = self
            .arrays
            .get(name)
            .ok_or_else(|| format!("missing array {name}"))?;
        if array.dtype != dtype || array.shape != dims {
            return Err(format!("invalid dtype/shape for {name}"));
        }
        self.raw
            .get(array.range.clone())
            .ok_or_else(|| "invalid array storage".into())
    }

    fn floats(&self, name: &str, dims: &[usize], half_only: bool) -> Result<Vec<f32>, String> {
        let array = self
            .arrays
            .get(name)
            .ok_or_else(|| format!("missing array {name}"))?;
        let mut values = Vec::new();
        if array.dtype == "<f2" {
            let bytes = self.array(name, "<f2", dims)?;
            values
                .try_reserve_exact(bytes.len() / 2)
                .map_err(|e| e.to_string())?;
            for b in bytes.as_chunks::<2>().0 {
                values.push(half(u16::from_le_bytes(*b)));
            }
        } else if array.dtype == "<f4" && !half_only {
            let bytes = self.array(name, "<f4", dims)?;
            values
                .try_reserve_exact(bytes.len() / 4)
                .map_err(|e| e.to_string())?;
            for b in bytes.as_chunks::<4>().0 {
                values.push(f32::from_le_bytes(*b));
            }
        } else {
            return Err(format!("unsupported float storage for {name}"));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite array {name}"));
        }
        Ok(values)
    }

    fn only(&self, arrays: &[&str], metadata: &[&str]) -> Result<(), String> {
        if self.arrays.keys().any(|n| !arrays.contains(&n.as_str())) {
            return Err("unsupported decoder side arrays".into());
        }
        let object = self.meta.as_object().ok_or("invalid metadata object")?;
        if object.keys().any(|n| !metadata.contains(&n.as_str())) {
            return Err("unsupported decoder metadata".into());
        }
        Ok(())
    }

    fn layout(&self, expected: &[usize]) -> Result<(Vec<usize>, Layout), String> {
        let dims = shape(self.meta.get("shape").ok_or("missing tensor shape")?, false)?;
        if dims != expected {
            return Err("record shape differs from model table".into());
        }
        let _count = dims
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or("tensor size overflow")?;
        let layout = match text(&self.meta, "kind")? {
            "raw_f32" => {
                self.only(&["values"], &["kind", "shape"])?;
                self.array("values", "<f4", &dims)?;
                Layout::F32(
                    self.arrays
                        .get("values")
                        .ok_or("missing values")?
                        .range
                        .clone(),
                )
            }
            "raw_bf16" => {
                self.only(&["values"], &["kind", "shape"])?;
                self.array("values", "<u2", &dims)?;
                Layout::Bf16(
                    self.arrays
                        .get("values")
                        .ok_or("missing values")?
                        .range
                        .clone(),
                )
            }
            "constant" => {
                self.only(&["value"], &["kind", "shape"])?;
                Layout::Constant(f32::from_bits(u32_at(self.array("value", "<f4", &[1])?)?))
            }
            "scalar" => {
                self.only(
                    &["codes", "scales", "levels", "lowrank_u", "lowrank_v"],
                    &[
                        "kind",
                        "shape",
                        "config",
                        "bits",
                        "group",
                        "padded_cols",
                        "alphabet",
                        "rotation",
                        "lowrank_effective_rank",
                        "repair",
                        "repair_strength",
                    ],
                )?;
                if dims.len() != 2 {
                    return Err("scalar record must be a matrix".into());
                }
                if self.meta.get("rotation").is_some() && number(&self.meta, "rotation")? != 0 {
                    return Err("native A22 rotations are unsupported".into());
                }
                if self
                    .meta
                    .get("repair")
                    .is_some_and(|v| v.as_str() != Some("none"))
                    || self
                        .meta
                        .get("repair_strength")
                        .is_some_and(|v| v.as_f64() != Some(0.0))
                {
                    return Err("native A22 repair stages are unsupported".into());
                }
                let bits = usize::try_from(number(&self.meta, "bits")?)
                    .map_err(|_| "bit width overflow")?;
                if ![2, 3, 4, 6, 8].contains(&bits) {
                    return Err("unsupported scalar bit width".into());
                }
                let group =
                    usize::try_from(number(&self.meta, "group")?).map_err(|_| "group overflow")?;
                if group == 0 {
                    return Err("zero scalar group".into());
                }
                let (rows, cols) = (dims[0], dims[1]);
                let padded = cols
                    .div_ceil(group)
                    .checked_mul(group)
                    .ok_or("padding overflow")?;
                if number(&self.meta, "padded_cols")? != padded as u64 {
                    return Err("invalid scalar padding".into());
                }
                let codes_count = rows.checked_mul(padded).ok_or("code count overflow")?;
                let bytes = codes_count
                    .checked_mul(bits)
                    .ok_or("packed extent overflow")?
                    .div_ceil(8);
                let codes = self.array("codes", "|u1", &[bytes])?;
                let groups = padded / group;
                let scales = self.floats("scales", &[rows, groups], false)?;
                if scales.iter().any(|v| *v <= 0.0) {
                    return Err("nonpositive scalar scale".into());
                }
                let table_shape = &self
                    .arrays
                    .get("levels")
                    .ok_or("missing scalar levels")?
                    .shape;
                if table_shape.len() != 1 || table_shape[0] == 0 || table_shape[0] > (1 << bits) {
                    return Err("invalid scalar alphabet".into());
                }
                let levels = self.floats("levels", table_shape, true)?;
                if levels.len() < 1 << bits {
                    for index in 0..codes_count {
                        let bit = index * bits;
                        let word = codes[bit / 8] as u16
                            | (codes.get(bit / 8 + 1).copied().map_or(0, u16::from) << 8);
                        if ((word >> (bit % 8)) & ((1 << bits) - 1)) as usize >= levels.len() {
                            return Err("code outside stored alphabet, including padding".into());
                        }
                    }
                }
                let has_correction =
                    self.arrays.contains_key("lowrank_u") || self.arrays.contains_key("lowrank_v");
                let (rank, left, right) = if has_correction {
                    let shape = &self
                        .arrays
                        .get("lowrank_u")
                        .ok_or("missing left correction")?
                        .shape;
                    if shape.len() != 2 || shape[0] != rows || shape[1] > 2 {
                        return Err("native A22 supports correction ranks zero through two".into());
                    }
                    let rank = shape[1];
                    if self.meta.get("lowrank_effective_rank").is_some()
                        && number(&self.meta, "lowrank_effective_rank")? != rank as u64
                    {
                        return Err("correction rank metadata differs".into());
                    }
                    (
                        rank,
                        self.floats("lowrank_u", &[rows, rank], true)?,
                        self.floats("lowrank_v", &[rank, cols], true)?,
                    )
                } else {
                    if self.meta.get("lowrank_effective_rank").is_some() {
                        return Err("rank metadata without correction arrays".into());
                    }
                    (0, Vec::new(), Vec::new())
                };
                let mut table = [0.0f32; 256];
                table[..levels.len()].copy_from_slice(&levels);
                Layout::Scalar(Box::new(Scalar {
                    bits,
                    group,
                    padded,
                    codes: self
                        .arrays
                        .get("codes")
                        .ok_or("missing codes")?
                        .range
                        .clone(),
                    scales,
                    levels: table,
                    correction: if has_correction { Some(rank) } else { None },
                    left,
                    right,
                }))
            }
            kind => return Err(format!("unsupported native A22 recipe {kind}")),
        };
        Ok((dims, layout))
    }

    pub(super) fn prepare(self, expected: &[usize]) -> Result<PackedTensor, String> {
        let (shape, layout) = self.layout(expected)?;
        Ok(PackedTensor {
            raw: self.raw,
            shape,
            layout,
        })
    }

    pub(super) fn decode<F: FnMut(&[u8]) -> Result<(), String>>(
        &self,
        expected: &[usize],
        emit: F,
    ) -> Result<(), String> {
        let (shape, layout) = self.layout(expected)?;
        let mut output = Output::new(emit);
        layout.each(&self.raw, &shape, |value| output.push(value))?;
        output.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_conversion_matches_finite_ieee_values_including_signed_zero() {
        for bits in 0u16..=u16::MAX {
            let exponent = (bits >> 10) & 31;
            if exponent == 31 {
                continue;
            }
            let fraction = (bits & 1023) as f64;
            let magnitude = if exponent == 0 {
                fraction * 2f64.powi(-24)
            } else {
                (1.0 + fraction / 1024.0) * 2f64.powi(exponent as i32 - 15)
            };
            let expected = (if bits & 0x8000 != 0 {
                -magnitude
            } else {
                magnitude
            }) as f32;
            assert_eq!(half(bits).to_bits(), expected.to_bits());
        }
    }
}
