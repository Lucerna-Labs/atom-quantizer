//! A complete decoder recipe around the block codec.
//!
//! Every operation used by the picker is represented here, so measurement and
//! artifact consumers reconstruct the same tensor without original weights.

use crate::{gguf, stacks, wq};

pub fn storage_type_for_source(dtype: u32) -> u32 {
    if gguf::writable_passthrough(dtype) {
        dtype
    } else {
        gguf::GGML_F32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    None,
    RowHadamard8,
}

#[derive(Clone, Debug)]
pub enum Payload {
    Blocks(wq::CompressedTensor),
    F32(Vec<f32>),
}

#[derive(Clone, Debug)]
pub struct EncodedTensor {
    pub payload: Payload,
    /// GGUF order: innermost dimension first.
    pub dims: Vec<usize>,
    pub transform: Transform,
    /// Original row norms, applied after the inverse transform. Empty = bypass.
    pub row_norms: Vec<f32>,
    /// Final rounding implied by the OQ03 record's original GGUF storage type.
    pub output_type: u32,
}

impl EncodedTensor {
    pub fn plain(codec: wq::CompressedTensor) -> Self {
        Self {
            dims: vec![codec.len],
            payload: Payload::Blocks(codec),
            transform: Transform::None,
            row_norms: Vec::new(),
            output_type: gguf::GGML_F32,
        }
    }

    pub fn exact(values: Vec<f32>, dims: Vec<usize>) -> Self {
        Self {
            payload: Payload::F32(values),
            dims,
            transform: Transform::None,
            row_norms: Vec::new(),
            output_type: gguf::GGML_F32,
        }
    }

    pub fn len(&self) -> usize {
        match &self.payload {
            Payload::Blocks(c) => c.len,
            Payload::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn strategy(&self) -> Option<wq::QuantStrategy> {
        match &self.payload {
            Payload::Blocks(c) => Some(c.strategy),
            Payload::F32(_) => None,
        }
    }

    pub fn strategy_name(&self) -> &'static str {
        self.strategy().map(wq::strategy_name).unwrap_or("F32")
    }

    pub fn validate(&self) -> Result<(), String> {
        if !gguf::writable_passthrough(self.output_type) {
            return Err("unsupported decoded storage type".into());
        }
        if self.dims.is_empty() || self.dims.len() > 8 {
            return Err("tensor must have 1 through 8 dimensions".into());
        }
        let len = self
            .dims
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or("tensor dimensions overflow")?;
        if len != self.len() {
            return Err("tensor dimensions do not match its element count".into());
        }
        let cols = self.dims[0];
        let rows = len.checked_div(cols).unwrap_or(0);
        if self.transform == Transform::RowHadamard8
            && (cols < 8 || !cols.is_multiple_of(8) || rows == 0)
        {
            return Err("rowH8 requires nonempty rows divisible by eight".into());
        }
        if !self.row_norms.is_empty() && self.row_norms.len() != rows {
            return Err("row norm count does not match the tensor shape".into());
        }
        if self.row_norms.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("row norms must be finite and nonnegative".into());
        }
        let c = match &self.payload {
            Payload::Blocks(c) => c,
            Payload::F32(v) => {
                if self.transform != Transform::None
                    || !self.row_norms.is_empty()
                    || v.iter().any(|v| !v.is_finite())
                    || v.iter()
                        .any(|&v| gguf::round_to_storage(v, self.output_type).ok() != Some(v))
                {
                    return Err(
                        "exact payload must be finite and have no lossy decoder stages".into(),
                    );
                }
                return Ok(());
            }
        };
        let blocks = len.div_ceil(wq::BLOCK);
        let code_bytes = len
            .checked_mul(wq::bits(c.strategy) as usize)
            .ok_or("packed code size overflow")?
            .div_ceil(8);
        if c.packed.len() != code_bytes {
            return Err("packed code length does not match tensor shape".into());
        }
        if c.scales.len() != blocks || (!c.mins.is_empty() && c.mins.len() != blocks) {
            return Err("block scale/minimum counts do not match tensor shape".into());
        }
        if c.scales.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || c.mins.iter().any(|v| !v.is_finite())
            || !c.repair_strength.is_finite()
            || !(0.0..=1.0).contains(&c.repair_strength)
        {
            return Err("invalid quantization metadata".into());
        }
        if !c.verify() {
            return Err("packed code integrity check failed".into());
        }
        Ok(())
    }

    pub fn decode(&self) -> Result<Vec<f32>, String> {
        self.validate()?;
        let mut values = match &self.payload {
            Payload::Blocks(c) => wq::expand(c),
            Payload::F32(v) => return Ok(v.clone()),
        };
        let cols = self.dims[0];
        let rows = values.len().checked_div(cols).unwrap_or(0);
        if self.transform == Transform::RowHadamard8 {
            stacks::row_hadamard_8_apply(&mut values, rows, cols);
        }
        if !self.row_norms.is_empty() {
            values = stacks::preserve_row_norm(values, &self.row_norms, rows, cols);
        }
        if self.output_type != gguf::GGML_F32 {
            for value in &mut values {
                *value = gguf::round_to_storage(*value, self.output_type)?;
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err("decoder produced nonfinite values".into());
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_requires_complete_shape_and_side_state() {
        let c = wq::encode_with(&[0.1; 64], wq::QuantStrategy::Q4, &wq::WqConfig::default());
        let mut t = EncodedTensor::plain(c);
        t.dims = vec![8, 8];
        t.transform = Transform::RowHadamard8;
        t.row_norms = vec![1.0; 7];
        assert!(t.decode().is_err());
        t.row_norms.push(1.0);
        assert!(t.decode().is_ok());
        if let Payload::Blocks(c) = &mut t.payload {
            c.packed.pop();
        }
        assert!(t.decode().is_err());
    }
}
