//! Prepared native storage and exact-weight streaming matrix operations.
use std::mem::size_of;
use std::ops::Range;

pub(super) struct Scalar {
    pub bits: usize,
    pub group: usize,
    pub padded: usize,
    pub codes: Range<usize>,
    pub scales: Vec<f32>,
    pub levels: [f32; 256],
    pub correction: Option<usize>,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

pub(super) enum Layout {
    F32(Range<usize>),
    Bf16(Range<usize>),
    Constant(f32),
    Scalar(Box<Scalar>),
}

pub(super) struct Output<F> {
    bytes: Vec<u8>,
    emit: F,
}

impl<F: FnMut(&[u8]) -> Result<(), String>> Output<F> {
    pub(super) fn new(emit: F) -> Self {
        Self {
            bytes: Vec::with_capacity(1024 * 1024),
            emit,
        }
    }
    pub(super) fn push(&mut self, value: f32) -> Result<(), String> {
        if !value.is_finite() {
            return Err("decoded nonfinite value".into());
        }
        self.bytes.extend_from_slice(&value.to_le_bytes());
        if self.bytes.len() >= 1024 * 1024 {
            (self.emit)(&self.bytes)?;
            self.bytes.clear();
        }
        Ok(())
    }
    pub(super) fn finish(mut self) -> Result<(), String> {
        if !self.bytes.is_empty() {
            (self.emit)(&self.bytes)?;
        }
        Ok(())
    }
}

impl Scalar {
    fn row_values<
        const BITS: usize,
        const CORRECTION: usize,
        F: FnMut(usize, f32) -> Result<(), String>,
    >(
        &self,
        raw: &[u8],
        row: usize,
        cols: usize,
        emit: &mut F,
    ) -> Result<(), String> {
        let codes = &raw[self.codes.clone()];
        let groups = self.padded / self.group;
        let rank = self.correction.unwrap_or(0);
        for group in 0..groups {
            let scale = self.scales[row * groups + group];
            let begin = group * self.group;
            let end = (begin + self.group).min(cols);
            for col in begin..end {
                let bit = (row * self.padded + col) * BITS;
                let word = codes[bit / 8] as u16
                    | (codes.get(bit / 8 + 1).copied().map_or(0, u16::from) << 8);
                let code = ((word >> (bit % 8)) & ((1 << BITS) - 1)) as usize;
                let base = scale * self.levels[code];
                let mut correction = 0.0f32;
                if CORRECTION >= 2 {
                    correction += self.left[row * rank] * self.right[col];
                }
                if CORRECTION >= 3 {
                    correction += self.left[row * rank + 1] * self.right[cols + col];
                }
                emit(
                    col,
                    if CORRECTION == 0 {
                        base
                    } else {
                        base + correction
                    },
                )?;
            }
        }
        Ok(())
    }

    fn row_bits<const BITS: usize, F: FnMut(usize, f32) -> Result<(), String>>(
        &self,
        raw: &[u8],
        row: usize,
        cols: usize,
        emit: &mut F,
    ) -> Result<(), String> {
        match self.correction {
            None => self.row_values::<BITS, 0, F>(raw, row, cols, emit),
            Some(0) => self.row_values::<BITS, 1, F>(raw, row, cols, emit),
            Some(1) => self.row_values::<BITS, 2, F>(raw, row, cols, emit),
            Some(2) => self.row_values::<BITS, 3, F>(raw, row, cols, emit),
            _ => Err("invalid prepared correction rank".into()),
        }
    }
}

impl Layout {
    pub(super) fn row<F: FnMut(usize, f32) -> Result<(), String>>(
        &self,
        raw: &[u8],
        row: usize,
        cols: usize,
        mut emit: F,
    ) -> Result<(), String> {
        match self {
            Self::F32(range) => {
                let data = &raw[range.clone()][row * cols * 4..(row + 1) * cols * 4];
                for (col, bytes) in data.as_chunks::<4>().0.iter().enumerate() {
                    emit(col, f32::from_le_bytes(*bytes))?;
                }
            }
            Self::Bf16(range) => {
                let data = &raw[range.clone()][row * cols * 2..(row + 1) * cols * 2];
                for (col, bytes) in data.as_chunks::<2>().0.iter().enumerate() {
                    emit(
                        col,
                        f32::from_bits((u16::from_le_bytes(*bytes) as u32) << 16),
                    )?;
                }
            }
            Self::Constant(value) => {
                for col in 0..cols {
                    emit(col, *value)?;
                }
            }
            Self::Scalar(scalar) => match scalar.bits {
                2 => scalar.row_bits::<2, F>(raw, row, cols, &mut emit)?,
                3 => scalar.row_bits::<3, F>(raw, row, cols, &mut emit)?,
                4 => scalar.row_bits::<4, F>(raw, row, cols, &mut emit)?,
                6 => scalar.row_bits::<6, F>(raw, row, cols, &mut emit)?,
                8 => scalar.row_bits::<8, F>(raw, row, cols, &mut emit)?,
                _ => return Err("invalid prepared bit width".into()),
            },
        }
        Ok(())
    }

    pub(super) fn each<F: FnMut(f32) -> Result<(), String>>(
        &self,
        raw: &[u8],
        shape: &[usize],
        mut emit: F,
    ) -> Result<(), String> {
        let cols = if matches!(self, Self::Scalar(_)) {
            shape[1]
        } else {
            shape.iter().product()
        };
        let rows = if matches!(self, Self::Scalar(_)) {
            shape[0]
        } else {
            1
        };
        for row in 0..rows {
            self.row(raw, row, cols, |_, value| emit(value))?;
        }
        Ok(())
    }

    fn extra_bytes(&self) -> usize {
        match self {
            Self::Scalar(s) => {
                size_of::<Scalar>()
                    + (s.scales.capacity() + s.left.capacity() + s.right.capacity()) * 4
            }
            _ => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Fused,
    Row,
}

pub struct PackedTensor {
    pub(super) raw: Vec<u8>,
    pub(super) shape: Vec<usize>,
    pub(super) layout: Layout,
}

pub struct DenseTensor {
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn zeros<T: Default + Clone>(count: usize) -> Result<Vec<T>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(count).map_err(|e| e.to_string())?;
    values.resize(count, T::default());
    Ok(values)
}

fn inputs(
    shape: &[usize],
    values: &[f32],
    batch: usize,
) -> Result<(usize, usize, Vec<f64>), String> {
    if shape.len() != 2 || batch == 0 {
        return Err("matmul requires a matrix and nonzero batch".into());
    }
    let (rows, cols) = (shape[0], shape[1]);
    let count = batch.checked_mul(cols).ok_or("input extent overflow")?;
    rows.checked_mul(batch).ok_or("output extent overflow")?;
    if values.len() != count || values.iter().any(|v| !v.is_finite()) {
        return Err("matmul input shape differs or contains nonfinite values".into());
    }
    let mut transposed = zeros::<f64>(count)?;
    for sample in 0..batch {
        for col in 0..cols {
            transposed[col * batch + sample] = values[sample * cols + col] as f64;
        }
    }
    Ok((rows, cols, transposed))
}

fn accumulate(acc: &mut [f64], x: &[f64], weight: f32) {
    let value = weight as f64;
    for (a, input) in acc.iter_mut().zip(x) {
        *a += value * input;
    }
}

fn finish_row(output: &mut [f32], row: usize, rows: usize, acc: &[f64]) -> Result<(), String> {
    for (sample, value) in acc.iter().enumerate() {
        let value = *value as f32;
        if !value.is_finite() {
            return Err("matmul produced a nonfinite output".into());
        }
        output[sample * rows + row] = value;
    }
    Ok(())
}

impl PackedTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    /// Owned representation and side-state allocation; process RSS is measured separately.
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + self.raw.capacity()
            + self.shape.capacity() * size_of::<usize>()
            + self.layout.extra_bytes()
    }
    pub(super) fn decode_chunks<F: FnMut(&[u8]) -> Result<(), String>>(
        &self,
        emit: F,
    ) -> Result<(), String> {
        let mut out = Output::new(emit);
        self.layout.each(&self.raw, &self.shape, |v| out.push(v))?;
        out.finish()
    }
    pub fn to_dense(&self) -> Result<DenseTensor, String> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.elements())
            .map_err(|e| e.to_string())?;
        self.layout.each(&self.raw, &self.shape, |v| {
            values.push(v);
            Ok(())
        })?;
        Ok(DenseTensor {
            shape: self.shape.clone(),
            values,
        })
    }
    pub fn matmul(&self, values: &[f32], batch: usize, kernel: Kernel) -> Result<Vec<f32>, String> {
        let (rows, cols, x) = inputs(&self.shape, values, batch)?;
        let mut result = zeros::<f32>(rows * batch)?;
        let mut acc = zeros::<f64>(batch)?;
        let mut scratch = if kernel == Kernel::Row {
            zeros::<f32>(cols)?
        } else {
            Vec::new()
        };
        for row in 0..rows {
            acc.fill(0.0);
            if kernel == Kernel::Fused {
                self.layout.row(&self.raw, row, cols, |col, weight| {
                    accumulate(&mut acc, &x[col * batch..(col + 1) * batch], weight);
                    Ok(())
                })?;
            } else {
                self.layout.row(&self.raw, row, cols, |col, weight| {
                    scratch[col] = weight;
                    Ok(())
                })?;
                for (col, weight) in scratch.iter().enumerate() {
                    accumulate(&mut acc, &x[col * batch..(col + 1) * batch], *weight);
                }
            }
            finish_row(&mut result, row, rows, &acc)?;
        }
        Ok(result)
    }
}

impl DenseTensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn owned_bytes(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * 4 + self.shape.capacity() * size_of::<usize>()
    }
    pub fn matmul(&self, values: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        let (rows, cols, x) = inputs(&self.shape, values, batch)?;
        let mut result = zeros::<f32>(rows * batch)?;
        let mut acc = zeros::<f64>(batch)?;
        for row in 0..rows {
            acc.fill(0.0);
            for (col, weight) in self.values[row * cols..(row + 1) * cols].iter().enumerate() {
                accumulate(&mut acc, &x[col * batch..(col + 1) * batch], *weight);
            }
            finish_row(&mut result, row, rows, &acc)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_packed_methods_preserve_existing_double_accumulation() {
        let values: Vec<f32> = (0..63).map(|i| ((i as f32) - 19.0) / 7.0).collect();
        let raw: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let count = raw.len();
        let tensor = PackedTensor {
            raw,
            shape: vec![7, 9],
            layout: Layout::F32(0..count),
        };
        let dense = tensor.to_dense().unwrap();
        for batch in [1, 3, 9] {
            let input: Vec<f32> = (0..batch * 9).map(|i| ((i as f32) - 7.0) / 11.0).collect();
            let rows = input
                .as_chunks::<9>()
                .0
                .iter()
                .map(|x| x.to_vec())
                .collect::<Vec<_>>();
            let expected = crate::metrics::cpu_matmul_batch(&values, &rows, 7, 9).concat();
            for actual in [
                tensor.matmul(&input, batch, Kernel::Fused).unwrap(),
                tensor.matmul(&input, batch, Kernel::Row).unwrap(),
                dense.matmul(&input, batch).unwrap(),
            ] {
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        assert!(tensor.matmul(&[], 0, Kernel::Fused).is_err());
        assert!(tensor.matmul(&[0.0; 8], 1, Kernel::Row).is_err());
        assert!(tensor.matmul(&[f32::NAN; 9], 1, Kernel::Fused).is_err());
    }
}
