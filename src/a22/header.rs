//! Bounded authoritative F32 GGUF-prefix validation for A22 archives.
use std::collections::{BTreeMap, HashSet};

pub(super) struct Extent {
    pub shape: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
}

pub(super) struct Header {
    pub tensors: BTreeMap<String, Extent>,
    pub end: u64,
    pub padded_end: u64,
}

struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.offset.checked_add(n).ok_or("prefix extent overflow")?;
        let value = self
            .data
            .get(self.offset..end)
            .ok_or("truncated GGUF prefix")?;
        self.offset = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn length(&mut self) -> Result<usize, String> {
        usize::try_from(self.u64()?).map_err(|_| "prefix length exceeds platform".into())
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.length()?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "invalid GGUF name/key UTF-8".into())
    }
    fn skip_string(&mut self) -> Result<(), String> {
        let n = self.length()?;
        self.take(n)?;
        Ok(())
    }
    fn width(kind: u32) -> Option<usize> {
        match kind {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        }
    }
    fn skip(&mut self, kind: u32) -> Result<(), String> {
        if let Some(width) = Self::width(kind) {
            self.take(width)?;
        } else if kind == 8 {
            self.skip_string()?;
        } else if kind == 9 {
            let element = self.u32()?;
            let count = self.length()?;
            if let Some(width) = Self::width(element) {
                self.take(count.checked_mul(width).ok_or("metadata array overflow")?)?;
            } else if element == 8 && count <= (self.data.len() - self.offset) / 8 {
                for _ in 0..count {
                    self.skip_string()?;
                }
            } else {
                return Err("unsupported or invalid GGUF metadata array".into());
            }
        } else {
            return Err("unknown GGUF metadata type".into());
        }
        Ok(())
    }
}

pub(super) fn parse(data: &[u8]) -> Result<Header, String> {
    let mut r = Cursor { data, offset: 0 };
    if r.take(4)? != b"GGUF" || !(2..=3).contains(&r.u32()?) {
        return Err("unsupported stored GGUF header".into());
    }
    let count = r.length()?;
    let keys_count = r.length()?;
    if count == 0 || count > 1 << 20 || keys_count > 1 << 20 {
        return Err("invalid GGUF counts".into());
    }
    let mut keys = HashSet::new();
    let mut alignment = 32u64;
    for _ in 0..keys_count {
        let key = r.string()?;
        if !keys.insert(key.clone()) {
            return Err("duplicate GGUF metadata key".into());
        }
        let kind = r.u32()?;
        if key == "general.alignment" {
            if kind != 4 {
                return Err("invalid GGUF alignment type".into());
            }
            alignment = r.u32()? as u64;
        } else {
            r.skip(kind)?;
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("invalid GGUF alignment".into());
    }
    let mut tensors = BTreeMap::new();
    for _ in 0..count {
        let name = r.string()?;
        let dimensions = r.u32()? as usize;
        if name.is_empty() || !(1..=8).contains(&dimensions) || tensors.contains_key(&name) {
            return Err("invalid or duplicate GGUF tensor".into());
        }
        let mut shape = Vec::new();
        let mut bytes = 4u64;
        for _ in 0..dimensions {
            let dim = r.length()?;
            if dim == 0 {
                return Err("empty GGUF tensor dimension".into());
            }
            bytes = bytes
                .checked_mul(dim as u64)
                .ok_or("tensor extent overflow")?;
            shape.push(dim);
        }
        shape.reverse();
        if r.u32()? != 0 {
            return Err("A22 native export requires stored F32 GGUF types".into());
        }
        let relative = r.u64()?;
        if relative % alignment != 0 {
            return Err("unaligned tensor offset".into());
        }
        let offset = (data.len() as u64)
            .checked_add(relative)
            .ok_or("tensor offset overflow")?;
        tensors.insert(
            name,
            Extent {
                shape,
                offset,
                bytes,
            },
        );
    }
    let padded = (r.offset as u64)
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or("prefix size overflow")?;
    if padded != data.len() as u64 {
        return Err("stored prefix length differs from GGUF table".into());
    }
    let mut ranges: Vec<_> = tensors.values().map(|t| (t.offset, t.bytes)).collect();
    ranges.sort_unstable();
    let mut end = data.len() as u64;
    for (offset, bytes) in ranges {
        if offset < end {
            return Err("overlapping GGUF tensors".into());
        }
        end = offset
            .checked_add(bytes)
            .ok_or("GGUF tensor end overflow")?;
    }
    let padded_end = end
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or("GGUF size overflow")?;
    Ok(Header {
        tensors,
        end,
        padded_end,
    })
}
