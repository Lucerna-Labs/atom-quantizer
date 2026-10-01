//! q2_composer — Composer 2: a Q2-targeting composer that runs *on top of*
//! Composer 1's dequantized output tensor.
//!
//! Composition contract:
//!   [Original W] -> [Composer 1 (untouched)] -> [W1] -> [Composer 2 (this)] -> [W2]
//!
//! Composer 2 does NOT replace Composer 1. It composes on top of Composer 1's
//! output — the atoms in `stacks.rs` are UNTOUCHED, the picker in `main.rs`
//! is UNTOUCHED. This module carries its own primitive implementations so it
//! is genuinely independent of the current kit.

use atom_quantizer::wq;

const BLOCK: usize = 32;

// ─────────────────────── self-contained shape primitives ───────────────────────

pub fn row_dc_remove(w: &[f32], out_len: usize, in_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut means = Vec::with_capacity(out_len);
    let mut resid = vec![0f32; w.len()];
    for r in 0..out_len {
        let base = r * in_len;
        let mut s = 0.0f64;
        for c in 0..in_len {
            s += w[base + c] as f64;
        }
        let m = (s / in_len as f64) as f32;
        means.push(m);
        for c in 0..in_len {
            resid[base + c] = w[base + c] - m;
        }
    }
    (means, resid)
}

pub fn row_dc_restore(
    mut resid: Vec<f32>,
    means: &[f32],
    out_len: usize,
    in_len: usize,
) -> Vec<f32> {
    for r in 0..out_len {
        let base = r * in_len;
        let m = means.get(r).copied().unwrap_or(0.0);
        for c in 0..in_len {
            resid[base + c] += m;
        }
    }
    resid
}

fn fwht_orthonormal(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for x in v.iter_mut() {
        *x *= s;
    }
}

pub fn row_hadamard_8_apply(v: &mut [f32], out_len: usize, in_len: usize) {
    if in_len < 8 {
        return;
    }
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + 8 <= in_len {
            let seg = &mut v[base + c..base + c + 8];
            fwht_orthonormal(seg);
            c += 8;
        }
    }
}

pub fn dct_row_8_apply(v: &mut [f32], out_len: usize, in_len: usize, inverse: bool) {
    if in_len < 8 {
        return;
    }
    let n = 8;
    let c0 = (1.0f32 / n as f32).sqrt();
    let ck = (2.0f32 / n as f32).sqrt();
    let mut m = [[0f32; 8]; 8];
    for (k, row) in m.iter_mut().enumerate().take(n) {
        for (i, cell) in row.iter_mut().enumerate().take(n) {
            let scale = if k == 0 { c0 } else { ck };
            *cell = scale
                * ((core::f32::consts::PI * (2 * i + 1) as f32 * k as f32) / (2.0 * n as f32))
                    .cos();
        }
    }
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + 8 <= in_len {
            let mut seg = [0f32; 8];
            for i in 0..8 {
                seg[i] = v[base + c + i];
            }
            let mut y = [0f32; 8];
            for k in 0..8 {
                let mut s = 0f32;
                for i in 0..8 {
                    let mm = if inverse { m[i][k] } else { m[k][i] };
                    s += mm * seg[i];
                }
                y[k] = s;
            }
            for i in 0..8 {
                v[base + c + i] = y[i];
            }
            c += 8;
        }
    }
}

fn haar8_matrix() -> &'static [[f32; 8]; 8] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    CACHE.get_or_init(|| {
        let mut state: u32 = 0xdead_beef;
        let mut lcg = || -> f32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32;
            2.0 * u - 1.0
        };
        let mut m = [[0f32; 8]; 8];
        for row in m.iter_mut() {
            for cell in row.iter_mut() {
                *cell = lcg();
            }
        }
        for j in 0..8 {
            for k in 0..j {
                let dot = m.iter().map(|row| row[k] * row[j]).sum::<f32>();
                for row in &mut m {
                    let prior = row[k];
                    row[j] -= dot * prior;
                }
            }
            let nrm = m.iter().map(|row| row[j] * row[j]).sum::<f32>();
            let nrm = nrm.sqrt().max(1e-20);
            for row in &mut m {
                row[j] /= nrm;
            }
        }
        m
    })
}

pub fn haar_mix_8_apply(v: &mut [f32], out_len: usize, in_len: usize, inverse: bool) {
    if in_len < 8 {
        return;
    }
    let m = haar8_matrix();
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + 8 <= in_len {
            let mut seg = [0f32; 8];
            for i in 0..8 {
                seg[i] = v[base + c + i];
            }
            let mut y = [0f32; 8];
            for k in 0..8 {
                let mut s = 0f32;
                for i in 0..8 {
                    let mm = if inverse { m[i][k] } else { m[k][i] };
                    s += mm * seg[i];
                }
                y[k] = s;
            }
            for i in 0..8 {
                v[base + c + i] = y[i];
            }
            c += 8;
        }
    }
}

pub fn outlier_extract(w: &mut [f32], k_per_block: usize) -> Vec<(usize, f32)> {
    if k_per_block == 0 {
        return Vec::new();
    }
    let mut side = Vec::new();
    for (bi, blk) in w.chunks_mut(BLOCK).enumerate() {
        let mut order: Vec<usize> = (0..blk.len()).collect();
        order.sort_by(|&a, &b| {
            blk[b]
                .abs()
                .partial_cmp(&blk[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in order.iter().take(k_per_block.min(blk.len())) {
            side.push((bi * BLOCK + j, blk[j]));
            blk[j] = 0.0;
        }
    }
    side
}

pub fn outlier_restore(mut w: Vec<f32>, side: &[(usize, f32)]) -> Vec<f32> {
    for &(idx, v) in side {
        if idx < w.len() {
            w[idx] = v;
        }
    }
    w
}

pub fn row_l2_norms(v: &[f32], out_len: usize, in_len: usize) -> Vec<f32> {
    let mut n = Vec::with_capacity(out_len);
    for r in 0..out_len {
        let base = r * in_len;
        let mut s = 0f64;
        for c in 0..in_len {
            let x = v[base + c] as f64;
            s += x * x;
        }
        n.push(s.sqrt() as f32);
    }
    n
}

pub fn preserve_row_norm(
    mut v: Vec<f32>,
    target_norms: &[f32],
    out_len: usize,
    in_len: usize,
) -> Vec<f32> {
    for r in 0..out_len {
        let base = r * in_len;
        let mut cur = 0f64;
        for c in 0..in_len {
            let x = v[base + c] as f64;
            cur += x * x;
        }
        let cur = cur.sqrt() as f32;
        if cur > 1e-12 && r < target_norms.len() {
            let s = target_norms[r] / cur;
            for c in 0..in_len {
                v[base + c] *= s;
            }
        }
    }
    v
}

// ─────────────────────── Q2 quantizer primitives ───────────────────────

fn sym_q2_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let max_abs = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if max_abs == 0.0 {
            out.extend(blk.iter().copied());
            continue;
        }
        let scale = max_abs;
        for &v in blk {
            let q = (v / scale).round().clamp(-1.0, 1.0);
            out.push(q * scale);
        }
    }
    out
}

fn asym_q2_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let lo = blk.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let hi = blk.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let scale = if hi > lo { (hi - lo) / 3.0 } else { 1.0 };
        for &v in blk {
            let q = ((v - lo) / scale).round().clamp(0.0, 3.0);
            out.push(lo + q * scale);
        }
    }
    out
}

fn superpose_q2_recon(data: &[f32]) -> Vec<f32> {
    let a = sym_q2_recon(data);
    let b = asym_q2_recon(data);
    let n = a.len().min(b.len());
    (0..n).map(|i| 0.5 * a[i] + 0.5 * b[i]).collect()
}

fn refract_q2_recon(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    let mut off = 0usize;
    for blk in data.chunks(BLOCK) {
        let max_abs = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if max_abs == 0.0 {
            for (i, &v) in blk.iter().enumerate() {
                out[off + i] = v;
            }
            off += blk.len();
            continue;
        }
        let thr = 0.30 * max_abs;
        let bulk: Vec<f32> = blk.iter().copied().filter(|&v| v.abs() < thr).collect();
        let tail: Vec<f32> = blk.iter().copied().filter(|&v| v.abs() >= thr).collect();
        let bulk_r = if bulk.is_empty() {
            Vec::new()
        } else {
            sym_q2_recon(&bulk)
        };
        let tail_r = if tail.is_empty() {
            Vec::new()
        } else {
            asym_q2_recon(&tail)
        };
        let mut bi = 0usize;
        let mut ti = 0usize;
        for (i, &v) in blk.iter().enumerate() {
            if v.abs() >= thr {
                out[off + i] = tail_r[ti];
                ti += 1;
            } else {
                out[off + i] = bulk_r[bi];
                bi += 1;
            }
        }
        off += blk.len();
    }
    out
}

/// Estimated packed payload, excluding an as-yet undefined record/container header.
/// Side data uses the f32 precision actually used by the reconstruction.
pub fn estimated_bits_per_weight(n: usize, rows: usize, cols: usize, stack: &Q2Stack) -> f64 {
    let blocks = n.div_ceil(BLOCK);
    let streams = if stack.quant == Q2Quant::Superpose {
        2
    } else {
        1
    };
    let code_bytes = n.div_ceil(4) * streams;
    let scale_bytes = blocks
        * match stack.quant {
            Q2Quant::Sym => 4,
            Q2Quant::Asym => 8,
            Q2Quant::Superpose | Q2Quant::Refract => 12,
        };
    let mask_bytes = if stack.quant == Q2Quant::Refract {
        n.div_ceil(8)
    } else {
        0
    };
    let k = stack.k_extract.min(BLOCK);
    let exceptions = n / BLOCK * k + (n % BLOCK).min(k);
    let row_fields = if rows > 1 && cols > 1 {
        usize::from(stack.pre == Q2Pre::RowDc) + usize::from(stack.post == Q2Post::Preserve)
    } else {
        0
    };
    let bytes = code_bytes + scale_bytes + mask_bytes + exceptions * 8 + row_fields * rows * 4;
    8.0 * bytes as f64 / n.max(1) as f64
}

#[cfg(test)]
mod storage_tests {
    use super::*;

    #[test]
    fn estimates_charge_all_streams_masks_row_state_and_partial_blocks() {
        let base = Q2Stack {
            name: "test",
            pre: Q2Pre::Null,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Null,
        };
        assert_eq!(estimated_bits_per_weight(32, 1, 32, &base), 3.0);
        let superposed = Q2Stack {
            quant: Q2Quant::Superpose,
            ..base
        };
        assert_eq!(estimated_bits_per_weight(32, 1, 32, &superposed), 7.0);
        let refracted = Q2Stack {
            quant: Q2Quant::Refract,
            ..base
        };
        assert_eq!(estimated_bits_per_weight(32, 1, 32, &refracted), 6.0);
        let rows = Q2Stack {
            pre: Q2Pre::RowDc,
            post: Q2Post::Preserve,
            k_extract: 4,
            ..base
        };
        assert_eq!(estimated_bits_per_weight(64, 2, 32, &rows), 13.0);
        let partial = Q2Stack {
            k_extract: 4,
            ..base
        };
        assert_eq!(
            estimated_bits_per_weight(33, 1, 33, &partial) * 33.0 / 8.0,
            57.0
        );
    }
}

// ─────────────────────── Composer 2 candidate stacks ───────────────────────

#[derive(Clone, Copy, Debug)]
pub struct Q2Stack {
    pub name: &'static str,
    pub pre: Q2Pre,
    pub k_extract: usize,
    pub quant: Q2Quant,
    pub post: Q2Post,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q2Pre {
    Null,
    RowDc,
    RowH8,
    RowDct8,
    HaarMix8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q2Quant {
    Sym,
    #[expect(dead_code, reason = "reserved for the affine Q2 experiment surface")]
    Asym,
    Superpose,
    Refract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q2Post {
    Null,
    Preserve,
}

pub fn q2_candidates() -> Vec<Q2Stack> {
    vec![
        Q2Stack {
            name: "naked sym-Q2",
            pre: Q2Pre::Null,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Null,
        },
        Q2Stack {
            name: "rowDC + Q2 + preserve",
            pre: Q2Pre::RowDc,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "rowH8 + Q2 + preserve",
            pre: Q2Pre::RowH8,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "rowDCT8 + Q2 + preserve",
            pre: Q2Pre::RowDct8,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "haarMix8 + Q2 + preserve",
            pre: Q2Pre::HaarMix8,
            k_extract: 0,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "top1 + Q2 + preserve",
            pre: Q2Pre::Null,
            k_extract: 1,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "top2 + Q2 + preserve",
            pre: Q2Pre::Null,
            k_extract: 2,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "top4 + Q2 + preserve",
            pre: Q2Pre::Null,
            k_extract: 4,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "rowDC + top4 + Q2 + preserve",
            pre: Q2Pre::RowDc,
            k_extract: 4,
            quant: Q2Quant::Sym,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "rowDC + top4 + superposeQ2 + preserve",
            pre: Q2Pre::RowDc,
            k_extract: 4,
            quant: Q2Quant::Superpose,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "rowDC + top4 + refractQ2 + preserve",
            pre: Q2Pre::RowDc,
            k_extract: 4,
            quant: Q2Quant::Refract,
            post: Q2Post::Preserve,
        },
        Q2Stack {
            name: "haarMix8 + top4 + refractQ2 + preserve",
            pre: Q2Pre::HaarMix8,
            k_extract: 4,
            quant: Q2Quant::Refract,
            post: Q2Post::Preserve,
        },
    ]
}

pub fn apply_q2_composer(w1: &[f32], stack: &Q2Stack, out_len: usize, in_len: usize) -> Vec<f32> {
    let is_2d = out_len > 1 && in_len > 1;

    let mut cur = w1.to_vec();
    let mut row_means: Option<Vec<f32>> = None;
    if is_2d {
        match stack.pre {
            Q2Pre::Null => {}
            Q2Pre::RowDc => {
                let (m, r) = row_dc_remove(&cur, out_len, in_len);
                row_means = Some(m);
                cur = r;
            }
            Q2Pre::RowH8 if in_len.is_multiple_of(8) => {
                row_hadamard_8_apply(&mut cur, out_len, in_len)
            }
            Q2Pre::RowDct8 if in_len.is_multiple_of(8) => {
                dct_row_8_apply(&mut cur, out_len, in_len, false)
            }
            Q2Pre::HaarMix8 if in_len.is_multiple_of(8) => {
                haar_mix_8_apply(&mut cur, out_len, in_len, false)
            }
            _ => {}
        }
    }

    let target_norms = if is_2d && stack.post == Q2Post::Preserve {
        Some(row_l2_norms(w1, out_len, in_len))
    } else {
        None
    };

    let outlier_side = if stack.k_extract > 0 {
        outlier_extract(&mut cur, stack.k_extract)
    } else {
        Vec::new()
    };

    let mut recon = match stack.quant {
        Q2Quant::Sym => sym_q2_recon(&cur),
        Q2Quant::Asym => asym_q2_recon(&cur),
        Q2Quant::Superpose => superpose_q2_recon(&cur),
        Q2Quant::Refract => refract_q2_recon(&cur),
    };

    if !outlier_side.is_empty() {
        recon = outlier_restore(recon, &outlier_side);
    }

    if is_2d {
        match stack.pre {
            Q2Pre::Null => {}
            Q2Pre::RowDc => {
                if let Some(m) = &row_means {
                    recon = row_dc_restore(recon, m, out_len, in_len);
                }
            }
            Q2Pre::RowH8 if in_len.is_multiple_of(8) => {
                row_hadamard_8_apply(&mut recon, out_len, in_len)
            }
            Q2Pre::RowDct8 if in_len.is_multiple_of(8) => {
                dct_row_8_apply(&mut recon, out_len, in_len, true)
            }
            Q2Pre::HaarMix8 if in_len.is_multiple_of(8) => {
                haar_mix_8_apply(&mut recon, out_len, in_len, true)
            }
            _ => {}
        }
    }

    if let Some(n) = &target_norms {
        recon = preserve_row_norm(recon, n, out_len, in_len);
    }

    recon
}

// ─────────────────────── Legacy cosine-only Composer 1 approximation ───────

pub fn apply_composer1(w: &[f32], out_len: usize, in_len: usize) -> (Vec<f32>, wq::QuantStrategy) {
    use wq::{cosine, encode_with, expand, WqConfig, CANDIDATES};
    let is_2d = out_len > 1 && in_len > 1;

    let use_pre = is_2d && in_len >= 8 && in_len.is_multiple_of(8);
    let encode_input: Vec<f32> = if use_pre {
        let mut v = w.to_vec();
        row_hadamard_8_apply(&mut v, out_len, in_len);
        v
    } else {
        w.to_vec()
    };

    let cfg = WqConfig {
        protect: false,
        key: WqConfig::default().key,
        asym: false,
    };
    let mut fallback: Option<wq::QuantStrategy> = None;
    let mut fallback_recon: Option<Vec<f32>> = None;
    for &s in &CANDIDATES {
        let cand = encode_with(&encode_input, s, &cfg);
        let mut recon = expand(&cand);
        if use_pre {
            row_hadamard_8_apply(&mut recon, out_len, in_len);
        }
        let cos = cosine(w, &recon);
        if cos >= 0.99 {
            if is_2d {
                let target = row_l2_norms(w, out_len, in_len);
                recon = preserve_row_norm(recon, &target, out_len, in_len);
            }
            return (recon, s);
        }
        fallback = Some(s);
        fallback_recon = Some(recon);
    }
    let strat = fallback.expect("CANDIDATES non-empty");
    let mut recon = fallback_recon.expect("CANDIDATES non-empty");
    if is_2d {
        let target = row_l2_norms(w, out_len, in_len);
        recon = preserve_row_norm(recon, &target, out_len, in_len);
    }
    (recon, strat)
}
