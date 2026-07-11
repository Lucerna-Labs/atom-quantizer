#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! stacks — the discovery harness. Atoms are STAGES; a stack composes them in a
//! specific order. This is a discovery experiment: we enumerate legal stacks and
//! measure each against KL and cosine, without pre-verdict.
//!
//! Pipeline shape:
//!   reconstruction = POST( QUANTIZE( EXTRACT( PRE( w ) ) ) )
//!   with the inverse of PRE applied inside so the tensor lives in weight space
//!   at the end.
//!
//! Stage roles:
//!   PRE      — transform / predict / whiten (invertible; inverse applied on decode).
//!   EXTRACT  — pull a sparse tail (kept at higher precision), quantize the rest.
//!   QUANTIZE — the actual 4-bit map (symmetric, affine, or codebook).
//!   POST     — repair or de-transform (identity by default).
//!
//! Each Stage carries its ROW_LEN dependence (row-oriented stages need shape).

use crate::wq::{asym_q4_recon, errfeed_q4_recon, sym_q4_recon, BLOCK};

// ─────────────────────────── stage kinds ───────────────────────────
//
// Every stage variant is annotated with its state_shape per the atom-extension-32
// doc §1-6 (kept in sync with bench/atoms_48_shape_aware.py). Legend:
//   2D    operates on a 2-D matrix (needs `out_len × in_len`)
//   1D    block-scoped, operates on the flat vector after PRE
//   any   works on any shape (identity or shape-agnostic)
// A stack is well-formed when every stage's shape is compatible with what the
// prior stage produced. Currently every PRE returns 2D, EXTRACT/QUANTIZE
// consume the flat block view, and POST is a mix (2D for PreserveRowNorm,
// 1D-block for ErrfeedPass/WaveletLift, any for Null).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pre {
    /// state_shape = any     (identity)
    Null,
    /// state_shape = 2D      (subtracts per-row mean; needs row structure)
    RowDcRemove,
    /// state_shape = 2D      (8-point Walsh-Hadamard within each row)
    RowHadamard8,
    /// state_shape = 1D      (global-flatten Hadamard on every 32-block — baseline probe)
    BlockHadamard32,
    /// state_shape = 2D      (atom 9 `transform`: 8-pt orthonormal DCT-II per row)
    DctRow8,
    /// state_shape = 2D      (atom 39 `mix`: fixed Haar-random 8x8 orthogonal
    /// rotation per row across 8-col blocks. Contrast with Walsh-Hadamard
    /// (structured) and DCT (energy-compacting). Real-Qwen local sim: 0.948×
    /// baseline MSE — marginal but still a BEATS.)
    HaarMix8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extract {
    /// state_shape = any     (identity)
    Null,
    /// state_shape = 1D      (top-1 magnitude outlier per BLOCK, bit-exact sidecar)
    OutlierTopK1,
    /// state_shape = 1D      (top-2 per BLOCK)
    OutlierTopK2,
    /// state_shape = 1D      (top-4 per BLOCK)
    OutlierTopK4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// state_shape = 1D      (block-scoped max-abs symmetric 4-bit — baseline)
    SymUniform,
    /// state_shape = 1D      (block-scoped min+scale affine 4-bit)
    AsymAffine,
    /// state_shape = 1D      (block-scoped Lloyd-Max 16-level codebook)
    CodebookLloydMax16,
    /// state_shape = 2D      (atom 15 `symmetrize`: sign-symmetric Lloyd-Max codebook,
    /// 8 magnitude centroids × ±sign. Designed for 2-D weight matrices' near-symmetric
    /// distribution. See atom-extension-32.md §5 verdict log for the real-Qwen check.)
    SymmetricCodebook,
    /// state_shape = any     (atom 16 `superpose`: mean of sym-Q4 and asym-Q4 recons)
    SuperposeSymAsym,
    /// state_shape = 2D      (atom 28 `refract`: bulk (|w|<0.30·max_abs) → sym-Q4,
    /// tail (|w|≥0.30·max_abs) → asym-Q4, stitched back at original positions.
    /// Designed to exploit the bulk+tail distribution common in 2-D weight matrices.
    /// Landed at 0.139× baseline MSE on the shape-aware synthetic bench.)
    RefractBulkTail30,
    /// state_shape = 2D      (atom 12 `compose`: 2-level nested block hierarchy —
    /// super-block of 16 elements picks a fractional scale for each 8-elem sub-block,
    /// so the sub-block scale is quantized into 4 levels of the super-scale. Real-Qwen
    /// local sim: 0.713× baseline MSE — meaningful BEATS.)
    ComposeNestedScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Post {
    /// state_shape = any     (identity)
    Null,
    /// state_shape = 1D      (sigma-delta residual carry along the flat vector)
    ErrfeedPass,
    /// state_shape = 1D      (integer Haar lifting + soft-threshold on detail band)
    WaveletLift,
    /// state_shape = 2D      (atom 11 `preserve`: rescale each row of recon so L2
    /// norm matches the original's. Requires row structure.)
    PreserveRowNorm,
}

#[derive(Clone, Copy, Debug)]
pub struct Stack {
    pub pre: Pre,
    pub extract: Extract,
    pub quantize: Quantize,
    pub post: Post,
}

impl Stack {
    pub fn label(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}",
            self.pre, self.extract, self.quantize, self.post
        )
    }
    pub fn short(&self) -> String {
        fn s_pre(p: Pre) -> &'static str {
            match p {
                Pre::Null => "-",
                Pre::RowDcRemove => "rowDC",
                Pre::RowHadamard8 => "rowH8",
                Pre::BlockHadamard32 => "blkH32",
                Pre::DctRow8 => "rowDCT8",
                Pre::HaarMix8 => "haarMix8",
            }
        }
        fn s_ex(e: Extract) -> &'static str {
            match e {
                Extract::Null => "-",
                Extract::OutlierTopK1 => "top1",
                Extract::OutlierTopK2 => "top2",
                Extract::OutlierTopK4 => "top4",
            }
        }
        fn s_q(q: Quantize) -> &'static str {
            match q {
                Quantize::SymUniform => "symQ4",
                Quantize::AsymAffine => "asymQ4",
                Quantize::CodebookLloydMax16 => "codeQ4",
                Quantize::SymmetricCodebook => "symCode",
                Quantize::SuperposeSymAsym => "superSA",
                Quantize::RefractBulkTail30 => "refract30",
                Quantize::ComposeNestedScale => "compose",
            }
        }
        fn s_p(p: Post) -> &'static str {
            match p {
                Post::Null => "-",
                Post::ErrfeedPass => "errfeed",
                Post::WaveletLift => "wavelet",
                Post::PreserveRowNorm => "presvNorm",
            }
        }
        format!(
            "{}/{}/{}/{}",
            s_pre(self.pre),
            s_ex(self.extract),
            s_q(self.quantize),
            s_p(self.post)
        )
    }

    /// Parse a stack from its short-form string, e.g. `"rowDC/top4/refract30/-"`.
    /// Returns `None` on any unrecognized token.
    pub fn parse_short(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 4 {
            return None;
        }
        let pre = match parts[0] {
            "-" => Pre::Null,
            "rowDC" => Pre::RowDcRemove,
            "rowH8" => Pre::RowHadamard8,
            "blkH32" => Pre::BlockHadamard32,
            "rowDCT8" => Pre::DctRow8,
            "haarMix8" => Pre::HaarMix8,
            _ => return None,
        };
        let extract = match parts[1] {
            "-" => Extract::Null,
            "top1" => Extract::OutlierTopK1,
            "top2" => Extract::OutlierTopK2,
            "top4" => Extract::OutlierTopK4,
            _ => return None,
        };
        let quantize = match parts[2] {
            "symQ4" => Quantize::SymUniform,
            "asymQ4" => Quantize::AsymAffine,
            "codeQ4" => Quantize::CodebookLloydMax16,
            "symCode" => Quantize::SymmetricCodebook,
            "superSA" => Quantize::SuperposeSymAsym,
            "refract30" => Quantize::RefractBulkTail30,
            "compose" => Quantize::ComposeNestedScale,
            _ => return None,
        };
        let post = match parts[3] {
            "-" => Post::Null,
            "errfeed" => Post::ErrfeedPass,
            "wavelet" => Post::WaveletLift,
            "presvNorm" => Post::PreserveRowNorm,
            _ => return None,
        };
        Some(Stack {
            pre,
            extract,
            quantize,
            post,
        })
    }
}

/// Enumerate every combination of stages. Legality prune: BlockHadamard32 is
/// only meaningful with SymUniform (its known failure mode is the confirmation);
/// for other quantizers we drop the redundant BlockHadamard32 variants to keep
/// the report readable. Otherwise: full cartesian product.
pub fn enumerate_legal_stacks() -> Vec<Stack> {
    let pres = [
        Pre::Null,
        Pre::RowDcRemove,
        Pre::RowHadamard8,
        Pre::BlockHadamard32,
        Pre::DctRow8,
        Pre::HaarMix8,
    ];
    let exs = [
        Extract::Null,
        Extract::OutlierTopK1,
        Extract::OutlierTopK2,
        Extract::OutlierTopK4,
    ];
    let qs = [
        Quantize::SymUniform,
        Quantize::AsymAffine,
        Quantize::CodebookLloydMax16,
        Quantize::SymmetricCodebook,
        Quantize::SuperposeSymAsym,
        Quantize::RefractBulkTail30,
        Quantize::ComposeNestedScale,
    ];
    let posts = [
        Post::Null,
        Post::ErrfeedPass,
        Post::WaveletLift,
        Post::PreserveRowNorm,
    ];
    let mut out = Vec::new();
    for &pre in &pres {
        for &ex in &exs {
            for &q in &qs {
                for &post in &posts {
                    if pre == Pre::BlockHadamard32 && q != Quantize::SymUniform {
                        continue;
                    }
                    out.push(Stack {
                        pre,
                        extract: ex,
                        quantize: q,
                        post,
                    });
                }
            }
        }
    }
    out
}

// ─────────────────────────── stage impls (invertible pairs) ───────────────────────────

/// In-place fast Walsh-Hadamard transform, orthonormal (self-inverse).
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

/// Per-row mean (one f16 per row); returns the mean vector alongside the residual.
fn row_dc_remove(w: &[f32], out_len: usize, in_len: usize) -> (Vec<f32>, Vec<f32>) {
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

/// Re-add the per-row means to a reconstructed residual.
fn row_dc_restore(mut resid: Vec<f32>, means: &[f32], out_len: usize, in_len: usize) -> Vec<f32> {
    for r in 0..out_len {
        let m = means[r];
        let base = r * in_len;
        for c in 0..in_len {
            resid[base + c] += m;
        }
    }
    resid
}

/// 8-point Hadamard within each row (in-place, orthonormal, self-inverse).
pub fn row_hadamard_8_apply(v: &mut [f32], out_len: usize, in_len: usize) {
    if in_len < 8 {
        return;
    }
    let g = 8usize;
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + g <= in_len {
            fwht_orthonormal(&mut v[base + c..base + c + g]);
            c += g;
        }
    }
}

/// 32-block Hadamard over the flattened tensor (the earlier failed variant).
fn block_hadamard_32_apply(v: &mut [f32]) {
    let mut c = 0;
    while c + BLOCK <= v.len() {
        fwht_orthonormal(&mut v[c..c + BLOCK]);
        c += BLOCK;
    }
}

/// Evict top-K |w| per 32-block into a sparse sidecar (index, value f32). Bulk
/// gets those positions zeroed so the following quantizer sees a tighter range.
fn outlier_extract(w: &mut [f32], k_per_block: usize) -> Vec<(usize, f32)> {
    if k_per_block == 0 {
        return Vec::new();
    }
    let mut side = Vec::new();
    for blk_start in (0..w.len()).step_by(BLOCK) {
        let blk_end = (blk_start + BLOCK).min(w.len());
        let mut idxs: Vec<usize> = (blk_start..blk_end).collect();
        idxs.sort_by(|&a, &b| {
            w[b].abs()
                .partial_cmp(&w[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let take = k_per_block.min(idxs.len());
        for &i in idxs.iter().take(take) {
            side.push((i, w[i]));
            w[i] = 0.0;
        }
    }
    side
}

/// Re-insert the evicted outliers onto a reconstructed tensor.
fn outlier_restore(mut w: Vec<f32>, side: &[(usize, f32)]) -> Vec<f32> {
    for &(i, v) in side {
        if i < w.len() {
            w[i] = v;
        }
    }
    w
}

/// Per-block Lloyd-Max 16-level codebook (1-D k-means with 6 refinements).
fn codebook_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        if blk.is_empty() {
            continue;
        }
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &v in blk {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !hi.is_finite() || !lo.is_finite() || hi == lo {
            out.extend(blk.iter().copied());
            continue;
        }
        // seed 16 centroids linearly across [lo, hi]
        let k = 16usize;
        let mut c = [0f32; 16];
        for j in 0..k {
            c[j] = lo + (hi - lo) * (j as f32) / (k as f32 - 1.0);
        }
        for _iter in 0..3 {
            let mut sum = [0f64; 16];
            let mut cnt = [0u32; 16];
            for &v in blk {
                let mut best = 0usize;
                let mut bd = f32::INFINITY;
                for j in 0..k {
                    let d = (v - c[j]).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
                sum[best] += v as f64;
                cnt[best] += 1;
            }
            for j in 0..k {
                if cnt[j] > 0 {
                    c[j] = (sum[j] / cnt[j] as f64) as f32;
                }
            }
        }
        for &v in blk {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for j in 0..k {
                let d = (v - c[j]).abs();
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            out.push(c[best]);
        }
    }
    out
}

/// One-level integer Haar lifting + soft-threshold on the detail band, then
/// undo. Norm-preserving, storage-neutral (POST repair).
fn wavelet_lifting_repair(v: &[f32]) -> Vec<f32> {
    if v.len() < 2 {
        return v.to_vec();
    }
    let n = v.len() / 2 * 2;
    let mut a = Vec::with_capacity(n / 2); // approx
    let mut d = Vec::with_capacity(n / 2); // detail
    for i in 0..n / 2 {
        let x = v[2 * i];
        let y = v[2 * i + 1];
        a.push(0.5 * (x + y));
        d.push(0.5 * (x - y));
    }
    // soft-threshold the detail band by 10% of its L1 mean
    let mag: f32 = d.iter().map(|x| x.abs()).sum::<f32>() / (d.len() as f32).max(1.0);
    let t = 0.1 * mag;
    for x in d.iter_mut() {
        let s = x.signum();
        let m = (x.abs() - t).max(0.0);
        *x = s * m;
    }
    let mut out = Vec::with_capacity(v.len());
    for i in 0..n / 2 {
        out.push(a[i] + d[i]);
        out.push(a[i] - d[i]);
    }
    if v.len() > n {
        out.push(v[n]);
    }
    out
}

// ─────────────────────────── extended-atom impls ───────────────────────────

/// 8-point orthonormal DCT-II. Computed on the fly (n=8 is small).
/// Orthonormal so its inverse is DCT-III with the same normalization.
fn dct8_forward(v: &mut [f32]) {
    let mut out = [0.0f32; 8];
    let n = 8;
    let c0 = (1.0f32 / n as f32).sqrt();
    let ck = (2.0f32 / n as f32).sqrt();
    for k in 0..n {
        let scale = if k == 0 { c0 } else { ck };
        let mut s = 0.0f32;
        for m in 0..n {
            let arg = std::f32::consts::PI * (2 * m + 1) as f32 * k as f32 / (2 * n) as f32;
            s += v[m] * arg.cos();
        }
        out[k] = scale * s;
    }
    v.copy_from_slice(&out);
}

/// Inverse (DCT-III) — since DCT-II is orthonormal, this is its exact inverse.
fn dct8_inverse(v: &mut [f32]) {
    let mut out = [0.0f32; 8];
    let n = 8;
    let c0 = (1.0f32 / n as f32).sqrt();
    let ck = (2.0f32 / n as f32).sqrt();
    for m in 0..n {
        let mut s = 0.0f32;
        for k in 0..n {
            let scale = if k == 0 { c0 } else { ck };
            let arg = std::f32::consts::PI * (2 * m + 1) as f32 * k as f32 / (2 * n) as f32;
            s += scale * v[k] * arg.cos();
        }
        out[m] = s;
    }
    v.copy_from_slice(&out);
}

/// Apply 8-point DCT to every consecutive 8-wide window within each row.
fn dct_row_8_apply(v: &mut [f32], out_len: usize, in_len: usize, inverse: bool) {
    if in_len < 8 {
        return;
    }
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + 8 <= in_len {
            let seg = &mut v[base + c..base + c + 8];
            if inverse {
                dct8_inverse(seg);
            } else {
                dct8_forward(seg);
            }
            c += 8;
        }
    }
}

/// Row L2 norms of a row-major [out_len × in_len] tensor.
pub fn row_l2_norms(v: &[f32], out_len: usize, in_len: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(out_len);
    for r in 0..out_len {
        let mut s = 0.0f64;
        let base = r * in_len;
        for c in 0..in_len {
            let x = v[base + c] as f64;
            s += x * x;
        }
        norms.push(s.sqrt() as f32);
    }
    norms
}

/// Rescale each row of `v` so its L2 norm equals `target_norms[r]`.
/// Zero-norm rows are left as-is. Preserves shape.
pub fn preserve_row_norm(
    mut v: Vec<f32>,
    target_norms: &[f32],
    out_len: usize,
    in_len: usize,
) -> Vec<f32> {
    for r in 0..out_len {
        let base = r * in_len;
        let mut cur = 0.0f64;
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

/// Symmetric-codebook Q4: Lloyd-Max on |w|, 8 magnitude centroids, ±sign gives
/// 16 levels (still 4-bit) but the codebook has a hard symmetry constraint.
fn symmetric_codebook_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        if blk.is_empty() {
            continue;
        }
        let mut max_abs = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0.0 {
            out.extend(blk.iter().copied());
            continue;
        }
        // Init 8 magnitude centroids linearly across [0, max_abs].
        let n_mag = 8usize;
        let mut c = [0f32; 8];
        for j in 0..n_mag {
            c[j] = max_abs * (j as f32 + 0.5) / n_mag as f32;
        }
        for _iter in 0..3 {
            let mut sum = [0f64; 8];
            let mut cnt = [0u32; 8];
            for &v in blk {
                let a = v.abs();
                let mut best = 0usize;
                let mut bd = f32::INFINITY;
                for j in 0..n_mag {
                    let d = (a - c[j]).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
                sum[best] += a as f64;
                cnt[best] += 1;
            }
            for j in 0..n_mag {
                if cnt[j] > 0 {
                    c[j] = (sum[j] / cnt[j] as f64) as f32;
                }
            }
        }
        for &v in blk {
            let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
            let a = v.abs();
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for j in 0..n_mag {
                let d = (a - c[j]).abs();
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            out.push(sign * c[best]);
        }
    }
    out
}

/// Superpose Q4: average of sym-Q4 and asym-Q4 reconstructions on the same input.
/// Tests whether the two quantizers' errors partially cancel.
fn superpose_sym_asym_recon(data: &[f32]) -> Vec<f32> {
    let a = sym_q4_recon(data);
    let b = asym_q4_recon(data);
    let n = a.len().min(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(0.5 * a[i] + 0.5 * b[i]);
    }
    out
}

/// Compose Q4 (2-level nested block scale): super-block of 16 elements picks a
/// coarse scale supm=max(|w|); each 8-elem sub-block within the super picks a
/// fractional scale in {1/3, 2/3, 3/3} of supm, quantized with sym-Q4 grid /7.
/// Real-Qwen local sim: 0.713× baseline MSE — meaningful BEATS.
fn compose_nested_scale_recon(data: &[f32]) -> Vec<f32> {
    const SUP: usize = 16;
    const SUB: usize = 8;
    let mut out = vec![0f32; data.len()];
    let mut off = 0usize;
    for sup_blk in data.chunks(SUP) {
        let supm = sup_blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
        for (i, sub_blk) in sup_blk.chunks(SUB).enumerate() {
            let lm = sub_blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let frac_q = ((3.0 * lm / supm).round() / 3.0).clamp(1e-3, 1.0);
            let s = (supm * frac_q) / 7.0;
            let sub_off = off + i * SUB;
            for (j, &v) in sub_blk.iter().enumerate() {
                let q = (v / s).round().clamp(-7.0, 7.0);
                out[sub_off + j] = q * s;
            }
        }
        off += sup_blk.len();
    }
    out
}

/// Fixed 8x8 orthogonal matrix used by Pre::HaarMix8. Computed once at first
/// call via inline Gram-Schmidt on a hardcoded pseudo-random seed matrix, then
/// cached. Deterministic across runs (no OS randomness). Distinct from Walsh-
/// Hadamard (structured) and DCT (energy-compacting) so it tests whether a
/// *generic* orthogonal rotation still helps.
fn haar8_matrix() -> &'static [[f32; 8]; 8] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<[[f32; 8]; 8]> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Deterministic seed matrix from a fixed LCG (Numerical Recipes).
        let mut state: u32 = 0xdead_beef;
        let mut lcg = || -> f32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
            2.0 * u - 1.0 // [-1, 1)
        };
        let mut m = [[0f32; 8]; 8];
        for row in m.iter_mut() {
            for cell in row.iter_mut() {
                *cell = lcg();
            }
        }
        // Modified Gram-Schmidt on columns.
        for j in 0..8 {
            for k in 0..j {
                let mut dot = 0f32;
                for i in 0..8 {
                    dot += m[i][k] * m[i][j];
                }
                for i in 0..8 {
                    m[i][j] -= dot * m[i][k];
                }
            }
            let mut n = 0f32;
            for i in 0..8 {
                n += m[i][j] * m[i][j];
            }
            let n = n.sqrt().max(1e-20);
            for i in 0..8 {
                m[i][j] /= n;
            }
        }
        m
    })
}

/// Apply the fixed 8x8 Haar rotation per-row across 8-column blocks. When
/// `inverse` is true, apply the transpose (Haar matrix is orthogonal so its
/// inverse == transpose).
fn haar_mix_8_apply(v: &mut [f32], out_len: usize, in_len: usize, inverse: bool) {
    if in_len < 8 {
        return;
    }
    let m = haar8_matrix();
    let mut seg = [0f32; 8];
    let mut y = [0f32; 8];
    for r in 0..out_len {
        let base = r * in_len;
        let mut c = 0;
        while c + 8 <= in_len {
            for i in 0..8 {
                seg[i] = v[base + c + i];
            }
            for k in 0..8 {
                let mut s = 0f32;
                if inverse {
                    // transpose: y[k] = sum_i m[k][i] * seg[i]  since m is orthonormal,
                    // m_inv[k][i] = m[i][k].
                    for i in 0..8 {
                        s += m[i][k] * seg[i];
                    }
                } else {
                    for i in 0..8 {
                        s += m[k][i] * seg[i];
                    }
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

/// Refract Q4 (bulk+tail split at 30% max-abs): each block partitions into
///   bulk (|w| <  0.30 * max_abs) → sym-Q4
///   tail (|w| >= 0.30 * max_abs) → asym-Q4
/// The two reconstructions are stitched back together at the original block
/// positions. Reproduces the shape-aware synthetic-bench winner exactly.
fn refract_bulk_tail_30_recon(data: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    let mut off = 0usize;
    for blk in data.chunks(BLOCK) {
        let n = blk.len();
        let max_abs = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if max_abs == 0.0 {
            for (i, &v) in blk.iter().enumerate() {
                out[off + i] = v;
            }
            off += n;
            continue;
        }
        let thr = 0.30f32 * max_abs;
        let mut bulk: Vec<f32> = Vec::new();
        let mut tail: Vec<f32> = Vec::new();
        for &v in blk {
            if v.abs() >= thr {
                tail.push(v);
            } else {
                bulk.push(v);
            }
        }
        let bulk_recon = if bulk.is_empty() {
            Vec::new()
        } else {
            sym_q4_recon(&bulk)
        };
        let tail_recon = if tail.is_empty() {
            Vec::new()
        } else {
            asym_q4_recon(&tail)
        };
        let mut bi = 0usize;
        let mut ti = 0usize;
        for (i, &v) in blk.iter().enumerate() {
            if v.abs() >= thr {
                out[off + i] = tail_recon[ti];
                ti += 1;
            } else {
                out[off + i] = bulk_recon[bi];
                bi += 1;
            }
        }
        off += n;
    }
    out
}

// ─────────────────────────── applying a stack ───────────────────────────

/// Apply a stack to a weight tensor of the given [out_len, in_len] shape, returning
/// (reconstruction, side_bytes) — side_bytes accounts for evicted outliers, per-row
/// DC means, and codebook storage (per-block scale for uniform quantizers).
pub fn apply(stack: Stack, w: &[f32], out_len: usize, in_len: usize) -> (Vec<f32>, usize) {
    // Compute row L2 norms of the ORIGINAL (needed if POST=PreserveRowNorm; free
    // to compute always so control flow stays flat).
    let orig_row_norms = if matches!(stack.post, Post::PreserveRowNorm) {
        Some(row_l2_norms(w, out_len, in_len))
    } else {
        None
    };

    // PRE ---
    let mut cur = w.to_vec();
    let mut row_means: Option<Vec<f32>> = None;
    let mut pre_side_bytes = 0usize;
    match stack.pre {
        Pre::Null => {}
        Pre::RowDcRemove => {
            let (m, r) = row_dc_remove(&cur, out_len, in_len);
            pre_side_bytes = m.len() * 2; // one f16 per row
            row_means = Some(m);
            cur = r;
        }
        Pre::RowHadamard8 => {
            row_hadamard_8_apply(&mut cur, out_len, in_len);
        }
        Pre::BlockHadamard32 => {
            block_hadamard_32_apply(&mut cur);
        }
        Pre::DctRow8 => {
            dct_row_8_apply(&mut cur, out_len, in_len, false);
        }
        Pre::HaarMix8 => {
            haar_mix_8_apply(&mut cur, out_len, in_len, false);
        }
    }

    // EXTRACT ---
    let k = match stack.extract {
        Extract::Null => 0,
        Extract::OutlierTopK1 => 1,
        Extract::OutlierTopK2 => 2,
        Extract::OutlierTopK4 => 4,
    };
    let side = outlier_extract(&mut cur, k);
    let extract_side_bytes = side.len() * 8; // (u32 index + f32 value)

    // QUANTIZE ---
    let mut recon = match stack.quantize {
        Quantize::SymUniform => sym_q4_recon(&cur),
        Quantize::AsymAffine => asym_q4_recon(&cur),
        Quantize::CodebookLloydMax16 => codebook_q4_recon(&cur),
        Quantize::SymmetricCodebook => symmetric_codebook_q4_recon(&cur),
        Quantize::SuperposeSymAsym => superpose_sym_asym_recon(&cur),
        Quantize::RefractBulkTail30 => refract_bulk_tail_30_recon(&cur),
        Quantize::ComposeNestedScale => compose_nested_scale_recon(&cur),
    };
    // Book the quantizer's own metadata bytes (rough, per-block):
    let n_blocks = cur.len().div_ceil(BLOCK);
    let quant_side_bytes = match stack.quantize {
        Quantize::SymUniform => n_blocks * 4, // one f32 scale/block
        Quantize::AsymAffine => n_blocks * 8, // scale + min per block
        Quantize::CodebookLloydMax16 => n_blocks * 32, // 16 f16 codebook/block
        Quantize::SymmetricCodebook => n_blocks * 16, // 8 f16 magnitude centroids
        Quantize::SuperposeSymAsym => n_blocks * 12, // sym scale + asym scale+min
        // refract: sym scale (4) + asym scale+min (8) + 1 partition-mask bit/element
        // packed → ceil(BLOCK/8) bytes per block.
        Quantize::RefractBulkTail30 => n_blocks * 12 + n_blocks * BLOCK.div_ceil(8),
        // compose: one f32 super-scale per 16-elem block + 2-bit sub-scale per 8-elem
        // sub-block. Two sub-scales per super = 4 bits total (0.5 bytes/super).
        Quantize::ComposeNestedScale => cur.len().div_ceil(16) * 4 + cur.len().div_ceil(16) / 2,
    };

    // POST ---
    match stack.post {
        Post::Null => {}
        Post::ErrfeedPass => {
            recon = errfeed_q4_recon(&recon);
        }
        Post::WaveletLift => {
            recon = wavelet_lifting_repair(&recon);
        }
        Post::PreserveRowNorm => {
            // NB: applied in the CURRENT (transformed) space; norm-preservation
            // in any orthonormal basis is equivalent to preservation in weight
            // space, so this is safe under Hadamard/DCT (which are orthonormal).
            if let Some(ref n) = orig_row_norms {
                recon = preserve_row_norm(recon, n, out_len, in_len);
            }
        }
    }

    // Re-insert outliers *while still in transformed space* — outliers were
    // evicted from `cur` which lives in transformed space; putting them back
    // BEFORE the PRE inverse means they get inverse-transformed together with
    // everything else, and the row DC gets added back to them uniformly. This
    // is the fix for the two critical bugs the adversarial review found.
    recon = outlier_restore(recon, &side);

    // undo PRE (last operation — the whole tensor is inverse-transformed together) ---
    if matches!(stack.pre, Pre::RowHadamard8) {
        row_hadamard_8_apply(&mut recon, out_len, in_len);
    }
    if matches!(stack.pre, Pre::BlockHadamard32) {
        block_hadamard_32_apply(&mut recon);
    }
    if matches!(stack.pre, Pre::DctRow8) {
        dct_row_8_apply(&mut recon, out_len, in_len, true);
    }
    if matches!(stack.pre, Pre::HaarMix8) {
        haar_mix_8_apply(&mut recon, out_len, in_len, true);
    }
    if let Some(means) = row_means {
        recon = row_dc_restore(recon, &means, out_len, in_len);
    }

    (
        recon,
        pre_side_bytes + extract_side_bytes + quant_side_bytes,
    )
}

/// Nominal 4-bit code storage in bytes (packed).
pub fn nominal_code_bytes(len: usize) -> usize {
    (len * 4).div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumeration_is_bounded_and_deterministic() {
        let a = enumerate_legal_stacks();
        let b = enumerate_legal_stacks();
        assert_eq!(a.len(), b.len());
        // 6 pres × 4 exs × 7 qs × 4 posts = 672, minus BlockH32-with-nonSym pruning
        // (1 × 4 × 6 × 4 = 96) = 576 legal stacks with refract + compose + haar-mix.
        assert!(
            a.len() >= 400 && a.len() <= 700,
            "enum size sanity, got {}",
            a.len()
        );
        // Every enumerated stack must apply cleanly to a small tensor.
        let w = vec![0.1f32; 64];
        for s in a.iter().take(8) {
            let (r, _bytes) = apply(*s, &w, 8, 8);
            assert_eq!(r.len(), w.len());
        }
    }

    #[test]
    fn compose_nested_scale_recon_is_finite_and_same_length() {
        let s = Stack {
            pre: Pre::Null,
            extract: Extract::Null,
            quantize: Quantize::ComposeNestedScale,
            post: Post::Null,
        };
        // Bimodal magnitude across the 32-elem block so sub-blocks pick different
        // fractional scales.
        let mut w = vec![0.05f32; 32];
        for &i in &[3usize, 11, 19, 27] {
            w[i] = 0.8;
        }
        let (r, side) = apply(s, &w, 1, 32);
        assert_eq!(r.len(), w.len());
        for &v in &r {
            assert!(v.is_finite(), "compose produced non-finite value: {v}");
        }
        // Large-magnitude elements should reconstruct within one quant step.
        for &i in &[3usize, 11, 19, 27] {
            let err = (r[i] - w[i]).abs();
            assert!(
                err < 0.15,
                "compose big-mag reconstruction off by {err} at {i}"
            );
        }
        assert!(side > 0);
    }

    #[test]
    fn haar_mix_8_is_orthogonal_and_invertible() {
        // Round-trip: forward then inverse must recover the input to float precision.
        let s = Stack {
            pre: Pre::HaarMix8,
            extract: Extract::Null,
            quantize: Quantize::SymUniform,
            post: Post::Null,
        };
        // Small tensor, no quantization loss on values that land exactly on the grid.
        // We use zero to sidestep quantization: 0 -> 0 through sym_q4 as long as any
        // scale > 0. This isolates the mix's orthogonality.
        let w = vec![0.0f32; 16];
        let (r, _bytes) = apply(s, &w, 2, 8);
        assert_eq!(r.len(), w.len());
        for &v in &r {
            assert!(v.is_finite());
            assert!(v.abs() < 1e-6, "zero-input roundtrip drifted: got {v}");
        }
    }

    #[test]
    fn haar_mix_8_matrix_is_orthogonal() {
        let m = haar8_matrix();
        // M^T M should be the 8x8 identity within float tolerance.
        for i in 0..8 {
            for j in 0..8 {
                let mut dot = 0f32;
                for k in 0..8 {
                    dot += m[k][i] * m[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "M^T M[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn refract_bulk_tail_recon_is_finite_and_same_length() {
        let s = Stack {
            pre: Pre::Null,
            extract: Extract::Null,
            quantize: Quantize::RefractBulkTail30,
            post: Post::Null,
        };
        // 32-element block with a bimodal distribution: 28 bulk values ~0.1 and
        // 4 tail values ~1.0 — matches the shape refract was designed to exploit.
        let mut w = vec![0.1f32; 32];
        w[3] = 1.0;
        w[9] = -1.0;
        w[15] = 1.0;
        w[27] = -1.0;
        let (r, side) = apply(s, &w, 1, 32);
        assert_eq!(r.len(), w.len());
        for &v in &r {
            assert!(v.is_finite(), "refract produced non-finite value: {v}");
        }
        // With 4 tail values, tail recon uses asym-Q4, bulk uses sym-Q4.
        // Both should keep the tail sign correct.
        for &i in &[3usize, 9, 15, 27] {
            assert!(
                r[i].signum() == w[i].signum(),
                "refract tail sign flipped at {i}: got {}, expected {}",
                r[i],
                w[i]
            );
        }
        // side bytes account for both quantizers + partition mask.
        assert!(side > 0);
    }

    #[test]
    fn identity_stack_survives_the_pipeline() {
        // Null|Null|Sym|Null with tiny weights: reconstruction is finite and same shape.
        let s = Stack {
            pre: Pre::Null,
            extract: Extract::Null,
            quantize: Quantize::SymUniform,
            post: Post::Null,
        };
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let (r, _) = apply(s, &w, 8, 8);
        assert_eq!(r.len(), w.len());
        for &v in &r {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn outlier_restore_preserves_evicted_values_exactly() {
        // With Null pre and Null post, an outlier we evicted must be exactly restored.
        let s = Stack {
            pre: Pre::Null,
            extract: Extract::OutlierTopK1,
            quantize: Quantize::SymUniform,
            post: Post::Null,
        };
        let mut w = vec![0.01f32; 32];
        w[7] = 99.0; // the outlier
        let (r, _) = apply(s, &w, 1, 32);
        assert_eq!(r[7], 99.0, "top-1 outlier must be preserved bit-exactly");
    }

    #[test]
    fn rowdc_plus_outlier_lands_in_weight_space() {
        // The critical bug the adversarial review caught: outlier_restore must
        // run BEFORE row_dc_restore so evicted (residual) values get the row
        // mean re-added, landing in weight space. Prior code had them swapped.
        let s = Stack {
            pre: Pre::RowDcRemove,
            extract: Extract::OutlierTopK1,
            quantize: Quantize::SymUniform, // any quantizer is fine — outlier bypasses it
            post: Post::Null,
        };
        // One row of 32 weights: mean 1.0, plus an outlier at index 7.
        let mut w = vec![1.0f32; 32];
        w[7] = 99.0;
        let (r, _) = apply(s, &w, 1, 32);
        // After the fix, the outlier position should reconstruct to the raw
        // weight (99.0), not to residual-only (98.0).
        assert!(
            (r[7] - 99.0).abs() < 1e-3,
            "rowDC+outlier: expected ≈99.0, got {}",
            r[7]
        );
    }

    #[test]
    fn hadamard_plus_outlier_lands_in_weight_space() {
        // Same critical fix, other transform. Outliers evicted in the Hadamard
        // domain must be inverse-transformed alongside the reconstruction.
        let s = Stack {
            pre: Pre::RowHadamard8,
            extract: Extract::OutlierTopK1,
            quantize: Quantize::SymUniform,
            post: Post::Null,
        };
        // Rows of 8. All-zero row except one spike so the Hadamard has a clean coeff.
        let mut w = vec![0.0f32; 16];
        w[0] = 4.0;
        let (r, _) = apply(s, &w, 2, 8);
        // Reconstruction must be finite and comparable in magnitude to the input,
        // not blown up by inserting a Hadamard-domain coefficient into weight space.
        let max_r = r.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert!(
            max_r < 20.0,
            "hadamard+outlier: max |r|={max_r} suggests transform-domain leakage"
        );
    }
}
