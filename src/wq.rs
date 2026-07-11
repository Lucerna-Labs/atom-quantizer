#![allow(dead_code)]
//! wq — adaptive weight-quantization codec for LLM tensors.
//!
//! This is the **KV-cache codec technique applied to model weights**: severe
//! (down-to-2-bit) per-block quantization, a *blind* low-pass reconstruction that
//! knocks down high-frequency quantization noise on read ("RF signal repair"),
//! and a `mix_u32` integrity tag over the packed codes. The adaptive policy runs
//! at compress time (original weights available): it picks the most aggressive
//! bit-width whose *blind* reconstruction still clears a fidelity bar, and tunes
//! the repair strength that is then applied blindly on read.
//!
//! Weights need higher fidelity than a KV cache (error compounds through layers),
//! so the default bar is stricter, and "protected" tensors (embeddings, the output
//! projection, norms/biases) hold an even higher bar.
//!
//! Zero dependencies — pure std, same doctrine primitives as the PIE engine.

// ── doctrine primitives (mirror sim-atoms::hash + sim-cross-domain::convolve_1d) ──

/// Integer avalanche mix (mirrors `sim_atoms::atoms::hash::mix_u32`).
pub fn mix_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Same-length, center-aligned, zero-padded 1-D convolution.
fn convolve_1d(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    let (n, k) = (signal.len(), kernel.len());
    if k == 0 || n == 0 {
        return vec![0.0; n];
    }
    let half = (k / 2) as isize;
    let mut out = vec![0.0f32; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &kv) in kernel.iter().enumerate() {
            let si = i as isize + j as isize - half;
            if si >= 0 && (si as usize) < n {
                sum += signal[si as usize] * kv;
            }
        }
        *slot = sum;
    }
    out
}

/// Quantization strategies (bit-width tiers).
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantStrategy {
    Q2, // 2-bit
    Q4, // 4-bit
    Q8, // 8-bit
}

/// Block size (32-wide quant blocks, matching the engine).
pub const BLOCK: usize = 32;

/// Adaptive candidates, most-aggressive → safest (the "compression floor" search).
/// Exposed so the KL-driven picker in `main.rs` walks the same tiering as before.
pub const CANDIDATES: [QuantStrategy; 3] =
    [QuantStrategy::Q2, QuantStrategy::Q4, QuantStrategy::Q8];

fn levels(s: QuantStrategy) -> i32 {
    match s {
        QuantStrategy::Q2 => 1, // codes ∈ {-1,0,1}
        QuantStrategy::Q4 => 7,
        QuantStrategy::Q8 => 127,
    }
}

/// Packed code bit-width for a strategy.
pub fn bits(s: QuantStrategy) -> u32 {
    match s {
        QuantStrategy::Q2 => 2,
        QuantStrategy::Q4 => 4,
        QuantStrategy::Q8 => 8,
    }
}

pub fn strategy_name(s: QuantStrategy) -> &'static str {
    match s {
        QuantStrategy::Q2 => "Q2",
        QuantStrategy::Q4 => "Q4",
        QuantStrategy::Q8 => "Q8",
    }
}

/// Compress-time policy.
///
/// `protect` is retained so callers can flag protected tensors (embeddings,
/// output projection, norms/biases) — the KL-driven picker in `main.rs` reads
/// this field to select a stricter KL ceiling for those tensors.
#[derive(Clone, Copy, Debug)]
pub struct WqConfig {
    /// Protected tensors (embeddings / output projection / norms) — flags the
    /// caller that a stricter KL ceiling should apply.
    #[allow(dead_code)]
    pub protect: bool,
    /// Key for the integrity tag.
    pub key: u64,
    /// Use the asymmetric affine tier (per-block min + scale) instead of the
    /// symmetric max-abs one — the validated +cosine atom, at +4 bytes/block.
    pub asym: bool,
}

impl Default for WqConfig {
    fn default() -> Self {
        Self {
            protect: false,
            key: 0xADA7_79B9_5EED_C0DE,
            asym: false,
        }
    }
}

/// A severely-compressed weight tensor. `expand` reconstructs from this alone.
#[derive(Clone, Debug)]
pub struct CompressedTensor {
    pub strategy: QuantStrategy,
    pub len: usize,
    pub scales: Vec<f32>, // one per BLOCK-wide block
    pub mins: Vec<f32>,   // affine offsets, one per block; EMPTY ⇒ symmetric
    pub packed: Vec<u8>,  // bit-packed unsigned codes
    pub repair_strength: f32,
    pub integrity: u64,
    pub key: u64,
}

impl CompressedTensor {
    /// Bytes actually stored (codes + block scales + affine mins + small header).
    pub fn compressed_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4 + self.mins.len() * 4 + 32
    }
    /// Compression ratio vs dense f32.
    pub fn ratio_vs_f32(&self) -> f32 {
        if self.len == 0 {
            return 1.0;
        }
        (self.len * 4) as f32 / self.compressed_bytes().max(1) as f32
    }
    /// Integrity check over the packed codes.
    pub fn verify(&self) -> bool {
        bytes_tag(&self.packed, self.key) == self.integrity
    }
}

fn bytes_tag(packed: &[u8], key: u64) -> u64 {
    let mut h = (key as u32) ^ 0x9E37_79B9;
    for &b in packed {
        h = mix_u32(h ^ b as u32);
    }
    ((h as u64) << 32) | (mix_u32(h) as u64)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Accumulate in f64: billion-element weight tensors overflow f32 precision
    // and would otherwise yield impossible cosines > 1.0.
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return if na == 0.0 && nb == 0.0 { 1.0 } else { 0.0 };
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn quantize_codes(data: &[f32], s: QuantStrategy) -> (Vec<f32>, Vec<u32>) {
    let l = levels(s);
    let lf = l as f32;
    let mut scales = Vec::with_capacity(data.len().div_ceil(BLOCK));
    let mut codes = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let max_abs = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        scales.push(scale);
        for &w in blk {
            let q = (w / scale * lf).round().clamp(-lf, lf) as i32;
            codes.push((q + l) as u32); // signed [-l, l] → unsigned [0, 2l]
        }
    }
    (scales, codes)
}

fn dequantize_codes(scales: &[f32], codes: &[u32], s: QuantStrategy) -> Vec<f32> {
    let l = levels(s);
    let lf = l as f32;
    let mut out = Vec::with_capacity(codes.len());
    for (bi, blk) in codes.chunks(BLOCK).enumerate() {
        let scale = scales.get(bi).copied().unwrap_or(1.0);
        for &c in blk {
            out.push((c as i32 - l) as f32 / lf * scale);
        }
    }
    out
}

fn pack(codes: &[u32], bits: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((codes.len() * bits as usize).div_ceil(8));
    let mask = (1u32 << bits) - 1;
    let (mut acc, mut nbits) = (0u32, 0u32);
    for &c in codes {
        acc |= (c & mask) << nbits;
        nbits += bits;
        while nbits >= 8 {
            out.push((acc & 0xFF) as u8);
            acc >>= 8;
            nbits -= 8;
        }
    }
    if nbits > 0 {
        out.push((acc & 0xFF) as u8);
    }
    out
}

fn unpack(packed: &[u8], bits: u32, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let mask = (1u32 << bits) - 1;
    let (mut acc, mut nbits, mut bi) = (0u32, 0u32, 0usize);
    for _ in 0..count {
        while nbits < bits {
            let byte = packed.get(bi).copied().unwrap_or(0);
            bi += 1;
            acc |= (byte as u32) << nbits;
            nbits += 8;
        }
        out.push(acc & mask);
        acc >>= bits;
        nbits -= bits;
    }
    out
}

/// Blind RF signal-repair: a magnitude-preserving low-pass (kernel Σ = 1) blended
/// into the degraded signal to knock down high-frequency quantization noise. Uses
/// only the degraded data — no original at read time.
fn blind_repair(degraded: &[f32], strength: f32) -> Vec<f32> {
    if strength <= 0.0 || degraded.len() < 5 {
        return degraded.to_vec();
    }
    let kernel = [0.1f32, 0.2, 0.4, 0.2, 0.1]; // Σ = 1.0
    let smooth = convolve_1d(degraded, &kernel);
    degraded
        .iter()
        .zip(smooth.iter())
        .map(|(&d, &sm)| d + strength * (sm - d))
        .collect()
}

/// Pick the repair strength that maximizes fidelity (original available at compress).
fn tune_repair(orig: &[f32], degraded: &[f32]) -> f32 {
    let mut best_s = 0.0f32;
    let mut best_c = cosine(orig, degraded);
    for k in 1..=8 {
        let s = k as f32 / 8.0;
        let c = cosine(orig, &blind_repair(degraded, s));
        if c > best_c {
            best_c = c;
            best_s = s;
        }
    }
    best_s
}

fn empty_tensor(strategy: QuantStrategy, key: u64) -> CompressedTensor {
    CompressedTensor {
        strategy,
        len: 0,
        scales: vec![],
        mins: vec![],
        packed: vec![],
        repair_strength: 0.0,
        integrity: bytes_tag(&[], key),
        key,
    }
}

/// Encode with an explicit strategy, **symmetric** (max-abs) — the original path.
pub fn compress_with(data: &[f32], strategy: QuantStrategy, key: u64) -> CompressedTensor {
    if data.is_empty() {
        return empty_tensor(strategy, key);
    }
    let (scales, codes) = quantize_codes(data, strategy);
    let packed = pack(&codes, bits(strategy));
    let degraded = dequantize_codes(&scales, &codes, strategy);
    let repair_strength = tune_repair(data, &degraded);
    CompressedTensor {
        strategy,
        len: data.len(),
        integrity: bytes_tag(&packed, key),
        scales,
        mins: vec![],
        packed,
        repair_strength,
        key,
    }
}

/// Max unsigned code for a strategy's bit-width (affine uses the full 2^bits grid).
fn max_code(s: QuantStrategy) -> f32 {
    ((1u32 << bits(s)) - 1) as f32
}

/// Encode with an explicit strategy, **asymmetric affine** (per-block min+scale) —
/// the validated +cosine atom. Costs one extra f32 (the min) per block. No repair
/// (blind low-pass was measured inert for weights).
pub fn compress_with_asym(data: &[f32], strategy: QuantStrategy, key: u64) -> CompressedTensor {
    if data.is_empty() {
        return empty_tensor(strategy, key);
    }
    let mc = max_code(strategy);
    let mut scales = Vec::with_capacity(data.len().div_ceil(BLOCK));
    let mut mins = Vec::with_capacity(data.len().div_ceil(BLOCK));
    let mut codes = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &w in blk {
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let scale = if hi > lo { (hi - lo) / mc } else { 1.0 };
        scales.push(scale);
        mins.push(lo);
        for &w in blk {
            codes.push(((w - lo) / scale).round().clamp(0.0, mc) as u32);
        }
    }
    let packed = pack(&codes, bits(strategy));
    CompressedTensor {
        strategy,
        len: data.len(),
        integrity: bytes_tag(&packed, key),
        scales,
        mins,
        packed,
        repair_strength: 0.0,
        key,
    }
}

fn encode(data: &[f32], s: QuantStrategy, cfg: &WqConfig) -> CompressedTensor {
    if cfg.asym {
        compress_with_asym(data, s, cfg.key)
    } else {
        compress_with(data, s, cfg.key)
    }
}

/// Encode with an explicit strategy honoring the config's asym flag. Public so
/// the KL-driven picker in `main.rs` can try each candidate strategy.
pub fn encode_with(data: &[f32], strategy: QuantStrategy, cfg: &WqConfig) -> CompressedTensor {
    encode(data, strategy, cfg)
}

/// Reference **asymmetric** 4-bit quant (affine: per-block `min` + `scale`, 16
/// levels) — the core idea behind llama.cpp's Q4_K. Returns the in-memory
/// reconstruction only; nothing is written to disk. Used to benchmark our
/// symmetric codec against the asymmetric technique on identical weights.
pub fn asym_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &w in blk {
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let scale = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
        for &w in blk {
            let q = ((w - lo) / scale).round().clamp(0.0, 15.0);
            out.push(lo + q * scale);
        }
    }
    out
}

/// Plain symmetric per-block Q4 reconstruction (max-abs scale, no repair) — the
/// bare baseline, to isolate what the blind repair and other atoms actually buy.
pub fn sym_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let s = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        for &w in blk {
            let q = (w / s * 7.0).round().clamp(-7.0, 7.0);
            out.push(q / 7.0 * s);
        }
    }
    out
}

/// Non-uniform (μ-law **companding**) Q4 — a compression-theory atom applied to
/// quantization: compress before the uniform grid so levels are denser near zero,
/// matching the peaked (Gaussian-ish) weight distribution. Currency: cosine at Q4.
pub fn nonuniform_q4_recon(data: &[f32]) -> Vec<f32> {
    let mu = 15.0f32;
    let ln1mu = (1.0 + mu).ln();
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let s = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        for &w in blk {
            let x = (w / s).clamp(-1.0, 1.0);
            let c = x.signum() * (1.0 + mu * x.abs()).ln() / ln1mu; // compress → [-1,1]
            let q = (c * 7.0).round().clamp(-7.0, 7.0) / 7.0; // 4-bit uniform
            let d = q.signum() * ((1.0 + mu).powf(q.abs()) - 1.0) / mu; // expand
            out.push(d * s);
        }
    }
    out
}

/// Error-feedback / **noise-shaping** Q4 (sigma-delta / Floyd-Steinberg / GPTQ in
/// 1-D): carry each weight's quantization residual into the next target. A
/// perceptual-coding / audio atom bridged into quantization. Currency: cosine at Q4.
pub fn errfeed_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let s = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        let mut err = 0.0f32;
        for &w in blk {
            let target = w + err;
            let q = (target / s * 7.0).round().clamp(-7.0, 7.0);
            let recon = q / 7.0 * s;
            err = target - recon; // propagate the residual forward
            out.push(recon);
        }
    }
    out
}

/// Asymmetric affine Q4 **with** error-feedback — do the two winning atoms stack?
pub fn asym_errfeed_q4_recon(data: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &w in blk {
            lo = lo.min(w);
            hi = hi.max(w);
        }
        let scale = if hi > lo { (hi - lo) / 15.0 } else { 1.0 };
        let mut err = 0.0f32;
        for &w in blk {
            let target = w + err;
            let q = ((target - lo) / scale).round().clamp(0.0, 15.0);
            let recon = lo + q * scale;
            err = target - recon;
            out.push(recon);
        }
    }
    out
}

/// In-place orthonormal fast Walsh–Hadamard transform on a power-of-two block
/// (±1 butterfly, no multiplies; normalized so it is its own inverse).
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

/// **Hadamard-rotated** Q4 (transforms → quantization glue, à la QuaRot/QuIP):
/// rotate each 32-block into an incoherent basis (spreads a lone outlier across
/// all lanes so the max-abs scale isn't hostage to it), quantize there, rotate
/// back. Norm-preserving ⇒ cosine is an honest A/B. `affine` picks the tier.
pub fn hadamard_q4_recon(data: &[f32], affine: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for blk in data.chunks(BLOCK) {
        if blk.len() == BLOCK {
            let mut b = blk.to_vec();
            fwht_orthonormal(&mut b); // into the rotated basis
            let mut r = if affine {
                asym_q4_recon(&b)
            } else {
                sym_q4_recon(&b)
            };
            fwht_orthonormal(&mut r); // orthonormal ⇒ inverse == forward
            out.extend_from_slice(&r);
        } else {
            out.extend_from_slice(&sym_q4_recon(blk)); // partial tail: no rotation
        }
    }
    out
}

/// Blind decode: reconstruct from the compressed struct alone. Uses the affine
/// (min+scale) path when `mins` is present, else the symmetric (+repair) path.
pub fn expand(c: &CompressedTensor) -> Vec<f32> {
    if c.len == 0 {
        return vec![];
    }
    let codes = unpack(&c.packed, bits(c.strategy), c.len);
    if !c.mins.is_empty() {
        // asymmetric affine: w = min + code * scale
        let mut out = Vec::with_capacity(c.len);
        for (bi, blk) in codes.chunks(BLOCK).enumerate() {
            let scale = c.scales.get(bi).copied().unwrap_or(1.0);
            let lo = c.mins.get(bi).copied().unwrap_or(0.0);
            for &code in blk {
                out.push(lo + code as f32 * scale);
            }
        }
        return out;
    }
    let degraded = dequantize_codes(&c.scales, &codes, c.strategy);
    blind_repair(&degraded, c.repair_strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trained-weight-like data: near-Gaussian with a few outliers per block.
    fn weight_like(n: usize, seed: u32) -> Vec<f32> {
        let mut h = seed;
        (0..n)
            .map(|i| {
                h = mix_u32(h);
                let u1 = (h as f32 / u32::MAX as f32).max(1e-6);
                h = mix_u32(h);
                let u2 = h as f32 / u32::MAX as f32;
                let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                let outlier = if i % 128 == 0 { 4.0 } else { 1.0 };
                g * 0.02 * outlier
            })
            .collect()
    }

    #[test]
    fn round_trip_reconstructs_within_tolerance() {
        let data = weight_like(4096, 1);
        // Post-KL-picker: the picker lives in main.rs; the codec only encodes
        // at an explicit strategy. Q4 is the historical "safe middle" tier.
        let c = encode_with(&data, QuantStrategy::Q4, &WqConfig::default());
        let recon = expand(&c);
        println!(
            "weights round-trip: {} cos={:.4} ratio={:.1}x repair={:.2}",
            strategy_name(c.strategy),
            cosine(&data, &recon),
            c.ratio_vs_f32(),
            c.repair_strength
        );
        assert_eq!(recon.len(), data.len());
        assert!(cosine(&data, &recon) >= 0.9);
        assert!(c.ratio_vs_f32() > 3.0);
    }

    #[test]
    fn two_bit_is_really_packed() {
        let data = weight_like(2048, 2);
        let c = compress_with(&data, QuantStrategy::Q2, 1);
        assert_eq!(c.packed.len(), (data.len() * 2).div_ceil(8));
        assert!(c.ratio_vs_f32() > 8.0);
    }

    #[test]
    fn decode_is_blind_and_deterministic() {
        let data = weight_like(512, 3);
        let c = compress_with(&data, QuantStrategy::Q4, 7);
        assert_eq!(expand(&c), expand(&c));
        assert_eq!(expand(&c).len(), data.len());
    }

    #[test]
    fn blind_repair_helps_or_holds() {
        let data = weight_like(1024, 4);
        let c = compress_with(&data, QuantStrategy::Q2, 3);
        let codes = unpack(&c.packed, bits(c.strategy), c.len);
        let raw = dequantize_codes(&c.scales, &codes, c.strategy);
        assert!(cosine(&data, &expand(&c)) >= cosine(&data, &raw) - 1e-4);
    }

    // The `protect_never_more_aggressive` test used to validate that the
    // cosine-driven picker in wq::compress() upgraded protected tensors to a
    // less aggressive strategy. The picker has moved to main.rs (KL-driven),
    // so that invariant is no longer part of the wq codec's contract. Removed.

    #[test]
    fn integrity_tag_detects_corruption() {
        let data = weight_like(1024, 6);
        let mut c = compress_with(&data, QuantStrategy::Q2, 0xDEAD);
        assert!(c.verify());
        c.packed[10] ^= 0x01;
        assert!(!c.verify());
    }

    #[test]
    fn empty_input_is_safe() {
        let c = encode_with(&[], QuantStrategy::Q4, &WqConfig::default());
        assert_eq!(expand(&c), Vec::<f32>::new());
        assert!(c.verify());
    }
}
