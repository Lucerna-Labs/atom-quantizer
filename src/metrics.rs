#![allow(clippy::needless_range_loop)]
//! metrics — KL divergence as the primary discovery currency.
//!
//! Cosine measures angular similarity of the weights themselves. KL measures the
//! divergence of the **output distribution** the weights induce — the standard
//! proxy for real-world quantization quality. This module ships:
//!   * `kl_symmetric_output(w_orig, w_quant, [out, in], k, seed)` — for 2-D linear
//!     weight tensors: push k random unit-Gaussian inputs through both matrices,
//!     softmax the outputs, report ½(KL(p‖q)+KL(q‖p)) averaged over inputs. This
//!     is the currency.
//!   * `kl_histogram(orig, quant, bins)` — a fast, activation-free fallback for
//!     1-D tensors (norms, biases) that don't have an activation semantic. Bins
//!     both distributions over their joint support and returns symmetric KL.
//!
//! Zero-dependency, pure std. The Gaussian sampler is a deterministic Box-Muller
//! seeded from `mix_u32` so runs are reproducible bit-for-bit.

use crate::wq::mix_u32;

/// Deterministic uniform (0,1] from a 32-bit avalanche state (excludes 0 to keep
/// log-taking Box-Muller safe).
fn next_uniform(state: &mut u32) -> f32 {
    *state = mix_u32(*state).max(1);
    (*state as f32) / (u32::MAX as f32)
}

/// Deterministic standard-normal sample via Box-Muller. Returns two independent
/// samples per call — the caller should pair reads.
fn box_muller(state: &mut u32) -> (f32, f32) {
    let u1 = next_uniform(state).max(1e-9);
    let u2 = next_uniform(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let t = std::f32::consts::TAU * u2;
    (r * t.cos(), r * t.sin())
}

/// Fill a slice with unit-variance Gaussian noise (deterministic seed).
fn gaussian_fill(out: &mut [f32], seed: u32) {
    let mut s = seed.max(1);
    let mut i = 0;
    while i + 1 < out.len() {
        let (a, b) = box_muller(&mut s);
        out[i] = a;
        out[i + 1] = b;
        i += 2;
    }
    if i < out.len() {
        out[i] = box_muller(&mut s).0;
    }
}

/// Row-major matvec y = W · x   (W is [out, in], x is [in], y is [out]).
fn matvec(w: &[f32], x: &[f32], out_len: usize, in_len: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_len];
    for r in 0..out_len {
        let base = r * in_len;
        let mut acc = 0.0f64;
        for c in 0..in_len {
            acc += (w[base + c] as f64) * (x[c] as f64);
        }
        y[r] = acc as f32;
    }
    y
}

/// Numerically stable softmax (max-subtraction) into a probability vector.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f64;
    let mut out: Vec<f32> = logits
        .iter()
        .map(|&z| {
            let e = ((z - m) as f64).exp();
            sum += e;
            e as f32
        })
        .collect();
    let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 };
    for p in out.iter_mut() {
        *p = (*p as f64 * inv) as f32;
    }
    out
}

/// KL(p‖q) = Σ p_i · log(p_i / q_i), with ε-smoothing for numerical safety.
fn kl(p: &[f32], q: &[f32]) -> f64 {
    let eps = 1e-12f64;
    let mut d = 0.0f64;
    let n = p.len().min(q.len());
    for i in 0..n {
        let pi = (p[i] as f64).max(eps);
        let qi = (q[i] as f64).max(eps);
        d += pi * (pi / qi).ln();
    }
    d.max(0.0)
}

/// Directional KL(p‖q) — the perplexity direction (reference on the left).
pub fn kl_directional(p: &[f32], q: &[f32]) -> f64 {
    kl(p, q)
}

/// Jeffreys divergence: ½·(KL(p‖q)+KL(q‖p)). Symmetric but unbounded — NOT
/// Jensen-Shannon (which uses a mixture and is bounded). Kept as an optional
/// diagnostic; the discovery primary is `kl_directional` because that's what
/// perplexity / cross-entropy loss integrate.
#[allow(dead_code)]
pub fn kl_jeffreys(p: &[f32], q: &[f32]) -> f64 {
    0.5 * (kl(p, q) + kl(q, p))
}

/// Both directions of output KL in a single pass — the interesting quantity for
/// discovery. Returns `(fwd, rev)` = `(KL(orig ‖ quant), KL(quant ‖ orig))`.
/// The asymmetry `fwd / rev` diagnoses the failure MODE:
///   fwd ≫ rev  — quant SMEARED probability mass away from orig's peak (mode loss);
///   fwd ≪ rev  — quant HALLUCINATED a peak orig didn't put mass on (spurious mode);
///   fwd ≈ rev  — symmetric distortion (rare in real quant).
/// Perplexity integrates `fwd`.
#[allow(dead_code)]
pub fn kl_bidirectional_output(
    w_orig: &[f32],
    w_quant: &[f32],
    out_len: usize,
    in_len: usize,
    k: usize,
    seed: u32,
) -> Option<(f64, f64)> {
    if out_len == 0 || in_len == 0 || out_len < 2 {
        return None;
    }
    if w_orig.len() != out_len * in_len || w_quant.len() != out_len * in_len {
        return None;
    }
    let mut x = vec![0.0f32; in_len];
    let inv_sqrt_in = 1.0 / (in_len as f32).sqrt();
    let (mut acc_fwd, mut acc_rev) = (0.0f64, 0.0f64);
    for step in 0..k {
        let s = seed.wrapping_add(step as u32).max(1);
        gaussian_fill(&mut x, s);
        for v in x.iter_mut() {
            *v *= inv_sqrt_in;
        }
        let y_orig = matvec(w_orig, &x, out_len, in_len);
        let y_quant = matvec(w_quant, &x, out_len, in_len);
        let p = softmax(&y_orig);
        let q = softmax(&y_quant);
        acc_fwd += kl_directional(&p, &q);
        acc_rev += kl_directional(&q, &p);
    }
    let n = k.max(1) as f64;
    Some((acc_fwd / n, acc_rev / n))
}

/// **Directional output KL** — `KL(softmax(W·x) ‖ softmax(W'·x))`, the perplexity
/// direction, averaged over `k` deterministic Gaussian inputs. Reference is on
/// the left (original weights).
///
/// Honest caveats (from adversarial review, kept in-source):
/// (1) `softmax(W·x)` has clean semantics only for a classifier/lm_head. For
///     hidden-state tensors it manufactures a distribution whose entropy
///     depends on `out_len` — use as a **per-tensor sensitivity proxy**, not a
///     drop-in perplexity number.
/// (2) Isotropic unit-Gaussian inputs miss the **outlier-feature regime** that
///     dominates real LLM quantization failure (LLM.int8/SmoothQuant/AWQ). A
///     stack that specifically handles outlier channels may show "no gain" here.
/// (3) Per-tensor KLs on different `out_len` are on different scales; averaging
///     across the tensor set favors the largest-out_len tensor. Report also
///     per-tensor and consider normalizing by `ln(out_len)`.
///
/// Returns `None` if the shape doesn't match.
#[allow(dead_code)]
pub fn kl_directional_output(
    w_orig: &[f32],
    w_quant: &[f32],
    out_len: usize,
    in_len: usize,
    k: usize,
    seed: u32,
) -> Option<f64> {
    if out_len == 0 || in_len == 0 || out_len < 2 {
        return None;
    }
    if w_orig.len() != out_len * in_len || w_quant.len() != out_len * in_len {
        return None;
    }
    let mut x = vec![0.0f32; in_len];
    let inv_sqrt_in = 1.0 / (in_len as f32).sqrt();
    let mut acc = 0.0f64;
    for step in 0..k {
        // Seed already mixes stack + tensor identity at the call site so different
        // tensors get independent Gaussian samples.
        let s = seed.wrapping_add(step as u32).max(1);
        gaussian_fill(&mut x, s);
        for v in x.iter_mut() {
            *v *= inv_sqrt_in;
        }
        let y_orig = matvec(w_orig, &x, out_len, in_len);
        let y_quant = matvec(w_quant, &x, out_len, in_len);
        let p = softmax(&y_orig);
        let q = softmax(&y_quant);
        acc += kl_directional(&p, &q);
    }
    Some(acc / k.max(1) as f64)
}

/// **Histogram KL** (directional, orig→quant) for 1-D tensors with no activation
/// semantic. Bins joint support `[min, max]` and returns KL(p_orig ‖ p_quant).
#[allow(dead_code)]
pub fn kl_histogram(orig: &[f32], quant: &[f32], bins: usize) -> f64 {
    if orig.is_empty() || quant.is_empty() || bins < 2 {
        return 0.0;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in orig.iter().chain(quant.iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return 0.0;
    }
    let width = hi - lo;
    let mut p = vec![0.0f32; bins];
    let mut q = vec![0.0f32; bins];
    for &v in orig {
        let b = (((v - lo) / width) * bins as f32).floor() as isize;
        p[b.clamp(0, bins as isize - 1) as usize] += 1.0;
    }
    for &v in quant {
        let b = (((v - lo) / width) * bins as f32).floor() as isize;
        q[b.clamp(0, bins as isize - 1) as usize] += 1.0;
    }
    let sp: f32 = p.iter().sum();
    let sq: f32 = q.iter().sum();
    if sp <= 0.0 || sq <= 0.0 {
        return 0.0;
    }
    for x in p.iter_mut() {
        *x /= sp;
    }
    for x in q.iter_mut() {
        *x /= sq;
    }
    kl_directional(&p, &q)
}

/// Deterministic Gaussian inputs (unit-norm activation vectors, scaled 1/sqrt(in)).
/// Exposed so the discover harness can compute them ONCE per tensor and reuse
/// across all stacks — the caller was previously recomputing them and W_orig.
pub fn make_gaussian_inputs(in_len: usize, k: usize, seed: u32) -> Vec<Vec<f32>> {
    let inv_sqrt_in = 1.0 / (in_len as f32).sqrt();
    (0..k)
        .map(|step| {
            let s = seed.wrapping_add(step as u32).max(1);
            let mut x = vec![0.0f32; in_len];
            gaussian_fill(&mut x, s);
            for v in x.iter_mut() {
                *v *= inv_sqrt_in;
            }
            x
        })
        .collect()
}

/// CPU matvec batch — matches the shape the GPU path returns so callers can be
/// backend-agnostic. Used when the CUDA backend isn't available.
pub fn cpu_matmul_batch(
    w: &[f32],
    xs: &[Vec<f32>],
    out_len: usize,
    in_len: usize,
) -> Vec<Vec<f32>> {
    xs.iter().map(|x| matvec(w, x, out_len, in_len)).collect()
}

/// Bidirectional KL over PRE-COMPUTED y-batches. This is where the reuse pays:
/// callers softmax each batch and feed both here.
#[allow(dead_code)]
pub fn bidirectional_kl_from_ys(ys_orig: &[Vec<f32>], ys_quant: &[Vec<f32>]) -> (f64, f64) {
    let k = ys_orig.len().min(ys_quant.len()).max(1);
    let (mut fwd, mut rev) = (0.0f64, 0.0f64);
    for i in 0..k {
        let p = softmax(&ys_orig[i]);
        let q = softmax(&ys_quant[i]);
        fwd += kl_directional(&p, &q);
        rev += kl_directional(&q, &p);
    }
    (fwd / k as f64, rev / k as f64)
}

/// Precomputed softmax of the reference (`p_orig`) so we don't re-softmax per stack.
pub fn softmax_batch(ys: &[Vec<f32>]) -> Vec<Vec<f32>> {
    ys.iter().map(|y| softmax(y)).collect()
}

/// Bidirectional KL when the reference softmax is already cached.
pub fn bidirectional_kl_cached(ps_orig: &[Vec<f32>], ys_quant: &[Vec<f32>]) -> (f64, f64) {
    let k = ps_orig.len().min(ys_quant.len()).max(1);
    let (mut fwd, mut rev) = (0.0f64, 0.0f64);
    for i in 0..k {
        let q = softmax(&ys_quant[i]);
        fwd += kl_directional(&ps_orig[i], &q);
        rev += kl_directional(&q, &ps_orig[i]);
    }
    (fwd / k as f64, rev / k as f64)
}

/// Bidirectional histogram KL — same binning, both directions.
pub fn kl_histogram_bidirectional(orig: &[f32], quant: &[f32], bins: usize) -> (f64, f64) {
    if orig.is_empty() || quant.is_empty() || bins < 2 {
        return (0.0, 0.0);
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in orig.iter().chain(quant.iter()) {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return (0.0, 0.0);
    }
    let width = hi - lo;
    let mut p = vec![0.0f32; bins];
    let mut q = vec![0.0f32; bins];
    for &v in orig {
        let b = (((v - lo) / width) * bins as f32).floor() as isize;
        p[b.clamp(0, bins as isize - 1) as usize] += 1.0;
    }
    for &v in quant {
        let b = (((v - lo) / width) * bins as f32).floor() as isize;
        q[b.clamp(0, bins as isize - 1) as usize] += 1.0;
    }
    let sp: f32 = p.iter().sum();
    let sq: f32 = q.iter().sum();
    if sp <= 0.0 || sq <= 0.0 {
        return (0.0, 0.0);
    }
    for x in p.iter_mut() {
        *x /= sp;
    }
    for x in q.iter_mut() {
        *x /= sq;
    }
    (kl_directional(&p, &q), kl_directional(&q, &p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_identical_distributions_is_zero() {
        let p = [0.2, 0.3, 0.5];
        assert!(kl_directional(&p, &p) < 1e-9);
        assert!(kl_jeffreys(&p, &p) < 1e-9);
    }

    #[test]
    fn kl_disjoint_distributions_is_large() {
        let p = [1.0, 0.0, 0.0];
        let q = [0.0, 0.0, 1.0];
        assert!(kl_jeffreys(&p, &q) > 5.0);
        assert!(kl_directional(&p, &q) > 3.0);
    }

    #[test]
    fn kl_jeffreys_is_symmetric() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.6, 0.3, 0.1];
        let a = kl_jeffreys(&p, &q);
        let b = kl_jeffreys(&q, &p);
        assert!((a - b).abs() < 1e-9);
    }

    #[test]
    fn kl_directional_asymmetric() {
        // Directional KL must be asymmetric when p and q have DIFFERENT shapes
        // (not just rotated). Sharp p vs broad q → KL(p||q) small, KL(q||p) large.
        let p = [0.9, 0.05, 0.05]; // sharp
        let q = [0.33, 0.34, 0.33]; // uniform
        let ab = kl_directional(&p, &q);
        let ba = kl_directional(&q, &p);
        assert!(
            (ab - ba).abs() > 0.1,
            "directional KL must be asymmetric here: ab={ab}, ba={ba}"
        );
    }

    #[test]
    fn kl_output_identical_weights_is_zero() {
        let out_len = 4;
        let in_len = 8;
        let mut w = vec![0f32; out_len * in_len];
        gaussian_fill(&mut w, 42);
        let k = kl_directional_output(&w, &w, out_len, in_len, 4, 7).unwrap();
        assert!(k < 1e-6, "identical weights must give KL≈0, got {k}");
    }

    #[test]
    fn kl_output_perturbed_weights_is_positive() {
        let out_len = 8;
        let in_len = 16;
        let mut w = vec![0f32; out_len * in_len];
        gaussian_fill(&mut w, 1);
        let mut w2 = w.clone();
        let mut jitter = vec![0f32; w.len()];
        gaussian_fill(&mut jitter, 999);
        for i in 0..w.len() {
            w2[i] += 0.5 * jitter[i];
        }
        let k = kl_directional_output(&w, &w2, out_len, in_len, 4, 7).unwrap();
        assert!(k > 1e-3, "perturbed weights must give KL>0, got {k}");
    }

    #[test]
    fn gaussian_fill_is_deterministic() {
        let mut a = vec![0f32; 100];
        let mut b = vec![0f32; 100];
        gaussian_fill(&mut a, 12345);
        gaussian_fill(&mut b, 12345);
        assert_eq!(a, b);
    }

    #[test]
    fn gaussian_fill_is_seed_sensitive() {
        let mut a = vec![0f32; 100];
        let mut b = vec![0f32; 100];
        gaussian_fill(&mut a, 1);
        gaussian_fill(&mut b, 2);
        assert!(a != b);
    }

    #[test]
    fn kl_histogram_identical_is_zero() {
        let v = vec![0.1, -0.2, 0.05, 0.3];
        assert!(kl_histogram(&v, &v, 16) < 1e-9);
    }
}
