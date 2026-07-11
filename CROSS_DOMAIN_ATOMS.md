# Cross-domain atoms in Atom Quantizer

Atom Quantizer is interesting because its architecture does not treat quantization as one isolated numerical trick. It turns mechanisms from other disciplines into bounded atoms, composes them, and judges the composed result against explicit fidelity gates.

> **Status boundary:** “Current path” means the atom is wired into the v0.2.3 adaptive codec path. “Discovery kit” means real source exists and can be exercised by stack search or the companion composer, but it is not part of every normal encode.

## Current v0.2.3 path

| Source domain | Atom | Borrowed invariant or behavior | Role in Atom Quantizer | Source surface |
|---|---|---|---|---|
| Fast signal transforms | Row-wise Walsh–Hadamard (`rowH8`) | Orthonormal, self-inverse rotation that preserves norm while redistributing concentrated energy | Spreads outliers before block quantization, then returns the reconstruction to weight space | `src/main.rs`, `src/stacks.rs` |
| Digital signal processing | Blind finite receptive-field repair | Magnitude-aware low-pass reconstruction over a five-tap neighborhood | Reduces local decode noise without requiring the original weights at decode time | `src/wq.rs` |
| Linear algebra / vector geometry | Preserve row norm | L2 norm is restored independently for each eligible row | Repairs magnitude after inverse transform and decode | `src/main.rs`, `src/stacks.rs` |
| Information theory | Bidirectional KL gate | Compare reference and reconstructed output distributions in both directions | Rejects a candidate when sampled distribution drift exceeds the ceiling | `src/main.rs`, `src/metrics.rs` |
| Vector geometry | Cosine floor | Angular similarity catches structured weight drift that the Gaussian-input KL proxy missed | Mandatory second acceptance gate after the KL-only failure | `src/main.rs`, `src/wq.rs` |
| Hashing / data integrity | `mix_u32` avalanche tag | Deterministic bit diffusion makes packed-code corruption detectable | Verifies every compressed tensor and every streamed `OQ02` record | `src/wq.rs`, `src/oq.rs` |

The current composition is:

```text
FAST TRANSFORMS        QUANTIZATION       DSP REPAIR        GEOMETRY
rowH8              →   Q2 / Q4 / Q8   →   blind RF      →   preserve L2
                                                                  ↓
DATA INTEGRITY      ←                INFORMATION THEORY + VECTOR GEOMETRY
mix_u32 verify                         KL ceiling + cosine floor
```

## Discovery and composer atoms

| Source domain | Atom | What crosses the boundary | Current status |
|---|---|---|---|
| Telephony / audio coding | μ-law companding | Compress the amplitude domain before a uniform code grid so levels become denser near zero | Discovery implementation in `src/wq.rs` |
| Audio, control, and perceptual coding | Error feedback / noise shaping | Carry each quantization residual into the next target, analogous to sigma-delta and error-diffusion families | Discovery stack in `src/wq.rs` and `src/stacks.rs` |
| Image and signal compression | Row DCT-8 | Rotate short row windows into an energy-compacting frequency basis before quantization | Stack-search and Composer 2 candidate |
| Multiresolution signal analysis | Haar mixing and wavelet lifting | Orthogonal mixing plus approximation/detail-band repair | Stack-search and Composer 2 candidate |
| Classical scalar quantization | Lloyd–Max codebook | Learn non-uniform reconstruction centroids instead of assuming an evenly spaced grid | Discovery quantizer in `src/stacks.rs` |
| Sparse representation | Top-K outlier extraction | Remove the rare high-magnitude tail from the lossy path and preserve it in a sidecar | Discovery extract stage in `src/stacks.rs` |
| Optics-inspired routing metaphor | `refract` bulk/tail split | Route the dense bulk and sparse tail through different quantizers, then stitch original positions back together | Stack-search and Composer 2 candidate |
| Ensemble error cancellation | `superpose` | Blend symmetric and asymmetric reconstructions so partially independent errors can cancel | Discovery quantizer in `src/stacks.rs` |

“Optics-inspired” describes the routing abstraction; the implementation is a measurable bulk/tail partition, not a claim that it simulates physical optics. Similarly, the integrity tag is an avalanche check, not a cryptographic authentication primitive.

## The architectural rule

A cross-domain analogy becomes an Atom Quantizer atom only when it has:

1. a narrow input/output contract;
2. a reversible or explicitly lossy boundary;
3. a measurable storage and fidelity cost;
4. a legal place in the PRE → EXTRACT → QUANTIZE → POST → VERIFY chain; and
5. evidence that the composed stack beats or clarifies a baseline.

This keeps cross-domain thinking from becoming decorative metaphor. An idea earns its place by surviving composition and measurement.
