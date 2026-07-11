# THESIS — Atom Quantizer

> **Development record, not a release specification.** This document intentionally preserves failed hypotheses and historical benchmark snapshots. Section 13 supersedes the earlier KL-only claims. The checked-in v0.2.3 source additionally wires `rowH8` and row-norm preservation; a fresh full-model v0.2.3 benchmark has not yet replaced the v0.2.2 baseline.

**Dual-gate adaptive Q2/Q4/Q8 weight quantization: bidirectional KL currency + cosine floor, blind RF-repair, per-block integrity for LLM GGUFs.**

Research record through version 0.2.3 · see the README for the current public status.

> **Read §13 first if you're evaluating the picker's decision function.** v0.2.0-0.2.1 shipped with a KL-only picker whose currency (softmax(W·x) over random Gaussian inputs) was empirically insufficient — it accepted Q2 aggression that broke real inference in LM Studio. §13 documents the failure mode and the v0.2.2 dual-gate fix. Sections §2-§5 have been amended to reflect the correction; the "cosine is a lying metric" framing has been substantially walked back.

---

## Abstract

Atom Quantizer is a per-tensor adaptive weight codec whose strategy picker (Q2 / Q4 / Q8, per block) is driven by the **maximum of forward and reverse KL divergence** over `softmax(W · x)` — an activation-based currency — rather than by weight cosine similarity. On Qwen3.5-0.8B (bf16 GGUF, 320 tensors, 0.752 B params) this yields **10.66× compression** (3009.6 MB → 282.2 MB) at bidirectional KL discipline, versus 4.99× for the same code path with a cosine picker. The cosine picker over-conservatively forced Q4 on tensors whose actual output-distribution shift under Q2 was measured at 0.0000-0.0009. The KL picker sees that clearly and accepts Q2 for 205 of the 320 tensors; the remaining 115 are all tiny 1-D norm/bias vectors (16-1024 elements) that fall back to Q8 as the safest candidate. Integrity is verified per tensor via a `mix_u32` avalanche hash over packed codes. Optional cuBLAS matvec accelerates the KL bar; a CPU fallback is functionally identical but slower.

---

## 1. Problem statement

Aggressive quantization of LLM weights is now table stakes for consumer-hardware inference. State-of-the-art shipping codecs (llama.cpp K-quants, DwarfStar's IQ2_XXS + Q2_K mix for DeepSeek V4, AWQ / GPTQ per-group / per-channel affine schemes, exllamav2 GPTQ) all pick a per-tensor bit budget and a within-tensor quantization scheme, then apply them. The differences between codecs are:

1. **Which tensors get quantized how aggressively.** (llama.cpp uses fixed per-tensor rules; AWQ/GPTQ use activation-aware calibration.)
2. **The within-block quantization primitive.** (Symmetric max-abs / asymmetric affine / codebook / Lloyd-Max.)
3. **Optional per-block or per-row corrections.** (Zero-point, per-row means, imatrix scale adjustment.)
4. **The fidelity currency that drives the acceptance decision.** (Almost always: reconstruction MSE or cosine.)

Atom Quantizer takes point (4) as the point where existing codecs are weakest: **weight cosine is a lying metric for output-distribution preservation.** A tensor's reconstruction can hit cosine 0.9965 while its induced output distribution — the thing the model actually consumes — has drifted so badly the tensor's KL divergence ranks near the bottom of a 576-stack sweep on the same substrate. The compression ratio a codec "chooses" is downstream of the currency the picker measures against. Fix the currency and the same picker chooses much more aggressive strategies with honest fidelity.

The specific target Atom Quantizer positions itself against is **ds4 / DwarfStar**, a DeepSeek V4 Flash inference engine that ships imatrix-tuned 2-bit quantizations (IQ2_XXS for up/gate MoE experts, Q2_K for down) and reports that "the 2-bit quants are not a joke: they behave well, work under coding agents, call tools in a reliable way." Atom Quantizer's contribution is orthogonal: at any given bit budget, use a better fidelity currency (bidirectional KL over softmax outputs) to make the picker's per-tensor bit-width call. The block primitive is the same block-symmetric max-abs quantizer as llama.cpp's Q2/Q4/Q8. What changes is which tensors get which strategy.

---

## 2. The KL currency — why forward and reverse

### 2.1 Why cosine is not the right currency

Weight cosine between an original tensor W and its reconstruction W′ is dominated by the largest-magnitude weights. Two failure modes:

- **Scale-invariance.** A tensor whose reconstruction is a scalar multiple of the original hits cosine 1.0 exactly, even if that scalar is wrong. Weight quantization rarely produces exact scalar multiples, but per-block max-abs quantization does produce *approximate* per-block scalar corrections that the cosine metric absorbs silently.
- **Large-weight dominance.** In a 4096-element weight vector where 20 weights carry 60% of the L2 norm, cosine is essentially a measurement of how well those 20 weights are preserved. The remaining 4076 can drift substantially without cosine noticing. On real-model activations, those 4076 weights carry meaningful signal through the residual stream — meaningful enough that their perturbation shows up in the output distribution even when cosine says the tensor is fine.

The empirical evidence for this from the discovery sweep in this project: on Qwen3.5-0.8B, naked block-symmetric Q4 (the primitive Atom Quantizer v0.1 used as its "safe middle" tier) landed at **KL rank 574 out of 576** in a 576-stack composition sweep. Its cosine on the same tensor sample was 0.9965. Cosine reported "very good"; KL reported "second-worst possible strategy at this bit budget." The cosine picker on this substrate was forcing Q4 on tensors whose actual output-distribution shift under Q2 (with blind repair) was measured at ≤ 0.001 forward KL. It was over-conservative by an unnecessarily large margin.

### 2.2 The KL currency, formalized

For a 2-D linear weight tensor `W ∈ ℝ^{out × in}` and its reconstruction `W′ ∈ ℝ^{out × in}`, define the reference and quantized output distributions as:

- Sample `k` random input vectors `x_i ~ N(0, I_in / √in)` — unit-variance activations, deterministic seed derived from the tensor index.
- Compute `y_orig_i = W · x_i` and `y_quant_i = W′ · x_i`.
- Define the softmax distributions with temperature τ = 1: `p_i = softmax(y_orig_i)`, `q_i = softmax(y_quant_i)`.

Then the two KL directions are:

- **Forward KL** (perplexity direction): `KL(p || q) = Σ_j p_j · log(p_j / q_j)` — averaged over `k` inputs.
- **Reverse KL** (mode-seeking direction): `KL(q || p) = Σ_j q_j · log(q_j / p_j)` — averaged over `k` inputs.

For 1-D tensors (norms, biases), there is no meaningful matvec-through-activations semantic, so Atom Quantizer falls back to a **256-bin histogram KL**: bin both distributions over their joint support `[−max(|orig|, |quant|), +max(|orig|, |quant|)]`, add a small ε to avoid log(0), and compute both directions on the histogram probabilities. This is a coarser, activation-free proxy; it works well on high-count 1-D tensors (1024+ elements) and is inherently noisy on very small ones (16-128 elements), which is why 115 of the 320 tensors on Qwen3.5-0.8B fall back to Q8 with a `!` marker — the histogram signal is too noisy at those sizes to prove Q2 or Q4 clears the bar.

### 2.3 Why bidirectional and not one direction

- **Forward KL alone** (KL(p || q), the perplexity direction) — a codec optimizing this is happy to drop a mode in the original as long as it does not invent a spurious one. Under sample-based estimation this means: the codec can collapse a low-probability but real feature of the original distribution and forward KL will barely notice. That's a subtle but real fidelity loss.
- **Reverse KL alone** (KL(q || p), the mode-seeking direction) — a codec optimizing this is happy to invent spurious mass anywhere as long as it does not lose a mode. The codec will happily introduce a broad "smear" of probability across the output support and reverse KL will be forgiving of it.
- **The safe metric is `max(fwd, rev)`.** Reject any strategy where EITHER direction exceeds the ceiling. A candidate that has both directions below the ceiling is safe against both failure modes.

An additional diagnostic emerges: the **fwd/rev asymmetry ratio** `fwd / rev` per tensor tells us what kind of distortion the quantizer is introducing. Symmetric distortion (ratio near 1.0) is the well-behaved case: the quantized distribution is a "flat" perturbation of the original. Strongly asymmetric ratios reveal specific failure modes worth investigating. On Qwen3.5-0.8B under the current picker, mean asymmetry is 1.018 across the 320 tensors — essentially symmetric, which is the expected behavior of a well-tuned block-uniform quantizer with blind repair.

### 2.4 The ceiling

The ceiling is the empirical threshold under which a candidate strategy is "acceptable." Two ceilings are baked in:

- **Normal tensors: `max(fwd, rev) ≤ 1.0`**. On the Qwen3.5-0.8B substrate, this rejects naked sym-Q4 (which measured at KL 2.9) but accepts Q2-with-blind-repair on every big 2-D linear tensor (measured at KL 0.0000-0.0009). The ceiling was chosen so that the naked sym-Q4 baseline (the pre-KL Atom Quantizer v0.1) would be rejected outright — a stricter bar than any codec that reports cosine.
- **Protected tensors: `max(fwd, rev) ≤ 0.25`**. Tensors where error compounds through the residual stream — embeddings (`token_embd`), output projection (`output.weight`, `lm_head`), norms, biases — hold this stricter ceiling. On Qwen3.5-0.8B this pushes 7 tensors (all `attn_output.weight` variants + `token_embd.weight`) from Q4 into Q8 territory when the picker measures them.

Both ceilings are runtime constants in `main::run_quantize`. Retuning them per-model or per-architecture is a straightforward extension (measure the discover-sweep top KL band for that substrate, set the ceiling just above it).

---

## 3. The picker

`main::compress_kl_adaptive` implements the algorithm:

```
for strategy in [Q2, Q4, Q8]:
    candidate = encode_with(data, strategy, cfg)
    recon = expand(candidate)
    if is_2d:
        y_quant = matvec(recon, xs, out_len, in_len)   # cuBLAS or CPU
        (fwd, rev) = bidirectional_kl_cached(ps_orig, softmax(y_quant))
    else:
        (fwd, rev) = kl_histogram_bidirectional(data, recon, 256_bins)
    if max(fwd, rev) <= ceiling:
        return (candidate, fwd, rev)     # accepted at this strategy
    last = (candidate, fwd, rev)
return last                              # nothing cleared; keep the safest (Q8) result
```

Key properties:

- **Deterministic.** The Gaussian input sampler is a seeded Box-Muller derived from the `mix_u32` avalanche hash. Same tensor index → same `x_i` → same `ps_orig` → same picker decision.
- **Reference reused.** `ps_orig` is computed once per tensor via one matvec of the original W against `xs`. All three candidate strategies reuse it. The dominant cost per tensor is the `k=4` matvecs (one for the reference, one per candidate strategy for the reconstructions).
- **cuBLAS optional.** With `--features cuda`, the matvecs route through `cuda_backend::CudaMatvec::matmul_batch` (cuBLAS SGEMM). Without the feature, `metrics::cpu_matmul_batch` handles them on CPU. The picker's decisions are bit-identical modulo IEEE-754 accumulation order in the matmul — in practice, on Qwen3.5-0.8B the strategy mix is identical between GPU and CPU builds on the same seed.
- **Fallback is Q8.** If Q2, Q4, and Q8 all exceed the ceiling, the codec keeps the Q8 candidate anyway and marks the tensor with `!` in the per-tensor report. The final summary reports the count of such misses. This is a real risk indicator — but on Qwen3.5-0.8B, every `!` tensor is a tiny 1-D vector (16-1024 elems) where the histogram KL is inherently noisy at 256 bins. The Q8 fallback is safe.

The picker walks Q2 → Q4 → Q8 (most-aggressive first). This ordering matters: on tensors where Q2 clears the ceiling, we save two candidate encodes + two matvecs + two KL computes per tensor. Q2 is the most likely acceptance on the big 2-D weights — the picker essentially never gets past Q2 for `ffn_*`, `attn_*_proj`, `token_embd`, etc.

---

## 4. Atoms actually applied per tensor

The codec applies exactly the following ten atoms per tensor. This is the complete list — there are no atoms silently in the loop; there is no dead wiring; there is no atom that "would be nice to add later."

| # | Atom | Source location | When applied |
|---|------|-----------------|--------------|
| 1 | **Block symmetric max-abs quantize** — per-block f32 scale, symmetric grid at 2/4/8 bits | `wq::quantize_codes` + `wq::dequantize_codes` | Every tensor (default) |
| 2 | **Block asymmetric affine quantize** — per-block `min` + `scale`, 2^bits unsigned levels | `wq::compress_with_asym` | Every tensor when `--asym` set |
| 3 | **Blind RF-repair** — 5-tap symmetric low-pass conv `[0.1, 0.2, 0.4, 0.2, 0.1]` (Σ = 1), blended into the degraded signal at a stored strength ∈ [0, 1] | `wq::blind_repair` | Every tensor on decode |
| 4 | **Cosine-tuned repair strength** — compress-time search over 8 candidate strengths, picks the strength maximizing cosine vs original. Uses cosine because the search is within a chosen strategy, not across strategies. | `wq::tune_repair` | Every tensor at encode |
| 5 | **`mix_u32` integrity tag** — Numerical Recipes-style avalanche hash over the packed codes; recomputed and compared on every read | `wq::bytes_tag` | Every tensor |
| 6 | **KL-driven strategy picker** — bidirectional-KL adaptive walk over Q2/Q4/Q8 with a max(fwd, rev) ≤ ceiling accept criterion | `main::compress_kl_adaptive` | Every tensor |
| 7 | **cuBLAS SGEMM matvec** — used to compute `y = W · x` for the softmax reference distribution and each candidate's reconstruction; falls back to CPU matmul if cuBLAS is not compiled in or fails to initialize | `cuda_backend::CudaMatvec::matmul_batch` (feature `cuda`) or `metrics::cpu_matmul_batch` | Every 2-D linear tensor's fidelity check |
| 8 | **Histogram KL** — 256-bin joint-support histogram, symmetric ε regularization, both directions | `metrics::kl_histogram_bidirectional` | Every 1-D tensor's fidelity check |
| 9 | **Gaussian input sampler** — deterministic Box-Muller from a `mix_u32` LCG-style state, per-tensor seed | `metrics::make_gaussian_inputs` + `metrics::box_muller` | Reference distribution generation per 2-D tensor |
| 10 | **Protected-tensor stricter ceiling** — `is_protected(name)` matches embeddings, output projection, norms, biases, and applies the 0.25 KL ceiling instead of 1.0 | `main::is_protected` | Every tensor whose name matches |

---

## 5. Empirical results

### 5.1 Qwen3.5-0.8B (bf16 GGUF, 320 tensors, 0.752 B params)

Verified end-to-end run:

```
=== build --features cuda -D warnings ===  build passed, warnings: 0
=== build (no cuda) -D warnings ===        build passed, warnings: 0
=== cargo test --release -D warnings ===   18 passed, 0 failed, warnings: 0
=== full end-to-end (RTX 5070 Ti + cuBLAS) ===

processed 320 tensors  |  strategy mix: Q2=205 Q4=0 Q8=115  (* = protected)
dense f32 3009.6 MB  →  Atom Quantizer 282.2 MB   (10.66× smaller)
mean KL_fwd: 1.9683   mean KL_rev: 1.6104   fwd/rev asymmetry: 1.018
currency: softmax(W·x) with k=4 Gaussian inputs for 2-D; 256-bin histogram for 1-D
worst-tensor max(KL_fwd, KL_rev): 17.0902  (blk.17.ssm_a)
! 115 tensor(s) exceeded the KL ceiling even at Q8 (kept at Q8 anyway; marked '!')
integrity: all codes verified ✓

wall clock: 29.8 s
```

Same code path with the pre-KL cosine picker (Atom Quantizer v0.1):

```
processed 320 tensors  |  strategy mix: Q2=45 Q4=268 Q8=7
dense f32 3009.6 MB  →  Atom Quantizer 603.7 MB   (4.99× smaller)
mean reconstruction cosine: 0.9959
integrity: all codes verified ✓
```

The v0.1 → v0.2 delta is entirely the currency change: cosine → max(fwd, rev) KL. Same wq encoder, same blind repair, same integrity, same GGUF loader. Same model.

### 5.2 Where the compression came from

Of the 320 tensors:

- **205 big 2-D linear tensors** land at Q2. Every `ffn_*` (down/gate/up, ~3.67 M elems each), every `attn_qkv` (~6.29 M), every `attn_gate` (~2.10 M), every `attn_q/k/v` (~0.52-4.19 M), every `ssm_out/alpha/beta` (~16 K-2.10 M), and `token_embd.weight` (~254 M — the biggest tensor, 34% of the model by parameters). Measured KL_fwd on these: 0.0000-0.0009 under Q2 with blind repair. The KL currency confirms Q2 preserves the output distribution to floating-point noise on these substrates. The cosine picker was blind to this — cosine 0.994-0.995 tripped its "not quite good enough" heuristic and forced Q4.
- **115 tensors fall back to Q8** (all marked `!`). Every one is a tiny 1-D vector — norm weights (128-1024 elems), SSM state parameters (16 elems), biases (16 elems). At 16-1024 elements against 256 histogram bins, the KL signal is inherently noisy: 16 elements across 256 bins gives 0.06 elements per bin, which the ε-regularized log dominates. The Q8 fallback is appropriate — these tensors are individually small (a few KB each) and their aggregate contribution to model size is < 0.5 MB.

### 5.3 fwd/rev asymmetry as diagnostic

The 320-tensor sample yields mean asymmetry `fwd / rev = 1.018` — essentially symmetric distortion. This is the expected behavior of block-symmetric max-abs quantization with blind repair: the reconstruction is a "flat" perturbation of the original rather than a mode-biased one. Two tensor classes that show more asymmetry:

- The 1-D norm/bias tensors that hit the `!` fallback have `fwd = rev` exactly (histogram KL is symmetric by construction for these).
- The protected tensors that land at Q8 (`attn_output.weight` variants, `token_embd`) show slightly asymmetric ratios — `fwd/rev` ranging 0.5-2.0 — because Q8's per-block scale is fine-grained enough that the reconstruction sits in a slightly different "shape" than the original, not just a rescaled version.

Neither is a failure mode. Both are consistent with a well-behaved quantizer.

---

## 6. Reversibility, integrity, in-RAM by default

Every compressed tensor stores:

- The strategy (Q2 / Q4 / Q8).
- The packed codes (bit-packed unsigned integers).
- The per-block scales (`Vec<f32>`).
- The per-block affine mins (`Vec<f32>`, empty for symmetric).
- The blind-repair strength (`f32`).
- A `mix_u32` integrity tag over the packed codes.

`wq::expand(&c)` reconstructs the f32 weights from the compressed representation alone — no side channel, no oracle. The reconstruction is deterministic. On a call to `c.verify()`, the codec re-hashes the packed codes with the same `mix_u32` and compares to the stored tag. Any mutation of the packed bytes (bit-flip, truncation, replacement) makes verify fail.

The summary line `integrity: all codes verified ✓` reports whether every one of the 320 tensors passed its own integrity check. This is the reversibility guarantee: any tensor whose `verify()` passes will `expand()` to the same f32 vector every time. On write-to-`.oq`, `verify_file(path)` re-opens the file, re-reads every record, and re-runs the integrity check — proving the file is not a write-only dead end.

Without `--out`, the codec runs entirely in RAM. The `CompressedTensor` structs for every tensor are held in memory; nothing hits disk. The RAM footprint of the compressed state is the same as the reported "Atom Quantizer MB" summary — 282.2 MB for Qwen3.5-0.8B, versus the 3009.6 MB f32 reconstruction that would result from calling `expand()` on all of them at once.

---

## 7. Design lineage — from KV codec to weight codec

Atom Quantizer's core primitives (block-scoped max-abs quantization + blind low-pass RF-repair + `mix_u32` integrity) are directly ported from the PIE inference engine's KV-cache codec, where the same construction achieves 2-bit KV compression with per-token repair. Weights need higher fidelity than KV caches because errors compound through layers, so the tiering (Q2/Q4/Q8 candidates) is stricter and protected tensors get a stricter ceiling — but the atom stack is the same.

Two things that were **not** ported from the KV codec because they don't apply to weights:

- **Per-token scale rotation.** The KV codec rotates the per-block scale schedule across tokens to prevent temporal correlation of quantization error. Weights are static — no time axis, so this atom does not apply.
- **Blind crypto layer.** The KV codec optionally XORs the packed codes with a keystream so a stolen dump is unreadable. Weights are usually meant to be reproducible/comparable; the crypto layer is off by default in the weight codec. The infrastructure (the `key` field on `WqConfig`) is present but exposes only the integrity-tag seed.

Two things that are **added** for weights:

- **The adaptive picker.** KV blocks all use 2 bits; weights get an adaptive Q2/Q4/Q8 choice per tensor. The KL currency drives the choice.
- **The protected-tensor stricter ceiling.** No equivalent in the KV codec — every KV block is protected symmetrically.

---

## 8. Verification protocol

The codec exposes three orthogonal correctness checks that are all machine-verified on every run:

1. **Per-tensor integrity tag** (`CompressedTensor::verify()`). Re-hash the packed codes with `mix_u32` and compare to the stored tag. On the 320-tensor run this fires 320 times; the summary reports `all codes verified ✓` iff every one passed.

2. **Writeback round-trip** (`--out model.oq` followed by `oq::verify_file(path)`). Stream the compressed tensors to disk in the `OQ02` container format, then re-open the file, re-read every record, and re-run the integrity check. The `writeback` summary block reports the count of successfully verified tensors and shows the first decoded tensor's stats as proof-of-life.

3. **KL ceiling accounting.** Every tensor's `max(KL_fwd, KL_rev)` is compared against its ceiling (0.25 for protected, 1.0 for normal). Ceiling misses are counted and marked with `!` in the per-tensor rows. On Qwen3.5-0.8B this is 115/320 (all tiny 1-D vectors, all falling back to Q8 as the safest candidate). The summary reports the count so the miss rate is not hidden.

Additional external checks:

4. **`compare a.gguf b.gguf` subcommand.** Cosine of matched tensors between two GGUFs — used for an absolute fidelity number against a higher-precision reference. This is intentionally still cosine-based because the comparison is between two static GGUFs with no activation semantic. It is a diagnostic, not the picker's fidelity bar.

5. **Full unit-test suite.** 18 tests covering the codec's core round-trip (`round_trip_reconstructs_within_tolerance`), integrity tag (`integrity_tag_detects_corruption`), blind determinism (`decode_is_blind_and_deterministic`), Q2 bit-packing (`two_bit_is_really_packed`), empty-input safety (`empty_input_is_safe`), blind-repair monotonicity (`blind_repair_helps_or_holds`), `.oq` write-read round-trip (`oq_write_read_roundtrip`), GGUF Q4_K and Q6_K dequant against hand-computed reference, KL primitives (7 tests: identical distributions → 0, disjoint distributions → large, symmetric Jeffreys equality, asymmetric directional, output on identical weights → 0, output on perturbed weights → positive, histogram identical → 0), Gaussian sampler (seed sensitivity and determinism). All 18 must pass with `RUSTFLAGS="-D warnings"`.

---

## 9. Reproducibility

The full verification protocol, exactly as executed:

```bash
# Environment: Windows 11, RTX 5070 Ti (Blackwell), CUDA 12.9 driver + toolkit,
# Rust 1.90 stable, cargo release profile with LTO thin.

cd C:\projects\atom-quantizer

# 1. Build with cuBLAS (uses libcuda + libcublas at runtime via dynamic-loading)
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin:$PATH"
RUSTFLAGS="-D warnings" cargo build --release --features cuda

# 2. Build without cuBLAS (pure CPU path, no cudarc dep)
RUSTFLAGS="-D warnings" cargo build --release

# 3. Unit test suite with warnings denied
RUSTFLAGS="-D warnings" cargo test --release

# 4. End-to-end run on the target model
./target/release/atom-quantizer.exe C:/path/to/Qwen_Qwen3.5-0.8B-bf16.gguf

# 5. Writeback round-trip
./target/release/atom-quantizer.exe C:/path/to/model.gguf --out /tmp/out.oq
```

All five must produce exit code 0. Steps 1-3 must emit no warnings. Step 4 must end with `integrity: all codes verified ✓`. Step 5 must end with `re-read verify: all integrity tags OK ✓`.

---

## 10. Limitations

- **Histogram KL on very small vectors is noisy.** At 16 elements against 256 bins, the ε-regularized log dominates the signal. The 115 fall-back tensors on Qwen3.5-0.8B all sit in this regime. The fallback is Q8, which is safe but wastes bit budget on tensors whose actual sensitivity is unknown at this resolution. A finer bin count for small tensors (32 bins for ≤ 128-element inputs) is a natural extension.
- **Fixed KL ceilings.** The 0.25 / 1.0 ceilings are tuned against the Qwen3.5-0.8B substrate. For different architectures (deep dense, MoE-heavy, MLA-style attention), the discover-sweep top KL band shifts and the ceilings should be re-calibrated. A calibration subcommand that runs the discover sweep on a target model and reports the recommended ceiling would automate this.
- **Random-Gaussian activation proxy.** The reference distribution `ps_orig` is computed against `x ~ N(0, I/√in)`, not real inference activations. This is a well-known proxy in the quantization literature (AWQ, GPTQ, imatrix all address this). Real activations from a small calibration corpus would be more accurate — but they require running the actual model, which the codec deliberately does not do. Atom Quantizer is a codec, not an inference engine.
- **The discovered stack primitives from the R&D sibling are not applied.** `refract` (bulk sym + tail asym, discover-ranked #1 at KL 0.622), `rowDC` (per-row DC removal, discover-ranked in top 10), `top-K outlier extraction`, and `preserve-row-norm` post-repair all live in `PARENT_RD/src/stacks.rs` as tested code paths. Wiring them into Atom Quantizer's base quantizer would require a container format bump (`OQ02` → `OQ03`) to carry the per-tensor stack spec and side data (partition masks, per-row means, outlier sidecars). This is genuine additional engineering, not a placeholder.

---

## 11. What this document supersedes

- The design comments in `wq.rs::compress()` (the cosine-driven picker). That function was removed in v0.2. The picker lives in `main::compress_kl_adaptive` now.
- The v0.1 README's claim that Atom Quantizer was zero-dep pure-std. As of v0.2, the recommended build path adds one optional dependency: `cudarc` behind the `cuda` feature. The no-cuda build path remains pure-std.
- Any earlier framing of Atom Quantizer as an "alternative to ds4" via bit-rate matching alone. The claim is more precise: at the same bit budget, Atom Quantizer's KL-driven picker preserves output distribution better than the cosine-driven picker on the same block primitive. The block primitive itself (sym-Q4 with blind repair) is comparable to K-quant; the difference is in which tensors get which strategy.

---

## 12. Version log

- **0.2.3** — wired `rowH8` into the PRE stage and row-norm `preserve` into POST, then isolated Composer 2 in the `atom-quantizer-q2` workspace crate. The v0.2.2 dual gate remains mandatory. A fresh full-model benchmark is required before attaching new compression or inference claims to this pipeline.
- **0.2.2** — **dual gate: KL ceiling AND cosine floor, both must clear.** After v0.2.1's KL-only picker produced a passthrough GGUF that hallucinated in LM Studio, added a cosine ≥ 0.99 (0.995 protected) as a mandatory second gate on every strategy candidate. Q2 no longer clears on any big 2-D tensor (cos 0.81 fails the floor) even when KL 0.0001 passes. Strategy mix returns to Q2=0/Q4=198/Q8=122 on Qwen3.5-0.8B — matches v0.1 behavior. Compression 2.51× (source GGUF → .oq). **Passthrough GGUF loads in LM Studio and produces coherent output — verified manually.** See §13.
- **0.2.1** — bidirectional KL: picker used `max(fwd, rev)` as the acceptance metric. Reverse KL computed for every candidate (previously discarded via `_rev`). **Note (0.2.2 hindsight):** reverse KL rejected zero additional strategies on Qwen3.5-0.8B because the forward-KL bound was already the binding constraint; the real problem was that neither direction of a random-Gaussian-input KL was tight enough for real inference. See §13.
- **0.2.0** — replaced the cosine-driven strategy picker with a KL-driven one. Added `metrics.rs` (KL primitives) and `cuda_backend.rs` (optional cuBLAS matvec). Claimed 10.66× on Qwen3.5-0.8B. **Note (0.2.2 hindsight):** the passthrough GGUF built from this picker hallucinated in real LM Studio inference. The claimed compression was against a currency that did not correlate with real inference quality on this substrate. See §13.
- **0.1.0** — initial Atom Quantizer drop from the parent R&D project codec. Zero-dep pure-std, cosine-driven picker (0.975 normal / 0.995 protected). BF16 dequant supported. Verified on Qwen3.5-0.8B → 4.99× smaller, cos 0.9959, integrity OK. **Note (0.2.2 hindsight):** this was the empirically-safe pick. The KL-only detour did not improve on it in practice.

---

## 13. Post-mortem: the KL-only currency failure

**What happened.** v0.2.0-0.2.1 replaced the v0.1 cosine-driven strategy picker with a picker driven by KL divergence over `softmax(W · x)` where `x ~ N(0, I / √in)` with `k = 4` random Gaussian samples per tensor. On Qwen3.5-0.8B this metric reported KL_fwd = 0.0000-0.0009 on every big 2-D linear tensor at Q2, so the picker accepted Q2 on 205 of 320 tensors, giving a reported compression of 10.66× (versus v0.1's 4.99×). §5 celebrated this as a validation of the KL currency doctrine.

The passthrough GGUF produced by v0.2.1 was then loaded in LM Studio (real llama.cpp inference on the reconstructed weights). **The model hallucinated.** The claimed KL fidelity did not translate to coherent inference behavior.

**Why the KL currency was insufficient.** Two failure modes, both structural:

1. **Random Gaussian input is a nearly-uniform reference distribution for softmax.** For a 6144×2048 linear tensor with `x ~ N(0, I/√in)`, the output `y = W · x` has approximately i.i.d.-normal entries. `softmax(y)` over 6144 dimensions is nearly uniform — every entry is close to 1/6144 with small perturbation. In this regime, KL(orig || quant) is dominated by the small-perturbation regime and is insensitive to which weights got quantized: the reference distribution has no structure to disturb. A weight perturbation that would shift a specific attention pattern by 35° in weight space produces a tiny KL because the "attention pattern" doesn't exist in the random-Gaussian reference.
2. **Per-tensor KL misses cross-layer error accumulation.** Even if the per-tensor output distribution is preserved to within KL 0.0001 (which it wasn't, in the sense that matters), errors from 24 layers of transformer + SSM blocks compound. If each layer's reconstruction has a small systematic bias in a common eigen-direction, the compounded bias through 24 layers is not the sum of the individual biases — it's an exponential in the relevant subspace. Per-tensor KL cannot see this.

**How cosine caught what KL missed.** Weight cosine is not scale-invariant to structured drift. Q2 quantization on `ffn_down.weight` gave cosine 0.81 — a real 35° angle between original and reconstructed weight vectors. That angle is the primary evidence of the drift that compounds catastrophically through inference. Cosine as a per-tensor floor gates this drift directly. It is not a proxy for KL — it's an independent signal that catches a failure mode KL over random Gaussians is provably blind to.

**The v0.2.2 dual gate.** Both metrics must clear on every candidate strategy for a tier to be accepted. KL ceiling (max(fwd, rev) ≤ 1.0 normal / ≤ 0.25 protected) rejects tensors whose output distribution genuinely shifts; cosine floor (≥ 0.99 normal / ≥ 0.995 protected) rejects tensors whose weights drift too far in any direction the KL was blind to. In practice on Qwen3.5-0.8B, the cosine floor is the tighter constraint on every big 2-D tensor: Q2 clears KL but fails cosine (0.81 < 0.99), so the picker walks to Q4 which clears both (cos 0.994-0.995, KL essentially 0). Compression returns to 2.51× against source GGUF (603.7 MB .oq vs 1516.7 MB source), matching v0.1 behavior. **The model runs coherently in LM Studio.**

**What KL-with-random-Gaussian is still useful for.** As a diagnostic column and a secondary rejection criterion. Some future strategy could hit high weight cosine while genuinely rotating output distributions (e.g., an orthogonal Haar rotation before quantization). KL catches that; cosine doesn't. So both stay in the report and both are gates. But the "cosine is a lying metric" framing from v0.2.0's THESIS §2.1 was overreach. Cosine was the safety net that v0.2.0's KL-only picker removed and v0.2.2's dual gate restored.

**What would actually let us push past 2.51× reliably.** The literature consensus is imatrix-style calibration: run a small prompt corpus through the source model in bf16, capture real activation patterns at every layer's input, use THOSE (not random Gaussians) as `x` in the KL currency. Real activations have the structural asymmetry that reveals Q2 aggression as harmful before it hits inference. AWQ, GPTQ, and llama.cpp's imatrix use exactly this. Atom Quantizer does not, currently. Extending Atom Quantizer to accept an imatrix file and use it for KL reference computation is the natural path to actually pushing sub-3-bits-per-weight without breaking inference. That would justify walking the cosine floor back down. Until that lands, dual gate at cosine 0.99 is the safety belt.

**What this section supersedes.** THESIS v0.2.0-0.2.1 §2.1 argued "cosine is a lying metric" and framed KL as unambiguously superior. That claim was empirically wrong on this substrate. §2.1 remains in the document for historical context but is superseded by this section: cosine catches real weight drift; the KL currency Atom Quantizer uses is a weak proxy for real inference fidelity; both are needed.

---

## 14. The repair-and-fortification chain

**We are not done, not by a long shot.** The dual gate in §13 is the safety belt that keeps Atom Quantizer functional; it is not the full codec Atom Quantizer is aiming to be. What v0.2.2 ships occupies two stations of a five-station pipeline. The remaining three stations exist in the discovery kit (`stacks.rs` in the parent R&D project) as tested, benchmarked primitives — they are not part of the codec's per-tensor path at this version.

### 14.1 The five-station chain

```
PRE-transform   →   QUANT   →   POST-repair   →   OUTLIER preservation   →   VERIFY
```

Each station handles one class of quantization damage:

1. **PRE-transform** — rotate the tensor into a basis where quantization is friendlier (energy compaction, decorrelation, DC removal).
2. **QUANT** — the actual bit-width map. Atom Quantizer's Q2/Q4/Q8 block-symmetric max-abs (or asym-affine with `--asym`) sits here.
3. **POST-repair** — active fidelity correction after decode: restore what quant took away.
4. **OUTLIER preservation** — pull the values that matter most (largest-magnitude weights) out of the quant path entirely, keep them at higher precision.
5. **VERIFY** — invariant checks + integrity so the pipeline is provably reversible and any corruption is caught.

Where Atom Quantizer v0.2.2 currently sits:

| Station | Wired in Atom Quantizer v0.2.2 | Not wired (available in the kit) |
|---------|--------------------------|-----------------------------------|
| PRE-transform | — | `mix` (random Haar rotation), `transform` (DCT-II row basis), `rowDC` (per-row DC removal), `rowHadamard8`, `haarMix8` |
| QUANT | `SymUniform` Q2/Q4/Q8 (default), `AsymAffine` (via `--asym`) | `CodebookLloydMax16`, `SymmetricCodebook`, `SuperposeSymAsym`, `RefractBulkTail30`, `ComposeNestedScale` |
| POST-repair | Blind RF-repair (single 5-tap low-pass conv, cosine-tuned strength) | `preserve` (row-norm restore), `errfeed` (residual carry), `waveletLift` (multi-resolution), `phase_align` (row-mean removal) |
| OUTLIER preservation | — | `tunnel` / `OutlierTopK1/2/4` (top-K bit-exact sidecar) |
| VERIFY | `mix_u32` integrity tag per tensor | `verify` (invariant check), `audit` (hash-chained trail) |

### 14.2 The two primitive categories

The kit's primitives fall into two functionally distinct roles:

#### Cosine-repair primitives — active fidelity correction

These primitives directly improve reconstruction cosine against the original, at the same or slightly higher bit budget:

| Primitive | Category | How it helps cosine |
|-----------|----------|---------------------|
| `preserve` (row-norm) | POST | Restores per-row L2 norm — magnitudes are restored, direction was already preserved by quant. |
| `superpose` (sym+asym blend) | QUANT | Blending two reconstructions often improves cosine because errors partially cancel. |
| `refract` (bulk sym + tail asym) | QUANT | Asymmetric tail treatment can improve direction preservation on outliers. |
| `compose` (super-block scales) | QUANT | Hierarchical scaling can correct for direction drift at the block level. |
| `phase_align` (row-mean) | POST | Removes DC component — can improve direction by removing the bulk offset. |

#### Fortification primitives — robustness, not repair

These primitives make the model *robust* rather than *more accurately reconstructed*. They spread risk, catch errors, and reduce the blast radius of any single miss:

| Primitive | Category | What it fortifies against |
|-----------|----------|---------------------------|
| `mix` (random Haar rotation) | PRE | Spreads outliers across coefficients — model becomes robust to single-tensor corruption. |
| `transform` (DCT-II row basis) | PRE | Energy compaction in a different basis — quant errors become predictable. |
| `tunnel` (top-K bit-exact) | OUTLIER | Outliers preserved bit-exact — model is robust to the most damaging errors. |
| `errfeed` (error feedback) | POST | Iteratively quantize the residual — like error-correcting codes. |
| `waveletLift` (wavelet lifting) | POST | Multi-resolution representation — model is robust to errors at any scale. |
| `verify` (invariant check) | VERIFY | Catches when repair went wrong. |
| `audit` (hash-chained trail) | VERIFY | Logs what got repaired, so it can be re-verified. |

### 14.3 The composer pattern

A per-tensor pipeline that exploits the kit fully looks like:

```
PRE:      rowDC  or  rowDCT8  or  mix          (rotate to compress-friendly basis)
QUANT:    Q2/Q4/Q8  with refract  or  superpose  (the codec's bit-width map, potentially blended)
POST:     preserve  +  errfeed  or  waveletLift (blend/restore reconstructions, correct residuals)
OUTLIER:  tunnel  (top-K bit-exact)              (keep the values that matter most exact)
VERIFY:   mix_u32  +  verify  +  audit           (invariant checks + integrity + audit trail)
```

Each layer is a real primitive. Each exists in the kit as tested code (see the discover harness in `stacks.rs`). The dual-gate picker in v0.2.2 already knows how to reject a bad candidate; extending the picker to walk through composed stacks rather than only the Q2/Q4/Q8 bit-width tier is the mechanical piece that ties the composer to the codec.

### 14.4 What v0.2.3 currently applies vs the full chain

Comparing the codec's per-tensor path against the full chain:

| Station | v0.2.3 delivers | Fraction of the chain wired |
|---------|-----------------|------------------------------|
| PRE-transform | `rowH8` on eligible 2-D tensors | 1/5 |
| QUANT | 2 primitives (sym-Q4/Q2/Q8, asym-affine via `--asym`) | 2/7 |
| POST-repair | 2 primitives (blind RF-repair and row-norm `preserve`) | 2/4 |
| OUTLIER preservation | none | 0/1 |
| VERIFY | 1 primitive (`mix_u32` integrity tag) | 1/3 |

**Total: 6 of 20 kit primitives are represented in the current path.** Eligibility and CLI choices mean not every primitive runs on every tensor. The dual gate remains the safety boundary; v0.2.3 needs a new end-to-end benchmark before these additions can be called an improvement.

### 14.5 The rationale for capturing this here

Two reasons to fix this in the THESIS rather than treat it as informal context:

1. **The dual gate at cosine 0.99 is not the destination.** §13 established that KL-only was insufficient; the dual gate restored safety. But the ceiling `cosine ≥ 0.99` is what forces the picker off Q2 on every big 2-D tensor and caps compression at 2.51×. Wiring in the composer's repair primitives (preserve, superpose, refract, compose) is the direct path to letting Q2 clear the cosine floor at aggressive bit budgets — because those primitives specifically raise reconstruction cosine at fixed bits.
2. **The R&D discovery already ranked the primitives.** The parent R&D project's discover sweep on Qwen3.5-0.8B ranked `rowDC/top4/refract30/-` at KL 0.622 (rank #1 of 576), versus naked sym-Q4 at KL 2.9 (rank 574). Even under the KL currency that §13 established as an imperfect proxy, the composed stacks are three-to-four-times more information-preserving than the naked stations. §14.3's composer pattern is the concrete path to exploit that ranking inside Atom Quantizer.

---
