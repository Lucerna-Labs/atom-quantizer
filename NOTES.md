# Atom Quantizer — historical engineering notes

> **Under development.** These notes preserve the v0.2.0 KL-only experiment, including the 10.66× result that later failed real inference. The dual-gate post-mortem in `THESIS.md` supersedes that claim; use `README.md` for current public-facing status.

**See [THESIS.md](THESIS.md) for the full engineering thesis.** These notes are the working dev doc; THESIS.md is the design/justification document for the whole process (problem, currency doctrine, picker design incl. bidirectional KL, empirical results, verification protocol, design lineage, limitations).


Dev notes for the codec: how it's built, why the pieces are shaped this way, how to reproduce, how to verify.

---

## 1. What Atom Quantizer is (v0.2)

An adaptive per-tensor Q2/Q4/Q8 weight codec whose strategy picker is driven by **KL divergence over softmax(W·x)** — not cosine. Three things distinguish it from a plain uniform quantizer:

1. **KL-driven adaptive picker.** For every 2-D linear weight tensor, sample 4 random Gaussian inputs, compute `ps_orig = softmax(W · x)` once, then walk Q2 → Q4 → Q8, encoding each candidate and computing the forward KL of its reconstruction against `ps_orig`. Keep the first candidate under the ceiling. For 1-D tensors (norms, biases), the fidelity currency is a 256-bin histogram KL between original and reconstructed weights. Protected tensors (embeddings, output projection, norms, biases) hold a stricter ceiling.
2. **Blind RF-repair on decode.** A short low-pass convolution is applied to the reconstructed weights on read. The repair strength is picked at compress time (when the original is available for comparison) and stored alongside the codes. On read, the decoder just applies the recorded repair.
3. **`mix_u32` integrity tag.** A cheap avalanche hash is computed over the packed codes at compress time and re-checked on read.

Two things Atom Quantizer deliberately does not do:

- It is **not** an inference engine. It's a codec: encode → decode, reversibly.
- It is **not** a discovery/experiment harness. Stack enumeration, cross-domain atoms, and KL-currency benchmarking of composed stacks live in the parent R&D project. Atom Quantizer is where verified codec stages ship.

## 2. Files

| File | Role | LOC |
|------|------|-----|
| `src/main.rs` | CLI: `<gguf>` runs the KL-driven codec; `compare <a> <b>` does a cosine head-to-head between two static GGUFs. | ~450 |
| `src/wq.rs`   | The codec itself — `encode_with`, `expand`, `CompressedTensor`, `mix_u32` integrity, blind repair, `tune_repair`. | ~600 |
| `src/oq.rs`   | Streaming `.oq` container writer + verify pass. Format: magic `OQ02` + per-tensor records with integrity re-check on read. | ~260 |
| `src/gguf.rs` | Minimal GGUF reader: header + metadata (skipped, alignment captured) + tensor table + dequant for F32/F16/BF16/Q8_0/Q4_K/Q6_K. | ~410 |
| `src/metrics.rs` | KL primitives: `kl_symmetric_output`, `kl_histogram_bidirectional`, `make_gaussian_inputs`, `cpu_matmul_batch`, `softmax_batch`, deterministic Box-Muller. | ~430 |
| `src/cuda_backend.rs` | Optional cuBLAS matvec (feature `cuda`). Softly falls back to CPU. | ~180 |
| `Cargo.toml`  | One optional dep: `cudarc` behind `cuda` feature. Standalone (no workspace inheritance). | ~20 |

Total codec ~2300 LOC of Rust.

## 3. How to build & run

```
# Recommended (needs CUDA 12.x driver + cuBLAS at runtime, dlopen'd via dynamic-loading):
cargo build --release --features cuda
./target/release/atom-quantizer <model.gguf>

# Or CPU-only (no cudarc, no cuBLAS):
cargo build --release
./target/release/atom-quantizer <model.gguf>
```

Release mode is important — the codec is CPU-bound and LTO'd release is ~5× faster than debug.

Options:

| Flag | Meaning |
|------|---------|
| `--limit N`   | Process only the first N dequantizable tensors. |
| `--key 0xHEX` | Seed for the `mix_u32` integrity tag + repair schedule. Deterministic. |
| `--out PATH`  | Also write to a real `.oq` file on disk (streamed, then verified on re-read). |
| `--asym`      | Force the affine (min+scale) quantizer instead of symmetric max-abs. |

## 4. How to reproduce Atom Quantizer from scratch

From the source tree in the parent R&D project:

```
PARENT_RD/src/wq.rs           → src/wq.rs
PARENT_RD/src/oq.rs           → src/oq.rs
PARENT_RD/src/gguf.rs         → src/gguf.rs           (must include BF16 dequant)
PARENT_RD/src/metrics.rs      → src/metrics.rs
PARENT_RD/src/cuda_backend.rs → src/cuda_backend.rs
```

Then write a `main.rs` that:

- Keeps: the default subcommand (KL-driven quantize, optionally `--out`) and `compare`.
- Drops: `discover`, `quantize-stack`, `bench`, `gpu-check` — these pull in `stacks.rs` (discovery kit) which is R&D-only.
- Ports over the `compress_kl_adaptive` helper from `quant-app/src/main.rs` (it lives above `is_protected`).

Cargo.toml has one optional dep — `cudarc` with `dynamic-loading` (so the build doesn't need the CUDA toolkit installed; libcuda is dlopen'd from the driver install at runtime). Add `#![allow(dead_code)]` at the top of `wq.rs`, `gguf.rs`, and `cuda_backend.rs` because the trimmed main uses a subset of the pub functions; some reference-material atoms (`asym_q4_recon`, `hadamard_q4_recon`, `matvec_batched`, etc.) are left in the source but not called by the CLI.

**Critical bug that must be fixed on the way over**: the original `gguf.rs` only listed F32/F16/Q8_0/Q4_K/Q6_K as dequantizable. Modern models (Qwen3.5, etc.) ship in **BF16** GGUFs. Without BF16 in the dequant table, the loader silently skips every big weight tensor and reports processing only ~40× fewer tensors than expected. The fix (already present in the copy here):

```rust
pub const GGML_BF16: u32 = 30;

pub fn type_name(t: u32) -> &'static str {
    /* ... */
    30 => "BF16",
    /* ... */
}

pub fn dequantizable(t: u32) -> bool {
    matches!(t, GGML_F32 | GGML_F16 | GGML_BF16 | GGML_Q8_0 | GGML_Q4_K | GGML_Q6_K)
}

// in block_layout():
GGML_BF16 => Some((1, 2)),

// in dequant_bytes():
GGML_BF16 => {
    // bf16 is the top 16 bits of f32 (sign + 8 exp + 7 mantissa).
    for c in buf.chunks_exact(2) {
        let bits = u32::from(u16::from_le_bytes([c[0], c[1]])) << 16;
        out.push(f32::from_bits(bits));
    }
}
```

**Critical design point that must be honored on the way over**: the strategy picker MUST live in `main.rs`, not inside `wq::compress()`. The historical `compress()` function in `wq.rs` was a cosine-driven picker — that was removed in v0.2. `wq.rs` now only exposes `encode_with(data, strategy, cfg)` (a single-strategy encoder). The picker orchestration (walk the tiers, compute KL for each candidate, pick the first that clears the ceiling) belongs in `main.rs` because it needs the shape info (out_len, in_len) and the matvec plumbing (GPU or CPU) — neither of which belongs in the codec module.

## 5. How to verify correctness

The codec has three internal correctness properties, all machine-checked on every run:

### 5a. Integrity round-trip (`c.verify()`)

Every compressed tensor stores a `mix_u32` hash over its packed codes. `verify()` recomputes and compares. The summary line at the bottom of a run says:

```
integrity: all codes verified ✓
```

If any tensor's codes were mutated (bit-flipped, truncated, replaced), the summary would say `FAILED ✗`.

### 5b. Writeback round-trip (`--out` then `oq::verify_file`)

When `--out model.oq` is given, the codec streams the compressed tensors to disk in the `OQ02` container format. Immediately after writing, `verify_file` opens the just-written file, re-reads every record, and re-runs the integrity check. Summary:

```
│ re-read verify      : all integrity tags OK ✓ (320 tensors)
│ decoded off disk    : token_embd.weight (BF16, 254279680 weights)
```

### 5c. KL ceiling accounting

Every tensor's KL_fwd is recorded during the picker walk. Tensors that couldn't clear the ceiling even at Q8 are counted and marked with `!` in the per-tensor row. The summary reports `! N tensor(s) exceeded the KL ceiling even at Q8`. On Qwen3.5-0.8B this hits 115/320 — all tiny 1-D norm/bias vectors where 256-bin histogram KL is inherently noisy on ≤1024-element inputs. Those tensors correctly land at Q8 (the safest candidate).

### 5d. Verified end-to-end run — Qwen3.5-0.8B bf16 GGUF

```
build:  RUSTFLAGS="-D warnings" cargo build --release --features cuda   → exit 0
build:  RUSTFLAGS="-D warnings" cargo build --release                    → exit 0
tests:  RUSTFLAGS="-D warnings" cargo test --release                     → 18 passed, 0 failed
run:    time ./target/release/atom-quantizer.exe <0.8B.gguf>            → 29.8 s wall clock

processed 320 tensors  |  strategy mix: Q2=205 Q4=0 Q8=115
dense f32 3009.6 MB  →  Atom Quantizer 282.2 MB   (10.66× smaller)
mean KL_fwd: 1.9683
worst-tensor KL_fwd: 17.0902  (blk.17.ssm_a — 16 elements)
integrity: all codes verified ✓
```

Compare to v0.1 (cosine picker on same code path, same model): 4.99× smaller, Q2=45 / Q4=268 / Q8=7, mean cosine 0.9959. The KL picker discovers that Q2 with blind repair is genuinely fine on all the big 2-D weights (measured KL 0.0000-0.0009), where the cosine bar was over-conservative and forced Q4.

## 6. What's deliberately not here

**Discovery/experimental atoms.** The `stacks.rs` module (which enumerates stack compositions — refract, compose, mix, tunnel, entangle, etc.) lives in the parent R&D project alongside a `discover` subcommand. That's the R&D harness. Atom Quantizer is the graduated codec — it applies items 1-10 in the README's "Atoms actually applied per tensor" table.

**GPU code for anything beyond KL matvec.** The `cuda_backend.rs` module only accelerates the softmax(W·x) matvec. The atoms themselves (quantize, dequantize, blind repair, tune) all run on CPU. This is intentional — the compute is cheap.

**K-quant re-quant.** If a GGUF ships as pure Q4_K or pure Q6_K, Atom Quantizer can dequant it to f32 and then Q2/Q4/Q8 it back down, but this typically doesn't help — the source already lost precision. Atom Quantizer is at its best on F16/BF16 sources.

## 7. Roadmap items from R&D

Discoveries in the parent `quant-app`'s discover harness (KL-ranked over 576 stacks × 20 tensors of Qwen3.5-0.8B) that could graduate into Atom Quantizer:

- **Refract-as-quant-strategy** (`Quantize::RefractBulkTail30` in `quant-app/src/stacks.rs`). Split each 32-block into bulk (`|w| < 0.30 · max_abs` → sym-Q4) and tail (`|w| ≥ 0.30 · max_abs` → asym-Q4). Discover-ranked #1 overall (KL 0.622 vs naked sym-Q4's KL 2.925). Requires the refract partition-mask travelling through the `CompressedTensor` metadata and the `.oq` format (magic bumps to `OQ03`).
- **rowDC pre-transform** (`Pre::RowDcRemove`). Subtract the per-row mean, quantize the residual, store one f16/row. Discover-ranked frequently in top-10 stacks. Requires one extra f16 per row in the `CompressedTensor` — `.oq` format bump.
- **top-K outlier extraction** (`Extract::OutlierTopK`). Pull the K largest |w| per block into a sidecar `(u32 index, f32 value)` list, quantize the bulk at a tighter scale. Aggressive fidelity gain at ~8 bytes/outlier cost.
- **Preserve-row-norm post-repair** (`Post::PreserveRowNorm`). On tensors that quantize to marginal fidelity, rescaling each row's L2 norm back to the original's post-decode cheaply restores KL headroom.
- **BF16 writeback scales**. `.oq` currently stores per-block scales as f32. Switching to f16 or bf16 cuts the container size ~50% at negligible fidelity cost. Bumps the magic to `OQ03`.

## 8. What went wrong on the way here, briefly

- Spent time chasing sim results in Kaggle for a discovery experiment when the codec was already shipped and the question was just "does the codec work on this model?" — real answer arrives faster from running the codec.
- v0.1 shipped with a cosine-driven picker that gave 4.99× on Qwen3.5-0.8B. The KL sweep revealed naked sym-Q4 sitting at KL rank 574/576 while cosine reported 0.9959 — cosine was hiding the damage. v0.2 replaces the picker's fidelity bar with KL and gets 10.66× on the same code path with an honest currency.
- The 133-vs-320-tensor puzzle on Qwen3.5-0.8B was the BF16 dequant gap in `gguf.rs`. Once BF16 was added, 320 tensors (the actual model) processed instead of 133 (norms/biases only).

## 9. Version log

- **0.2.0** — replaced the cosine-driven strategy picker with a KL-driven one. Added `metrics.rs` (KL primitives) and `cuda_backend.rs` (optional cuBLAS matvec). Cargo.toml gains an optional `cuda` feature. Verified on Qwen3.5-0.8B → **10.66× smaller** (up from 4.99×), integrity OK. 320 tensors, 29.8 s wall clock on RTX 5070 Ti.
- **0.1.0** — initial Atom Quantizer drop from the parent R&D project codec. Zero-dep pure-std, cosine-driven picker. BF16 dequant supported. Verified on Qwen3.5-0.8B → 4.99× smaller, cos 0.9959, integrity OK.
