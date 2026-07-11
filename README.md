# Atom Quantizer

> [!WARNING]
> **Under active development.** Atom Quantizer is a research codec, not a production release. Its CLI, thresholds, container format, and performance can change. Keep the original model and validate any generated artifact before use.

Atom Quantizer is an atom-composed adaptive Q2/Q4/Q8 weight codec for GGUF language models. It evaluates each tensor, walks from the most aggressive candidate toward safer bit widths, and accepts a result only when both its KL ceiling and cosine floor pass.

The current v0.2.3 path composes:

```text
rowH8 pre-transform → Q2/Q4/Q8 block quantization → blind RF repair
→ row-norm preservation → KL + cosine dual gate → integrity verification
```

This is an Atom-lineage project: small, measurable transformation primitives are composed into a codec, then rejected or promoted by explicit fidelity gates.

## Current status

- F32, F16, BF16, Q8_0, Q4_K, and Q6_K GGUF tensors can be read.
- Symmetric max-abs quantization is the default; affine quantization is available with `--asym`.
- Two-dimensional tensors use bidirectional KL over sampled `softmax(W · x)` outputs and a cosine safety floor.
- One-dimensional tensors use bidirectional histogram KL and the same cosine safety principle.
- Optional CUDA support routes the sampled matrix multiplication through cuBLAS. Quantization and repair remain CPU work.
- `.oq` artifacts use the existing `OQ02` format and are verified on read. The rename does not change the file magic.
- The separate `atom-quantizer-q2` crate explores more aggressive Composer 2 stacks without enlarging the main codec crate.

## Evidence, with the important caveat

The last inference-validated baseline recorded in [THESIS.md](THESIS.md) is the v0.2.2 dual-gate run on Qwen3.5-0.8B:

| Measurement | Recorded result |
|---|---:|
| Strategy mix | Q2=0 / Q4=198 / Q8=122 |
| `.oq` size | 603.7 MB |
| Source GGUF size | 1516.7 MB |
| Compression vs source GGUF | 2.51× |
| Runtime validation | Coherent output in LM Studio |

That is an R&D snapshot, not a release guarantee. The older 10.66× result in the historical notes came from a KL-only picker that later failed real inference. v0.2.2 restored the cosine gate. v0.2.3 then added rowH8 and row-norm preservation; its full model-level benchmark must be rerun before a current performance claim is published.

## Build and run

CPU-only:

```powershell
cargo build --release --workspace
.\target\release\atom-quantizer.exe C:\path\to\model.gguf
```

CUDA-assisted fidelity checks:

```powershell
cargo build --release -p atom-quantizer --features cuda
.\target\release\atom-quantizer.exe C:\path\to\model.gguf
```

Composer 2 research binary:

```powershell
cargo run --release -p atom-quantizer-q2 -- C:\path\to\model.gguf
```

Main CLI options:

| Flag | Meaning |
|---|---|
| `--limit N` | Process the first `N` dequantizable tensors. |
| `--key 0xHEX` | Set the deterministic integrity and repair seed. |
| `--out PATH` | Stream a verified `.oq` container to disk. |
| `--asym` | Use affine min/scale blocks instead of symmetric max-abs blocks. |

Compare two compatible GGUFs:

```powershell
.\target\release\atom-quantizer.exe compare C:\path\to\a.gguf C:\path\to\b.gguf
```

## Architecture

| Surface | Role |
|---|---|
| `src/wq.rs` | Q2/Q4/Q8 encoding, packed codes, blind repair, integrity tags. |
| `src/gguf.rs` | GGUF parsing and supported tensor dequantization. |
| `src/metrics.rs` | Cosine, sampled output KL, histogram KL, and CPU matmul. |
| `src/stacks.rs` | Composable transform, quantization, repair, and verification atoms. |
| `src/oq.rs` | Streaming `OQ02` writer and verified reader. |
| `src/cuda_backend.rs` | Optional cuBLAS matrix multiplication for fidelity sampling. |
| `src/main.rs` | Main adaptive CLI and stack-search surfaces. |
| `crates/atom-quantizer-q2` | Isolated Composer 2 research executable. |

## What still has to happen

- Add real-activation or imatrix calibration instead of relying only on Gaussian proxy inputs.
- Re-run a full v0.2.3 model benchmark and downstream task/perplexity validation.
- Stabilize the CLI and `.oq` format before declaring compatibility.
- Extend CUDA regression coverage across supported cards and driver versions.
- Decide and publish a project license before release.

See [DEVELOPMENT.md](DEVELOPMENT.md) for readiness rules and [THESIS.md](THESIS.md) for the full, intentionally preserved research trail.

## License

No license has been selected yet. Public source visibility does not grant reuse rights; licensing will be finalized before a release.
