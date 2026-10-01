# Atom Quantizer

> [!WARNING]
> **Under active development.** Atom Quantizer is a research codec, not a production release. Its CLI, thresholds, container format, and performance can change. Keep the original model and validate any generated artifact before use.

Atom Quantizer is an atom-composed adaptive Q2/Q4/Q8 weight codec for GGUF language models. It measures the reconstruction produced by its serialized decoder, keeps mandatory KL/cosine gates, and optionally adds real-activation output measurements and a whole-model byte budget. Tensors that cannot pass through a lossy candidate use an exact F32 escape.

Version 0.2.4 implements four foundations from the [first-principles study](research/first-principles-2026-09-05/REPORT.md): complete decoding, activation calibration, optimized Q2 scales, and global byte allocation. See [usage and boundaries](docs/FOUNDATIONS.md), [format details](docs/OQ03.md), and [the real-model benchmark](docs/FOUNDATION_BENCHMARK.md).

## The unusual idea: cross-domain atoms

The compelling part is not simply that Atom Quantizer can choose Q2, Q4, or Q8. It is that the codec is assembled from **small mechanisms borrowed across domains that rarely meet inside one weight format**:

- Walsh–Hadamard transforms from fast signal transforms redistribute outliers before quantization.
- A finite receptive-field low-pass repair borrows from digital signal processing.
- Row-norm restoration borrows a geometric invariant from linear algebra.
- Bidirectional KL divergence contributes an information-theory gate.
- Cosine similarity contributes an independent vector-geometry gate.
- A `mix_u32` avalanche tag contributes deterministic corruption detection without pretending to be cryptography.

The discovery kit deliberately reaches further: μ-law companding from telephony and audio, sigma-delta-style residual carry, DCT and Haar/wavelet transforms, Lloyd–Max scalar codebooks, sparse outlier sidecars, and an optics-inspired `refract` split that routes bulk and tail weights through different quantizers.

The method is: **borrow one useful invariant or error behavior, make it a bounded atom, compose it with atoms from other fields, then measure the resulting stack without granting any domain automatic authority.** See [CROSS_DOMAIN_ATOMS.md](CROSS_DOMAIN_ATOMS.md) for the source-backed map and the live-versus-research boundary.

The current v0.2.4 path composes:

```text
rowH8 → Q2/Q4/Q8 block quantization/repair → inverse rowH8
→ stored row norms → source-dtype rounding → serialized decode
→ KL + cosine + optional real-output MSE gates → complete-record verification
```

This is an Atom-lineage project: small, measurable primitives cross domain boundaries, compose into a codec, then get rejected or promoted by explicit fidelity gates.

## Current status

- F32, F16, BF16, Q8_0, Q4_K, and Q6_K GGUF tensors can be read.
- Q2 tries fitted scales and the original grid; `--q2-scale maxabs` selects the baseline. Affine quantization and its Q2 scale fitting are available with `--asym`.
- Two-dimensional tensors use bidirectional KL over sampled `softmax(W · x)` outputs and a cosine safety floor.
- One-dimensional tensors use bidirectional histogram KL and the same cosine safety principle.
- Optional CUDA support routes the sampled matrix multiplication through cuBLAS. Quantization and repair remain CPU work.
- New `.oq` artifacts use `OQ03`, including tensor shapes, decoder state, original GGUF metadata, and complete-record CRC64. OQ02 remains a labeled legacy read path; its missing transform/norm state cannot be recovered.
- `--calibration` consumes source-bound real operator inputs. `--budget-bytes` or `--budget-mib` allocates complete artifact bytes across accepted candidates without relaxing their gates.
- `decode model.oq --out new.gguf` restores a complete dense-source model without its original weights. F32/F16/BF16 output rounding is included in candidate measurement.
- `apply-packed` applies a verified A22-2 matrix directly with lower retained weight memory. It preserves decoded-weight arithmetic but was slower than dense controls in the measured CPU trial. See [usage and measured tradeoffs](docs/PACKED_OPERATORS.md).
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

That is a historical R&D snapshot. The older 10.66× result came from a KL-only picker that later failed real inference. The v0.2.4 work additionally found and fixed missing decoder state in v0.2.3's OQ writeback.

The new SmolLM2-135M-Instruct benchmark evaluated models decoded from their saved containers, using 128 WikiText-2 test chunks at context 512. Calibration used separate training documents. Lower perplexity is better:

| Configuration | Complete file size | Held-out perplexity |
|---|---:|---:|
| Source F32 GGUF | 539.8 MB | 19.6648 |
| Corrected proxy baseline | 86.8 MB | 29.1611 |
| Calibrated adaptive codec | 106.8 MB | 23.6701 |
| Calibrated, 120 MiB budget | 125.8 MB | 21.6583 |

This shows a quality/storage tradeoff on one model, with remaining loss relative to the source. No Q2 tensor passed the retained gates. Inference used decoded dense GGUFs; these are disk compression results, not compressed-kernel speed or resident-memory claims. Exact hashes, sizes, settings, and limitations are in the [benchmark record](docs/FOUNDATION_BENCHMARK.md).

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
| `--key 0xHEX` | Set the deterministic packed-code integrity seed. |
| `--out PATH` | Stream a verified `.oq` container to disk. |
| `--asym` | Use affine min/scale blocks instead of symmetric max-abs blocks. |
| `--q2-scale mse\|maxabs` | Enable fitted Q2 scales (default) or use the original scale baseline. |
| `--calibration PATH` | Load AC01 real operator inputs bound to this GGUF's SHA-256. |
| `--max-output-mse VALUE` | Set the calibrated relative output-MSE ceiling; protected tensors use one quarter of it. |
| `--budget-bytes N` / `--budget-mib N` | Allocate the complete artifact budget across passing candidates; requires calibration. |
| `--out-gguf PATH` | Export a complete dense-source GGUF from saved decoder records. |

Output paths must be new. `--limit` can produce a partial OQ tensor sample, which cannot be exported as a whole model. Standalone `verify` and `decode` commands are documented in [FOUNDATIONS.md](docs/FOUNDATIONS.md).

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
| `src/oq.rs` | Staged OQ03 writer, verified reader, legacy OQ02 reads, and standalone GGUF reconstruction. |
| `src/encoded.rs` | Complete decoder recipe, exact escape, and storage rounding. |
| `src/calibration.rs` | SHA-bound real input batches and transformed feature importance. |
| `src/allocation.rs` | Deterministic allocation under a complete byte budget. |
| `src/cuda_backend.rs` | Optional cuBLAS matrix multiplication for fidelity sampling. |
| `src/main.rs` | Main adaptive CLI and stack-search surfaces. |
| `crates/atom-quantizer-q2` | Isolated Composer 2 research executable. |

## What still has to happen

- Expand real-activation collector coverage across architectures and operator types.
- Validate additional models, long contexts, and downstream tasks beyond the initial v0.2.4 perplexity benchmark.
- Stabilize the CLI and `.oq` format before declaring compatibility.
- Extend CUDA regression coverage across supported cards and driver versions.
- Decide and publish a project license before release.

See [DEVELOPMENT.md](DEVELOPMENT.md) for readiness rules and [THESIS.md](THESIS.md) for the full, intentionally preserved research trail.

The experimental full22 harness has [archive and resume regression checks](docs/EXPERIMENT_VALIDATION.md), including damaged-cache recovery and standalone GGUF export validation.

The [B01 architecture-discovery trial](research/basis-discovery-2026-09-30/OUTCOME.md) added two native Rust component crates and tested 1,280 serialized-record conditions. All principal candidates failed their benefit gate; the measurements remain available, and none was promoted to the main codec.

[B02's local-coupling trial](research/local-basis-2026-09-30/OUTCOME.md) completed another 1,472 conditions. One candidate improved mean error slightly but caused large tensor regressions and lost to controls, so all principal candidates remain rejected.

[B03's block-local resource trial](research/resource-discovery-2026-09-30/OUTCOME.md) completed 3,120 conditions. Its reversible column scales moved the quality tradeoff between tensor classes; all eight principal candidates failed acceptance and remain experimental.

[D01's functional-residual codec](research/direction-retention-2026-09-30/OUTCOME.md) adds a stored calibration-aware low-rank representation through the primitive-toolkit approach. It passed tensor screening and complete archive checks, but its selected rank-one settings did not beat the existing correction in the required full-model comparisons. The option remains experimental.

[D02's calibration-coverage trial](research/calibration-coverage-2026-10-01/OUTCOME.md) adds traced stratified sampling as an experiment option. It increased sample diversity but worsened all six tested model settings; endpoint sampling remains the default.

[D03's separate-calibration function](research/calibration-roles-2026-10-01/OUTCOME.md) fits base weights on interior observations and a stored correction on endpoint observations. Selected 3-bit and 4-bit models passed their size and fresh-final perplexity gates. Main-codec fidelity admission remains unmet, and an initial byte-audit failure remains unexplained despite successful repeats; the function remains experimental.

[R1's export consistency check](research/decoder-reproducibility-2026-10-01/OUTCOME.md) adds A22-2 decoded-tensor commitments. New archives reject changed reconstruction before publishing a GGUF; older archives remain readable. Both selected D03 models retained identical decoded bytes, with 22,848 added archive bytes each. This contains mismatched exports without claiming the original intermittent failure's cause is resolved.

The main CLI now has a read-only [native fidelity assessment command](docs/NATIVE_ASSESSMENT.md). Its [N1 validation](research/native-admission-2026-10-01/OUTCOME.md) measured all 272 tensors of four models across four observation regimes. The 4-bit research model still fails 25 tensors on interior fit data and 24 on selection data; the historical reference passes all regimes. The quantizer and assessor share their metric and gate implementation.

The experimental [adaptive precision builder](docs/ADAPTIVE_PRECISION.md) raises only tensors that fail native fitting gates and preserves passing records. [D04](research/adaptive-fidelity-2026-10-01/OUTCOME.md) produced native-admitted models at 83.6/85.0 MB, but its corrected candidate narrowly lost its required scalar-control PPL comparison. The conditional final test remains unrun; the uniform 6-bit control is a separate promising selection result.

[D05's verified six-bit recipe](docs/SIX_BIT_RECIPE.md) passed native fitting/selection checks and a new final inference test. Its 109.3 MB archive scored 17.4507 perplexity versus 19.0007 for the 125.8 MB historical reference on the same final segment. This is an experimental A22-2 quality/size improvement; native OQ03 integration and packed inference remain separate work.

The main release executable now [verifies and decodes that A22-2 representation natively](docs/NATIVE_A22.md), with no Python or fitting data required for consumption. I1 matched all five D05 model exports byte-for-byte and preserved old OQ behavior. Fitting remains in the experimental builder; unsupported research recipes fail explicitly.

The main app's [build-functional command](docs/FUNCTIONAL_BUILD.md) now runs the actual fitter and independently checks the resulting archive, provenance and both native fidelity views before publication. I2 reproduced D05's validated model byte-for-byte through this command. Fitting uses the existing Python engine; consumption and final acceptance checks are native.

## License

See [PolyForm Small Business License 1.0.0](LICENSE), preserved from the upstream repository.
