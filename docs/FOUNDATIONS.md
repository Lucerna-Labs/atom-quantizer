**Using the four v0.2.4 foundations**

The normal codec now has complete OQ03 decoding, source-bound activation calibration, optimized symmetric and affine Q2 scales, and optional global byte allocation. These are operational features of the main CLI. The separate Composer 2 binary retains its earlier research behavior; it is not a calibrated v0.2.4 codec or a compressed inference runtime.

Build the CPU path:

```bash
cargo build --release --workspace
cargo test --workspace --all-targets
```

The main tensor path is rowH8 on eligible matrices, block quantization/repair, inverse rowH8, stored row-norm restoration, and source-dtype rounding. The exact serialized decoder output must pass KL and cosine gates. With calibration, unsoftmaxed operator output error must also pass. If all lossy candidates fail, the tensor is stored through an exact F32 escape. Output names must be new; original models and existing artifacts are not overwritten.

```bash
target/release/atom-quantizer source.gguf --out model.oq
target/release/atom-quantizer verify model.oq
target/release/atom-quantizer decode model.oq --out reconstructed.gguf
```

The last command needs no original model weights. For a complete dense source, `--out-gguf reconstructed.gguf` can also be passed during encoding; it uses the saved container decoder. See [the format contract](OQ03.md) for legacy and dtype boundaries.

**Real operator inputs**

`scripts/capture_calibration.py` runs local Hugging Face forward passes with hooks on mapped linear layers and token embeddings. It verifies the GGUF weights against the checkpoint before capture, including the supported llama.cpp Q/K permutation. It rejects missing mappings or unverifiable conversions. The collector was validated on SmolLM2-135M-Instruct; other architectures may require explicit additional operator adapters. Tied embeddings currently use lookup samples for the stored embedding tensor; downstream effects through a shared output head still require whole-model evaluation.

Install the optional Python tools in this project:

```bash
uv venv .venv --python python3
uv pip install --python .venv/bin/python -r scripts/requirements-calibration.txt
```

Provide a local checkpoint, a GGUF exported from exactly those weights, and a JSONL corpus containing strings or objects with a `text` field:

```bash
.venv/bin/python scripts/capture_calibration.py \
  --model models/checkpoint \
  --gguf source.gguf \
  --corpus calibration.jsonl \
  --out real.acal \
  --samples 32 --prompts 16 --context 128 --threads 8

target/release/atom-quantizer source.gguf \
  --calibration real.acal --out calibrated.oq
```

The AC01 file contains actual input vectors or embedding indices, a source GGUF SHA-256, corpus SHA-256, and a file checksum. A JSON sidecar records capture configuration, tokenized-input hash, coverage, and software version. The Rust CLI refuses a different source model or missing matrix calibration; it does not quietly fall back to Gaussian probes within a calibrated run. Small one-dimensional tensors retain their histogram/cosine checks and can escape to exact storage.

The added metric is `sum((reference_output - decoded_output)^2) / sum(reference_output^2)`. The default ceiling is 0.02, or 0.005 for protected tensors. `--max-output-mse VALUE` changes the normal ceiling; the protected ceiling is one quarter of that value. It requires calibration. Existing cosine floors (0.99/0.995) and KL ceilings (1.0/0.25) remain mandatory. These per-tensor tests are useful filters, not a theorem about whole-model quality.

**Optimized Q2 scales**

`--q2-scale mse` is the default. The encoder searches scale/assignment refinements, retaining the original max-abs or affine min/max grid as a candidate. With real linear samples it uses column second moments in the transformed coordinate basis; without calibration it fits unweighted reconstruction error. Both optimized and original Q2 candidates can be evaluated. `--asym` uses the affine four-level family. No new scale metadata or code bits are introduced by the fit.

`--q2-scale maxabs` disables the fitted candidate, providing a controlled baseline. Symmetric Q2 remains a ternary {-scale, 0, scale} alphabet packed in two bits. A four-level nonuniform symmetric codebook, lattice coding, and other proposed families have not been added by this foundation work.

**Whole-artifact byte budgets**

```bash
target/release/atom-quantizer source.gguf \
  --calibration real.acal \
  --budget-mib 120 \
  --out budgeted.oq
```

`--budget-bytes N` specifies an exact integer budget; use one budget flag. Both include the model prefix, record frames, checksums, scales/minima, transforms, row norms, and any exact escapes. The first pass measures all passing candidates one tensor at a time. A deterministic Lagrangian search with marginal-gain filling selects the allocation; this is a feasible heuristic, not an exact discrete optimum. Its objective sums normalized operator errors, with weight error for uncalibrated one-dimensional vectors. It is not measured perplexity.

The second pass reconstructs the chosen records and checks their bytes and checksums against the plan. Source hashes are checked again, the final size must fit the budget, and failed encodes are not published at the requested artifact path. A budget below the cheapest passing allocation is rejected with its minimum feasible size. The allocator does not relax fidelity thresholds to force a smaller file.

**Evaluation boundary**

[The recorded foundation benchmark](FOUNDATION_BENCHMARK.md) uses a real model, disjoint WikiText-2 train/test data, and llama.cpp inference on models decoded from the compressed artifacts. It finds a quality/storage tradeoff, not lossless compression. No Q2 tensor passed the retained gates in that run. The broader cross-domain candidates in the research study remain proposals or separate experiments.

This is a disk codec. The measured `.oq` savings do not imply the same reduction in resident inference memory: current evaluation exports dense GGUF weights. Fused compressed matmul, larger-model and multi-architecture validation, downstream task evaluations, and broader CUDA runtime testing remain future work.
