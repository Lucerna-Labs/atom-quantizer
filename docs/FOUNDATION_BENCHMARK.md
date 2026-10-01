**v0.2.4 foundation benchmark — 2026-09-05**

The four initial foundations are implemented in the main codec and exercised on a real model. Calibration and a larger byte budget improve the measured quality/storage tradeoff. They do not recover the source model's full quality, and no Q2 tensor passes the retained gates in this run.

| Configuration | Actual file bytes | Effective bits/parameter, including metadata | Held-out perplexity |
|---|---:|---:|---:|
| Source F32 GGUF | 539,846,112 | — | 19.6648 |
| Corrected proxy baseline | 86,816,410 | 5.1632 | 29.1611 |
| Calibrated adaptive | 106,833,562 | 6.3537 | 23.6701 |
| Calibrated, 120 MiB budget | 125,809,756 | 7.4823 | 21.6583 |

The calibrated artifact reduces perplexity by about 18.8% against the corrected proxy baseline while using about 23.1% more bytes. The 120 MiB allocation improves further at additional storage cost. All compressed variants remain worse than the source. These points do not establish an optimal rate–distortion frontier.

**What was measured**

- Model: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`; 134,515,008 stored parameters in 272 tensors.
- Source: F32 GGUF exported with the official llama.cpp converter. Source SHA-256: `53f86c219b099fa84cd005f1b22572f6c4e53f8ba242fb8074f600712abe9c5b`.
- Dataset: [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- Calibration: first 16 training rows whose stripped text exceeds 400 characters; 128-token context; 1,973 actual input tokens; 32 samples per operator. The collector verified all 272 source tensors and captured 211 linear/embedding operator batches.
- Evaluation: separate test split; first 128 chunks, context 512, batch and microbatch 512, eight CPU threads. llama.cpp scores 255 next-token predictions per chunk in this configuration: 32,640 predictions total.
- Runtime: `llama-perplexity` from llama.cpp commit `74a7c897f049c17e7080423aa2111776eff6ebbf`. Every compressed evaluation loaded a GGUF decoded from its saved OQ03 artifact, without the original weights being required for decode.

The source precision is explicitly F32. Compression ratios in this benchmark use that actual GGUF file as the denominator, not the checkpoint's BF16 storage size. Inference is on decoded dense weights: this benchmark makes no compressed-kernel throughput, energy, or resident-memory claim. Some perplexity jobs ran concurrently, so their elapsed times are not used for performance comparisons.

The corrected proxy baseline includes the decoder/integrity fixes and exact escape, retains Gaussian KL and cosine checks, and disables Q2 scale fitting with `--q2-scale maxabs`. Calibrated runs retain those cosine/KL thresholds and additionally require relative operator-output MSE ≤ 0.02, or ≤ 0.005 for protected tensors. No Q2 tensor is selected in either case, so the model-level quality differences do not establish a Q2 scale-fitting gain.

| Configuration | Q2 tensors | Q4 tensors | Q8 tensors | Exact F32 tensors |
|---|---:|---:|---:|---:|
| Proxy baseline | 0 | 211 | 6 | 55 |
| Calibrated adaptive | 0 | 159 | 58 | 55 |
| 120 MiB budget | 0 | 67 | 144 | 61 |

The budget allocator's minimum feasible complete size is **106,833,562 bytes**. Running at exactly that budget reproduces the calibrated artifact byte-for-byte. At 120 MiB, the cap is **125,829,120 bytes**, and the actual artifact leaves **19,364 bytes** unused. Its summed local error surrogate falls from 1.156729 to 0.433231. That surrogate is distinct from perplexity. A subsequent encode with the final numerical implementation reproduced the evaluated 120 MiB artifact byte-for-byte.

**Verification**

The validation suite has 42 library tests and six CLI integration tests, all passing. Checks include complete transform/norm reconstruction, exact escape, dtype rounding and exported-value equality for F32/F16/BF16, source-hash rejection, impossible budgets, byte-budget enforcement, altered records, truncation, existing-file preservation, and failed-output publication. Format checks, Clippy with CUDA enabled, and release workspace builds pass. CUDA was compile-checked; these model and capture runs used CPU.

The final capture script reproduced the calibration payload, and the checked-in dataset preparation script reproduced both input files. This is targeted validation, not sustained fuzzing or cross-platform runtime qualification.

**Reproduction**

Use Python 3.13 with the pinned packages in `scripts/requirements-benchmark.txt`, the model and llama.cpp revisions above, and the [calibration instructions](FOUNDATIONS.md). Prepare inputs and capture actual activations:

```bash
.venv/bin/python scripts/prepare_benchmark.py --out artifacts/new-run
.venv/bin/python scripts/capture_calibration.py \
  --model models/SmolLM2-135M-Instruct \
  --gguf models/SmolLM2-135M-Instruct-f32.gguf \
  --corpus artifacts/new-run/calibration.jsonl \
  --out artifacts/new-run/real.acal \
  --samples 32 --prompts 16 --context 128 --threads 8

target/release/atom-quantizer models/SmolLM2-135M-Instruct-f32.gguf \
  --calibration artifacts/new-run/real.acal \
  --budget-mib 120 --out artifacts/new-run/budgeted.oq

target/release/atom-quantizer decode artifacts/new-run/budgeted.oq \
  --out artifacts/new-run/budgeted.gguf

third_party/llama.cpp/build/bin/llama-perplexity \
  -m artifacts/new-run/budgeted.gguf \
  -f artifacts/new-run/heldout.txt \
  --chunks 128 -c 512 -b 512 -ub 512 -t 8
```

For the calibrated adaptive comparison, omit the budget. For the corrected proxy comparison, omit calibration and use `--q2-scale maxabs`. Evaluate the source GGUF with the same perplexity arguments. Keep all output paths distinct.

The machine-readable [results and artifact hashes](../research/foundations-2026-09-05/results.json) are retained in the repository. Larger local artifacts and raw logs are under `artifacts/foundations-2026-09-05/` and are ignored by Git. The original failed decoder study remains preserved under `research/first-principles-2026-09-05/`.

This is one small Llama-family model and one held-out text subset. Larger models, other architectures, instruction/task accuracy, longer contexts, richer atom families, and compressed inference kernels remain unvalidated by this benchmark.
