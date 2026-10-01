# Native decoded-model assessment

`assess` evaluates a decoded candidate GGUF using the same measurement functions
and acceptance decisions as the main quantizer. It reads both models and their
source-bound AC01 observations; it does not fit or rewrite weights.

```bash
cargo run --release -- assess source.gguf candidate.gguf \
  --calibration fit.acal > assessment.json
```

The N1 verification build is also available at
`target/native-admission-current/release/atom-quantizer`. The incumbent
`target/release/atom-quantizer` was preserved during that experiment.

The complete JSON report goes to stdout. Progress and errors go to stderr. Exit
status 0 means every tensor passed, 2 means assessment completed with fidelity
failures, and 1 means invalid input or an execution error. A gate failure therefore
still produces a complete report. There is no partial-tensor or Gaussian fallback.

The default limits are:

| Metric | Ordinary tensors | Protected tensors |
|---|---:|---:|
| Both directional KL scores | ≤1.0 | ≤0.25 |
| Weight cosine | ≥0.99 | ≥0.995 |
| Calibrated relative output MSE | ≤0.02 | ≤0.005 |

Protection follows the quantizer's existing name classification: `token_embd`,
`output.weight`, `lm_head`, `norm` or `bias`. The optional `--max-output-mse VALUE`
uses the same positive finite base ceiling as quantization, with its protected
quarter-scale limit. N1 used the defaults throughout; no thresholds were relaxed.

Reports include file and header hashes, complete tensor counts, dimensions in
GGUF's innermost-first order, observation kinds/sample counts, scores, actual
native thresholds, each gate result and unused calibration entries. Nonfinite
output-MSE values use JSON `null` with an explicit `output_mse_state`, so an
unbounded error is distinguishable from a non-applicable histogram score.

Embeddings use the main codec's lookup observations. Extra tied-output-head
observations are listed as unused; they are not silently substituted for lookup
validation. One-dimensional tensors use the existing histogram-KL path. Assessment
always uses CPU, including when the program is compiled with the CUDA feature.

Duplicate, missing, extra or differently shaped tensors, unsupported types,
nonfinite weights and invalid/missing source-bound calibration fail assessment.
Source, candidate and calibration hashes are checked again before the complete
report is emitted. Header equality is reported separately: this is tensor fidelity
assessment, not proof of metadata equivalence, OQ03 representability, packed-kernel
behavior or whole-model inference quality.

[N1's measured results](../research/native-admission-2026-10-01/OUTCOME.md) show why
all gates matter: the selected 4-bit research model has four cosine misses, but
its total failing-tensor count reaches 25 on interior fit observations. The
historical product reference passes all four tested observation regimes.
