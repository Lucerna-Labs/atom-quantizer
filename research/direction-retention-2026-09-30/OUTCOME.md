# D01 outcome — 2026-10-01

The stored functional-residual function is implemented and operational, but
**neither selected global rank-one setting passed the complete-model acceptance
protocol**. No method was promoted to the main codec. The broader goal remains
active; the tensor screen's success was not treated as model-quality proof.

This branch uses Jesse's conventional mathematical primitive-kit/composition
approach: quantize, decode, project the error through real fitting inputs,
factorize, store the factors and reconstruct blindly. It is explicit numerical
optimization, not an Atom synthetic-emergence result. The source catalog's
projection and spectral-decomposition primitives motivated the function.

## Tensor screen

Protocol precommitted as `b51875a`. All 160 declared records completed and were
audited: 144 candidates over eight tensors, 3/4 bits, ranks 1/2/4 and three
geometries, plus 16 scalar references. All six aligned width/rank combinations
passed the frozen tensor-screen gate. The smallest eligible rank was one at both
widths, as required by the precommitted selection rule.

At rank one, mean selection-output NMSE fell approximately 64.35% at 3 bits and
65.43% at 4 bits versus scalar, with 2.62%/2.04% more complete raw-record bytes.
Every sampled tensor improved; aligned fitting beat the same-rank diagonal and
shuffled controls. The saved factors include f16 rounding. Minimum weight cosine
was only .9526/.9787: passing this experimental functional-error screen did not
establish the main codec's cosine/KL safety gates, which were left unchanged.

All 704 earlier full22 records still decoded bit-identically after the decoder
extension. See [SCREEN_SUMMARY.JSON](SCREEN_SUMMARY.JSON),
[SCREEN_RECORDS.json](SCREEN_RECORDS.json), [SCREEN_AUDIT.json](SCREEN_AUDIT.json)
and [DECODER_REGRESSION.json](DECODER_REGRESSION.json).

## Complete models and actual inference

Six complete archives were built and exported. Every one of their 272 tensors
was compared exactly against its saved-record reconstruction (1,632 checks).
Standalone export completed in subprocesses that denied Python opens of original
source/calibration paths. The preserved v0.2.4 budgeted OQ artifact matched its
benchmark hash, and a fresh decode matched the preserved GGUF byte-for-byte.

All inference used the installed llama-perplexity build at commit `74a7c89`,
CPU-only, four threads, context/batch/microbatch 512 and 64 chunks per corpus.
Eight selection runs and the five conditionally required final runs completed.
Model/corpus/log/runtime-library hashes and exact commands are retained. Lower
perplexity is better; values below are new measurements on the stated corpora,
not the historical benchmark's different test segment.

| Model | Complete stored bytes | Selection PPL | Final PPL |
|---|---:|---:|---:|
| Source F32 GGUF | 539,846,112 | 26.7164 | 17.8191 |
| Historical v0.2.4 120-MiB artifact | 125,809,756 | 30.0986 | 19.5882 |
| 3-bit scalar | 58,709,408 | 90.0866 | 54.5865 |
| 3-bit existing diagonal residual, rank 1 | 59,446,131 | 79.8858 | 48.4632 |
| 3-bit functional residual, rank 1 | 59,448,909 | 79.6965 | 48.9310 |
| 4-bit scalar | 72,316,615 | 35.2827 | UNRUN |
| 4-bit existing diagonal residual, rank 1 | 73,065,043 | 33.8939 | UNRUN |
| 4-bit functional residual, rank 1 | 73,065,954 | 34.0391 | UNRUN |

The 3-bit candidate passed selection but failed the final no-worse-than-existing-
correction requirement. It improves final PPL about 10.36% over scalar at 1.26%
extra archive bytes, yet the existing same-rank correction measured slightly
better and slightly smaller. The small gap is not asserted statistically
significant; it fails the precommitted comparison as written.

The 4-bit candidate improved selection PPL only about 3.52%, below the required
5%, and also lost to the existing correction. Its final evaluation was therefore
not triggered by the conditional protocol and remains UNRUN. Higher ranks
passed the tensor screen but were not selected for D01's model stage; their
complete-model performance is still unmeasured.

The historical product reference uses a different calibration procedure and a
larger storage budget; it is a product reference, not the controlled geometry
ablation. All inference here loads dense decoded GGUFs. No compressed-kernel
speed or resident-memory reduction is claimed.

## Interpretation

The correlation-aware geometry helped measured operators substantially, but
did not beat the existing correction on the required model boundary. Fixed
source-activation error is an incomplete predictor of compressed-model behavior.
The experiment does not identify the unique cause: rank capacity, unobserved
directions, changing activations and objective mismatch remain hypotheses.

The next trial should evaluate those questions at the model boundary rather than
assume the larger local reduction is sufficient. Higher ranks and geometry that
balances observed directions with unobserved weight error are candidates for a
new frozen comparison. Do not retune D01 or claim its untested ranks passed.
The first 64 chunks of full22 final_test.txt have now been used; they must not be
presented as untouched holdout data for adaptive successor trials.

## Implementation, verification and retained failures

- New pure function: `experiments/full22/residual.py`; option
  `rank_geometry="activation"` in the existing experimental codec. The default
  diagonal geometry is retained. Factors, effective rank and decoding checks are
  stored in the real record, with no reconstruction-time calibration dependency.
- Six new tests cover the SVD-tail identity after storage rounding, shuffled
  geometry, zero numerical rank, invalid data/factor state, repeated decode and
  complete GGUF export after original/calibration deletion. All 36 Python tests
  pass; `git diff --check` passes. Rust numerical code was unchanged.
- Initial fixture tests caught a module/local-variable name collision; it was
  corrected before the tensor trial. The first model-stage launch rejected tiny
  floating-point differences caused by filesystem receipt order. Restoring the
  declared reduction order reproduced the frozen summary exactly; no metric or
  threshold was changed. Its failed launch log is preserved.
- Run `64031` and independent model audit `43460` exited 0. The earlier
  admission-only launch `40825` exited 1 before model work. No job remains live.

Durable evidence: [MODEL_AUDIT.json](MODEL_AUDIT.json),
[MODEL-SUMMARY.JSON](MODEL-SUMMARY.JSON), [MODELS.JSON](MODELS.JSON),
[MODEL-PROVENANCE.JSON](MODEL-PROVENANCE.JSON), [RUNTIME.JSON](RUNTIME.JSON) and
the screen receipts above. Large archives, decoded models, all 13 inference logs
and source snapshots remain under `artifacts/direction-retention-2026-09-30/`.

Example research use (experimental; no main-codec promotion):

```bash
.venv/bin/python -m experiments.full22.harness build \
  --config '{"bits":4,"rank":1,"rank_geometry":"activation"}' \
  --source models/SmolLM2-135M-Instruct-f32.gguf \
  --calibration artifacts/full22-2026-09-06/fit.acal \
  --out artifacts/NEW-functional.a22
.venv/bin/python -m experiments.full22.harness decode \
  artifacts/NEW-functional.a22 artifacts/NEW-functional.gguf
```
