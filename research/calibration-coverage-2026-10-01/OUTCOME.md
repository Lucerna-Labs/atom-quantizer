# D02 outcome — 2026-10-01

**The same-budget stratified-sampling hypothesis FAILED for all six tested global
codec settings.** No seed repeats or final evaluation were triggered, and the
default sampler remains the legacy endpoint method. The new sampler and detailed
position traces are operational research capabilities, not an accepted quality
improvement. The broader goal remains active.

Protocol precommit: `c1d78d4`. This is a conventional primitive-kit observation
function (scan, partition, hash, order and record), not an Atom-emergence result.

## What was established

The old fitting capture selected two endpoints per document. All 128 last-token
embedding IDs are 3717, a paragraph-ending space/newline. Half the first-layer
Q/K observations are consequently identical; first endpoints carry 98.7% of the
captured final FFN-down input energy. These observations motivated testing broader
coverage; they did not establish that replacing endpoints would help inference.

Both endpoint fit and selection captures reproduced the previous AC01 bytes
exactly after the collector extension. D01 control models, corpora and inference
runtime hashes also matched, permitting their explicitly labeled reuse.

| Capture | Samples/operator | First-Q unique rows | Largest identical group | Endpoint positions |
|---|---:|---:|---:|---:|
| Endpoint fit | 256 | 75 | 128 | 256 |
| Stratified fit, seed 0 | 256 | 178 | 14 | 2 |
| Endpoint selection | 64 | 25 | 32 | 64 |
| Stratified selection, seed 0 | 64 | 54 | 4 | 1 |

All four captures contain 212 real operator batches, verified against all 272
source tensors. Every recorded position was audited for bounds, uniqueness and,
where applicable, exact reproduction from the declared hash/stratum formula.

## Complete-model results

All six new archives were exported and checked against every saved tensor
reconstruction (1,632 exact comparisons). All six new 64-chunk CPU inference runs
completed. Endpoint values below are verified reused D01 measurements on the
identical selection corpus/runtime. Lower PPL is better.

| Configuration | Endpoint PPL | Stratified PPL | Stratified archive bytes |
|---|---:|---:|---:|
| 3-bit scalar | 90.0866 | 485.1447 | 58,500,756 |
| 3-bit diagonal residual, rank 1 | 79.8858 | 121.3120 | 59,237,141 |
| 3-bit functional residual, rank 1 | 79.6965 | 484.6687 | 59,240,770 |
| 4-bit scalar | 35.2827 | 44.3903 | 71,817,460 |
| 4-bit diagonal residual, rank 1 | 33.8939 | 42.3958 | 72,562,598 |
| 4-bit functional residual, rank 1 | 34.0391 | 53.4461 | 72,563,888 |

Every quality comparison fails. Files were about 0.35–0.69% smaller, which does
not compensate for the loss. Per the frozen conditional protocol, seeds 1/2,
repeatability acceptance and fresh final inference are UNRUN, not missing results
silently treated as passes. Existing main-codec safety gates were unchanged.

## Why the diagnostic was still useful

Re-scoring all 160 existing D01 tensor records on interior observations reduced
the apparent rank-one functional-correction advantage. Its mean error reduction
over scalar was about 64–65% on endpoint observations, but only about 8–9% on the
new interior observations. This directly demonstrates sensitivity of the proxy
result to the observation distribution; it does not alone identify every cause
of D01's model-level behavior.

At the same time, simply removing boundary emphasis caused severe model damage.
The results suggest preserving high-impact boundary information while adding
representative coverage. They do not prove that stratified sampling is universally
bad, that endpoints alone are optimal, or that diversity is a quality metric.

A useful next hypothesis is to separate fitting coverage from preserved boundary
constraints, or use a controlled hybrid capture. Both need new protocols and
model-level tests. Do not rescue D02 by choosing a different lucky seed, changing
its acceptance thresholds or relabeling its statistical coverage as model benefit.

## Fresh holdout remains available

The native tokenizer verified a new suffix beginning at character 219523 of
full22 final_test.txt. The excluded prefix matches 50,449 original token IDs,
beyond D01's at-most-32,768-token input window; the suffix contains 101,799 tokens.
Its SHA-256 is `42ad320614e02063e1a484efb02fcf2c58f81552aef71ef0468cbd4cb7f68bbd`.
No perplexity run has used it. It remains available for a later frozen final test.
The HF/native token streams differed at a later position, so native tokenization
was used for the exclusion proof rather than guessed HF offsets.

## Implementation and verification

- `scripts/calibration_sampling.py` supplies deterministic stratified indices and
  create-new, complete-file publication. Existing captures/manifests are preserved.
- Both collectors expose `--sampling stratified --seed N`; endpoint behavior
  remains the default. Every selected index, source/corpus identity and file hash
  is recorded. Position traces are explicitly before the final sample cap.
- Both fit and selection replay checks were exact. An additional real generic-CLI
  smoke capture verified all 272 weights and collected 211 operator batches.
- Four new boundary/property tests and all 40 Python tests passed;
  `git diff --check` passed. No new third-party dependency or Atom crate was added.
- The existing llama-tokenize executable was built from unchanged vendored source
  for boundary verification; inference binaries/libraries remained unchanged.
- Run `21107`, generic smoke `69717` and independent audit `66448` exited 0.
  No D02 process remains live.

Receipts: [AUDIT.json](AUDIT.json), [SUMMARY.JSON](SUMMARY.JSON),
[ENDPOINT-REPLAY.JSON](ENDPOINT-REPLAY.JSON), [FRESHNESS.JSON](FRESHNESS.JSON),
[CROSS_SAMPLING_SUMMARY.json](CROSS_SAMPLING_SUMMARY.json),
[CROSS-SAMPLING.JSON](CROSS-SAMPLING.JSON), [MANIFEST.JSON](MANIFEST.JSON) and
[RUNTIME.JSON](RUNTIME.JSON). Large captures, source snapshots, model archives and
logs remain under `artifacts/calibration-coverage-2026-10-01/`.

Example research capture (choose a new output path):

```bash
.venv/bin/python -m experiments.full22.capture --split fit \
  --sampling stratified --seed 0 --out artifacts/NEW-stratified-fit.acal
```

That command is a supported experiment option, not a recommendation to replace
the validated calibration or ignore the failed quality result above.
