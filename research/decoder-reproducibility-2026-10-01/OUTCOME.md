# R1 outcome: export consistency implemented and verified

Precommit: `52105ff`, [protocol](PROTOCOL.md). The previous goal turn made progress
through D03's real model-quality gains and an open audit failure. This turn adds
a fail-closed export boundary and completes the declared reproducibility probe.
The initial D03 failure's cause remains unknown; it is not relabeled as fixed.

## What changed

New experimental builds now use **A22-2**, with a `decoded_sha256` commitment for
every tensor: SHA-256 over row-major little-endian F32 decoded bytes. The exporter
checks those values before publishing a GGUF. Any mismatch identifies the tensor
and expected/actual digests, rejects publication and removes the temporary file.
A22-2 rejects missing or malformed commitments before decoding starts.

The updated reader still decodes A22-1. Its result reports the format and actual
`decoded_verified_tensors` count; an old archive without commitments gets zero
verification credit. The exact pre-R1 reader was also tested and rejects A22-2
with `unknown model format`, so an old reader cannot silently skip the new rule.
No existing artifact was overwritten or upgraded in place.

This is a conventional integrity/compare function in the primitive toolkit. It
commits reconstructed values as well as the recipe that generated them. It is
not an Atom-emergence or cryptographic-authentication claim. Decoder arithmetic,
quantization algorithms, native Atom primitives, main defaults and fidelity
thresholds are unchanged. The guard detects changed reconstruction; it does not
guarantee that every numerical runtime will successfully reproduce it.

## Bounded diagnostic

The probe checked D03's q4-IE-r1 and selected q3-IE-r1/q4-IE-r2 models. Archive and
fixed decoded-file identities were verified before and after execution.

- All 6,528 tensor comparisons passed: three complete models, four stored-array
  offsets modulo 64 (0, 1, 31, 63), and BLAS thread limits of 1 and 4.
- All 3,376 rank-one comparisons against elementwise outer-product arithmetic
  passed, without using matrix multiplication for that reference calculation.
- All 4,063,232 finite-half conversion comparisons passed: all 63,488 finite
  binary16 bit patterns at 64 offsets, checked against an integer expansion to
  binary32. The check includes signed zeros and subnormal values.
- Stored record state remained unchanged, and the floating-point rounding mode
  did not change across the tested decoder calls.

The frozen runtime records NumPy 2.5.2, OpenBLAS 0.3.34.0.0 (Haswell, pthreads),
Python and relevant binary/source hashes. A recurrence would save the original
record, expected/actual arrays, byte differences, alignment/strides, thread
limit and rounding mode. No recurrence was captured. Passing this bounded matrix
is evidence about these runs, not proof of the original failure's cause.

## Complete-model results

Both selected D03 configurations were rebuilt with the same source and separate
fit inputs. Standalone export denied Python opens of source and both calibration
files. Every decoded commitment passed. All underlying tensor records, prefixes
and complete decoded GGUF files matched D03 exactly.

| Configuration | Identical records | A22-1 bytes | A22-2 bytes | Added bytes | Prior fresh-final PPL |
|---|---:|---:|---:|---:|---:|
| 3-bit IE rank 1 | 272 | 59,240,812 | 59,263,660 | 22,848 | 44.9737 |
| 4-bit IE rank 2 | 272 | 73,268,797 | 73,291,645 | 22,848 | 21.6630 |

The extra bytes are included in actual whole-file accounting. PPL receipts are
explicitly **reused**, bound to the identical decoded GGUF SHA-256 values; no new
inference or fresh holdout consumption occurred during R1. This is a format and
export verification, not a new quality trial or compression-kernel benchmark.

The new hashes of the decoded models remain:
- q3-IE-r1: `c124c0d1f48219e0b28a2dbfe5bcb42bd4a14ca9b32b32be5f36c23bcd4016ad`
- q4-IE-r2: `7208fd04dab974cd6997a3ce62c99ea600afea87ef12543dc28ccc505b4fa89b`

## Failure paths and compatibility

All 50 Python tests passed, including five new commitment tests. They cover
malformed/missing digests before decoder execution, one-ULP and signed-zero
changes, existing-destination preservation, staging cleanup and legacy reads.
Existing tests also confirm export after deleting source and both calibrations.
A separate real CLI smoke did the same and reported two verified A22-2 tensors.

For the complete 4-bit model, a **separate injected-fault archive** changed only
the final tensor's commitment to a valid but incorrect digest. Export processed
the preceding tensors and then rejected `output_norm.weight`. No destination or
staged GGUF remained. The original model archive was preserved. This test proves
the failure path; it does not explain the spontaneous D03 audit failure.

A legacy A22-1 4-bit model was independently re-exported, with an identical full
GGUF hash and zero decoded-commitment verification credit. The original D03
reader rejected the new format before publication. `git diff --check` passed.

## Remaining work

The initial D03 audit failure remains open and documented with its original log
and exact auditor source. Do not erase it or assign an unmeasured cause. A22-2
now contains mismatched exports when its committed values differ; successful
exports carry explicit verification of the reconstructed bytes.

Main-codec promotion still requires its actual native metrics and a representation
that passes them. Reading current code confirmed default relative output-MSE
ceilings of 0.02 (normal) and 0.005 (protected), bidirectional KL ceilings of 1.0
and 0.25, and cosine floors of 0.99 and 0.995. The four previously recorded 4-bit
cosine misses are only known necessary failures, not a complete admission result.
The main code treats embeddings as lookup observations, whereas the D03 functional
fit also uses the tied output-head observations. A native assessment should use
the existing operator and metric paths rather than treating Python proxy scores
as proof of native admission. No such assessment or integration is claimed here.

D03's used final segment remains used. The broader app-improvement goal, native
admission work and full22 programme remain active; B04 remains UNRUN. Only the
new R1 protocol was committed, preserving earlier dirty work.

## Reproduce and inspect

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m experiments.decode_reproducibility --out artifacts/new-R1-probe
.venv/bin/python -m experiments.decoded_commitments --out artifacts/new-R1-model-check
```

Both runners require new output locations. The first reproduces the bounded
probe; the second rebuilds and verifies the two fixed models without a new PPL
run. Source snapshots, logs, real-model manifests and test evidence are retained
under `artifacts/decoder-reproducibility-2026-10-01/`. Compact JSON receipts in this
directory bind the exact observations and model hashes.
