# Experimental archive and resume validation

The full22 Python harness is a research workflow separate from the Rust OQ03
codec. Run its regression suite from the repository root with the project's
Python environment (NumPy, SciPy, GGUF and threadpoolctl are required):

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
cargo test --workspace --offline
```

The Python suite creates disposable GGUFs and source-bound synthetic calibration
files. It performs real serialization, compression, screening, archive creation
and standalone CLI decoding. It does not download a model or run inference.

## Behavior covered

- Both raw and compressed records retain their decoded values. Truncation,
  byte corruption, invalid compression, extra streams and malformed array
  descriptors fail validation, including checksum-valid malformed metadata.
- Bad cached JSON and compressed records cause a cache miss instead of aborting
  a resumed screen. The previous result and artifact are retained under the
  run's `history/` directory before replacement. A successfully recomputed
  result can be resumed normally.
- Changing sampled rows or candidate configuration produces a different run
  fingerprint. Invalidated runs cannot resume. Run identity also incorporates
  source, fit and selection hashes, implementation files and dependency versions.
- Cache and archive records must declare the expected tensor shape before their
  decoder runs. Archive shapes and offsets are checked against the stored GGUF
  table; an entry checksum alone is insufficient.
- A complete lossless archive decodes byte-identically to the fixture GGUF after
  removal of the source. Mismatched names, offsets, shapes, source sizes and
  duplicate tensor entries are rejected without publishing an output or leaving
  a staging file. Existing output files are preserved.

## Verification on 2026-09-30

All 17 Python regression tests passed. The Rust workspace also passed its 55
existing tests: 44 main-library, 6 main-CLI, 1 Q2 accounting and 4 Q2-CLI tests.
The Rust export and accounting fixes were already present when this improvement
pass began; this pass changed Python record validation, cache recovery and the
archive's pre-decode shape check.

All 704 existing full22 tensor artifacts (69,858,158 bytes in total) were loaded
with the previous and updated record readers. Every decoded array was identical;
zero failed. Per-artifact hashes and shapes are retained in the local diagnostic
report `target/improvement-2026-09-30/compatibility.json`. The initial regression
failures remain in `target/improvement-2026-09-30/before-tests.log`. These local
files are ignored build output; the repository test source is the reproducible
regression coverage.

This establishes record compatibility and the tested file-handling behavior.
It does not establish improved model quality, faster inference, lower resident
memory or completion of the full 22-candidate experiment. Full-model and runtime
evaluation remain separate work under the existing experiment protocol.

## Separate calibration roles (D03)

The experimental builder accepts `--residual-calibration` in addition to its
base `--calibration`. Base observations fit the quantized weights; residual
observations fit only the stored low-rank correction. Each file is bound to the
original GGUF by SHA-256. The archive records both identities and decoding uses
neither file. Omit the new option to retain the previous single-file behavior.

For example, using the existing D02 interior capture for the base and endpoint
capture for the correction:

```bash
.venv/bin/python -m experiments.full22.harness build \
  --config '{"bits":3,"rank":1,"rank_geometry":"activation"}' \
  --calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal \
  --residual-calibration artifacts/full22-2026-09-06/fit.acal \
  --out artifacts/new-calibration-roles.a22
.venv/bin/python -m experiments.full22.harness decode \
  artifacts/new-calibration-roles.a22 artifacts/new-calibration-roles.gguf
```

Both destinations must be new. This is the conventional primitive-toolkit
research codec; the Rust OQ03 defaults and safety gates are unchanged. The dense
GGUF export supports model-quality evaluation, not compressed-kernel or resident
memory claims. [D03's frozen protocol](../research/calibration-roles-2026-10-01/PROTOCOL.md)
defines its model comparisons and conditional final evaluation.

Five added boundary tests cover default byte compatibility, independent input
roles with unchanged base payloads, invalid/missing/wrong-source observations,
calibration mutation before publication, and decoding after removal of all fit
files and source. The full Python suite contains 45 passing tests. A separate
CLI build/decode smoke verified both fixture tensors after removing those files.

## Decoded tensor commitments (R1)

New builds use **A22-2**. Each manifest tensor includes `decoded_sha256`, the
SHA-256 of its decoded row-major little-endian F32 bytes. Export validates every
commitment before publishing the GGUF. A mismatch raises `DecodedChecksumError`
with the tensor name and expected/actual digests; the incomplete temporary file
is removed. A22-2 rejects missing or malformed commitments before decoding.

The updated decoder still reads A22-1 archives. Export reports its format and
`decoded_verified_tensors`: a legacy archive without these commitments receives
zero commitment-verification credit. Older decoders reject A22-2 explicitly.
Existing archives are not rewritten automatically.

R1 rebuilt the two selected D03 models with all 544 tensor records and both
complete decoded GGUF hashes unchanged. Each archive adds exactly 22,848 bytes.
Their PPL evidence is reused for the identical decoded files; no new inference
is claimed. The full Python suite now has 50 passing tests, including one-ULP and
signed-zero fault injection. A complete-model bad-hash test rejected the final
tensor without publishing an output or leaving a staged file.

The original intermittent D03 audit failure remains unexplained. A bounded probe
passed 6,528 tensor comparisons across buffer alignments and thread counts, plus
3,376 independent rank-one arithmetic comparisons and 4,063,232 finite-half
conversion checks. The new contract detects changed reconstructions; it does not
establish the cause of that old failure or guarantee export on every numerical
runtime. [R1 outcome](../research/decoder-reproducibility-2026-10-01/OUTCOME.md)
records the evidence and remaining limits.
