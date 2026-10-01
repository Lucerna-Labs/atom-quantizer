# I1 outcome: validated A22-2 models now decode in the main native app

Precommit: `299a835`, [protocol](PROTOCOL.md). The previous D05 turn established
an independent model-quality/size gain. I1 completes native verification and
consumption of that stored representation in the main Rust application.

The normal `target/release/atom-quantizer` now recognizes A22-2 in its ordinary
verify/decode commands. It reconstructs the packed scalar values and stored
rank-two correction itself, without launching Python or reading original model,
calibration or decoded-reference files. [Native usage](../../docs/NATIVE_A22.md)
and the [six-bit recipe](../../docs/SIX_BIT_RECIPE.md) are updated accordingly.

## Native implementation

`src/a22.rs`, `src/a22/header.rs` and `src/a22/record.rs` implement bounded native
ZIP/member checks, authoritative GGUF-prefix validation, AR01/ACZ1 parsing,
zlib completion checks, array-layout validation and streaming f32 reconstruction.
Supported independent recipes include scalar widths 2/3/4/6/8, stored levels/scales,
f16 correction ranks 0/1/2 and raw F32/BF16 or constant exact tensors. Shapes,
groups and padding come from validated records.

Every prefix/record/decoded commitment is checked. Duplicate or missing members,
overlapping tensor extents, checksum-valid malformed descriptors, nonfinite data,
wrong shapes and unsupported decoder side state are rejected. Transform/predictor/
repair/gain stages, higher ranks and other research families fail explicitly;
there is no Python fallback or raw-reference substitution. New export stages a
complete output, preserves existing destinations (including dangling links), and
publishes only after all tensors verify. A staging-name collision cannot delete
an existing file.

Dependencies zip 6.0.0 (default features off), flate2 1.1.10 (pure Rust backend)
and serde_json 1.0.149 were already cached and built offline. They belong to the
conventional app package; native Atom components are unchanged. Existing OQ02/
OQ03 consumption and the default encoder remain intact. Fitting is still handled
by the existing builder; this is native consumption, not native SVD fitting.

## Real-model and independent checks

All five D05 stage archives were verified and exported with the native executable.
All 1,360 tensor arrays and complete GGUF SHA-256 values exactly match the saved
Python-produced files. The selected six-activation output remains:
`4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8`.

Consequently D05's measured PPL 17.4507 belongs to exactly the native-exported
weights. Its archive remains 109,251,501 bytes, unchanged. No new PPL inference is
claimed or needed for identical decoded bytes; the used final segment stays used.
No packed-inference or reduced resident-memory claim is made.

A copied executable and selected archive were placed in an isolated directory.
Verify and export passed with PATH excluding Python/tools. System-call traces
show exactly one execve per operation and no Python, source-model, calibration
or research-script access. The resulting whole GGUF hash matches D05. Only normal
system libraries, the executable/archive and owned output/staging paths were used.

A separate complete-model fault archive changed the last output_norm.weight
commitment. Native export reached that check, rejected it and left neither a
published model nor a staging file. The original archive was preserved. The old
D03 intermittent cause remains unknown; no recurrence occurred in these tests.

Six earlier quantization cases retained identical OQ files, decoded GGUF bytes and
reported scores. The updated normal release also reproduced all 1,088 native
fidelity scores/decisions from the selected model's two fit/two selection views.
This validates the actual updated app path as well as the separate test build.

## Test and build status

- All 71 Rust tests passed, including exhaustive finite-half conversion and staging
  collision preservation.
- All 71 Python tests passed, including ten new native CLI integration cases.
  Independently generated records cover every supported width, padding, raw and
  constant values, f32 scales, activation/diagonal corrections and ranks 0/1/2.
- Corruption tests cover ZIP CRC, truncation, duplicate/missing/extra members,
  inner zlib truncation/suffixes, wrong commitments, malformed array descriptors,
  checksum-valid bad GGUF prefixes, unsupported state, nonfinite values and
  existing/dangling destination preservation.
- CUDA feature compilation passed; actual native consumption tests used CPU.
- Ordinary Clippy reports the same six pre-existing GGUF style warnings and no
  new implementation warnings. No globally clean strict-Clippy claim is made.
- Changed-file formatting and git diff checks pass.

Adding serde_json exposed an ambiguous sum type in an existing allocation unit
test. An explicit u64 annotation resolves that compilation error without changing
runtime code. Its initial failed build log is retained. An unrelated formatter
change in stacks.rs was restored to exact pre-edit bytes. No other pre-existing
source outside the intended app/dependency files and that test annotation changed.

## Release path and recovery

The old release was backed up and its hash verified before updating the normal
path. The tested development binary and normal release are byte-identical:
`22ac0988e9b5e9d32e260ef04b301408f8664416d2a0f96e9abd3cb69c7d577a`.
The package still reports 0.2.4; this is a local development build, not a tagged
or published release. Current executable size is 1,268,976 bytes, compared with
861,824 for the preserved predecessor.

Backup: `artifacts/native-a22-2026-10-01/atom-quantizer-before-release`, SHA
`69b4d49770cd532069aa2d70bb7e6a894d76997ac842c6aac48b1cecdfd4e08f`.
The full pre-edit source and both earlier executables are also in before-source.tar.gz.
Only the protocol was committed. Existing dirty work and all model artifacts remain.

## Evidence and remaining boundary

SUMMARY.JSON, AUDIT.json, MANIFEST.JSON, RELEASE.json, RELEASE_ASSESS.json and
BACKUP.json are compact receipts. Full exports, process/file-access traces,
negative tests, native command logs, old-format regressions and source snapshots
are under `artifacts/native-a22-2026-10-01/`. Test/build logs are under target/.

The app now consumes the validated representation natively. The next integration
boundary is the user-facing fitting/build workflow: the learned fitting engine
remains Python, and OQ03 encoding is unchanged. Do not claim native fitting,
all-family A22 support, packed kernels, broad model generalization or completed
full22 research from this result. The broader goal remains active; B04 remains
UNRUN, and earlier failed hypotheses keep their recorded status.
