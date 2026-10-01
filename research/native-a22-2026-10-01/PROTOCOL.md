# I1: native app consumption of the validated A22-2 representation

Precommit status: UNRUN. D05 was progress: a fixed, native-admitted recipe beat
the historical size/quality point on an independent final segment. I1 integrates
its stored representation into the main Rust app; it is not another quality search.

## Operational boundary

Extend the main `verify` and `decode ... --out NEW.gguf` commands to recognize
A22-2 and perform actual native Rust validation/reconstruction. No Python process,
original model, calibration file or separately decoded reference may be required
by native consumption. The existing OQ02/OQ03 paths and default encoder remain.

Keep the current compressed archives unchanged: ZIP-stored members containing
AR01 or zlib-compressed ACZ1 records, manifest and GGUF prefix. Use bounded native
ZIP/zlib/JSON readers and preserve record/prefix/decoded SHA commitments. These
are conventional application dependencies; no native Atom component is changed.

Implement the representation actually validated in D05: independent scalar packed
codes, stored levels/scales, optional rank0/1/2 f16 corrections, and raw F32/BF16 or
constant exact records. Decode scalar products and rank-two corrections in f32
with separate correction accumulation, matching the saved representation. Read
shapes, groups, padding and arrays from the archive; do not assume model-specific
names or dimensions. Support the scalar bit widths already represented by this
codec (2/3/4/6/8), with finite validated values and bounded allocation.

Unsupported transforms, predictors, repair/gain side state, higher ranks or other
research codec families must fail explicitly. Do not silently ignore fields,
substitute raw weights, or call Python as a native-decoding fallback. Wider A22
coverage remains separate work; this boundary must fully support the validated
recipe rather than claim every full22 family.

Validate the authoritative GGUF tensor table against the manifest, exact ZIP
member set including duplicates, array descriptors/extents/checksums, expected
shapes and decoded commitments. Verification must actually decode every tensor.
Export must stage complete output, reject mismatches, preserve existing files
including dangling symlinks, and publish only after complete verification. No
partial or failed model may be published.

## Required verification

Preserve the current source/executables first. Test native reconstruction against
independently produced Python records across scalar widths, padding, rank0/1/2,
raw/constant values and f16 edge cases. Test truncation, record/ZIP corruption,
duplicate/missing/extra members, malformed metadata, shape/offset mismatch,
unsupported recipes, nonfinite data and wrong decoded commitments. No test-only
stub may stand in for native numerical decoding.

Use a disposable fixture archive to prove main-CLI verify/export works when the
original model and all calibration/reference files are removed and Python is
unavailable. Preserve old OQ behavior using prior baseline fixtures/artifacts;
run Rust workspace tests and relevant feature/build checks. New code must avoid
panics/unwrap/expect outside tests; do not alter old unrelated code to hide lints.

Verify and export the five D05 stage archives with the native app. Require every
tensor commitment and complete exported GGUF SHA-256 to match its recorded Python
counterpart exactly. Reuse PPL evidence only for identical decoded hashes; do not
spend another holdout on unchanged weights. Independently audit the real command,
executable/source identity, archives, exported values and old/new format behavior.

Decode the selected complete archive from an isolated directory using only the
native executable/archive and a fresh output path, with no Python or fitting data
available through that workflow. Keep the existing release available until these
checks finish. Report the native result and any unsupported scope honestly.

Success establishes native archive consumption, not native SVD fitting, OQ03
representation, packed inference, resident-memory savings or universal model
quality. Preserve D05's exact winner, D04's failed comparison, all used-final data
boundaries and the unresolved original D03 audit cause. The broader goal remains
active while required app integration and full22 work remain.
