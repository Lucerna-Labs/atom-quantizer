# Native A22-2 verification and export

The main Rust app now consumes the validated six-bit representation directly:

```bash
target/release/atom-quantizer verify \
  artifacts/six-bit-frontier-2026-10-01/activation/model.a22
target/release/atom-quantizer decode \
  artifacts/six-bit-frontier-2026-10-01/activation/model.a22 \
  --out artifacts/new-native-six-bit.gguf
```

The output destination must be new. Format detection uses the file signature.
Existing OQ02/OQ03 verification/export behavior remains available. A22-2 native
consumption needs only the executable and archive, not Python, the source model,
calibration captures or an existing decoded GGUF.

Supported stored recipes are raw F32, raw BF16, constant tensors and independent
packed scalar codes at 2/3/4/6/8 bits, with stored levels/scales and optional f16
rank-zero, rank-one or rank-two corrections. Shapes, groups and padding come from
the records. Other research families, rotations, predictors, repair/gain side
state and higher correction ranks are rejected explicitly. They are not silently
ignored or routed through Python. Existing research tooling retains its wider
format support.

Verification performs the real numerical reconstruction. It checks the ZIP member
set and duplicates, stored GGUF extents, manifest identities, record SHA checksums,
bounded zlib completion, array layouts, finite values and every decoded-tensor
commitment. Export writes to an owned temporary file and publishes only after all
checks pass. Existing files and dangling destination links are preserved; failed
exports leave no partial model or staging file.

The native implementation uses pure Rust ZIP/zlib/JSON dependencies in the
conventional app package. It changes no native Atom primitive or main encoding
threshold. The existing default encoder still writes OQ03; learned fitting remains
in the documented experimental builder. This is native archive consumption, not
native SVD fitting or a packed-inference kernel.

I1 verified/exported all five D05 stage archives: 1,360 tensor byte comparisons and
complete GGUF hashes matched the Python-produced references. The selected output
remains `4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8`, so its measured
D05 PPL applies to exactly the same weights. No new inference is claimed.

An isolated executable/archive export with Python unavailable on PATH passed.
System-call traces showed one executed program and no source-model, calibration
or research-script access. A full-model wrong commitment on the final tensor was
rejected without publication. [I1's evidence](../research/native-a22-2026-10-01/OUTCOME.md)
includes all regression, isolation and release-path checks.
