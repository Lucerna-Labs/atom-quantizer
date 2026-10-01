# Verified six-bit functional recipe

The D05 experiment selected this whole-model recipe before evaluating its unused
final segment. On SmolLM2-135M it passed the existing native fidelity rules and
produced a 109,251,501-byte archive with final perplexity 17.4507. The historical
125,809,756-byte reference scored 19.0007 on the same segment: 13.16% fewer stored
bytes and 8.16% lower measured perplexity.

Reproduce the fit through the main app using the source-bound captures in this checkout:

```bash
target/release/atom-quantizer build-functional \
  models/SmolLM2-135M-Instruct-f32.gguf \
  --base-calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal \
  --residual-calibration artifacts/full22-2026-09-06/fit.acal \
  --out artifacts/new-six-bit-model.a22
```

Use a new archive destination and work directory. This fits the scalar base on interior observations
and a rank-two functional correction on endpoint observations, then checks both
native fitting views. The measured model needed no precision escalation: all
211 matrices stayed at 6 bits and 61 non-matrix tensors remained exact.

The measured files are:

- `artifacts/six-bit-frontier-2026-10-01/activation/model.a22`
- `artifacts/six-bit-frontier-2026-10-01/activation/model.gguf`
- Decoded GGUF SHA256: `4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8`.

The A22-2 archive is self-contained and validates every decoded tensor. Export it
to a new destination with the main native app:

```bash
target/release/atom-quantizer decode \
  artifacts/new-six-bit-model.a22 --out artifacts/new-six-bit-decoded.gguf
```

This is an experimentally verified recipe for the specified source/captures and
evaluation data, not a guarantee for other models or workloads. It uses the
conventional primitive-toolkit branch. The decoded file is dense F32; reduced
resident memory and packed-kernel acceleration were not measured. Native archive
consumption is verified; fitting still uses the Python builder and OQ03 encoding
is unchanged. The [main build command](FUNCTIONAL_BUILD.md) orchestrates that actual
fitting engine and independently enforces native checks. The original intermittent D03 audit
failure remains unexplained; A22-2 rejects changed reconstruction at export.

The final text is now used evaluation data. Do not reuse it as an untouched
holdout for changed models. [D05's full record](../research/six-bit-frontier-2026-10-01/OUTCOME.md)
includes all controls, selection rules and verification evidence. D04's failed
comparison remains unchanged.
