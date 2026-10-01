# Atom Quantizer development status

Atom Quantizer is **under active development**. It should not be presented as a finished quantizer or as a drop-in replacement for established GGUF tooling.

## Current readiness boundary

- The source builds as two focused Rust crates and has unit coverage for codec, metric, container, and composer primitives.
- The v0.2.4 pipeline adds complete serialized decoding, real-activation input capture and measurement, fitted Q2 scales, global byte budgets, exact escape, and full-record checksums. The main codec preserves the existing transform and KL/cosine design.
- Cross-domain atoms remain labeled by execution status. DCT, wavelet, companding, error-feedback, codebook, outlier, refract, and superpose research candidates are distinct from the main codec.
- OQ03 is a research format with a documented decoder contract. OQ02 remains readable as a legacy representation, with explicit limits on missing metadata.
- The most aggressive historic result was invalidated by inference. A new v0.2.4 SmolLM2 benchmark measures held-out perplexity from serialized-then-decoded artifacts, but still shows quality loss versus the source. Keep both records labeled as model-specific R&D results.

## Exit criteria for an alpha release

1. Extend the initial v0.2.4 benchmark to additional model sizes and architectures from documented hashes.
2. Measure perplexity and representative downstream tasks against the source model.
3. Extend the real-activation/proxy comparison across corpora and supported operators; an imatrix diagonal does not replace actual input samples.
4. Freeze a documented container version and compatibility policy.
5. Extend the new corruption, truncation, dtype and CLI round-trip tests with sustained fuzzing and cross-platform execution.
6. Publish release artifacts with checksums and include the repository's license.

Until those checks pass, repository descriptions, site copy, and benchmark tables must retain the **Under Development** label.
