# Atom Quantizer development status

Atom Quantizer is **under active development**. It should not be presented as a finished quantizer or as a drop-in replacement for established GGUF tooling.

## Current readiness boundary

- The source builds as two focused Rust crates and has unit coverage for codec, metric, container, and composer primitives.
- The latest checked-in pipeline is v0.2.3: rowH8, adaptive Q2/Q4/Q8, blind repair, row-norm preservation, dual KL/cosine gates, and integrity verification.
- Cross-domain atoms are labeled by execution status: the v0.2.3 live path is distinct from DCT, wavelet, companding, error-feedback, codebook, outlier, refract, and superpose research candidates.
- The `OQ02` container is a research format. Compatibility is not promised yet.
- The most aggressive historic result was invalidated by real inference. Only the dual-gate baseline should be cited, and even that must be labeled as a recorded R&D snapshot.

## Exit criteria for an alpha release

1. Reproduce a clean full-model v0.2.3 benchmark from a documented source model hash.
2. Measure perplexity and representative downstream tasks against the source model.
3. Add real-activation or imatrix calibration and compare it with the Gaussian proxy.
4. Freeze a documented container version and compatibility policy.
5. Add corruption, truncation, fuzz, and cross-platform read/write tests.
6. Select a license and publish release artifacts with checksums.

Until those checks pass, repository descriptions, site copy, and benchmark tables must retain the **Under Development** label.
