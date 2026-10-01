# B03 outcome — 2026-09-30

**All eight principal order/readout/width candidates FAILED the frozen benefit
gate.** The declared experiment is complete. The automatic architecture goal and
full22 work remain active and incomplete; no candidate was promoted.

Protocol precommit: `c9bcbc0`. The native dynamics now expose their existing
resource state through a batched block-field executable. B03 measures independent
block-local fields and stores reversible f32 column scales in real codec records.
Both resource and inverse-resource readouts were tested without correction,
clipping, score-driven routing or per-tensor winner selection.

## Evidence motivating the change

[Post-run B02 diagnostics](../local-basis-2026-09-30/FAILURE_ANALYSIS.json) showed
that the final FFN-down tensor's input energy varied about 976x across blocks.
The shared field obscured strong block-specific correlation structure. Its
reconstruction weight error fell about 17% while operator-output error rose about
87%. This motivated changing the field's locality and its codec readout; it did
not authorize reclassifying the failed B02 candidates as successful.

## Complete comparison

All 19,584 native block executions completed in 768 batches. All 3,120 unique
scored records completed: 3,072 candidates plus 48 conventional references.
Every declared order, arm, seed, readout, width and tensor is present. Mean
selection-output NMSE across the declared tensors/seeds:

| Width | Scalar | Diagonal reference | Conductance order 0 | Conductance order 1 | Resistance order 0 | Resistance order 1 |
|---|---:|---:|---:|---:|---:|---:|
| 3 bits | .01165559 | .01286262 | .01284880 | .01281695 | .01206243 | .01194546 |
| 4 bits | .00310705 | .00278022 | .00350165 | .00351655 | .00316124 | .00315663 |

All full candidates lose to plain scalar on mean error. Conductance increases
mean error roughly 10–13%; resistance increases it roughly 1.6–3.5%. The stored
scale state adds approximately 3.92% raw bytes at 3 bits and 3.05% at 4 bits.

Inverse-resource order 1 improves the final FFN-down tensor's 3-bit error by
28.56%, but increases embedding/output-head error by 69.19%. Its 4-bit embedding
regression is 47.46%. Conversely, the conductance readout improves that embedding
metric while hurting other tensors, including more than doubling final FFN-down
error at 4 bits. Neither universal readout passes the per-tensor constraints.

| Gate | Status | Evidence |
|---|---|---|
| G1 native invariants | MEASURED | Max transport orthogonality error 4.44e-15; finite states/bounds pass; resources in [.55426752, 1] |
| G2 stored reconstruction | MEASURED | Worst unquantized reconstruction NMSE 4.873e-16; scales match native readouts exactly after declared f32 rounding |
| G3 benefit | FAILED | All eight candidates lose to scalar and required controls; each has disallowed tensor regressions |
| G4 contributions | MEASURED | All required removals and coupling/drive controls change the stored observation and measured result in every seed |
| G5 evidence | MEASURED | Exact 3,120 count; frozen hashes unchanged; all artifacts and native readouts independently audited |

The independent audit checked every artifact hash, actual raw and compressed
length and finite blind decode. All 3,072 stored scale vectors matched both the
encoded record and the declared readout of the saved native resource. The
resource minima/maxima matched native diagnostics in every block, and the 768
unique batches summed to 19,584 blocks. See [AUDIT.json](AUDIT.json),
[SUMMARY.JSON](SUMMARY.JSON), [MANIFEST.JSON](MANIFEST.JSON) and the full compact
[RECORDS.json](RECORDS.json). Payloads and the exact source/binary snapshot are in
`artifacts/resource-discovery-2026-09-30/`. The run and audit exited successfully.

## Narrow observation and next question

For 3-bit inverse-resource order 1, the full multi-chaos stack lowers mean error
2.54% versus no-chaos, 1.81% versus no-rotor and 2.30% versus no-population. That
is a measured local comparison within a rejected experiment. It does not satisfy
the overall benefit gate, establish general chaos superiority or justify product
promotion. Quantum-base removal also changes and worsens that arm's mean, but
scalar still wins and embedding quality remains unacceptable.

The universal direction of the resource readout trades one tensor class's damage
for another's. The joint stimulus also collapses weight and activation energies
into one product, losing their separate roles. A successor should question that
representation and interaction rather than select favorable tensors, tune gates,
or add a quality-directed correction. Separate interacting weight and activation
resource fields are a possible new hypothesis, requiring their own frozen
standalone, disconnected, coupled and domain/chaos controls. No successor is
implemented or established by this outcome.

## Implementation and verification

Added `examples/discover_resources.rs`, a resource accessor in the existing
primitive crate, `experiments/resource_discovery.py`, and a stored f32 resource
scale in the experimental codec. Weight and input coordinate changes cancel
before quantization; the serialized scale alone suffices for inverse decoding.
Both readouts exported complete GGUF fixtures after deletion of original weights,
calibration and scale-source files, reproducing expected decoded tensors exactly.
These are operational serialization tests, not full-model inference evidence.

All 62 Rust and 25 Python tests passed; strict Clippy for the native crates and
`git diff --check` passed. The two owning crates contain 346 and 155 Rust lines
and no external dependencies. No native numerical mechanism was changed for
B03; the new accessor and batch executable expose already-existing state.
Only the protocol was committed in this pass; existing dirty work, prior
receipts and guard settings were preserved.

```bash
cargo build --offline --release --example discover_basis --example discover_local_basis --example discover_resources
cargo test --workspace --offline
cargo clippy --offline -p basis-primitives -p basis-engine --all-targets -- -D warnings
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m experiments.resource_discovery --out artifacts/resource-discovery-new-run
```
