# B01 outcome — 2026-09-30

The declared experiment is complete. **All principal candidates FAILED the
frozen benefit gate and were rejected.** The broader automatic app-improvement
goal remains active. No candidate was promoted to the main codec and no
full-model quality, runtime-speed or physical-quantum claim is made.

The protocol was committed as `567908a` before implementation. The two owning
Rust crates execute real primitive compositions, and the emitted transport is
used by the experimental codec. The source hash manifest records the exact
implementation and binary used. Existing user changes were preserved; only the
new protocol was committed during this pass.

## Measurements

All 624 declared native executions produced matrices. All 1,280 unique record
conditions completed with `MEASURED` receipts: 1,248 candidates and 32 conventional
references, over eight real SmolLM2 tensors, two orders, thirteen arms, three
seeds and two code widths. No declared case is missing or failed execution.
Passing execution is separate from passing the quality hypothesis.

Mean selection-output NMSE, averaged across the declared tensors and seeds:

| Width | Plain scalar control | Signed H32 control | Full order 0 | Full order 1 |
|---|---:|---:|---:|---:|
| 3 bits | 0.01165559 | 0.01569666 | 0.01513478 | 0.01513426 |
| 4 bits | 0.00310705 | 0.00418061 | 0.00388719 | 0.00399815 |

Full candidates had about 25.1–29.8% more mean error than the plain scalar
control. They improved on the signed-H32 reference, but that was insufficient:
the protocol required beating both references and the relevant ablations.
Complete raw records were 4.81% larger at 3 bits and 3.75% larger at 4 bits than
the smaller reference. Matrix state and metadata were charged, not estimated.

| Gate | Status | Evidence |
|---|---|---|
| G1 native invariants | MEASURED | Worst orthogonality error 3.33e-15; finite states and resource/occupancy bounds passed |
| G2 stored reconstruction | MEASURED | Worst unquantized reconstruction NMSE 2.165e-14; all saved records decoded |
| G3 benefit | FAILED | All four full order/width combinations lost to scalar and some ablations; order 1 at 3 bits also exceeded the per-tensor H32 regression limit |
| G4 contribution | MEASURED | Every declared domain/base removal changed the matrix and result in every seed on at least one tensor; this does not establish useful contribution |
| G5 complete evidence | MEASURED | All 1,280 unique conditions retained; input and implementation hashes unchanged |

An independent post-run audit reread all 1,280 artifacts, verified hashes, checked
their actual raw and compressed lengths, decoded finite values, and matched all
1,248 stored matrices to the native matrix files. See [AUDIT.json](AUDIT.json),
[SUMMARY.JSON](SUMMARY.JSON), [MANIFEST.JSON](MANIFEST.JSON) and the compact
[1,280 result receipts](RECORDS.json). Large payloads remain in
`artifacts/basis-discovery-2026-09-30/`; the process exited successfully.

## What the failure changes

The no-H and no-base controls consistently outperform the full candidate on mean
error. In this trial, the initial global mixing was harmful relative to retaining
the original coordinates. Both complex orders also lose to their no-chaos
controls. Small gains against one Hadamard reference do not rescue these failures.
Chemistry and activity memory materially affect the dynamics, but their useful
contribution is inconsistent across orders and widths.

The next hypothesis should investigate more local, data-coupled transport and
how the initial stimulus represents structure, with the required domains and
chaos controls retained. B01 supplies no justification for selecting a per-tensor
winner, removing required domains from a principal candidate, changing these
gates, or fitting the component against reconstruction error. Any successor
needs a new frozen protocol. Full-model evaluation stays conditional on a
candidate passing the declared screen; the older full22 scope is still open.

## Reproduction and implementation checks

```bash
cargo build --offline --release --example discover_basis
cargo test --offline --workspace
cargo clippy --offline -p basis-primitives -p basis-engine --all-targets -- -D warnings
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m experiments.basis_discovery --out artifacts/basis-discovery-new-run
```

The workspace's 60 Rust tests and 20 Python tests passed. The five new native
tests include all declared composition masks and seeds, deterministic repetition,
local numerical fixtures and fail-closed inputs. The three new Python boundary
tests invoke the actual native executable, preserve linear-operator coordinates,
reject invalid matrices, and decode a saved record after removing its matrix
source file. Strict Clippy passed for the two new crates. They contain 293 and
99 Rust lines respectively, have no external dependencies, and the orchestrator
entry point does not inspect content or scores. The surrounding codec/evaluator
remain conventional software; their presence does not establish that the whole
application is emergent.
