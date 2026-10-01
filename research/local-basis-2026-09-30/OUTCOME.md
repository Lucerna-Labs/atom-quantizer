# B02 outcome — 2026-09-30

**All four principal order/width candidates FAILED the frozen benefit gate.**
The bounded experiment is complete; the automatic architecture-improvement goal
remains active. Nothing was promoted to the main codec or full-model evaluation.

Protocol precommit: `b39119c`. B02 adds an explicit local-coupling entry point to
the existing two native component crates. Relative phase reversal replaces the
initial global H32 operation, while fit-activation correlations modulate local
unitary transport. Chemistry, activity memory, information potential, harmonic
dynamics and both chaos processes remain active in the principal candidate.
Uniform and shuffled coupling are controls. No quality objective enters the
native component.

## Complete measured comparison

All 720 native executions and all 1,472 unique scored records completed: 1,440
candidate records and 32 references. Every required control and both orders ran.
No case failed execution or serialization. Mean selection-output NMSE:

| Width | Plain scalar | Signed H32 | Full order 0 | Full order 1 |
|---|---:|---:|---:|---:|
| 3 bits | 0.01165559 | 0.01569666 | 0.01167794 | 0.01157843 |
| 4 bits | 0.00310705 | 0.00418061 | 0.00314811 | 0.00311178 |

Full order 1 at 3 bits lowers mean error by 0.662% against scalar, but fails other
requirements. Its final-layer FFN-down tensor error increases 64.38%, and its mean
error is higher than the no-chaos and shuffled-coupling controls. Full order 0
at 3 bits increases mean error 0.192%. At 4 bits, orders 0/1 increase mean error
1.322%/0.152%. The final-layer FFN-down tensor increases 85.79%/86.86% at 4 bits.
All full candidates exceed the per-tensor regression limit. Complete raw records
are 4.809% larger at 3 bits and 3.744% larger at 4 bits than the smaller reference.

| Gate | Status | Evidence |
|---|---|---|
| G1 native mechanics | MEASURED | Worst orthogonality error 3.89e-15; norm, finite-state and resource bounds passed |
| G2 stored reconstruction | MEASURED | Worst unquantized reconstruction NMSE 1.642e-14; deterministic blind decode passed |
| G3 useful benefit | FAILED | All four candidates fail controls and per-tensor regression checks |
| G4 contributions | MEASURED | Required removals and coupling controls alter outputs in every seed; beneficial contribution is not established |
| G5 evidence | MEASURED | All 1,472 distinct cases present; frozen hashes unchanged |

Independent audit reloaded every saved artifact, verified its SHA-256 and actual
raw/compressed lengths, decoded finite values and matched all 1,440 stored native
matrices. Compact durable receipts are [RECORDS.json](RECORDS.json),
[SUMMARY.JSON](SUMMARY.JSON), [MANIFEST.JSON](MANIFEST.JSON) and
[AUDIT.json](AUDIT.json). Payloads and a source/binary snapshot remain under
`artifacts/local-basis-2026-09-30/`. The run and audit exited successfully.

## Interpretation and next question

Local coupling closes most of B01's average-error gap, but that is not an accepted
app improvement. Shuffled coupling beats the aligned full stack in every mean
comparison, so this test does not establish a benefit from alignment to folded
activation correlations. The full stack also loses to no-chaos on every mean
comparison. The large final-layer regression remains even in base-only controls:
about 55.3% at 3 bits and 67.1% at 4 bits. It cannot be blamed solely on chaos.

The next investigation should separate harmful base transport from the way
correlations are folded across column blocks, and question whether these dynamics
should act directly on coordinates at all. A different coupling or function
family needs a new protocol; do not rescue B02 by selecting favorable tensors,
discarding required domains or modifying its frozen limits. The mean gain in one
case is insufficient for promotion. Full-model and inference-runtime claims
remain unmeasured, and the broader full22 work is still incomplete.

## Operational checks and preservation

The current code passes 62 Rust tests and 22 Python tests. Strict Clippy passes for
both native crates (342 and 155 Rust lines; no external dependencies). Its initial
range-loop lint was corrected before the frozen scientific run; no numerical
gate or parameter was changed. The new boundary test creates and exports a
complete GGUF fixture after deleting its original weights, calibration and basis
source; decoded tensors exactly match saved-record reconstruction. This is a
fixture export test, not LLM inference evidence.

All 624 B01 native matrices were regenerated bit-identically after extending the
shared crates: [B01_REGRESSION.json](B01_REGRESSION.json). B01's original source and
binary snapshot is in its own artifact directory. Historical receipts are intact.
The observer summary helper was parameterized for B02's extra controls; historical
B01 summaries and gates were not rewritten. Existing user work and guard settings
were preserved. Only the B02 protocol was committed in this pass.

```bash
cargo build --offline --release --example discover_basis --example discover_local_basis
cargo test --offline --workspace
cargo clippy --offline -p basis-primitives -p basis-engine --all-targets -- -D warnings
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
.venv/bin/python -m experiments.local_basis_discovery --out artifacts/local-basis-new-run
```
