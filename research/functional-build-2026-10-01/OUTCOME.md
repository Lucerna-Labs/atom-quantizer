# I2 outcome: operational functional builds from the main app

Precommit: `b124e38`, [protocol](PROTOCOL.md). The previous I1 turn was progress
through native consumption. I2 completes the user-facing build/verify/decode
workflow for the validated functional-six profile. The larger research objective
is not reduced to this one successful integration.

## Main command and authority

`atom-quantizer build-functional SOURCE.gguf --base-calibration BASE.acal
--residual-calibration RESIDUAL.acal --out NEW.a22` runs real fitting with the
existing Python engine. Defaults use the checkout's environment; explicit Python,
engine and new work-directory options are available. The profile is fixed at
start-six, rank-two activation correction, both fitting views and the unchanged
native thresholds. No shell is used to pass arguments, and module context is
explicit. [Usage](../../docs/FUNCTIONAL_BUILD.md).

The Rust app does not trust a backend success/status message. It validates
source/calibration identities and stored profile/configuration, reconstructs with
its own native A22-2 decoder, and independently invokes its actual fidelity rules
on both requested views. Those checks use the native reconstruction, not the
backend's supplied GGUF. Input, interpreter, executable and fitting-source hashes
are rechecked before copying/checksumming a staged archive and publishing to a new
path. Work and failures remain inspectable; existing files/links are preserved.

Fitting is explicitly Python. Final validation and archive consumption are native.
This is an operational engine orchestration, not a claim of native SVD fitting,
OQ03 representation or packed inference. The old default encoder remains unchanged.

## Real end-to-end result

The main command fitted the real SmolLM2 source using the D05 input roles. It
created a new complete archive, then independently passed all 272 tensors in each
of the two parent-native checks. Archive and native-exported GGUF exactly match
the D05 selected files:

- Archive: 109,251,501 bytes, SHA
  `27d36037a432c5cefeeb0c1f62505377f469365027e00372d30b1840c4608a83`.
- Decoded GGUF SHA:
  `4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8`.
- Output: `artifacts/functional-build-2026-10-01/functional.a22`.
- Work/proof: `artifacts/functional-build-2026-10-01/fit-work/`.

Independent auditing checked all 272 tensor byte arrays, 544 parent-native decisions,
source/backend/executable identities and that verification preceded publication.
Standalone native export with Python unavailable reproduced the same GGUF. The
validated D05 size/PPL gain applies to identical weights; no new inference run
or untouched holdout consumption is claimed.

## Failure tests and compatibility

All 83 Python tests and 71 Rust tests passed. Twelve main-build integration tests
include an actual fitter run and explicitly labeled fault backends. False success
without an archive, wrong provenance/profile, bad decoded commitments, native
fidelity rejection, changed fitting code, duplicate observations, invalid/missing
arguments/interpreter and destination creation during fitting all fail correctly.
Those fault fixtures do not stand in for the real full-model fitting run.

Existing and dangling destinations are preserved. A backend may finish while its
model still fails independent native checks; that result is not published. Logs
and failed work remain. The verified receipt records pre-publication acceptance,
and the PUBLISHED receipt identifies completed output. A secondary receipt-write
failure after publication is reported explicitly as such, not as a silent success
or a claim that no archive exists.

Six earlier quantization cases retained exact OQ bytes, decoded GGUFs and scores.
CUDA feature compilation passes; CPU paths were executed. Ordinary Clippy still
shows the same six pre-existing GGUF style warnings, with no new implementation
warning. Changed-file formatting and diff checks pass; no globally green strict
Clippy/workspace-format claim is made.

## Release path and reproducibility

After preserving a hashed predecessor, the normal release path was updated to
the tested executable. A real build smoke through that normal path passed. The
local package still reports 0.2.4; no remote/tagged release was published.

The current executable is `target/release/atom-quantizer`, SHA
d45eaa37c05ae4d08001e7b08ab3d90a0cfe9c970e03c0f06351558d6c9a400b.
The old I1 executable remains at
`artifacts/functional-build-2026-10-01/atom-quantizer-before-release`, with its
hash/size in BACKUP.json. Source and both before/current binaries are archived.
Only the protocol was committed; earlier dirty work and model data are preserved.

Sources: src/build_functional.rs, the main command dispatch, and native A22 profile
checks. No fitting numerical implementation or native Atom primitive changed.
README, the six-bit recipe and docs/FUNCTIONAL_BUILD.md expose the actual workflow.
Full logs, native reports, checksummed request/verification/publication receipts,
regression outputs and integration.patch are under the I2 artifact directory.
SUMMARY.JSON, AUDIT.json, RELEASE.json and OQ_REGRESSION.json are compact evidence.

## Remaining objective

This closes the tested main-app workflow for the new primitive-toolkit recipe.
The original full22 protocol still includes wider candidate/composition comparisons,
sequential W→W1→W2 experiments, recurrent-model trajectories, selection uncertainty,
text/code/mathematics teacher-fidelity checks and actual packed runtime evidence.
Those requirements must be inspected and completed at their stated boundaries;
this one successful recipe does not complete them. The original D03 intermittent
cause stays open, with A22-2 commitments containing changed reconstructions.
Previously used final segments remain used, and B04 remains UNRUN.

The broad goal therefore remains active. The next pass should audit and execute
the remaining concrete full22 requirements rather than keep treating the now
implemented build/verify/decode workflow as an unimplemented integration task.
