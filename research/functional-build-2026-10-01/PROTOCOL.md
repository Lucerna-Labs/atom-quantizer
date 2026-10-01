# I2: operational functional build workflow in the main app

Precommit status: UNRUN. I1 was progress: the main Rust app now consumes the
validated representation natively. I2 connects real fitting to a normal app
command and independently enforces native acceptance before publication.

## User-facing contract

Add `atom-quantizer build-functional SOURCE.gguf --base-calibration BASE.acal
--residual-calibration RESIDUAL.acal --out NEW.a22`, with optional new work-directory
and explicit Python/engine-directory overrides. Default fitting uses the existing
project Python environment and engine. Say so accurately: this is real Python
fitting orchestrated by the app, not native SVD fitting. Native consumption stays
independent and the old default encoder stays unchanged.

Fix the selected D05 recipe: start6, rank2 activation correction, 6 ->8 ->exact
escalation, and both distinct fitting observation files as native gate views.
Do not expose a shortcut that loosens native limits or skips a view. No hard-coded
tensor names, synthetic substitution or copied prebuilt artifact in the real run.

Pass arguments directly without a shell. Invoke the fitting engine with explicit
module search context. Preserve all work/logs, especially failures. A successful
backend exit or claimed FIT_ADMITTED state is not authoritative by itself.

The main app must validate source/calibration identities and the declared profile,
decode the archive with its own native decoder, and independently run its native
fidelity rules on both supplied views. Recheck input/backend/executable/archive
identities before publication. Publish only a completely checked archive to a new
destination, preserving existing files and dangling links. Keep a receipt and
actual native reports in the work directory. Backend, verification, input-change,
fidelity or publication errors must not be reported as successful builds.

## Required verification

Preserve current source/release first. Exercise the real command with a disposable
model and actual fitter/native assessor. Test required/duplicate/invalid options,
missing backend, existing destinations, wrong-source calibration and publication
preservation. Explicit fault fixtures may test false backend success, invalid
archive/provenance or failing native gates; do not present those as real fitting.

Independently enforce profile metadata and supported stored configuration, rather
than trusting only a backend status string. Ensure the decoded file used for
native checks came from the app's native decoder, not a backend-supplied GGUF.
Keep the separate roles/paths intact and require distinct observation identities.

Run the main command end-to-end on the actual SmolLM2 source and D05 observations.
Require a newly produced complete archive, all272 native decoded commitments,
both complete native fitting views, and exact archive/decoded equality to D05's
selected model. The existing PPL evidence may be reused only for matching decoded
hashes. No new holdout is needed for identical weights; do not claim a new inference.

Run relevant Rust/Python tests, old-format/quantizer regression and release-path
checks. Build the verified implementation into the normal release only after
preserving its predecessor. Audit commands, interpreter/engine/native executable
identities, artifact bytes, reports and publication from actual files/logs.

Success establishes the real app build/verify/decode workflow for this profile.
It does not establish native fitting, packed inference/RAM savings, all-family
native A22 support, broader-model quality or completion of the full22 programme.
Keep prior failures, used-final boundaries and the unresolved D03 concern intact.
