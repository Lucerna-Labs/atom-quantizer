# M1 post-trial destination ownership fix

The complete numerical M1 experiment and its independent audit finished before
this change. The current mamba_models CLI can add failure.json to a pre-existing
output directory after mkdir rejects it. Resolving a dangling output link before
mkdir can also create its target. Both were reproduced using temporary controls;
see artifacts/mamba-temporal-2026-10-01/output-guard-before.json. The actual M1
build used a new owned directory, and its results remain valid.

Require atomic creation of the raw absolute destination before the failure
handler owns it. Only a successfully created directory may receive failure.json.
Keep the existing build body unchanged. Predicted validation: existing directories
and dangling links reject without changes; a newly owned failing run retains its
failure receipt. Preserve the frozen audited source and explicitly record the
post-trial source difference. No numerical implementation, recipe, gate or result
changes. Tests and AST comparison of the original build body must pass.
