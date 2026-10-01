# D05: native-admitted six-bit quality/size candidates

Precommit status: UNRUN. D04 was progress but its lower-footprint corrected
principal FAILED its matched-control comparison. That outcome and unused final
segment remain intact. D05 tests a different, higher-quality operating region
suggested by D04's uniform6 control; it does not relabel D04 or prove that a
particular correction geometry is uniquely best.

## Fixed candidate set and implementation

Extend the operational adaptive builder with an explicit starting precision and
correction geometry. Preserve existing start4/activation defaults. For this study,
start all matrices at6 bits and use the fixed6 ->8 ->exact escalation rule, driven
only by union-of-failures across endpoint/interior **fit** observations. Passing
records remain byte-identical. Non-matrices remain exact. Execution, identity,
commitment or exact-tier failures stop fitting; native thresholds are unchanged.

Use three global settings, all with the same interior seed0 base observations:
1. Scalar: no correction.
2. Diagonal: rank2 existing diagonal-weighted correction using endpoint observations.
3. Activation: rank2 functional-output correction using endpoint observations.

No rank search, manually selected tensor names, selection-data fitting or change
of geometry by tensor. A global geometry is selected only after all candidates
have been built and evaluated. At most three stages per setting, nine new stage
archives total. Use the existing source and four calibration captures from N1,
the native assessor SHA628c4d10a9513379c6a37b14d617469bffd5333de81da8fffe642cf4324417bd,
and A22-2 decoded commitments. This is conventional primitive-toolkit composition,
not Atom synthetic emergence. Keep the native component and main defaults intact.

Every stage must receive both complete native fit assessments. Every final
candidate receives both complete native selection assessments. Both kinds of
observations must pass before that candidate is eligible. Do not adapt afterward.

## Global selection and final validation

Measure all three final candidates on the existing 64-chunk selection corpus with
the frozen D01 CPU llama-perplexity runtime, four threads, context/batch/microbatch
512 and no GPU. Include unchanged source F32 and historical budget120 references,
reusing their scores only after exact identity/command/log/runtime checks.

Eligibility requires every native fit/selection gate passing, complete archive
bytes <=90% of historical budget120, and selection PPL <=97% of historical
budget120. This requires a material gain in both stored size and measured model
quality. Among eligible settings select lowest PPL, then lower bytes, then the
fixed order scalar/diagonal/activation. This is one whole-model setting, not
per-tensor PPL selection. Keep all measured tradeoffs and failed candidates.

If none qualifies, final remains UNRUN. Otherwise freeze the selection and run
all three candidates plus source and historical references on D04's still-unused
new-final suffix. The selected setting passes final only if its PPL <=97% of
historical on that segment and its byte cap still holds. Do not reselect from
final results if another geometry wins there. Report that comparison honestly;
the final gate establishes improvement over the product reference, not universal
optimality or a unique advantage of the selected correction mechanism.

Any required execution failure makes the affected stage inconclusive. All planned
independent cases must be retained. At most three new selection and five new final
inference runs. No new data capture or new dataset is required.

## Holdout and verification

Recheck that D04 did not consume the suffix SHA
d757b51672161dc9f32eef6a01a54a84839893869842153ade19fb13ab4ed6e3.
Reproduce its native-token proof: character439133, excluded common prefix100418
tokens, suffix51831; prior-use separation exceeds84241. Preserve the source text
and proof and record consumption if final runs. Never call used text untouched.

Test default compatibility, starting-tier/rank-geometry validation, bounded
escalation, actual CLI wiring, stable passing-record reuse and publication.
Each model must independently export every committed tensor without fitting-file
access. Audit source/calibration/configuration identities, all native decisions,
precision transitions, tensor bytes, inference logs, actual archive size and the
global selection/final rule. Preserve exact implementation snapshots and failures.

The original D03 byte-audit cause stays unresolved; A22-2 must reject any changed
reconstruction. A successful experiment is not by itself native OQ03 integration,
packed inference, a resident-memory claim, or completion of the wider full22 work.
