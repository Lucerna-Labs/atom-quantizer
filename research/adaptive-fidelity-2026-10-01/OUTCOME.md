# D04 outcome: native admission achieved; principal comparison failed

Precommit: `33c14a9`, [protocol](PROTOCOL.md). The operational adaptive-precision
function completed its frozen experiment. Both adaptive models now pass every
native gate on fitting and separate selection observations. The principal still
**FAILED** its required PPL comparison with the scalar-adaptive control. The
conditional final stage remains **UNRUN**; no threshold or comparison was changed.

## Implemented function

[build-adaptive](../../docs/ADAPTIVE_PRECISION.md) starts matrices at 4 bits, builds
and exports a complete A22-2 model, and invokes the actual native assessor on both
fitting views. A failure in either view raises that tensor to 6 bits, then 8 bits,
then exact storage. Non-matrix tensors are exact. Passing records remain unchanged.
A failed exact tensor or an execution/integrity error stops the build.

The principal fits its scalar base on interior observations and a rank-two stored
correction on endpoint observations. The control uses the same base and native
policy without a correction. This is a conventional primitive-toolkit composition,
not Atom synthetic emergence. Native equations, thresholds and main defaults are
unchanged. No tensor names were hard-coded into the precision policy.

Reuse checks complete source/calibration/configuration/prefix/record identities
and decoded commitments. It is limited to independent scalar/lossless recipes;
external-file and predictor dependencies cannot be silently reused. Every stage
and native report is retained. Final model files and FIT_ADMITTED status appear
only after all fitting views pass. That status does not imply final model quality.

## Precision progression

| Model | Stage | Fit-view failure union | Archive bytes | Passing records preserved |
|---|---:|---:|---:|---:|
| Corrected adaptive | 4-bit start | 25 | 73,289,998 | — |
| Corrected adaptive | Raise failures to 6 bits | 0 | 83,573,330 | 247 |
| Scalar adaptive | 4-bit start | 50 | 71,840,979 | — |
| Scalar adaptive | Raise failures to 6 bits | 2 | 83,576,761 | 222 |
| Scalar adaptive | Raise remaining failures to 8 bits | 2 | 83,961,027 | 270 |
| Scalar adaptive | Exact fallback | 0 | 84,983,143 | 270 |

The corrected result has 186 matrices at 4 bits, 25 at 6 bits and 61 exact
non-matrix tensors. The scalar result has 161 matrices at 4 bits, 48 at 6 bits,
two exact matrices and 61 exact non-matrix tensors. Its exact matrices are
blk.27.ffn_down.weight and blk.28.ffn_down.weight. At 8 bits their endpoint-fit
output errors were 0.064858 and 0.228998, above the ordinary limit of 0.02,
although their cosines exceeded 0.9998 and their KL scores were tiny. This is an
observed reason to retain independent output-error checks.

Both adaptive results and the uniform 6-bit corrected control passed every
native gate on both selection views. No precision choice was changed after
selection. The initial corrected/scalar decoded models exactly reproduce the
R1 fixed4 and D02 interior-scalar files, respectively.

## Whole-file size and model quality

All new scores are actual 64-chunk CPU llama-perplexity runs on the original
selection text, with four threads and context/batch/microbatch 512. Source,
historical and fixed4 scores were explicitly reused after identity checks.

| Model | Stored bytes | Selection PPL |
|---|---:|---:|
| Source F32 | 539,846,112 | 26.7164 |
| Historical budget120 | 125,809,756 | 30.0986 |
| Fixed4 corrected reference | 73,291,645 | 31.2429 |
| Corrected adaptive | 83,573,330 | 29.3378 |
| Scalar adaptive | 84,983,143 | 29.2825 |
| Uniform 6-bit corrected control | 109,251,501 | 26.9661 |

The corrected adaptive model is 33.57% smaller than historical budget120 and
has 2.53% lower selection PPL. The scalar-adaptive control is 32.45% smaller
and has 2.71% lower selection PPL. The corrected model saves 1,409,813 bytes
against that control but loses the frozen PPL comparison by 0.0553. It therefore
fails D04, despite passing the native and historical-reference comparisons.
Do not turn this into a pass by changing the comparison or treating the small
difference as a tie after seeing the result.

The uniform 6-bit corrected control is 13.16% smaller than historical budget120
and has 10.41% lower selection PPL. This is a separate useful lead and measured
tradeoff point. It was not the admitted D04 principal, has no new final result,
and is not automatically promoted. All selection data here were already used in
research; none of these comparisons substitutes for the required final test.

## Verification and preserved limits

Seven unique new complete archives were built and standalone-exported with source
and fitting-file Python opens denied. The independent audit passed 1,904 tensor
byte/commitment comparisons, all 20 native reports (5,440 tensor decisions),
1,009 unchanged-record comparisons, the escalation rule, archive sizes, all three
new inference logs and three reused inference receipts. Uniform-control settings
and the failed conditional-final decision were checked explicitly. No recurrence
of the original D03 byte mismatch was observed; its cause remains unknown.

The pre-trial Python suite had 56 passing tests, including real native-report
validation and CLI publication. The exact completed-trial source was preserved
before a subsequent resource-cleanup fix: when opening a publication source fails,
the staging descriptor is now closed before temporary-file cleanup. That change
is nonnumerical and alters no trial artifact. A new regression test raises the
current suite to 57 passing tests. [POST_TRIAL_FIX.json](POST_TRIAL_FIX.json) records
the source change; no inference was rerun or claimed for that cleanup.

The new final suffix remains unused: character cut 439133, common excluded prefix
100418 native tokens, suffix 51831 tokens, SHA
`d757b51672161dc9f32eef6a01a54a84839893869842153ade19fb13ab4ed6e3`.
Its separation exceeds D03's used segment plus the declared gap. No new-final
receipt, log or consumption marker exists. A future precommitted study may use
this still-unevaluated segment after rechecking provenance.

This is a working experimental builder and a measured native-admission result,
not native OQ03 integration, compressed-kernel execution or a resident-memory
claim. The complete decoded outputs remain dense F32 GGUFs. The principal's final
quality condition is unproven because selection failed. The broader goal and
full22 programme remain active, and B04 remains UNRUN.

## Evidence and next work

[SUMMARY.JSON](SUMMARY.JSON), [AUDIT.json](AUDIT.json), [FRESHNESS.JSON](FRESHNESS.JSON)
and [INITIAL_MODELS.json](INITIAL_MODELS.json) contain compact receipts. Full stage
archives, native reports, logs and the untouched suffix are under
`artifacts/adaptive-fidelity-2026-10-01/`. The exact trial implementation is in
implementation-trial.tar.gz; later current-source and evidence snapshots preserve
both the cleanup and the original record. Only the protocol was committed.

A successor should keep the failed principal comparison on record and investigate
the actual size/quality frontier or a stronger corrected representation under a
new protocol. The uniform 6-bit result is a promising selection lead, not a proven
final winner. Any promotion still needs independent final evidence and an honest
app-boundary integration claim. Do not reclassify these conventional functions as
native Atom emergence or relax the main fidelity rules.
