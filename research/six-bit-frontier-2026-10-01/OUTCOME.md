# D05 outcome: selected six-bit setting passed final validation

Precommit: `99a6bf9`, [protocol](PROTOCOL.md). The previous D04 turn was progress,
but its lower-footprint corrected principal failed its scalar-control comparison.
That outcome remains unchanged. D05 tested a distinct higher-quality region and
selected one global method before its first final evaluation.

The selected **six-activation** setting passed all native fit/selection gates,
the frozen size/selection criteria and the independent final test. It uses
**109,251,501 stored bytes** and scores **17.4507 final PPL**, compared with
**125,809,756 bytes /19.0007 PPL** for historical budget120 on the same segment.
That is **13.16% fewer stored bytes and 8.16% lower measured perplexity**.

## Operational implementation

The public adaptive builder now accepts --start-bits 4/6/8 and global correction
geometry activation/diagonal. Old start4/activation defaults are preserved.
Source-bound input roles remain separate; scalar mode omits the correction.
A geometry request without correction observations fails. The fixed ladder raises
only tensors failing actual native fitting checks. Passing records and their
commitments stay unchanged, and exact/identity/execution failures stop the build.

This is conventional primitive-toolkit composition. It does not modify the native
Atom component or claim synthetic emergence. [The verified recipe](../../docs/SIX_BIT_RECIPE.md)
provides a usable CLI command and the actual model/archive paths.

## Candidates and native checks

All settings started at six bits and used the same interior base observations.
The two corrections had rank2 and used endpoint observations. There was no
per-tensor geometry choice, selection-data fitting or rank search.

- Scalar: two matrices failed at six and eight bits, so the fixed rule used exact
  storage for blk.27.ffn_down.weight and blk.28.ffn_down.weight. Final archive:
  109,317,095 bytes. Other matrices stayed at six bits.
- Diagonal correction: every matrix passed both fitting views at six bits.
  Final archive: 109,255,265 bytes.
- Activation correction: every matrix passed both fitting views at six bits.
  Final archive: 109,251,501 bytes. It is byte-identical, archive and decoded GGUF,
  to D04's uniform6 control; rebuilding did not change the numerical result.

All three final candidates passed every tensor on both native selection views.
The assessor and all limits stayed unchanged. Five unique stage archives were
built in total, with 16 complete native reports covering 4,352 tensor decisions.
There were 540 unchanged-record comparisons across the scalar escalation steps.

## Selection and final inference

| Model | Complete archive bytes | Selection PPL | New final PPL |
|---|---:|---:|---:|
| source-f32 | 539,846,112 | 26.7164 | 17.4036 |
| historical-budget120 | 125,809,756 | 30.0986 | 19.0007 |
| six-scalar | 109,317,095 | 27.1168 | 17.6123 |
| six-diagonal | 109,255,265 | 26.9946 | 17.4759 |
| six-activation | 109,251,501 | 26.9661 | 17.4507 |

All three candidates met the precommitted selection requirements: every native
view passed, bytes <=90% of historical, and selection PPL <=97% of historical.
Activation had the lowest selection PPL and was selected before opening final.
Its final result also led the two other tested settings, but no final reselection
was allowed or performed. The selected result is 0.27% above the source F32 PPL
on this final segment; that is a corpus/model-specific measurement, not a general
quality-equivalence guarantee.

The final gate required the frozen selected setting to retain >=10% stored-size
reduction and >=3% PPL improvement over historical. Both passed. All candidate
comparisons remain in the record; this does not prove that activation geometry
is universally optimal, nor does it turn D04's failed comparison into a pass.

Eight new real CPU inference runs completed: three selection and five final.
The two source/historical selection references were reused after identity checks.
Every run used 64 chunks, four threads, context/batch/microbatch 512, no GPU, and
the unchanged llama-perplexity executable/shared-library hashes from D01.

## Integrity and tests

The independent audit passed all 1,360 exported tensor byte/commitment comparisons,
16 native reports, the fixed starting/escalation rules, stable passing records,
actual archive sizes, all new/reused inference logs and the pre-final global
selection/final decision. Every model exported independently with fitting-file
Python opens denied. No original-D03-style byte mismatch recurred in these checks;
the old failure's cause remains unresolved and A22-2 continues to reject changed
reconstruction.

All 61 Python tests passed, including real CLI start-six/diagonal wiring, explicit
parameter validation, eligibility/tie rules and a check that final evaluation
never reselects a better unchosen candidate. Existing adaptive/default tests still
pass. An auditor-only bracket typo initially prevented its launch; it was fixed
before validation and its failed log remains in artifacts/d05-audit.log.

Exact trial sources were preserved before a small post-trial input-validation
fix: explicit empty/invalid geometry values no longer silently select the default.
Valid configurations and all measured artifacts remain unchanged. The final test
suite includes that case. See POST_TRIAL_FIX.json; use the saved trial sources for
exact frozen-state audit replay rather than weakening identity checks.

## Data-use boundary and remaining integration

The final suffix is now **consumed**. SHA:
`d757b51672161dc9f32eef6a01a54a84839893869842153ade19fb13ab4ed6e3`.
Its native proof retains character cut 439133, common excluded prefix 100418 and
suffix 51831 tokens, beyond the 84241 minimum separation from earlier evaluations.
D05 used the first 64 chunks (at most 32768 input tokens) for five models. Final-use
records exist in the D05 directory and alongside D04's formerly unused suffix.
Do not reuse this segment as untouched data for changed models.

This is a verified experimental A22-2 artifact/quality improvement on SmolLM2-135M
and the stated observations/corpora. Its decoder reconstructs a dense F32 GGUF;
packed-kernel speed and resident-memory savings have not been measured. Native
OQ03 representation/integration, wider-model generalization and the broader
full22 programme remain separate outstanding work. Main Rust defaults and the
incumbent release binary were preserved. B04 remains UNRUN.

## Evidence

SUMMARY.JSON, AUDIT.json, MANIFEST.JSON, RUNTIME.JSON, FRESHNESS.JSON, FINAL_USE.json
and ACTIVATION_REPLAY.json are compact source-bound receipts. Full stage models,
fit/selection reports, PPL logs and results are in
`artifacts/six-bit-frontier-2026-10-01/`. Source/evidence snapshots preserve the
exact trial and current implementation. Only the D05 protocol was committed;
prior dirty work and all earlier successes/failures remain available.
