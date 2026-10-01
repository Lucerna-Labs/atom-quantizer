# S1 outcome: the tested second stage is rejected

Precommit `f68d67e`; historical-metadata correction `77e0546`; loader-validation
precommit `331dc0f`. The exact Composer 1 -> second-stage architecture was executed
on the complete SmolLM2 model. Both candidates passed all four native views
against their intermediate W1 model, but failed fidelity against the original W.
The correction-enabled candidate also lost to the current direct recipe on both
archive size and selection perplexity. Neither gate was relaxed. Final evaluation
is UNRUN because selection failed; no fresh final segment was consumed.

## Complete-model measurements

All PPL values below use the same existing validation corpus,64chunks,context512,
4CPUthreads and pinned llama.cpp runtime. Four control results were identity- and
log-verified before reuse; the two sequential candidates were newly evaluated.
These are selection results, not fresh final evidence or statistical-significance
claims. Archive bytes include every record and decoder state.

| Model | Complete stored bytes | Selection PPL |
|---|---:|---:|
| Original F32 GGUF | 539,846,112 | 26.7164 |
| Exact Composer 1 W1 | 125,809,756 | 30.0986 |
| Direct six-bit scalar control | 109,317,095 | 27.1168 |
| Current direct functional recipe | 109,251,501 | 26.9661 |
| W1 -> scalar second stage | 112,135,705 | 30.6050 |
| W1 -> functional second stage | 110,271,260 | 30.5199 |

The functional sequence is0.93% larger and has13.18% higher PPL than the direct
recipe on this selection segment. Relative to W1, its archive is12.35% smaller
but PPL is1.40% higher, exceeding the precommitted1% allowance. It has slightly
lower measured PPL than the sequential scalar control; that does not establish
significance or rescue the failed product gates.

## What actually ran

The main executable freshly decoded the pinned125,809,756-byte historical OQ
artifact. W1 matched SHA c5fb3e2c4e516823058a730d9234e267f0a4e1d94f061449828826c2ba02b7f5.
Every one of272 W1 arrays was installed in the actual HF model and forward-converted
back exactly, including Q/K row permutation and tied embedding state. Four actual
forward-pass captures used the same training/selection documents and token
positions as the direct controls:256/64 samples per operator,212 operators,
context512. All211 linear input arrays changed from W as measured; embedding
lookup indices remained identical. No original-W observation header was rebound.

The scalar6->8->exact ladder failed3 tensors initially,2 at the next stage and0
at the last. It retained269 and270 previously passing records byte-for-byte.
The final scalar model has208 six-bit matrices,one eight-bit matrix,two exact
matrices and61 exact one-dimensional tensors. The functional model passed its
W1 fitting views at six bits for all211 matrices, with rank-two factors;61
one-dimensional tensors remain exact. Both final archives exported natively with
all272 tensor arrays equal to their stored commitments. Every stage also had a
standalone export with fitting-file opens denied.

Each final model was separately assessed against W and W1 with endpoint/interior
fit/selection views. Both have zero failures in all four W1 views. Against W,
both have one cosine failure on endpoint views and an additional output-MSE
failure on interior views. The affected attention-output tensors are
blk.0.attn_output.weight (cosine) and blk.28.attn_output.weight (output MSE).
The actual gates, protected thresholds and source identities remain unchanged.

The independent audit passed four stored stages,1,088 reconstructed arrays and
24 native reports/6,528 tensor decisions. It independently recalculated the
failure unions, escalation, retained records, all comparisons and inference
identities/logs. The weight-error decomposition also verifies E_total=E_first+
E_second with its measured cross term. For the functional candidate, normalized
weight error rises from0.00433312 for W1 to0.00496167 for W2 against W. The cross
term is a small negative contribution (-26.14 versus2,833.04 second-stage error
energy); cancellation did not remove the added error. This is weight-space
accounting, separate from the actual model PPL results.

## Preserved failures and validated loader improvement

The first process stopped before candidate fitting because the September6
endpoint manifest lacked the newer sampled-position field. That run, its one
W1 capture, failure.json and original implementation snapshot remain. D02's
position-bearing endpoint replay payloads were verified identical to the original
captures before using their metadata. The corrected trial used a new directory.

Review reproduced a loader defect with a small real Llama: differing attention
head counts can share all parameter shapes and roundtrip their stored arrays,
yet compute different logits (measured maximum difference0.09108). The loader now
checks12 standard architecture fields, supported activation/bias/RoPE state,
complete mapping, dimensions, F32 types, finiteness and tied aliases before changing
weights. Missing/different/unsupported architecture state fails closed. All91
Python tests pass, including wrong-head and wrong-norm controls and exact logits
under a matching installation.

The real trial's original/W1 headers and all12 architecture settings were checked
and match. After the hardening, original-W installation still reproduces all272
checkpoint parameters, and a fresh W1 endpoint selection capture is byte-identical
to the trial payload (SHA d52abd319ccf0f2e5cff4ecf80e77670e7d8a403a52c72733bd29dc479312e34).
Thus this input-validation fix does not change the measured numerical trial.
The audit passed before hardening; only capture.py and gguf_weights.py differ
from the frozen trial source. The exact audited source remains in
implementation-corrected.tar.gz. Historical identity checks must not be weakened
to pretend the current source is that snapshot.

## Operational scope and next work

The experimental collector now supports actual decoded-model calibration:

```bash
.venv/bin/python -m experiments.full22.capture --split selection \
  --source PATH_TO_MATCHING_F32_GGUF --load-gguf-weights \
  --out NEW_CALIBRATION.acal
```

Its supported scope is standard Llama with complete matching metadata and the
matching local checkpoint/tokenizer. This trial used only SmolLM2. Run the whole
comparison with `python -m experiments.sequential_functional --out NEW_DIRECTORY`
in the existing environment. All destinations must be new.

The normal main executable and the accepted direct recipe remain unchanged.
Fitting uses Python; decoding and fidelity checks used the real native executable.
Decoded GGUFs are dense F32,539,846,112 bytes each. No packed-matmul, resident-memory
or throughput improvement follows from these archive sizes. Build intermediates,
calibrations and logs are retained separately from each standalone output.

Lesson: acceptance relative to an intermediate model does not imply acceptance
relative to the original. A future sequential candidate needs an explicit
original-reference error budget and must still beat the direct recipe. This
one failed composition is not a proof that all sequential architectures fail.

The broader goal remains active. The full22 coverage audit retains all22 original
requirements and separates704 verified tensor-screen artifacts from the remaining
whole-model, recurrent, attention, task-fidelity, uncertainty and packed-runtime
work. S1 closes this declared sequential comparison, including its negative result.
The earlier D03 intermittent cause remains open; B04 remains UNRUN.

Evidence: SUMMARY.JSON, AUDIT.json, MANIFEST.JSON, ARCHITECTURE.json,
POST_TRIAL_FIX.json; full data and logs under
artifacts/sequential-functional-2026-10-01/{run,run-corrected,post-hardening}/.
Original and corrected source snapshots, the negative loader probe and a final
source snapshot are retained. Existing dirty work and all failed data are preserved.
