# S1: exact Composer 1 followed by functional compression

Precommitted before implementation or candidate outcomes. This executes the
sequential W -> W1 -> W2 requirement in the original full22 protocol. The preceding
I2 milestone completed the real app workflow; it did not complete this experiment.
No incumbent encoder, floor, source archive, failed trial or reference folder changes.

## Mechanism and boundary

W is the fixed SmolLM2 F32 source (SHA53f86c219b099fa84cd005f1b22572f6c4e53f8ba242fb8074f600712abe9c5b).
W1 is a fresh native decode of the exact historical Composer 1 budget120.oq:
125,809,756 bytes, SHA6d8e7d7cf6c734866751630a96be7315133a04c1852bc1fdef1b6e1686e3fde1.
Decoded W1 must equal c5fb3e2c4e516823058a730d9234e267f0a4e1d94f061449828826c2ba02b7f5.

The second stage uses the catalog's prescribed mathematical toolkit: fixed scalar
quantization and rank-two functional residual projection. This is a conventional
composition experiment, not a new synthetic-emergence Atom primitive. The fixed
order is Composer 1 then the experimental compressor; direct compression of W is
its control. W1 cannot reconstruct W's original errors merely from its own weights.
The hypothesis is that structure created by Composer 1 might reduce second-stage
stored cost enough to compensate for cumulative error. Its negation is equally
informative: accumulated errors or stored correction costs may remove all benefit.

Install every decoded W1 tensor into the existing HF architecture, invert the exact
Q/K row permutation, preserve tied embeddings, and verify all 272 forward-converted
parameter arrays byte-for-byte. Capture actual W1 forward passes, not source-W
activations with a replaced header. Use the same training/selection documents,
context512 and sample budgets256/64 as D05, both endpoints and stratified seed0.
Both fitting views drive only the pre-existing 6->8->exact admission ladder.

Two complete new candidates: sequential-scalar (no correction) and
sequential-activation (rank2 correction). Both use W1 interior inputs for scalar
fitting; only the latter uses W1 endpoint inputs for residual fitting. Other
settings equal D05. All stages, failures, complete archives and decoded commitments
are retained. Decode final candidates independently with the native main app and
compare all 272 arrays. Count only the complete standalone W2 archive as inference
storage; report the separate retained build intermediates and dense decoded state.

## Fixed comparisons and predictions

Controls: original W; exact W1; D05 direct-six-scalar and direct-six-activation.
No new numerical configuration is selected from validation outcomes in S1.

G0 — MEASURED requires native W1 export exactly matching the pinned hash; all272
parameters verified on installation; original-W installation reproduces its values.
Prediction: pass, based on the invertible row permutation and preserved OQ decoder.

G1 — MEASURED requires four new actual W1 captures (fit/selection x endpoint/interior),
all212 operators and all272 verified tensors with W1 identity, finite observations,
matching document/token positions and distinct contents. Prediction: pass.

G2 — Both candidates must finish the fixed ladder on both W1 fitting views.
Independently assess both final models against W1 and W using each one's own
endpoint/interior fit/selection captures. Report every failure against both
references; no W-based repair/escalation after these results. Prediction: W1 fit
admission passes; W-based admission may fail from cumulative errors. Production
floors remain KL<=1/.25, cosine>=.99/.995 and output NMSE<=.02/.005 as implemented.

G3 — Practical recompression benefit requires sequential-activation to pass every
native W and W1 view, archive<=0.90*W1 archive, PPL<=1.01*W1, archive<=1.05*sequential
scalar, and PPL<=sequential scalar. Prediction: uncertain; lower incremental error
need not preserve the original model. These are complete-model measurements.

G4 — Improvement over the current app recipe additionally requires archive<=0.95*
direct-six-activation and PPL<=1.01*direct-six-activation. Prediction: likely FAILED
because W1 has already lost information. This is the principal promotion gate;
passing only G3 cannot replace the current direct recipe. Report scalar comparison
and all controls regardless of the benefit outcome.

Use existing validation selection.txt,64chunks,context512,4CPUthreads,ngl0, the
pinned llama.cpp runtime. Reuse control inference only after checking decoded,
corpus,runtime,log identities and successful64chunk receipt. Time is diagnostic,
not a concurrent throughput benchmark. Run both new candidates even if G2 fails
against W. No quality metric or gate changes after seeing results.

Only if G3 and G4 qualify, freeze that result and run all six models on a new final
segment. D05 consumed 32768 inputs starting after native common prefix100418 in
original final_test.txt. Choose the earliest subsequent newline boundary with
native prefix>=100418+32768+1024 and at least32768 remaining tokens for the existing
64chunk evaluator. If that is unavailable, leave final UNRUN and obtain a separately
precommitted fresh corpus in a later experiment; do not reuse consumed tokens or
silently shorten the final evaluation. Final uses identical gates and marks the
segment consumed before inference. Failed selection preserves final data.

## Verification and falsifiers

Test inverse Q/K transforms, full mapping/shape/dtype/finite checks, tied aliases,
wrong/omitted sources and reference-gate behavior. Replay the original endpoint
selection capture to verify unchanged existing capture output. Every new capture
must come from actual W1 inference. Native reports must identify their correct
source and candidate. Independently re-decode records and audit all-stage checksums,
source/calibration links and preserved passing records. Freeze implementation,
model,corpus,calibration,runtime and protocol hashes before the trial and recheck
after it. Preserve snapshots, logs and negative results; no inferred improvement.

Broader full22 candidate coverage, uncertainty, Mamba, attention, text/code/math and
packed-kernel requirements remain separate. This experiment closes only the
explicit sequential-model comparison at its actual measured boundary.
