# M1: actual Mamba temporal fidelity under stored compression

Precommit before candidate implementation or outcomes. S1 was concrete progress:
a full sequential trial was rejected and real decoded-model calibration was
implemented. This trial addresses full22 item18 using the matching Mamba model,
including real recurrent state, increasing context and complete stored artifacts.
It does not complete the broader22-candidate programme.

## Mechanism and preflight evidence

The local checkpoint is state-spaces/mamba-130m-hf:129,135,360 unique parameters,
242 tensors,24 layers, hidden768, intermediate1536, state16, convolution4 and
time-step rank48. Its README's2.8b text is not treated as model identity.
The unmodified official llama.cpp converter produced F32 GGUF
artifacts/mamba-temporal-2026-10-01/preflight/source.gguf,518,323,392 bytes,
SHA3a4b925d3e7e6bde08b6f63fbc9fca48e632f7746847653d795c34cfc099f793.
No vendor code is changed or committed.

The installed Transformers5.16.1/PyTorch2.14.0 CPU implementation is inspected
and will be hashed. State lives in cache.layers[i].recurrent_states[0] and
conv_states[0], dictionaries whose keys are integers. Metadata is not a tensor.
The dt projection is a direct weight matmul; capture its exact input from the
first48 columns of x_proj's actual output, rather than expecting a dt_proj hook.
Use real token-by-token cached inference for trajectories. Full/cached source
execution at64 tokens differed by max0.000328 logits and about1e-13 final-state
relative MSE in preflight; this is runtime compatibility evidence, not compression.

GGUF stores A=-exp(A_log). Decoding for the HF runtime must recover a float32
preimage whose actual -exp equals every stored A value exactly. Direct log missed
2,374 values across the24 source tensors; bounded adjacent-float search recovered
all values in preflight. Preserve that exact transition, all other mapped weights,
convolution reshaping, and tied embedding semantics. Instantiate from GGUF metadata
without reading source checkpoint weights/config during standalone loading.

This is prescribed numerical tooling from the user's mathematical primitive-kit
branch: observe, project, compare, decompose and serialize. It is not claimed as
synthetic-emergence Atom behavior. The Mamba paper provides the selective-state
mechanism context (https://arxiv.org/abs/2312.00752); applying the functional
correction here is an untested engineering hypothesis. Negative A and positive
softplus(dt) do not alone prove small accumulated model error.

## Frozen complete-model candidate set

Reuse A22-2, existing scalar/functional codecs and native main-app decoding.
All97 linear matrices (96 block projections plus tied token/output embedding)
are candidates for compression. Keep all other145 tensors exact, including A,
D, convolution weights/biases, norms and dt bias. This isolates projection and
input-dependent selector rounding; no synthetic linear proxy is substituted for
the nonlinear transition or depthwise convolution.

1. Lossless source archive and independently decoded source model.
2. q4-scalar: four-bit scalar matrices, no correction.
3. q6-scalar: six-bit scalar matrices, no correction.
4. q6-functional: six-bit matrices, rank2 activation-output correction.
5. q6-functional-selectors-exact: same correction, but all48 x_proj/dt_proj
   matrices exact to isolate the effect of protecting the selective recurrence.

No validation-driven tensor escalation or recipe selection. Native decoder
commitments apply to every tensor. Complete bytes include prefix, codes, factors,
exact state and all metadata. Export final archives natively to complete GGUF.
HF loading must use only that decoded GGUF plus the installed runtime, not the
original checkpoint. Decoder factors/working memory and dense exported size are
reported separately. No packed-inference speed claim.

## Data and actual operator capture

Use the existing128 training documents and32 selection documents, context512,
with the same fixed endpoint and stratified-seed0 sampling mechanics as D05.
Capture256 fitting rows per operator and64 selection rows. Source-bind AC01 to
the new Mamba GGUF. There are98 entries:97 linear roles (including the tied head)
and the token embedding lookup. Verify every source GGUF tensor against the
actual executing model's converted operator values. The scalar fit uses interior
rows; functional factors use endpoint rows. Fitting is exclusively training data.

For temporal selection, tokenize selection.txt with the checkpoint tokenizer,
freeze the first four nonoverlapping1025-token blocks and use1024 inputs plus the
next-token target in each. This is validation, already used for selection in
other experiments; it is not a fresh final set. Inspect prefix lengths64,256,1024
from the same continuously carried state, and retain per-token/per-layer state
and convolution error, norms, finite flags, teacher KL, top1 agreement and NLL.
Aggregate ratios from summed energies, also report per-layer and per-sequence
values so tiny denominators or one large layer cannot hide failures. Store raw
source/candidate states at the declared prefix boundaries with checksums.

Include a source-weight reset-state control (new cache each token) on the same
four blocks. It must differ from carried-state execution; its measured quality
is diagnostic. Compare full-forward versus token-by-token source logits and end
states on the first1024-token block before interpreting compression trajectories.

Run actual native llama-perplexity on all five complete decoded models using the
existing selection.txt,64chunks,context512,batch512,microbatch512,4threads,ngl0.
Do not reuse SmolLM scores. Also score all temporal candidates against the same
source teacher under actual streamed HF execution. Freeze model/corpus/runtime,
implementation and protocol identities before runs and recheck after them.

## Gates and predictions (unchanged after outcomes)

G0 correctness: all242 converted operator arrays exact after source loading;
all serialized tensor hashes/decoded commitments exact; every final native GGUF
matches Python export. Source full/stream max-logit difference<=0.01 and final
recurrent/conv relative MSE<=1e-8. Lossless source reconstruction produces identical
actual forward outputs to the checkpoint for matched execution. Reset-state
control differs. Prediction: MEASURED.

G1 scope: all98 actual capture roles, all four captures, all five complete archives,
four validation trajectories per candidate through1024 inputs, raw checkpoints
at64/256/1024, and five successful64chunk native PPL runs. Prediction: MEASURED.
No scope substitution from isolated tensors, simulated state or shorter contexts.

G2 principal functional benefit: q6-functional complete bytes<=1.05*q6-scalar;
native selection PPL<=q6-scalar; mean teacher KL<=q6-scalar; aggregate recurrent
state NMSE over the last256 positions<=0.95*q6-scalar, with no nonfinite trajectory.
Prediction: uncertain. Low local error can fail after nonlinear recurrent use.

G3 exact-selector tradeoff: selectors-exact bytes<=1.20*q6-functional; native PPL
and teacher KL<=q6-functional; last256 recurrent NMSE<=0.80*q6-functional.
Prediction: uncertain; preserving selectors may improve temporal fidelity, but
embedding/in/out errors may dominate. Count the exact-state storage cost.

q4 and lossless controls define distortion/size context. Failed/dominated results
remain. Do not choose a different metric or threshold after results. A favorable
research result is not native full-model admission: the existing native assessor
models matrices as linear operators and cannot yet represent Mamba's nonlinear
A transition/depthwise convolution. Its production thresholds and supported
contracts remain unchanged. Promotion requires a separate actual operator-aware
admission implementation; no unsupported gate is silently relabeled as passing.

No fresh final corpus is consumed in M1. Any candidate chosen for product promotion
needs a separately precommitted fresh evaluation, task checks and actual admission.
This declared temporal trial remains useful whether its benefit gates pass or fail.

## Tests and evidence

Test dictionary cache extraction, missing/nonfinite/wrong-shaped state rejection,
exact transition-preimage recovery and impossible values, complete mapping and
tied alias loading, metadata incompatibility, actual source full/stream behavior,
capture dt-projection coverage, stored commitment corruption and existing output
preservation. Use tiny real Mamba models for boundary tests and the actual130M
model for all substantive evidence. Preserve source snapshots, negative results,
raw logs, standalone-source-access denials and an independent artifact/metric audit.
Append the build log and update shared Obsidian context. No default/release change
is inferred from this experiment; no other Atom checkout is accessed.
