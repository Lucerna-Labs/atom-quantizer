# M1 outcome: output quality improved, temporal benefit gates failed

Precommit `ab29527`; post-trial destination fix `ac872a7`. The complete Mamba
trial ran at the declared boundaries: five stored models, four actual calibration
views, five native inference runs and real recurrent trajectories through 1,024
tokens. Both benefit gates **FAILED** because the late-sequence state-error
improvements fell short of the fixed criteria. No threshold changed, no fresh
final set was consumed and no candidate was promoted.

## Complete-model selection results

Native PPL uses the same selection corpus and 64 chunks, context 512, batch and
microbatch 512, four CPU threads and zero GPU layers. Temporal results use four
nonoverlapping 1,025-token blocks from that validation corpus, with 1,024 carried
inputs and next-token targets in each. These are different scoring windows;
absolute native and streamed PPL values are not interchangeable. Every comparison
within each evaluation uses identical inputs and runtime conditions.

| Stored model | Complete archive bytes | Native selection PPL | Mean teacher KL | Late-256 SSM NMSE |
|---|---:|---:|---:|---:|
| Lossless control | 481,939,978 | 31.2313 | reference | reference |
| Four-bit scalar | 72,439,191 | 102.5970 | 1.170476 | 0.02657206 |
| Six-bit scalar | 107,225,395 | 33.2949 | 0.069924 | 0.00177252 |
| Six-bit functional correction | 108,287,812 | 32.7018 | 0.052558 | 0.00173546 |
| Functional, selectors exact | 121,690,522 | 32.6325 | 0.051532 | 0.00162300 |

The functional correction reduced measured teacher KL by **24.84%** and improved
native PPL from 33.2949 to 32.7018, at about 1% additional stored bytes. Its
late-sequence state NMSE improved by only **2.09%**, below the precommitted 5%.
Keeping the 48 selector matrices exact improved late state NMSE by **6.48%** over
the functional candidate, below the required 20%, at about 12.4% additional bytes.
The other fixed checks passed. These remain failed conjunctive benefit gates;
individual favorable measurements do not convert them into passes.

Functional correction's state-error reduction versus scalar was 15.14% over the
first 64 positions, 9.32% over the first 256 and 3.96% over all 1,024. The last 256
positions improved by 2.09%. This locates a weakness in this candidate's late-prefix
fidelity; it does not isolate context length as the cause, because token content
also changes across those windows. No significance or universal-optimality claim.
The raw per-layer and per-sequence results remain available.

## Actual model, storage and execution

The local standard Mamba checkpoint has 129,135,360 unique parameters, 242 tensors
and 24 layers. Its README's stray 2.8b label was not used as model identity. The
unmodified official converter produced a 518,323,392-byte F32 GGUF, SHA
3a4b925d3e7e6bde08b6f63fbc9fca48e632f7746847653d795c34cfc099f793.
The source corpus, checkpoint, runtime and implementation identities are
preserved in phase manifests. Converter files were checked after the trial as
clean at the recorded vendor commit; CONVERTER.json records their hashes and
the actual conversion command. Every converted operator was also checked
against the source checkpoint.

All 97 linear matrices were compressed in the principal controls. The other
145 tensors, including transition A, convolution, D, biases and norms, were exact.
The selector control additionally kept its 48 x/dt matrices exact. Independent
auditing proved the other 194 stored records byte-identical between the two
functional models, so this comparison did not introduce unrelated refits.
Complete A22-2 bytes include the GGUF prefix, codes, factors, exact state and
metadata. Every candidate was exported by both Python and the actual native main
executable; the complete GGUF hashes match. The lossless decoded file also matches
the source file exactly. All decoded files are dense F32: no packed-kernel speed,
energy or resident-memory benefit follows from the archive sizes.

The four capture views contain all 98 roles: 97 linear roles, including the tied
output head, plus embedding lookup. Fit views have 256 samples per role; selection
views have 64. All 242 converted source tensors were verified against the executing
checkpoint. The time-step projection uses direct matmul in this runtime, so its
real input is captured from the selector output prefix. A runtime-dispatch test
checks the actual matmul operands rather than trusting an uncalled layer hook.

## New operational functions and actual-state proof

`experiments/mamba_runtime.py` constructs the supported Mamba model from GGUF
metadata and loads every operator without the original checkpoint. GGUF stores
A=-exp(A_log). Simple log inversion missed 2,374 source values; a bounded search
among adjacent float32 values recovered exact exponential preimages for all of
them. Verification checks the actual reconstructed A, complete mapping, shapes,
F32 types, finiteness, convolution reshape and tied embeddings. Unsupported
metadata or transitions without a verified preimage reject explicitly.

The restored source model produced bit-identical logits and both state kinds to
the checkpoint at 64 tokens. At 1,024 tokens, source full-pass versus streamed
execution had maximum logit difference 0.0004501, recurrent relative MSE
8.57e-14 and convolution relative MSE 4.42e-13, within the committed numerical
limits. Cache extraction uses dictionary tensor values at key zero, with explicit
shape/type checks; integer keys and metadata are not treated as tensors.

`experiments/mamba_trajectories.py` measured 20,480 paired steps across the reset
control and four compressed candidates. It retained complete per-token/per-layer
energies, probabilities and losses, plus 60 raw checkpoints at 64/256/1,024 tokens.
Original checkpoint and original GGUF opens were denied during this evaluation,
including explicit rejected probes. Only independently decoded artifacts supplied
weights. Resetting the source cache each token changed behavior substantially:
streamed PPL was 1,177.09 versus the carried-state teacher's 32.15 on these blocks.
That is a negative control, not an alternative compression result.

The independent audit passed all **1,210 stored tensor reconstructions**, all
capture identities/roles, raw checkpoint metrics, full trace summaries, matching
source-teacher losses, native logs/settings and the two fixed rejection decisions.
It separately reran the first 64 actual steps of every comparison and reproduced
all state arrays and logits exactly. This replay scope is stated explicitly;
the raw audit does not claim a second execution of every complete trajectory.

Mechanism context: the [original Mamba paper](https://arxiv.org/abs/2312.00752)
describes input-dependent selective state propagation. The quantization hypotheses
and measurements here are this checkout's engineering experiment. This work uses
the user's mathematical primitive-kit branch and is not a new synthetic-emergence
Atom component. Existing native Atom crates and their failed trials remain intact.

## Validation and preserved failures

All **103 Python tests** pass. New tests cover exact transition reconstruction,
invalid transitions/metadata, complete real-model loading, cache dictionaries,
actual dt operands, energy accounting, fixed gates and destination preservation.
An early test attempted bitwise comparison of recomputed matmul outputs and saw
3.725e-9 differences. Its logs and the unsuccessful instrumentation attempts remain;
the final test observes the actual input operands exactly. No numerical gate was
loosened and no cause is invented for the older D03 reproducibility concern.

After the independent audit, temporary controls reproduced a new build-command
bug: a failed call could add failure.json inside an existing output directory, or
follow a dangling destination. The wrapper now creates a new raw absolute
output directory before it owns any failure receipt. Actual CLI checks preserve
existing directories and dangling links, and new owned failures retain diagnostics.
The numerical build body is AST-identical to the audited version. Only
mamba_models.py differs from the frozen runtime files; the exact audited source is
retained in implementation-evaluation.tar.gz. Historical identity checks must not
be weakened to impersonate that snapshot with later source.

## Scope retained

M1 completes this declared real recurrent-model comparison, including its negative
benefit result. It adds operational GGUF reconstruction, actual operator capture
and state-fidelity measurement. Existing archive decoding handles the new complete
models. The normal executable, default recipe and production thresholds remain
unchanged.

The current native assessor treats matrix tensors as linear operators; it does
not yet model Mamba's nonlinear transition and depthwise convolution. Complete
native admission and any promotion therefore require a separate operator-aware
implementation and fresh evaluation. No unsupported native gate is called passed.
Further calibration/temporal-function discovery can use the observed short/late
fidelity difference as a hypothesis, with matched controls before claiming a cause.

The broader full22 objective remains active: other candidate/composition frontiers,
attention coupling, uncertainty/sample allocation, text/code/math teacher checks
and actual packed-runtime work still need their required evidence. No fresh final
segment was used by M1. B04 remains UNRUN and D03's intermittent root cause stays open.

Evidence: SUMMARY.JSON, AUDIT.json, GATES.json, MANIFEST.JSON, SOURCE.json,
CAPTURES.json, SOURCE_CONSISTENCY.json, POST_TRIAL_FIX.json and EVIDENCE.json.
Complete local artifacts and logs are under artifacts/mamba-temporal-2026-10-01/.
The capture, build and evaluation source snapshots, all failed test logs and the
before/after destination probes are retained. Use new output paths for every run.
