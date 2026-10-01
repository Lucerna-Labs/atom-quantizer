# K1: actual packed matrix execution and resident-memory measurement

Precommit before implementation or performance outcomes. M1 was concrete progress:
actual recurrent-model evaluation completed, with both benefit gates failed.
K1 addresses full22 item22 at the native runtime-kernel boundary. It does not claim
whole-model inference speed from storage size or from a collection of operators.

## Mechanism and fixed scope

Use the catalog's conventional scan/fold/project toolkit to consume verified packed
weights directly. This is a prescribed native numerical implementation, not a new
synthetic-emergence Atom component. Native Atom crates, thresholds and incumbents
remain intact. No vendor implementation is modified.

The target representation is the existing native-supported A22-2 scalar family:
2/3/4/6/8 bits, stored levels/scales, rank0/1/2 correction, raw F32/BF16 and constants.
Share its existing record validation. Preserve every decoded weight's exact F32
semantics, including scale/level multiplication, correction products/addition,
zero-rank correction and signed zero. Do not replace elementwise rounded corrected
weights with U(VX): that is a different floating-point computation.

Implement two bounded-memory native methods: fused per-element decoding/accumulation,
and decode-one-row/reuse across the input batch. Neither materializes a complete
matrix. Double accumulators follow ascending column order, matching the app's
existing CPU dense reference. A matched dense kernel uses the same input layout
and accumulation; compare the existing CPU reference too. Independently measure
an optimized NumPy/BLAS F32 dense baseline, with its own bounded numerical check.
Do not describe beating a scalar reference as beating optimized dense execution.

Provide an operational apply-packed CLI and library surface. Inputs/outputs are
explicit little-endian F32 batches with declared batch count and authoritative
stored matrix dimensions. Reject invalid shapes, nonfinite input/output, unsupported
state, damaged records/commitments and existing/dangling destinations. Publish only
a verified complete result to a new path. A single-tensor operation verifies its
selected tensor and must label that scope; it is not a full-archive quality claim.
Complete model loading independently verifies every tensor.

## Actual data and controls

Use the accepted SmolLM D05 activation archive,109251501 bytes,
SHA27d36037a432c5cefeeb0c1f62505377f469365027e00372d30b1840c4608a83.
Its decoded GGUF SHA is4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8.
Use stratified0 selection AC01 from the real D02 capture:
artifacts/calibration-coverage-2026-10-01/captures/stratified0-selection.acal,
SHA8e357a6712d5d288aa53a2e088f574c26246868db14eb864fd2fb73c9b3e4d99.
The tied embedding uses its captured output-head input role. No new calibration,
encoder, model weights, final-test data or inference PPL is introduced.

Correctness covers all211 complete matrix tensors at batches1,8,32 from those
actual inputs, both packed methods, the matched dense kernel and existing app
CPU dense reference. Probe all supported bit widths/raw forms/correction ranks
and non-group-aligned sizes with boundary tests. Existing A22 decode/verify and
six OQ regression cases must remain unchanged.

Timing uses the eight original full22 tensor roles: token_embd, blk0 q/k/attention
output, blk3 ffn_up, blk9 ffn_gate, blk17/29 ffn_down. Use complete matrices,
not slices, at batches1,8,32. Pin one allowed CPU and limit native/BLAS work to one
thread. Record hardware, compiler, libraries and environment relevant to execution.
Use two warmups and eleven measured calls per method/case. Interleave methods in
a fixed seeded order where possible and retain every sample. Report median,
spread, startup verification/decompression and operator latency separately.
Every result must be consumed outside optimization; report output commitments.

Memory uses fresh isolated native processes for complete D05 model weights,
including all272 tensors and metadata. Compare retained packed representation
against owned dense F32 weights loaded from the same archive with identical
verification. Record actual Linux RSS/HWM before/after loading and after an actual
operation, plus exact owned representation/side-state bytes. Loading must touch
and verify all records. Do not present encoded byte accounting alone as RSS.
Report transient initialization peaks and steady resident measurements separately.
Kernel scratch and output allocations are included/accounted; no KV-cache or
end-to-end-model memory saving is inferred.

## Gates and predictions

G0 correctness: bit-identical outputs to the existing CPU dense reference on every
real tensor/batch case; decoded commitments unchanged; both methods agree; malformed
inputs fail without publication; existing/dangling outputs stay intact. Prediction:
MEASURED, because each stored weight and accumulation order is preserved.

G1 complete measurement: all declared real operator cases, complete model residency,
both packed algorithms, both native dense controls and optimized dense baseline,
with pinned identities and raw timing samples. Prediction: MEASURED. No substitute
from model file size or a process existing without actually executing the kernel.

G2 memory benefit: packed complete-model loaded RSS <=40% of dense loaded RSS,
with all272 tensors verified in each isolated process. Prediction: likely MEASURED;
actual allocation/side-state/transient costs may defeat this and must be measured.

G3 latency benefit is separate: for a packed method, geometric-mean operator median
must be <=90% of the faster native dense control for a batch regime. Evaluate each
method and batch regime as declared; no cherry-picked operator-only speed claim.
Also compare with optimized F32 BLAS. Do not claim speed over BLAS unless the same
criterion holds there. Prediction: uncertain, likely FAILED for some regimes;
unpacking and exact correction arithmetic may cost more than reduced memory traffic.
A lower-memory operation remains useful if G0/G2 pass even when speed does not.

Dense BLAS numerical output must be finite with relative squared output error
<=1e-10 against the existing double-accumulating dense reference. This is an
independent floating arithmetic check, not a replacement/relaxation of G0's strict
identity for the packed implementation or any production model fidelity floor.

## Integration and evidence

A validated optional packed operation can be added to the normal local executable
if G0 and G2 pass. Preserve a hashed predecessor and run the actual normal-path
command. Do not reroute dense model inference or default fitting from kernel-only
evidence. Keep G3 failures and all benchmarks, source snapshots and tests.

Freeze implementation/inputs/executables/runtime before measured runs. Independently
recompute output checksums, byte/RSS accounting and summary statistics from raw
artifacts. Run appropriate Rust/Python integration tests and CUDA compilation;
only CPU execution is claimed. Preserve existing dirty work and .atom-guard.json.
Append BUILD-LOG, update Obsidian and the full22 coverage record. Other candidate,
attention, uncertainty and task-fidelity requirements remain active.
