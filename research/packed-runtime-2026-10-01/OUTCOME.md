# K1 outcome: verified lower-memory matrix execution, slower kernels

Precommit `bcceda3`. The actual packed native operator passed correctness and
complete-model residency gates. Its latency gates **FAILED** in every declared
batch/method regime. The optional `apply-packed` command is installed in the
normal local executable because the precommitted correctness and memory conditions
passed. Existing encoder defaults and model inference paths are unchanged.

## Measured resident memory

Fresh native processes loaded and verified all 272 tensors of the accepted D05
SmolLM archive, retained the representation and performed an actual output-head
matrix operation on a captured input. Linux process RSS, rather than file-size
estimates, gives:

| Representation | Loaded RSS | After actual operation | Owned tensor representation/state bytes |
|---|---:|---:|---:|
| Packed native tensors | 131.49 MiB | 131.49 MiB | 130,850,892 |
| Dense F32 tensors | 528.02 MiB | 528.02 MiB | 538,076,952 |

Loaded packed RSS is **24.90%** of the dense control, a **75.10% reduction**, meeting
the fixed 40% ceiling. Both modes represent 538,060,032 decoded parameter bytes
and retain the same 1,786,080-byte metadata prefix. Side arrays, expanded scales,
correction factors and raw-record storage are counted. RSS includes allocator and
runtime effects beyond those owned-byte counts. Startup/verification and transient
HWM values are retained separately in the raw reports.

A second pair of isolated executions corroborated the internal HWM through
`/usr/bin/time` OS accounting and repeated the memory verdict. This measures
complete weight residency and one actual operation. It does not establish a full
model inference engine's KV-cache, activation, total-RAM or throughput benefit.

## Latency result, including optimized dense controls

Eight complete real tensor roles were timed at batches 1/8/32. Each case retained
two warmups and eleven measured calls per method. Native methods were interleaved
in a fixed seeded order; tensor/batch case order was also fixed and shuffled.
All work was pinned to CPU0 and single-threaded. The host was an Intel i7-13700K,
Rust1.98.1 with ordinary release settings, and NumPy2.5.2/OpenBLAS single-thread
F32 execution. The independent BLAS output check passed its declared relative
squared-error ceiling against the F64-accumulating reference.

The table shows geometric means of per-operator median latency ratios. Below1
would be faster; the precommitted speed gate required <=0.9. No slow cases or
individual timing samples were removed.

| Batch | Fused / faster native dense | Row / faster native dense | Fused / optimized F32 BLAS | Row / optimized F32 BLAS |
|---|---:|---:|---:|---:|
| 1 | 5.53x | 6.21x | 46.28x | 51.97x |
| 8 | 2.17x | 2.18x | 11.11x | 11.19x |
| 32 | 1.54x | 1.44x | 10.20x | 9.53x |

Both kernels are slower in this trial. The native dense controls include the
app's existing calculation and a matched kernel with the same input layout and
accumulation. The BLAS comparison is reported independently; beating a simple
reference would not establish a speed advantage over optimized dense execution.
Startup integrity checking, decompression, dense expansion and hot operator calls
are separate costs in the saved records. This is a memory/latency tradeoff.

## Exact arithmetic and real-boundary verification

A shared prepared layout now serves native export, integrity checking and packed
execution. It supports the existing native scalar 2/3/4/6/8-bit records, correction
ranks zero through two, raw F32/BF16 and constants. Unknown decoder state still
fails closed. The selected record and decoded-weight commitment are verified
before a tensor is exposed. Complete model loading verifies every tensor.

Each corrected coefficient preserves the exact decoder's F32 arithmetic. The
correction is not moved outside the rounded-weight expression as U(VX). The
fused kernel decodes coefficients directly into ascending-column F64 accumulation;
the row kernel uses one row of scratch space. Neither creates a full dense matrix.
The existing CPU dense reference uses the same accumulation order.

All **633 real cases** (211 complete matrices x batches1/8/32, from actual held-out
operator inputs) matched both packed methods and the matched dense kernel
bit-for-bit against the existing dense implementation. An independent NumPy
ascending-column F64 calculation, using the pinned dense GGUF, reproduced every
one of those output hashes. The auditor checked all24 timing cases and all1,320
retained measured samples, output files, identities, thread/affinity settings,
summary statistics, memory receipts and fixed gate decisions.

Boundary tests cover every supported bit width, correction rank including stored
zero rank, non-group-aligned dimensions, raw forms, signed zero, invalid shapes,
nonfinite inputs/output overflow, damaged commitments and output preservation.
The existing native archive and functional-build tests were explicitly rerun
against the new binary. All **109 Python tests** and **72 Rust tests** pass.
CUDA compilation passes; only CPU execution is claimed. Clippy retains the known
six GGUF style warnings and the existing discover_resources example warning;
there are no new warnings in the changed implementation. No global strict-Clippy
claim is made.

Six legacy OQ cases retained byte-identical archives, dense exports and score
lines. The normal executable also verified and exported the complete D05 model
with unchanged SHA4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8.
No encoder, archive, fidelity floor, training input or final-test segment changed.

## Operational integration

See [PACKED_OPERATORS.md](../../docs/PACKED_OPERATORS.md) for the command and library
interfaces. Inputs/outputs are sample-major little-endian F32 with explicit batch
and stored matrix dimensions. Existing files, dangling links and staging collisions
are preserved; failures do not publish partial outputs. The receipt reports
`scope: selected_tensor`; it does not imply verification of untouched records.

An isolated executable/archive/input-only run succeeded with Python unavailable.
The system-call trace shows one executable and no source weights, calibration,
Python or dense GGUF access. The actual normal-path command produced the same
output-head result SHA68e8efdda72b7b008e2e1bba75c23121d05361f762d7f8cea4bc2aec10314ed5.

After preserving the predecessor, the normal release was rebuilt and is identical
to the benchmarked/tested binary:
e4c7b6f39ccc3d7509fd457d7b78dae247afe8b95bfebbace72cd3bef416b1cb.
Backup artifacts/packed-runtime-2026-10-01/atom-quantizer-before-release preserves
I2's d45eaa37c05ae4d08001e7b08ab3d90a0cfe9c970e03c0f06351558d6c9a400b.
The local version remains0.2.4; this is not a tagged/remote release.

This conventional scan/fold implementation follows the mathematical toolkit branch.
Native Atom primitives and their separate emergence requirements remain untouched.
Existing dirty work was preserved; only the experiment protocol was committed.
The task delta is retained against its own before-source snapshot, avoiding
misattribution of older changes already present in the worktree.

## Remaining scope

K1 completes this declared native packed-kernel comparison for the supported A22
family, including its failed speed gates. The optional memory-saving operation
is operational. Further SIMD/packing work would need a new measured trial; the
current result does not justify routing whole-model inference through it.
Other full22 candidate/composition frontiers, attention coupling, uncertainty and
text/code/math fidelity requirements remain active. M1's temporal benefit gates
remain failed, S1 remains rejected, B04 is UNRUN and D03's root cause stays open.

Evidence: SUMMARY.JSON, AUDIT.json, MANIFEST.JSON, RUNTIME.JSON, RELEASE.json,
ISOLATION.json, OQ_REGRESSION.json, BACKUP.json and EVIDENCE.json. Complete local
results, all raw samples/output hashes, external memory checks, system-call trace,
source snapshots, task patch and binaries are retained under
artifacts/packed-runtime-2026-10-01/.
