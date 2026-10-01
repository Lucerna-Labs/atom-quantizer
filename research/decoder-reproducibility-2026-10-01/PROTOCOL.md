# R1: capture and contain decoded-value mismatches

Precommit status: UNRUN. This is codec reliability work following D03, not a new
quality hypothesis or Atom-emergence trial. Preserve D03's fixed decoded files,
inference receipts, successful repeats and unexplained initial audit failure.

## Observed boundary

The first stricter audit failed a byte comparison in the inferred q4-IE-r1 model
without recording the tensor or differences. The isolated model, four thread
settings and two full-audit repeats subsequently matched. No cause is established.
A read-only preflight found no cast mismatch for all63,488 finite binary16 values
at64 byte offsets against an independent integer expansion to binary32. Local
NumPy is2.5.2, OpenBLAS0.3.34.0.0, Haswell, pthreads. Record actual runtime hashes.

## Bounded diagnostic

Probe q4-IE-r1 and the selected q3-IE-r1/q4-IE-r2 archives, preserving all272
tensors per model. Verify original artifact and decoded hashes first. Repeat
complete record decoding with stored-array offsets0,1,31,63 and thread counts1,4:
24 model passes,6,528 tensor comparisons. Keep original array bits, shapes and
metadata identical; vary only storage alignment and the declared BLAS thread count.
Also repeat the finite-half cast preflight and compare rank-one corrections with
an elementwise outer-product implementation that does not call matrix multiply.

Any mismatch must capture the model/tensor, archive record, expected and actual
arrays, hashes, alignment, strides, dtype, finite/numeric/bitwise comparisons,
runtime identity and floating-point rounding mode. Stop on the first mismatch
after preserving evidence. Passing samples do not prove the old failure's cause
or authorize deleting it. Do not replace failure with a numerical tolerance.

## Export consistency contract

Add A22-2 as a conventional archive-format extension. New builds commit a SHA-256
of each decoded row-major little-endian F32 tensor into the manifest. A22-2 must
contain a valid commitment for every tensor. The exporter must compare actual
decoded bytes before publication, reject any mismatch with the tensor name and
both digests, and retain existing temporary-file cleanup/new-destination behavior.
Old decoders must reject the new version; the updated decoder must still read
A22-1. Report verification counts/version honestly: legacy archives without these
commitments cannot receive new hash-verification credit. Do not rewrite old files.

This detects and contains changed reconstruction; it does not make an unknown
floating-point failure disappear or guarantee successful decoding on every host.
The manifest is an integrity commitment, not a signature against malicious edits.

Required tests: missing/malformed commitments; one-ULP and signed-zero mismatches;
no published or leftover staged output after rejection; unaffected destinations;
legacy compatibility; and correct standalone export after deleting source and
both calibration files. Injected faults are explicitly tests, not explanations
of the real D03 failure. Run the existing Python suite.

## Real-model verification

Build A22-2 versions of D03's selected3-bit rank1 and4-bit rank2 IE configurations
using the exact same source/fit files, algorithm and four math threads. Each must
export272 commitment-verified tensors with source/calibration opens denied. Compare
all underlying AR records with D03 and require exact decoded-GGUF equality to its
existing measured files. Include the actual new manifest bytes in size reporting.
If those fixed decoded hashes match, reuse their explicitly labeled PPL evidence;
do not consume a new holdout merely to remeasure identical files. If they differ,
retain the difference and investigate; do not claim D03's PPL for changed weights.

Exercise the failure path on a complete real model by changing one manifest
commitment in a separate test archive. The exporter must reject it and publish
no GGUF, while preserving the original archive. Keep the byte-level original
failure open unless new evidence actually identifies its cause.

No source model, calibration capture, old archive, native Atom primitive, safety
floor, main-codec default or D03 gate is changed. The broader goal remains active.
