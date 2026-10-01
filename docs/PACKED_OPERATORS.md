# Apply a stored matrix directly

The native CPU command applies one verified A22-2 matrix without expanding it
into a complete dense matrix:

```bash
target/release/atom-quantizer apply-packed model.a22 \
  --tensor blk.0.attn_q.weight \
  --input inputs.f32 --batch 8 --out new-outputs.f32
```

For a stored matrix shaped `[rows, columns]`, the input file contains exactly
`batch * columns` little-endian F32 values in sample-major order. The new output
contains `batch * rows` F32 values in the same order. The destination must be new;
existing files and dangling links are preserved. Nonfinite inputs, outputs,
unsupported recipes and failed record/decoded commitments reject publication.

The JSON receipt identifies archive, input and output hashes, dimensions and
`scope: selected_tensor`. This operation verifies the selected tensor's complete
recipe and reconstructed commitment. Use `verify` to verify every tensor in an
archive. No original weights, calibration, Python or exported dense GGUF is needed.

The supported native representation is scalar 2/3/4/6/8-bit A22-2, correction
ranks zero through two, raw F32/BF16 and constants. Matmul requires a matrix.
Each corrected weight retains the decoder's F32 rounding before multiplication;
the dot product accumulates in ascending column order in F64, as the app's
existing CPU reference does. The low-rank factors are not algebraically moved
outside that rounded-weight calculation.

The library exposes `a22::load_tensor`, `load_packed_model`, `load_dense_model`,
`PackedTensor::matmul` and `DenseTensor::matmul`. Complete model loaders verify
every tensor. `Kernel::Fused` streams values directly; `Kernel::Row` reconstructs
one row of scratch space at a time. Both preserve the same numerical result.
Retained allocation accounting and actual process RSS are separate measurements.

## Measured tradeoff

On the accepted SmolLM model, complete verified weight residency measured
**131.5 MiB packed versus 528.0 MiB dense**, approximately 75% lower. These are
isolated native processes retaining all 272 tensors, with an actual matrix
operation and OS memory-accounting checks. This does not measure a full inference
engine's activations, KV cache or total RAM use.

All 633 actual operator cases (211 matrices at batches 1/8/32) matched the
existing dense CPU reference bit-for-bit. The implementation was slower in all
tested batch regimes. Fused execution took roughly 5.53x, 2.17x and 1.54x the
faster native dense control's latency at batches 1, 8 and 32 respectively, and
was further behind optimized F32 BLAS. This is an optional lower-memory matrix
operation. It does not change the default encoder or route model inference.

The current evidence is CPU/Linux on one host and model. See the
[K1 results](../research/packed-runtime-2026-10-01/OUTCOME.md) for raw timings,
startup costs, RSS, exact identities, failures and independent verification.
