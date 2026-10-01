# D01: stored functional-residual directions

Precommit status: UNRUN. This branch applies Jesse's mathematical primitive-kit /
composition architecture in the existing **conventional experimental codec**.
It is explicit numerical optimization, not an Atom synthetic-emergence claim.
The native B01–B03 failures remain unchanged; B04 remains UNRUN after the user's
catalog-driven steering. The active goal and full22 work are not declared complete.

## Mechanism and new function

The catalog-driven observer found concentrated fit/selection operator-error
directions. D01 tests whether a complete stored representation can exploit them.
Compose existing quantize/decode, matrix projection, spectral factorization,
packing and blind reconstruction primitives. The research codec already stores
low-rank factors; this adds a functional geometry for discovering their direction.

Let W be source weights, Q the reconstruction of the existing saved scalar
record, E=float64(W)-float64(Q), and X the real fit inputs. Compute Y=E X^T.
Take the first k left singular vectors U_k of Y and B=U_k^T E. Store U_k and B
as f16. Blind decode is Q+float32(U_k)@float32(B). Every measurement uses that
saved representation. Neither fit inputs nor the original weights are required
by the decoder. Selection/test inputs never choose the factors.

Without storage rounding, ||(E-U_k B)X^T||_F^2 is the tail sum of Y's squared
singular values. This is a fit-space identity, not a generalization guarantee.
Only nonzero numerical singular directions count: singular > s_max *
max(Y.shape) * eps_f64. The effective rank is bounded by the requested rank,
matrix shape and measured numerical rank. Zero measured rank stores empty
factors and does not invent a correction in unobserved directions.

Compare against the existing diagonal-importance-weighted residual factorizer
(unchanged default) and a shuffled-input geometry: apply a seed-0 permutation
to X's columns only while deriving Y. All methods start from the identical scalar
record and store the same factor precision. The shuffle tests functional alignment.

## Stage 1: frozen tensor screen

Same source/captures/eight catalog tensors as prior studies; up-to-256 evenly
spaced rows, 3/4 bits, group32 fitted scalar coding, nearest rounding, ranks 1/2/4.
Three geometries: activation, existing diagonal and activation-shuffled. No
additional transform, correction, per-tensor winner or parameter search.

Exactly 8 tensors * 2 widths * 3 ranks * 3 geometries = 144 candidate records,
plus 16 scalar references = **160** unique records or explicit failure receipts.

S1: all records deserialize/blind-decode with valid shapes and finite values;
repeated decode is exact. Count all actual raw/compressed record bytes and hashes.
S2: for a width/rank, aligned mean selection-output NMSE is <=75% of scalar,
strictly lower than both same-rank geometry controls, and no tensor exceeds
105% of its scalar selection-output NMSE. Complete raw bytes summed over tensors
must be <=110% of scalar at that width. Predict substantial fit-error reduction;
selection benefit remains uncertain and can fail from overfitting or f16 factors.
S3: source/fit/selection/implementation/protocol identities recorded and unchanged;
all 160 cases retained, including failures and rejected ranks.

At each width, choose the smallest rank meeting all stage-1 criteria. This is a
single global configuration for the next stage, not per-tensor selection. If no
rank qualifies, that width has no full-model candidate in D01. Model-stage
eligibility is not main-codec promotion; its existing KL/cosine gates are unchanged.

## Stage 2: conditional complete-model validation

For each eligible width, build three complete archives using the same real fit
capture: scalar, existing diagonal residual at the selected rank, and aligned
functional residual at that rank. All 2D tensors use the one declared global
configuration; non-2D tensors retain the existing exact record path. Missing
required linear calibration rejects the build. Fully export and verify every
tensor from the saved artifact, without relying on original weights for decode.

Evaluate source F32 and these models with installed llama-perplexity, CPU-only,
4 threads, context/batch/microbatch 512, 64 chunks of full22 selection.txt.
Preserve command/version/model/corpus hashes and complete logs. Selection-model
gate: aligned PPL <=95% of its scalar reference, no worse than the same-rank
diagonal control, and complete archive size <=110% of its scalar archive.

For each width passing that gate, evaluate its three artifacts plus source on
64 chunks of the separately retained final_test.txt, with identical arguments.
Apply the same PPL/size requirements on final-test results. No tuning from these
test results. At most six candidate/control archives and seven models per corpus;
source is shared between widths. Retain every evaluated point and any failure.

If the preserved v0.2.4 120-MiB decoded model is present and matches its benchmark
hash, also evaluate it on those exact corpora as an additional historical product
comparison. Label its different calibration procedure; it is not the controlled
factor-geometry ablation. Existing benchmark PPL from another corpus segment is
not substituted for a current result.

These are dense-decoded inference measurements. They establish no packed-kernel
speed, resident-memory reduction, additional architecture generality or physical
quantum effect. Full model size includes archive metadata and stored decoder
state. Do not present tensor screening alone as improvement of model quality.

## Numerical preflight

An independent seed-17 matrix check with E shape12x9, X shape7x9 and k=2 gives
residual fit energy 180.94804946544517 versus SVD-tail energy 180.9480494654452
(difference 2.842e-14). f16 factor payload is 2*k*(12+9)=84 bytes before framing.
Tests must cover the identity, shuffled direction, zero/rank-deficient inputs,
storage rounding, invalid state and complete fixture export before real trials.
