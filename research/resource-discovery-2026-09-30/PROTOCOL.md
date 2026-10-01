# B03: block-local resource conditioning

Precommit status: UNRUN. B01/B02 are rejected, preserved experiments. This is a
new readout and interaction-boundary hypothesis under the active automatic goal,
not a reinterpretation of their benefit gates. The previous goal turn progressed
through a completed measured rejection; it was not a blocker or a verified wait.

## Evidence and new question

Post-run analysis of B02 found that its final FFN-down tensor had approximately
976x variation in block input energy. Its block correlations differ from the
pooled field by RMS .724 versus block-correlation RMS .736. At 4 bits, the full
order-1 candidate reduced weight NMSE 16.7% but increased output NMSE 86.9%.
These are diagnostics of the failed run, not a newly accepted candidate.

B03 asks whether the same complex native dynamics can supply useful **local
resource observations** without rotating weight coordinates. Each 32-column
block runs its own field. Its available resource fractions r are observed; the
conventional codec tests two reversible coordinate-unit readouts, conductance
d=r and resistance d=1/r. These are two declared hypotheses, not a choice made
from a tensor's score. Every declared case uses each readout and both code widths.
Native amplitudes/transport do not directly replace model weights in B03.

The two existing owning Rust crates retain their mechanics. A new resource
accessor exposes the already-evolved state; it does not compute a quantization
answer. The native component receives no code width, error, target scaling,
selection input, score, desired bit count or target output. Application of the
observed resource as a conventional reversible codec scale is explicitly an
outer experiment, not evidence that the whole codec is emergent.

## Intrinsic composition and local inputs

Use B02's Z-relative-phase and Q-unitary base, C reversible binding, R reduced
activity-memory depression/recovery, I self-information potential, P harmonic
transport, K kicked rotor and L logistic population. Their equations, dt=.125,
16 cycles, bounds and seed mapping are unchanged. This preserves all required
domains and the two distinct chaos mechanisms. Their interaction can affect
the resource trajectory through the changing amplitude-density field.

For block g and lane i, the observer's raw stimulus is
sqrt(mean_rows(W_gi^2) * mean_fit_samples(X_gi^2)): the product of RMS weight and
activation, a local contribution-energy observation before quantization. Local
coupling uses that block's uncentered fit correlation C_ij/sqrt(C_ii*C_jj), with
zero coupling for unexcited lanes. No error minimizer, corrective branch or
per-tensor routing is introduced. Zero total stimulus fails closed and is
retained as a failed execution if encountered.

Initial bound=0, resource=1 and other state/seeds as B02. Z initializes the field.
Order 0 cycles Q,C,R,I,P,K,L. Order 1 cycles Q,K,L,C,R,I,P. These remain the two
mechanically justified orders: chemistry/memory sees pre- or post-chaos fields.
No parameter sweep or additional ordering is permitted in B03.

Sixteen arms per order:

- Full principal complex stack.
- Each single removal Z/Q/C/R/I/P/K/L.
- No-base (Z+Q), no-chaos (K+L), base-only, identity.
- Uniform coupling and the same within-block shuffled-coupling permutation
  pi(i)=(5*i+7) mod32.
- Folded drive: pool weight/input second moments and correlations across the
  tensor's column blocks and repeat that one input field for all blocks. This
  isolates local versus shared observations while retaining the full stack.

All smaller arms are controls. The native result r is not corrected, optimized,
clipped or renormalized. A no-memory or identity arm may yield r=1; that is an
observed control, not a fallback selected on performance.

## Exact experiment and stored representation

Use the same source-bound fit and disjoint selection captures, eight catalog
tensors, up-to-256 evenly spaced rows, seeds 1/2/3, and fitted group32 nearest
scalar coding at 3 and 4 bits. Three conventional references: plain scalar,
signed H32 seed0, and diagonal preconditioning exponent .5. The extra reference
tests a conventional technique with the same coordinate-scaling role.

Store every chosen scale as f32 in the actual record: W'=W*d, X'=X/d, decoded
W_hat=W'_hat/d. Store the same scale used for measurement; decoding cannot need
the original model, resource field, calibration, native binary or external scale
file. Count all scales, codes, metadata and framing. No estimated tiny recipe is
credited. Report both actual raw and compressed bytes; raw bytes govern G3 as in
the prior trials.

Bound: 2 orders * 16 arms * 3 seeds * 8 tensors * 2 readouts * 2 code widths =
3072 candidate records, plus 3 references * 8 tensors * 2 widths = 48 references,
for exactly 3120 scored records or explicit failed receipts. Each block field is
shared across both readouts/widths. There are 204 blocks across the eight tensors,
so 204 * 2 * 16 * 3 = 19584 native block executions, batched into 768 processes.
This larger count is justified by the explicit block-specific failure question,
not broader order search. No unregistered model/tensor/parameter exploration is
included in this experiment.

## Frozen gates and predictions

G1: all native fields retain f64 transport orthogonality and amplitude-norm errors
<=1e-10, finite states and intrinsic occupancy/resource bounds. From
dr/dt=.5*(1-r)-b*r with b in [0,1], r starts at 1 and stays >=1/3; readout values
must agree with the native state. Any violation is FAILED, without repair.

G2: f32 scales are finite in [1/4,4] (rounding-safe envelope around the mechanical
[1/3,3] readout bounds). Unquantized reconstruction NMSE <=1e-11, exact repeated
saved-record decode, correct shapes and valid hashes. Predict G1/G2 pass from the
mechanics, then verify them on every case and a complete GGUF fixture.

G3, uncertain benefit: for an order/readout/width candidate, mean selection-output
NMSE over eight tensors and three seeds must be strictly lower than all three
conventional references and every control arm. Its summed complete raw bytes
must be <=110% of the smaller reference. Every tensor's seed-mean output NMSE
must be <=110% of that tensor's plain-scalar reference. No switch from the output
metric to weight error is permitted. This retains the prior rejection discipline.

G4: removal of every required domain/base/primitive, and each coupling/drive
control, changes the stored scale vector by >1e-7 relative norm and changes the
measured result on at least one tensor in every seed. Changed trajectories alone
cannot justify promotion. G5: exact 3120 unique count, all outcomes and hashes
retained, unchanged source/fit/selection/protocol/implementation, independent
artifact, actual-size and decoder audit. Count failed cases explicitly.

Failing any required gate rejects that principal candidate. No per-tensor winner,
quality-driven resource correction, relaxed gate, skipped control or relabeling
of a conventional reference as Atom is permitted. Full-model inference remains
conditional on a passing screen and a subsequent frozen model protocol.

Numerical preflight before implementation: the lower resource bound is 1/3;
constant maximal drive over time 2 gives 1/3+(2/3)*exp(-3)=.36652471224524263.
For w=[.2,-.5,1.75,2.5], x=[2,-1,.25,.75] and r=[.4,.7,.9,1], conductance and
resistance coordinate changes preserve the 3.2125 operator result within f32
rounding and recover weights with relative MSE below 1e-14. This predicts local
arithmetic correctness, not an improvement in quantization.
