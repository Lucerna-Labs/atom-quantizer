# B02: local, data-coupled basis dynamics

Precommit status: UNRUN. B01 is preserved as a failed experiment; its full
1280-condition result and implementation snapshot are not overwritten. The
automatic app-improvement goal and full22 scope remain open.

## Hypothesis and architectural boundary

B01's initial global Hadamard mixing and multiple-chaos field lost to the plain
scalar control. B02 tests a distinct hypothesis: interference through local
couplings derived from actual fit-input correlations may disturb useful weight
structure less while retaining interactions among all required domains.

Reuse the two owning crates `basis-primitives` and `basis-engine`. Add an explicit
local-coupling entry point, leaving B01's entry point and numerical behavior
intact. The component receives a 32-value stimulus, a symmetric 32x32 coupling
field, seed and fixed composition. No quantization error, codes, thresholds,
desired basis, model name or selection inputs enter the component.

The surrounding conventional observer computes **measurements**, not a target:
stimulus is RMS weight magnitude by column residue modulo 32 over the same
selected rows; coupling is the uncentered correlation of fit activations folded
into those 32 positions. For X with rows formed from each consecutive 32-column
block of every fit sample, C=X^T X/n and g_ij=C_ij/sqrt(C_ii*C_jj). An unexcited
feature has zero coupling. No optimization or parameter search uses these
measurements. Coupling is signed, fixed during a run, symmetric and bounded by
one; it multiplies the generator angle for each interacting pair, rather than
selecting a behavioral route.

## Mechanics and interactions

Replace B01's one-time global Hadamard H with one-time relative phase reversal
Z: odd amplitudes and their transport rows change sign. This preserves squared
norm and changes interference under subsequent pair mixing. It is the real
[Pauli-Z phase mechanic](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.ZGate),
not physical qubit execution. Q remains the real two-level unitary transport.
Z by itself is a control, not a claim of compression benefit.

Retain C binding, R activity-memory depletion/recovery, I self-information
potential, P driven harmonic transport, K kicked rotor and L nonlinear logistic
population from B01. Their exact equations, dt=.125, 16 cycles, parameter values
and intrinsic bounds remain as declared in B01. All amplitude rotations now use
angle*g_ij. Chemistry and memory observe the actual locally transported field;
their states influence Q and P. Both chaos processes are retained at two stack
positions. Domains remain quantum, mechanics/theory, physics, information theory,
chemistry and cognitive science. This is a changed interaction substrate, not
removal of required domains or a relabeling of the conventional scalar quantizer.

Initialization: normalized positive RMS stimulus; bound=0, resources=1, spring
position/velocity=0, rotor momentum=0. The same seed mapping based on fractional
multiples of sqrt(2) initializes rotor phase and population. Z is applied once.

Cycles: order 0 is Q,C,R,I,P,K,L; order 1 is Q,K,L,C,R,I,P. The comparison asks
whether binding and memory should observe the chaos-perturbed field. No other
order is tested, and no factorial or parameter sweep is claimed.

Fifteen arms per order:

- Full complex candidate.
- Eight single removals: Z, Q, C, R, I, P, K, L.
- No-base (Z+Q removed), no-chaos (K+L removed), base-only (Z+Q), identity.
- Uniform coupling (all g_ij=1) to test whether weak/data-dependent coupling
  contributes rather than merely providing another phase trajectory.
- Shuffled coupling: apply the fixed permutation pi(i)=(5*i+7) mod 32 to both
  coupling axes while preserving the stimulus coordinates. This preserves field
  values and symmetry but breaks their alignment to observed features.

The eight simpler removals, no-base, no-chaos, base-only and identity are controls,
not substitute principal candidates. Uniform/shuffled coupling use the full
complex stack. A coupling mismatch or a loss is retained as evidence.

## Frozen execution and acceptance

Use the same source, source-bound real fit and separate selection captures, eight
`full22.catalog.TENSORS`, up-to-256 evenly spaced tensor rows, seeds 1/2/3 and
3-/4-bit fitted group32 scalar codec as B01. References: plain scalar and signed
Hadamard32 seed0. Native outputs are rounded to stored f32 matrices, charged in
every record, applied identically to weights and fit inputs, then inverted by
transpose during blind decode. No matrix repair or output correction is allowed.

Exactly 2 orders * 15 arms * 3 seeds * 8 tensors * 2 widths = 1440 candidate
records plus 32 references = 1472 unique scored records or explicit failure
receipts. The same basis is shared across widths: 720 native component executions.

G1: native f64 max |U U^T-I| <= 1e-10, amplitude norm error <= 1e-10, finite
states and binding/resource/population bounds. G2: stored f32 matrix error <=
2e-6, unquantized reconstruction relative MSE <= 1e-11, correct shape, verified
artifact hashes, exact repeated decode without native executable/source state.
Predicted G1/G2 outcome: pass; these follow the preserved conservative mechanism
and remain independently tested, not assumed.

G3, uncertain benefit hypothesis: at a given width, the full order's mean
selection-output NMSE over all eight tensors and three seeds must be strictly
lower than both conventional references and every control arm. Complete raw
record bytes summed over tensors/seeds must be <=110% of the smaller reference.
Each tensor's seed-mean error must be <=110% of that tensor's plain scalar
reference. Both orders and widths are reported regardless of outcome.

G4: each base/domain/primitive removal and both coupling controls must change
the matrix by >1e-7 relative norm and the measured result on at least one tensor
in every seed. Changed matrices without G3 benefit do not justify promotion.
G5: all 1472 unique cases and failures retained, source/fit/selection/protocol/
implementation hashes unchanged, and independent artifact/size/decoder audit.

Failing any required gate rejects that principal candidate. No per-tensor winner
selection, tuned quality constants, looser thresholds or post-hoc success metric
is allowed. Only passing candidates may proceed to complete-model inference;
this screen alone cannot complete the app-improvement goal.

Independent numerical preflight: for centered feature columns [1,-1,2,-2] and
[2,-2,1,-1], g01=.8; the zero feature has zero coupling. At dt=.125 the first Q
angle is .1 with initial resource=1/bound=0. Applying Z then this Q to [.6,.8]
gives [.676869232484278,-.7361032822343239], with orthogonality error below 1e-16.
These fixtures validate local arithmetic, not the benefit hypothesis.
