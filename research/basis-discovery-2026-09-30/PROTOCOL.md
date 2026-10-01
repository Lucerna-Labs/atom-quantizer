# B01: composed basis dynamics for quantization

Status at precommit: UNRUN. User objective: improve this app through automatic,
trial-and-error discovery using Jesse's architectures. This is the first native
Atom candidate in this conventional codec checkout. Existing codecs, previous
experiments and the full22 objective remain intact.

## Ownership and unknown

Two new owning crates: `crates/basis-primitives` (intrinsic mechanics, no external
dependencies, at most 500 Rust lines including tests) and `crates/basis-engine`
(content-independent composition and execution, depending only on primitives).
The conventional app example and Python evaluator are observers/integration.
The stack receives 32 real stimulus values, a seed and a fixed composition. It
never receives bitwidth, reconstruction error, desired codes, quality thresholds,
model names, labels or a target matrix. It emits its accumulated transport.

Unknown: can interacting conservative transport, binding and activity memory
produce coordinates that improve the existing packed scalar codec? Orthogonal
transport alone is not a discovered quantizer. Success must be measured outside
the component after actual serialization and blind decoding. Failed candidates
stay research artifacts. A new function class or physical quantum benefit is not
implied by this experiment.

## Mechanics and domains

The common state carries 32 real amplitudes, a 32x32 transport matrix initialized
to identity, binding occupancies, activity resources, oscillator coordinates and
momenta, kicked-rotor phases/momenta and logistic populations. Every amplitude
rotation also left-multiplies the transport by the identical rotation. The state
is classical native Rust execution. Its matrix is saved in each codec record;
the decoder does not need the state, source weights, executable or calibration.

| ID | Intrinsic mechanic and interaction | Invariant and falsifier |
|---|---|---|
| H | Quantum-derived tensor-product Hadamard amplitude interference, once at initialization | Conserves squared amplitude norm; remove H independently and with Q |
| Q | Real two-level unitary transport: `(a,b) -> (cos(t)a-sin(t)b, sin(t)a+cos(t)b)`; t=dt*r_i*r_j*(1+b_i+b_j) on neighboring disjoint pairs | Orthogonal transport; chemistry/resource removal must change the measured dynamics to count |
| C | Chemistry, reversible occupancy: db/dt=p*(1-b)-0.5*b; exact constant-drive flow over dt, with p from current amplitude energy | b in [0,1], free+bound=1; remove C |
| R | Cognitive-science activity memory, reduced continuous-rate synaptic depression: dr/dt=0.5*(1-r)-b*r, exact flow; resource gates Q | r in [0,1]; response history must matter; not a full neuron or cognition claim |
| I | Information-theory self-information u=-ln(p), with p=(a^2+1e-12)/(sum(a^2)+32e-12); adjacent potential differences couple into conservative amplitude rotations t=dt*(u_j-u_i) | Finite conserved transport, not an entropy optimizer or quality objective; remove I |
| P | Physics, driven harmonic transport: v+=dt*(b_i-b_j-x); x+=dt*v, followed by t=dt*x rotation on shifted pairs | Symplectic oscillator step with finite state; forcing exchanges energy, so oscillator-energy conservation is not claimed; remove P |
| K | Mechanics/chaos: Chirikov kicked rotor, v+=5*sin(q), q+=v, reduced modulo 2pi; phase drives t=dt*sin(q) rotations on stride-two pairs | Bounded phases, conservative amplitude transport; remove K and change its stage position |
| L | Nonlinear population chaos: z'=3.9*z*(1-z); population phase drives t=dt*(z-0.5) on stride-four pairs | z in [0,1]; remove L and change its stage position |

dt=0.125; 16 repeated cycles. Initial bound=0, resources=1, spring x=v=0,
rotor v=0. Seeded rotor phase and logistic population use fractional multiples
of sqrt(2), independent of measured quality. Initial amplitudes are normalized
stimulus; zero stimulus fails closed. Stimulus is the mean weight over selected
rows and column blocks of width 32, before any quantization. No parameter search
or target-error feedback is allowed in B01.

Each chosen mechanic has a distinct interaction: H/Q modify interference; current
amplitude energies drive occupancy; occupancy drives resource depletion and
mechanical forcing; resources/occupancy alter subsequent transport; surprise,
oscillator displacement and two nonlinear phase processes alter later amplitude
interactions. Domain names are not acceptance evidence. Their ablations measure
whether these implemented interactions contribute or merely complicate the run.

Sources for the individual mechanics (the composition is an untested hypothesis):
[IBM RY](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYGate),
[binding kinetics](https://pmc.ncbi.nlm.nih.gov/articles/PMC5521016/),
[Tsodyks and Markram 1997](https://doi.org/10.1073/pnas.94.2.719),
[Shannon 1948](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf),
[Chirikov and Shepelyansky](https://var.scholarpedia.org/article/Standard_map),
[May 1976](https://doi.org/10.1038/261459a0).

## Two orders and matched controls

H initializes the quantum-derived substrate in both orders. The cycles are:

1. Q,C,R,I,P,K,L: binding and memory observe the transported field before chaos.
2. Q,K,L,C,R,I,P: binding and memory observe the chaos-perturbed transported field.

These are the only declared orders. Both preserve Q as a base. Their disagreement
tests the placement of both chaos mechanisms relative to occupancy and memory;
factorial exploration is not required by the revised Rule 2.

Thirteen arms per order: full; each of the eight single-primitive removals H/Q/C/
R/I/P/K/L; no-base (H+Q removed); no-chaos (K+L removed); base-only (only H+Q);
identity (all removed). No-K and no-L are the matched single-chaos controls.
All simpler arms are controls. The complex full candidate is the principal arm.

Seeds 1,2,3; the existing eight `catalog.TENSORS`; the same up-to-256 evenly spaced
rows per tensor; bits 3 and 4, group32, fitted scales and nearest rounding from
the existing scalar encoder. Real fit and selection activation captures remain
separate and source-bound. Two conventional references: no transform and signed
Hadamard32 (seed0). No baseline quality floor is weakened.

Bound: 2 orders * 13 arms * 3 seeds * 8 tensors * 2 widths = 1,248 candidate
record trials, plus 2 references * 8 tensors * 2 widths = 32 reference trials.
Exactly 1,280 scored records or explicit failure receipts are required. Basis
generation is shared across the two bitwidths (624 native component executions).
Records include actual packed codes, f16 scales, f32 saved matrix and metadata.
Artifact bytes include all of this state; no hypothetical compact representation
is credited. Matrix application to weights and fit inputs uses the same saved
f32 matrix. Inverse application uses its transpose, whose finite-precision error
is measured. Existing full22 configurations retain their own behavior.

## Frozen predictions, gates and falsifiers

- G1: native f64 transport max |U U^T-I| <= 1e-10; every state finite; occupancy
  and resource bounds hold. Every arm/seed/tensor must pass. No post-hoc QR repair.
- G2: saved f32 transport max |U U^T-I| <= 2e-6 and unquantized reconstruction
  relative MSE <= 1e-11. Every record decodes without source/state; shape and
  checksums must match. Exact numerical equality is required on repeated decode.
- G3 (benefit hypothesis): at either width, a full order's mean selection-output
  NMSE across tensors and seeds is strictly lower than both conventional
  references, no-chaos, each single-chaos, no-base, base-only and every individual
  domain/primitive ablation for that order. Complete raw-record bytes summed over
  tensors/seeds must be <= 110% of the smaller reference. Worst per-tensor
  mean selection-output NMSE must not exceed 110% of signed-H32's corresponding
  tensor. Results at both widths and for both orders are retained regardless.
- G4: every required domain and the base alters the emitted matrix and measured
  result beyond 1e-7 relative matrix distance on at least one tensor in each seed.
  This is only a load-bearing check; changed outputs alone do not pass G3.
- G5: source, calibration, selection, protocol and implementation hashes recorded;
  exact 1,280 count and all failures present; no edits to these gates after data.

Prediction: G1/G2 should pass from orthogonal mechanics; G3 is explicitly uncertain
and may fail due to added state cost or destructive mixing. G4 may expose a weak
or redundant domain. A failure is not repaired by changing gates or forcing a
matrix. Only a candidate passing all gates proceeds to complete-model selection-
split inference, followed by separately held-out evaluation. No screen result
alone is a production promotion or completion of the user's app-improvement goal.

Independent numerical preflight (before implementation): 2x2 H orthogonality
error 2.220446049250313e-16; RY(t=.125) error 1.1102230246251565e-16;
C(b=0,p=.25,dt=.125)=.029829879539988613; R(r=1,b=.25)=.9701701204600114;
K(q=.3,v=0) yields q=1.7776010333066978,v=1.4776010333066978;
L(z=.3)=.819. These numerical fixtures validate local equations, not G3.
