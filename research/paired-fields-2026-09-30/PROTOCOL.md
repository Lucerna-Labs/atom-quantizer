# B04: separate interacting weight and activation fields

Precommit status: UNRUN. Previous goal turn: progress, through B03 implementation,
complete execution and audited rejection. B01–B03 remain failed experiments with
unchanged receipts. The automatic app-improvement goal and full22 scope remain
active. No standalone screen constitutes full-model improvement.

## Unknown and ownership

B03's universal resource/inverse-resource readouts traded embedding damage for
FFN improvements. Its product stimulus erased the distinction between weight
and activation structure. B04 keeps those two observations separate and asks
whether their interacting resource dynamics yield useful relative conditioning.

Use the same two owning Rust crates. One component contains two mechanical
fields; this is not a new third behavior-owning crate. Each field retains the
quantum-derived Z/Q base, C binding, R activity-memory, I self-information,
P harmonic dynamics and K/L chaos mechanisms. The new interaction is symmetric
chemical occupancy exchange, implemented in the primitive crate. The engine
composes and observes the fields, with no quantization score, desired scale,
target codes, bit count, model label or selection-input access.

The conventional codec uses one fixed readout, d=r_weight/r_activation, stored
as f32. It does not choose a readout per tensor or score. This ratio is an outer
reversible coordinate-unit experiment, not a claim that the entire quantizer
emerges from the native component.

## Inputs and intrinsic interaction

Each 32-column block supplies two raw input fields:

- Weight field: sqrt(mean_rows(W^2)) stimulus and that block's uncentered weight
  correlation W^T W / sqrt(diag products).
- Activation field: sqrt(mean_fit_samples(X^2)) stimulus and the block's
  uncentered fit-activation correlation.

Zero-energy lanes have zero pair coupling; a wholly zero stimulus fails closed.
No stimulus is optimized against reconstruction error. The native mechanics,
initial states, dt=.125, 16 cycles and seeds 1/2/3 remain those of B02/B03.

Exchange E is the exact constant-rate mass-action flow for equal-volume
compartments: db_w/dt=k*(b_x-b_w), db_x/dt=k*(b_w-b_x), k=.5.
For each corresponding site, mean=(b_w+b_x)/2 and
delta=(b_w-b_x)/2*exp(-2*k*dt); new occupancies are mean+delta and mean-delta.
This conserves combined occupied amount during E, contracts the concentration
difference, and preserves [0,1]. Binding with each field's own reservoir still
changes occupancy between exchanges; full-run bound-mass conservation is not
claimed. E does not touch amplitudes, resources, weights or target scales.
Its subsequent influence occurs through C/R/Q/P interactions.

Z initializes each active field. Two fixed pair schedules are declared:

1. Q,C,E,R,I,P,K,L — memory observes freshly exchanged occupancy.
2. Q,C,R,I,P,K,L,E — exchange acts at the cycle boundary before the next Q/C.

Q/C/R/I/P/K/L acts on both active fields independently; E alone couples them.
Moving E changes its position relative to both chaos mechanisms and memory.
No other order or rate is searched. The same seed is used in both fields so an
unrelated seed mismatch is not credited as coupling benefit.

## Controls and bounded scope

Eighteen arms per schedule: full; eight simultaneous single-primitive removals
Z/Q/C/R/I/P/K/L in both fields; no-base Z+Q; no-chaos K+L; base-only Z+Q; identity;
disconnected (both fields, no E); weight-only; activation-only; uniform coupling
within both fields; shuffled within-field couplings via pi(i)=(5*i+7) mod32.

Standalone controls execute only their actual active field. The absent field is
explicitly represented as unit resource in the outer ratio: d=r_weight for
weight-only, d=1/r_activation for activation-only. It is not reported as an
executed native field. All other arms execute two fields. E runs only in the
full/ablated/uniform/shuffled coupled arms, including zero-change identity and
base-only controls. Disconnected/standalone schedules should agree because they
contain the same per-field order once E is removed; test this prediction.

Use the same eight catalog tensors, source-bound fit and separate selection
captures, up-to-256 evenly spaced rows, 3-/4-bit fitted group32 nearest scalar
coding and seeds 1/2/3. References: scalar, signed H32 seed0 and diagonal .5.
All arrays, metadata and framing count in the actual stored byte measurement.

Exact scope: 2 schedules * 18 arms * 3 seeds * 8 tensors * 2 widths = 1728
candidate records, plus 48 references = 1776 unique scored conditions or explicit
failure receipts. Across 204 column blocks this is 22032 pair/block requests in
864 batches, containing 41616 actual active-field executions. Connected arms
perform 293760 exchange steps total (16 each); absent fields and disconnected
exchanges are not counted as executed. Each native result is shared across the
two code widths. This larger native count tests genuine multi-stack interaction,
not a parameter or factorial-order sweep.

## Frozen predictions and acceptance

G1: each active field remains finite with norm/transport-orthogonality errors
<=1e-10, occupancy in [0,1], resources in [1/3,1]. Each E conserves per-site
combined occupancy to <=1e-12 and preserves bounds. Native diagnostics must
distinguish absent fields and count exchanges actually performed. Predict these
mechanical invariants pass; test rather than assume them.

G2: stored f32 d is finite in [1/4,4], exactly the declared rounded resource ratio.
Unquantized reconstruction NMSE <=1e-11, exact repeated blind decode, proper
shape/hash checks and standalone complete-GGUF-fixture export. Decoding needs no
original model, calibration, native state/executable or external scale file.

G3, uncertain benefit hypothesis: at a width/schedule, full mean selection-output
NMSE across the eight tensors and three seeds must be strictly lower than every
reference and every control (including disconnected and both standalone arms).
Complete raw bytes must be <=110% of the smaller reference, and each tensor's
seed-mean output NMSE <=110% of its scalar reference. No metric substitution,
per-tensor winner, quality-directed readout or post-hoc threshold adjustment.

G4: every required primitive/domain/base removal plus disconnected, standalone
and within-field coupling controls must change the stored scale by >1e-7 relative
norm and change the measured result on at least one tensor in every seed. Useful
coupling requires G3, not merely different trajectories. G5: exactly 1776 unique
receipts, all failures preserved, hashes unchanged and independent artifact,
actual-size, decoder, ratio, active-field and exchange-count audit.

Any failed required gate rejects the principal candidate. Only a passing screen
may proceed to a separately frozen full-model protocol. Simpler controls are not
substitute principal candidates. The earlier failed trials are not rescued by
this new hypothesis.

Independent numerical preflight: E(.8,.2)=(.7647490707753787,.23525092922462137),
sum=1 and difference contraction exp(-.125)=.8824969025845955. No exchange and
either standalone mode must reproduce existing single-field mechanics exactly.
No numerical prediction of quantization benefit is asserted.
