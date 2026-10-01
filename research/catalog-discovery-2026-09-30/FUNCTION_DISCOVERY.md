# Catalog-driven function discovery

User steering: use the desktop first-principles/axioms collection to create new
functions that might advance the quantizer's goals. The authorized reference
folder is `/home/jesse/Desktop/first priniples`. It was inspected read-only;
96 material files excluding the ZIP are present. Selected-source hashes are in
[SOURCES.json](SOURCES.json). This was a targeted review, not a claim that every
catalog entry or physical analogy has been verified.

## What changes in the work

The failed B01–B03 trials explored mixing and scale conditioning. The reference
collection suggests additional function families: preserving shared directions,
forming representations through self-organization, and exploiting shared
residual structure. B04's paired-field protocol was committed as `aa57d5d` before
the new user steering, but remains **UNRUN and unimplemented**. Its gates and
declared cases have not been changed or presented as complete. The active goal
continues; the catalog review informs prioritization before further native work.

The folder contains both the broader mathematical primitive-kit/composition
approach and the stricter Atom synthetic-emergence rules. Keep their boundaries
explicit: a conventional observer or SVD baseline is not an emergent Atom
function. An Atom candidate still needs intrinsic mechanics, the two owning
crates, required domains, justified orders, matched controls and measured benefit.

## First new function: functional-error profiling — implemented

`experiments/error_geometry.py` composes scan/fold/project/compare and spectral
decomposition into an observer. It does not alter, select or correct weights.
It measures:

1. Actual operator error versus a diagonal prediction that ignores correlations.
2. Signed cross terms: whether errors cancel or reinforce within/across blocks.
3. How much selection error lies in low-rank output directions derived only from
   fit errors, plus unframed factor-storage cost as a clearly labeled estimate.
4. The difference between float64 linear algebra and actual f32 matmul scoring.

Tests explicitly distinguish cancellation from reinforcement, reject false
transfer into unseen selection directions, detect rank-one error, preserve
existing output files and reject nonfinite operator evaluation. All five new
tests and all 30 Python tests pass. The CLI analyzed 320 principal/scalar cases
from B02/B03, checked saved artifact identities and reproduced the original f32
output-error metrics. The final checked report is
[error-geometry-checked.json](error-geometry-checked.json); the earlier instrument
report is retained separately. [SUMMARY.json](SUMMARY.json) contains compact
selected examples.

For the sampled final FFN-down tensor at 4 bits, four directions fitted on the
fit-error matrix capture 99.325% of the scalar control's selection-error energy.
The B02 full order-1 cases average 99.615%; B03 inverse-resource order-1 averages
99.489%. These are operator-error subspaces, not proof that a four-mode model is
sufficient or that storing a correction improves a complete model. The original
weights, correction factors, decoding precision and all metadata must still be
charged in an actual codec trial.

On that tensor, output error relative to the diagonal prediction changes from
.929 in the scalar control to 1.127 in B02 and 1.218 in B03's inverse-resource
cases. The former errors partially cancel; the later errors reinforce. This
helps explain why lower weight error alone was misleading.

## Function hypotheses to develop

| Function hypothesis | Reference mechanisms | Possible app benefit | Required evidence before claiming benefit |
|---|---|---|---|
| Preserve important shared directions | Projection/SVD as observer and baseline; coherent propagation, coactivation memory and conservation as candidate native mechanics | Retain a small, important component while compressing less consequential variation | Useful directions must transfer to separate inputs; store/redecode the full representation and test actual bytes and inference |
| Form useful quantization groups/codebooks | Nucleation, phase separation, activator/inhibitor dynamics, competitive learning and information measures | Discover nonuniform groups or representational levels from interacting dynamics | Groups and level count must be observed rather than assigned as the desired answer; compare real packed artifacts to scalar/codebook baselines and domain/chaos controls |
| Represent shared residual structure compactly | Spectral decomposition, prediction/delta coding, memory and decay | Store common structure once and encode smaller differences | Prediction must transfer; include every factor, dictionary, index and residual byte; compare independent tensors and shuffled/incorrect-reference controls |

These are research hypotheses, not established functions or a construction
protocol. They broaden the search beyond another universal rotation or scalar
direction. Start with the shared-direction question because the new measured
profiles provide a concrete reason; do not choose mechanisms solely because a
catalog ranks them highly or because their names sound compatible.

Primary starting points include [Oja's 1982 activity-based feature-learning model](https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf)
and [Cahn and Hilliard's interfacial free-energy formulation](https://doi.org/10.1063/1.1744102).
They ground individual mechanisms; applying them to a new quantization function
is our untested engineering hypothesis, not a result established by those papers.

An eventual native candidate must receive raw observations and intrinsic state,
not desired codes, held-out answers, quality-driven repair instructions or a
target output. Conventional baselines may implement explicit objectives, but
remain labeled as baselines. The observer's finding of a low-rank error component
is not permission to hide a prescribed correction inside an Atom primitive.

## Source map and verification discipline

- `03_Root_Atoms_and_Master_Taxonomy/ROOT_ATOMS.md`: scan, fold, project, compare,
  combine and scale vocabulary for the observer.
- `04_Domain_Primitive_Reserves/linear-algebra-matrix/PRIMITIVES.md`: project-vec,
  SVD, eigen decomposition and QR; used for the diagnostic and future baselines.
- `04_Domain_Primitive_Reserves/information-theory-coding/PRIMITIVES.md`: residual
  coding, rate/distortion and relevance-preserving compression hypotheses.
- `04_Domain_Primitive_Reserves/physics-diffusion/PRIMITIVES.md`: reaction/diffusion,
  Turing instability and phase separation; candidate self-organization mechanics.
- Thermodynamics and cognitive reserves: symmetry breaking, coarse-graining,
  coactivation and competition; candidates requiring specific dynamics/controls.
- `06_Research_and_Emergence/Quantization App Development Research...`: telemetry,
  operator sensitivity and low-precision research context, not verified performance
  promises for this checkout.

Equations and claims must be checked. For example, the linear-algebra catalog's
statement equating symmetric-matrix eigenvalues with singular values needs the
positive-semidefinite qualification: diag(-2,1) has eigenvalues -2,1 and singular
values 2,1. Some PDF-extracted companions also contain broken font mappings.
The reference folder was not edited; do not treat corrupted extraction or an
analogy as an implementation specification. The glue-ranking document itself
labels its rankings heuristic and requires empirical validation.

## Reproduce the implemented observer

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_error_geometry.py' -v
.venv/bin/python -m experiments.error_geometry \
  --run artifacts/local-basis-2026-09-30 \
  --run artifacts/resource-discovery-2026-09-30 \
  --out artifacts/new-error-geometry.json
```

Output paths must be new. Fit directions and selection measurements remain
separate. No new compressed model or accepted quantization improvement was
produced by this diagnostic pass.
