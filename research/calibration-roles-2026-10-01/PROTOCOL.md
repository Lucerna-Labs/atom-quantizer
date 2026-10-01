# D03: separate base and correction observations

Precommit status: UNRUN. This tests the conventional primitive-kit/composition
branch identified in the user's first-principles collection. Projection, SVD,
residual coding and addition have explicit mathematical objectives here; this is
not an Atom synthetic-emergence claim. B01–B03 and D01–D02 retain their failed
benefit statuses; B04 remains UNRUN. Main-codec safety gates remain unchanged.

## Hypothesis and function contract

D02 demonstrated that broader token coverage alone can seriously harm model
quality. D01's endpoint-fitted correction also transfers poorly to interior
observations. Hypothesis: use interior observations to fit the base quantizer,
then preserve endpoint responses in an independently fitted stored correction.
Neither role is sufficient evidence of quality by itself.

The encoder will accept separate base and residual calibration inputs. Only the
residual factorization sees the residual inputs. Existing transforms, scalar
fitting and other steps keep using base inputs. Omitting the new argument must
preserve the existing encoder behavior exactly. Explicit missing, nonfinite,
misaligned or wrong-source residual calibration must fail, not fall back silently.

For E=W-Q, Y=E X_correction^T, take the leading left singular vectors U and
store f16 U and B=U^T E, as in D01. Decode Q+U B using stored values only.
Changing correction observations must leave all base codes, scales and levels
bit-identical. Both calibration hashes must be retained in the complete archive.

## Frozen design

Source: existing SmolLM2-135M-Instruct F32 GGUF, SHA256
53f86c219b099fa84cd005f1b22572f6c4e53f8ba242fb8074f600712abe9c5b.
Endpoint fit: original fit.acal, SHA256
f5e6d79eb6b552da48cf976ea4c114d0ac387689ef42de5fa8746e257056431f.
Interior fit: D02 stratified seed0 fit, SHA256
29d09e1701dc329cee26fd636dd1d492c3e29b1e1c3d17d28e2c777a52e9a2c4.
Each contains the same 256-observation budget per operator from the same fit
documents. No new capture, position tuning or selection/final input fitting.

At each of 3 and 4 bits, test ranks1,2,4 with four activation-residual roles:

- EE: endpoint base, endpoint correction (same-role control).
- II: interior base, interior correction (same-role control).
- IE: interior base, endpoint correction (principal candidate).
- EI: endpoint base, interior correction (reversed-role control).

That is 24 complete functional-residual settings. Also include six endpoint
diagonal-residual controls (3/4 bits, ranks1/2/4), four scalar controls (endpoint
and interior at3/4 bits), source F32 and historical budget120 references.
Same global setting on all2D tensors, existing exact handling on non2D tensors;
no per-tensor winner selection and no rank/geometry tuning beyond this matrix.

The four same-role rank1 activation models, two rank1 diagonal models, four
scalar models and source/history selection scores may be reused only after
verifying archive/decoded hashes, calibration/configuration identity, complete
272-tensor export, inference corpus/runtime and log hashes. Label reuse. Fresh
EE/II fixture encoding must reproduce old/default behavior first. This leaves
24 new complete models and 24 new selection evaluations, plus12 reused controls.
If a reused model fails identity, stop and retain evidence; do not substitute it.
At most two independent build/evaluation workers, each using4 CPU math threads.
This is not a timing benchmark. No native Atom equations are changed.

## Precommitted admission and final gates

Selection: original selection.txt, 64 chunks, existing llama-perplexity binary
and shared-library hashes from D01. CPU only, threads4, context/batch/microbatch512.
Evaluate all declared cases even if early candidates fail. Every required run
must be finite and complete; failed execution is INCONCLUSIVE, not benefit failure.

For IE at a given width/rank to qualify, require all of:

1. PPL <=95% of the best of EE, II and EI at that same width/rank.
2. PPL no worse than endpoint diagonal at the same width/rank, and no worse
   than the best previously tested endpoint rank1 correction at that width.
3. Total archive bytes <=110% of the smaller endpoint/interior scalar archive
   at that width. Include headers, entropy framing, factors and all tensors.

At each width select the smallest qualifying rank. If none qualifies, reject
this role-separation hypothesis for that width; do not evaluate its final stage.
The controls may reveal useful higher-rank leads, but cannot be relabeled as the
principal candidate or promoted by changing these gates after seeing results.

For each qualifying width evaluate the selected IE, its three same-rank role
controls, the same-rank diagonal control, both endpoint rank1 corrections and
both scalar controls on the D02 fresh suffix. Deduplicate exact same models.
Also evaluate source F32 and historical budget120 once. At most20 final runs.
Require the same PPL and byte gates on this fresh final corpus. A final failure
remains a failure even if selection passed. No new model is fitted after final
evaluation. Report selected rank, every control, bytes and PPL with claim limits.

Fresh corpus: artifacts/calibration-coverage-2026-10-01/fresh-final.txt, SHA256
42ad320614e02063e1a484efb02fcf2c58f81552aef71ef0468cbd4cb7f68bbd.
D02 native-token proof excludes50449 common prefix tokens, beyond D01's32768
input-token upper bound. D02 ran no model on this suffix. Verify its saved proof,
hash and absence of prior fresh-final evaluations before first use; keep record
when consumed so an adaptive successor cannot call it untouched.

## Integrity and verification

Freeze implementation, protocol, source, calibration, corpus, runtime and reused
receipt identities before real trials; recheck at completion. Validate both role
calibrations against the source before fitting and recheck bytes at publication.
Preserve earlier files, failed attempts and dirty work. No overwritten archives.

Tests: default compatibility; independent-role routing and unchanged base arrays;
explicit missing/misaligned/nonfinite/source-mismatched inputs; changed calibration
rejection; and complete archive export after deleting both calibrations and source.
Run the existing Python suite. Independently compare all new model base arrays
against their scalar control and all272 exported tensors against stored decode.
Audit actual receipt/log hashes, whole-file bytes and the frozen decision rule.

Predictions: independent correction fitting changes stored factors without
changing base codes; IE recovers useful boundary responses. The latter is an
untested model-quality prediction and may fail. No tensor proxy can substitute
for the declared PPL test. Dense-decoded inference does not prove packed-kernel
speed, resident-memory reduction, main-codec promotion or broad generalization.
