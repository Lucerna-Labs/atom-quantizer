# N1: native decoded-model fidelity assessment

Precommit status: UNRUN. This adds a conventional read-only assessment function
to the main Rust CLI and measures the D03/R1 candidates at the native fidelity
boundary. It does not change their weights, the quality gates or the native Atom
component. The previous R1 turn was progress, with a verified export guard.

## Shared implementation and constraints

Add `atom-quantizer assess SOURCE.gguf CANDIDATE.gguf --calibration INPUT.acal`.
Report complete machine-readable JSON on stdout and progress/errors on stderr.
Successful completed assessment exits0 when every tensor passes and2 when any
fidelity gate fails; invalid input or execution failure exits1 without a complete
report. Require real source-bound calibration; do not silently use Gaussian data,
partial tensor limits, missing operators or a different model. Reject duplicate,
missing, extra or mismatched tensor shapes and nonfinite candidate values.

Share measurement and acceptance code with the existing quantizer. Reuse its
actual operator outputs, reference construction, KL, cosine and relative output
MSE functions. The CPU path must be explicit in the report. Preserve existing
thresholds and protected-tensor classification. The default output-MSE ceiling
is0.02 normal/0.005 protected; KL ceilings are1.0/0.25; cosine floors are0.99/0.995.
If the existing configurable output-MSE option is exposed, validate and report it;
the frozen experiment below uses defaults only. Do not change the comparisons.

The main codec uses embedding lookup observations and linear samples for other
matrices. It does not consume the extra tied-output-head entry as a second
operator. Report observation kinds/counts and unused calibration entries so that
this scope is visible. One-dimensional tensors use the existing histogram path.

Bind source, candidate and calibration hashes, tensor names/shapes/types, each
metric, actual thresholds, individual gate results and full counts. Recheck input
identities before printing the completed report. Assessment writes no models and
does not itself establish OQ03 representability, packed inference or model quality.

## Regression and interface checks

Before editing, build the current dirty source into a separate baseline target
directory and preserve that executable/source snapshot. After refactoring, require
exact OQ bytes and decoded GGUF equality for fixed fixtures under default,
calibrated, explicit scale/asymmetry, budget and exact-escape settings. Compare
selected tensor modes and scores too. Do not attribute existing dirty work to N1.

Test the real CLI on identical and altered matrices, finite gate failures, missing
or wrong-source calibration, duplicate/missing/extra/shape-mismatched tensors,
nonfinite weights, invalid flags and valid JSON escaping. A calibrated common-mode
shift that passes cosine/KL but fails output-MSE must remain rejected by the same
gate used by quantization. Run Rust workspace tests and relevant checks. No new
dependency is needed. Preserve the installed/incumbent release binary.

## Frozen real-model matrix

Use the existing F32 SmolLM2 source and four candidates: the source itself as an
identity control, historical budget120 as a product reference, and R1's unchanged
3-bit IE rank1 and4-bit IE rank2 decoded models. Historical budget120 has different
original fitting data and is not assumed to pass the current observations.

Assess each candidate on four existing source-bound calibration captures:
endpoint fit/selection and interior seed0 fit/selection. That is16 complete native
assessments,272 tensors each (4,352 total). Fit captures have256 samples/operator;
selection captures have64. No fitting or per-tensor representation selection in N1.
Freeze file hashes, source/compiler/executable identity and commands before runs.
At most two independent CPU assessment processes; no GPU or timing claim.

Every identity-control tensor must pass, with identity metrics. For other models,
report all misses by KL/cosine/output MSE and calibration regime. A failed native
gate is evidence for the next representation trial, not grounds to loosen it.
Independently audit complete reports, file identities, native score/decision
consistency and the source/candidate contracts. Do not infer admission from the
previous Python cosine screen or select an easier observation regime after seeing
the results. No new perplexity run or untouched final segment is consumed.

The goal remains broader than this diagnostic: a useful representation must still
meet native requirements and be integrated/verified at its eventual app boundary.
The original D03 reproducibility concern remains open; A22-2 contains changed
reconstruction at export. The wider full22 programme and B04 status are unchanged.
