# N1 outcome: native assessment implemented; candidates remain unadmitted

Precommit: `dae4053`, [protocol](PROTOCOL.md). The new main-CLI assessment function
is operational and uses the quantizer's actual measurement and acceptance code.
All sixteen complete assessments and the independent decision audit completed.
Neither D03/R1 candidate meets all existing native requirements. This does not
undo D03's measured size/PPL gains; it identifies why promotion remains incomplete.

## Shared native boundary

`atom-quantizer assess SOURCE.gguf CANDIDATE.gguf --calibration INPUT.acal` reads
complete models and reports JSON to stdout, with progress/errors on stderr.
`src/main.rs` now shares `measure_fidelity`, `FidelityLimits` and `Fidelity::gates`
between quantization and `src/assess.rs`. The assessor reuses `make_reference`,
`operator_outputs`, native KL/cosine/output-MSE functions and existing protection
classification. No copied Python metric is presented as native measurement.

The report binds source/candidate/calibration and header hashes, all tensor
names/dimensions/types, operator observations, sample counts, scores, actual
thresholds, individual decisions and aggregate counts. It rechecks input hashes
before emitting the complete report. Missing/ambiguous tensor sets, shape/type
errors, nonfinite candidates, wrong/missing calibration and invalid flags fail.
Unbounded output MSE is represented as null with an explicit state, not invalid
JSON or a passing gate. Assessment writes no model.

Exit 0 means every tensor passed, exit 2 means complete assessment with gate
failures, and exit 1 means invalid input or execution failure. Assessment uses CPU,
including in a CUDA-enabled build. [Usage and boundaries](../../docs/NATIVE_ASSESSMENT.md).

## Real-model matrix

Every row assessed all 272 tensors. Existing fit captures contain 256 samples per
operator, selection captures 64. These selection captures were already used in
prior research; they are not a newly untouched evaluation set. No fitting, new
perplexity run or final-holdout consumption took place in N1.

| Model | Observations | Passed | Failed | KL misses | Cosine misses | Output-MSE misses |
|---|---|---:|---:|---:|---:|---:|
| source-f32 | endpoint-fit | 272 | 0 | 0 | 0 | 0 |
| historical-budget120 | endpoint-fit | 272 | 0 | 0 | 0 | 0 |
| q3-IE-r1 | endpoint-fit | 61 | 211 | 0 | 211 | 36 |
| q4-IE-r2 | endpoint-fit | 268 | 4 | 0 | 4 | 1 |
| source-f32 | endpoint-selection | 272 | 0 | 0 | 0 | 0 |
| historical-budget120 | endpoint-selection | 272 | 0 | 0 | 0 | 0 |
| q3-IE-r1 | endpoint-selection | 61 | 211 | 0 | 211 | 36 |
| q4-IE-r2 | endpoint-selection | 268 | 4 | 0 | 4 | 1 |
| source-f32 | interior-fit | 272 | 0 | 0 | 0 | 0 |
| historical-budget120 | interior-fit | 272 | 0 | 0 | 0 | 0 |
| q3-IE-r1 | interior-fit | 61 | 211 | 0 | 211 | 112 |
| q4-IE-r2 | interior-fit | 247 | 25 | 0 | 4 | 22 |
| source-f32 | interior-selection | 272 | 0 | 0 | 0 | 0 |
| historical-budget120 | interior-selection | 272 | 0 | 0 | 0 | 0 |
| q3-IE-r1 | interior-selection | 61 | 211 | 0 | 211 | 111 |
| q4-IE-r2 | interior-selection | 248 | 24 | 0 | 4 | 21 |

Gate misses can overlap on one tensor, so their sum need not equal the failing
count. All four source-identity controls have identity metrics. Historical
budget120 passes all four current observation regimes, despite using different
original fitting data. Neither research candidate receives native admission.

The selected 4-bit candidate has four cosine misses across every regime:
`token_embd.weight`, `blk.0.attn_output.weight`, `blk.1.attn_output.weight` and
`blk.29.ffn_down.weight`. The embedding also fails lookup output MSE: approximately
0.0100/0.0101 on endpoint fit/selection and 0.0110/0.0117 on interior fit/selection,
against its protected limit of 0.005.

Interior fit data expose 21 additional attention-output projection failures from
output MSE, for 25 failing tensors total. Interior selection has 20 such additional
failures, for 24 total. The fitting-failure union contains every selection failure;
there are no selection-only failing names. This is a lead for a new representation
trial, not a preselected list of per-tensor winners or a performed repair.

All KL scores pass their existing limits. That does not override the independent
cosine/output-error requirements. The earlier four-tensor cosine screen was
insufficient to describe the full native admission boundary.

The main codec validates embeddings as lookups. In every report,
`token_embd.weight::output` is explicitly listed as unused. D03's functional fit
uses that extra tied-head view, which is a different operator observation. The
assessor preserves existing native semantics and makes this limitation visible.

## Regression and verification

The baseline release build from the pre-edit dirty workspace is byte-identical
to the incumbent release executable. Both identify as atom-quantizer 0.2.4. It and
its exact source snapshot are preserved under the N1 artifact directory.

Six before/after quantization cases (default, calibrated, max-abs scale,
asymmetric, exact escape and global byte budget) retained identical OQ bytes,
decoded GGUFs and reported tensor scores. No existing source outside main.rs was
changed; new Rust files are assess.rs and its CLI tests. The incumbent
`target/release/atom-quantizer` remains unchanged. The tested new executable is
`target/native-admission-current/release/atom-quantizer`.

All 69 Rust tests passed, including seven new CLI tests. They exercise identity
and JSON escaping, independent KL/cosine/output-error failures, protected limits,
embedding lookup semantics, unused tied-head data, unbounded output error,
incomplete/duplicate/extra/mismatched tensor sets, nonfinite values, missing or
wrong-source calibration and strict flags. The CUDA feature compiles; no live
GPU result is claimed.

The native reports were independently audited against file identities, GGUF
names/dimensions, observation kinds/budgets, actual f32 cosine thresholds and
f64 error limits. All 4,352 reported tensor decisions and aggregate counts match
the frozen rule. Every report is complete, with the expected exit status. All
model headers match the original source header in this experiment.

Changed-file rustfmt and git diff checks pass. Strict Clippy with -Dwarnings does
not pass because unchanged src/gguf.rs has six existing chunks_exact-to-as_chunks
style lints under Rust 1.98.1. Ordinary Clippy completes with those six library
warnings and no new binary warnings. A whole-workspace formatting check also
reported unrelated pre-existing Q2 formatting; those files were not reformatted.
These limitations are retained rather than mislabeled as globally green checks.

## Next work and claim limits

A new compression trial must address all native conditions across observation
views, particularly protected attention-output accuracy, while retaining the
successful boundary-response correction. Fixing only four cosine misses would
leave native output-error failures. A universal, precommitted precision or
representation rule can be studied using fitting data; do not hand-pick changes
from selection results or relabel the old D03 representation as admitted.

Native tensor fidelity is separate from OQ03 representability, packed-kernel
performance, main-codec integration and whole-model inference quality. The next
changed representation still needs its own actual artifact accounting and a new
untouched final segment. D03's consumed segment remains used. The initial D03
reproducibility concern remains open, with R1's A22-2 export commitments enforcing
consistency for new archives. The broad goal and full22 programme remain active;
B04 remains UNRUN.

## Evidence and reproduction

[SUMMARY.JSON](SUMMARY.JSON), [AUDIT.json](AUDIT.json), [REGRESSION.json](REGRESSION.json)
and [FAILURE_ANALYSIS.json](FAILURE_ANALYSIS.json) contain the compact results.
Complete stdout/stderr, native reports, source/compiler/executable fingerprints,
baseline source/binary snapshot and a before/after main.rs patch are preserved in
`artifacts/native-admission-2026-10-01/`. Rust test/check logs are under target/.
Only the new protocol was committed; previous dirty work and artifacts remain.

```bash
cargo test --offline --workspace --target-dir target/native-admission-current
rustfmt --edition 2021 --check src/main.rs src/assess.rs tests/cli_assess.rs
target/native-admission-current/release/atom-quantizer assess   models/SmolLM2-135M-Instruct-f32.gguf   artifacts/decoder-reproducibility-2026-10-01/models-check/models/q4-IE-r2.gguf   --calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal
```

The last command is expected to exit 2 with a complete failing assessment. The
frozen matrix runner requires its baseline/regression receipts and a new
evaluations directory; do not overwrite the completed N1 evidence to rerun it.
