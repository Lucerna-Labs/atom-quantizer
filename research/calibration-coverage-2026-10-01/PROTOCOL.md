# D02: calibrated token-position coverage

Precommit status: UNRUN. The previous goal turn made progress through complete
D01 model tests and an audited rejection. This is a conventional primitive-kit
observation function, not an Atom-emergence claim. Prior trials and main-codec
KL/cosine gates remain unchanged; B04 remains UNRUN after the user's steering.

## Observed problem and hypothesis

The current collectors choose evenly spaced endpoints. With 256 samples across
128 fit documents, each contributes its first and last token. All 128 last-token
embedding IDs are 3717 (the paragraph-ending space/newline). Consequently half
the first-layer Q/K inputs are identical. In the final FFN-down tensor, first
endpoints account for 98.701% of fit input energy and 98.689% of selection energy.
These are measured sampling facts, not proof that they caused D01's model loss.

Hypothesis: replacing endpoint-only sampling with deterministic stratified token
sampling at the **same sample budget** improves actual quantized-model quality.
Compose scan, hash, partition/order and record/fold primitives. No weight, target
label, reconstruction error or quality score participates in selecting positions.

## Sampling function

For an operator invocation of length L and budget k, n=min(L,k). Partition its
indices into n disjoint integer strata [floor(iL/n),floor((i+1)L/n)). Within each
stratum choose start + H mod width. H is the little-endian integer from an
8-byte BLAKE2b digest of UTF-8 "seed|document_key|L|i". document_key binds the
corpus SHA-256 and document index/ID. This is deterministic pseudo-random
stratification, not physical randomness or a chaos claim.

Record every selected position and available invocation length, plus strategy,
seed, corpus/source/implementation identity. Keep legacy endpoint sampling as
an explicit unchanged option and default. Existing captures must not be
overwritten. AC01's existing binary source-binding/checksum format remains valid.

Predictions: selected positions are distinct, ordered and within their strata;
the exact requested count is produced when enough positions exist; repeated
inputs reproduce exactly. The observations should cover document interiors,
but improved diversity or changed moments alone cannot establish model benefit.

## Frozen experiment

Use the same source SmolLM2-135M, fit.jsonl (128 documents), selection_inputs.jsonl
(32 documents), context512, sample caps256/64, eager CPU forward passes and
8 capture threads. Model weights must again match every source GGUF tensor.

1. Reproduce endpoint fit/selection captures with the extended collector. Require
   exact AC01 equality to the preserved originals before reusing D01 controls.
   If equality fails, retain diagnostics and rebuild/re-evaluate the six endpoint
   controls from the fresh endpoint captures; do not silently reuse old scores.
2. Capture stratified fit/selection at fixed seed0 and identical budgets. Re-score
   all 160 D01 screen records on the new selection observations as a diagnostic,
   without changing their old statuses or using this proxy to select models.
3. Build all six seed0 full-model configurations: scalar, diagonal rank1 and
   activation rank1 at 3/4 bits. Same global configuration on all 2D tensors,
   existing exact handling on non-2D tensors; only fit observations change.
4. Compare on the same 64-chunk selection corpus and D01's exact CPU runtime/
   settings (threads4, context/batch/microbatch512, no GPU). Verified unchanged
   D01 endpoint-model/control measurements may be reused and labeled reused.

Seed0 admission: for a setting, selection PPL <=95% of its endpoint counterpart,
no worse than the best endpoint setting at that width, and complete archive bytes
<=105% of its endpoint counterpart. At each width, select the admitted setting
with lowest PPL (then smaller archive, then scalar/diagonal/activation order).
This is a global setting, not per-tensor winner selection.

For admitted widths only, capture fit data at seeds1 and2 and build the selected
setting at each seed. Require all three seeds to meet the same selection gate.
A width failing this repeatability gate does not enter final evaluation. Failed
seeds/settings remain in the report; do not choose the luckiest seed.

On the fresh final corpus, evaluate source, historical budget120, all three
endpoint controls for each surviving width, and all three stratified-seed models
for its selected setting. Require every seed's PPL <=95% of its endpoint
counterpart and no worse than the best endpoint control at that width; preserve
the same size cap. Otherwise reject sampler promotion for that setting.

Bounds: four initial captures, six initial stratified models; at most two further
fit captures and four further stratified models. Up to six endpoint rebuilds
only if replay differs. Six seed0 selection evaluations plus at most four seed
repeats; endpoint/source/history selection results reused only after exact
identity verification, otherwise rerun at most eight controls. Final at most
14 models (source/history plus six endpoint and six stratified). All declared
and conditionally required cases must be retained and counted.

## Freshness and integrity

D01 consumed the first64 chunks of full22 final_test.txt (at most32768 input
tokens). Prepare a new suffix starting at the first newline after one third of
that file's characters. The installed native tokenizer must prove that the
excluded prefix shares at least33792 original token IDs and that the suffix
contains at least32768 input tokens. The 1024-token minimum gap guards the
boundary; do not use HF offsets as proof (its full stream differs by one token).
Record original/suffix/prefix/tokenizer hashes and exact cut location before
any model evaluation. In the read-only feasibility check the cut is character
219523, the common prefix is50449 native tokens and the suffix has101799 tokens.
No model was evaluated on this suffix during that check.

Stored model archives must be independently exported/redecoded with all tensors
verified. Captures must be finite, source-bound, immutable and accompanied by
position traces. Compare actual whole-file bytes, not estimated scale overhead.
Only real dense-decoded inference is measured; no compressed-kernel throughput,
resident-memory or broad architecture-generalization claim. An improvement in
sampling statistics alone cannot pass the goal's requested boundary.

## Required checks and failure retention

Before model trials: endpoint compatibility, deterministic/unique/stratum-correct
sampling, invalid input handling, source/output preservation and actual forward
capture checks. Frozen identities must remain unchanged during the experiment.
The existing toolkit function suite and codec tests remain gates. Results may
disprove the sampling hypothesis; preserve them without retuning the position
seed, metric, budgets or thresholds.
