# D03 outcome: quality gates passed; promotion remains open

Precommit: `565a0c8`, [frozen protocol](PROTOCOL.md). All 24 new model cases and
12 reused controls completed. Both selected configurations passed the declared
size and fresh-final PPL gates. One unexplained initial byte-comparison failure
remains an open reproducibility concern, despite successful full replays. This
is a measured research-codec improvement, not unconditional release validation.

## Implemented function

The primitive-toolkit composition now accepts independent observations for the
base quantizer and the stored residual correction. `codec.encode` adds a keyword
`residual_inputs`; the complete builder exposes `--residual-calibration`. Base
fitting still sees only base observations. Both source-bound calibration hashes
are stored and rechecked before publication. Missing, wrong-source, misaligned,
nonfinite and changed role files fail. Omission preserves the old input routing.

This composes the catalog's projection, spectral decomposition, residual coding
and addition mechanisms. It is an explicit conventional mathematical function,
not Atom synthetic emergence. No native Atom equations, main-codec defaults or
admission floors changed. Ten selected catalog source hashes remain unchanged.
See the [source map](../catalog-discovery-2026-09-30/FUNCTION_DISCOVERY.md).

EE uses endpoint observations for both roles; II uses interiors for both; IE
fits the base on interiors and the correction on endpoints; EI reverses IE.
The saved f16 correction is still U B, with U derived from (W-Q) X_correction^T
and B=U^T(W-Q). Decoding uses only the complete archive. The 8,160 corrected-model
base records checked in each successful audit exactly match their scalar controls.

## Selection matrix

Lower PPL is better. These are real 64-chunk complete-model CPU evaluations;
previous matching rank1/scalar controls were verified and explicitly reused.

| Bits | Rank | EE | II | IE candidate | EI | Endpoint diagonal | IE status |
|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 1 | 79.6965 | 484.6687 | 63.5125 | 69.9121 | 79.8858 | QUALIFIED |
| 3 | 2 | 76.7363 | 252.8625 | 55.6607 | 67.5575 | 75.7728 | QUALIFIED |
| 3 | 4 | 73.8688 | 207.8346 | 54.5375 | 64.0238 | 70.0476 | QUALIFIED |
| 4 | 1 | 34.0391 | 53.4461 | 32.1262 | 33.3114 | 33.8939 | FAILED |
| 4 | 2 | 33.6375 | 59.1437 | 31.2429 | 33.0536 | 33.3695 | QUALIFIED |
| 4 | 4 | 33.4055 | 52.7063 | 31.0350 | 32.3063 | 33.0654 | FAILED |

The precommitted smallest-qualifying-rank rule selected 3-bit rank1 and 4-bit
rank2. Four-bit ranks1 and4 failed the required 5% margin over the strongest
other role. Their absolute PPL improvements do not change their failed status.
Higher-rank 3-bit candidates were not evaluated on final after rank1 was chosen.

## Fresh final results

The D02-prepared suffix was unused before D03. All 18 required final runs
completed; it is now used evaluation data. Full source F32 PPL is 18.3348;
historical budget120 is 20.3825 with 125,809,756 stored bytes. The latter is a
product reference with different calibration, not a controlled role ablation.

| Model | Stored bytes | Selection PPL | Fresh final PPL |
|---|---:|---:|---:|
| historical-budget120 | 125,809,756 | 30.0986 | 20.3825 |
| q3-EE-r1 | 59,448,909 | 79.6965 | 57.2457 |
| q3-EI-r1 | 59,449,442 | 69.9121 | 50.1208 |
| q3-IE-r1 | 59,240,812 | 63.5125 | 44.9737 |
| q3-II-r1 | 59,240,770 | 484.6687 | 348.3955 |
| q3-diagonal-r1 | 59,446,131 | 79.8858 | 55.4800 |
| q3-scalar-E | 58,709,408 | 90.0866 | 63.6691 |
| q3-scalar-I | 58,500,756 | 485.1447 | 361.4485 |
| q4-EE-r1 | 73,065,954 | 34.0391 | 23.7608 |
| q4-EE-r2 | 73,769,974 | 33.6375 | 23.4656 |
| q4-EI-r2 | 73,771,183 | 33.0536 | 22.9624 |
| q4-IE-r2 | 73,268,797 | 31.2429 | 21.6630 |
| q4-II-r2 | 73,268,542 | 59.1437 | 41.1029 |
| q4-diagonal-r1 | 73,065,043 | 33.8939 | 23.4880 |
| q4-diagonal-r2 | 73,777,626 | 33.3695 | 23.0088 |
| q4-scalar-E | 72,316,615 | 35.2827 | 24.5831 |
| q4-scalar-I | 71,817,460 | 44.3903 | 30.7330 |
| source-f32 | 539,846,112 | 26.7164 | 18.3348 |

The 3-bit rank1 IE model stores 59,240,812 bytes (56.496 MiB). Its final PPL 44.9737
is 10.27% below the strongest other same-rank role (50.1208) and 18.94% below the
existing diagonal correction (55.4800). It uses 1.27% more bytes than the smaller
scalar archive and fewer bytes than either endpoint rank1 correction.

The 4-bit rank2 IE model stores 73,268,797 bytes (69.875 MiB). Its final PPL 21.6630
is 5.66% below the strongest other same-rank role (22.9624) and 5.85% below the
matched diagonal correction (23.0088). It uses 2.02% more bytes than the smaller
scalar archive. Relative to historical budget120 it is 41.76% smaller, with 6.28%
higher PPL; it does not dominate that product reference on both dimensions.

These are reductions in measured perplexity, not percentages of general model
quality. Evidence is limited to this source model, fitting observations, fixed
settings and evaluation corpus. Dense F32 exports were evaluated; packed-kernel
speed and resident-memory benefits were not tested.

## Validation and the retained failure

- All 45 Python tests passed, including five new calibration-role boundary tests.
- A real CLI build/decode smoke verified two fixture tensors after deleting the
  original GGUF and both calibration files.
- All 24 new complete models were exported with Python opens of source and both
  calibration files denied; each builder checked all 272 stored tensor values.
- There were 42 new successful inference runs: 24 selection and 18 fresh final.
  Another 12 selection controls were reused with identity/log checks.
- The first stricter byte-level audit stopped after q4-EI-r4, at the next model's
  tensor comparison. The original failure recorded neither the tensor name nor
  the differing values. Numerical equality at that failure is unknown.
- An isolated q4-IE-r1 decode and four more complete decodes at 1/4/8/4 threads
  found no byte differences. The instrumented full audit and an unchanged-source
  replay of the original auditor each passed 9,248 tensor comparisons across 34
  archives, 8,160 unchanged base records, every log/receipt and the independent
  decision calculation. No decoder, model payload, quality gate or threshold was
  changed between the failed audit and these replays.

The initial failure is **unexplained and open**, not erased by passing repeats.
Do not label it signed-zero behavior, rounding, hardware failure or a fixed bug
without evidence. Fixed decoded model hashes bind all reported PPL measurements.
See [REPRODUCIBILITY_CONCERN.json](REPRODUCIBILITY_CONCERN.json),
[AUDIT.json](AUDIT.json) and [original-auditor replay](AUDIT_ORIGINAL_REPEAT.json).
Failed/successful logs and original auditor source are preserved with the artifacts.

## Main-codec boundary and next work

The measured weight-cosine comparison still falls below current main-codec floors:
all 211 corrected 3-bit tensors, and four tensors in the selected 4-bit model:

| 4-bit tensor | Measured cosine | Current floor |
|---|---:|---:|
| token_embd.weight | 0.9932546212 | 0.995 |
| blk.0.attn_output.weight | 0.9932393892 | 0.995 |
| blk.1.attn_output.weight | 0.9947600755 | 0.995 |
| blk.29.ffn_down.weight | 0.9899663330 | 0.990 |

This is a measured cosine comparison, not a native OQ03 admission run. Native
KL/output-error checks and integration remain untested. No floor was relaxed.
Next work must resolve or bound the decode reproducibility issue and study a
precommitted fidelity-preserving representation before main-codec promotion.
The four 4-bit tensors are a concrete lead, not permission for post-hoc per-tensor
winner selection or a claimed completed successor.

The fresh-final SHA256 is 42ad320614e02063e1a484efb02fcf2c58f81552aef71ef0468cbd4cb7f68bbd.
D03 consumed its first 64 chunks (at most 32768 input tokens); adaptive successors
must not call this segment untouched. Earlier B01–B03/D01–D02 failures and B04's
UNRUN status remain unchanged. The broader improvement goal remains ACTIVE.

## Reproduction and evidence

[Usage](../../docs/EXPERIMENT_VALIDATION.md#separate-calibration-roles-d03) documents
both calibration arguments. The frozen runner and independent audit are:

```bash
.venv/bin/python -m experiments.calibration_roles --out artifacts/new-D03-run
.venv/bin/python -m experiments.audit_calibration_roles --run artifacts/new-D03-run
```

A fresh D03 run now rejects the already-consumed final suffix; do not bypass this
freshness guard. Use saved data for a labeled regression, or a new precommitted
holdout for adaptive research. The saved completed run can be audited directly
using `--run artifacts/calibration-roles-2026-10-01`. Source changes intentionally
invalidate its frozen current-workspace identity; exact sources are preserved in
`artifacts/calibration-roles-2026-10-01/implementation.tar.gz`.

Compact receipts are in this research directory; full models, captures references,
logs and evidence are under `artifacts/calibration-roles-2026-10-01/`. Only the
protocol was committed. Existing dirty work and earlier artifacts are preserved.
