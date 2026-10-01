# Native-gated adaptive precision

The experimental `build-adaptive` command builds complete A22-2 models using a
fixed precision ladder and the main Rust codec's native fidelity decisions.

```bash
.venv/bin/python -m experiments.full22.harness build-adaptive \
  --source models/SmolLM2-135M-Instruct-f32.gguf \
  --base-calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal \
  --residual-calibration artifacts/full22-2026-09-06/fit.acal \
  --gate-calibration artifacts/full22-2026-09-06/fit.acal \
  --gate-calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal \
  --assessor target/native-admission-current/release/atom-quantizer \
  --out artifacts/new-adaptive-model
```

The output directory must be new. Supply source-bound observations and a native
binary containing the `assess` command. The example uses the verified N1 build;
`cargo build --release --target-dir target/native-admission-current` builds it
from the current source. Omit `--residual-calibration` for the scalar control.

Matrices start at 4 bits. With a correction file, each uses the existing rank-two
activation-residual correction fitted on that file. The base uses only its own
observations. Every complete stage is exported with decoded-tensor commitments
and assessed on every supplied gate-calibration file. A failure in any view moves
that tensor one step along **4 → 6 → 8 → exact**. Passing records stay unchanged.
Non-matrix tensors remain exact. No gate is loosened, and no tensor names are
hard-coded as exceptions.

`--start-bits 6` or `--start-bits 8` chooses a higher initial tier while preserving
the same remaining ladder and native thresholds. With a residual-calibration file,
`--rank-geometry diagonal` selects the existing diagonal-weighted rank-two
correction; `activation` retains the functional-output geometry. The default is
still start4/activation, and geometry cannot be requested without correction
observations. A geometry is fixed for the whole build.

Native report identities, model coverage, thresholds, scores and decisions are
validated before use. Execution errors stop the build. A failed exact tensor
also stops it. The builder retains all stages and native reports, so rejected
representations remain inspectable. Verified record reuse requires matching
source, calibration, independent configuration and decoded commitments; predictors
and external-file-dependent transforms are outside this reuse path.

After every fitting view passes, the directory contains `model.a22`, `model.gguf`
and `result.json`. `FIT_ADMITTED` describes only those fitting observations. It
does not claim unseen-input fidelity, perplexity, native OQ03 integration or
compressed-kernel execution. The intermediate `stage-*` files are trial artifacts.

Model-quality studies must keep gate-fitting observations separate from selection
and final evaluation. [D04's frozen protocol](../research/adaptive-fidelity-2026-10-01/PROTOCOL.md)
defines those additional checks, comparisons and whole-file byte accounting.

Decode the self-contained archive independently:

```bash
.venv/bin/python -m experiments.full22.harness decode \
  artifacts/new-adaptive-model/model.a22 artifacts/new-decoded-model.gguf
```

No source weights or calibration inputs are required for decoding. The export
checks every saved decoded-tensor commitment and rejects changed reconstruction.
