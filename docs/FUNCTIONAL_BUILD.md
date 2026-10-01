# Build the verified functional profile from the main app

```bash
target/release/atom-quantizer build-functional \
  models/SmolLM2-135M-Instruct-f32.gguf \
  --base-calibration artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal \
  --residual-calibration artifacts/full22-2026-09-06/fit.acal \
  --out artifacts/new-functional-model.a22
```

The archive destination must be new. The default work directory is the archive
path plus `.work`; it must also be new. Use `--work-dir NEW_DIRECTORY` to choose
another location. Work files, fitting stages and logs are retained on success or
failure. Completed JSON goes to stdout; progress and errors go to stderr.

This command fixes the D05 profile: matrices start at six bits, with a rank-two
activation-output correction; only native fitting failures advance to eight bits
or exact storage. The base and residual observations are separate, source-bound
files with distinct contents. Both are always checked by the native fidelity rules.
No option bypasses a view or loosens the thresholds.

The existing Python engine performs fitting. By default, the app uses this
checkout's `.venv/bin/python` and engine directory. `--python PATH` and
`--engine-dir DIRECTORY` support explicit installations. The engine must contain
`experiments/full22/harness.py` and its dependencies. Invocation uses argument
vectors and an explicit isolated module context, not a shell. This is operational
Python fitting, not a claim of a native Rust SVD encoder.

Before publication the Rust app independently:

1. Checks source/calibration identities and the fixed profile metadata.
2. Reconstructs the archive through its native A22-2 decoder.
3. Applies the actual native fidelity rules to that reconstruction on both views.
4. Rechecks inputs, fitting source files, interpreter and executable identities.
5. Copies and hashes a staged archive, then publishes to a new destination.

A backend exit or `FIT_ADMITTED` message cannot bypass these checks. Missing or
invalid output, wrong provenance/profile, failed native gates, changed inputs or
destination conflicts fail the command. Diagnostic work remains available.

The work directory contains `request.json`, backend logs, the engine's stages,
the native `native.gguf`, two `native-fit-*.json` reports, `verified.json` and,
on successful publication, `result.json`. `verified.json` records completed native
checks before publication; the final `PUBLISHED` receipt identifies the output.
If writing the final receipt fails after publication, the command explicitly
reports that the archive was published and where the verified receipt remains.

Consume the result without the fitting environment:

```bash
target/release/atom-quantizer verify artifacts/new-functional-model.a22
target/release/atom-quantizer decode artifacts/new-functional-model.a22 \
  --out artifacts/new-functional-model.gguf
```

The actual I2 main-app build recreated D05's archive and decoded GGUF byte-for-byte.
Thus its measured 13.16% archive reduction and 8.16% lower final PPL against the
historical reference apply to the same weights. Different models or captures
still require their own quality evaluation; native fitting checks alone do not
guarantee whole-model quality. OQ03 encoding, packed inference and resident-memory
claims remain separate. [I2 evidence](../research/functional-build-2026-10-01/OUTCOME.md).
