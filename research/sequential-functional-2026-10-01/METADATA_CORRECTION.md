# S1 metadata correction before candidate fitting

The first run ended with KeyError(sampled_positions_before_cap) after source
installation, exact Composer 1 decode, an unchanged source selection replay and
one real W1 endpoint-fit capture. No candidate was built or scored. The process
is terminal (exit1); the original run directory, failure.json, log and
implementation-trial.tar.gz are retained.

The original September 6 endpoint manifests predate position reporting. D02's
endpoint-fit/selection replays include that field. Their AC01 payload hashes have
just been checked against the original files and are identical for both splits.
The correction uses those verified replay manifests for position comparisons,
freezes their metadata/payload identities, and independently repeats the checks
in the auditor. It does not replace observations, alter numerical settings,
relax a gate or discard any failure. A new run directory is required.

Original fit SHA f5e6d79eb6b552da48cf976ea4c114d0ac387689ef42de5fa8746e257056431f.
Original selection SHA d8a0411a3ac9b4033502dbec560b3a5390c5571d83c177f1a52260bd9a74a7d0.
Predicted corrected metadata gate: MEASURED, with equal token/document/position
selections and equal input payloads for the source replays. All original benefit
predictions and thresholds remain unchanged.
