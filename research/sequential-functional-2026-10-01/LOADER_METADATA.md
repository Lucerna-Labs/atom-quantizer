# S1 post-trial loader metadata hardening

Before editing the frozen trial implementation, a small actual Llama forward-pass
control reproduced a new-loader defect: hidden size16, Q matrix16x16 and KV matrix
8x16 can fit either4/2 attention/KV heads or8/4 heads. The current loader accepted
the different configuration and its20 stored tensor arrays roundtripped, yet
logits differed by up to0.09107697010040283. Shape checks alone do not verify the
operator. The probe is retained in artifacts/sequential-functional-2026-10-01/
loader-metadata-probe.json. The fixture had no architecture dimension metadata;
that omission must also be rejected instead of treated as matching metadata.

The S1 real trial uses a fixed original checkpoint and an exact native W1 decode
with the original source header. Verify that header equality and every supported
architecture setting before relying on the trial. Preserve and audit the frozen
trial source before applying this hardening. Do not rerun or reinterpret quality
scores because of a check-only change to already matching inputs.

Fix: require matching standard Llama GGUF dimensions, context, vocabulary, Q/K/V
head dimensions, head counts, norm epsilon and default RoPE base before copying
any parameter. Reject missing, different and unsupported architecture state;
require the supported SiLU/no-bias/default-RoPE configuration. Keep complete
weight/alias/finite checks. The matching tokenizer remains the supplied fixed
checkpoint tokenizer, recorded in the trial source manifest.

Predictions: the wrong-head, missing-field and wrong-norm controls fail before
mutation; original W and W1 load successfully with matching metadata. The W1
endpoint selection capture is repeated after hardening and must reproduce its
payload hash exactly. This is a conventional input-validation fix, with no
quantization math, fidelity floor, trial gate or accepted recipe change.
