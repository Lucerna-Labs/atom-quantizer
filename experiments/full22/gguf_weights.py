"""Install a complete decoded Llama GGUF into its actual HF architecture.

This is an explicit model-loading utility, not an activation-header rebinding.
The caller still runs real forward passes to collect new observations.
"""
import hashlib

import gguf
import numpy as np
import torch

from scripts.capture_calibration import converted_weight


def validate_architecture(model, reader):
    architecture = reader.get_field("general.architecture")
    config = model.config
    if architecture is None or architecture.contents() != "llama" or config.model_type != "llama":
        raise ValueError("decoded model installation requires the matching Llama architecture")
    rope = getattr(config, "rope_parameters", None)
    if (not isinstance(rope, dict) or rope.get("rope_type") != "default"
            or set(rope) != {"rope_type", "rope_theta"} or config.hidden_act != "silu"
            or config.attention_bias or config.mlp_bias):
        raise ValueError("unsupported Llama activation, bias or RoPE configuration")
    expected = {
        "llama.block_count": config.num_hidden_layers,
        "llama.context_length": config.max_position_embeddings,
        "llama.embedding_length": config.hidden_size,
        "llama.feed_forward_length": config.intermediate_size,
        "llama.attention.head_count": config.num_attention_heads,
        "llama.attention.head_count_kv": config.num_key_value_heads,
        "llama.rope.freq_base": float(np.float32(rope["rope_theta"])),
        "llama.attention.layer_norm_rms_epsilon": float(np.float32(config.rms_norm_eps)),
        "llama.attention.key_length": config.head_dim,
        "llama.attention.value_length": config.head_dim,
        "llama.vocab_size": config.vocab_size,
        "llama.rope.dimension_count": config.head_dim,
    }
    actual = {key: field.contents() for key, field in reader.fields.items() if key.startswith("llama.")}
    if set(actual) != set(expected):
        raise ValueError("missing or unsupported GGUF Llama architecture fields")
    for key, value in expected.items():
        if not np.isscalar(actual[key]) or actual[key] != value:
            raise ValueError(f"GGUF/checkpoint architecture mismatch: {key}")
    return expected


def inverse_weight(weight, name, config):
    if name.endswith(".attn_q.weight"):
        heads = config.num_attention_heads
    elif name.endswith(".attn_k.weight"):
        heads = config.num_key_value_heads
    else:
        return weight
    if weight.ndim != 2 or heads <= 0 or weight.shape[0] % (2 * heads):
        raise ValueError(f"invalid Q/K dimensions: {name}")
    rows, columns = weight.shape
    return weight.reshape(heads, rows // heads // 2, 2, columns).swapaxes(1, 2).reshape(weight.shape)


def install(model, reader):
    architecture = validate_architecture(model, reader)
    source = {tensor.name: tensor for tensor in reader.tensors}
    if len(source) != len(reader.tensors):
        raise ValueError("duplicate source tensor")
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, model.config.num_hidden_layers)
    parameters = list(model.named_parameters(remove_duplicate=False))
    aliases = dict(parameters)
    plans, covered, skipped = [], set(), []
    for name, parameter in parameters:
        mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if mapped not in source:
            if (name == "lm_head.weight" and model.config.tie_word_embeddings
                    and "model.embed_tokens.weight" in aliases
                    and parameter.data_ptr() == aliases["model.embed_tokens.weight"].data_ptr()):
                skipped.append(name)
                continue
            raise ValueError(f"unmapped model parameter: {name}")
        tensor = source[mapped]
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32 or tensor.data.dtype != np.dtype("<f4"):
            raise ValueError(f"decoded source must be F32: {mapped}")
        actual = tensor.data
        if tuple(parameter.shape) != actual.shape or parameter.dtype != torch.float32:
            raise ValueError(f"source shape/dtype differs: {mapped}")
        if not np.isfinite(actual).all():
            raise ValueError(f"nonfinite source: {mapped}")
        converted = inverse_weight(actual, mapped, model.config)
        plans.append((mapped, parameter, converted))
        covered.add(mapped)
    if covered != set(source):
        raise ValueError(f"source tensors not installed: {sorted(set(source) - covered)}")
    # Validate the complete mapping before changing any model parameter.
    with torch.no_grad():
        for _, parameter, values in plans:
            parameter.copy_(torch.from_numpy(np.array(values, copy=True)))
    receipts = {}
    for mapped, parameter, _ in plans:
        weight = parameter.detach().cpu().numpy()
        converted = converted_weight(weight, mapped, model.config)
        values = weight if converted is None else converted
        if values.tobytes() != source[mapped].data.tobytes():
            raise ValueError(f"installed tensor failed exact roundtrip: {mapped}")
        receipts[mapped] = hashlib.sha256(values.tobytes()).hexdigest()
    return {"verified_tensors": len(receipts), "tensor_sha256": receipts,
            "tied_aliases": skipped, "all_parameters_installed": True,
            "verified_architecture": architecture}
