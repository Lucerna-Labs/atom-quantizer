"""Exact standard-Mamba GGUF loading and observation of actual recurrent state."""
import hashlib
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import MambaConfig, MambaForCausalLM


def configuration(reader):
    architecture = reader.get_field("general.architecture")
    if architecture is None or architecture.contents() != "mamba":
        raise ValueError("requires a standard Mamba GGUF")
    values = {k: f.contents() for k, f in reader.fields.items() if k.startswith("mamba.")}
    required = {"mamba.context_length", "mamba.embedding_length", "mamba.feed_forward_length",
                "mamba.attention.head_count", "mamba.block_count", "mamba.ssm.conv_kernel",
                "mamba.ssm.inner_size", "mamba.ssm.state_size", "mamba.ssm.time_step_rank",
                "mamba.attention.layer_norm_rms_epsilon", "mamba.ssm.dt_b_c_rms"}
    if set(values) != required or values["mamba.ssm.dt_b_c_rms"] is not False:
        raise ValueError("missing or unsupported Mamba architecture fields")
    for key in required - {"mamba.attention.layer_norm_rms_epsilon", "mamba.ssm.dt_b_c_rms"}:
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError("invalid Mamba integer metadata")
    if (values["mamba.feed_forward_length"] != 0 or values["mamba.attention.head_count"] != 0
            or any(values[k] <= 0 for k in required if k not in {
                "mamba.feed_forward_length", "mamba.attention.head_count", "mamba.ssm.dt_b_c_rms"})
            or not np.isfinite(values["mamba.attention.layer_norm_rms_epsilon"])
            or values["mamba.ssm.inner_size"] != 2 * values["mamba.embedding_length"]):
        raise ValueError("unsupported Mamba dimensions or normalization")
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors) or "token_embd.weight" not in tensors:
        raise ValueError("duplicate tensors or missing embedding")
    embedding = tensors["token_embd.weight"].data
    if embedding.ndim != 2 or embedding.shape[1] != values["mamba.embedding_length"]:
        raise ValueError("embedding and metadata dimensions differ")
    return MambaConfig(vocab_size=int(embedding.shape[0]), hidden_size=values["mamba.embedding_length"],
        num_hidden_layers=values["mamba.block_count"], state_size=values["mamba.ssm.state_size"],
        expand=2, conv_kernel=values["mamba.ssm.conv_kernel"], time_step_rank=values["mamba.ssm.time_step_rank"],
        layer_norm_epsilon=values["mamba.attention.layer_norm_rms_epsilon"],
        use_bias="blk.0.ssm_in.bias" in tensors, use_conv_bias="blk.0.ssm_conv1d.bias" in tensors,
        hidden_act="silu", residual_in_fp32=True, tie_word_embeddings="output.weight" not in tensors,
        bos_token_id=0, eos_token_id=0, pad_token_id=0, use_cache=True)


@torch.inference_mode()
def transition_preimage(values):
    target = torch.as_tensor(np.array(values, dtype=np.float32, copy=True))
    if not torch.isfinite(target).all() or not (target < 0).all():
        raise ValueError("transition values must be finite and strictly negative")
    result = torch.log(-target)
    found = -result.exp() == target
    lower, upper = result.clone(), result.clone()
    for _ in range(4):
        if bool(found.all()):
            break
        lower = torch.nextafter(lower, torch.full_like(lower, float("-inf")))
        upper = torch.nextafter(upper, torch.full_like(upper, float("inf")))
        for candidate in (lower, upper):
            match = ~found & (-candidate.exp() == target)
            result[match] = candidate[match]
            found |= match
    if not bool(found.all()):
        raise ValueError("stored transition has no verified nearby float32 exponential preimage")
    return result.numpy().copy()


def converted(parameter, name):
    values = parameter.detach().cpu()
    if name.endswith(".A_log"):
        values = -values.exp()
    elif name.endswith(".conv1d.weight"):
        values = values.squeeze(1)
    return values.numpy()


def validate_configuration(model, expected):
    fields = ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
              "state_size", "conv_kernel", "time_step_rank", "use_bias", "use_conv_bias", "hidden_act",
              "residual_in_fp32", "tie_word_embeddings")
    if any(getattr(model.config, k) != getattr(expected, k) for k in fields):
        raise ValueError("checkpoint and GGUF Mamba configurations differ")
    if np.float32(model.config.layer_norm_epsilon) != np.float32(expected.layer_norm_epsilon):
        raise ValueError("Mamba normalization differs")


def verify(model, reader):
    validate_configuration(model, configuration(reader))
    tensors = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MAMBA, model.config.num_hidden_layers)
    verified, aliases = {}, []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if name == "lm_head.weight" and mapped not in tensors and model.config.tie_word_embeddings:
            if parameter.data_ptr() != model.backbone.embeddings.weight.data_ptr():
                raise ValueError("tied Mamba output head is not shared")
            aliases.append(name)
            continue
        if mapped not in tensors:
            raise ValueError(f"unmapped Mamba parameter: {name}")
        tensor = tensors[mapped]
        if parameter.dtype != torch.float32 or tensor.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError("Mamba reconstruction requires F32")
        values = converted(parameter, name)
        if values.shape != tensor.data.shape or values.tobytes() != tensor.data.tobytes():
            raise ValueError(f"Mamba operator values differ: {mapped}")
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite Mamba parameter: {mapped}")
        verified[mapped] = hashlib.sha256(values.tobytes()).hexdigest()
    if set(verified) != set(tensors):
        raise ValueError("not every Mamba tensor was verified")
    return {"verified_tensors": len(verified), "operator_sha256": verified, "tied_aliases": aliases}


@torch.inference_mode()
def load(path):
    reader = gguf.GGUFReader(Path(path))
    config = configuration(reader)
    model = MambaForCausalLM(config).float().eval()
    tensors = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MAMBA, config.num_hidden_layers)
    plans, covered = [], set()
    for name, parameter in model.named_parameters():
        mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if mapped not in tensors:
            raise ValueError(f"missing Mamba tensor: {mapped}")
        tensor = tensors[mapped]
        if tensor.tensor_type != gguf.GGMLQuantizationType.F32 or not np.isfinite(tensor.data).all():
            raise ValueError("invalid Mamba F32 source")
        values = tensor.data
        if name.endswith(".A_log"):
            values = transition_preimage(values)
        elif name.endswith(".conv1d.weight"):
            values = values[:, None, :]
        if values.shape != tuple(parameter.shape):
            raise ValueError(f"Mamba tensor shape differs: {mapped}")
        plans.append((parameter, values))
        covered.add(mapped)
    if covered != set(tensors):
        raise ValueError("unexpected Mamba tensors")
    for parameter, values in plans:
        parameter.copy_(torch.from_numpy(np.array(values, copy=True)))
    receipt = verify(model, reader)
    return model, receipt


def states(cache, config):
    if cache is None or len(cache.layers) != config.num_hidden_layers:
        raise ValueError("missing or incomplete Mamba cache")
    result = {}
    for kind, width in (("recurrent_states", config.state_size), ("conv_states", config.conv_kernel)):
        values = []
        for layer in cache.layers:
            entries = getattr(layer, kind, None)
            if not isinstance(entries, dict) or set(entries) != {0}:
                raise ValueError("unsupported Mamba cache state layout")
            value = entries[0]
            if (not torch.is_tensor(value) or value.dtype != torch.float32
                    or tuple(value.shape) != (1, config.intermediate_size, width)
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("invalid Mamba state tensor")
            values.append(value[0])
        result[kind] = torch.stack(values).detach()
    return result
