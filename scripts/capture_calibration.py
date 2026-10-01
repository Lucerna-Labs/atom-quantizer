#!/usr/bin/env python3
"""Capture real HF operator inputs after verifying the corresponding GGUF weights.

Supports mapped torch Linear and token Embedding modules. Unmapped or transformed
weights are rejected unless their exact supported conversion is verified. No
remote model code is enabled. The Rust consumer checks source and file SHA-256.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import gguf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
if __package__:
    from .calibration_sampling import require_new_capture, stratified_indices, write_capture
else:
    from calibration_sampling import require_new_capture, stratified_indices, write_capture


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").digest()


def converted_weight(weight, name, config):
    """The documented llama.cpp Q/K row permutation; input columns stay fixed."""
    if name.endswith(".attn_q.weight"):
        heads = config.num_attention_heads
    elif name.endswith(".attn_k.weight"):
        heads = config.num_key_value_heads
    else:
        return None
    if weight.shape[0] % (2 * heads):
        return None
    return weight.reshape(heads, 2, weight.shape[0] // heads // 2, *weight.shape[1:]).swapaxes(1, 2).reshape(weight.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--gguf", required=True, help="GGUF exported from these same weights")
    parser.add_argument("--corpus", required=True, help="JSONL containing text strings or {text: ...}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--sampling", choices=["endpoints", "stratified"], default="endpoints")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.samples, args.prompts, args.context, args.threads) <= 0 or args.seed < 0:
        parser.error("sample, prompt, context and thread counts must be positive")
    output = Path(args.out)
    require_new_capture(output)
    corpus_hash = digest(args.corpus)
    source_hash = digest(args.gguf)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").eval()
    reader = gguf.GGUFReader(args.gguf)
    arch_name = reader.get_field("general.architecture").contents()
    arch = next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch_name)
    name_map = gguf.get_tensor_name_map(arch, model.config.num_hidden_layers)
    source = {t.name: t for t in reader.tensors}

    # Verify every mapped source parameter, including norms. A shape-only match
    # would permit capturing from the wrong checkpoint with the same architecture.
    verified = set()
    for hf_name, parameter in model.named_parameters(remove_duplicate=False):
        name = name_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        if name not in source:
            continue
        actual = gguf.dequantize(source[name].data, source[name].tensor_type)
        expected = parameter.detach().float().cpu().numpy()
        if expected.shape != actual.shape:
            raise ValueError(f"Source shape differs for {name}: {expected.shape} vs {actual.shape}")
        if not np.array_equal(actual, expected):
            permuted = converted_weight(expected, name, model.config) if arch_name == "llama" else None
            if permuted is None or not np.array_equal(actual, permuted):
                raise ValueError(f"GGUF weights do not exactly match this HF checkpoint/conversion: {name}")
        verified.add(name)
    missing_weights = sorted(set(source) - verified)
    if missing_weights:
        raise ValueError(f"Cannot verify source tensors: {missing_weights}")

    texts = []
    with open(args.corpus, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            text = item if isinstance(item, str) else item["text"]
            if text.strip():
                texts.append(text)
            if len(texts) == args.prompts:
                break
    if not texts:
        raise ValueError("Calibration corpus is empty")
    per_prompt = max(1, (args.samples + len(texts) - 1) // len(texts))
    batches, kinds, handles, positions = {}, {}, [], {}
    current_prompt = None

    def hook_for(name, embedding):
        batches[name] = []
        positions[name] = []
        kinds[name] = 1 if embedding else 0
        def hook(module, inputs):
            values = inputs[0].detach().cpu()
            values = values.reshape(-1) if embedding else values.reshape(-1, values.shape[-1])
            if args.sampling == "endpoints":
                indices = torch.linspace(0, len(values) - 1, min(per_prompt, len(values))).long()
            else:
                indices = torch.tensor(stratified_indices(len(values), per_prompt, args.seed,
                    f"{corpus_hash.hex()}:{current_prompt}"), dtype=torch.long)
            batches[name].append(values[indices].numpy().copy())
            positions[name].append({"prompt": current_prompt, "available": len(values), "indices": indices.tolist()})
        return hook

    for hf_name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        name = name_map.get_name(hf_name + ".weight", try_suffixes=(".weight",))
        if name not in source:
            continue
        embedding = isinstance(module, torch.nn.Embedding)
        if embedding and name != "token_embd.weight":
            raise ValueError(f"Unsupported embedding operator: {name}")
        if name in batches:
            raise ValueError(f"Multiple operators map to {name}; an explicit combined collector is needed")
        handles.append(module.register_forward_pre_hook(hook_for(name, embedding)))

    needed = {name for name, t in source.items() if len(t.shape) >= 2 and min(t.shape) > 1}
    if needed - set(batches):
        raise ValueError(f"Missing real operator hooks: {sorted(needed - set(batches))}")
    token_hash = hashlib.sha256()
    token_count = 0
    with torch.inference_mode():
        for i, text in enumerate(texts):
            current_prompt = i
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.context)
            token_hash.update(inputs["input_ids"].numpy().astype("<i8").tobytes())
            token_count += inputs["input_ids"].numel()
            model(**inputs, use_cache=False)
            print(f"Captured prompt {i + 1}/{len(texts)}", flush=True)
    for handle in handles:
        handle.remove()

    if digest(args.corpus) != corpus_hash or digest(args.gguf) != source_hash:
        raise ValueError("calibration source or corpus changed during capture")
    record = bytearray(b"AC01" + source_hash + corpus_hash + struct.pack("<I", len(batches)))
    counts = {}
    for name in sorted(batches):
        if not batches[name]:
            raise ValueError(f"Operator was never executed: {name}")
        values = np.concatenate(batches[name])[:args.samples]
        name_bytes = name.encode("utf-8")
        record += struct.pack("<I", len(name_bytes)) + name_bytes
        record += struct.pack("<BI", kinds[name], len(values))
        if kinds[name] == 0:
            record += struct.pack("<I", values.shape[1])
            record += values.astype("<f4").tobytes()
        else:
            record += values.astype("<u4").tobytes()
        counts[name] = len(values)
    record += hashlib.sha256(record).digest()
    manifest = {
        "model_directory": str(Path(args.model).resolve()),
        "source_gguf_sha256": source_hash.hex(), "corpus_sha256": corpus_hash.hex(),
        "tokenized_inputs_sha256": token_hash.hexdigest(), "tokens": token_count,
        "prompts": len(texts), "context": args.context, "samples_per_operator": counts,
        "verified_source_tensors": len(verified), "torch": torch.__version__,
        "real_forward_passes": True, "device": "cpu", "dtype": "float32",
        "sampling": args.sampling, "sampling_seed": args.seed, "sampled_positions_before_cap": positions,
        "collector_sha256": digest(__file__).hex(),
    }
    write_capture(output, record, manifest)
    print(f"Wrote {output}: {len(batches)} real operator batches, {len(record)} bytes")


if __name__ == "__main__":
    main()
