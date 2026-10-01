"""Source-bound real Mamba inputs, including its directly evaluated dt projection."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import gguf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.calibration_sampling import require_new_capture, stratified_indices, write_capture
from .full22.data import ROOT, OUT, sha256
from .mamba_runtime import verify

MODEL = ROOT / "models/mamba-130m-hf"
SOURCE = ROOT / "artifacts/mamba-temporal-2026-10-01/preflight/source.gguf"


def capture(corpus, output, samples, context, sampling, seed=0):
    corpus, output = Path(corpus), Path(output)
    if sampling not in ("endpoints", "stratified") or min(samples, context) <= 0 or seed < 0:
        raise ValueError("invalid Mamba capture settings")
    require_new_capture(output)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    source_hash, corpus_hash = sha256(SOURCE), sha256(corpus)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    reader = gguf.GGUFReader(SOURCE)
    verified = verify(model, reader)
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MAMBA, model.config.num_hidden_layers)
    source_names = {t.name for t in reader.tensors}
    documents = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    if not documents:
        raise ValueError("empty Mamba fitting corpus")
    per_doc = max(1, (samples + len(documents) - 1) // len(documents))
    batches, kinds, positions, methods, handles = {}, {}, {}, {}, []
    current_document = None

    def register(name, embedding, method):
        if name in batches:
            raise ValueError("duplicate capture role")
        batches[name], positions[name] = [], []
        kinds[name], methods[name] = int(embedding), method

    def collect(name, values):
        values = values.detach().cpu()
        values = values.reshape(-1) if kinds[name] else values.reshape(-1, values.shape[-1])
        if sampling == "endpoints":
            index = torch.linspace(0, len(values)-1, min(per_doc, len(values))).long()
        else:
            index = torch.tensor(stratified_indices(len(values), per_doc, seed,
                                 f"{corpus_hash}:{current_document}"), dtype=torch.long)
        sampled = values[index].numpy().copy()
        if not np.isfinite(sampled).all():
            raise ValueError("nonfinite actual Mamba observation")
        batches[name].append(sampled)
        positions[name].append({"document": current_document, "available": len(values), "indices": index.tolist()})

    def pre_hook(name):
        return lambda module, inputs: collect(name, inputs[0])

    def time_hook(name):
        def observe(module, inputs, output):
            if output.shape[-1] != model.config.time_step_rank + 2 * model.config.state_size:
                raise ValueError("Mamba selector layout changed")
            collect(name, output[..., :model.config.time_step_rank])
        return observe

    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or name.endswith(".dt_proj"):
            continue
        mapped = mapping.get_name(name + ".weight", try_suffixes=(".weight",))
        if mapped == "output.weight" and mapped not in source_names and model.config.tie_word_embeddings:
            mapped = "token_embd.weight::output"
        if mapped not in source_names and mapped != "token_embd.weight::output":
            raise ValueError(f"unmapped real Mamba operator: {name}")
        register(mapped, isinstance(module, torch.nn.Embedding), "forward_pre_hook")
        handles.append(module.register_forward_pre_hook(pre_hook(mapped)))
        if name.endswith(".x_proj"):
            dt_name = mapping.get_name(name.removesuffix(".x_proj") + ".dt_proj.weight", try_suffixes=(".weight",))
            if dt_name not in source_names:
                raise ValueError("missing time-step projection source")
            register(dt_name, False, "actual_x_proj_output_prefix")
            handles.append(module.register_forward_hook(time_hook(dt_name)))
    expected = 4 * model.config.num_hidden_layers + 2
    if len(batches) != expected:
        raise ValueError("not every linear/embedding role is observed")
    tokens, token_hash = 0, hashlib.sha256()
    with torch.inference_mode():
        for index, document in enumerate(documents):
            current_document = document.get("doc_index", index)
            inputs = tokenizer(document["text"], return_tensors="pt", truncation=True, max_length=context)
            token_hash.update(inputs["input_ids"].numpy().astype("<i8").tobytes())
            tokens += inputs["input_ids"].numel()
            model(**inputs, use_cache=False)
            if (index + 1) % 16 == 0:
                print(f"Captured {index + 1}/{len(documents)} Mamba documents", flush=True)
    for handle in handles:
        handle.remove()
    if sha256(SOURCE) != source_hash or sha256(corpus) != corpus_hash:
        raise ValueError("Mamba source/corpus changed")
    data = bytearray(b"AC01" + bytes.fromhex(source_hash) + bytes.fromhex(corpus_hash) + struct.pack("<I", len(batches)))
    counts = {}
    for name in sorted(batches):
        if not batches[name]:
            raise ValueError(f"operator never executed: {name}")
        values = np.concatenate(batches[name])[:samples]
        counts[name] = len(values)
        encoded = name.encode()
        data += struct.pack("<I", len(encoded)) + encoded + struct.pack("<BI", kinds[name], len(values))
        if kinds[name]:
            data += values.astype("<u4").tobytes()
        else:
            data += struct.pack("<I", values.shape[1]) + values.astype("<f4").tobytes()
    data += hashlib.sha256(data).digest()
    manifest = {"source_sha256": source_hash, "corpus_sha256": corpus_hash, "counts": counts,
        "verified_source": verified, "tokenized_input_sha256": token_hash.hexdigest(), "tokens": tokens,
        "documents": [d["doc_index"] for d in documents], "context": context, "sample_cap": samples,
        "sampling": sampling, "sampling_seed": seed, "sampled_positions_before_cap": positions,
        "capture_methods": methods, "collector_sha256": sha256(__file__), "torch": torch.__version__}
    write_capture(output, data, manifest)
    print(f"Saved {output}: {len(batches)} actual roles, {len(data)} bytes", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("fit", "selection"), required=True)
    parser.add_argument("--sampling", choices=("endpoints", "stratified"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    capture(OUT / ("fit.jsonl" if args.split == "fit" else "selection_inputs.jsonl"),
            args.out, 256 if args.split == "fit" else 64, 512, args.sampling)
