"""Real capture for the sweep, including the tied embedding's output-head role."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
import gguf
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.capture_calibration import converted_weight
from scripts.calibration_sampling import require_new_capture, stratified_indices, write_capture
from .data import ROOT, OUT, MODEL, SOURCE, sha256


def capture(corpus, output, samples, context, sampling="endpoints", seed=0, *,
            source=SOURCE, checkpoint=MODEL, load_gguf_weights=False):
    corpus, output = Path(corpus), Path(output)
    source = Path(source)
    if sampling not in ("endpoints", "stratified") or min(samples, context) <= 0 or seed < 0:
        raise ValueError("invalid capture configuration")
    require_new_capture(output)
    corpus_hash = sha256(corpus)
    source_hash = sha256(source)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    reader = gguf.GGUFReader(source)
    installation = None
    if load_gguf_weights:
        from . import gguf_weights
        installation = {**gguf_weights.install(model, reader),
                        "loader_sha256": sha256(gguf_weights.__file__)}
    source_path = source
    source = {t.name: t for t in reader.tensors}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, model.config.num_hidden_layers)
    verified = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        if mapped not in source:
            continue
        w = parameter.detach().numpy()
        actual = source[mapped].data
        if not np.array_equal(w, actual):
            candidate = converted_weight(w, mapped, model.config)
            if candidate is None or not np.array_equal(candidate, actual):
                raise ValueError(f"checkpoint mismatch: {mapped}")
        verified.add(mapped)
    if verified != set(source):
        raise ValueError("not all source tensors verified")
    documents = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    if not documents:
        raise ValueError("empty calibration corpus")
    per_doc = max(1, (samples+len(documents)-1)//len(documents))
    batches, kinds, handles, positions = {}, {}, [], {}
    current_document = None
    def make_hook(name, embedding):
        batches[name] = []
        positions[name] = []
        kinds[name] = int(embedding)
        def hook(module, inputs):
            x = inputs[0].detach().cpu()
            x = x.reshape(-1) if embedding else x.reshape(-1, x.shape[-1])
            if sampling == "endpoints":
                index = torch.linspace(0, len(x)-1, min(per_doc, len(x))).long()
            else:
                index = torch.tensor(stratified_indices(len(x), per_doc, seed,
                    f"{corpus_hash}:{current_document}"), dtype=torch.long)
            batches[name].append(x[index].numpy().copy())
            positions[name].append({"document": current_document, "available": len(x), "indices": index.tolist()})
        return hook
    for name, module in model.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        mapped = mapping.get_name(name+".weight", try_suffixes=(".weight",))
        if mapped == "output.weight" and mapped not in source and model.config.tie_word_embeddings:
            mapped = "token_embd.weight::output"
        if mapped not in source and mapped != "token_embd.weight::output":
            continue
        if mapped in batches:
            raise ValueError("ambiguous operator mapping")
        handles.append(module.register_forward_pre_hook(make_hook(mapped, isinstance(module, torch.nn.Embedding))))
    token_hash = hashlib.sha256()
    tokens, rows_per_doc = 0, []
    with torch.inference_mode():
        for i, doc in enumerate(documents):
            current_document = doc.get("doc_index", i)
            inputs = tokenizer(doc["text"], return_tensors="pt", truncation=True, max_length=context)
            token_hash.update(inputs["input_ids"].numpy().astype("<i8").tobytes())
            tokens += inputs["input_ids"].numel()
            rows_per_doc.append(min(per_doc, inputs["input_ids"].numel()))
            model(**inputs, use_cache=False)
            if (i+1) % 16 == 0:
                print(f"Captured {i+1}/{len(documents)} documents", flush=True)
    for handle in handles:
        handle.remove()
    if sha256(corpus) != corpus_hash or sha256(source_path) != source_hash:
        raise ValueError("calibration source or corpus changed during capture")
    data = bytearray(b"AC01" + bytes.fromhex(source_hash) + bytes.fromhex(corpus_hash) + struct.pack("<I", len(batches)))
    counts = {}
    for name in sorted(batches):
        x = np.concatenate(batches[name])[:samples]
        counts[name] = len(x)
        encoded = name.encode()
        data += struct.pack("<I", len(encoded))+encoded+struct.pack("<BI", kinds[name], len(x))
        if kinds[name] == 0:
            data += struct.pack("<I", x.shape[1])+x.astype("<f4").tobytes()
        else:
            data += x.astype("<u4").tobytes()
    data += hashlib.sha256(data).digest()
    manifest = {"source_sha256": source_hash, "corpus_sha256": corpus_hash,
        "tokenized_input_sha256": token_hash.hexdigest(), "tokens": tokens, "context": context,
        "documents": [d["doc_index"] for d in documents], "rows_per_document_before_cap": rows_per_doc,
        "sample_cap": samples, "counts": counts, "verified_tensors": len(verified),
        "tied_head_captured": "token_embd.weight::output" in counts, "torch": torch.__version__,
        "sampling": sampling, "sampling_seed": seed, "sampled_positions_before_cap": positions,
        "collector_sha256": sha256(Path(__file__))}
    if installation is not None:
        manifest["installed_gguf"] = {"source": str(source_path.resolve()), **installation}
    write_capture(output, data, manifest)
    print(f"Saved {output}: {len(batches)} operators, {len(data)} bytes", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["fit", "selection"], required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--sampling", choices=["endpoints", "stratified"], default="endpoints")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--checkpoint", type=Path, default=MODEL)
    p.add_argument("--load-gguf-weights", action="store_true")
    args = p.parse_args()
    if args.split == "fit":
        capture(OUT/"fit.jsonl", args.out or OUT/"fit.acal", 256, 512, args.sampling, args.seed,
                source=args.source, checkpoint=args.checkpoint, load_gguf_weights=args.load_gguf_weights)
    else:
        capture(OUT/"selection_inputs.jsonl", args.out or OUT/"selection.acal", 64, 512, args.sampling, args.seed,
                source=args.source, checkpoint=args.checkpoint, load_gguf_weights=args.load_gguf_weights)
