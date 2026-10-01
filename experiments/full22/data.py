"""Pinned, separate fit/selection/final data and source-bound activation loading."""
import hashlib
import json
import os
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/full22-2026-09-06"
SOURCE = ROOT / "models/SmolLM2-135M-Instruct-f32.gguf"
MODEL = ROOT / "models/SmolLM2-135M-Instruct"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_calibration(path, source=SOURCE):
    data = Path(path).read_bytes()
    if data[:4] != b"AC01" or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("invalid calibration checksum")
    if data[4:36].hex() != sha256(source):
        raise ValueError("calibration source mismatch")
    count = struct.unpack_from("<I", data, 68)[0]
    offset, result = 72, {}
    for _ in range(count):
        n = struct.unpack_from("<I", data, offset)[0]; offset += 4
        name = data[offset:offset+n].decode(); offset += n
        kind, samples = struct.unpack_from("<BI", data, offset); offset += 5
        if kind == 0:
            cols = struct.unpack_from("<I", data, offset)[0]; offset += 4
            size = samples * cols * 4
            result[name] = np.frombuffer(data, dtype="<f4", count=samples * cols, offset=offset).reshape(samples, cols)
        elif kind == 1:
            size = samples * 4
            result[name] = np.frombuffer(data, dtype="<u4", count=samples, offset=offset)
        else:
            raise ValueError("unknown calibration kind")
        offset += size
    if offset != len(data) - 32:
        raise ValueError("calibration extent mismatch")
    return result


def prepare():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/datasets"))
    from datasets import load_dataset
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "data_manifest.json").exists():
        return json.loads((OUT / "data_manifest.json").read_text())
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=DATA_REVISION, token=False)
    rng = np.random.default_rng(220906)
    train = [(i, r["text"]) for i, r in enumerate(data["train"]) if len(r["text"].strip()) > 400]
    valid = [(i, r["text"]) for i, r in enumerate(data["validation"]) if len(r["text"].strip()) > 300]
    train = [train[i] for i in rng.permutation(len(train))[:128]]
    valid = [valid[i] for i in rng.permutation(len(valid))]
    def jsonl(name, rows):
        (OUT / name).write_bytes("".join(json.dumps({"doc_index": i, "text": text}) + "\n" for i, text in rows).encode())
    jsonl("fit.jsonl", train)
    jsonl("selection_inputs.jsonl", valid[:32])
    (OUT / "selection.txt").write_bytes("\n".join(text for _, text in valid[32:]).encode())
    split = len(data["test"]) // 2
    prefix = "\n".join(data["test"][:split]["text"])
    tail = "\n".join(data["test"][split:]["text"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    if prefix_tokens <= 65536:
        raise ValueError("final test overlaps prior evaluation")
    (OUT / "final_test.txt").write_bytes(tail.encode())
    result = {"dataset": "Salesforce/wikitext", "revision": DATA_REVISION, "seed": 220906,
        "fit_documents": 128, "selection_input_documents": 32,
        "final_test_start_row": split, "excluded_test_prefix_tokens": prefix_tokens,
        "previous_evaluation_max_input_tokens": 65536,
        "files": {name: sha256(OUT / name) for name in ["fit.jsonl", "selection_inputs.jsonl", "selection.txt", "final_test.txt"]}}
    (OUT / "data_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2))
