#!/usr/bin/env python3
"""Prepare disjoint, reproducible WikiText-2 calibration and evaluation inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/datasets"))

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    parser.add_argument("--documents", type=int, default=16)
    args = parser.parse_args()
    if args.documents <= 0:
        parser.error("--documents must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    for name in ["calibration.jsonl", "heldout.txt", "dataset.json"]:
        if (output / name).exists():
            parser.error(f"output already exists: {output / name}")
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=args.revision, token=False)
    calibration = [r["text"] for r in data["train"] if len(r["text"].strip()) > 400][:args.documents]
    heldout = "\n".join(r["text"] for r in data["test"])
    calibration_bytes = "".join(json.dumps({"text": t}) + "\n" for t in calibration).encode("utf-8")
    heldout_bytes = heldout.encode("utf-8")
    (output / "calibration.jsonl").write_bytes(calibration_bytes)
    (output / "heldout.txt").write_bytes(heldout_bytes)
    manifest = {
        "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
        "dataset_revision": args.revision, "calibration_split": "train", "heldout_split": "test",
        "calibration_documents": len(calibration), "test_characters": len(heldout),
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "heldout_sha256": hashlib.sha256(heldout_bytes).hexdigest(),
    }
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
