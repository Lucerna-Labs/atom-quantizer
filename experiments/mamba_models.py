"""Build M1's fixed complete-model controls and verify native A22 exports."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import gguf
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .full22 import precision
from .full22.data import ROOT, sha256
from .mamba_capture import SOURCE

CAPTURES = ROOT / "artifacts/mamba-temporal-2026-10-01/captures"
MAIN = ROOT / "target/release/atom-quantizer"
ORDER = ("source-lossless", "q4-scalar", "q6-scalar", "q6-functional", "q6-functional-selectors-exact")
LINEAR_SUFFIXES = (".ssm_in.weight", ".ssm_x.weight", ".ssm_dt.weight", ".ssm_out.weight")


def configurations(reader, variant):
    if variant not in ORDER:
        raise ValueError("unknown M1 candidate")
    config = {"family": "lossless"} if variant == "source-lossless" else {
        "bits": 4 if variant == "q4-scalar" else 6,
        **({"rank": 2, "rank_geometry": "activation"} if "functional" in variant else {})}
    tensors, linear = {}, []
    for tensor in reader.tensors:
        is_linear = tensor.name == "token_embd.weight" or tensor.name.endswith(LINEAR_SUFFIXES)
        if is_linear: linear.append(tensor.name)
        selected = dict(config) if is_linear else {"family": "lossless"}
        if variant.endswith("selectors-exact") and tensor.name.endswith((".ssm_x.weight", ".ssm_dt.weight")):
            selected = {"family": "lossless"}
        tensors[tensor.name] = selected
    if len(reader.tensors) != 242 or len(linear) != 97:
        raise ValueError("M1 requires the declared complete 130M model")
    return config, tensors


def _run(root):
    threadpool_limits(4)
    captures = json.loads((CAPTURES / "summary.json").read_text())
    if captures["status"] != "MEASURED" or any(sha256(v["path"]) != v["sha256"] for v in captures["captures"].values()):
        raise ValueError("actual Mamba captures changed or incomplete")
    if sha256(SOURCE) != "3a4b925d3e7e6bde08b6f63fbc9fca48e632f7746847653d795c34cfc099f793":
        raise ValueError("Mamba source changed")
    if sha256(MAIN) != "d45eaa37c05ae4d08001e7b08ab3d90a0cfe9c970e03c0f06351558d6c9a400b":
        raise ValueError("native decoder changed")
    base, residual = CAPTURES / "stratified-fit.acal", CAPTURES / "endpoints-fit.acal"
    files = [SOURCE, MAIN, base, residual, Path(__file__), ROOT / "experiments/mamba_runtime.py",
             ROOT / "research/mamba-temporal-2026-10-01/PROTOCOL.md"]
    files += list((ROOT / "experiments/full22").glob("*.py"))
    frozen = {str(p): sha256(p) for p in files}
    save_json(root / "manifest.json", {"sha256": frozen, "candidate_order": ORDER})
    models = {}
    reader = gguf.GGUFReader(SOURCE)
    for name in ORDER:
        config, per_tensor = configurations(reader, name)
        print("BUILD", name, flush=True)
        model = precision.build_stage(SOURCE, base, residual if "functional" in name else None,
                                      per_tensor, root / name, default_config=config)
        native = root / name / "native.gguf"
        command = [str(MAIN), "decode", model["archive"], "--out", str(native)]
        with (root / name / "native-decode.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        if sha256(native) != model["decoded_sha256"]:
            raise ValueError("Mamba native and Python A22 export differ")
        if name == "source-lossless" and sha256(native) != sha256(SOURCE):
            raise ValueError("lossless Mamba archive changed source bytes")
        models[name] = dict(model, name=name, decoded=str(native), native_export_identical=True)
        save_json(root / "progress.json", models)
        print(json.dumps({"name": name, "bytes": model["archive_bytes"], "sha256": model["archive_sha256"]}), flush=True)
    if any(sha256(p) != h for p, h in frozen.items()):
        raise ValueError("M1 build inputs or implementation changed")
    save_json(root / "summary.json", {"status": "MEASURED", "models": models})
    return models


def run(out):
    root = Path(os.path.abspath(out))
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run(root)
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
