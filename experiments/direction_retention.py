"""D01 stored functional-residual comparison, conventional primitive toolkit."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import measured_record, save_json
from .full22 import codec
from .full22.catalog import TENSORS
from .full22.data import ROOT, OUT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record

PROTOCOL = ROOT/"research/direction-retention-2026-09-30/PROTOCOL.md"
GEOMETRIES = ("activation", "diagonal", "activation-shuffled")
RANKS = (1, 2, 4)


def manifest():
    paths = [PROTOCOL, Path(__file__), ROOT/"experiments/basis_discovery.py"]
    paths += sorted((ROOT/"experiments/full22").glob("*.py"))
    return {"experiment": "D01-screen", "records": 160, "rows": 256,
        "source_sha256": sha256(SOURCE), "fit_sha256": sha256(OUT/"fit.acal"),
        "selection_sha256": sha256(OUT/"selection.acal"), "python": sys.version,
        "dependencies": {n:importlib.metadata.version(n) for n in ["numpy", "scipy", "gguf", "threadpoolctl"]},
        "implementation": {str(p.relative_to(ROOT)):sha256(p) for p in paths}}


def summarize(records, frozen):
    candidates, eligible = [], []
    for bits in (3, 4):
        baseline = [r for r in records if r["bits"] == bits and r["rank"] == 0 and r["status"] == "MEASURED"]
        for rank in RANKS:
            groups = {g:[r for r in records if r["bits"] == bits and r["rank"] == rank
                         and r["geometry"] == g and r["status"] == "MEASURED"] for g in GEOMETRIES}
            case = {"bits": bits, "rank": rank, "status": "INCONCLUSIVE"}
            if len(baseline) != 8 or any(len(v) != 8 for v in groups.values()):
                case["reason"] = "required measured record missing or failed"
                candidates.append(case)
                continue
            means = {g:float(np.mean([r["selection"]["output_nmse"] for r in group])) for g,group in groups.items()}
            mean_base = float(np.mean([r["selection"]["output_nmse"] for r in baseline]))
            byte_ratio = sum(r["raw_bytes"] for r in groups["activation"]) / sum(r["raw_bytes"] for r in baseline)
            ratios = {r["tensor"]:r["selection"]["output_nmse"] /
                      next(b["selection"]["output_nmse"] for b in baseline if b["tensor"] == r["tensor"])
                      for r in groups["activation"]}
            gates = {"S1_stored_records": True,
                "S2_benefit": means["activation"] <= 0.75*mean_base and byte_ratio <= 1.1
                    and max(ratios.values()) <= 1.05
                    and all(means["activation"] < means[g] for g in ("diagonal", "activation-shuffled")),
                "S3_complete": len(records) == 160 and manifest() == frozen}
            case.update(status="MEASURED" if all(gates.values()) else "FAILED", gates=gates,
                mean_selection_output_nmse={"scalar": mean_base, **means},
                raw_bytes_ratio=byte_ratio, per_tensor_output_ratios=ratios,
                minimum_weight_cosine=min(r["selection"]["cosine"] for r in groups["activation"]))
            candidates.append(case)
        passing = [c for c in candidates if c["bits"] == bits and c["status"] == "MEASURED"]
        if passing:
            eligible.append({"bits": bits, "rank": min(c["rank"] for c in passing)})
    return {"experiment": "D01-screen", "records": len(records),
        "measured": sum(r["status"] == "MEASURED" for r in records), "candidates": candidates,
        "eligible_for_model_trials": eligible, "promoted_to_main_codec": []}


def run(output):
    threadpool_limits(4)
    output = Path(output).resolve()
    frozen = manifest()
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != frozen:
            raise ValueError("D01 inputs changed; preserve this run and use a new directory")
    else:
        save_json(path, frozen)
    fit, selection = load_calibration(OUT/"fit.acal", SOURCE), load_calibration(OUT/"selection.acal", SOURCE)
    reader = gguf.GGUFReader(SOURCE)
    tensors = {t.name:t for t in reader.tensors}
    records = []
    for name in TENSORS:
        full = np.asarray(tensors[name].data, np.float32)
        indices = np.linspace(0, len(full)-1, min(256, len(full))).astype(np.int64)
        weight = np.ascontiguousarray(full[indices])
        x, xv = inputs_for(fit, name), inputs_for(selection, name)
        if x is None or xv is None:
            raise ValueError("D01 needs both real fit and selection inputs")
        for bits in (3, 4):
            configs = [(0, "scalar", {"bits": bits})] + [(rank, g,
                {"bits": bits, "rank": rank, "rank_geometry": g}) for rank in RANKS for g in GEOMETRIES]
            for rank, geometry, config in configs:
                identity = {"tensor": name, "bits": bits, "rank": rank, "geometry": geometry}
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                receipt, artifact = output/(key+".json"), output/(key+".ar")
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    if any(result.get(k) != v for k,v in identity.items()):
                        raise ValueError("D01 receipt identity mismatch")
                    if result["status"] == "MEASURED":
                        if sha256(artifact) != result["artifact_sha256"]:
                            raise ValueError("D01 artifact changed")
                        codec.decode(Record.loads(artifact.read_bytes()))
                else:
                    result = dict(identity, config=config)
                    try:
                        result.update(measured_record(weight, x, xv, config, artifact))
                        loaded = Record.loads(artifact.read_bytes())
                        result["effective_rank"] = loaded.arrays["lowrank_u"].shape[1] if rank else 0
                    except Exception as error:
                        result.update(status="FAILED", error=repr(error))
                    save_json(receipt, result)
                records.append(result)
        print(f"{name}: {len(records)}/160 records", flush=True)
    if manifest() != frozen:
        save_json(output/"invalidated.json", {"reason": "inputs changed during execution"})
        raise ValueError("D01 inputs changed during execution")
    summary = summarize(records, frozen)
    save_json(output/"summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2, allow_nan=False), flush=True)
