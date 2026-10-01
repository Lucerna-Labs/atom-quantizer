"""Native M1 perplexity and the precommitted temporal benefit comparisons."""
import argparse
import json
import os
from pathlib import Path

from .basis_discovery import save_json
from .direction_models import BINARY, evaluate
from .full22.data import ROOT, OUT, sha256
from .mamba_models import ORDER


def benefit_gates(models, scores, trajectories):
    if (set(models) != set(ORDER) or any(scores.get(n, {}).get("status") != "MEASURED" for n in ORDER)
            or trajectories.get("status") != "MEASURED"):
        return {"status": "INCONCLUSIVE"}
    traces = trajectories["models"]
    required = {"source-reset", *ORDER[1:]}
    if set(traces) != required or any(v["status"] != "MEASURED" for v in traces.values()):
        return {"status": "INCONCLUSIVE"}
    plain, functional, protected = "q6-scalar", "q6-functional", "q6-functional-selectors-exact"
    def summary(name): return traces[name]["summary"]
    g2 = {
        "complete_bytes": models[functional]["archive_bytes"] <= 1.05*models[plain]["archive_bytes"],
        "native_ppl": scores[functional]["perplexity"] <= scores[plain]["perplexity"],
        "teacher_kl": summary(functional)["teacher_kl"] <= summary(plain)["teacher_kl"],
        "tail_state_nmse": summary(functional)["tail_256_ssm_nmse"] <= .95*summary(plain)["tail_256_ssm_nmse"],
        "finite": summary(functional)["finite"],
    }
    g3 = {
        "complete_bytes": models[protected]["archive_bytes"] <= 1.20*models[functional]["archive_bytes"],
        "native_ppl": scores[protected]["perplexity"] <= scores[functional]["perplexity"],
        "teacher_kl": summary(protected)["teacher_kl"] <= summary(functional)["teacher_kl"],
        "tail_state_nmse": summary(protected)["tail_256_ssm_nmse"] <= .80*summary(functional)["tail_256_ssm_nmse"],
        "finite": summary(protected)["finite"],
    }
    return {"status": "MEASURED", "G2": {"status": "MEASURED" if all(g2.values()) else "FAILED", "checks": g2},
            "G3": {"status": "MEASURED" if all(g3.values()) else "FAILED", "checks": g3}}


def _native(models_path, root):
    models_path = Path(models_path).resolve()
    models = json.loads(models_path.read_text())["models"]
    if set(models) != set(ORDER):
        raise ValueError("M1 requires all five complete models")
    runtime = json.loads((ROOT / "artifacts/six-bit-frontier-2026-10-01/runtime.json").read_text())
    if sha256(BINARY) != runtime["executable"] or any(sha256(p) != h for p, h in runtime["libraries"].items()):
        raise ValueError("native inference runtime changed")
    save_json(root / "runtime.json", runtime)
    files = [models_path, Path(__file__), ROOT / "experiments/direction_models.py", OUT / "selection.txt", BINARY]
    files += [Path(p) for p in runtime["libraries"]]
    for model in models.values():
        for key in ("archive", "decoded"):
            if sha256(model[key]) != model[key+"_sha256"]:
                raise ValueError("stored Mamba model changed")
            files.append(Path(model[key]))
    frozen = {str(p): sha256(p) for p in files}
    save_json(root / "manifest.json", {"sha256": frozen})
    scores = {}
    for name in ORDER:
        scores[name] = evaluate(models[name], OUT / "selection.txt", "selection", root, sha256(root / "runtime.json"))
        save_json(root / "progress.json", scores)
    if any(sha256(p) != h for p, h in frozen.items()):
        raise ValueError("native inference inputs changed")
    result = {"status": "MEASURED" if all(v["status"] == "MEASURED" for v in scores.values()) else "FAILED", "scores": scores}
    save_json(root / "summary.json", result)
    return result


def run_native(models_path, out):
    root = Path(os.path.abspath(out))
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _native(models_path, root)
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_native(args.models, args.out)
