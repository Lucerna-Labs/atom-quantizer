"""Execute D01's conditional complete-model and held-out inference protocol."""
import argparse
import contextlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .direction_retention import GEOMETRIES, manifest as screen_manifest, summarize
from .full22.catalog import TENSORS
from .full22 import codec, harness
from .full22.data import ROOT, OUT, SOURCE, sha256
from .full22.records import Record

BINARY = ROOT/"third_party/llama.cpp/build/bin/llama-perplexity"


def verified_screen(screen):
    frozen = json.loads((screen/"manifest.json").read_text())
    if screen_manifest() != frozen:
        raise ValueError("screen implementation or data changed before model trials")
    records = []
    for path in screen.glob("*.json"):
        value = json.loads(path.read_text())
        if "geometry" in value:
            if value["status"] == "MEASURED" and sha256(path.with_suffix(".ar")) != value["artifact_sha256"]:
                raise ValueError("screen artifact changed")
            records.append(value)
    records.sort(key=lambda r: (TENSORS.index(r["tensor"]), r["bits"], r["rank"],
        -1 if r["rank"] == 0 else GEOMETRIES.index(r["geometry"])))
    summary = summarize(records, frozen)
    if summary != json.loads((screen/"summary.json").read_text()):
        raise ValueError("screen summary disagrees with current receipts")
    return summary["eligible_for_model_trials"], frozen


def validate_export(artifact, destination):
    tensors = {t.name:t for t in gguf.GGUFReader(destination).tensors}
    with zipfile.ZipFile(artifact) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if len(tensors) != len(manifest["tensors"]):
            raise ValueError("exported tensor count differs")
        for entry in manifest["tensors"]:
            data = archive.read(entry["entry"])
            expected = codec.decode(Record.loads(data))
            if not np.array_equal(tensors[entry["name"]].data, expected):
                raise ValueError("exported values differ from stored decoder")
    return len(tensors)


def build(config, name, root, calibration=None, residual_calibration=None):
    calibration = Path(calibration) if calibration is not None else OUT/"fit.acal"
    calibration_hash = sha256(calibration)
    residual_hash = sha256(residual_calibration) if residual_calibration is not None else calibration_hash
    def check_calibrations(saved_archive):
        saved = json.loads(saved_archive.read("manifest.json"))
        if (saved["calibration_sha256"] != calibration_hash
                or saved.get("residual_calibration_sha256", saved["calibration_sha256"]) != residual_hash):
            raise ValueError("saved model calibration roles differ")
    archive, decoded = root/(name+".a22"), root/(name+".gguf")
    receipt = root/(name+".model.json")
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["config"] != config or result["archive_sha256"] != sha256(archive) or result["decoded_sha256"] != sha256(decoded):
            raise ValueError("saved model receipt changed")
        with zipfile.ZipFile(archive) as saved_archive:
            check_calibrations(saved_archive)
        return result
    print(f"BUILD {name}", flush=True)
    if archive.exists():
        saved = json.loads(archive.with_suffix(".a22.json").read_text())
        if saved["config"] != config or saved["sha256"] != sha256(archive):
            raise ValueError("unverified existing archive")
        with zipfile.ZipFile(archive) as saved_archive:
            check_calibrations(saved_archive)
    else:
        with (root/(name+".build.log")).open("x") as log, contextlib.redirect_stdout(log):
            harness.build_model(config, archive, SOURCE, calibration, residual_calibration=residual_calibration)
    if not decoded.exists():
        # Standalone export is tested with Python opens of original/calibration denied.
        program = """
import os,sys
from experiments.full22.archive import export_model
blocked={os.path.realpath(p) for p in sys.argv[3:]}
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)) and os.path.realpath(os.fsdecode(args[0])) in blocked:
        raise PermissionError('source access forbidden during standalone export')
sys.addaudithook(audit)
print(export_model(sys.argv[1],sys.argv[2]))
"""
        with (root/(name+".export.log")).open("x") as log:
            subprocess.run([sys.executable, "-c", program, str(archive), str(decoded), str(SOURCE), str(calibration),
                            str(residual_calibration or calibration), str(OUT/"fit.acal")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    count = validate_export(archive, decoded)
    result = {"name": name, "status": "MEASURED", "config": config, "archive": str(archive),
        "calibration_sha256": calibration_hash,
        "residual_calibration_sha256": residual_hash,
        "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
        "decoded": str(decoded), "decoded_sha256": sha256(decoded), "verified_tensors": count,
        "standalone_source_and_calibration_opens_denied": True}
    save_json(receipt, result)
    print(f"VERIFIED {name}: {count} tensors; {result['archive_bytes']} archive bytes", flush=True)
    return result


def evaluate(model, corpus, corpus_name, root, runtime_hash):
    receipt, log_path = root/f"{corpus_name}-{model['name']}.json", root/f"{corpus_name}-{model['name']}.log"
    identity = {"model_sha256": sha256(model["decoded"]), "corpus_sha256": sha256(corpus), "runtime_sha256": runtime_hash}
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if any(result.get(k) != v for k,v in identity.items()):
            raise ValueError("perplexity receipt identity changed")
        return result
    command = [str(BINARY), "-m", model["decoded"], "-f", str(corpus), "--chunks", "64",
        "-c", "512", "-b", "512", "-ub", "512", "-t", "4", "-tb", "4", "-ngl", "0"]
    print(f"EVALUATE {corpus_name} {model['name']}", flush=True)
    started = time.perf_counter()
    execution_error = None
    with log_path.open("x") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            returncode = completed.returncode
        except (subprocess.TimeoutExpired, OSError) as error:
            returncode, execution_error = None, repr(error)
    text = log_path.read_text()
    match = re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)", text)
    chunks = re.search(r"calculating perplexity over (\d+) chunks", text)
    result = dict(identity, name=model["name"], corpus=corpus_name, command=command,
        seconds=time.perf_counter()-started, returncode=returncode, log_sha256=sha256(log_path))
    if returncode == 0 and match and chunks and int(chunks[1]) == 64:
        ppl, se = float(match[1]), float(match[2])
        if np.isfinite(ppl) and ppl > 0 and np.isfinite(se):
            result.update(status="MEASURED", perplexity=ppl, standard_error=se, chunks=64)
        else:
            result.update(status="FAILED", error="nonfinite perplexity")
    else:
        result.update(status="FAILED", error=execution_error or "missing successful 64-chunk perplexity result")
    save_json(receipt, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def comparison(bits, models, evaluations):
    names = [f"q{bits}-scalar", f"q{bits}-diagonal", f"q{bits}-activation"]
    if any(models[n].get("status") != "MEASURED" or evaluations.get(n, {}).get("status") != "MEASURED" for n in names):
        return {"bits": bits, "status": "INCONCLUSIVE", "reason": "a model or inference run failed"}
    scalar, diagonal, aligned = [evaluations[n]["perplexity"] for n in names]
    ratio = models[names[2]]["archive_bytes"] / models[names[0]]["archive_bytes"]
    passed = aligned <= .95*scalar and aligned <= diagonal and ratio <= 1.1
    return {"bits": bits, "status": "MEASURED" if passed else "FAILED",
        "scalar_ppl": scalar, "diagonal_ppl": diagonal, "activation_ppl": aligned, "archive_bytes_ratio": ratio}


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    models_dir, eval_dir = root/"models", root/"evaluations"
    models_dir.mkdir(parents=True, exist_ok=True); eval_dir.mkdir(exist_ok=True)
    eligible, frozen = verified_screen(root/"screen")
    if not eligible:
        raise ValueError("no screen configuration qualified for model trials")
    dependencies = sorted({p.resolve() for p in BINARY.parent.glob("*.so*")})
    version = subprocess.run([str(BINARY), "--version"], capture_output=True, text=True, check=True)
    runtime = {"executable": sha256(BINARY), "libraries": {str(p):sha256(p) for p in dependencies},
        "version": version.stdout + version.stderr}
    save_json(root/"runtime.json", runtime)
    runtime_hash = sha256(root/"runtime.json")
    stage = {"runner_sha256": sha256(Path(__file__)), "screen_manifest_sha256": sha256(root/"screen/manifest.json"),
        "eligible": eligible, "runtime_sha256": runtime_hash, "selection_sha256": sha256(OUT/"selection.txt"),
        "final_test_sha256": sha256(OUT/"final_test.txt")}
    stage_path = root/"model-provenance.json"
    if stage_path.exists() and json.loads(stage_path.read_text()) != stage:
        raise ValueError("model-stage provenance changed")
    save_json(stage_path, stage)
    models = {"source-f32": {"name": "source-f32", "status": "MEASURED", "decoded": str(SOURCE),
        "decoded_sha256": sha256(SOURCE), "archive_bytes": SOURCE.stat().st_size}}
    history = json.loads((ROOT/"research/foundations-2026-09-05/results.json").read_text())
    budget = next(r for r in history["results"] if r["name"] == "budget120")
    preserved = ROOT/"artifacts/foundations-2026-09-05/budget120-final-decoded.gguf"
    if Path(budget["artifact"]).exists() and sha256(budget["artifact"]) == budget["sha256"] and preserved.exists():
        decoded = models_dir/"historical-budget120.gguf"
        if not decoded.exists():
            with (models_dir/"historical-budget120.decode.log").open("x") as log:
                subprocess.run([str(ROOT/"target/release/atom-quantizer"), "decode", budget["artifact"], "--out", str(decoded)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha256(decoded) != sha256(preserved):
            raise ValueError("historical model no longer reproduces its preserved decoded artifact")
        models["historical-budget120"] = {"name": "historical-budget120", "status": "MEASURED", "decoded": str(decoded),
            "decoded_sha256": sha256(decoded), "archive_bytes": budget["bytes"], "archive_sha256": budget["sha256"],
            "note": "historical product reference with different calibration; not a controlled geometry ablation"}
    for selected in eligible:
        bits, rank = selected["bits"], selected["rank"]
        for geometry in ("scalar", "diagonal", "activation"):
            name = f"q{bits}-{geometry}"
            config = {"bits": bits} if geometry == "scalar" else {"bits": bits, "rank": rank, "rank_geometry": geometry}
            try:
                models[name] = build(config, name, models_dir)
            except Exception as error:
                models[name] = {"name": name, "status": "FAILED", "config": config, "error": repr(error)}
                save_json(models_dir/(name+".failed.json"), models[name])
            save_json(root/"models.json", models)
    selection = {name:evaluate(model, OUT/"selection.txt", "selection", eval_dir, runtime_hash)
                 for name,model in models.items() if model["status"] == "MEASURED"}
    selection_results = [comparison(s["bits"], models, selection) for s in eligible]
    passing = [c["bits"] for c in selection_results if c["status"] == "MEASURED"]
    save_json(root/"selection-summary.json", {"comparisons": selection_results, "eligible_for_final": passing})
    final = {}
    if passing:
        names = [n for n in models if n in ("source-f32", "historical-budget120") or any(n.startswith(f"q{b}-") for b in passing)]
        final = {n:evaluate(models[n], OUT/"final_test.txt", "final", eval_dir, runtime_hash) for n in names}
    if screen_manifest() != frozen:
        raise ValueError("implementation or calibration changed during model trials")
    if (sha256(Path(__file__)) != stage["runner_sha256"] or sha256(BINARY) != runtime["executable"]
            or any(sha256(p) != value for p,value in runtime["libraries"].items())
            or sha256(OUT/"selection.txt") != stage["selection_sha256"]
            or sha256(OUT/"final_test.txt") != stage["final_test_sha256"]):
        raise ValueError("model runner, runtime or evaluation corpus changed during execution")
    result = {"experiment": "D01-models", "models": models, "selection": selection_results,
        "final": [comparison(b, models, final) for b in passing],
        "reference_perplexities": {"selection": {n:r.get("perplexity") for n,r in selection.items()},
                                   "final": {n:r.get("perplexity") for n,r in final.items()}},
        "dense_decoded_inference_only": True, "main_codec_gates_changed": False}
    save_json(root/"model-summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2, allow_nan=False), flush=True)
