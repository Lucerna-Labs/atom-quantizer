"""Independent S1 identity, source-role, cumulative error and decision audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .adaptive_fidelity import FIT, SELECT
from .audit_adaptive_fidelity import artifact, native_receipt
from .basis_discovery import save_json
from .full22.data import SOURCE, OUT, ROOT, load_calibration, sha256

ORDER = ("sequential-scalar", "sequential-activation")


def compare_weights(w1, candidate):
    original = {t.name: t for t in gguf.GGUFReader(SOURCE).tensors}
    intermediate = {t.name: t for t in gguf.GGUFReader(w1).tensors}
    final = {t.name: t for t in gguf.GGUFReader(candidate).tensors}
    rows = []
    total = np.zeros(6, dtype=np.float64)
    for name, tensor in original.items():
        w = tensor.data.astype(np.float64)
        a = intermediate[name].data.astype(np.float64)
        b = final[name].data.astype(np.float64)
        first, second = a - w, b - a
        combined = b - w
        values = np.array([np.sum(w*w), np.sum(a*a), np.sum(first*first),
                           np.sum(second*second), np.sum(combined*combined), np.sum(first*second)])
        if not np.isfinite(values).all() or not np.isclose(values[4], values[2]+values[3]+2*values[5], rtol=1e-11, atol=1e-15):
            raise ValueError("cumulative error identity failed")
        total += values
        rows.append({"tensor": name, "first_error_energy": float(values[2]),
                     "second_error_energy": float(values[3]), "total_error_energy": float(values[4]),
                     "cross_term": float(2*values[5])})
    save = dict(zip(("W_energy", "W1_energy", "first_error_energy", "second_error_energy",
                     "total_error_energy", "error_dot_product"), map(float, total)))
    save.update(relative_to_W=save["total_error_energy"]/save["W_energy"],
                relative_to_W1=save["second_error_energy"]/save["W1_energy"],
                first_stage_relative_to_W=save["first_error_energy"]/save["W_energy"],
                cross_term=2*save["error_dot_product"],
                scope="decoded_weight_error_not_inference", tensors=rows)
    return save


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    output = root / "audit-data"
    output.mkdir(exist_ok=False)
    result = json.loads((root / "summary.json").read_text())
    for name in ("manifest.json", "capture-manifest.json"):
        frozen = json.loads((root / name).read_text())["sha256"]
        if any(sha256(p) != h for p, h in frozen.items()):
            raise ValueError("frozen inputs or implementation changed")
    w1 = root / "W1.gguf"
    if sha256(w1) != "c5fb3e2c4e516823058a730d9234e267f0a4e1d94f061449828826c2ba02b7f5":
        raise ValueError("Composer 1 identity differs")
    installation = json.loads((root / "source-install.json").read_text())
    original_hashes = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in gguf.GGUFReader(SOURCE).tensors}
    installed_hashes = {t.name: hashlib.sha256(t.data.tobytes()).hexdigest() for t in gguf.GGUFReader(w1).tensors}
    if (installation["tensor_sha256"] != original_hashes or not installation["original_checkpoint_unchanged"]
            or installation["verified_tensors"] != 272):
        raise ValueError("original source install not verified")
    if sha256(root / "captures/endpoints-selection.acal") != sha256(SELECT[0]):
        raise ValueError("existing capture changed")
    captured, change_counts = {}, {}
    for split, old_paths in (("fit", FIT), ("selection", SELECT)):
        for index, mode in enumerate(("endpoints", "stratified")):
            path = root / "captures/W1" / f"{mode}-{split}.acal"
            meta = json.loads(path.with_suffix(".acal.json").read_text())
            replay = ROOT / "artifacts/calibration-coverage-2026-10-01/captures" / f"{'endpoint' if index == 0 else 'stratified0'}-{split}.acal"
            if sha256(replay) != sha256(old_paths[index]):
                raise ValueError("historical capture replay differs")
            old_meta = json.loads(replay.with_suffix(".acal.json").read_text())
            if meta["installed_gguf"]["tensor_sha256"] != installed_hashes or meta["source_sha256"] != sha256(w1):
                raise ValueError("new observations were not captured with W1 weights")
            for key in ("tokenized_input_sha256", "documents", "sampled_positions_before_cap", "counts"):
                if meta[key] != old_meta[key]:
                    raise ValueError("observation positions differ")
            current, prior = load_calibration(path, w1), load_calibration(old_paths[index], SOURCE)
            if len(current) != 212 or set(current) != set(prior):
                raise ValueError("missing real operators")
            changed = [n for n in current if current[n].tobytes() != prior[n].tobytes()]
            if "token_embd.weight" in changed:
                raise ValueError("embedding token indices changed")
            captured[(split, index)] = path
            change_counts[f"{split}-{mode}"] = {"changed_linear_inputs": len(changed), "identical_embedding_indices": True}
    stages, native_count, checked_tensors, independent_reports, weight_errors = [], 0, 0, {}, {}
    initial = {t.name: 1 if t.data.ndim == 2 else 3 for t in gguf.GGUFReader(w1).tensors}
    for name in ORDER:
        build = result["builds"][name]
        if build["status"] != "FIT_ADMITTED" or not 1 <= len(build["stages"]) <= 3:
            raise ValueError("missing bounded sequential build")
        previous, failures = None, set()
        for index, stage in enumerate(build["stages"]):
            model = stage["model"]
            manifest = artifact(model, output)
            checked_tensors += 272
            if stage["iteration"] != index or (index == 0 and stage["levels"] != initial):
                raise ValueError("incorrect initial precision or stage order")
            corrected = name == "sequential-activation"
            if (manifest["source_sha256"] != sha256(w1)
                    or manifest["calibration_sha256"] != sha256(captured[("fit", 1)])
                    or manifest.get("residual_calibration_sha256") != (sha256(captured[("fit", 0)]) if corrected else None)):
                raise ValueError("second-stage source/calibration roles differ")
            for entry in manifest["tensors"]:
                level = (4, 6, 8, "exact")[stage["levels"][entry["name"]]]
                expected = {"family": "lossless"} if level == "exact" else {
                    "bits": level, **({"rank": 2, "rank_geometry": "activation"} if corrected else {})}
                if entry["config"] != expected:
                    raise ValueError("candidate numerical configuration differs")
            retained = 0
            if previous is not None:
                if stage["levels"] != {n: v + int(n in failures) for n, v in previous["levels"].items()}:
                    raise ValueError("escalation used a different reference or a selection observation")
                with zipfile.ZipFile(previous["model"]["archive"]) as old, zipfile.ZipFile(model["archive"]) as new:
                    entries = {t["name"]: t for t in json.loads(old.read("manifest.json"))["tensors"]}
                    for entry in manifest["tensors"]:
                        if entry["name"] not in failures:
                            if not entry["reused"] or old.read(entries[entry["name"]]["entry"]) != new.read(entry["entry"]):
                                raise ValueError("passing stored record changed")
                            retained += 1
            failures = set()
            if len(stage["native_reports"]) != 2:
                raise ValueError("missing native fitting view")
            for view, path in enumerate(stage["native_reports"]):
                report = json.loads(Path(path).read_text())
                if report["report"]["source_sha256"] != sha256(w1):
                    raise ValueError("fitting decisions used the wrong model")
                failures.update(native_receipt(path, model, captured[("fit", view)]))
                native_count += 1
            if failures != set(stage["failed_tensors"]):
                raise ValueError("stage failure union differs")
            stages.append({"name": name, "stage": index, "failures": len(failures),
                           "retained_records": retained, "archive_bytes": model["archive_bytes"]})
            previous = stage
        if failures:
            raise ValueError("final model did not pass its fitting inputs")
        model = result["models"][name]
        if sha256(model["archive"]) != previous["model"]["archive_sha256"] or sha256(model["decoded"]) != previous["model"]["decoded_sha256"]:
            raise ValueError("native final model differs from stored stages")
        independent_reports[name] = {}
        for reference, source, views in (("W", SOURCE, (FIT, SELECT)), ("W1", w1, (
                [captured[("fit", i)] for i in range(2)], [captured[("selection", i)] for i in range(2)]))):
            for split, observations in zip(("fit", "selection"), views):
                for view, calibration in enumerate(observations):
                    key = f"{reference}-{split}-{view}"
                    path = root / "native" / f"{name}-{key}.json"
                    receipt = json.loads(path.read_text())
                    if receipt != result["native"][name][key] or receipt["report"]["source_sha256"] != sha256(source):
                        raise ValueError("reference-native report changed")
                    failed = native_receipt(path, model, calibration)
                    independent_reports[name][key] = {"failed_tensors": sorted(failed),
                        "gate_failures": receipt["report"]["gate_failures"]}
                    native_count += 1
        weight_errors[name] = compare_weights(w1, model["decoded"])
    inference = []
    for split, corpus in (("selection_scores", "selection"), ("final_scores", "new-final")):
        for name, score in result[split].items():
            path = Path(score["reused_from"]) if "reused_from" in score else root / "evaluations" / f"{corpus}-{name}.json"
            saved, log = json.loads(path.read_text()), path.with_suffix(".log")
            for key in ("status", "command", "model_sha256", "corpus_sha256", "runtime_sha256", "log_sha256", "chunks", "returncode", "perplexity"):
                if saved[key] != score[key]:
                    raise ValueError("inference receipt differs")
            if (score["status"] != "MEASURED" or score["returncode"] != 0 or score["chunks"] != 64
                    or sha256(result["models"][name]["decoded"]) != score["model_sha256"]
                    or sha256(score["command"][4]) != score["corpus_sha256"]
                    or sha256(log) != score["log_sha256"] or sha256(root / "runtime.json") != score["runtime_sha256"]
                    or score["command"][5:] != ["--chunks", "64", "-c", "512", "-b", "512", "-ub", "512", "-t", "4", "-tb", "4", "-ngl", "0"]):
                raise ValueError("inference identity or conditions differ")
            text = log.read_text()
            match = re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)", text)
            count = re.search(r"calculating perplexity over (\d+) chunks", text)
            if not match or not count or int(count[1]) != 64 or float(match[1]) != score["perplexity"]:
                raise ValueError("inference does not match actual log")
            inference.append({"name": name, "split": split, "ppl": score["perplexity"], "reused": "reused_from" in score})
    p, s, history, direct = "sequential-activation", "sequential-scalar", "historical-budget120", "six-activation"
    models, scores = result["models"], result["selection_scores"]
    expected_G3 = {
        "all_W_and_W1_native_views": all(not r["failed_tensors"] for r in independent_reports[p].values()),
        "W1_bytes": models[p]["archive_bytes"] <= .9 * models[history]["archive_bytes"],
        "W1_ppl": scores[p]["perplexity"] <= 1.01 * scores[history]["perplexity"],
        "scalar_bytes": models[p]["archive_bytes"] <= 1.05 * models[s]["archive_bytes"],
        "scalar_ppl": scores[p]["perplexity"] <= scores[s]["perplexity"],
    }
    expected_G4 = {"direct_bytes": models[p]["archive_bytes"] <= .95 * models[direct]["archive_bytes"],
                   "direct_ppl": scores[p]["perplexity"] <= 1.01 * scores[direct]["perplexity"]}
    qualified = all(expected_G3.values()) and all(expected_G4.values())
    if (result["selection_gate"]["G3"]["checks"] != expected_G3 or result["selection_gate"]["G4"]["checks"] != expected_G4
            or result["selection_gate"]["status"] != ("QUALIFIED" if qualified else "FAILED")):
        raise ValueError("benefit decision differs from precommit")
    if not qualified and (result["final_scores"] or result["final_gate"]["status"] != "UNRUN" or (root / "final-use.json").exists()):
        raise ValueError("failed candidate consumed final data")
    save_json(root / "weight-errors.json", weight_errors)
    report = {"status": "PASSED", "stages": stages, "checked_tensors": checked_tensors,
              "native_reports": native_count, "native_tensor_decisions": native_count * 272,
              "capture_changes": change_counts, "native_comparisons": independent_reports,
              "inference": inference, "selection_gate": result["selection_gate"], "final_gate": result["final_gate"],
              "weight_errors_sha256": sha256(root / "weight-errors.json"), "summary_sha256": sha256(root / "summary.json"),
              "auditor_sha256": sha256(Path(__file__))}
    save_json(root / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("status", "checked_tensors", "native_reports", "selection_gate", "final_gate")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    run(parser.parse_args().run)
