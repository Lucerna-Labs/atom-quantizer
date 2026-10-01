"""S1: real Composer 1 -> functional compressor, with both references retained."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .adaptive_fidelity import ASSESSOR, FIT, SELECT
from .basis_discovery import save_json
from .calibration_coverage import TOKENIZER, native_ids
from .direction_models import BINARY, evaluate, validate_export
from .full22 import native, precision
from .full22.data import MODEL, OUT, ROOT, SOURCE, load_calibration, sha256

D05 = ROOT / "artifacts/six-bit-frontier-2026-10-01"
PROTOCOL = ROOT / "research/sequential-functional-2026-10-01/PROTOCOL.md"
MAIN = ROOT / "target/release/atom-quantizer"
HISTORICAL = ROOT / "artifacts/foundations-2026-09-05/budget120.oq"
D02_CAPTURES = ROOT / "artifacts/calibration-coverage-2026-10-01/captures"
ORDER = ("sequential-scalar", "sequential-activation")
CONTROLS = ("source-f32", "historical-budget120", "six-scalar", "six-activation")


def gates(models, scores, reports):
    required = (*CONTROLS, *ORDER)
    if any(n not in models or scores.get(n, {}).get("status") != "MEASURED" for n in required):
        return {"status": "INCONCLUSIVE"}
    p, s, w1, direct = "sequential-activation", "sequential-scalar", "historical-budget120", "six-activation"
    rows = reports.get(p, {})
    complete = set(rows) == {f"{ref}-{split}-{i}" for ref in ("W", "W1")
                             for split in ("fit", "selection") for i in range(2)}
    checks = {
        "all_W_and_W1_native_views": complete and all(r["report"]["all_passed"] for r in rows.values()),
        "W1_bytes": models[p]["archive_bytes"] <= .90 * models[w1]["archive_bytes"],
        "W1_ppl": scores[p]["perplexity"] <= 1.01 * scores[w1]["perplexity"],
        "scalar_bytes": models[p]["archive_bytes"] <= 1.05 * models[s]["archive_bytes"],
        "scalar_ppl": scores[p]["perplexity"] <= scores[s]["perplexity"],
    }
    improvement = {
        "direct_bytes": models[p]["archive_bytes"] <= .95 * models[direct]["archive_bytes"],
        "direct_ppl": scores[p]["perplexity"] <= 1.01 * scores[direct]["perplexity"],
    }
    return {"status": "QUALIFIED" if all(checks.values()) and all(improvement.values()) else "FAILED",
            "G3": {"status": "MEASURED" if all(checks.values()) else "FAILED", "checks": checks},
            "G4": {"status": "MEASURED" if all(improvement.values()) else "FAILED", "checks": improvement}}


def run_command(command, log):
    print("RUN", log.name, flush=True)
    with log.open("x") as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)


def capture(root, source, split, sampling, *, install=True):
    output = root / f"{sampling}-{split}.acal"
    command = [sys.executable, "-m", "experiments.full22.capture", "--split", split,
               "--sampling", sampling, "--seed", "0", "--source", source, "--out", output]
    if install:
        command.append("--load-gguf-weights")
    run_command(command, output.with_suffix(".log"))
    manifest = json.loads(output.with_suffix(".acal.json").read_text())
    values = load_calibration(output, source)
    count = 256 if split == "fit" else 64
    if (manifest["verified_tensors"] != 272 or len(values) != 212
            or set(manifest["counts"].values()) != {count}
            or any(not np.isfinite(v).all() or len(v) != count for v in values.values())
            or (install and manifest.get("installed_gguf", {}).get("verified_tensors") != 272)):
        raise ValueError("capture is incomplete or not bound to installed weights")
    return output, manifest


def check_original_installation(root):
    program = """
import hashlib,json,sys
import gguf,torch
from transformers import AutoModelForCausalLM
from experiments.full22.gguf_weights import install
torch.set_num_threads(4)
model=AutoModelForCausalLM.from_pretrained(sys.argv[1],local_files_only=True,
    trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval()
def hashes():
    return {n:hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
            for n,p in model.named_parameters(remove_duplicate=False)}
before=hashes()
receipt=install(model,gguf.GGUFReader(sys.argv[2]))
if hashes()!=before: raise ValueError('source install differs from the actual HF checkpoint')
print(json.dumps(dict(receipt,original_checkpoint_unchanged=True)))
"""
    log = root / "source-install.log"
    run_command([sys.executable, "-c", program, MODEL, SOURCE], log)
    receipt = json.loads(log.read_text().splitlines()[-1])
    if receipt["verified_tensors"] != 272 or not receipt["original_checkpoint_unchanged"]:
        raise ValueError("source installation coverage incomplete")
    save_json(root / "source-install.json", receipt)


def reference_models(root, runtime_hash):
    previous = json.loads((D05 / "summary.json").read_text())
    models, scores = {}, {}
    for name in CONTROLS:
        model = dict(previous["models"][name], name=name)
        if name == "historical-budget120":
            model["archive"] = str(HISTORICAL)
        if sha256(model["decoded"]) != model["decoded_sha256"]:
            raise ValueError("reference model changed")
        if "archive" in model and (sha256(model["archive"]) != model["archive_sha256"]
                or Path(model["archive"]).stat().st_size != model["archive_bytes"]):
            raise ValueError("reference archive changed")
        score = previous["selection_scores"][name]
        receipt = Path(score.get("reused_from", D05 / "evaluations" / f"selection-{name}.json"))
        log = receipt.with_suffix(".log")
        if (score["status"] != "MEASURED" or score["chunks"] != 64 or score["returncode"] != 0
                or score["model_sha256"] != model["decoded_sha256"]
                or score["corpus_sha256"] != sha256(OUT / "selection.txt")
                or score["runtime_sha256"] != runtime_hash or score["log_sha256"] != sha256(log)):
            raise ValueError("reference inference identity changed")
        measured = json.loads(receipt.read_text())
        for key in ("status", "model_sha256", "corpus_sha256", "runtime_sha256", "log_sha256", "perplexity"):
            if measured[key] != score[key]:
                raise ValueError("reference receipt differs from recorded score")
        models[name], scores[name] = model, dict(score, reused_from=str(receipt))
    save_json(root / "references.json", {"models": models, "scores": scores})
    return models, scores


def fresh_final(root):
    original = OUT / "final_test.txt"
    text = original.read_text()
    full = native_ids(text)
    lower = 100418 + 32768 + 1024
    if len(full) < lower + 32768:
        result = {"status": "UNRUN", "reason": "remaining final tail is shorter than 64 chunks",
                  "native_total_tokens": len(full), "required_prefix": lower, "required_tail": 32768,
                  "source_sha256": sha256(original), "tokenizer_sha256": sha256(TOKENIZER)}
        save_json(root / "freshness.json", result)
        return None
    # Only boundaries after the already consumed character region are considered.
    for cut in (i + 1 for i, c in enumerate(text) if c == "\n" and i >= 439133):
        prefix = native_ids(text[:cut])
        common = next((i for i, (a, b) in enumerate(zip(full, prefix)) if a != b), min(len(full), len(prefix)))
        if common < lower:
            continue
        suffix = text[cut:]
        count = len(native_ids(suffix.removesuffix("\n")))
        if count < 32768:
            break
        digest = hashlib.sha256(suffix.encode()).hexdigest()
        for path in (ROOT / "artifacts").glob("**/evaluations/*.json"):
            if json.loads(path.read_text()).get("corpus_sha256") == digest:
                raise ValueError("prospective final data already used")
        output = root / "new-final.txt"
        with output.open("x") as stream:
            stream.write(suffix)
        save_json(root / "freshness.json", {"status": "MEASURED", "cut_character": cut,
                  "excluded_common_prefix_tokens": common, "fresh_tokens": count,
                  "source_sha256": sha256(original), "fresh_sha256": digest,
                  "tokenizer_sha256": sha256(TOKENIZER)})
        return output
    save_json(root / "freshness.json", {"status": "UNRUN", "reason": "no qualifying newline boundary"})
    return None


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("captures", "native", "evaluations"):
        (root / name).mkdir()
    runtime = json.loads((D05 / "runtime.json").read_text())
    if sha256(BINARY) != runtime["executable"] or any(sha256(p) != h for p, h in runtime["libraries"].items()):
        raise ValueError("inference runtime changed")
    prior_hashes = json.loads((D05 / "manifest.json").read_text())["sha256"]
    if any(sha256(p) != prior_hashes[str(p)] for p in (SOURCE, *FIT, *SELECT, OUT / "selection.txt", OUT / "final_test.txt")):
        raise ValueError("D05 source or observations changed")
    if (sha256(MAIN) != "d45eaa37c05ae4d08001e7b08ab3d90a0cfe9c970e03c0f06351558d6c9a400b"
            or sha256(ASSESSOR) != "628c4d10a9513379c6a37b14d617469bffd5333de81da8fffe642cf4324417bd"
            or sha256(SOURCE) != "53f86c219b099fa84cd005f1b22572f6c4e53f8ba242fb8074f600712abe9c5b"
            or sha256(HISTORICAL) != "6d8e7d7cf6c734866751630a96be7315133a04c1852bc1fdef1b6e1686e3fde1"):
        raise ValueError("precommitted source or executable changed")
    save_json(root / "runtime.json", runtime)
    runtime_hash = sha256(root / "runtime.json")
    files = [SOURCE, HISTORICAL, *FIT, *SELECT, MAIN, ASSESSOR, BINARY, TOKENIZER, PROTOCOL,
             OUT / "fit.jsonl", OUT / "selection_inputs.jsonl", OUT / "selection.txt", OUT / "final_test.txt"]
    files += [Path(p) for p in runtime["libraries"]]
    files += sorted((ROOT / "experiments").rglob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.py"))
    files += [p for p in MODEL.iterdir() if p.is_file()]
    files += [PROTOCOL.with_name("METADATA_CORRECTION.md")]
    files += [D02_CAPTURES / f"{mode}-{split}{suffix}" for mode in ("endpoint", "stratified0")
              for split in ("fit", "selection") for suffix in (".acal", ".acal.json")]
    frozen = {str(p): sha256(p) for p in files}
    save_json(root / "manifest.json", {"sha256": frozen, "experiment": "S1", "candidate_order": ORDER})
    models, scores = reference_models(root, runtime_hash)
    for model in models.values():
        for key in ("archive", "decoded"):
            if key in model:
                frozen[model[key]] = sha256(model[key])
    save_json(root / "manifest.json", {"sha256": frozen, "experiment": "S1", "candidate_order": ORDER})
    check_original_installation(root)
    w1 = root / "W1.gguf"
    run_command([MAIN, "decode", HISTORICAL, "--out", w1], root / "W1-decode.log")
    if sha256(w1) != "c5fb3e2c4e516823058a730d9234e267f0a4e1d94f061449828826c2ba02b7f5":
        raise ValueError("fresh Composer 1 decode differs")
    replay, replay_manifest = capture(root / "captures", SOURCE, "selection", "endpoints", install=False)
    if sha256(replay) != sha256(SELECT[0]):
        raise ValueError("existing capture behavior changed")
    # Keep the source replay separate from four W1 captures.
    new_captures = root / "captures" / "W1"
    new_captures.mkdir()
    paths, captures = {}, {}
    for split in ("fit", "selection"):
        for sampling in ("endpoints", "stratified"):
            path, meta = capture(new_captures, w1, split, sampling)
            old_capture = D02_CAPTURES / f"{'endpoint' if sampling == 'endpoints' else 'stratified0'}-{split}.acal"
            reference = (FIT if split == "fit" else SELECT)[0 if sampling == "endpoints" else 1]
            if sha256(old_capture) != sha256(reference):
                raise ValueError("historical replay payload differs from original observations")
            old = old_capture.with_suffix(".acal.json")
            old_meta = json.loads(old.read_text())
            for key in ("tokenized_input_sha256", "documents", "sampled_positions_before_cap", "counts"):
                if meta[key] != old_meta[key]:
                    raise ValueError("sequential capture inputs differ from the direct control")
            paths[(split, sampling)], captures[f"{split}-{sampling}"] = path, meta
    if len(set(sha256(p) for p in paths.values())) != 4:
        raise ValueError("new observation files are not distinct")
    save_json(root / "capture-summary.json", {"replay_sha256": sha256(replay), "captures": captures})
    generated = {str(p): sha256(p) for p in (w1, replay, *paths.values())}
    save_json(root / "capture-manifest.json", {"sha256": generated})
    builds, reports = {}, {}
    fit = [paths[("fit", mode)] for mode in ("endpoints", "stratified")]
    selection = [paths[("selection", mode)] for mode in ("endpoints", "stratified")]
    for name in ORDER:
        correction = name == "sequential-activation"
        build = precision.run(w1, fit[1], fit[0] if correction else None, fit, ASSESSOR, root / name,
                              start_bits=6, rank_geometry="activation" if correction else None)
        builds[name] = build
        model = dict(build["model"], name=name)
        decoded = root / f"{name}-native.gguf"
        run_command([MAIN, "decode", model["archive"], "--out", decoded], root / f"{name}-native.log")
        if sha256(decoded) != model["decoded_sha256"] or validate_export(model["archive"], decoded) != 272:
            raise ValueError("native second-stage export differs")
        model["decoded"] = str(decoded)
        models[name] = model
        reports[name] = {}
        for reference, source, views in (("W", SOURCE, (FIT, SELECT)), ("W1", w1, (fit, selection))):
            for split, observations in zip(("fit", "selection"), views):
                for index, calibration in enumerate(observations):
                    key = f"{reference}-{split}-{index}"
                    reports[name][key] = native.assess(ASSESSOR, source, decoded, calibration,
                                                     root / "native" / f"{name}-{key}.json")
        scores[name] = evaluate(model, OUT / "selection.txt", "selection", root / "evaluations", runtime_hash)
        save_json(root / "progress.json", {"models": models, "builds": builds, "reports": reports, "scores": scores})
    selected = gates(models, scores, reports)
    save_json(root / "selection-gate.json", selected)
    final, final_gate = {}, {"status": "UNRUN", "reason": "selection did not qualify"}
    if selected["status"] == "QUALIFIED":
        holdout = fresh_final(root)
        if holdout is None:
            final_gate = {"status": "UNRUN", "reason": "a new final corpus must be precommitted"}
        else:
            use = {"corpus_sha256": sha256(holdout), "models": list(models), "status": "STARTED"}
            save_json(root / "final-use.json", use)
            for name, model in models.items():
                final[name] = evaluate(model, holdout, "new-final", root / "evaluations", runtime_hash)
                save_json(root / "final-progress.json", final)
            final_gate = gates(models, final, reports)
            save_json(root / "final-use.json", dict(use, status="COMPLETED"))
    if any(sha256(p) != h for p, h in {**frozen, **generated}.items()):
        raise ValueError("frozen input, implementation or runtime changed")
    result = {"experiment": "S1", "models": models, "builds": builds, "native": reports,
              "selection_scores": scores, "selection_gate": selected, "final_scores": final,
              "final_gate": final_gate, "source_replay_identical": True,
              "new_inference_runs": len(ORDER) + len(final), "defaults_changed": False}
    save_json(root / "summary.json", result)
    print(json.dumps({"selection_gate": selected, "final_gate": final_gate}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    destination = Path(parser.parse_args().out).resolve()
    try:
        run(destination)
    except Exception as error:
        if destination.is_dir() and not (destination / "failure.json").exists():
            save_json(destination / "failure.json", {"status": "FAILED", "error": repr(error)})
        raise
