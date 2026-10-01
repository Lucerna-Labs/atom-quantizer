"""D03: independently fit base quantization and a stored residual correction."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .calibration_coverage import native_ids, TOKENIZER
from .direction_models import BINARY, build, evaluate, validate_export
from .full22.data import ROOT, OUT, SOURCE, sha256
from .full22.records import Record

D01 = ROOT/"artifacts/direction-retention-2026-09-30"
D02 = ROOT/"artifacts/calibration-coverage-2026-10-01"
PROTOCOL = ROOT/"research/calibration-roles-2026-10-01/PROTOCOL.md"
CALIBRATIONS = {"E": OUT/"fit.acal", "I": D02/"captures/stratified0-fit.acal"}
EXPECTED = {SOURCE: "53f86c219b099fa84cd005f1b22572f6c4e53f8ba242fb8074f600712abe9c5b",
    CALIBRATIONS["E"]: "f5e6d79eb6b552da48cf976ea4c114d0ac387689ef42de5fa8746e257056431f",
    CALIBRATIONS["I"]: "29d09e1701dc329cee26fd636dd1d492c3e29b1e1c3d17d28e2c777a52e9a2c4",
    D02/"fresh-final.txt": "42ad320614e02063e1a484efb02fcf2c58f81552aef71ef0468cbd4cb7f68bbd"}


def case_name(bits, rank, role):
    return f"q{bits}-{role}-r{rank}"


def base_identity(artifact, scalar_artifact):
    """Deleting only stored correction state must reproduce the entire scalar record."""
    checked, corrected = 0, 0
    with zipfile.ZipFile(artifact) as candidate, zipfile.ZipFile(scalar_artifact) as scalar:
        manifest = json.loads(candidate.read("manifest.json"))
        control = json.loads(scalar.read("manifest.json"))
        if (manifest["source_sha256"] != control["source_sha256"]
                or manifest["calibration_sha256"] != control["calibration_sha256"]
                or len(manifest["tensors"]) != len(control["tensors"])):
            raise ValueError("base identity inputs differ")
        for entry, reference in zip(manifest["tensors"], control["tensors"]):
            if entry["name"] != reference["name"]:
                raise ValueError("base tensor order differs")
            record = Record.loads(candidate.read(entry["entry"]))
            expected = Record.loads(scalar.read(reference["entry"]))
            if "lowrank_u" in record.arrays:
                corrected += 1
            record.arrays.pop("lowrank_u", None)
            record.arrays.pop("lowrank_v", None)
            record.meta.pop("lowrank_effective_rank", None)
            for key in ("rank", "rank_geometry"):
                record.meta.get("config", {}).pop(key, None)
            if record.dumps() != expected.dumps():
                raise ValueError(f"correction changed the stored base: {entry['name']}")
            checked += 1
    return {"identical_base_records": checked, "corrected_records": corrected,
            "scalar_archive_sha256": sha256(scalar_artifact)}


def reuse(old_root, old_name, name, config, role, runtime_hash):
    if old_name in ("source-f32", "historical-budget120"):
        model = json.loads((D01/"models.json").read_text())[old_name]
    else:
        model = json.loads((old_root/"models"/(old_name+".model.json")).read_text())
    if model["status"] != "MEASURED" or sha256(model["decoded"]) != model["decoded_sha256"]:
        raise ValueError("reused decoded model changed")
    if config is not None:
        if (model["config"] != config or sha256(model["archive"]) != model["archive_sha256"]
                or Path(model["archive"]).stat().st_size != model["archive_bytes"]):
            raise ValueError("reused archive changed")
        with zipfile.ZipFile(model["archive"]) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if (manifest["config"] != config or manifest["source_sha256"] != EXPECTED[SOURCE]
                    or manifest["calibration_sha256"] != EXPECTED[CALIBRATIONS[role]]
                    or manifest.get("residual_calibration_sha256", manifest["calibration_sha256"])
                        != EXPECTED[CALIBRATIONS[role]]):
                raise ValueError("reused archive role identity differs")
        if validate_export(model["archive"], model["decoded"]) != 272:
            raise ValueError("incomplete reused export")
    if old_name == "historical-budget120":
        history = json.loads((ROOT/"research/foundations-2026-09-05/results.json").read_text())
        reference = next(r for r in history["results"] if r["name"] == "budget120")
        if sha256(reference["artifact"]) != model["archive_sha256"]:
            raise ValueError("historical artifact changed")
    score_path = old_root/"evaluations"/f"selection-{old_name}.json"
    score = json.loads(score_path.read_text())
    if (score["status"] != "MEASURED" or score["chunks"] != 64 or score["returncode"] != 0
            or score["model_sha256"] != model["decoded_sha256"]
            or score["corpus_sha256"] != sha256(OUT/"selection.txt")
            or score["runtime_sha256"] != runtime_hash
            or score["log_sha256"] != sha256(score_path.with_suffix(".log"))):
        raise ValueError("reused inference identity differs")
    return dict(model, name=name, reused_from=str(old_root), original_name=old_name), dict(
        score, name=name, reused_from=str(score_path))


def freshness(root):
    proof = json.loads((D02/"freshness.json").read_text())
    if (sha256(TOKENIZER) != proof["tokenizer_sha256"] or sha256(OUT/"final_test.txt") != proof["source_sha256"]
            or sha256(D02/"fresh-final.txt") != proof["fresh_sha256"]):
        raise ValueError("freshness proof inputs changed")
    text = (OUT/"final_test.txt").read_text()
    cut = text.find("\n", len(text)//3)+1
    full, prefix = native_ids(text), native_ids(text[:cut])
    common = next((i for i, (a,b) in enumerate(zip(full,prefix)) if a != b), min(len(full),len(prefix)))
    fresh_text = (D02/"fresh-final.txt").read_text()
    if (fresh_text != text[cut:] or cut != proof["cut_character"] or common != proof["excluded_common_prefix_tokens"]
            or common < 33792 or len(native_ids(fresh_text.removesuffix("\n"))) != proof["fresh_tokens"]
            or hashlib.sha256(np.asarray(full,dtype="<i4").tobytes()).hexdigest() != proof["full_ids_sha256"]):
        raise ValueError("native token separation did not reproduce")
    prior = [str(p) for p in (ROOT/"artifacts").glob("**/evaluations/*fresh-final*") if not p.is_relative_to(root)]
    if prior:
        raise ValueError(f"fresh final was previously used: {prior}")
    save_json(root/"freshness.json", dict(proof, prior_external_fresh_evaluations=prior))


def check_gate(bits, rank, models, scores):
    principal = case_name(bits, rank, "IE")
    roles = [case_name(bits, rank, r) for r in ("EE", "II", "EI")]
    diagonal = case_name(bits, rank, "diagonal")
    previous = [case_name(bits, 1, r) for r in ("EE", "diagonal")]
    scalar = [f"q{bits}-scalar-{r}" for r in "EI"]
    names = list(dict.fromkeys([principal, *roles, diagonal, *previous, *scalar]))
    if any(models.get(n, {}).get("status") != "MEASURED" or scores.get(n, {}).get("status") != "MEASURED" for n in names):
        return {"bits":bits, "rank":rank, "status":"INCONCLUSIVE"}
    ppl = scores[principal]["perplexity"]
    best_roles = min(scores[n]["perplexity"] for n in roles)
    best_previous = min(scores[n]["perplexity"] for n in previous)
    size_ratio = models[principal]["archive_bytes"] / min(models[n]["archive_bytes"] for n in scalar)
    gates = {"role_improvement":ppl <= .95*best_roles,
             "matched_diagonal":ppl <= scores[diagonal]["perplexity"],
             "previous_rank1":ppl <= best_previous, "bytes":size_ratio <= 1.10}
    return {"bits":bits, "rank":rank, "status":"QUALIFIED" if all(gates.values()) else "FAILED",
        "perplexity":ppl, "best_other_role_ppl":best_roles, "diagonal_ppl":scores[diagonal]["perplexity"],
        "best_previous_ppl":best_previous, "archive_bytes_ratio":size_ratio, "gates":gates}


def execute_case(args):
    bits, rank, role, root, runtime_hash, scalar_artifact = args
    threadpool_limits(4)
    name = case_name(bits,rank,role)
    geometry = "diagonal" if role == "diagonal" else "activation"
    config = {"bits":bits, "rank":rank, "rank_geometry":geometry}
    base, correction = ("E", "E") if role == "diagonal" else tuple(role)
    try:
        model = build(config,name,root/"models",CALIBRATIONS[base],CALIBRATIONS[correction])
        identity = base_identity(model["archive"],scalar_artifact)
        save_json(root/"models"/(name+".base.json"),identity)
        score = evaluate(model,OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
    except Exception as error:
        model = {"name":name,"config":config,"status":"FAILED","error":repr(error)}
        score = {"name":name,"status":"FAILED","error":repr(error)}
        save_json(root/"models"/(name+".failure.json"),model)
    return name, model, score


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    for directory in ("models", "evaluations"):
        (root/directory).mkdir(parents=True,exist_ok=True)
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"precommitted input changed: {path}")
    runtime = json.loads((D01/"runtime.json").read_text())
    if sha256(BINARY) != runtime["executable"] or any(sha256(p)!=h for p,h in runtime["libraries"].items()):
        raise ValueError("inference runtime changed")
    save_json(root/"runtime.json",runtime)
    runtime_hash = sha256(root/"runtime.json")
    files = [Path(__file__), PROTOCOL, ROOT/"experiments/direction_models.py", ROOT/"experiments/direction_retention.py",
        ROOT/"experiments/calibration_coverage.py", ROOT/"experiments/basis_discovery.py",
        ROOT/"tests/test_calibration_roles.py"] + sorted((ROOT/"experiments/full22").glob("*.py"))
    identities = {str(p):sha256(p) for p in [*files, *EXPECTED, OUT/"selection.txt", D01/"runtime.json",
        BINARY, *map(Path,runtime["libraries"])]}
    frozen = {"experiment":"D03", "sha256":identities, "workers":2}
    path = root/"manifest.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("D03 implementation or inputs changed; retain this run")
    save_json(path,frozen)
    freshness(root)
    models, scores = {}, {}
    for name in ("source-f32", "historical-budget120"):
        models[name],scores[name] = reuse(D01,name,name,None,None,runtime_hash)
    for bits in (3,4):
        for role,old_root,prefix in (("E",D01,""),("I",D02,"strat0-")):
            name = f"q{bits}-scalar-{role}"
            models[name],scores[name] = reuse(old_root,f"{prefix}q{bits}-scalar",name,{"bits":bits},role,runtime_hash)
            name = case_name(bits,1,role*2)
            models[name],scores[name] = reuse(old_root,f"{prefix}q{bits}-activation",name,
                {"bits":bits,"rank":1,"rank_geometry":"activation"},role,runtime_hash)
        name = case_name(bits,1,"diagonal")
        models[name],scores[name] = reuse(D01,f"q{bits}-diagonal",name,
            {"bits":bits,"rank":1,"rank_geometry":"diagonal"},"E",runtime_hash)
    reused = {"models":dict(models),"selection":dict(scores)}
    for name,model in models.items():
        if "-r1" in name:
            base = "I" if "-II-" in name else "E"
            save_json(root/"models"/(name+".base.json"),base_identity(model["archive"],models[f"{name[:2]}-scalar-{base}"]["archive"]))
    save_json(root/"reused.json",reused)
    work = []
    for bits in (3,4):
        for rank in (1,2,4):
            for role in ("EE","II","IE","EI","diagonal"):
                if case_name(bits,rank,role) not in models:
                    base = "E" if role == "diagonal" else role[0]
                    work.append((bits,rank,role,root,runtime_hash,models[f"q{bits}-scalar-{base}"]["archive"]))
    if len(work) != 24 or len(reused["models"]) != 12:
        raise ValueError("unexpected case count")
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(execute_case,item) for item in work]):
            name,model,score = future.result()
            models[name],scores[name] = model,score
            save_json(root/"progress.json",{"models":models,"selection":scores})
            print(f"COMPLETE {name}: {score.get('perplexity',score['status'])}",flush=True)
    selection = [check_gate(b,r,models,scores) for b in (3,4) for r in (1,2,4)]
    selected = {}
    if all(s.get("status") == "MEASURED" for s in scores.values()):
        for bits in (3,4):
            ranks = [g["rank"] for g in selection if g["bits"] == bits and g["status"] == "QUALIFIED"]
            if ranks:
                selected[bits] = min(ranks)
    save_json(root/"selection-summary.json",{"decisions":selection,"selected":selected})
    final, final_decisions = {}, []
    if selected:
        names = {"source-f32","historical-budget120"}
        for bits,rank in selected.items():
            names.update(case_name(bits,rank,r) for r in ("EE","II","IE","EI","diagonal"))
            names.update(case_name(bits,1,r) for r in ("EE","diagonal"))
            names.update(f"q{bits}-scalar-{r}" for r in "EI")
        save_json(root/"fresh-final-use.json",{"status":"STARTED","models":sorted(names),"corpus_sha256":EXPECTED[D02/"fresh-final.txt"]})
        for name in sorted(names):
            final[name] = evaluate(models[name],D02/"fresh-final.txt","fresh-final",root/"evaluations",runtime_hash)
            save_json(root/"final-progress.json",final)
        final_decisions = [check_gate(bits,rank,models,final) for bits,rank in selected.items()]
        save_json(root/"fresh-final-use.json",{"status":"COMPLETED","models":sorted(names),"corpus_sha256":EXPECTED[D02/"fresh-final.txt"]})
    for path,expected in identities.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen identity changed during D03: {path}")
    result = {"experiment":"D03","models":models,"selection":scores,"selection_decisions":selection,
        "selected_ranks":selected,"fresh_final":final,"final_decisions":final_decisions,
        "new_cases":len(work),"reused_cases":len(reused["models"]),"defaults_changed":False,
        "dense_decoded_inference_only":True}
    save_json(root/"summary.json",result)
    print(json.dumps({k:result[k] for k in ("selection_decisions","selected_ranks","final_decisions")}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
