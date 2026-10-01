"""D02: same-budget calibration sampling, followed by real model comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .direction_models import BINARY, build, evaluate
from .full22 import codec
from .full22.data import ROOT, OUT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record

TOKENIZER = BINARY.parent/"llama-tokenize"
D01 = ROOT/"artifacts/direction-retention-2026-09-30"
GEOMETRIES = ("scalar", "diagonal", "activation")
PROTOCOL = ROOT/"research/calibration-coverage-2026-10-01/PROTOCOL.md"


def native_ids(text):
    command = [str(TOKENIZER), "-m", str(SOURCE), "--stdin", "--ids", "--no-parse-special"]
    result = subprocess.run(command, input=text, text=True, capture_output=True, check=True)
    ids = json.loads(result.stdout)
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ValueError("invalid native token stream")
    return ids


def fresh_holdout(root):
    original = OUT/"final_test.txt"
    text = original.read_text()
    if "\\" in text:
        raise ValueError("evaluation escaping needs a new boundary analysis")
    cut = text.find("\n", len(text)//3)+1
    if cut <= 0:
        raise ValueError("no declared fresh boundary")
    full, prefix = native_ids(text), native_ids(text[:cut])
    common = next((i for i,(a,b) in enumerate(zip(full,prefix)) if a != b), min(len(full),len(prefix)))
    suffix = text[cut:]
    fresh = native_ids(suffix.removesuffix("\n"))
    if common < 33792 or len(fresh) < 32768:
        raise ValueError("fresh evaluation scope is insufficient")
    path = root/"fresh-final.txt"
    if path.exists():
        if path.read_text() != suffix:
            raise ValueError("fresh corpus changed")
    else:
        with path.open("x") as f:
            f.write(suffix)
    report = {"source_sha256": sha256(original), "fresh_sha256": sha256(path), "cut_character": cut,
        "excluded_common_prefix_tokens": common, "prior_input_upper_bound": 32768,
        "fresh_tokens": len(fresh), "tokenizer_sha256": sha256(TOKENIZER),
        "full_ids_sha256": hashlib.sha256(np.asarray(full,dtype="<i4").tobytes()).hexdigest()}
    save_json(root/"freshness.json", report)
    return path


def capture(root, split, sampling, seed=0):
    name = f"endpoint-{split}" if sampling == "endpoints" else f"{sampling}{seed}-{split}"
    destination = root/"captures"/(name+".acal")
    metadata = destination.with_suffix(".acal.json")
    if destination.exists() and metadata.exists():
        data = json.loads(metadata.read_text())
        if (data["sampling"] != sampling or data["sampling_seed"] != seed
                or data["file_sha256"] != sha256(destination) or data["source_sha256"] != sha256(SOURCE)):
            raise ValueError("capture identity mismatch")
        load_calibration(destination, SOURCE)
        return destination
    print(f"CAPTURE {name}", flush=True)
    command = [sys.executable,"-m","experiments.full22.capture","--split",split,
               "--sampling",sampling,"--seed",str(seed),"--out",str(destination)]
    with (root/(name+".capture.log")).open("x") as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return destination


def rescore(selection_path, root):
    selection = load_calibration(selection_path, SOURCE)
    reader = gguf.GGUFReader(SOURCE)
    tensors = {t.name:t for t in reader.tensors}
    rows = []
    for path in sorted((D01/"screen").glob("*.json")):
        r = json.loads(path.read_text())
        if "geometry" not in r:
            continue
        if r["status"] != "MEASURED" or sha256(path.with_suffix(".ar")) != r["artifact_sha256"]:
            raise ValueError("D01 screen record unavailable for re-scoring")
        name = r["tensor"]
        full = np.asarray(tensors[name].data,np.float32)
        indices = np.linspace(0,len(full)-1,min(256,len(full))).astype(np.int64)
        decoded = codec.decode(Record.loads(path.with_suffix(".ar").read_bytes()))
        rows.append({"tensor":name,"bits":r["bits"],"rank":r["rank"],"geometry":r["geometry"],
            "original_endpoint_nmse":r["selection"]["output_nmse"],
            "stratified_nmse":codec.distortion(full[indices],decoded,inputs_for(selection,name))["output_nmse"]})
    if len(rows) != 160:
        raise ValueError("incomplete cross-sampling diagnostic")
    save_json(root/"cross-sampling.json", {"records":160,"selection_sha256":sha256(selection_path),"results":rows})


def config_for(bits, geometry):
    return {"bits":bits} if geometry == "scalar" else {"bits":bits,"rank":1,"rank_geometry":geometry}


def qualifies(new, old, best_old, new_model, old_model):
    return (new.get("status") == old.get("status") == "MEASURED"
        and new["perplexity"] <= .95*old["perplexity"] and new["perplexity"] <= best_old
        and new_model["archive_bytes"] <= 1.05*old_model["archive_bytes"])


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    for name in ("captures","models","evaluations"):
        (root/name).mkdir(parents=True,exist_ok=True)
    runtime = json.loads((D01/"runtime.json").read_text())
    if sha256(BINARY) != runtime["executable"] or any(sha256(p)!=v for p,v in runtime["libraries"].items()):
        raise ValueError("inference runtime changed from endpoint controls")
    save_json(root/"runtime.json",runtime)
    runtime_hash = sha256(root/"runtime.json")
    files = [Path(__file__),PROTOCOL,ROOT/"scripts/calibration_sampling.py",ROOT/"scripts/capture_calibration.py",
        ROOT/"experiments/direction_models.py",ROOT/"experiments/basis_discovery.py"]
    files += sorted((ROOT/"experiments/full22").glob("*.py"))
    frozen = {"source_sha256":sha256(SOURCE),"fit_corpus_sha256":sha256(OUT/"fit.jsonl"),
        "selection_corpus_sha256":sha256(OUT/"selection_inputs.jsonl"),
        "inference_selection_sha256":sha256(OUT/"selection.txt"),
        "implementation":{str(p.relative_to(ROOT)):sha256(p) for p in files},"runtime_sha256":runtime_hash}
    manifest_path = root/"manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != frozen:
        raise ValueError("D02 provenance changed; preserve run and choose a new directory")
    save_json(manifest_path,frozen)
    fresh = fresh_holdout(root)
    old_fit = capture(root,"fit","endpoints")
    old_selection = capture(root,"selection","endpoints")
    replay = {"fit_identical":sha256(old_fit)==sha256(OUT/"fit.acal"),
              "selection_identical":sha256(old_selection)==sha256(OUT/"selection.acal")}
    save_json(root/"endpoint-replay.json",replay)
    fit = capture(root,"fit","stratified",0)
    selection = capture(root,"selection","stratified",0)
    rescore(selection,root)
    originals = json.loads((D01/"models.json").read_text())
    old_models, old_scores = {}, {}
    for name, model in originals.items():
        if model["status"] != "MEASURED" or sha256(model["decoded"]) != model["decoded_sha256"]:
            raise ValueError("endpoint model identity changed")
        if name.startswith("q") and not all(replay.values()):
            model = build(model["config"],"endpoint-"+name,root/"models",old_fit)
        old_models[name] = model
        if all(replay.values()):
            score = json.loads((D01/"evaluations"/f"selection-{name}.json").read_text())
            if (score["model_sha256"] != sha256(model["decoded"]) or score["corpus_sha256"] != sha256(OUT/"selection.txt")
                    or score["runtime_sha256"] != runtime_hash):
                raise ValueError("endpoint inference identity changed")
            old_scores[name] = dict(score,reused_from=str(D01/"evaluations"/f"selection-{name}.json"))
        else:
            old_scores[name] = evaluate(model,OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
    models, scores = {}, {}
    for bits in (3,4):
        for geometry in GEOMETRIES:
            key = f"q{bits}-{geometry}"
            name = "strat0-"+key
            models[name] = build(config_for(bits,geometry),name,root/"models",fit)
            scores[name] = evaluate(models[name],OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
            save_json(root/"progress.json",{"models":models,"scores":scores})
    chosen = {}
    for bits in (3,4):
        best_old = min(old_scores[f"q{bits}-{g}"]["perplexity"] for g in GEOMETRIES)
        passing = [g for g in GEOMETRIES if qualifies(scores[f"strat0-q{bits}-{g}"],old_scores[f"q{bits}-{g}"],best_old,
            models[f"strat0-q{bits}-{g}"],old_models[f"q{bits}-{g}"])]
        if passing:
            chosen[bits] = min(passing,key=lambda g:(scores[f"strat0-q{bits}-{g}"]["perplexity"],
                models[f"strat0-q{bits}-{g}"]["archive_bytes"],GEOMETRIES.index(g)))
    save_json(root/"seed0-selection.json",{"chosen":chosen,"models":models,"scores":scores,"endpoint_scores":old_scores})
    for seed in (1,2):
        if not chosen:
            break
        fit_seed = capture(root,"fit","stratified",seed)
        for bits,geometry in chosen.items():
            name = f"strat{seed}-q{bits}-{geometry}"
            models[name] = build(config_for(bits,geometry),name,root/"models",fit_seed)
            scores[name] = evaluate(models[name],OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
            save_json(root/"progress.json",{"models":models,"scores":scores})
    surviving = {}
    for bits,geometry in chosen.items():
        best_old = min(old_scores[f"q{bits}-{g}"]["perplexity"] for g in GEOMETRIES)
        if all(qualifies(scores[f"strat{s}-q{bits}-{geometry}"],old_scores[f"q{bits}-{geometry}"],best_old,
            models[f"strat{s}-q{bits}-{geometry}"],old_models[f"q{bits}-{geometry}"]) for s in (0,1,2)):
            surviving[bits] = geometry
    final = {}
    if surviving:
        final_models = {k:v for k,v in old_models.items() if k in ("source-f32","historical-budget120")
            or any(k.startswith(f"q{b}-") for b in surviving)}
        for bits,g in surviving.items():
            final_models.update({f"strat{s}-q{bits}-{g}":models[f"strat{s}-q{bits}-{g}"] for s in (0,1,2)})
        final = {k:evaluate(v,fresh,"fresh-final",root/"evaluations",runtime_hash) for k,v in final_models.items()}
    acceptance = []
    for bits,g in surviving.items():
        best_old = min(final[f"q{bits}-{v}"]["perplexity"] for v in GEOMETRIES)
        passed = all(qualifies(final[f"strat{s}-q{bits}-{g}"],final[f"q{bits}-{g}"],best_old,
            models[f"strat{s}-q{bits}-{g}"],old_models[f"q{bits}-{g}"]) for s in (0,1,2))
        acceptance.append({"bits":bits,"geometry":g,"status":"MEASURED" if passed else "FAILED"})
    freshness = json.loads((root/"freshness.json").read_text())
    if (any(sha256(p)!=h for p,h in frozen["implementation"].items()) or sha256(SOURCE)!=frozen["source_sha256"]
            or sha256(OUT/"fit.jsonl")!=frozen["fit_corpus_sha256"]
            or sha256(OUT/"selection_inputs.jsonl")!=frozen["selection_corpus_sha256"]
            or sha256(OUT/"selection.txt")!=frozen["inference_selection_sha256"]
            or sha256(fresh)!=freshness["fresh_sha256"]
            or sha256(BINARY)!=runtime["executable"] or any(sha256(p)!=h for p,h in runtime["libraries"].items())):
        raise ValueError("D02 implementation, source, corpus or runtime changed during execution")
    result = {"experiment":"D02","endpoint_replay":replay,"seed0_chosen":chosen,"repeatable_selection":surviving,
        "final_acceptance":acceptance,"models":models,"endpoint_models":old_models,"selection":scores,
        "endpoint_selection":old_scores,"fresh_final":final,"defaults_changed":False,"dense_decoded_inference_only":True}
    save_json(root/"summary.json",result)
    print(json.dumps({k:result[k] for k in ("experiment","endpoint_replay","seed0_chosen","repeatable_selection","final_acceptance")}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    args = parser.parse_args()
    run(args.out)
