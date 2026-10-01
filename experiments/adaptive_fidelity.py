"""D04: precommitted native-gated adaptive precision and real inference trial."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .calibration_coverage import native_ids,TOKENIZER
from .direction_models import BINARY,evaluate
from .full22 import native,precision
from .full22.data import ROOT,OUT,SOURCE,sha256

D01 = ROOT/"artifacts/direction-retention-2026-09-30"
D02 = ROOT/"artifacts/calibration-coverage-2026-10-01"
D03 = ROOT/"artifacts/calibration-roles-2026-10-01"
R1 = ROOT/"artifacts/decoder-reproducibility-2026-10-01/models-check"
N1 = ROOT/"artifacts/native-admission-2026-10-01"
ASSESSOR = ROOT/"target/native-admission-current/release/atom-quantizer"
PROTOCOL = ROOT/"research/adaptive-fidelity-2026-10-01/PROTOCOL.md"
FIT = (OUT/"fit.acal",D02/"captures/stratified0-fit.acal")
SELECT = (OUT/"selection.acal",D02/"captures/stratified0-selection.acal")
FRESH_HASH = "d757b51672161dc9f32eef6a01a54a84839893869842153ade19fb13ab4ed6e3"


def fresh(root):
    original = OUT/"final_test.txt"
    text = original.read_text()
    cut = text.find("\n",2*len(text)//3)+1
    full,prefix = native_ids(text),native_ids(text[:cut])
    common = next((i for i,(a,b) in enumerate(zip(full,prefix)) if a!=b),min(len(full),len(prefix)))
    suffix = text[cut:]
    ids = native_ids(suffix.removesuffix("\n"))
    if (cut!=439133 or common!=100418 or common<84241 or len(ids)!=51831
            or hashlib.sha256(suffix.encode()).hexdigest()!=FRESH_HASH):
        raise ValueError("precommitted fresh segment did not reproduce")
    prior = []
    for path in (ROOT/"artifacts").glob("**/evaluations/*.json"):
        value = json.loads(path.read_text())
        if value.get("corpus_sha256")==FRESH_HASH: prior.append(str(path))
    if prior: raise ValueError(f"new final corpus already evaluated: {prior}")
    path = root/"new-final.txt"
    with path.open("x") as stream: stream.write(suffix)
    save_json(root/"freshness.json",{"source_sha256":sha256(original),"fresh_sha256":sha256(path),
        "cut_character":cut,"excluded_common_prefix_tokens":common,"fresh_tokens":len(ids),
        "minimum_required_prefix":84241,"previous_D03_start":50449,"previous_input_upper_bound":32768,
        "tokenizer_sha256":sha256(TOKENIZER),"prior_matching_evaluations":prior})
    return path


def ladder(root, correction):
    name = "adaptive-activation" if correction else "adaptive-scalar"
    try:
        result = precision.run(SOURCE,FIT[1],FIT[0] if correction else None,FIT,ASSESSOR,root/name)
        return name,result
    except Exception as error:
        result = {"status":"FAILED","error":repr(error)}
        save_json(root/(name+"-failure.json"),result)
        return name,result


def reused(root, runtime_hash):
    d01 = json.loads((D01/"models.json").read_text())
    r1 = json.loads((R1/"summary.json").read_text())
    specs = [("source-f32",d01["source-f32"],D01/"evaluations/selection-source-f32.json"),
        ("historical-budget120",d01["historical-budget120"],D01/"evaluations/selection-historical-budget120.json"),
        ("fixed4-activation",r1["models"]["q4-IE-r2"],D03/"evaluations/selection-q4-IE-r2.json")]
    models,scores = {},{}
    for name,model,path in specs:
        if sha256(model["decoded"])!=model["decoded_sha256"]: raise ValueError("reference decoded model changed")
        if "archive" in model and (sha256(model["archive"])!=model["archive_sha256"] or Path(model["archive"]).stat().st_size!=model["archive_bytes"]):
            raise ValueError("reference archive changed")
        if name=="historical-budget120":
            old=json.loads((ROOT/"research/foundations-2026-09-05/results.json").read_text())
            artifact=next(x["artifact"] for x in old["results"] if x["name"]=="budget120")
            if sha256(artifact)!=model["archive_sha256"] or Path(artifact).stat().st_size!=model["archive_bytes"]:
                raise ValueError("historical reference archive changed")
        score=json.loads(path.read_text())
        if (score["status"]!="MEASURED" or score["model_sha256"]!=model["decoded_sha256"]
                or score["corpus_sha256"]!=sha256(OUT/"selection.txt") or score["runtime_sha256"]!=runtime_hash
                or score["log_sha256"]!=sha256(path.with_suffix(".log")) or score["chunks"]!=64):
            raise ValueError("reference inference identity changed")
        models[name]=dict(model,name=name)
        scores[name]=dict(score,name=name,reused_from=str(path))
    save_json(root/"reused.json",{"models":models,"selection":scores})
    return models,scores


def gate(models, scores, native_ok):
    principal,scalar,historical = "adaptive-activation","adaptive-scalar","historical-budget120"
    required = (principal,scalar,historical,"source-f32","fixed4-activation","uniform6-activation")
    if any(n not in models or scores.get(n,{}).get("status")!="MEASURED" for n in required):
        return {"status":"INCONCLUSIVE"}
    p=models[principal]["archive_bytes"]; q=models[scalar]["archive_bytes"]; h=models[historical]["archive_bytes"]
    checks={"native_fit_and_selection":native_ok,"history_bytes":p<=.9*h,"scalar_bytes":p<=1.05*q,
        "history_ppl":scores[principal]["perplexity"]<=scores[historical]["perplexity"],
        "scalar_ppl":scores[principal]["perplexity"]<=scores[scalar]["perplexity"]}
    return {"status":"QUALIFIED" if all(checks.values()) else "FAILED","checks":checks,
        "principal_bytes":p,"scalar_bytes":q,"historical_bytes":h,
        "principal_ppl":scores[principal]["perplexity"],"scalar_ppl":scores[scalar]["perplexity"],
        "historical_ppl":scores[historical]["perplexity"]}


def run(root):
    threadpool_limits(4)
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    for name in ("native-selection","evaluations"): (root/name).mkdir()
    if sha256(ASSESSOR)!="628c4d10a9513379c6a37b14d617469bffd5333de81da8fffe642cf4324417bd":
        raise ValueError("precommitted native assessor changed")
    prior_inputs=json.loads((N1/"manifest.json").read_text())["sha256"]
    if any(sha256(p)!=prior_inputs[str(p)] for p in [SOURCE,*FIT,*SELECT]):
        raise ValueError("source or observation captures changed from N1")
    runtime=json.loads((D01/"runtime.json").read_text())
    if sha256(BINARY)!=runtime["executable"] or any(sha256(p)!=h for p,h in runtime["libraries"].items()):
        raise ValueError("inference runtime changed")
    save_json(root/"runtime.json",runtime); runtime_hash=sha256(root/"runtime.json")
    files=[SOURCE,*FIT,*SELECT,ASSESSOR,PROTOCOL,Path(__file__),OUT/"selection.txt",OUT/"final_test.txt",TOKENIZER,BINARY]
    files += [Path(p) for p in runtime["libraries"]]
    files += [ROOT/"experiments"/n for n in ("direction_models.py","calibration_coverage.py","basis_discovery.py")]
    files += sorted((ROOT/"experiments/full22").glob("*.py"))
    frozen={str(p):sha256(p) for p in files}; save_json(root/"manifest.json",{"sha256":frozen})
    holdout=fresh(root)
    models,scores=reused(root,runtime_hash)
    adaptive={}
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(ladder,root,c) for c in (True,False)]):
            name,result=future.result(); adaptive[name]=result
            if result["status"]=="FIT_ADMITTED": models[name]=result["model"]
            save_json(root/"adaptive-progress.json",adaptive)
    tensors=gguf.GGUFReader(SOURCE).tensors
    configs={t.name:precision.config_for(6 if t.data.ndim==2 else "exact",True) for t in tensors}
    uniform=precision.build_stage(SOURCE,FIT[1],FIT[0],configs,root/"uniform6",default_config=precision.config_for(6,True))
    models["uniform6-activation"]=dict(uniform,name="uniform6-activation")
    native_reports={}
    for name in ("adaptive-activation","adaptive-scalar","uniform6-activation"):
        if name not in models: continue
        native_reports[name]={}
        if name=="uniform6-activation":
            for index,cal in enumerate(FIT):
                native_reports[name][f"fit-{index}"]=native.assess(ASSESSOR,SOURCE,models[name]["decoded"],cal,
                    root/"native-selection"/f"{name}-fit-{index}.json")
        for index,cal in enumerate(SELECT):
            native_reports[name][f"selection-{index}"]=native.assess(ASSESSOR,SOURCE,models[name]["decoded"],cal,
                root/"native-selection"/f"{name}-selection-{index}.json")
        scores[name]=evaluate(models[name],OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
        save_json(root/"selection-progress.json",{"models":models,"native":native_reports,"scores":scores})
    native_ok=(adaptive.get("adaptive-activation",{}).get("status")=="FIT_ADMITTED"
        and all(r["report"]["all_passed"] for r in native_reports.get("adaptive-activation",{}).values())
        and len(native_reports.get("adaptive-activation",{}))==2)
    selection=gate(models,scores,native_ok)
    save_json(root/"selection-gate.json",selection)
    final={}; final_gate={"status":"UNRUN"}
    if selection["status"]=="QUALIFIED":
        save_json(root/"final-use.json",{"status":"STARTED","corpus_sha256":FRESH_HASH,"models":sorted(models)})
        for name,model in models.items():
            final[name]=evaluate(model,holdout,"new-final",root/"evaluations",runtime_hash)
            save_json(root/"final-progress.json",final)
        final_gate=gate(models,final,native_ok)
        save_json(root/"final-use.json",{"status":"COMPLETED","corpus_sha256":FRESH_HASH,"models":sorted(models)})
    if any(sha256(p)!=h for p,h in frozen.items()) or sha256(holdout)!=FRESH_HASH:
        raise ValueError("D04 inputs, implementation or runtime changed")
    result={"experiment":"D04","models":models,"adaptive":adaptive,"native":native_reports,
        "selection":scores,"selection_gate":selection,"final":final,"final_gate":final_gate,
        "new_inference_runs":sum("reused_from" not in x for x in scores.values())+len(final),
        "defaults_changed":False,"main_OQ03_integration":False}
    save_json(root/"summary.json",result)
    print(json.dumps({"selection_gate":selection,"final_gate":final_gate,"new_inference_runs":result["new_inference_runs"]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
