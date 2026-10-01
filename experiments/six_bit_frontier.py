"""D05: globally select a native-admitted six-bit quality/size candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path

from threadpoolctl import threadpool_limits

from .adaptive_fidelity import ASSESSOR,FIT,SELECT,FRESH_HASH,fresh
from .basis_discovery import save_json
from .direction_models import BINARY,evaluate
from .full22 import native,precision
from .full22.data import ROOT,OUT,SOURCE,sha256

D04=ROOT/"artifacts/adaptive-fidelity-2026-10-01"
N1=ROOT/"artifacts/native-admission-2026-10-01"
PROTOCOL=ROOT/"research/six-bit-frontier-2026-10-01/PROTOCOL.md"
ORDER=("scalar","diagonal","activation")


def build_candidate(root,geometry):
    try:
        result=precision.run(SOURCE,FIT[1],None if geometry=="scalar" else FIT[0],FIT,ASSESSOR,root/geometry,
            start_bits=6,rank_geometry=None if geometry=="scalar" else geometry)
        result["model"]["name"]="six-"+geometry
        return geometry,result
    except Exception as error:
        result={"status":"FAILED","error":repr(error)}
        save_json(root/(geometry+"-failure.json"),result)
        return geometry,result


def select(models,scores,native_ok):
    required=("source-f32","historical-budget120",*("six-"+g for g in ORDER))
    if any(n not in models or scores.get(n,{}).get("status")!="MEASURED" for n in required):
        return {"status":"INCONCLUSIVE","selected":None,"candidates":[]}
    history=models["historical-budget120"]["archive_bytes"]
    reference=scores["historical-budget120"]["perplexity"]
    choices=[]
    for geometry in ORDER:
        name="six-"+geometry
        checks={"native_fit_and_selection":native_ok.get(name,False),
            "stored_bytes":models[name]["archive_bytes"]<=.9*history,
            "perplexity":scores[name]["perplexity"]<=.97*reference}
        choices.append({"name":name,"status":"QUALIFIED" if all(checks.values()) else "FAILED","checks":checks})
    eligible=[r["name"] for r in choices if r["status"]=="QUALIFIED"]
    chosen=min(eligible,key=lambda n:(scores[n]["perplexity"],models[n]["archive_bytes"],ORDER.index(n.removeprefix("six-")))) if eligible else None
    return {"status":"SELECTED" if chosen else "FAILED","selected":chosen,"candidates":choices}


def final_gate(selected,models,scores):
    if selected is None: return {"status":"UNRUN"}
    if len(scores)!=5 or any(s["status"]!="MEASURED" for s in scores.values()):
        return {"status":"INCONCLUSIVE"}
    history="historical-budget120"
    size=models[selected]["archive_bytes"]<=.9*models[history]["archive_bytes"]
    quality=scores[selected]["perplexity"]<=.97*scores[history]["perplexity"]
    return {"status":"QUALIFIED" if size and quality else "FAILED","selected":selected,
        "size_passed":size,"ppl_passed":quality,"selected_ppl":scores[selected]["perplexity"],
        "historical_ppl":scores[history]["perplexity"],"selected_bytes":models[selected]["archive_bytes"]}


def run(root):
    threadpool_limits(4)
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    for name in ("native-selection","evaluations"): (root/name).mkdir()
    prior=json.loads((D04/"summary.json").read_text())
    if prior["final_gate"]["status"]!="UNRUN" or list((D04/"evaluations").glob("new-final-*")):
        raise ValueError("D04 final segment was consumed")
    pinned=json.loads((N1/"manifest.json").read_text())["sha256"]
    if any(sha256(p)!=pinned[str(p)] for p in [SOURCE,*FIT,*SELECT,ASSESSOR]):
        raise ValueError("source, observations or native assessor changed")
    runtime=json.loads((D04/"runtime.json").read_text())
    if sha256(BINARY)!=runtime["executable"] or any(sha256(p)!=h for p,h in runtime["libraries"].items()):
        raise ValueError("inference runtime changed")
    save_json(root/"runtime.json",runtime); runtime_hash=sha256(root/"runtime.json")
    files=[SOURCE,*FIT,*SELECT,ASSESSOR,BINARY,PROTOCOL,Path(__file__),OUT/"selection.txt",OUT/"final_test.txt"]
    files += [Path(p) for p in runtime["libraries"]]
    files += [ROOT/"experiments"/p for p in ("adaptive_fidelity.py","direction_models.py","calibration_coverage.py","basis_discovery.py")]
    files += list((ROOT/"experiments/full22").glob("*.py"))
    frozen={str(p):sha256(p) for p in files}; save_json(root/"manifest.json",{"sha256":frozen,"candidate_order":ORDER})
    holdout=fresh(root)
    models,scores={},{}
    for name in ("source-f32","historical-budget120"):
        model=prior["models"][name]; score=prior["selection"][name]
        path=Path(score["reused_from"])
        if (sha256(model["decoded"])!=model["decoded_sha256"] or score["model_sha256"]!=model["decoded_sha256"]
                or score["runtime_sha256"]!=runtime_hash or score["corpus_sha256"]!=sha256(OUT/"selection.txt")
                or score["log_sha256"]!=sha256(path.with_suffix(".log")) or score["status"]!="MEASURED"):
            raise ValueError("reference inference identity differs")
        models[name],scores[name]=model,score
    history=json.loads((ROOT/"research/foundations-2026-09-05/results.json").read_text())
    historical_archive=Path(next(x["artifact"] for x in history["results"] if x["name"]=="budget120"))
    if (sha256(historical_archive)!=models["historical-budget120"]["archive_sha256"]
            or historical_archive.stat().st_size!=models["historical-budget120"]["archive_bytes"]):
        raise ValueError("historical archive identity differs")
    frozen[str(historical_archive)]=sha256(historical_archive)
    save_json(root/"manifest.json",{"sha256":frozen,"candidate_order":ORDER})
    save_json(root/"references.json",{"models":models,"scores":scores})
    builds={}
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(build_candidate,root,g) for g in ORDER]):
            geometry,result=future.result(); builds[geometry]=result
            if result["status"]=="FIT_ADMITTED": models["six-"+geometry]=result["model"]
            save_json(root/"build-progress.json",builds)
    reports,native_ok={},{}
    for geometry in ORDER:
        name="six-"+geometry
        if name not in models: continue
        reports[name]={}
        for index,path in enumerate(SELECT):
            reports[name][str(index)]=native.assess(ASSESSOR,SOURCE,models[name]["decoded"],path,
                root/"native-selection"/f"{name}-{index}.json")
        native_ok[name]=all(x["report"]["all_passed"] for x in reports[name].values())
        scores[name]=evaluate(models[name],OUT/"selection.txt","selection",root/"evaluations",runtime_hash)
        save_json(root/"selection-progress.json",{"models":models,"scores":scores,"native":reports})
    choice=select(models,scores,native_ok)
    save_json(root/"selection.json",choice)
    final={}
    if choice["selected"] is not None:
        save_json(root/"final-use.json",{"status":"STARTED","corpus_sha256":FRESH_HASH,"selected":choice["selected"],"models":sorted(models)})
        for name in ("source-f32","historical-budget120",*("six-"+g for g in ORDER)):
            final[name]=evaluate(models[name],holdout,"new-final",root/"evaluations",runtime_hash)
            save_json(root/"final-progress.json",final)
        save_json(root/"final-use.json",{"status":"COMPLETED","corpus_sha256":FRESH_HASH,"selected":choice["selected"],"models":sorted(models)})
    result={"experiment":"D05","models":models,"builds":builds,"native_selection":reports,"selection_scores":scores,
        "selection":choice,"final_scores":final,"final_gate":final_gate(choice["selected"],models,final),
        "new_inference_runs":sum("reused_from" not in v for v in scores.values())+len(final),
        "defaults_changed":False,"main_OQ03_integration":False}
    if any(sha256(p)!=h for p,h in frozen.items()) or sha256(holdout)!=FRESH_HASH:
        raise ValueError("D05 frozen state changed")
    save_json(root/"summary.json",result)
    print(json.dumps({"selection":choice,"final_gate":result["final_gate"],"new_inference_runs":result["new_inference_runs"]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
