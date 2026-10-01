"""Independent D05 global choice, native admission and artifact audit."""
import argparse
import json
from pathlib import Path
import re
import zipfile

import gguf
from threadpoolctl import threadpool_limits

from .adaptive_fidelity import FIT,SELECT,FRESH_HASH
from .audit_adaptive_fidelity import artifact,native_receipt
from .basis_discovery import save_json
from .full22.data import SOURCE,sha256

ORDER=("scalar","diagonal","activation")


def run(root):
    threadpool_limits(4)
    root=Path(root).resolve(); output=root/"audit-data"; output.mkdir(exist_ok=False)
    result=json.loads((root/"summary.json").read_text())
    frozen=json.loads((root/"manifest.json").read_text())["sha256"]
    if any(sha256(p)!=h for p,h in frozen.items()): raise ValueError("frozen source/input differs")
    initial={t.name:1 if t.data.ndim==2 else 3 for t in gguf.GGUFReader(SOURCE).tensors}
    stages=[]; native_count=0; tensor_count=0; eligible={}
    for geometry in ORDER:
        build=result["builds"][geometry]; name="six-"+geometry
        if build["status"]!="FIT_ADMITTED" or not 1<=len(build["stages"])<=3:
            raise ValueError("candidate did not complete bounded native fitting")
        previous=None; failures=set()
        for index,stage in enumerate(build["stages"]):
            model=stage["model"]; manifest=artifact(model,output); tensor_count+=272
            if stage["iteration"]!=index or (index==0 and stage["levels"]!=initial):
                raise ValueError("incorrect starting precision or stage order")
            if (manifest["source_sha256"]!=sha256(SOURCE) or manifest["calibration_sha256"]!=sha256(FIT[1])
                    or manifest.get("residual_calibration_sha256")!=(None if geometry=="scalar" else sha256(FIT[0]))):
                raise ValueError("candidate fitting roles differ")
            for entry in manifest["tensors"]:
                level=(4,6,8,"exact")[stage["levels"][entry["name"]]]
                config={"family":"lossless"} if level=="exact" else {"bits":level,**({} if geometry=="scalar" else {"rank":2,"rank_geometry":geometry})}
                if entry["config"]!=config: raise ValueError("stored global geometry or tier differs")
            unchanged=0
            if previous is not None:
                if stage["levels"]!={n:v+int(n in failures) for n,v in previous["levels"].items()}:
                    raise ValueError("escalation used something other than fitting failures")
                with zipfile.ZipFile(previous["model"]["archive"]) as old,zipfile.ZipFile(model["archive"]) as new:
                    before={t["name"]:t for t in json.loads(old.read("manifest.json"))["tensors"]}
                    for entry in manifest["tensors"]:
                        if entry["name"] not in failures:
                            if not entry["reused"] or old.read(before[entry["name"]]["entry"])!=new.read(entry["entry"]):
                                raise ValueError("passing record changed")
                            unchanged+=1
            failures=set()
            if len(stage["native_reports"])!=2: raise ValueError("missing fitting view")
            for view,path in enumerate(stage["native_reports"]):
                failures.update(native_receipt(path,model,FIT[view])); native_count+=1
            if failures!=set(stage["failed_tensors"]): raise ValueError("failure union differs")
            stages.append({"name":name,"stage":index,"failures":len(failures),"unchanged_records":unchanged,"bytes":model["archive_bytes"]})
            previous=stage
        if failures: raise ValueError("final candidate fails fitting gates")
        model=result["models"][name]
        if sha256(model["archive"])!=previous["model"]["archive_sha256"] or sha256(model["decoded"])!=previous["model"]["decoded_sha256"]:
            raise ValueError("published model differs from fitting result")
        native_ok=True
        for view in (0,1):
            native_ok &= not native_receipt(root/"native-selection"/f"{name}-{view}.json",model,SELECT[view]); native_count+=1
        history=result["models"]["historical-budget120"]["archive_bytes"]
        eligible[name]=(native_ok and model["archive_bytes"]<=.9*history
            and result["selection_scores"][name]["perplexity"]<=.97*result["selection_scores"]["historical-budget120"]["perplexity"])
    inference=[]
    for split,corpus in (("selection_scores","selection"),("final_scores","new-final")):
        for name,score in result[split].items():
            path=Path(score["reused_from"]) if "reused_from" in score else root/"evaluations"/f"{corpus}-{name}.json"
            saved=json.loads(path.read_text()); log=path.with_suffix(".log")
            keys=("status","command","model_sha256","corpus_sha256","runtime_sha256","log_sha256","chunks","returncode","perplexity","standard_error")
            if any(saved[k]!=score[k] for k in keys) or score["returncode"]!=0 or score["status"]!="MEASURED" or score["chunks"]!=64:
                raise ValueError("inference receipt incomplete or changed")
            if (sha256(result["models"][name]["decoded"])!=score["model_sha256"]
                    or sha256(score["command"][4])!=score["corpus_sha256"] or sha256(log)!=score["log_sha256"]
                    or sha256(root/"runtime.json")!=score["runtime_sha256"]): raise ValueError("inference identity differs")
            if score["command"][5:]!=["--chunks","64","-c","512","-b","512","-ub","512","-t","4","-tb","4","-ngl","0"]:
                raise ValueError("inference options changed")
            text=log.read_text(); match=re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)",text)
            chunks=re.search(r"calculating perplexity over (\d+) chunks",text)
            if not match or not chunks or int(chunks[1])!=64 or float(match[1])!=score["perplexity"] or float(match[2])!=score["standard_error"]:
                raise ValueError("inference log disagrees")
            inference.append({"name":name,"split":split,"ppl":score["perplexity"],"reused":"reused_from" in score})
    candidates=[n for n,ok in eligible.items() if ok]
    selected=min(candidates,key=lambda n:(result["selection_scores"][n]["perplexity"],result["models"][n]["archive_bytes"],ORDER.index(n.removeprefix("six-")))) if candidates else None
    if selected!=result["selection"]["selected"]: raise ValueError("global selection differs from frozen rule")
    for row in result["selection"]["candidates"]:
        if (row["status"]=="QUALIFIED")!=eligible[row["name"]]: raise ValueError("eligibility report differs")
    if selected is None:
        if result["final_scores"] or result["final_gate"]["status"]!="UNRUN": raise ValueError("unadmitted final execution")
    else:
        final=result["final_scores"]
        if len(final)!=5 or sha256(root/"new-final.txt")!=FRESH_HASH: raise ValueError("final scope differs")
        passed=(final[selected]["perplexity"]<=.97*final["historical-budget120"]["perplexity"]
            and result["models"][selected]["archive_bytes"]<=.9*result["models"]["historical-budget120"]["archive_bytes"])
        if result["final_gate"]["selected"]!=selected or result["final_gate"]["status"]!=("QUALIFIED" if passed else "FAILED"):
            raise ValueError("final was reselected or judged by different conditions")
    report={"status":"PASSED","stages":stages,"tensors":tensor_count,"native_reports":native_count,
        "selection":result["selection"],"final_gate":result["final_gate"],"inference":inference,
        "auditor_sha256":sha256(Path(__file__)),"summary_sha256":sha256(root/"summary.json")}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="inference"}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run",required=True)
    run(parser.parse_args().run)
