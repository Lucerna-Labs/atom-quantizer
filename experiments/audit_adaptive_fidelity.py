"""Independent D04 stage, native-decision, artifact and inference audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .decode_reproducibility import capture_failure
from .full22 import codec,native,archive as archive_module
from .full22.data import ROOT,OUT,SOURCE,sha256
from .full22.records import Record


def native_receipt(path, model, calibration):
    r=json.loads(Path(path).read_text()); report=r["report"]; command=r["command"]
    if sha256(command[0])!=r["binary_sha256"] or sha256(Path(path).with_suffix(".stdout"))!=r["stdout_sha256"]:
        raise ValueError("native executable/stdout identity differs")
    if sha256(Path(path).with_suffix(".stderr"))!=r["stderr_sha256"] or json.loads(Path(path).with_suffix(".stdout").read_text())!=report:
        raise ValueError("native report/log differs")
    if report["candidate_sha256"]!=model["decoded_sha256"]:
        raise ValueError("assessment does not belong to the built model")
    if report["calibration_sha256"]!=sha256(calibration):
        raise ValueError("assessment used the wrong fit/selection view")
    native.validate(report,command[2],command[3],command[5],r["returncode"])
    failures=set()
    for t in report["tensors"]:
        protected=any(k in t["name"].lower() for k in ("token_embd","output.weight","lm_head","norm","bias"))
        kl=.25 if protected else 1.; cos=float(np.float32(.995 if protected else .99)); mse=.005 if protected else .02
        passed=(t["kl_forward"] is not None and t["kl_reverse"] is not None and max(t["kl_forward"],t["kl_reverse"])<=kl
            and t["cosine"] is not None and t["cosine"]>=cos
            and (t["observation"]=="histogram" or (t["output_mse"] is not None and t["output_mse"]<=mse)))
        if passed!=t["passed"]: raise ValueError("native gate differs from independently computed decision")
        if not passed: failures.add(t["name"])
    return failures


def artifact(model, output):
    if (sha256(model["archive"])!=model["archive_sha256"] or Path(model["archive"]).stat().st_size!=model["archive_bytes"]
            or sha256(model["decoded"])!=model["decoded_sha256"]):
        raise ValueError("model identity differs")
    tensors={t.name:t for t in gguf.GGUFReader(model["decoded"]).tensors}
    with zipfile.ZipFile(model["archive"]) as z:
        manifest=json.loads(z.read("manifest.json"))
        archive_module.validate_manifest(z.read("gguf-prefix.bin"),manifest)
        if manifest["format"]!="A22-2" or len(manifest["tensors"])!=272 or len(tensors)!=272:
            raise ValueError("incomplete committed model")
        for entry in manifest["tensors"]:
            data=z.read(entry["entry"])
            if len(data)!=entry["bytes"] or hashlib.sha256(data).hexdigest()!=entry["sha256"]:
                raise ValueError("record identity differs")
            record=Record.loads(data); decoded=codec.decode(record)
            expected=np.asarray(tensors[entry["name"]].data,dtype=np.float32)
            if decoded.shape!=expected.shape or record.meta["shape"]!=entry["shape"]:
                raise ValueError("record shape differs from exported tensor")
            if decoded.tobytes()!=expected.tobytes():
                capture_failure(output,{"archive":model["archive"],"tensor":entry["name"]},record,expected,decoded)
                raise ValueError("decoded commitment mismatch captured")
            if hashlib.sha256(decoded.tobytes()).hexdigest()!=entry["decoded_sha256"]:
                raise ValueError("decoded commitment differs")
        return manifest


def run(root, destination=None):
    threadpool_limits(4)
    root=Path(root).resolve(); output=Path(destination).resolve() if destination else root/"audit-data"; output.mkdir(exist_ok=False)
    result=json.loads((root/"summary.json").read_text())
    frozen=json.loads((root/"manifest.json").read_text())["sha256"]
    if any(sha256(p)!=h for p,h in frozen.items()): raise ValueError("frozen D04 input changed")
    stage_rows=[]; total_tensors=0; native_reports=0
    fit=(OUT/"fit.acal",ROOT/"artifacts/calibration-coverage-2026-10-01/captures/stratified0-fit.acal")
    selection_cal=(OUT/"selection.acal",ROOT/"artifacts/calibration-coverage-2026-10-01/captures/stratified0-selection.acal")
    initial={t.name:0 if t.data.ndim==2 else 3 for t in gguf.GGUFReader(SOURCE).tensors}
    for name,adaptive in sorted(result["adaptive"].items()):
        if adaptive["status"]!="FIT_ADMITTED": raise ValueError("adaptive build did not complete")
        previous=None; previous_failed=set()
        if not 1<=len(adaptive["stages"])<=4: raise ValueError("invalid stage count")
        for index,stage in enumerate(adaptive["stages"]):
            model=stage["model"]; manifest=artifact(model,output); total_tensors+=272
            levels=stage["levels"]
            if stage["iteration"]!=index or (index==0 and levels!=initial):
                raise ValueError("initial precision or stage order differs")
            if (manifest["source_sha256"]!=sha256(SOURCE) or manifest["calibration_sha256"]!=sha256(fit[1])
                    or manifest.get("residual_calibration_sha256")!=(sha256(fit[0]) if name=="adaptive-activation" else None)):
                raise ValueError("stage calibration roles differ")
            for entry in manifest["tensors"]:
                level=levels[entry["name"]]; choice=(4,6,8,"exact")[level]
                config={"family":"lossless"} if choice=="exact" else {"bits":choice,**({"rank":2,"rank_geometry":"activation"} if name=="adaptive-activation" else {})}
                if entry["config"]!=config: raise ValueError("stored configuration differs from declared ladder")
            failed=set()
            if len(stage["native_reports"])!=2: raise ValueError("missing fitting view")
            for view,path in enumerate(stage["native_reports"]):
                failed.update(native_receipt(path,model,fit[view])); native_reports+=1
            if failed!=set(stage["failed_tensors"]): raise ValueError("fit-view failure union differs")
            unchanged=0
            if previous is not None:
                old_levels=previous["levels"]
                if levels!={n:v+int(n in previous_failed) for n,v in old_levels.items()}:
                    raise ValueError("precision was changed outside the fixed fitting rule")
                with zipfile.ZipFile(previous["model"]["archive"]) as a,zipfile.ZipFile(model["archive"]) as b:
                    old={e["name"]:e for e in json.loads(a.read("manifest.json"))["tensors"]}
                    for entry in manifest["tensors"]:
                        if entry["name"] not in previous_failed:
                            if not entry.get("reused") or a.read(old[entry["name"]]["entry"])!=b.read(entry["entry"]):
                                raise ValueError("passing tensor was changed or refitted")
                            unchanged+=1
            stage_rows.append({"model":name,"stage":stage["iteration"],"failed_tensors":len(failed),
                "unchanged_records":unchanged,"bytes":model["archive_bytes"]})
            previous,previous_failed=stage,failed
        if previous_failed: raise ValueError("final model still fails native fitting gates")
        final=adaptive["model"]
        if sha256(final["archive"])!=previous["model"]["archive_sha256"] or sha256(final["decoded"])!=previous["model"]["decoded_sha256"]:
            raise ValueError("publication differs from the last passing stage")
    uniform=artifact(result["models"]["uniform6-activation"],output); total_tensors+=272
    if uniform["calibration_sha256"]!=sha256(fit[1]) or uniform.get("residual_calibration_sha256")!=sha256(fit[0]):
        raise ValueError("uniform control used different calibration roles")
    for entry in uniform["tensors"]:
        expected={"family":"lossless"} if initial[entry["name"]]==3 else {"bits":6,"rank":2,"rank_geometry":"activation"}
        if entry["config"]!=expected: raise ValueError("uniform control differs from the precommit")
    principal_native=True
    for name,reports in result["native"].items():
        keys={"selection-0","selection-1"}|({"fit-0","fit-1"} if name=="uniform6-activation" else set())
        if set(reports)!=keys: raise ValueError("missing final native control views")
        for key,report in reports.items():
            path=root/"native-selection"/f"{name}-{key}.json"
            cal=(fit if key.startswith("fit-") else selection_cal)[int(key[-1])]
            failed=native_receipt(path,result["models"][name],cal); native_reports+=1
            if name=="adaptive-activation": principal_native &= not failed
    inference=[]
    for split,corpus_name in (("selection","selection"),("final","new-final")):
        for name,score in result[split].items():
            path=Path(score["reused_from"]) if "reused_from" in score else root/"evaluations"/f"{corpus_name}-{name}.json"
            saved=json.loads(path.read_text()); log=path.with_suffix(".log")
            fields=("command","model_sha256","corpus_sha256","runtime_sha256","log_sha256","perplexity","standard_error","returncode","chunks","status")
            if any(saved[k]!=score[k] for k in fields) or score["returncode"]!=0 or score["chunks"]!=64:
                raise ValueError("inference receipt is incomplete or differs from summary")
            if score["command"][5:]!=["--chunks","64","-c","512","-b","512","-ub","512","-t","4","-tb","4","-ngl","0"]:
                raise ValueError("inference options differ from the protocol")
            if (score["status"]!="MEASURED" or score["model_sha256"]!=result["models"][name]["decoded_sha256"]
                    or score["log_sha256"]!=sha256(log) or score["corpus_sha256"]!=sha256(score["command"][4])
                    or score["runtime_sha256"]!=sha256(root/"runtime.json") or saved["perplexity"]!=score["perplexity"]):
                raise ValueError("inference receipt identity differs")
            text=log.read_text(); match=re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)",text)
            chunks=re.search(r"calculating perplexity over (\d+) chunks",text)
            if not match or not chunks or int(chunks[1])!=64 or float(match[1])!=score["perplexity"]:
                raise ValueError("inference log differs")
            inference.append({"model":name,"split":split,"ppl":score["perplexity"],"reused":"reused_from" in score})
    def qualified(scores):
        models=result["models"]; p=models["adaptive-activation"]["archive_bytes"]
        return (principal_native and p<=.9*models["historical-budget120"]["archive_bytes"]
            and p<=1.05*models["adaptive-scalar"]["archive_bytes"]
            and scores["adaptive-activation"]["perplexity"]<=min(scores[n]["perplexity"] for n in ("historical-budget120","adaptive-scalar")))
    selected=qualified(result["selection"])
    if result["selection_gate"]["status"]!=("QUALIFIED" if selected else "FAILED"):
        raise ValueError("selection gate differs from protocol")
    if selected:
        if len(result["final"])!=6 or result["final_gate"]["status"]!=("QUALIFIED" if qualified(result["final"]) else "FAILED"):
            raise ValueError("conditional final gate is incomplete or differs")
    elif result["final"] or result["final_gate"]["status"]!="UNRUN":
        raise ValueError("conditional final stage ran without admission")
    if not selected and (list((root/"evaluations").glob("new-final-*")) or (root/"final-use.json").exists()):
        raise ValueError("unadmitted final evaluation evidence exists")
    report={"status":"PASSED","stages":stage_rows,"verified_tensors":total_tensors,"native_reports":native_reports,
        "inference":inference,"selection_gate":result["selection_gate"],"final_gate":result["final_gate"],
        "source_sha256":sha256(Path(__file__)),"summary_sha256":sha256(root/"summary.json")}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="inference"}),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run",required=True)
    parser.add_argument("--out")
    args=parser.parse_args(); run(args.run,args.out)
