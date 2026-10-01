"""N1: run and audit the complete native fidelity matrix without fitting weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import gguf

from .basis_discovery import save_json
from .full22.data import ROOT, OUT, SOURCE, sha256

D03 = ROOT/"artifacts/calibration-roles-2026-10-01"
R1 = ROOT/"artifacts/decoder-reproducibility-2026-10-01/models-check"
D02 = ROOT/"artifacts/calibration-coverage-2026-10-01"
BINARY = ROOT/"target/native-admission-current/release/atom-quantizer"
CALIBRATIONS = {
    "endpoint-fit":OUT/"fit.acal",
    "endpoint-selection":OUT/"selection.acal",
    "interior-fit":D02/"captures/stratified0-fit.acal",
    "interior-selection":D02/"captures/stratified0-selection.acal",
}


def execute(root, name, model, regime, calibration):
    stem = root/"evaluations"/f"{name}-{regime}"
    command = [str(BINARY),"assess",str(SOURCE),model["decoded"],"--calibration",str(calibration)]
    started = time.perf_counter()
    try:
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=1800)
        stem.with_suffix(".stdout").write_text(result.stdout)
        stem.with_suffix(".stderr").write_text(result.stderr)
        if result.returncode not in (0,2):
            raise ValueError(f"assessment execution failed with exit {result.returncode}")
        report = json.loads(result.stdout)
        if (report["format"]!="native-fidelity-1" or report["backend"]!="cpu"
                or report["tensor_count"]!=272 or len(report["tensors"])!=272
                or report["max_output_mse"]!=.02
                or result.returncode!=(0 if report["all_passed"] else 2)
                or report["source_sha256"]!=sha256(SOURCE)
                or report["candidate_sha256"]!=model["decoded_sha256"]
                or report["calibration_sha256"]!=sha256(calibration)):
            raise ValueError("incomplete or mismatched native assessment")
        record = {"status":"MEASURED","model":name,"regime":regime,"command":command,
            "returncode":result.returncode,"seconds":time.perf_counter()-started,"report":report,
            "stdout_sha256":sha256(stem.with_suffix(".stdout")),"stderr_sha256":sha256(stem.with_suffix(".stderr"))}
    except Exception as error:
        record = {"status":"FAILED","model":name,"regime":regime,"command":command,
            "seconds":time.perf_counter()-started,"error":repr(error)}
    save_json(stem.with_suffix(".json"),record)
    print(json.dumps({"model":name,"regime":regime,"status":record["status"],
        "failures":record.get("report",{}).get("gate_failures")}),flush=True)
    return record


def audit(records, models, root):
    source_tensors = gguf.GGUFReader(SOURCE).tensors
    expected = {t.name:{"dimensions":list(t.shape),"kind":"histogram" if len(t.data.shape)==1
        else "embedding_lookup" if t.name=="token_embd.weight" else "linear"} for t in source_tensors}
    if len(records)!=16 or len({(r["model"],r["regime"]) for r in records})!=16:
        raise ValueError("incomplete native matrix")
    checks = []
    for record in records:
        if record["status"]!="MEASURED":
            raise ValueError("native matrix includes an execution failure")
        report = record["report"]
        stem = root/"evaluations"/f"{record['model']}-{record['regime']}"
        if (json.loads(stem.with_suffix(".stdout").read_text())!=report
                or sha256(stem.with_suffix(".stdout"))!=record["stdout_sha256"]
                or sha256(stem.with_suffix(".stderr"))!=record["stderr_sha256"]):
            raise ValueError("native stdout or log identity changed")
        if report["candidate_sha256"]!=sha256(models[record["model"]]["decoded"]):
            raise ValueError("measured candidate changed")
        rows = {t["name"]:t for t in report["tensors"]}
        if len(rows)!=272 or rows.keys()!=expected.keys():
            raise ValueError("tensor set is incomplete or duplicated")
        passed = 0
        counts = {k:0 for k in ("kl","cosine","output_mse")}
        for name, row in rows.items():
            protected = any(v in name.lower() for v in ("token_embd","output.weight","lm_head","norm","bias"))
            limits = {"kl":.25 if protected else 1.,"cosine":float(np.float32(.995 if protected else .99)),
                "output_mse":.005 if protected else .02}
            if row["protected"]!=protected or row["limits"]!=limits:
                raise ValueError("native thresholds differ from frozen policy")
            if row["dimensions"]!=expected[name]["dimensions"] or row["observation"]!=expected[name]["kind"]:
                raise ValueError("native tensor shape or operator kind differs")
            samples = 0 if row["observation"]=="histogram" else (256 if record["regime"].endswith("-fit") else 64)
            if row["samples"]!=samples or row["output_gate_required"]!=(samples!=0):
                raise ValueError("native observation budget or requirement differs")
            finite_kl = all(row[k] is not None and np.isfinite(row[k]) for k in ("kl_forward","kl_reverse"))
            gates = {"kl":finite_kl and max(row["kl_forward"],row["kl_reverse"])<=limits["kl"],
                "cosine":row["cosine"] is not None and row["cosine"]>=limits["cosine"],
                "output_mse":not row["output_gate_required"] or (row["output_mse"] is not None and row["output_mse"]<=limits["output_mse"])}
            if gates!=row["gates"] or row["passed"]!=all(gates.values()):
                raise ValueError("reported native score and decision disagree")
            if row["output_mse_state"]!=("not_applicable" if not samples else "finite"):
                raise ValueError("unexpected nonfinite model metric")
            if record["model"]=="source-f32" and (not row["passed"] or row["kl_forward"]!=0
                    or row["kl_reverse"]!=0 or row["cosine"]!=1 or (samples and row["output_mse"]!=0)):
                raise ValueError("identity control does not have identity metrics")
            passed += row["passed"]
            for key in counts:
                counts[key] += not gates[key]
        if (passed!=report["passed_count"] or report["failed_count"]!=272-passed
                or counts!=report["gate_failures"] or report["all_passed"]!=(passed==272)
                or report["unused_calibration_entries"]!=["token_embd.weight::output"]
                or not report["headers_identical"]):
            raise ValueError("native report summary or metadata scope differs")
        checks.append({"model":record["model"],"regime":record["regime"],"passed":passed,
            "failed":272-passed,"gate_failures":counts,
            "failed_tensors":[{k:t[k] for k in ("name","gates","kl_forward","kl_reverse","cosine","output_mse")}
                              for t in report["tensors"] if not t["passed"]]})
    result = {"status":"PASSED","reports":16,"tensor_decisions":16*272,"identity_controls":4,"checks":checks}
    save_json(root/"audit.json",result)
    return result


def run(root):
    root = Path(root).resolve()
    (root/"evaluations").mkdir(exist_ok=False)
    baseline = json.loads((root/"baseline.json").read_text())
    regression = json.loads((root/"regression.json").read_text())
    if regression["status"]!="PASSED" or regression["new_binary_sha256"]!=sha256(BINARY):
        raise ValueError("native preflight regression is missing or stale")
    if sha256(ROOT/"target/release/atom-quantizer")!=baseline["incumbent_release_sha256"]:
        raise ValueError("incumbent release binary changed")
    previous = json.loads((D03/"summary.json").read_text())
    upgraded = json.loads((R1/"summary.json").read_text())
    models = {"source-f32":previous["models"]["source-f32"],
        "historical-budget120":previous["models"]["historical-budget120"],
        **{name:upgraded["models"][name] for name in ("q3-IE-r1","q4-IE-r2")}}
    inputs = [SOURCE,*CALIBRATIONS.values(),BINARY,ROOT/"target/release/atom-quantizer"]
    for model in models.values():
        if sha256(model["decoded"])!=model["decoded_sha256"]:
            raise ValueError("reference model identity changed")
        inputs.append(Path(model["decoded"]))
    code = [ROOT/"Cargo.toml",ROOT/"Cargo.lock",Path(__file__),ROOT/"research/native-admission-2026-10-01/PROTOCOL.md",
        ROOT/"experiments/full22/data.py",ROOT/"experiments/basis_discovery.py",ROOT/"tests/cli_assess.rs"]
    code += sorted((ROOT/"src").rglob("*.rs"))
    frozen = {"sha256":{str(p):sha256(p) for p in [*inputs,*code]},"commands":[],"max_workers":2,
        "compiler":subprocess.run(["rustc","--version"],capture_output=True,text=True,check=True).stdout.strip(),
        "binary_version":subprocess.run([str(BINARY),"--version"],capture_output=True,text=True,check=True).stdout.strip()}
    work = [(root,name,model,regime,path) for regime,path in CALIBRATIONS.items() for name,model in models.items()]
    frozen["commands"] = [[str(BINARY),"assess",str(SOURCE),item[2]["decoded"],"--calibration",str(item[4])] for item in work]
    save_json(root/"manifest.json",frozen)
    save_json(root/"models.json",models)
    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,*item) for item in work]):
            records.append(future.result())
            save_json(root/"progress.json",records)
    records.sort(key=lambda r:(list(CALIBRATIONS).index(r["regime"]),list(models).index(r["model"])))
    if any(sha256(p)!=h for p,h in frozen["sha256"].items()):
        raise ValueError("native implementation or input changed during N1")
    save_json(root/"records.json",records)
    result = audit(records,models,root)
    summary = {"experiment":"N1","status":"MEASURED","native_reports":len(records),"audit":result["status"],
        "results":[{k:v for k,v in c.items() if k!="failed_tensors"} for c in result["checks"]],
        "no_models_fitted":True,"no_new_inference":True,"incumbent_release_unchanged":True}
    save_json(root/"summary.json",summary)
    print(json.dumps(summary),flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
