"""Validate complete native assess reports before using them as fitting decisions."""
import json
import hashlib
from pathlib import Path
import subprocess

import gguf
import numpy as np

from .data import load_calibration, sha256


def validate(report, source, candidate, calibration, returncode):
    reader = gguf.GGUFReader(source)
    tensors = reader.tensors
    expected = {t.name:t for t in tensors}
    observations = load_calibration(calibration,source)
    identity = {"source_sha256":sha256(source),"candidate_sha256":sha256(candidate),
        "calibration_sha256":sha256(calibration)}
    if (not isinstance(report,dict) or report.get("format")!="native-fidelity-1"
            or report.get("backend")!="cpu" or report.get("scope")!="decoded_tensor_fidelity"
            or report.get("max_output_mse")!=.02 or report.get("headers_identical") is not True
            or any(report.get(k)!=v for k,v in identity.items())
            or report.get("tensor_count")!=len(expected) or not isinstance(report.get("tensors"),list)
            or len(report["tensors"])!=len(expected)
            or any(type(report.get(k)) is not int for k in ("tensor_count","passed_count","failed_count"))):
        raise ValueError("native report has wrong identity or incomplete model coverage")
    with Path(source).open("rb") as stream: prefix = stream.read(reader.data_offset)
    with Path(candidate).open("rb") as stream: candidate_prefix = stream.read(reader.data_offset)
    prefix_hash = hashlib.sha256(prefix).hexdigest()
    if (candidate_prefix!=prefix or report.get("source_header_sha256")!=prefix_hash
            or report.get("candidate_header_sha256")!=prefix_hash):
        raise ValueError("native report model metadata differs")
    used, seen, failed = set(),set(),set()
    counts = {k:0 for k in ("kl","cosine","output_mse")}
    for row in report["tensors"]:
        if not isinstance(row,dict) or row.get("name") not in expected or row["name"] in seen:
            raise ValueError("native report has invalid or repeated tensors")
        name = row["name"]; seen.add(name)
        tensor = expected[name]
        protected = any(v in name.lower() for v in ("token_embd","output.weight","lm_head","norm","bias"))
        limits = {"kl":.25 if protected else 1.,"cosine":float(np.float32(.995 if protected else .99)),
            "output_mse":.005 if protected else .02}
        calibrated = bool(tensor.data.size//int(tensor.shape[0])>1 and tensor.shape[0]>1)
        if calibrated:
            if name not in observations:
                raise ValueError("native report omitted a required calibration operator")
            used.add(name)
            x = observations[name]
            kind,samples = ("linear" if x.ndim==2 else "embedding_lookup"),len(x)
        else:
            kind,samples = "histogram",0
        if (row.get("dimensions")!=list(tensor.shape) or any(type(n) is not int for n in row.get("dimensions",[]))
                or any(type(row.get(k)) is not int for k in ("source_type","candidate_type","samples"))
                or row.get("source_type")!=int(tensor.tensor_type)
                or row.get("candidate_type")!=int(tensor.tensor_type) or row.get("protected") is not protected
                or row.get("limits")!=limits or row.get("observation")!=kind or row.get("samples")!=samples
                or row.get("output_gate_required") is not calibrated):
            raise ValueError("native policy, tensor or observation contract differs")
        fwd,rev,cos,mse = [row.get(k) for k in ("kl_forward","kl_reverse","cosine","output_mse")]
        finite = lambda n:type(n) in (float,int) and np.isfinite(n)
        gates = {"kl":finite(fwd) and finite(rev) and max(fwd,rev)<=limits["kl"],
            "cosine":finite(cos) and cos>=limits["cosine"],
            "output_mse":not calibrated or (finite(mse) and mse<=limits["output_mse"])}
        if (not isinstance(row.get("gates"),dict) or any(type(v) is not bool for v in row["gates"].values())
                or row["gates"]!=gates or type(row.get("passed")) is not bool or row["passed"]!=all(gates.values())):
            raise ValueError("native decision disagrees with its scores")
        if not all(gates.values()): failed.add(name)
        for key in counts: counts[key]+=not gates[key]
    if (report.get("failed_count")!=len(failed) or report.get("passed_count")!=len(expected)-len(failed)
            or report.get("gate_failures")!=counts or report.get("all_passed") is not (not failed)
            or report.get("unused_calibration_entries")!=sorted(set(observations)-used)
            or returncode!=(2 if failed else 0)):
        raise ValueError("native aggregate decisions are inconsistent")
    return failed


def assess(executable, source, candidate, calibration, output):
    executable,source,candidate,calibration,output = map(Path,(executable,source,candidate,calibration,output))
    if output.exists() or output.with_suffix(".stderr").exists():
        raise ValueError("native assessment output exists")
    binary_hash = sha256(executable)
    command = [str(executable),"assess",str(source),str(candidate),"--calibration",str(calibration)]
    result = subprocess.run(command,capture_output=True,text=True,timeout=1800)
    output.with_suffix(".stderr").write_text(result.stderr)
    output.with_suffix(".stdout").write_text(result.stdout)
    if result.returncode not in (0,2):
        raise ValueError(f"native assessment failed with exit {result.returncode}")
    report = json.loads(result.stdout)
    validate(report,source,candidate,calibration,result.returncode)
    if sha256(executable)!=binary_hash:
        raise ValueError("native assessor changed during execution")
    receipt = {"command":command,"returncode":result.returncode,"binary_sha256":binary_hash,"report":report,
        "stdout_sha256":sha256(output.with_suffix(".stdout")),"stderr_sha256":sha256(output.with_suffix(".stderr"))}
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt
