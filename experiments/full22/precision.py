"""Operational fit-driven precision escalation with native full-model gates."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import gguf
from threadpoolctl import threadpool_limits

from . import harness, native
from .data import ROOT, sha256

LADDER = (4,6,8,"exact")


def save(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def publish_copy(source, destination, expected_hash):
    fd,name = tempfile.mkstemp(prefix=".publish-",suffix=".tmp",dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd,"wb") as dst,Path(source).open("rb") as src:
            shutil.copyfileobj(src,dst,1024*1024)
            dst.flush(); os.fsync(dst.fileno())
        if sha256(temporary)!=expected_hash:
            raise ValueError("final model publication changed bytes")
        os.link(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)


def config_for(level, correction, geometry="activation"):
    if geometry not in ("activation","diagonal"):
        raise ValueError("invalid correction geometry")
    if level=="exact": return {"family":"lossless"}
    if type(level) is not int or level not in LADDER[:-1]:
        raise ValueError("invalid precision tier")
    return {"bits":level,**({"rank":2,"rank_geometry":geometry} if correction else {})}


def advance(levels, failures):
    unknown = set(failures)-set(levels)
    if unknown: raise ValueError("native failures refer to unknown tensors")
    result = dict(levels)
    for name in failures:
        if levels[name]>=len(LADDER)-1:
            raise ValueError(f"exact representation failed native admission for {name}")
        result[name]+=1
    return result


def failure_union(reports):
    if not reports: raise ValueError("native fitting observations are required")
    return {row["name"] for receipt in reports for row in receipt["report"]["tensors"] if not row["passed"]}


def build_stage(source, base, residual, configs, root, previous=None, *, default_config=None):
    root.mkdir(parents=True,exist_ok=False)
    artifact,decoded = root/"model.a22",root/"model.gguf"
    default = dict(default_config) if default_config is not None else config_for(4,residual is not None)
    reuse = {} if previous is None else {"reuse_archive":previous["archive"],"reuse_sha256":previous["archive_sha256"]}
    with (root/"build.log").open("x") as log,contextlib.redirect_stdout(log):
        harness.build_model(default,artifact,source,base,tensor_configs=configs,residual_calibration=residual,**reuse)
    program = """
import json,os,sys
from experiments.full22.archive import export_model
blocked={os.path.realpath(p) for p in sys.argv[3:]}
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)) and os.path.realpath(os.fsdecode(args[0])) in blocked:
        raise PermissionError('fitting input access forbidden during standalone export')
sys.addaudithook(audit)
print(json.dumps(export_model(sys.argv[1],sys.argv[2])))
"""
    command = [sys.executable,"-c",program,str(artifact),str(decoded),str(source),str(base)]
    if residual is not None: command.append(str(residual))
    with (root/"export.log").open("x") as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    export = json.loads((root/"export.log").read_text())
    with zipfile.ZipFile(artifact) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if export["format"]!="A22-2" or export["decoded_verified_tensors"]!=len(manifest["tensors"]):
            raise ValueError("not every decoded commitment was verified")
        actual = {t.name:t for t in gguf.GGUFReader(decoded).tensors}
        for entry in manifest["tensors"]:
            if hashlib.sha256(actual[entry["name"]].data.tobytes()).hexdigest()!=entry["decoded_sha256"]:
                raise ValueError("exported tensor differs from its commitment")
    result = {"status":"MEASURED","archive":str(artifact),"archive_bytes":artifact.stat().st_size,
        "archive_sha256":sha256(artifact),"decoded":str(decoded),"decoded_sha256":sha256(decoded),
        "verified_tensors":len(actual),"tensor_configs":configs,"config":default,
        "source_sha256":sha256(source),"calibration_sha256":sha256(base),
        "residual_calibration_sha256":None if residual is None else sha256(residual),
        "standalone_fitting_opens_denied":True}
    if previous is not None:
        unchanged = 0
        with zipfile.ZipFile(previous["archive"]) as old,zipfile.ZipFile(artifact) as new:
            before = json.loads(old.read("manifest.json"))
            after = json.loads(new.read("manifest.json"))
            for a,b in zip(before["tensors"],after["tensors"]):
                if a["name"]!=b["name"]: raise ValueError("tensor order changed")
                if a["config"]==b["config"]:
                    if old.read(a["entry"])!=new.read(b["entry"]):
                        raise ValueError("passing tensor representation changed")
                    unchanged+=1
        result["unchanged_records"]=unchanged
    save(root/"model.json",result)
    return result


def run(source, base, residual, gates, assessor, out, *, start_bits=4, rank_geometry=None):
    threadpool_limits(4)
    if type(start_bits) is not int or start_bits not in LADDER[:-1]:
        raise ValueError("starting precision must be 4, 6 or 8 bits")
    if residual is None and rank_geometry is not None:
        raise ValueError("correction geometry requires residual observations")
    geometry = "activation" if rank_geometry is None else rank_geometry
    config_for(start_bits,residual is not None,geometry)
    start_index = LADDER.index(start_bits)
    source,base,assessor,out = [Path(p).resolve() for p in (source,base,assessor,out)]
    residual = Path(residual).resolve() if residual is not None else None
    gates = [Path(p).resolve() for p in gates]
    if not gates or len(set(map(sha256,gates)))!=len(gates):
        raise ValueError("provide distinct fitting observation files")
    out.mkdir(parents=True,exist_ok=False)
    frozen = {str(p):sha256(p) for p in [source,base,assessor,*gates,*([residual] if residual else [])]}
    frozen.update({str(p):sha256(p) for p in Path(__file__).parent.glob("*.py")})
    save(out/"manifest.json",{"sha256":frozen,"ladder":LADDER,"correction":residual is not None,
        "start_bits":start_bits,"rank_geometry":geometry if residual else None,
        "gate_calibrations":list(map(str,gates))})
    tensors = gguf.GGUFReader(source).tensors
    levels = {t.name:start_index if t.data.ndim==2 else len(LADDER)-1 for t in tensors}
    stages,previous = [],None
    for iteration in range(len(LADDER)-start_index):
        configs = {name:config_for(LADDER[level],residual is not None,geometry) for name,level in levels.items()}
        stage_dir = out/f"stage-{iteration}"
        model = build_stage(source,base,residual,configs,stage_dir,previous,
            default_config=config_for(start_bits,residual is not None,geometry))
        reports = []
        for index,path in enumerate(gates):
            print(f"ASSESS stage {iteration} fit view {index}",flush=True)
            report = native.assess(assessor,source,model["decoded"],path,stage_dir/f"fit-{index}.json")
            if report["report"]["candidate_sha256"]!=model["decoded_sha256"]:
                raise ValueError("native assessment is not for the built candidate")
            reports.append(report)
        failures = failure_union(reports)
        stage = {"iteration":iteration,"levels":dict(levels),"model":model,
            "failed_tensors":sorted(failures),"native_reports":[str(stage_dir/f"fit-{i}.json") for i in range(len(gates))]}
        stages.append(stage)
        save(out/"progress.json",{"stages":stages})
        print(f"STAGE {iteration}: {len(failures)} failed tensors; {model['archive_bytes']} bytes",flush=True)
        if not failures:
            break
        levels = advance(levels,failures)
        previous = model
    else:
        raise ValueError("precision ladder exhausted without native admission")
    if any(sha256(p)!=h for p,h in frozen.items()):
        raise ValueError("adaptive input or implementation changed during fitting")
    final = dict(model,name="adaptive-"+geometry if residual else "adaptive-scalar")
    for key,suffix in (("archive",".a22"),("decoded",".gguf")):
        destination = out/("model"+suffix)
        publish_copy(model[key],destination,model[key+"_sha256"])
        final[key]=str(destination)
    result = {"status":"FIT_ADMITTED","model":final,"stages":stages,"levels":levels,
        "start_bits":start_bits,"rank_geometry":geometry if residual else None,
        "ladder":LADDER,"gate_calibrations":list(map(str,gates)),"native_assessor_sha256":sha256(assessor)}
    save(out/"result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--base-calibration",required=True)
    parser.add_argument("--residual-calibration")
    parser.add_argument("--gate-calibration",action="append",required=True)
    parser.add_argument("--assessor",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--start-bits",type=int,choices=(4,6,8),default=4)
    parser.add_argument("--rank-geometry",choices=("activation","diagonal"))
    args = parser.parse_args()
    result = run(args.source,args.base_calibration,args.residual_calibration,args.gate_calibration,args.assessor,args.out,
        start_bits=args.start_bits,rank_geometry=args.rank_geometry)
    print(json.dumps({"status":result["status"],"model":result["model"],"stages":len(result["stages"])}))


if __name__=="__main__": main()
