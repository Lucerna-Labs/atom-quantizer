"""R1 real-model verification of A22-2 without reusing scores for changed weights."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import zipfile

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .direction_models import build
from .full22 import archive
from .full22.data import ROOT, OUT, SOURCE, sha256

D03 = ROOT/"artifacts/calibration-roles-2026-10-01"
D02 = ROOT/"artifacts/calibration-coverage-2026-10-01"
NAMES = ("q3-IE-r1", "q4-IE-r2")


def verify(old, new):
    count = 0
    tensors = {t.name:t for t in gguf.GGUFReader(new["decoded"]).tensors}
    with zipfile.ZipFile(old["archive"]) as legacy, zipfile.ZipFile(new["archive"]) as committed:
        before = json.loads(legacy.read("manifest.json"))
        after = json.loads(committed.read("manifest.json"))
        if before["format"]!="A22-1" or after["format"]!="A22-2":
            raise ValueError("unexpected archive versions")
        if len(before["tensors"])!=272 or len(after["tensors"])!=272 or len(tensors)!=272:
            raise ValueError("incomplete tensor table")
        for a,b in zip(before["tensors"],after["tensors"]):
            if a["name"]!=b["name"] or a["entry"]!=b["entry"] or legacy.read(a["entry"])!=committed.read(b["entry"]):
                raise ValueError("underlying tensor representation changed")
            data = tensors[b["name"]].data.tobytes(order="C")
            if hashlib.sha256(data).hexdigest()!=b["decoded_sha256"]:
                raise ValueError("stored decoded commitment differs from actual GGUF bytes")
            count += 1
        if legacy.read("gguf-prefix.bin")!=committed.read("gguf-prefix.bin"):
            raise ValueError("GGUF prefix changed")
        if {k:v for k,v in before.items() if k not in ("format","tensors")} != {k:v for k,v in after.items() if k not in ("format","tensors")}:
            raise ValueError("non-version archive metadata changed")
    if sha256(new["decoded"])!=old["decoded_sha256"]:
        raise ValueError("new decoded model differs from the model whose PPL was measured")
    result = {"identical_records":count,"committed_tensors":count,"decoded_sha256":old["decoded_sha256"],
        "old_archive_bytes":old["archive_bytes"],"new_archive_bytes":new["archive_bytes"],
        "added_bytes":new["archive_bytes"]-old["archive_bytes"]}
    export_log = Path(new["archive"]).with_suffix(".export.log")
    exported = ast.literal_eval(export_log.read_text().strip())
    if exported["format"]!="A22-2" or exported["decoded_verified_tensors"]!=272 or exported["sha256"]!=old["decoded_sha256"]:
        raise ValueError("standalone export did not verify every commitment")
    result["standalone_export"] = exported
    return result


def fail_closed_probe(root, model):
    corrupt = root/"wrong-commitment.a22"
    with zipfile.ZipFile(model["archive"]) as source, zipfile.ZipFile(corrupt,"w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name=="manifest.json":
                manifest = json.loads(data)
                victim = manifest["tensors"][-1]
                tensor = victim["name"]
                expected_actual = victim["decoded_sha256"]
                victim["decoded_sha256"] = "0"*64
                data = json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode()
            target.writestr(name,data)
    destination = root/"must-not-publish.gguf"
    try:
        archive.export_model(corrupt,destination)
    except archive.DecodedChecksumError as error:
        if error.tensor!=tensor or error.expected!="0"*64 or error.actual!=expected_actual:
            raise ValueError("wrong commitment failed at an unexpected boundary") from error
        if destination.exists() or list(root.glob(".must-not-publish.gguf.*.tmp")):
            raise ValueError("failed decode left a published or staged model")
        return {"status":"PASSED","injected_fault":True,"tensor":tensor,"actual_hash":error.actual,
            "publication_rejected":True,"staging_removed":True,"test_archive_sha256":sha256(corrupt)}
    raise ValueError("wrong decoded commitment was accepted")


def old_reader_check(root, new):
    path = root/"archive-before-R1.py"
    with tarfile.open(D03/"implementation.tar.gz") as source:
        data = source.extractfile("experiments/full22/archive.py").read()
    path.write_bytes(data)
    before = json.loads((D03/"manifest.json").read_text())["sha256"][str(ROOT/"experiments/full22/archive.py")]
    if sha256(path)!=before:
        raise ValueError("old decoder snapshot identity differs")
    spec = importlib.util.spec_from_file_location("experiments.full22.archive_before_R1",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = root/"old-reader-must-reject.gguf"
    try:
        module.export_model(new["archive"],destination)
    except ValueError as error:
        if str(error)!="unknown model format" or destination.exists():
            raise ValueError("old decoder failed for an unexpected reason") from error
        return {"status":"PASSED","old_reader_sha256":before,"error":str(error),"destination_absent":True}
    raise ValueError("old reader silently accepted new archive semantics")


def run(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    models_dir = root/"models"
    models_dir.mkdir(parents=True,exist_ok=False)
    scripts = [Path(__file__),ROOT/"experiments/direction_models.py",ROOT/"experiments/basis_discovery.py",
        ROOT/"research/decoder-reproducibility-2026-10-01/PROTOCOL.md"]
    scripts += sorted((ROOT/"experiments/full22").glob("*.py"))
    base, correction = D02/"captures/stratified0-fit.acal", OUT/"fit.acal"
    frozen = {str(p):sha256(p) for p in [*scripts,SOURCE,base,correction,D03/"summary.json"]}
    save_json(root/"manifest.json",frozen)
    previous = json.loads((D03/"summary.json").read_text())
    models, checks, scores = {}, {}, {}
    for name in NAMES:
        old = previous["models"][name]
        if sha256(old["archive"])!=old["archive_sha256"] or sha256(old["decoded"])!=old["decoded_sha256"]:
            raise ValueError("D03 reference changed")
        new = build(old["config"],name,models_dir,base,correction)
        models[name],checks[name] = new,verify(old,new)
        old_score = previous["fresh_final"][name]
        score_file = D03/"evaluations"/f"fresh-final-{name}.json"
        score = json.loads(score_file.read_text())
        if (score!=old_score or score["status"]!="MEASURED" or score["model_sha256"]!=new["decoded_sha256"]
                or score["log_sha256"]!=sha256(score_file.with_suffix(".log"))):
            raise ValueError("PPL evidence is not for this exact decoded file")
        scores[name] = dict(score,reused_from=str(score_file),reason="identical decoded GGUF SHA-256; no new inference")
        save_json(root/"progress.json",{"models":models,"checks":checks,"reused_ppl":scores})
    negative = fail_closed_probe(root,models[NAMES[-1]])
    old_reader = old_reader_check(root,models[NAMES[-1]])
    legacy = archive.export_model(previous["models"][NAMES[-1]]["archive"],root/"legacy-redecode.gguf")
    if legacy["format"]!="A22-1" or legacy["decoded_verified_tensors"]!=0 or legacy["sha256"]!=models[NAMES[-1]]["decoded_sha256"]:
        raise ValueError("legacy compatibility or verification accounting failed")
    if any(sha256(p)!=h for p,h in frozen.items()):
        raise ValueError("R1 implementation or inputs changed during verification")
    result = {"status":"PASSED","models":models,"checks":checks,"reused_ppl":scores,"new_inference_runs":0,
        "failure_injection":negative,"old_reader":old_reader,"legacy_export":legacy,
        "original_D03_failure_cause":"UNDETERMINED","default_numerics_changed":False}
    save_json(root/"summary.json",result)
    print(json.dumps({"status":result["status"],"checks":checks,"failure_injection":negative,"old_reader":old_reader}),flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",required=True)
    run(parser.parse_args().out)
