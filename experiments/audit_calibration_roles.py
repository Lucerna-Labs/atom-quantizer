"""Independent artifact/log and decision audit for the completed D03 experiment."""
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
from .full22 import codec
from .full22.data import ROOT, SOURCE, sha256
from .full22.records import Record


def audit(root):
    threadpool_limits(4)
    root = Path(root).resolve()
    result = json.loads((root/"summary.json").read_text())
    models = result["models"]
    if len(models) != 36 or result["new_cases"] != 24 or result["reused_cases"] != 12:
        raise ValueError("incomplete D03 case matrix")
    frozen = json.loads((root/"manifest.json").read_text())["sha256"]
    for path, expected in frozen.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    scalar = {}
    for name, model in models.items():
        if "-scalar-" in name:
            with zipfile.ZipFile(model["archive"]) as source:
                manifest = json.loads(source.read("manifest.json"))
                scalar[name] = {e["name"]:Record.loads(source.read(e["entry"])) for e in manifest["tensors"]}
    artifact_rows = []
    for name, model in sorted(models.items()):
        if model["status"] != "MEASURED" or sha256(model["decoded"]) != model["decoded_sha256"]:
            raise ValueError(f"unverified model: {name}")
        if "archive" not in model:
            continue
        if sha256(model["archive"]) != model["archive_sha256"] or Path(model["archive"]).stat().st_size != model["archive_bytes"]:
            raise ValueError("archive hash/size differs")
        exported = {t.name:t for t in gguf.GGUFReader(model["decoded"]).tensors}
        role = name.split("-")[1]
        base = "E" if role == "diagonal" else (name[-1] if role == "scalar" else role[0])
        correction = "E" if role == "diagonal" else (base if role == "scalar" else role[1])
        cal_hashes = {"E":"f5e6d79eb6b552da48cf976ea4c114d0ac387689ef42de5fa8746e257056431f",
            "I":"29d09e1701dc329cee26fd636dd1d492c3e29b1e1c3d17d28e2c777a52e9a2c4"}
        bases, tensors, corrections = 0, 0, 0
        with zipfile.ZipFile(model["archive"]) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            if (manifest["config"] != model["config"] or manifest["source_sha256"] != sha256(SOURCE)
                    or manifest["calibration_sha256"] != cal_hashes[base]
                    or manifest.get("residual_calibration_sha256",manifest["calibration_sha256"]) != cal_hashes[correction]
                    or len(manifest["tensors"]) != 272 or len(exported) != 272):
                raise ValueError("archive role or tensor identity differs")
            for entry in manifest["tensors"]:
                payload = archive.read(entry["entry"])
                if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise ValueError("record identity differs")
                record = Record.loads(payload)
                reconstructed = codec.decode(record)
                actual = np.asarray(exported[entry["name"]].data,dtype=np.float32)
                if actual.shape != reconstructed.shape or actual.tobytes() != reconstructed.tobytes():
                    different = actual.view(np.uint32) != reconstructed.view(np.uint32)
                    left, right = actual[different], reconstructed[different]
                    diagnostic = {"model":name,"tensor":entry["name"],"differing_values":int(different.sum()),
                        "numerically_equal":bool(np.array_equal(actual,reconstructed)),
                        "only_signed_zero":bool(np.all((left==0)&(right==0))),
                        "max_abs_difference":float(np.max(np.abs(actual-reconstructed))),
                        "repeat_decode_matches":codec.decode(record).tobytes()==reconstructed.tobytes(),
                        "examples":[{"exported_bits":int(a),"decoded_bits":int(b)}
                            for a,b in zip(left.view(np.uint32)[:8],right.view(np.uint32)[:8])]}
                    save_json(root/"audit-byte-mismatch.json",diagnostic)
                    raise ValueError("exported tensor bytes differ from saved decode")
                tensors += 1
                if role != "scalar":
                    reference = scalar[f"{name[:2]}-scalar-{base}"][entry["name"]]
                    extra = set(record.arrays) - set(reference.arrays)
                    if extra not in (set(), {"lowrank_u","lowrank_v"}):
                        raise ValueError("unexpected correction payload")
                    if set(reference.arrays)-set(record.arrays):
                        raise ValueError("base payload removed")
                    for key, expected in reference.arrays.items():
                        observed = record.arrays[key]
                        if observed.dtype != expected.dtype or observed.shape != expected.shape or observed.tobytes() != expected.tobytes():
                            raise ValueError("base array bytes changed")
                    stripped = json.loads(json.dumps(record.meta))
                    stripped.pop("lowrank_effective_rank",None)
                    stripped.get("config",{}).pop("rank",None)
                    stripped.get("config",{}).pop("rank_geometry",None)
                    if stripped != reference.meta:
                        raise ValueError("base decoder metadata changed")
                    bases += 1
                    corrections += bool(extra)
        artifact_rows.append({"name":name,"tensors":tensors,"base_records_unchanged":bases,
            "corrected_records":corrections,"archive_bytes":model["archive_bytes"],"archive_sha256":model["archive_sha256"]})
        print(f"AUDITED {name}: {tensors} tensors, {bases} unchanged base records",flush=True)
    history = json.loads((ROOT/"research/foundations-2026-09-05/results.json").read_text())
    historical = next(r for r in history["results"] if r["name"] == "budget120")
    if (sha256(historical["artifact"]) != models["historical-budget120"]["archive_sha256"]
            or Path(historical["artifact"]).stat().st_size != models["historical-budget120"]["archive_bytes"]):
        raise ValueError("historical reference changed")
    inference_rows = []
    for corpus_key, corpus_name in (("selection","selection"),("fresh_final","fresh-final")):
        for name, score in result[corpus_key].items():
            path = Path(score["reused_from"]) if "reused_from" in score else root/"evaluations"/f"{corpus_name}-{name}.json"
            saved = json.loads(path.read_text())
            keys = ("status","chunks","returncode","perplexity","standard_error","model_sha256","corpus_sha256","runtime_sha256","log_sha256")
            if any(score[k] != saved[k] for k in keys) or saved["status"] != "MEASURED" or saved["returncode"] != 0:
                raise ValueError("inference receipt differs or failed")
            command = saved["command"]
            if command[5:] != ["--chunks","64","-c","512","-b","512","-ub","512","-t","4","-tb","4","-ngl","0"]:
                raise ValueError("inference options changed")
            if (saved["model_sha256"] != sha256(models[name]["decoded"])
                    or saved["corpus_sha256"] != sha256(command[4])
                    or saved["runtime_sha256"] != sha256(root/"runtime.json")
                    or saved["log_sha256"] != sha256(path.with_suffix(".log"))):
                raise ValueError("inference identity differs")
            log = path.with_suffix(".log").read_text()
            match = re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)",log)
            chunks = re.search(r"calculating perplexity over (\d+) chunks",log)
            if (not match or not chunks or int(chunks[1]) != 64
                    or float(match[1]) != saved["perplexity"] or float(match[2]) != saved["standard_error"]):
                raise ValueError("inference log disagrees with receipt")
            inference_rows.append({"name":name,"corpus":corpus_name,"ppl":saved["perplexity"],"reused":"reused_from" in score})
    def admitted(bits, rank, scores):
        ppl = lambda role,r=rank:scores[f"q{bits}-{role}-r{r}"]["perplexity"]
        size = models[f"q{bits}-IE-r{rank}"]["archive_bytes"]
        scalar_bytes = min(models[f"q{bits}-scalar-{r}"]["archive_bytes"] for r in "EI")
        return (ppl("IE") <= .95*min(ppl(r) for r in ("EE","II","EI"))
                and ppl("IE") <= ppl("diagonal")
                and ppl("IE") <= min(ppl("EE",1),ppl("diagonal",1)) and size <= 1.10*scalar_bytes)
    expected_selection = []
    chosen = {}
    for bits in (3,4):
        for rank in (1,2,4):
            status = "QUALIFIED" if admitted(bits,rank,result["selection"]) else "FAILED"
            expected_selection.append({"bits":bits,"rank":rank,"status":status})
            if status == "QUALIFIED" and str(bits) not in chosen:
                chosen[str(bits)] = rank
    reported = [{k:g[k] for k in ("bits","rank","status")} for g in result["selection_decisions"]]
    if reported != expected_selection or chosen != result["selected_ranks"]:
        raise ValueError("selection decision does not follow the precommit")
    final_decisions = [{"bits":int(bits),"rank":rank,
        "status":"QUALIFIED" if admitted(int(bits),rank,result["fresh_final"]) else "FAILED"} for bits,rank in chosen.items()]
    if final_decisions != [{k:g[k] for k in ("bits","rank","status")} for g in result["final_decisions"]]:
        raise ValueError("final decision does not follow the precommit")
    if not chosen and (result["fresh_final"] or list((root/"evaluations").glob("*fresh-final*"))):
        raise ValueError("unauthorized conditional final execution")
    report = {"status":"PASSED","audit_source_sha256":sha256(Path(__file__)),"summary_sha256":sha256(root/"summary.json"),
        "archives":len(artifact_rows),"exported_tensors_compared":sum(r["tensors"] for r in artifact_rows),
        "base_records_unchanged":sum(r["base_records_unchanged"] for r in artifact_rows),
        "new_inference_runs":sum(not r["reused"] for r in inference_rows),
        "reused_inference_runs":sum(r["reused"] for r in inference_rows),
        "selection_decisions":expected_selection,"final_decisions":final_decisions,
        "artifacts":artifact_rows,"inference":inference_rows}
    save_json(root/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("artifacts","inference")}),flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    audit(parser.parse_args().run)
