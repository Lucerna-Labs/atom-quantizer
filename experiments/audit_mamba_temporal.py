"""Independent stored-model, real trace, raw checkpoint and M1 gate audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import gguf
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .full22 import archive, codec
from .full22.data import sha256, load_calibration
from .full22.records import Record
from .mamba_models import ORDER
from .mamba_runtime import load


def close(actual, expected, message):
    if not np.allclose(actual, expected, rtol=1e-11, atol=1e-13):
        raise ValueError(message)


def record_models(root, models):
    count = 0
    for name in ORDER:
        model = models[name]
        if sha256(model["archive"]) != model["archive_sha256"] or Path(model["archive"]).stat().st_size != model["archive_bytes"]:
            raise ValueError("stored model identity differs")
        if sha256(model["decoded"]) != model["decoded_sha256"]:
            raise ValueError("native decoded model identity differs")
        tensors = {t.name: t for t in gguf.GGUFReader(model["decoded"]).tensors}
        if len(tensors) != 242:
            raise ValueError("incomplete Mamba export")
        with zipfile.ZipFile(model["archive"]) as saved:
            manifest = json.loads(saved.read("manifest.json"))
            archive.validate_manifest(saved.read("gguf-prefix.bin"), manifest)
            if manifest["format"] != "A22-2" or len(manifest["tensors"]) != 242:
                raise ValueError("incomplete Mamba archive")
            for item in manifest["tensors"]:
                data = saved.read(item["entry"])
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("stored record checksum differs")
                record = Record.loads(data)
                decoded = codec.decode(record)
                if decoded.tobytes() != tensors[item["name"]].data.tobytes() or hashlib.sha256(decoded.tobytes()).hexdigest() != item["decoded_sha256"]:
                    failure = root / "audit-decode-failure.npz"
                    with failure.open("xb") as stream:
                        np.savez(stream, expected=tensors[item["name"]].data, actual=decoded)
                    raise ValueError(f"decoded Mamba commitment differs: {name}/{item['name']}")
                count += 1
    with zipfile.ZipFile(models["q6-functional"]["archive"]) as full, zipfile.ZipFile(models["q6-functional-selectors-exact"]["archive"]) as protected:
        before = {t["name"]: t for t in json.loads(full.read("manifest.json"))["tensors"]}
        after = json.loads(protected.read("manifest.json"))["tensors"]
        same, changed = 0, 0
        for item in after:
            old = before[item["name"]]
            selector = item["name"].endswith((".ssm_x.weight", ".ssm_dt.weight"))
            if selector:
                if item["config"] != {"family": "lossless"}:
                    raise ValueError("selector is not exact")
                changed += 1
            else:
                if full.read(old["entry"]) != protected.read(item["entry"]):
                    raise ValueError("non-selector representation changed in the selector control")
                same += 1
        if (same, changed) != (194, 48):
            raise ValueError("selector comparison scope differs")
    return {"decoded_tensors": count, "identical_nonselector_records": same, "exact_selector_records": changed}


def inspect_trace(result, sequences):
    path = Path(result["trace"]["path"])
    if sha256(path) != result["trace"]["sha256"]:
        raise ValueError("trajectory trace changed")
    with np.load(path, allow_pickle=False) as source:
        trace = {k: source[k] for k in source.files}
    expected = {f"{kind}_{q}" for kind in ("ssm", "conv") for q in ("error", "reference", "candidate")}
    expected |= {"teacher_kl", "source_nll", "candidate_nll", "top1_match"}
    if set(trace) != expected:
        raise ValueError("missing trajectory observations")
    for key, values in trace.items():
        shape = (4, 1024, 24) if key.startswith(("ssm_", "conv_")) else (4, 1024)
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid complete trajectory shape or values")
    if not np.isin(trace["top1_match"], (0, 1)).all():
        raise ValueError("invalid token agreement observations")
    summary = result["summary"]
    for length in (64, 256, 1024):
        row = summary[str(length)]
        close(row["teacher_kl"], trace["teacher_kl"][:, :length].mean(), "prefix KL differs")
        close(row["candidate_ppl"], np.exp(trace["candidate_nll"][:, :length].mean()), "prefix PPL differs")
        close(row["source_ppl"], np.exp(trace["source_nll"][:, :length].mean()), "source PPL differs")
        close(row["top1_agreement"], trace["top1_match"][:, :length].mean(), "agreement differs")
        for kind in ("ssm", "conv"):
            e, r = trace[kind+"_error"][:, :length], trace[kind+"_reference"][:, :length]
            close(row[kind+"_nmse"], e.sum()/max(r.sum(), 1e-30), "aggregate state NMSE differs")
            close(row[kind+"_per_layer_nmse"], e.sum(axis=(0, 1))/np.maximum(r.sum(axis=(0, 1)), 1e-30), "layer NMSE differs")
            close(row[kind+"_per_sequence_nmse"], e.sum(axis=(1, 2))/np.maximum(r.sum(axis=(1, 2)), 1e-30), "sequence NMSE differs")
    close(summary["tail_256_ssm_nmse"], trace["ssm_error"][:, 768:].sum()/trace["ssm_reference"][:, 768:].sum(), "tail NMSE differs")
    close(summary["teacher_kl"], trace["teacher_kl"].mean(), "global KL differs")
    seen = set()
    for checkpoint in result["checkpoints"]:
        seq, length = checkpoint["sequence"], checkpoint["prefix"]
        if (seq, length) in seen or seq not in range(4) or length not in (64, 256, 1024):
            raise ValueError("invalid/repeated raw checkpoint")
        seen.add((seq, length))
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("raw checkpoint changed")
        with np.load(checkpoint["path"], allow_pickle=False) as arrays:
            for kind, width in (("ssm", 16), ("conv", 4)):
                source, candidate = arrays["source_"+kind], arrays["candidate_"+kind]
                if source.shape != (24, 1536, width) or candidate.shape != source.shape:
                    raise ValueError("raw state coverage differs")
                a, b = source.astype(np.float64), candidate.astype(np.float64)
                for q, values in (("error", (a-b)**2), ("reference", a*a), ("candidate", b*b)):
                    close(values.sum(axis=(1, 2)), trace[kind+"_"+q][seq, length-1], "raw state disagrees with trace")
            logits = [arrays[k].astype(np.float64) for k in ("source_logits", "candidate_logits")]
            if any(x.shape != (50280,) or not np.isfinite(x).all() for x in logits):
                raise ValueError("raw vocabulary logits differ")
            logp = []
            for values in logits:
                shifted = values-values.max()
                logp.append(shifted-np.log(np.exp(shifted).sum()))
            target = sequences[seq][length]
            close(-logp[0][target], trace["source_nll"][seq, length-1], "raw source NLL differs")
            close(-logp[1][target], trace["candidate_nll"][seq, length-1], "raw candidate NLL differs")
            close(np.sum(np.exp(logp[0])*(logp[0]-logp[1])), trace["teacher_kl"][seq, length-1], "raw teacher KL differs")
    if len(seen) != 12:
        raise ValueError("missing prefix checkpoints")
    return trace


def replay_prefix(source_path, candidate_path, ids, checkpoint, reset):
    source, _ = load(source_path)
    candidate, _ = load(candidate_path)
    ac, bc = None, None
    with torch.inference_mode():
        for token in ids[:64]:
            inp = torch.tensor([[token]], dtype=torch.long)
            a = source(inp, cache_params=ac, use_cache=True)
            b = candidate(inp, cache_params=None if reset else bc, use_cache=True)
            ac, bc = a.cache_params, b.cache_params
    with np.load(checkpoint, allow_pickle=False) as expected:
        for label, cache in (("source", ac), ("candidate", bc)):
            for field, kind in (("recurrent_states", "ssm"), ("conv_states", "conv")):
                actual = np.stack([getattr(layer, field)[0][0].numpy() for layer in cache.layers])
                if actual.tobytes() != expected[label+"_"+kind].tobytes():
                    raise ValueError("independent actual-model state replay differs")
        for label, output in (("source", a), ("candidate", b)):
            if output.logits[0, 0].numpy().tobytes() != expected[label+"_logits"].tobytes():
                raise ValueError("independent actual-model logits replay differs")


def run(root):
    root = Path(root).resolve()
    threadpool_limits(4)
    torch.set_num_threads(4)
    for path in (root/"captures/manifest.json", root/"models/manifest.json", root/"evaluation-plan.json",
                 root/"native-evaluations/manifest.json", root/"trajectories/manifest.json"):
        if any(sha256(p) != h for p, h in json.loads(path.read_text())["sha256"].items()):
            raise ValueError(f"frozen M1 identity changed: {path}")
    models = json.loads((root/"models/summary.json").read_text())["models"]
    native = json.loads((root/"native-evaluations/summary.json").read_text())
    trajectories = json.loads((root/"trajectories/summary.json").read_text())
    sequences = json.loads((root/"sequences.json").read_text())["sequences"]
    if native["status"] != "MEASURED" or trajectories["status"] != "MEASURED":
        raise ValueError("M1 evaluation incomplete")
    source = root/"preflight/source.gguf"
    source_tensors = {t.name: t for t in gguf.GGUFReader(source).tensors}
    source_hashes = {n: hashlib.sha256(t.data.tobytes()).hexdigest() for n, t in source_tensors.items()}
    capture_rows = []
    captures = json.loads((root/"captures/summary.json").read_text())["captures"]
    if len(captures) != 4:
        raise ValueError("missing capture views")
    for name, receipt in captures.items():
        path = Path(receipt["path"])
        if sha256(path) != receipt["sha256"]:
            raise ValueError("capture payload changed")
        values = load_calibration(path, source)
        meta = json.loads(path.with_suffix(".acal.json").read_text())
        count = 256 if name.endswith("fit") else 64
        if (len(values) != 98 or meta["verified_source"]["operator_sha256"] != source_hashes
                or set(meta["counts"].values()) != {count}):
            raise ValueError("capture source/operator scope differs")
        for key, array in values.items():
            if len(array) != count or not np.isfinite(array).all():
                raise ValueError("invalid captured observations")
            if key == "token_embd.weight":
                if array.ndim != 1 or (array >= 50280).any():
                    raise ValueError("invalid embedding observations")
            else:
                weight = source_tensors["token_embd.weight" if key == "token_embd.weight::output" else key].data
                if array.ndim != 2 or array.shape[1] != weight.shape[1]:
                    raise ValueError("incorrect actual linear input dimensions")
        if sum(m == "actual_x_proj_output_prefix" for m in meta["capture_methods"].values()) != 24:
            raise ValueError("time projection capture incomplete")
        capture_rows.append({"name": name, "operators": 98, "samples": count, "verified_source_tensors": 242})
    artifacts = record_models(root, models)
    if not trajectories["original_checkpoint_and_source_gguf_opens_denied"] or len(trajectories["denied_probes"]) != 2:
        raise ValueError("standalone evaluation proof missing")
    consistency = trajectories["source_consistency"]
    if consistency["status"] != "MEASURED" or consistency["tokens"] != 1024 or consistency["max_logit_difference"] > .01 or any(v > 1e-8 for v in consistency["relative_state_mse"].values()):
        raise ValueError("source numerical consistency gate differs")
    original_nll, rows = None, []
    for name, result in trajectories["models"].items():
        trace = inspect_trace(result, sequences)
        if original_nll is not None and not np.array_equal(trace["source_nll"], original_nll):
            raise ValueError("source teacher changed across paired comparisons")
        original_nll = trace["source_nll"]
        checkpoint = next(c["path"] for c in result["checkpoints"] if c["sequence"] == 0 and c["prefix"] == 64)
        model = models["source-lossless"] if name == "source-reset" else models[name]
        replay_prefix(models["source-lossless"]["decoded"], model["decoded"], sequences[0], checkpoint, name == "source-reset")
        rows.append({"name": name, "observed_steps": 4096, "raw_checkpoints": 12, "independent_replayed_steps": 64})
    for name, score in native["scores"].items():
        path = root/"native-evaluations"/f"selection-{name}.json"
        if json.loads(path.read_text()) != score or score["returncode"] != 0 or score["chunks"] != 64:
            raise ValueError("native inference receipt differs")
        log = path.with_suffix(".log")
        if sha256(log) != score["log_sha256"] or sha256(models[name]["decoded"]) != score["model_sha256"]:
            raise ValueError("native inference identity differs")
        if (sha256(score["command"][4]) != score["corpus_sha256"]
                or sha256(root/"native-evaluations/runtime.json") != score["runtime_sha256"]
                or score["command"][5:] != ["--chunks", "64", "-c", "512", "-b", "512", "-ub", "512", "-t", "4", "-tb", "4", "-ngl", "0"]):
            raise ValueError("native inference corpus/runtime/settings differ")
        match = re.search(r"Final estimate: PPL = ([\d.eE+-]+) \+/- ([\d.eE+-]+)", log.read_text())
        chunks = re.search(r"calculating perplexity over (\d+) chunks", log.read_text())
        if not match or not chunks or int(chunks[1]) != 64 or float(match[1]) != score["perplexity"] or float(match[2]) != score["standard_error"]:
            raise ValueError("native PPL does not match actual log")
    traces, scores = trajectories["models"], native["scores"]
    g2 = (models["q6-functional"]["archive_bytes"] <= 1.05*models["q6-scalar"]["archive_bytes"]
          and scores["q6-functional"]["perplexity"] <= scores["q6-scalar"]["perplexity"]
          and traces["q6-functional"]["summary"]["teacher_kl"] <= traces["q6-scalar"]["summary"]["teacher_kl"]
          and traces["q6-functional"]["summary"]["tail_256_ssm_nmse"] <= .95*traces["q6-scalar"]["summary"]["tail_256_ssm_nmse"])
    g3 = (models["q6-functional-selectors-exact"]["archive_bytes"] <= 1.20*models["q6-functional"]["archive_bytes"]
          and scores["q6-functional-selectors-exact"]["perplexity"] <= scores["q6-functional"]["perplexity"]
          and traces["q6-functional-selectors-exact"]["summary"]["teacher_kl"] <= traces["q6-functional"]["summary"]["teacher_kl"]
          and traces["q6-functional-selectors-exact"]["summary"]["tail_256_ssm_nmse"] <= .80*traces["q6-functional"]["summary"]["tail_256_ssm_nmse"])
    report = {"status": "PASSED", "artifacts": artifacts, "captures": capture_rows, "trajectories": rows,
              "native_inference_runs": len(scores), "G2": "MEASURED" if g2 else "FAILED", "G3": "MEASURED" if g3 else "FAILED",
              "auditor_sha256": sha256(__file__)}
    save_json(root/"audit.json", report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    run(parser.parse_args().run)
