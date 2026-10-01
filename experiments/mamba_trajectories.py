"""Observe paired, real token-by-token Mamba recurrence from decoded artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .basis_discovery import save_json
from .full22.data import ROOT, sha256
from .mamba_runtime import load, states

PREFIXES = (64, 256, 1024)
KINDS = {"recurrent_states": "ssm", "conv_states": "conv"}


def write_arrays(path, arrays):
    with Path(path).open("xb") as stream:
        np.savez(stream, **arrays)
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def energy(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("unaligned state tensors")
    a, b = reference.double(), candidate.double()
    result = ((a-b).square().sum(dim=(1, 2)), a.square().sum(dim=(1, 2)), b.square().sum(dim=(1, 2)))
    if any(not bool(torch.isfinite(v).all()) for v in result):
        raise ValueError("nonfinite state energy")
    return [v.cpu().numpy().copy() for v in result]


def source_consistency(path, ids, root):
    model, verification = load(path)
    inputs = torch.tensor([ids[:1024]], dtype=torch.long)
    started = time.perf_counter()
    with torch.inference_mode():
        full = model(inputs, use_cache=True)
        expected_states = states(full.cache_params, model.config)
        cache, maximum = None, 0.
        for index in range(1024):
            output = model(inputs[:, index:index+1], cache_params=cache, use_cache=True)
            cache = output.cache_params
            difference = (output.logits[0, 0] - full.logits[0, index]).abs()
            if not bool(torch.isfinite(difference).all()):
                raise ValueError("nonfinite source full/stream comparison")
            maximum = max(maximum, float(difference.max()))
        actual_states = states(cache, model.config)
        errors = {}
        for kind in KINDS:
            e, reference, _ = energy(expected_states[kind], actual_states[kind])
            errors[kind] = float(e.sum() / max(reference.sum(), 1e-30))
    passed = maximum <= .01 and all(v <= 1e-8 for v in errors.values())
    result = {"status": "MEASURED" if passed else "FAILED", "tokens": 1024,
              "max_logit_difference": maximum, "relative_state_mse": errors,
              "model_sha256": sha256(path), "verified_tensors": verification["verified_tensors"],
              "seconds_including_observation": time.perf_counter()-started}
    save_json(root / "source-consistency.json", result)
    if not passed:
        raise ValueError("source full/stream gate failed")
    return result


def summaries(trace):
    result = {}
    for length in PREFIXES:
        row = {"tokens_per_sequence": length,
               "teacher_kl": float(trace["teacher_kl"][:, :length].mean()),
               "top1_agreement": float(trace["top1_match"][:, :length].mean()),
               "source_ppl": float(np.exp(trace["source_nll"][:, :length].mean())),
               "candidate_ppl": float(np.exp(trace["candidate_nll"][:, :length].mean()))}
        for prefix in KINDS.values():
            numerator, denominator = trace[prefix+"_error"][:, :length], trace[prefix+"_reference"][:, :length]
            row[prefix+"_nmse"] = float(numerator.sum()/max(denominator.sum(), 1e-30))
            row[prefix+"_per_layer_nmse"] = (numerator.sum(axis=(0, 1))/np.maximum(denominator.sum(axis=(0, 1)), 1e-30)).tolist()
            row[prefix+"_per_sequence_nmse"] = (numerator.sum(axis=(1, 2))/np.maximum(denominator.sum(axis=(1, 2)), 1e-30)).tolist()
        result[str(length)] = row
    result["tail_256_ssm_nmse"] = float(trace["ssm_error"][:, 768:].sum()/max(trace["ssm_reference"][:, 768:].sum(), 1e-30))
    result["teacher_kl"] = float(trace["teacher_kl"].mean())
    result["finite"] = bool(all(np.isfinite(v).all() for v in trace.values()))
    return result


def paired(source, candidate, sequences, root, reset=False):
    root.mkdir(exist_ok=False)
    teacher, source_receipt = load(source)
    student, candidate_receipt = load(candidate)
    if teacher.config.to_dict() != student.config.to_dict():
        raise ValueError("paired model configurations differ")
    layers = teacher.config.num_hidden_layers
    trace = {f"{prefix}_{quantity}": np.empty((4, 1024, layers), np.float64)
             for prefix in KINDS.values() for quantity in ("error", "reference", "candidate")}
    trace.update({key: np.empty((4, 1024), np.float64) for key in ("teacher_kl", "source_nll", "candidate_nll", "top1_match")})
    checkpoints = []
    started = time.perf_counter()
    with torch.inference_mode():
        for sequence_index, ids in enumerate(sequences):
            teacher_cache, student_cache = None, None
            for index in range(1024):
                token = torch.tensor([[ids[index]]], dtype=torch.long)
                a = teacher(token, cache_params=teacher_cache, use_cache=True)
                b = student(token, cache_params=None if reset else student_cache, use_cache=True)
                teacher_cache, student_cache = a.cache_params, b.cache_params
                original_states, candidate_states = states(teacher_cache, teacher.config), states(student_cache, student.config)
                for kind, prefix in KINDS.items():
                    measures = energy(original_states[kind], candidate_states[kind])
                    for quantity, values in zip(("error", "reference", "candidate"), measures):
                        trace[f"{prefix}_{quantity}"][sequence_index, index] = values
                pa = torch.log_softmax(a.logits[0, 0].double(), dim=-1)
                pb = torch.log_softmax(b.logits[0, 0].double(), dim=-1)
                divergence = float((pa.exp() * (pa-pb)).sum())
                if not bool(torch.isfinite(pa).all()) or not bool(torch.isfinite(pb).all()) or divergence < -1e-10:
                    raise ValueError("invalid actual model log probabilities")
                trace["teacher_kl"][sequence_index, index] = max(0., divergence)
                trace["source_nll"][sequence_index, index] = -float(pa[ids[index+1]])
                trace["candidate_nll"][sequence_index, index] = -float(pb[ids[index+1]])
                trace["top1_match"][sequence_index, index] = int(pa.argmax() == pb.argmax())
                if index+1 in PREFIXES:
                    arrays = {f"{role}_{KINDS[k]}": value.cpu().numpy()
                              for role, snapshot in (("source", original_states), ("candidate", candidate_states))
                              for k, value in snapshot.items()}
                    arrays.update(source_logits=a.logits[0, 0].cpu().numpy(), candidate_logits=b.logits[0, 0].cpu().numpy())
                    receipt = write_arrays(root / f"sequence-{sequence_index}-prefix-{index+1}.npz", arrays)
                    checkpoints.append(dict(receipt, sequence=sequence_index, prefix=index+1))
                if (index+1) % 128 == 0:
                    print(json.dumps({"model": root.name, "sequence": sequence_index, "tokens": index+1}), flush=True)
    result = {"status": "MEASURED", "source_sha256": sha256(source), "candidate_sha256": sha256(candidate),
              "source_verified_tensors": source_receipt["verified_tensors"],
              "candidate_verified_tensors": candidate_receipt["verified_tensors"],
              "reset_state_control": reset, "summary": summaries(trace), "checkpoints": checkpoints,
              "trace": write_arrays(root / "trace.npz", trace),
              "seconds_including_teacher_and_observation": time.perf_counter()-started}
    if not result["summary"]["finite"]:
        raise ValueError("nonfinite complete trajectory")
    save_json(root / "summary.json", result)
    return result


def _run(models_path, sequences_path, root):
    torch.set_num_threads(4)
    models_path, sequences_path = Path(models_path).resolve(), Path(sequences_path).resolve()
    models = json.loads(models_path.read_text())["models"]
    sequence_record = json.loads(sequences_path.read_text())
    sequences = sequence_record["sequences"]
    if (len(sequences) != 4 or any(len(s) != 1025 or any(type(i) is not int or not 0 <= i < 50280 for i in s) for s in sequences)):
        raise ValueError("M1 requires four complete pretokenized sequences")
    frozen = {str(p): sha256(p) for p in (models_path, sequences_path, Path(__file__), ROOT/"experiments/mamba_runtime.py")}
    for model in models.values():
        if sha256(model["decoded"]) != model["decoded_sha256"]:
            raise ValueError("trajectory model identity changed")
        frozen[model["decoded"]] = model["decoded_sha256"]
    save_json(root / "manifest.json", {"sha256": frozen})
    blocked = [os.path.realpath(ROOT / "models/mamba-130m-hf"),
               os.path.realpath(ROOT / "artifacts/mamba-temporal-2026-10-01/preflight/source.gguf")]
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if any(path == p or path.startswith(p+os.sep) for p in blocked):
                raise PermissionError("original Mamba inputs are forbidden during standalone trajectories")
    sys.addaudithook(audit)
    denied_probes = []
    for path in (ROOT / "models/mamba-130m-hf/config.json", Path(blocked[1])):
        try:
            with path.open("rb"):
                pass
        except PermissionError:
            denied_probes.append(str(path))
        else:
            raise ValueError("standalone source-access guard did not reject the probe")
    source = models["source-lossless"]["decoded"]
    consistency = source_consistency(source, sequences[0], root)
    results = {}
    for name in ("source-reset", "q4-scalar", "q6-scalar", "q6-functional", "q6-functional-selectors-exact"):
        candidate = source if name == "source-reset" else models[name]["decoded"]
        results[name] = paired(source, candidate, sequences, root/name, reset=name == "source-reset")
        save_json(root / "progress.json", results)
    if results["source-reset"]["summary"]["teacher_kl"] <= 0:
        raise ValueError("reset-state negative control did not differ")
    if any(sha256(p) != h for p, h in frozen.items()):
        raise ValueError("trajectory inputs or implementation changed")
    result = {"status": "MEASURED", "source_consistency": consistency, "models": results,
              "original_checkpoint_and_source_gguf_opens_denied": True, "denied_probes": denied_probes}
    save_json(root / "summary.json", result)
    return result


def run(models_path, sequences_path, out):
    root = Path(os.path.abspath(out))
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run(models_path, sequences_path, root)
    except Exception as error:
        save_json(root / "failure.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.models, args.sequences, args.out)
