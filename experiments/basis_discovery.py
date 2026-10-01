"""B01 observer: run native mechanics, serialize codecs, retain every outcome."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .full22 import codec, transforms
from .full22.catalog import TENSORS
from .full22.data import ROOT, OUT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record
from .full22.screen_cache import atomic_write

ARMS = {"full": 0, "no_h": 1, "no_q": 2, "no_chemistry": 4, "no_memory": 8,
    "no_information": 16, "no_physics": 32, "no_rotor": 64, "no_population": 128,
    "no_base": 3, "no_chaos": 192, "base_only": 252, "identity": 255}
PROTOCOL = ROOT / "research/basis-discovery-2026-09-30/PROTOCOL.md"
BINARY = ROOT / "target/release/examples/discover_basis"


def save_json(path, value):
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode())


def manifest():
    paths = [PROTOCOL, BINARY, Path(__file__)]
    paths += sorted((ROOT / "experiments/full22").glob("*.py"))
    for name in ["basis-primitives", "basis-engine"]:
        paths += sorted((ROOT / "crates" / name).rglob("*.rs"))
        paths += [ROOT / "crates" / name / "Cargo.toml"]
    paths += [ROOT / "examples/discover_basis.rs"]
    return {"experiment": "B01", "expected_records": 1280, "rows": 256,
        "tensors": TENSORS, "arms": ARMS, "seeds": [1, 2, 3], "bits": [3, 4],
        "source_sha256": sha256(SOURCE), "fit_sha256": sha256(OUT / "fit.acal"),
        "selection_sha256": sha256(OUT / "selection.acal"),
        "implementation": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def native_basis(signal, order, seed, disabled):
    result = subprocess.run([str(BINARY), str(order), str(seed), str(disabled)],
        input=" ".join(format(v, ".17g") for v in signal), text=True,
        capture_output=True, timeout=30, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        raise ValueError("unexpected native response")
    matrix = np.asarray([float(v) for v in lines[0].split()], np.float64).reshape(32, 32)
    diagnostics = np.asarray([float(v) for v in lines[1].split()], np.float64)
    if (diagnostics.shape != (6,) or not np.isfinite(diagnostics).all()
            or diagnostics[0] > 1e-10 or abs(diagnostics[1] - 1) > 1e-10
            or np.any(diagnostics[2:] < 0) or np.any(diagnostics[2:] > 1)):
        raise ValueError("native invariant failure")
    return matrix.astype(np.float32), diagnostics.tolist()


def measured_record(weight, x, xv, config, path):
    started = time.perf_counter()
    record = codec.encode(weight, x, config)
    raw, compressed = record.dumps(), record.dumps(True)
    atomic_write(path, compressed)
    decoded = codec.decode(Record.loads(path.read_bytes()))
    repeated = codec.decode(Record.loads(path.read_bytes()))
    if not np.array_equal(decoded, repeated):
        raise ValueError("nonrepeatable saved-record decode")
    return {"status": "MEASURED", "raw_bytes": len(raw), "archive_bytes": len(compressed),
        "raw_bits_per_weight": 8 * len(raw) / weight.size,
        "artifact_sha256": sha256(path), "fit": codec.distortion(weight, decoded, x),
        "selection": codec.distortion(weight, decoded, xv), "seconds": time.perf_counter() - started}


def summarize(records, output, frozen, arms=ARMS, contribution_arms=None,
              tensor_reference="signed_h32", current_manifest=manifest,
              reference_arms=("scalar", "signed_h32"), observation_path="basis_path", total_records=None):
    measured = [r for r in records if r["status"] == "MEASURED"]
    candidates = []
    for order in range(2):
        for bits in [3, 4]:
            groups = {arm: [r for r in measured if r["order"] == order and r["bits"] == bits
                            and r["arm"] == arm] for arm in arms}
            refs = {arm: [r for r in measured if r["order"] == -1 and r["bits"] == bits
                          and r["arm"] == arm] for arm in reference_arms}
            entry = {"order": order, "bits": bits, "status": "INCONCLUSIVE"}
            if any(len(g) != 24 for g in groups.values()) or any(len(g) != 8 for g in refs.values()):
                entry["reason"] = "a required record failed or is absent"
                candidates.append(entry)
                continue
            means = {name: float(np.mean([r["selection"]["output_nmse"] for r in group]))
                     for name, group in {**groups, **refs}.items()}
            bytes_ratio = sum(r["raw_bytes"] for r in groups["full"]) / (
                3 * min(sum(r["raw_bytes"] for r in group) for group in refs.values()))
            ratios = {name: means["full"] / score for name, score in means.items() if name != "full"}
            tensor_ratios = {}
            for tensor in TENSORS:
                full = np.mean([r["selection"]["output_nmse"] for r in groups["full"] if r["tensor"] == tensor])
                reference = next(r["selection"]["output_nmse"] for r in refs[tensor_reference] if r["tensor"] == tensor)
                tensor_ratios[tensor] = float(full / reference)
            contributions = {}
            for arm in (contribution_arms or ["no_h", "no_q", "no_base", "no_chemistry", "no_memory", "no_information", "no_physics"]):
                checks = []
                for seed in [1, 2, 3]:
                    changes = []
                    for full in [r for r in groups["full"] if r["seed"] == seed]:
                        control = next(r for r in groups[arm] if r["seed"] == seed and r["tensor"] == full["tensor"])
                        a, b = np.load(full[observation_path]), np.load(control[observation_path])
                        distance = float(np.linalg.norm(a.astype(float) - b) / np.linalg.norm(a))
                        changes.append(distance > 1e-7 and full["selection"] != control["selection"])
                    checks.append(any(changes))
                contributions[arm] = all(checks)
            benefit = all(ratio < 1 for ratio in ratios.values()) and bytes_ratio <= 1.1 and max(tensor_ratios.values()) <= 1.1
            gates = {"G1_native": True, "G2_storage": True, "G3_benefit": benefit,
                     "G4_contributions": all(contributions.values()),
                     "G5_complete": (len(records) if total_records is None else total_records) == frozen["expected_records"] and current_manifest() == frozen}
            entry.update(status="MEASURED" if all(gates.values()) else "FAILED", gates=gates,
                mean_selection_output_nmse=means, full_to_control_ratios=ratios,
                complete_raw_bytes_ratio=bytes_ratio, tensor_to_reference_ratios=tensor_ratios,
                tensor_reference=tensor_reference,
                contributions=contributions)
            candidates.append(entry)
    summary = {"experiment": frozen["experiment"], "scope": "real-tensor screen, not full-model inference",
        "records": len(records), "measured": len(measured), "failed_records": len(records)-len(measured),
        "candidates": candidates, "promoted": [], "eligible_for_model_trials":
        [{"order": c["order"], "bits": c["bits"]} for c in candidates if c["status"] == "MEASURED"]}
    save_json(output / "summary.json", summary)
    return summary


def run(output):
    threadpool_limits(4)
    output = Path(output).resolve()
    frozen = manifest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != frozen:
            raise ValueError("B01 inputs changed; preserve this run and use a new directory")
    else:
        save_json(manifest_path, frozen)
    fit = load_calibration(OUT / "fit.acal", SOURCE)
    selection = load_calibration(OUT / "selection.acal", SOURCE)
    reader = gguf.GGUFReader(SOURCE)
    tensors = {t.name: t for t in reader.tensors}
    records = []

    def trial(identity, weight, x, xv, config, extra):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt, artifact = output / (key + ".json"), output / (key + ".ar")
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if any(result.get(k) != v for k, v in identity.items()):
                raise ValueError("receipt identity mismatch")
            if result["status"] == "MEASURED":
                if sha256(artifact) != result["artifact_sha256"]:
                    raise ValueError("retained B01 artifact changed")
                codec.decode(Record.loads(artifact.read_bytes()))
        else:
            result = dict(identity, config=config, **extra)
            try:
                if "basis_error" in extra:
                    raise ValueError(extra["basis_error"])
                result.update(measured_record(weight, x, xv, config, artifact))
            except Exception as error:
                result.update(status="FAILED", error=repr(error))
            save_json(receipt, result)
        records.append(result)

    for tensor_name in TENSORS:
        full = np.asarray(tensors[tensor_name].data, np.float32)
        indices = np.linspace(0, len(full)-1, min(256, len(full))).astype(np.int64)
        weight = np.ascontiguousarray(full[indices])
        x, xv = inputs_for(fit, tensor_name), inputs_for(selection, tensor_name)
        if x is None or xv is None:
            raise ValueError("B01 requires real fit and selection inputs")
        for bits in [3, 4]:
            for arm, settings in [("scalar", {}), ("signed_h32", {"rotation": 32, "signed": True, "seed": 0})]:
                trial({"tensor": tensor_name, "order": -1, "seed": 0, "bits": bits, "arm": arm},
                      weight, x, xv, {"bits": bits, **settings}, {})
        signal = weight.astype(np.float64).reshape(-1, 32).mean(axis=0)
        for order in range(2):
            for arm, disabled in ARMS.items():
                for seed in [1, 2, 3]:
                    basis_name = f"{tensor_name}.o{order}.{arm}.s{seed}.npy"
                    basis_path = output / basis_name
                    extra = {"basis_path": str(basis_path)}
                    try:
                        matrix, diagnostics = native_basis(signal, order, seed, disabled)
                        if basis_path.exists():
                            if not np.array_equal(np.load(basis_path), matrix):
                                raise ValueError("native transport changed on repetition")
                        else:
                            with basis_path.open("xb") as f:
                                np.save(f, matrix, allow_pickle=False)
                        recovered = transforms.basis_apply(transforms.basis_apply(weight, matrix), matrix, True)
                        nmse = codec.distortion(weight, recovered)["weight_nmse"]
                        if nmse > 1e-11:
                            raise ValueError("saved basis unquantized reconstruction failed")
                        extra.update(native_diagnostics=diagnostics, unquantized_nmse=nmse,
                                     basis_sha256=sha256(basis_path))
                    except Exception as error:
                        extra["basis_error"] = repr(error)
                    for bits in [3, 4]:
                        trial({"tensor": tensor_name, "order": order, "seed": seed, "bits": bits, "arm": arm},
                              weight, x, xv, {"bits": bits, "basis_path": str(basis_path)}, extra)
        print(f"{tensor_name}: {len(records)}/1280 records; {sum(r['status'] != 'MEASURED' for r in records)} failed", flush=True)
    if manifest() != frozen:
        save_json(output / "invalidated.json", {"reason": "inputs changed during trial"})
        raise ValueError("B01 inputs changed during trial")
    return summarize(records, output, frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out), indent=2, allow_nan=False), flush=True)
