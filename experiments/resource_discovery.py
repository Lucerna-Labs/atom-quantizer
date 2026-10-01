"""B03: observe local native resource fields through reversible codec scales."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from . import basis_discovery as shared
from .full22 import codec, transforms
from .full22.catalog import TENSORS
from .full22.data import ROOT, OUT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record
from .local_basis_discovery import ARMS as LOCAL_ARMS

ARMS = {**LOCAL_ARMS, "folded_drive": 0}
READOUTS = ("conductance", "resistance")
REFERENCES = {"scalar": {}, "signed_h32": {"rotation": 32, "signed": True, "seed": 0},
              "diagonal": {"diagonal": 0.5}}
PROTOCOL = ROOT / "research/resource-discovery-2026-09-30/PROTOCOL.md"
BINARY = ROOT / "target/release/examples/discover_resources"


def manifest():
    paths = [PROTOCOL, BINARY, Path(__file__), Path(shared.__file__), ROOT / "experiments/local_basis_discovery.py"]
    paths += sorted((ROOT / "experiments/full22").glob("*.py"))
    for name in ["basis-primitives", "basis-engine"]:
        paths += sorted((ROOT / "crates" / name).rglob("*.rs"))
        paths += [ROOT / "crates" / name / "Cargo.toml"]
    paths += [ROOT / "examples/discover_resources.rs"]
    return {"experiment": "B03", "expected_records": 3120, "expected_native_blocks": 19584,
        "rows": 256, "tensors": TENSORS, "arms": ARMS, "seeds": [1, 2, 3], "bits": [3, 4],
        "readouts": READOUTS, "references": REFERENCES,
        "source_sha256": sha256(SOURCE), "fit_sha256": sha256(OUT / "fit.acal"),
        "selection_sha256": sha256(OUT / "selection.acal"), "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "gguf", "scipy", "threadpoolctl"]},
        "implementation": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def correlation(covariance):
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    denominator = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)


def measure_fields(weight, inputs):
    groups = weight.shape[1] // 32
    weights = np.asarray(weight, np.float64).reshape(len(weight), groups, 32)
    features = np.asarray(inputs, np.float64).reshape(len(inputs), groups, 32)
    weight_energy = np.mean(weights * weights, axis=0)
    covariance = np.einsum("sgi,sgj->gij", features, features) / len(features)
    input_energy = np.diagonal(covariance, axis1=1, axis2=2)
    signals = np.sqrt(weight_energy * input_energy)
    folded_signal = np.sqrt(weight_energy.mean(axis=0) * input_energy.mean(axis=0))
    return signals, correlation(covariance), folded_signal, correlation(covariance.mean(axis=0))


def native_resources(signals, couplings, order, seed, disabled):
    payload = np.concatenate([signals, couplings.reshape(len(signals), -1)], axis=1)
    result = subprocess.run([str(BINARY), str(order), str(seed), str(disabled)],
        input=" ".join(format(v, ".17g") for v in payload.reshape(-1)),
        text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise ValueError(f"native resource execution failed: {result.stderr.strip()}")
    values = np.asarray([[float(v) for v in line.split()] for line in result.stdout.splitlines()], np.float64)
    if values.shape != (len(signals), 38) or not np.isfinite(values).all():
        raise ValueError("invalid native resource response")
    resources, diagnostics = values[:, :32], values[:, 32:]
    if (np.any(resources < 1/3 - 1e-12) or np.any(resources > 1)
            or np.max(diagnostics[:, 0]) > 1e-10 or np.max(np.abs(diagnostics[:, 1]-1)) > 1e-10
            or np.any(diagnostics[:, 2:] < 0) or np.any(diagnostics[:, 2:] > 1)):
        raise ValueError("native resource invariant failure")
    return resources.reshape(-1), diagnostics


def preserve_array(path, values):
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), values):
            raise ValueError("native array changed on repetition")
    else:
        with path.open("xb") as f:
            np.save(f, values, allow_pickle=False)


def run(output):
    threadpool_limits(4)
    output = Path(output).resolve()
    frozen = manifest()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invalidated.json").exists():
        raise ValueError("invalidated B03 run; preserve it and use a new directory")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        # Canonical JSON compares tuple-valued declarations to their saved arrays.
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("B03 inputs changed; use a new output directory")
    else:
        shared.save_json(manifest_path, frozen)
    fit, selection = load_calibration(OUT / "fit.acal", SOURCE), load_calibration(OUT / "selection.acal", SOURCE)
    reader = gguf.GGUFReader(SOURCE)
    tensors = {t.name: t for t in reader.tensors}
    records, native_blocks = [], 0

    def trial(identity, weight, x, xv, config, extra):
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt, artifact = output / (key + ".json"), output / (key + ".ar")
        if receipt.exists():
            result = json.loads(receipt.read_text())
            if any(result.get(k) != v for k, v in identity.items()):
                raise ValueError("receipt identity mismatch")
            if result["status"] == "MEASURED":
                if sha256(artifact) != result["artifact_sha256"]:
                    raise ValueError("retained B03 artifact changed")
                codec.decode(Record.loads(artifact.read_bytes()))
        else:
            result = dict(identity, config=config, **extra)
            try:
                if "native_error" in extra:
                    raise ValueError(extra["native_error"])
                result.update(shared.measured_record(weight, x, xv, config, artifact))
            except Exception as error:
                result.update(status="FAILED", error=repr(error))
            shared.save_json(receipt, result)
        records.append(result)

    for tensor_name in TENSORS:
        full = np.asarray(tensors[tensor_name].data, np.float32)
        indices = np.linspace(0, len(full)-1, min(256, len(full))).astype(np.int64)
        weight = np.ascontiguousarray(full[indices])
        x, xv = inputs_for(fit, tensor_name), inputs_for(selection, tensor_name)
        if x is None or xv is None:
            raise ValueError("B03 requires real fit and selection inputs")
        for bits in [3, 4]:
            for arm, settings in REFERENCES.items():
                trial({"tensor": tensor_name, "order": -1, "seed": 0, "bits": bits, "arm": arm, "readout": "reference"},
                      weight, x, xv, {"bits": bits, **settings}, {})
        signals, coupling, folded_signal, folded_coupling = measure_fields(weight, x)
        preserve_array(output / f"{tensor_name}.signals.npy", signals)
        preserve_array(output / f"{tensor_name}.coupling.npy", coupling)
        for order in range(2):
            for arm, disabled in ARMS.items():
                current_signal, current_coupling = signals, coupling
                if arm == "uniform_coupling":
                    current_coupling = np.ones_like(coupling)
                elif arm == "shuffled_coupling":
                    permutation = (5 * np.arange(32) + 7) % 32
                    current_coupling = coupling[:, permutation][:, :, permutation]
                elif arm == "folded_drive":
                    current_signal = np.broadcast_to(folded_signal, signals.shape)
                    current_coupling = np.broadcast_to(folded_coupling, coupling.shape)
                for seed in [1, 2, 3]:
                    base = f"{tensor_name}.o{order}.{arm}.s{seed}"
                    resource_path = output / (base + ".resource.npy")
                    native = {"resource_path": str(resource_path), "native_blocks": len(signals)}
                    native_blocks += len(signals)
                    try:
                        resources, diagnostics = native_resources(current_signal, current_coupling, order, seed, disabled)
                        preserve_array(resource_path, resources)
                        native.update(native_diagnostics=diagnostics.tolist(), resource_sha256=sha256(resource_path),
                                      resource_min=float(resources.min()), resource_max=float(resources.max()))
                    except Exception as error:
                        native["native_error"] = repr(error)
                    for readout in READOUTS:
                        scale_path = output / (base + f".{readout}.npy")
                        extra = dict(native, scale_path=str(scale_path))
                        try:
                            if "native_error" in native:
                                raise ValueError(native["native_error"])
                            values = resources if readout == "conductance" else 1 / resources
                            scale = transforms.checked_native_scale(values, weight.shape[1])
                            preserve_array(scale_path, scale)
                            recovered = (weight * scale) / scale
                            nmse = codec.distortion(weight, recovered)["weight_nmse"]
                            if nmse > 1e-11:
                                raise ValueError("native scale unquantized reconstruction failed")
                            extra.update(scale_sha256=sha256(scale_path), unquantized_nmse=nmse)
                        except Exception as error:
                            extra["native_error"] = repr(error)
                        for bits in [3, 4]:
                            trial({"tensor": tensor_name, "order": order, "seed": seed, "bits": bits, "arm": arm, "readout": readout},
                                  weight, x, xv, {"bits": bits, "native_scale_path": str(scale_path)}, extra)
        print(f"{tensor_name}: {len(records)}/3120 records, {native_blocks}/19584 native blocks; "
              f"{sum(r['status'] != 'MEASURED' for r in records)} failed", flush=True)
    if manifest() != frozen:
        shared.save_json(output / "invalidated.json", {"reason": "inputs changed during trial"})
        raise ValueError("B03 inputs changed during trial")
    if native_blocks != frozen["expected_native_blocks"]:
        raise ValueError("native block count disagrees with protocol")
    summaries = []
    for readout in READOUTS:
        directory = output / readout
        directory.mkdir(exist_ok=True)
        selected = [r for r in records if r["readout"] in ("reference", readout)]
        summary = shared.summarize(selected, directory, frozen, arms=ARMS,
            contribution_arms=[a for a in ARMS if a != "full"], tensor_reference="scalar",
            current_manifest=manifest, reference_arms=tuple(REFERENCES), observation_path="scale_path",
            total_records=len(records))
        summaries.extend(dict(c, readout=readout) for c in summary["candidates"])
    result = {"experiment": "B03", "scope": "real-tensor screen, not full-model inference",
        "records": len(records), "native_blocks": native_blocks,
        "measured": sum(r["status"] == "MEASURED" for r in records), "candidates": summaries,
        "eligible_for_model_trials": [{k:c[k] for k in ["order", "bits", "readout"]}
            for c in summaries if c["status"] == "MEASURED"], "promoted": []}
    shared.save_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out), indent=2, allow_nan=False), flush=True)
