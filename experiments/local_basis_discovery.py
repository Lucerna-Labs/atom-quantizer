"""B02 observer: calibration-aligned local transport, frozen matched controls."""
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

from . import basis_discovery as b01
from .full22 import codec, transforms
from .full22.catalog import TENSORS
from .full22.data import ROOT, OUT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record

ARMS = {"full": 0, "no_phase": 256, "no_q": 2, "no_chemistry": 4, "no_memory": 8,
    "no_information": 16, "no_physics": 32, "no_rotor": 64, "no_population": 128,
    "no_base": 258, "no_chaos": 192, "base_only": 252, "identity": 510,
    "uniform_coupling": 0, "shuffled_coupling": 0}
PROTOCOL = ROOT / "research/local-basis-2026-09-30/PROTOCOL.md"
BINARY = ROOT / "target/release/examples/discover_local_basis"


def manifest():
    paths = [PROTOCOL, BINARY, Path(__file__), Path(b01.__file__)]
    paths += sorted((ROOT / "experiments/full22").glob("*.py"))
    for name in ["basis-primitives", "basis-engine"]:
        paths += sorted((ROOT / "crates" / name).rglob("*.rs"))
        paths += [ROOT / "crates" / name / "Cargo.toml"]
    paths += [ROOT / "examples/discover_local_basis.rs"]
    return {"experiment": "B02", "expected_records": 1472, "rows": 256,
        "tensors": TENSORS, "arms": ARMS, "seeds": [1, 2, 3], "bits": [3, 4],
        "source_sha256": sha256(SOURCE), "fit_sha256": sha256(OUT / "fit.acal"),
        "selection_sha256": sha256(OUT / "selection.acal"), "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "gguf", "scipy", "threadpoolctl"]},
        "implementation": {str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def measure_inputs(weight, inputs):
    """Only RMS/correlation statistics; no quantizer or reconstruction objective."""
    weights = np.asarray(weight, np.float64).reshape(-1, 32)
    features = np.asarray(inputs, np.float64).reshape(-1, 32)
    signal = np.sqrt(np.mean(weights * weights, axis=0))
    covariance = (features.T @ features) / len(features)
    energy = np.diag(covariance)
    denominator = np.sqrt(energy[:, None] * energy[None, :])
    coupling = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    return signal, coupling


def native_basis(signal, coupling, order, seed, disabled):
    stimulus = np.concatenate([signal, coupling.reshape(-1)])
    result = subprocess.run([str(BINARY), str(order), str(seed), str(disabled)],
        input=" ".join(format(v, ".17g") for v in stimulus), text=True,
        capture_output=True, timeout=30, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        raise ValueError("unexpected local native response")
    matrix = np.asarray([float(v) for v in lines[0].split()], np.float64).reshape(32, 32)
    diagnostics = np.asarray([float(v) for v in lines[1].split()], np.float64)
    if (diagnostics.shape != (6,) or not np.isfinite(diagnostics).all()
            or diagnostics[0] > 1e-10 or abs(diagnostics[1] - 1) > 1e-10
            or np.any(diagnostics[2:] < 0) or np.any(diagnostics[2:] > 1)):
        raise ValueError("local native invariant failure")
    return matrix.astype(np.float32), diagnostics.tolist()


def run(output):
    threadpool_limits(4)
    output = Path(output).resolve()
    frozen = manifest()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invalidated.json").exists():
        raise ValueError("B02 run invalidated; preserve evidence and use a new directory")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != frozen:
            raise ValueError("B02 inputs changed; preserve this run and use a new directory")
    else:
        b01.save_json(manifest_path, frozen)
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
                    raise ValueError("retained B02 artifact changed")
                codec.decode(Record.loads(artifact.read_bytes()))
        else:
            result = dict(identity, config=config, **extra)
            try:
                if "basis_error" in extra:
                    raise ValueError(extra["basis_error"])
                result.update(b01.measured_record(weight, x, xv, config, artifact))
            except Exception as error:
                result.update(status="FAILED", error=repr(error))
            b01.save_json(receipt, result)
        records.append(result)

    for tensor_name in TENSORS:
        full = np.asarray(tensors[tensor_name].data, np.float32)
        indices = np.linspace(0, len(full)-1, min(256, len(full))).astype(np.int64)
        weight = np.ascontiguousarray(full[indices])
        x, xv = inputs_for(fit, tensor_name), inputs_for(selection, tensor_name)
        if x is None or xv is None:
            raise ValueError("B02 requires real fit and selection inputs")
        for bits in [3, 4]:
            for arm, settings in [("scalar", {}), ("signed_h32", {"rotation": 32, "signed": True, "seed": 0})]:
                trial({"tensor": tensor_name, "order": -1, "seed": 0, "bits": bits, "arm": arm},
                      weight, x, xv, {"bits": bits, **settings}, {})
        signal, coupling = measure_inputs(weight, x)
        b01.save_json(output / (tensor_name + ".inputs.json"), {"signal": signal.tolist(),
            "coupling": coupling.tolist(), "sampled_rows_sha256": hashlib.sha256(indices.tobytes()).hexdigest()})
        for order in range(2):
            for arm, disabled in ARMS.items():
                current = coupling
                if arm == "uniform_coupling":
                    current = np.ones_like(coupling)
                elif arm == "shuffled_coupling":
                    permutation = (5 * np.arange(32) + 7) % 32
                    current = coupling[np.ix_(permutation, permutation)]
                for seed in [1, 2, 3]:
                    basis_path = output / f"{tensor_name}.o{order}.{arm}.s{seed}.npy"
                    extra = {"basis_path": str(basis_path)}
                    try:
                        matrix, diagnostics = native_basis(signal, current, order, seed, disabled)
                        if basis_path.exists():
                            if not np.array_equal(np.load(basis_path), matrix):
                                raise ValueError("local transport changed on repetition")
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
        print(f"{tensor_name}: {len(records)}/1472 records; {sum(r['status'] != 'MEASURED' for r in records)} failed", flush=True)
    if manifest() != frozen:
        b01.save_json(output / "invalidated.json", {"reason": "inputs changed during trial"})
        raise ValueError("B02 inputs changed during trial")
    controls = [name for name in ARMS if name != "full"]
    return b01.summarize(records, output, frozen, arms=ARMS, contribution_arms=controls,
                         tensor_reference="scalar", current_manifest=manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.out), indent=2, allow_nan=False), flush=True)
