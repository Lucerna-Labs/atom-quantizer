"""Observe cancellation and shared error directions; never correct or select weights.

Conventional primitive-toolkit instrumentation, not an Atom emergence claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .full22 import codec
from .full22.data import SOURCE, OUT, load_calibration, sha256
from .full22.harness import inputs_for
from .full22.records import Record


def energy(values):
    return float(np.sum(values * values, dtype=np.float64) / values.shape[1])


def relative(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def functional_error_profile(reference, reconstruction, fit, selection, block=32):
    w, q, xf, xv = [np.asarray(v, np.float32) for v in (reference, reconstruction, fit, selection)]
    if (w.ndim != 2 or q.shape != w.shape or xf.ndim != 2 or xv.ndim != 2
            or not w.size or not xf.size or not xv.size or xf.shape[1] != w.shape[1]
            or xv.shape[1] != w.shape[1] or type(block) is not int or block <= 0
            or any(not np.isfinite(v).all() for v in (w, q, xf, xv))):
        raise ValueError("need finite nonempty aligned weight and activation matrices")
    error = w.astype(np.float64) - q.astype(np.float64)
    fit_error = error @ xf.astype(np.float64).T
    linear_error = error @ xv.astype(np.float64).T
    with np.errstate(over="ignore", invalid="ignore"):
        reference_output = (w @ xv.T).astype(np.float64)
        actual_error = reference_output - (q @ xv.T).astype(np.float64)
    if any(not np.isfinite(v).all() for v in (reference_output, actual_error, fit_error, linear_error)):
        raise ValueError("nonfinite operator evaluation")
    total = energy(linear_error)
    diagonal = float(np.sum(error * error * np.mean(xv.astype(np.float64)**2, axis=0)))
    blocks = [energy(error[:, i:i+block] @ xv[:, i:i+block].astype(np.float64).T)
              for i in range(0, w.shape[1], block)]
    # Directions come only from fit errors. Selection data measures their transfer.
    u, singular, _ = np.linalg.svd(fit_error, full_matrices=False)
    tolerance = (singular[0] if len(singular) else 0) * max(fit_error.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular > tolerance))
    modes = []
    for count in (1, 2, 4, 8, 16):
        k = min(count, rank)
        basis = u[:, :k]
        captured = energy(basis.T @ linear_error) if k else 0.0
        fit_captured = float(np.sum(singular[:k]**2) / fit_error.shape[1])
        modes.append({"requested_rank": count, "measured_rank": k,
            "fit_error_energy_fraction": relative(fit_captured, energy(fit_error)),
            "selection_error_energy_fraction": relative(captured, total),
            "f32_factor_payload_bytes_without_framing": 4*k*(w.shape[0]+w.shape[1])})
    return {"weight_error_energy": float(np.sum(error**2)),
        "selection_output_error_energy": total, "diagonal_error_prediction": diagonal,
        "correlation_cross_term": total-diagonal,
        "output_to_diagonal_ratio": relative(total, diagonal),
        "sum_isolated_block_energies": sum(blocks),
        "cross_block_interference": total-sum(blocks), "block_error_energies": blocks,
        "actual_f32_output_nmse": relative(energy(actual_error), energy(reference_output)),
        "matmul_rounding_residual_energy": energy(actual_error-linear_error),
        "fit_error_rank": rank, "shared_error_modes": modes,
        "scope": "diagnostic only; no corrected model or compression benefit produced"}


def analyze_runs(runs, source=SOURCE, fit_path=OUT/"fit.acal", selection_path=OUT/"selection.acal"):
    source_hash, fit_hash, selection_hash = sha256(source), sha256(fit_path), sha256(selection_path)
    fit = load_calibration(fit_path, source)
    selection = load_calibration(selection_path, source)
    reader = gguf.GGUFReader(source)
    tensors = {t.name: t for t in reader.tensors}
    results = []
    for run in map(Path, runs):
        manifest = json.loads((run/"manifest.json").read_text())
        if (manifest["source_sha256"], manifest["fit_sha256"], manifest["selection_sha256"]) != (source_hash, fit_hash, selection_hash):
            raise ValueError("diagnostic inputs differ from the experiment")
        sampled = {}
        for path in sorted(run.glob("*.json")):
            receipt = json.loads(path.read_text())
            if receipt.get("arm") not in ("full", "scalar") or receipt.get("status") != "MEASURED":
                continue
            artifact = path.with_suffix(".ar")
            if sha256(artifact) != receipt["artifact_sha256"]:
                raise ValueError("saved experiment artifact changed")
            name = receipt["tensor"]
            if name not in sampled:
                full = np.asarray(tensors[name].data, np.float32)
                indices = np.linspace(0, len(full)-1, min(manifest["rows"], len(full))).astype(np.int64)
                sampled[name] = np.ascontiguousarray(full[indices])
            decoded = codec.decode(Record.loads(artifact.read_bytes()))
            profile = functional_error_profile(sampled[name], decoded, inputs_for(fit, name), inputs_for(selection, name))
            np.testing.assert_allclose(profile["actual_f32_output_nmse"], receipt["selection"]["output_nmse"], rtol=1e-12, atol=1e-15)
            identity = {k:receipt.get(k) for k in ("tensor", "arm", "order", "seed", "bits", "readout")}
            results.append({**identity, "experiment": manifest["experiment"], "receipt": str(path),
                            "artifact_sha256": receipt["artifact_sha256"], "profile": profile})
    if not results:
        raise ValueError("no measured principal/reference records found")
    if (sha256(source), sha256(fit_path), sha256(selection_path)) != (source_hash, fit_hash, selection_hash):
        raise ValueError("diagnostic inputs changed during execution")
    return {"source_sha256": source_hash, "fit_sha256": fit_hash, "selection_sha256": selection_hash,
        "implementation_sha256": sha256(Path(__file__)), "profiles": len(results), "results": results}


def save_new(path, result):
    path = Path(path)
    if os.path.lexists(path):
        raise ValueError("diagnostic destination exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--fit", default=str(OUT/"fit.acal"))
    parser.add_argument("--selection", default=str(OUT/"selection.acal"))
    args = parser.parse_args()
    threadpool_limits(4)
    result = analyze_runs(args.run, args.source, args.fit, args.selection)
    save_new(args.out, result)
    print(json.dumps({"profiles": result["profiles"], "output": args.out}), flush=True)
