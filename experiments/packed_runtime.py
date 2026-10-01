"""K1: real packed operators, isolated residency and matched dense controls."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import gguf
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .basis_discovery import save_json
from .full22.catalog import TENSORS
from .full22.data import ROOT, SOURCE, load_calibration, sha256
from .full22.harness import inputs_for

ARCHIVE = ROOT/"artifacts/six-bit-frontier-2026-10-01/activation/model.a22"
DECODED = ROOT/"artifacts/six-bit-frontier-2026-10-01/activation/model.gguf"
CALIBRATION = ROOT/"artifacts/calibration-coverage-2026-10-01/captures/stratified0-selection.acal"
PROBE = ROOT/"target/packed-runtime-current/release/examples/packed_probe"
BINARY = ROOT/"target/packed-runtime-current/release/atom-quantizer"
PROTOCOL = ROOT/"research/packed-runtime-2026-10-01/PROTOCOL.md"


def execute(command, result, log):
    with log.open("x") as stderr:
        completed = subprocess.run(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, timeout=3600)
    result.with_suffix(".stdout").write_text(completed.stdout)
    if completed.returncode != 0:
        raise ValueError(f"execution failed ({completed.returncode}): {log}")
    value = json.loads(completed.stdout)
    save_json(result, value)
    return value


def blas(tensor, input_path, reference_path, output):
    threadpool_limits(1)
    reader = gguf.GGUFReader(DECODED)
    weight = next(t.data for t in reader.tensors if t.name == tensor).copy(order="C")
    values = np.fromfile(input_path, dtype="<f4").reshape(-1, weight.shape[1])
    reference = np.fromfile(reference_path, dtype="<f4").reshape(len(values), len(weight))
    times, errors, hashes = [], [], []
    for iteration in range(13):
        start = time.perf_counter_ns()
        result = values @ weight.T
        elapsed = (time.perf_counter_ns()-start)*1e-9
        if not np.isfinite(result).all():
            raise ValueError("nonfinite BLAS output")
        error = float(np.sum((result.astype(np.float64)-reference)**2)/max(np.sum(reference.astype(np.float64)**2), 1e-30))
        if error > 1e-10:
            raise ValueError("dense BLAS arithmetic exceeds the fixed comparison bound")
        if iteration >= 2:
            times.append(elapsed)
            errors.append(error)
            import hashlib
            hashes.append(hashlib.sha256(result.astype("<f4").tobytes()).hexdigest())
    pools = threadpool_info()
    if any(x["num_threads"] != 1 for x in pools if x["user_api"] == "blas"):
        raise ValueError("dense BLAS was not single-threaded")
    value = {"status": "MEASURED", "tensor": tensor, "batch": len(values), "shape": list(weight.shape),
             "input_sha256": sha256(input_path), "reference_sha256": sha256(reference_path),
             "decoded_model_sha256": sha256(DECODED), "timings_seconds": times, "relative_mse": errors,
             "output_sha256": hashes, "blas": pools, "cpu_affinity": sorted(os.sched_getaffinity(0))}
    save_json(Path(output), value)
    return value


def summarize(results, memory):
    rows = []
    ratios = {str(batch): {mode: {"native": [], "blas": []} for mode in ("fused", "row")} for batch in (1, 8, 32)}
    for item in results:
        native, dense = item["native"], item["blas"]
        if any(len(v) != 11 for v in native["timings_seconds"].values()) or len(dense["timings_seconds"]) != 11:
            raise ValueError("missing measured timing samples")
        medians = {name: statistics.median(values) for name, values in native["timings_seconds"].items()}
        medians["blas"] = statistics.median(dense["timings_seconds"])
        fastest_native = min(medians["dense_shared"], medians["dense_existing"])
        for mode in ("fused", "row"):
            ratios[str(native["batch"])][mode]["native"].append(medians[mode]/fastest_native)
            ratios[str(native["batch"])][mode]["blas"].append(medians[mode]/medians["blas"])
        rows.append({"tensor": native["tensor"], "batch": native["batch"], "medians_seconds": medians,
                     "minimum_maximum_seconds": {name: [min(v), max(v)] for name, v in native["timings_seconds"].items()},
                     "blas_minimum_maximum_seconds": [min(dense["timings_seconds"]), max(dense["timings_seconds"])],
                     "packed_owned_bytes": native["packed_owned_bytes"], "dense_owned_bytes": native["dense_owned_bytes"]})
    for modes in ratios.values():
        for values in modes.values():
            for baseline, numbers in list(values.items()):
                ratio = statistics.geometric_mean(numbers)
                values[baseline] = {"geometric_mean_latency_ratio": ratio, "status": "MEASURED" if ratio <= .9 else "FAILED"}
    packed = memory["packed"]["measurements"]["loaded"]["VmRSS"]
    dense = memory["dense"]["measurements"]["loaded"]["VmRSS"]
    return {"operator_cases": rows, "G3": ratios,
            "G2": {"status": "MEASURED" if packed <= .4*dense else "FAILED", "packed_loaded_rss_kib": packed,
                   "dense_loaded_rss_kib": dense, "ratio": packed/dense}}


def _run(root):
    original_affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {original_affinity[0]})
    threadpool_limits(1)
    expected = {ARCHIVE: "27d36037a432c5cefeeb0c1f62505377f469365027e00372d30b1840c4608a83",
                DECODED: "4c9d806f45ec68dda196a966421857a7ff6d21c96e1dfe2393b378c82a2552d8",
                CALIBRATION: "8e357a6712d5d288aa53a2e088f574c26246868db14eb864fd2fb73c9b3e4d99"}
    if any(sha256(p) != h for p, h in expected.items()):
        raise ValueError("precommitted model or actual inputs changed")
    pools = threadpool_info()
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable")
    runtime = {"platform": platform.platform(), "processor": cpu, "affinity_before": original_affinity,
               "affinity": sorted(os.sched_getaffinity(0)), "blas": pools,
               "rustc": subprocess.check_output(["rustc", "-vV"], text=True), "python": sys.version,
               "RUSTFLAGS": os.environ.get("RUSTFLAGS"), "numpy": np.__version__}
    save_json(root/"runtime.json", runtime)
    files = [ARCHIVE, DECODED, CALIBRATION, SOURCE, PROBE, BINARY, PROTOCOL, Path(__file__),
             ROOT/"Cargo.toml", ROOT/"Cargo.lock", ROOT/"src/main.rs", ROOT/"src/packed_apply.rs",
             ROOT/"src/a22.rs", ROOT/"examples/packed_probe.rs", ROOT/"experiments/full22/data.py",
             ROOT/"experiments/full22/harness.py", ROOT/"experiments/full22/catalog.py"]
    files += list((ROOT/"src/a22").glob("*.rs"))
    files += [Path(p["filepath"]) for p in pools]
    frozen = {str(p): sha256(p) for p in files}
    save_json(root/"manifest.json", {"sha256": frozen, "timing_tensors": TENSORS, "batches": [1, 8, 32]})
    print("CHECK ALL REAL MATRICES", flush=True)
    correctness = execute([PROBE, "correctness", ARCHIVE, CALIBRATION, SOURCE], root/"correctness.json", root/"correctness.log")
    if correctness["matrices"] != 211 or correctness["tensors_verified"] != 272 or len(correctness["cases"]) != 633:
        raise ValueError("actual correctness scope incomplete")
    samples = load_calibration(CALIBRATION, SOURCE)
    inputs = root/"inputs"
    inputs.mkdir()
    cases = []
    for index, name in enumerate(TENSORS):
        values = inputs_for(samples, name)
        for batch in (1, 8, 32):
            path = inputs/f"tensor-{index}-batch-{batch}.f32"
            with path.open("xb") as stream: stream.write(values[:batch].astype("<f4").tobytes())
            cases.append((index, name, batch, path))
    memory = {}
    for mode in ("packed", "dense"):
        print("RESIDENT", mode, flush=True)
        memory[mode] = execute([PROBE, "resident", mode, ARCHIVE, TENSORS[0], cases[0][3], "1"],
                               root/f"memory-{mode}.json", root/f"memory-{mode}.log")
        if memory[mode]["measurements"]["tensors"] != 272:
            raise ValueError("incomplete model residency")
    if memory["packed"]["measurements"]["output_sha256"] != memory["dense"]["measurements"]["output_sha256"]:
        raise ValueError("resident process actual operations differ")
    results = []
    # Deterministic case shuffle reduces monotonic size/thermal ordering.
    np.random.default_rng(2201).shuffle(cases)
    for index, name, batch, path in cases:
        print("TIMING", name, batch, flush=True)
        destination = root/f"tensor-{index}-batch-{batch}"
        native = execute([PROBE, "operator", ARCHIVE, name, path, str(batch), destination],
                         root/f"tensor-{index}-batch-{batch}.json", root/f"tensor-{index}-batch-{batch}.log")
        command = [sys.executable, "-m", "experiments.packed_runtime", "--blas", "--tensor", name,
                   "--input", str(path), "--reference", str(destination/"dense_existing.f32"),
                   "--out", str(destination/"blas.json")]
        with (destination/"blas.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        dense = json.loads((destination/"blas.json").read_text())
        results.append({"native": native, "blas": dense})
        save_json(root/"progress.json", results)
    if any(sha256(p) != h for p, h in frozen.items()):
        raise ValueError("frozen K1 inputs or implementation changed")
    summary = {"status": "MEASURED", "correctness_cases": 633, "memory": memory,
               **summarize(results, memory), "whole_model_inference_benchmarked": False}
    save_json(root/"summary.json", summary)
    print(json.dumps({"status": summary["status"], "G2": summary["G2"], "G3": summary["G3"]}), flush=True)
    return summary


def run(out):
    root = Path(os.path.abspath(out))
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _run(root)
    except Exception as error:
        save_json(root/"failure.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--blas", action="store_true")
    parser.add_argument("--tensor")
    parser.add_argument("--input")
    parser.add_argument("--reference")
    args = parser.parse_args()
    if args.blas:
        blas(args.tensor, args.input, args.reference, args.out)
    else:
        run(args.out)
