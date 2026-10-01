"""Independently recompute K1 real outputs, timing summaries and RSS evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

import gguf
import numpy as np
from threadpoolctl import threadpool_limits

from .basis_discovery import save_json
from .full22.data import load_calibration, sha256
from .full22.harness import inputs_for
from .packed_runtime import ARCHIVE, DECODED, CALIBRATION, SOURCE, PROBE, TENSORS


def run(root):
    root = Path(root).resolve()
    threadpool_limits(1)
    runtime = json.loads((root/"runtime.json").read_text())
    os.sched_setaffinity(0, set(runtime["affinity"]))
    frozen = json.loads((root/"manifest.json").read_text())["sha256"]
    if any(sha256(p) != h for p, h in frozen.items()):
        raise ValueError("frozen K1 implementation or input changed")
    correctness = json.loads((root/"correctness.json").read_text())
    summary = json.loads((root/"summary.json").read_text())
    cases = {(r["tensor"], r["batch"]): r for r in correctness["cases"]}
    if len(cases) != 633 or correctness["matrices"] != 211 or correctness["tensors_verified"] != 272:
        raise ValueError("incomplete actual correctness scope")
    observations = load_calibration(CALIBRATION, SOURCE)
    tensors = [t for t in gguf.GGUFReader(DECODED).tensors if t.data.ndim == 2]
    if len(tensors) != 211:
        raise ValueError("incorrect model matrix count")
    outputs = {}
    for tensor in tensors:
        weights = tensor.data
        inputs = inputs_for(observations, tensor.name)[:32].astype(np.float64)
        actual = np.zeros((32, weights.shape[0]), np.float64)
        # Independent NumPy double products, with ascending-column accumulation.
        for col in range(weights.shape[1]):
            actual += inputs[:, col, None] * weights[None, :, col].astype(np.float64)
        values = actual.astype("<f4")
        for batch in (1, 8, 32):
            digest = hashlib.sha256(values[:batch].tobytes()).hexdigest()
            row = cases[(tensor.name, batch)]
            if row["output_sha256"] != digest or not row["bit_identical"]:
                raise ValueError(f"independent actual output differs: {tensor.name}, {batch}")
            outputs[(tensor.name, batch)] = digest
        print("AUDITED", tensor.name, flush=True)
    results = json.loads((root/"progress.json").read_text())
    if len(results) != 24:
        raise ValueError("missing timing cases")
    seen, ratios = set(), {str(b): {m: {"native": [], "blas": []} for m in ("fused", "row")} for b in (1, 8, 32)}
    for entry in results:
        native, blas = entry["native"], entry["blas"]
        name, batch = native["tensor"], native["batch"]
        if (name, batch) in seen or name not in TENSORS or batch not in (1, 8, 32):
            raise ValueError("unexpected/repeated timing case")
        seen.add((name, batch))
        index = TENSORS.index(name)
        directory = root/f"tensor-{index}-batch-{batch}"
        saved = json.loads(directory.with_suffix(".json").read_text())
        if native != saved or blas != json.loads((directory/"blas.json").read_text()):
            raise ValueError("timing progress differs from raw receipts")
        input_path = root/"inputs"/f"tensor-{index}-batch-{batch}.f32"
        if native["input_sha256"] != sha256(input_path) or blas["input_sha256"] != sha256(input_path):
            raise ValueError("timed input identity differs")
        if native["archive_sha256"] != sha256(ARCHIVE) or blas["decoded_model_sha256"] != sha256(DECODED):
            raise ValueError("timed model identity differs")
        expected = outputs[(name, batch)]
        for mode in ("fused", "row", "dense_shared", "dense_existing"):
            if sha256(directory/f"{mode}.f32") != expected:
                raise ValueError("timed output differs from independent actual calculation")
        if native["output_sha256"] != expected or blas["reference_sha256"] != expected:
            raise ValueError("timing reference differs")
        if (native["warmups"], native["repetitions"]) != (2, 11) or len(native["method_order"]) != 13:
            raise ValueError("warmup/repetition policy changed")
        if any(sorted(order) != [0, 1, 2, 3] for order in native["method_order"]):
            raise ValueError("timing methods not interleaved")
        groups = dict(native["timings_seconds"], blas=blas["timings_seconds"])
        if set(groups) != {"fused", "row", "dense_shared", "dense_existing", "blas"}:
            raise ValueError("missing dense or packed control")
        for values in groups.values():
            if len(values) != 11 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("invalid raw timing samples")
        if len(blas["relative_mse"]) != 11 or max(blas["relative_mse"]) > 1e-10:
            raise ValueError("BLAS numerical comparison failed")
        if blas["cpu_affinity"] != runtime["affinity"] or any(p["num_threads"] != 1 for p in blas["blas"] if p["user_api"] == "blas"):
            raise ValueError("BLAS execution conditions differ")
        medians = {k: statistics.median(v) for k, v in groups.items()}
        recorded = next(r for r in summary["operator_cases"] if r["tensor"] == name and r["batch"] == batch)
        if medians != recorded["medians_seconds"]:
            raise ValueError("median summary differs")
        for mode in ("fused", "row"):
            ratios[str(batch)][mode]["native"].append(medians[mode]/min(medians["dense_shared"], medians["dense_existing"]))
            ratios[str(batch)][mode]["blas"].append(medians[mode]/medians["blas"])
    for batch, modes in ratios.items():
        for mode, comparisons in modes.items():
            for baseline, values in comparisons.items():
                ratio = statistics.geometric_mean(values)
                report = summary["G3"][batch][mode][baseline]
                if report["geometric_mean_latency_ratio"] != ratio or report["status"] != ("MEASURED" if ratio <= .9 else "FAILED"):
                    raise ValueError("latency gate differs from declared comparison")
    memory = {mode: json.loads((root/f"memory-{mode}.json").read_text()) for mode in ("packed", "dense")}
    for mode, result in memory.items():
        measured = result["measurements"]
        if measured["tensors"] != 272 or measured["decoded_weight_bytes"] != 538060032:
            raise ValueError("incomplete model memory scope")
        for stage in (result["before"], measured["loaded"], measured["after_operation"]):
            if not 0 < stage["VmRSS"] <= stage["VmHWM"]:
                raise ValueError("invalid process memory reading")
        if measured["output_sha256"] != outputs[("token_embd.weight", 1)]:
            raise ValueError("resident process did not produce the actual expected output")
    ratio = memory["packed"]["measurements"]["loaded"]["VmRSS"]/memory["dense"]["measurements"]["loaded"]["VmRSS"]
    if summary["G2"]["ratio"] != ratio or summary["G2"]["status"] != ("MEASURED" if ratio <= .4 else "FAILED"):
        raise ValueError("resident-memory gate differs")
    # A second isolated execution corroborates internal HWM with OS wait4/time accounting.
    external = {}
    for mode in ("packed", "dense"):
        time_path, log_path = root/f"external-memory-{mode}.time", root/f"external-memory-{mode}.log"
        command = ["/usr/bin/time", "-v", "-o", str(time_path), str(PROBE), "resident", mode, str(ARCHIVE),
                   "token_embd.weight", str(root/"inputs/tensor-0-batch-1.f32"), "1"]
        with log_path.open("x") as log:
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True, check=True)
        value = json.loads(run.stdout)
        match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_path.read_text())
        if not match:
            raise ValueError("external maximum RSS unavailable")
        maximum = int(match[1])
        internal = value["measurements"]["after_operation"]["VmHWM"]
        if abs(maximum-internal) > max(4096, .02*internal):
            raise ValueError("OS peak accounting disagrees with internal process memory")
        external[mode] = {"result": value, "os_max_rss_kib": maximum, "time_log_sha256": sha256(time_path)}
    replay_ratio = external["packed"]["result"]["measurements"]["loaded"]["VmRSS"]/external["dense"]["result"]["measurements"]["loaded"]["VmRSS"]
    if (replay_ratio <= .4) != (ratio <= .4):
        raise ValueError("resident memory benefit did not repeat")
    report = {"status": "PASSED", "independent_real_output_cases": len(outputs), "timing_cases": len(seen),
              "retained_timing_samples": 24*5*11, "external_memory": external,
              "G2": summary["G2"], "G3": summary["G3"], "auditor_sha256": sha256(__file__)}
    save_json(root/"audit.json", report)
    print(json.dumps({k: report[k] for k in ("status", "independent_real_output_cases", "timing_cases", "G2")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    run(parser.parse_args().run)
