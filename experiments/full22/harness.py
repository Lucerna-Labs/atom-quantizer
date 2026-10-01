"""Resumable real-tensor screening and self-contained full-model archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
import zipfile
import numpy as np
import gguf
from threadpoolctl import threadpool_limits
from . import codec, reuse
from .records import Record
from .catalog import configurations, TENSORS
from .data import OUT, SOURCE, sha256, load_calibration
from .archive import export_model, validate_manifest
from .screen_cache import ScreenCache, atomic_write, provenance

threadpool_limits(8)


def inputs_for(calibration, name):
    key = name+"::output" if name == "token_embd.weight" and name+"::output" in calibration else name
    x = calibration.get(key)
    return x if x is not None and x.ndim == 2 else None


def screen(rows=256, names=None):
    if type(rows) is not int or rows <= 0:
        raise ValueError("screen rows must be a positive integer")
    names = list(TENSORS if names is None else names)
    if not names or len(set(names)) != len(names):
        raise ValueError("screen tensor names must be nonempty and unique")
    experiments = configurations()
    inputs = provenance(SOURCE, OUT/"fit.acal", OUT/"selection.acal", rows, names, experiments)
    fit = load_calibration(OUT/"fit.acal", SOURCE)
    validation = load_calibration(OUT/"selection.acal", SOURCE)
    source = {t.name: t for t in gguf.GGUFReader(SOURCE).tensors}
    if sha256(SOURCE) != inputs["source_sha256"]:
        raise ValueError("source changed while opening the screen")
    cache = ScreenCache(OUT/"screen", inputs)
    measured, resumed, failed = 0, 0, 0
    for tensor_name in names:
        tensor = source[tensor_name]
        full = np.asarray(tensor.data, np.float32)
        if full.ndim != 2 or full.size == 0:
            raise ValueError("screen requires nonempty two-dimensional tensors")
        indices = np.linspace(0, len(full)-1, min(rows, len(full))).astype(np.int64)
        weight = np.ascontiguousarray(full[indices])
        x, xv = inputs_for(fit, tensor_name), inputs_for(validation, tensor_name)
        for experiment in experiments:
            result_file, artifact = cache.paths(tensor_name, experiment)
            expected = {"tensor": tensor_name, "configuration": experiment["name"], "principles": experiment["principles"],
                "config": experiment["config"], "shape": list(weight.shape), "screening_only": True,
                "source_sha256": inputs["source_sha256"], "run_fingerprint": cache.fingerprint,
                "sampled_rows_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
            if cache.resume(result_file, artifact, expected):
                resumed += 1
                continue
            cache.retain_previous(result_file, artifact)
            started = time.perf_counter()
            result = dict(expected)
            try:
                record = codec.encode(weight, x, experiment["config"])
                raw, compressed = record.dumps(), record.dumps(True)
                atomic_write(artifact, compressed)
                decoded = codec.decode(Record.loads(artifact.read_bytes()))
                result.update(status="measured", raw_bytes=len(raw), archive_bytes=len(compressed),
                    raw_bits_per_weight=8*len(raw)/weight.size, archive_bits_per_weight=8*len(compressed)/weight.size,
                    resident_array_bytes=record.resident_array_bytes,
                    fit=codec.distortion(weight, decoded, x), validation=codec.distortion(weight, decoded, xv))
                # Fixed Gaussian proxy on the same rows for the observability comparison.
                gaussian = np.random.default_rng(91).normal(size=(4, weight.shape[1])).astype(np.float32)/np.sqrt(weight.shape[1])
                result["gaussian"] = codec.distortion(weight, decoded, gaussian)
                result["artifact_sha256"] = sha256(artifact)
                json.dumps(result, allow_nan=False)
                measured += 1
            except Exception as error:
                result = dict(expected, status="failed", error=repr(error))
                failed += 1
            result["elapsed_seconds"] = time.perf_counter()-started
            atomic_write(result_file, (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
            print(f"{tensor_name}: {experiment['name']}: {result['status']} ({result['elapsed_seconds']:.2f}s)", flush=True)
    if provenance(SOURCE, OUT/"fit.acal", OUT/"selection.acal", rows, names, experiments) != inputs:
        atomic_write(cache.directory / "invalidated.json", b'{"reason":"screen inputs or implementation changed during execution"}\n')
        raise ValueError("screen inputs changed during execution")
    summary = {"directory": str(cache.directory), "run_fingerprint": cache.fingerprint,
        "measured": measured, "resumed": resumed, "failed": failed}
    print(json.dumps(summary), flush=True)
    return summary


def zip_write(archive, name, data):
    info = zipfile.ZipInfo(name, (2026, 9, 6, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    archive.writestr(info, data)


def build_model(config, output, source=SOURCE, calibration=None, entropy=True, tensor_configs=None,
                *, residual_calibration=None, reuse_archive=None, reuse_sha256=None):
    output = Path(output)
    summary_path = output.with_suffix(output.suffix+".json")
    if os.path.lexists(output) or os.path.lexists(summary_path):
        raise ValueError(f"output exists: {output}")
    source = Path(source)
    calibration = calibration or OUT/"fit.acal"
    source_hash, calibration_hash = sha256(source), sha256(calibration)
    fit = load_calibration(calibration, source)
    correction_fit = None
    if residual_calibration is not None:
        if not any(c.get("rank", 0) for c in [config, *(tensor_configs or {}).values()]):
            raise ValueError("residual calibration requires a residual configuration")
        residual_hash = sha256(residual_calibration)
        correction_fit = load_calibration(residual_calibration, source)
    reader = gguf.GGUFReader(source)
    if any(t.tensor_type != gguf.GGMLQuantizationType.F32 for t in reader.tensors):
        raise ValueError("experimental GGUF export currently requires an F32 source")
    with source.open("rb") as f:
        prefix = f.read(reader.data_offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "A22-2", "source_sha256": source_hash, "source_bytes": source.stat().st_size,
        "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "calibration_sha256": calibration_hash,
        "config": config, "entropy": entropy, "tensors": []}
    if correction_fit is not None:
        manifest["residual_calibration_sha256"] = residual_hash
    reusable = {}
    if reuse_archive is not None:
        for selected in [config, *(tensor_configs or {}).values()]:
            reuse.require_independent(selected)
        identity = {key:manifest.get(key) for key in ("source_sha256","calibration_sha256","residual_calibration_sha256")}
        identity["entropy"] = entropy
        reusable = reuse.load(reuse_archive,reuse_sha256,prefix,identity)
        manifest["reused_archive_sha256"] = reuse_sha256
    elif reuse_sha256 is not None:
        raise ValueError("reuse checksum requires a source archive")
    anchors = {}
    total_parameters, distortion_sum = 0, 0.0
    started = time.perf_counter()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            zip_write(archive, "gguf-prefix.bin", prefix)
            for index, tensor in enumerate(reader.tensors):
                weight = np.ascontiguousarray(tensor.data, dtype=np.float32)
                selected = dict((tensor_configs or {}).get(tensor.name, config))
                x = inputs_for(fit, tensor.name)
                anchor = None
                if selected.get("predict", False) and tensor.name.startswith("blk."):
                    role = ".".join(tensor.name.split(".")[2:])
                    anchor_name = "blk.0."+role
                    if anchor_name in anchors and anchors[anchor_name].shape == weight.shape:
                        anchor = (anchor_name, anchors[anchor_name])
                correction_args = {}
                if correction_fit is not None and selected.get("rank", 0) and weight.ndim == 2:
                    xc = inputs_for(correction_fit, tensor.name)
                    for role, observations in (("base", x), ("residual", xc)):
                        if (observations is None or not observations.size
                                or observations.shape[1] != weight.shape[1]
                                or not np.isfinite(observations).all()):
                            raise ValueError(f"missing or invalid {role} calibration for {tensor.name}")
                    correction_args["residual_inputs"] = xc
                prior = reusable.get(tensor.name)
                reused = prior is not None and prior[0]["config"] == selected
                if reused:
                    data = prior[1]
                    record = Record.loads(data)
                else:
                    record = codec.encode(weight, x, selected, anchor, **correction_args)
                    data = record.dumps(entropy)
                loaded = Record.loads(data)
                reconstructed = codec.decode(loaded, anchors)
                decoded_hash = hashlib.sha256(reconstructed.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
                if reused and decoded_hash != prior[0]["decoded_sha256"]:
                    raise ValueError(f"reused decoded commitment differs for {tensor.name}: expected {prior[0]['decoded_sha256']}, got {decoded_hash}")
                if tensor.name.startswith("blk.0."):
                    anchors[tensor.name] = reconstructed
                entry = f"tensors/{index:04}.ar"
                zip_write(archive, entry, data)
                metrics = codec.distortion(weight.reshape(len(weight), -1) if weight.ndim == 1 else weight,
                    reconstructed.reshape(len(weight), -1) if weight.ndim == 1 else reconstructed, x if weight.ndim == 2 else None)
                distortion_sum += metrics.get("output_nmse", metrics["weight_nmse"])
                total_parameters += weight.size
                manifest["tensors"].append({"name": tensor.name, "shape": list(weight.shape), "gguf_offset": tensor.data_offset,
                    "entry": entry, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "decoded_sha256": decoded_hash,
                    "kind": record.meta["kind"], "config": selected, "metrics": metrics})
                if reuse_archive is not None:
                    manifest["tensors"][-1]["reused"] = reused
                if (index+1) % 20 == 0:
                    print(f"Encoded {index+1}/{len(reader.tensors)} tensors", flush=True)
            manifest.update(parameters=total_parameters, local_distortion_sum=distortion_sum)
            validate_manifest(prefix, manifest)
            if sha256(source) != manifest["source_sha256"]:
                raise ValueError("source changed during model encoding")
            if (sha256(calibration) != calibration_hash or (correction_fit is not None
                    and sha256(residual_calibration) != residual_hash)):
                raise ValueError("calibration changed during model encoding")
            if reuse_archive is not None and sha256(reuse_archive) != reuse_sha256:
                raise ValueError("reused archive changed during model encoding")
            zip_write(archive, "manifest.json", json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        with zipfile.ZipFile(temporary) as check:
            if check.testzip() is not None:
                raise ValueError("archive corruption")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {"artifact": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
        "bits_per_parameter": 8*output.stat().st_size/total_parameters, "parameters": total_parameters,
        "config": config, "entropy": entropy, "local_distortion_sum": distortion_sum,
        "seconds": time.perf_counter()-started, "source_sha256": manifest["source_sha256"]}
    with summary_path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screen"); s.add_argument("--rows", type=int, default=256); s.add_argument("--tensor", action="append")
    b = sub.add_parser("build"); b.add_argument("--config", required=True); b.add_argument("--out", required=True)
    b.add_argument("--source", default=str(SOURCE)); b.add_argument("--calibration"); b.add_argument("--uncompressed", action="store_true")
    b.add_argument("--residual-calibration", help="optional source-bound observations used only for residual fitting")
    a = sub.add_parser("build-adaptive", help="raise precision until all fitting views pass native fidelity gates")
    a.add_argument("--source", default=str(SOURCE)); a.add_argument("--base-calibration", required=True)
    a.add_argument("--residual-calibration"); a.add_argument("--gate-calibration", action="append", required=True)
    a.add_argument("--assessor", required=True); a.add_argument("--out", required=True)
    a.add_argument("--start-bits", type=int, choices=(4,6,8), default=4)
    a.add_argument("--rank-geometry", choices=("activation","diagonal"))
    d = sub.add_parser("decode"); d.add_argument("artifact"); d.add_argument("destination")
    args = parser.parse_args()
    if args.command == "screen":
        screen(args.rows, args.tensor)
    elif args.command == "build":
        named = {c["name"]: c["config"] for c in configurations()}
        config = named[args.config] if args.config in named else json.loads(args.config)
        build_model(config, args.out, args.source, args.calibration, not args.uncompressed,
                    residual_calibration=args.residual_calibration)
    elif args.command == "build-adaptive":
        from .precision import run
        result = run(args.source,args.base_calibration,args.residual_calibration,args.gate_calibration,args.assessor,args.out,
            start_bits=args.start_bits,rank_geometry=args.rank_geometry)
        print(json.dumps({"status":result["status"],"model":result["model"],"stages":len(result["stages"])}))
    else:
        print(json.dumps(export_model(args.artifact, args.destination), indent=2))


if __name__ == "__main__":
    main()
