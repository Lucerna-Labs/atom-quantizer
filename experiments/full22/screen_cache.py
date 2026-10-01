"""Source-bound screen namespaces with checked resume and retained failed attempts."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import uuid

from . import codec
from .data import sha256
from .records import Record


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provenance(source, fit, selection, rows, names, configurations):
    modules = Path(__file__).parent
    versions = {}
    for name in ("numpy", "scipy", "gguf", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {"schema": 2, "source_sha256": sha256(source), "fit_sha256": sha256(fit),
        "selection_sha256": sha256(selection), "rows": rows, "tensors": list(names),
        "configurations": configurations, "gaussian_seed": 91,
        "implementation": {p.name: sha256(p) for p in sorted(modules.glob("*.py"))},
        "python": sys.version, "dependencies": versions}


class ScreenCache:
    def __init__(self, root, inputs):
        self.fingerprint = hashlib.sha256(canonical(inputs)).hexdigest()
        self.directory = Path(root) / "runs" / self.fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "invalidated.json").exists():
            raise ValueError("screen run was invalidated; retain it and use a fresh output directory")
        manifest = self.directory / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != inputs:
                raise ValueError("screen manifest disagrees with its fingerprint")
        else:
            atomic_write(manifest, canonical(inputs) + b"\n")

    def paths(self, tensor, experiment):
        key = hashlib.sha256(canonical({"tensor": tensor, "experiment": experiment})).hexdigest()
        return self.directory / (key + ".json"), self.directory / (key + ".ar")

    def resume(self, result_path, artifact, expected):
        try:
            result = json.loads(result_path.read_text())
            if (not isinstance(result, dict) or result.get("status") != "measured"
                    or any(result.get(k) != v for k, v in expected.items())):
                return False
            if artifact.stat().st_size != result["archive_bytes"] or sha256(artifact) != result["artifact_sha256"]:
                return False
            record = Record.loads(artifact.read_bytes())
            if record.meta.get("shape") != expected["shape"]:
                return False
            decoded = codec.decode(record)
            return list(decoded.shape) == expected["shape"]
        except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
            return False

    def retain_previous(self, result_path, artifact):
        previous = [p for p in (result_path, artifact) if p.exists()]
        if previous:
            history = self.directory / "history" / (result_path.stem + "-" + uuid.uuid4().hex)
            history.mkdir(parents=True)
            for path in previous:
                os.replace(path, history / path.name)
