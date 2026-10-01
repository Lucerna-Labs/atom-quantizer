"""Verified reuse of independent scalar records during a precision ladder."""
import hashlib
import json
from pathlib import Path
import zipfile

from .archive import validate_manifest
from .data import sha256
from .records import Record


def require_independent(config):
    allowed = {"family", "bits", "rank", "rank_geometry"}
    if (not isinstance(config, dict) or set(config)-allowed
            or config.get("family", "scalar") not in ("scalar", "lossless")):
        raise ValueError("record reuse requires independent scalar/lossless configurations")


def load(path, expected_hash, prefix, identity):
    path = Path(path)
    if expected_hash is None or sha256(path) != expected_hash:
        raise ValueError("reused archive identity mismatch")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("format") != "A22-2" or archive.read("gguf-prefix.bin") != prefix:
            raise ValueError("reuse requires matching committed A22-2 model metadata")
        validate_manifest(prefix,manifest)
        if any(manifest.get(key)!=value for key,value in identity.items()):
            raise ValueError("reused source or calibration differs")
        names = archive.namelist()
        expected = {"manifest.json","gguf-prefix.bin"}|{e["entry"] for e in manifest["tensors"]}
        if len(names)!=len(set(names)) or set(names)!=expected:
            raise ValueError("invalid reusable archive members")
        records = {}
        for entry in manifest["tensors"]:
            require_independent(entry["config"])
            data = archive.read(entry["entry"])
            if len(data)!=entry["bytes"] or hashlib.sha256(data).hexdigest()!=entry["sha256"]:
                raise ValueError("reused record checksum mismatch")
            record = Record.loads(data)
            if (record.meta.get("shape")!=entry["shape"] or "anchor" in record.meta
                    or record.meta.get("kind")!=entry["kind"]
                    or record.meta.get("config",entry["config"])!=entry["config"]):
                raise ValueError("invalid or dependent reusable record")
            records[entry["name"]] = (entry,data)
    if sha256(path)!=expected_hash:
        raise ValueError("reused archive changed while loading")
    return records
