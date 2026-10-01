"""Validate A22 tensor placement against its GGUF prefix before exporting."""
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile

from . import codec
from .data import sha256
from .records import Record


class DecodedChecksumError(ValueError):
    def __init__(self, tensor, expected, actual):
        self.tensor, self.expected, self.actual = tensor, expected, actual
        super().__init__(f"decoded tensor checksum mismatch for {tensor}: expected {expected}, got {actual}")


class Prefix:
    def __init__(self, data):
        self.data = memoryview(data)
        self.offset = 0

    def take(self, size):
        if size < 0 or size > len(self.data) - self.offset:
            raise ValueError("truncated GGUF metadata")
        start = self.offset
        self.offset += size
        return self.data[start:self.offset]

    def number(self, fmt):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return bytes(self.take(self.number("<Q"))).decode("utf-8")

    def skip(self, kind):
        widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if kind in widths:
            self.take(widths[kind])
        elif kind == 8:
            self.take(self.number("<Q"))
        elif kind == 9:
            element, count = self.number("<I"), self.number("<Q")
            if element in widths:
                self.take(count * widths[element])
            elif element == 8 and count <= (len(self.data) - self.offset) // 8:
                for _ in range(count):
                    self.skip(8)
            else:
                raise ValueError("invalid GGUF metadata array")
        else:
            raise ValueError("unknown GGUF metadata type")


def validate_manifest(prefix, manifest):
    """Return authoritative F32 tensor extents without mapping untrusted data sizes."""
    if not isinstance(manifest, dict) or manifest.get("format") not in ("A22-1", "A22-2"):
        raise ValueError("unknown model format")
    if hashlib.sha256(prefix).hexdigest() != manifest.get("prefix_sha256"):
        raise ValueError("prefix checksum mismatch")
    if len(prefix) > 256 * 1024 * 1024:
        raise ValueError("GGUF metadata exceeds 256 MiB")
    p = Prefix(prefix)
    if p.take(4) != b"GGUF" or p.number("<I") not in (2, 3):
        raise ValueError("export requires little-endian GGUF v2/v3")
    count, metadata_count = p.number("<Q"), p.number("<Q")
    entries = manifest.get("tensors")
    if not isinstance(entries, list) or not 0 < count <= 2**20 or count != len(entries) or metadata_count > 2**20:
        raise ValueError("invalid or incomplete model tensor table")
    alignment, keys = 32, set()
    for _ in range(metadata_count):
        key, kind = p.string(), p.number("<I")
        if key in keys:
            raise ValueError("duplicate GGUF metadata key")
        keys.add(key)
        if key == "general.alignment":
            if kind != 4:
                raise ValueError("GGUF alignment must be UINT32")
            alignment = p.number("<I")
        else:
            p.skip(kind)
    if alignment == 0 or alignment & (alignment - 1):
        raise ValueError("GGUF alignment must be a nonzero power of two")
    tensors = {}
    for _ in range(count):
        name, dimensions = p.string(), p.number("<I")
        if not name or name in tensors or not 1 <= dimensions <= 8:
            raise ValueError("invalid or duplicate GGUF tensor")
        dims = [p.number("<Q") for _ in range(dimensions)]
        dtype, offset = p.number("<I"), p.number("<Q")
        if dtype != 0 or any(d == 0 for d in dims):
            raise ValueError("experimental export requires nonempty F32 tensors")
        size = 4
        for d in dims:
            size *= d
        if offset % alignment:
            raise ValueError("unaligned GGUF tensor offset")
        tensors[name] = {"shape": dims[::-1], "offset": len(prefix) + offset, "bytes": size}
    if (p.offset + alignment - 1) // alignment * alignment != len(prefix):
        raise ValueError("stored GGUF prefix has the wrong length")
    end = len(prefix)
    for tensor in sorted(tensors.values(), key=lambda t: t["offset"]):
        if tensor["offset"] < end:
            raise ValueError("overlapping GGUF tensor extents")
        end = tensor["offset"] + tensor["bytes"]
    source_bytes = manifest.get("source_bytes")
    if type(source_bytes) is not int or not end <= source_bytes <= (end + alignment - 1) // alignment * alignment:
        raise ValueError("model size disagrees with GGUF tensor extents")
    seen_names, seen_entries = set(), set()
    for entry in entries:
        name, member = entry.get("name"), entry.get("entry")
        if name not in tensors or name in seen_names or not isinstance(member, str) or member in seen_entries:
            raise ValueError("unknown or duplicate archive tensor")
        if member in {"manifest.json", "gguf-prefix.bin"}:
            raise ValueError("reserved tensor entry name")
        seen_names.add(name)
        seen_entries.add(member)
        tensor = tensors[name]
        shape = entry.get("shape")
        if (not isinstance(shape, list) or any(type(d) is not int for d in shape)
                or shape != tensor["shape"] or type(entry.get("gguf_offset")) is not int
                or entry["gguf_offset"] != tensor["offset"]):
            raise ValueError("manifest and GGUF tensor table disagree")
        if type(entry.get("bytes")) is not int or not 0 < entry["bytes"] <= 2**31:
            raise ValueError("invalid tensor entry size")
        if manifest["format"] == "A22-2" or "decoded_sha256" in entry:
            digest = entry.get("decoded_sha256")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("invalid or missing decoded tensor commitment")
    return tensors


def export_model(artifact, destination):
    destination = Path(destination)
    if os.path.lexists(destination):
        raise ValueError("destination exists")
    temporary = None
    verified_tensors = 0
    try:
        with zipfile.ZipFile(artifact) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("duplicate ZIP member")
            if archive.getinfo("manifest.json").file_size > 64 * 1024 * 1024:
                raise ValueError("manifest exceeds 64 MiB")
            if archive.getinfo("gguf-prefix.bin").file_size > 256 * 1024 * 1024:
                raise ValueError("GGUF metadata exceeds 256 MiB")
            manifest = json.loads(archive.read("manifest.json"))
            prefix = archive.read("gguf-prefix.bin")
            tensors = validate_manifest(prefix, manifest)
            expected_members = {"manifest.json", "gguf-prefix.bin"} | {e["entry"] for e in manifest["tensors"]}
            if set(members) != expected_members:
                raise ValueError("archive member set differs from manifest")
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            temporary = Path(name)
            with os.fdopen(fd, "wb") as f:
                f.write(prefix)
                context = {}
                for entry in manifest["tensors"]:
                    if archive.getinfo(entry["entry"]).file_size != entry["bytes"]:
                        raise ValueError("tensor entry length mismatch")
                    data = archive.read(entry["entry"])
                    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                        raise ValueError("tensor entry checksum mismatch")
                    tensor = tensors[entry["name"]]
                    record = Record.loads(data)
                    if record.meta.get("shape") != tensor["shape"]:
                        raise ValueError("record shape differs from GGUF table")
                    values = codec.decode(record, context)
                    if list(values.shape) != tensor["shape"] or values.nbytes != tensor["bytes"]:
                        raise ValueError("decoded tensor differs from GGUF table")
                    decoded = values.astype("<f4", copy=False).tobytes(order="C")
                    if "decoded_sha256" in entry:
                        actual_hash = hashlib.sha256(decoded).hexdigest()
                        if actual_hash != entry["decoded_sha256"]:
                            raise DecodedChecksumError(entry["name"], entry["decoded_sha256"], actual_hash)
                        verified_tensors += 1
                    if entry["name"].startswith("blk.0."):
                        context[entry["name"]] = values
                    f.seek(tensor["offset"])
                    f.write(decoded)
                f.truncate(manifest["source_bytes"])
                f.flush()
                os.fsync(f.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"destination": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination),
        "format": manifest["format"], "decoded_verified_tensors": verified_tensors}
