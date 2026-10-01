"""Deterministic position coverage and create-new calibration publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def stratified_indices(length, count, seed, document_key):
    if (type(length) is not int or length <= 0 or type(count) is not int or count <= 0
            or type(seed) is not int or seed < 0 or not isinstance(document_key, str)):
        raise ValueError("sampling requires positive lengths/counts, a nonnegative seed and document key")
    count = min(length, count)
    positions = []
    for i in range(count):
        start, end = i*length//count, (i+1)*length//count
        key = f"{seed}|{document_key}|{length}|{i}".encode()
        value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        positions.append(start+value % (end-start))
    return positions


def require_new_capture(output):
    output = Path(output)
    if os.path.lexists(output) or os.path.lexists(output.with_suffix(output.suffix+".json")):
        raise ValueError("calibration or manifest destination exists; choose a new path")


def write_capture(output, data, metadata):
    output = Path(output)
    require_new_capture(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata, file_sha256=hashlib.sha256(data).hexdigest())
    contents = [(output.with_suffix(output.suffix+".json"),
                 (json.dumps(metadata, indent=2, allow_nan=False)+"\n").encode()), (output, data)]
    # Publish the data last: a reader cannot see a partial AC01 payload.
    for destination, payload in contents:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return metadata
