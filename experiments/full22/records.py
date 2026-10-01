"""Self-contained, checked experimental tensor records (AR01/ACZ1)."""
from dataclasses import dataclass, field
import hashlib
import json
import struct
import zlib
import numpy as np

ALLOWED = {"|u1", "|i1", "<u2", "<i2", "<u4", "<i4", "<u8", "<f2", "<f4", "<f8"}


@dataclass
class Record:
    meta: dict
    arrays: dict = field(default_factory=dict)

    def dumps(self, entropy=False):
        descriptions, payload, offset = [], [], 0
        for name in sorted(self.arrays):
            array = np.ascontiguousarray(self.arrays[name])
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
            if array.dtype.str not in ALLOWED:
                raise ValueError(f"unsupported array dtype {array.dtype}")
            data = array.tobytes()
            descriptions.append({"name": name, "dtype": array.dtype.str, "shape": list(array.shape), "offset": offset, "bytes": len(data)})
            payload.append(data)
            offset += len(data)
        header = json.dumps({"version": 1, "meta": self.meta, "arrays": descriptions}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        data = b"AR01" + struct.pack("<I", len(header)) + header + b"".join(payload)
        data += hashlib.sha256(data).digest()
        if entropy:
            return b"ACZ1" + struct.pack("<Q", len(data)) + zlib.compress(data, level=6)
        return data

    @classmethod
    def loads(cls, data):
        if data[:4] == b"ACZ1":
            if len(data) < 12:
                raise ValueError("truncated compressed frame")
            size = struct.unpack_from("<Q", data, 4)[0]
            if size > 2**31:
                raise ValueError("record exceeds 2 GiB")
            stream = zlib.decompressobj()
            try:
                raw = stream.decompress(data[12:], size + 1)
            except zlib.error as error:
                raise ValueError("invalid compressed record") from error
            if len(raw) != size or not stream.eof or stream.unused_data or stream.unconsumed_tail:
                raise ValueError("invalid compressed record length")
            data = raw
        if len(data) < 40 or data[:4] != b"AR01":
            raise ValueError("invalid AR01 frame")
        if hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError("record checksum mismatch")
        length = struct.unpack_from("<I", data, 4)[0]
        if length > len(data) - 40:
            raise ValueError("invalid record metadata length")
        header = json.loads(data[8:8 + length])
        if not isinstance(header, dict):
            raise ValueError("record metadata must be an object")
        if type(header.get("version")) is not int or header["version"] != 1:
            raise ValueError("unknown record version")
        if not isinstance(header.get("meta"), dict) or not isinstance(header.get("arrays"), list):
            raise ValueError("invalid record metadata")
        body = memoryview(data)[8 + length:-32]
        arrays, offset = {}, 0
        for item in header["arrays"]:
            if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
                    or not item["name"] or item["name"] in arrays
                    or not isinstance(item.get("dtype"), str) or item["dtype"] not in ALLOWED
                    or type(item.get("offset")) is not int or item["offset"] != offset
                    or type(item.get("bytes")) is not int):
                raise ValueError("invalid array descriptor")
            shape = item.get("shape")
            if not isinstance(shape, list) or len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("invalid array shape")
            count = 1
            for n in shape:
                count *= n
            dtype = np.dtype(item["dtype"])
            nbytes = count * dtype.itemsize
            if nbytes != item["bytes"] or offset + nbytes > len(body):
                raise ValueError("invalid array extent")
            arrays[item["name"]] = np.frombuffer(body[offset:offset + nbytes], dtype=dtype).reshape(shape)
            offset += nbytes
        if offset != len(body):
            raise ValueError("unexpected payload suffix")
        return cls(header["meta"], arrays)

    def roundtrip(self, entropy=False):
        return Record.loads(self.dumps(entropy))

    @property
    def resident_array_bytes(self):
        return sum(a.nbytes for a in self.arrays.values())


def pack_codes(codes, bits):
    if not 1 <= bits <= 16:
        raise ValueError("unsupported code width")
    flat = np.asarray(codes).reshape(-1)
    if np.any(flat < 0) or np.any(flat >= 2**bits):
        raise ValueError("code outside alphabet")
    if bits == 8:
        return flat.astype(np.uint8)
    if bits == 16:
        return flat.astype("<u2").view(np.uint8)
    # Bounded chunks avoid a full model-sized bit matrix.
    chunk = 1 << 18
    output = []
    for start in range(0, len(flat), chunk):
        values = flat[start:start + chunk].astype(np.uint16)
        unpacked = ((values[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8).ravel()
        output.append(np.packbits(unpacked, bitorder="little"))
    return np.concatenate(output) if output else np.zeros(0, np.uint8)


def unpack_codes(packed, bits, count):
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if len(packed) != (count * bits + 7) // 8:
        raise ValueError("packed code length mismatch")
    if bits == 8:
        return packed.copy()
    if bits == 16:
        return packed.view("<u2").copy()
    chunk = 1 << 18
    output = []
    for start in range(0, count, chunk):
        n = min(chunk, count - start)
        # chunk is divisible by eight, so each chunk starts at a byte boundary.
        begin, end = start * bits // 8, ((start + n) * bits + 7) // 8
        values = np.unpackbits(packed[begin:end], bitorder="little")[:n * bits].reshape(n, bits)
        output.append((values.astype(np.uint16) << np.arange(bits, dtype=np.uint16)).sum(axis=1, dtype=np.uint16))
    return np.concatenate(output) if output else np.zeros(0, np.uint16)


def raw_record(weight):
    w = np.ascontiguousarray(weight, dtype="<f4")
    words = w.view("<u4")
    if np.all((words & 65535) == 0):
        return Record({"kind": "raw_bf16", "shape": list(w.shape)}, {"values": (words >> 16).astype("<u2")})
    return Record({"kind": "raw_f32", "shape": list(w.shape)}, {"values": w})
