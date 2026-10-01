"""Finite E8 lattice source coding, with an explicit canonical codebook."""
from functools import lru_cache
import numpy as np
from .records import pack_codes, unpack_codes
from .transforms import padded


def keys(doubled):
    digits = (doubled.astype(np.int32) + 8).astype(np.uint32)
    return np.sum(digits << (np.arange(8, dtype=np.uint32)*4), axis=-1, dtype=np.uint64).astype(np.uint32)


@lru_cache(maxsize=1)
def codebook():
    # E8 = D8 union (D8 + (1/2,...,1/2)); retain squared norm <= 10.
    points = []
    for choices in [range(-6, 7, 2), range(-5, 6, 2)]:
        minimum = min(x*x for x in choices)
        def visit(prefix, norm, total):
            left = 8-len(prefix)
            if norm + minimum*left > 40:
                return
            if left == 0:
                if total % 4 == 0:
                    points.append(prefix)
                return
            for v in choices:
                if norm+v*v <= 40:
                    visit(prefix+[v], norm+v*v, total+v)
        visit([], 0, 0)
    points = np.asarray(points, np.int8)
    order = np.argsort(keys(points))
    points = points[order]
    if len(points) != 56881:
        raise RuntimeError(f"unexpected E8 shell size: {len(points)}")
    return points.astype(np.float32)*0.5, keys(points)


def nearest_unbounded(x):
    results = []
    for shift in [0.0, 0.5]:
        y = x-shift
        z = np.rint(y).astype(np.int16)
        wrong = z.sum(axis=1) % 2 != 0
        residual = y-z
        coordinate = np.argmax(np.abs(residual), axis=1)
        ids = np.flatnonzero(wrong)
        change = np.where(residual[ids, coordinate[ids]] >= 0, 1, -1)
        z[ids, coordinate[ids]] += change.astype(np.int16)
        results.append(z.astype(np.float32)+shift)
    a, b = results
    use_b = np.sum((x-b)**2, axis=1) < np.sum((x-a)**2, axis=1)
    a[use_b] = b[use_b]
    return a


def nearest_bounded(x):
    table, table_keys = codebook()
    q = nearest_unbounded(x)
    bad = np.flatnonzero(np.sum(q*q, axis=1) > 10.00001)
    roots = table[np.sum(table*table, axis=1) == 2]
    for start in range(0, len(bad), 256):
        ids = bad[start:start+256]
        neighbors = q[ids, None, :] + roots[None]
        valid = np.sum(neighbors*neighbors, axis=-1) <= 10.00001
        distance = np.sum((neighbors-x[ids, None, :])**2, axis=-1)
        distance[~valid] = np.inf
        best = np.argmin(distance, axis=1)
        ok = np.isfinite(distance[np.arange(len(ids)), best])
        q[ids[ok]] = neighbors[np.arange(len(ids))[ok], best[ok]]
        for i in ids[~ok]:
            shrink = x[i:i+1].copy()
            for _ in range(64):
                shrink *= 0.9
                candidate = nearest_unbounded(shrink)
                if np.sum(candidate*candidate) <= 10:
                    q[i] = candidate[0]
                    break
            else:
                raise ValueError("E8 bounded-shell search failed")
    key = keys(np.rint(q*2).astype(np.int8))
    index = np.searchsorted(table_keys, key)
    if np.any(index >= len(table)) or not np.array_equal(table_keys[index], key):
        raise ValueError("point outside the specified E8 codebook")
    return index.astype(np.uint16)


def encode(values, inputs, config, record):
    group = int(config.get("group", 32))
    if group % 8:
        raise ValueError("E8 scale group must divide into eight-coordinate vectors")
    w = padded(values, group)
    subvectors = w.reshape(*w.shape[:2], group//8, 8)
    maximum_norm = np.sqrt(np.max(np.sum(subvectors*subvectors, axis=-1), axis=-1))
    scales = np.maximum(maximum_norm/np.sqrt(8), np.nextafter(np.float16(0), np.float16(1))).astype(np.float16)
    normalized = (subvectors/scales.astype(np.float32)[:, :, None, None]).reshape(-1, 8)
    codes = np.empty(len(normalized), np.uint16)
    for start in range(0, len(normalized), 65536):
        codes[start:start+65536] = nearest_bounded(normalized[start:start+65536])
    record.meta.update(kind="e8", group=group, padded_cols=w.shape[1]*group, codebook="E8_R10_v1", bits_per_vector=16)
    record.arrays["codes"] = pack_codes(codes, 16)
    record.arrays["scales"] = scales


def decode(record):
    if record.meta["codebook"] != "E8_R10_v1":
        raise ValueError("unknown lattice codebook")
    rows, cols = record.meta["shape"]
    width, group = record.meta["padded_cols"], record.meta["group"]
    table, _ = codebook()
    indices = unpack_codes(record.arrays["codes"], 16, rows*width//8)
    if np.any(indices >= len(table)):
        raise ValueError("invalid E8 index")
    values = table[indices].reshape(rows, width//group, group)
    values *= record.arrays["scales"].astype(np.float32)[:, :, None]
    return values.reshape(rows, width)[:, :cols]
