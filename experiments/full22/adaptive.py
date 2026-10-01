"""Dyadic groups selected by actual scale/partition costs and distortion."""
import numpy as np
from .records import pack_codes, unpack_codes
from .scalar import alphabet, nearest


def encode(values, inputs, config, record):
    bits = int(config.get("bits", 4))
    table = alphabet(bits).astype(np.float16).astype(np.float32)
    rows, cols = values.shape
    if cols % 16:
        raise ValueError("adaptive experiment requires rows divisible by sixteen")
    importance = np.ones(cols, np.float32) if inputs is None else np.maximum(np.mean(inputs*inputs, axis=0), 1e-12)
    # Work in distortion per original weight-energy unit; lambda prices metadata.
    unit = max(float(np.mean(values*values*importance[None])), 1e-20)
    price = float(config.get("partition_lambda", 0.05)) * unit
    choices, scales, lengths, codes = [], [], [], []
    candidates = {}
    for length in [16, 32, 64, 128]:
        if cols % length:
            continue
        w = values.reshape(rows, cols//length, length)
        h = importance.reshape(cols//length, length)
        scale = np.maximum(np.max(np.abs(w), axis=-1), np.nextafter(np.float16(0), np.float16(1))).astype(np.float16)
        q = nearest(w/scale.astype(np.float32)[:, :, None], table)
        error = np.sum(h[None]*(w-scale.astype(np.float32)[:, :, None]*table[q])**2, axis=-1)
        # 16 bits for one stored f16 scale plus 2 bits for this leaf's length.
        cost = error + price*18
        candidates[length] = (cost, scale, q)
    # Process row regions of at most 128, falling back to aligned 64/32/16 tails.
    def best(r, begin, length):
        group = begin//length
        own, scale, q = candidates[length]
        leaf = [(length, scale[r, group], q[r, group])]
        if length == 16:
            return float(own[r, group]), leaf
        left_cost, left = best(r, begin, length//2)
        right_cost, right = best(r, begin+length//2, length//2)
        if left_cost+right_cost < own[r, group]:
            return left_cost+right_cost, left+right
        return float(own[r, group]), leaf
    top = max(candidates)
    for r in range(rows):
        for begin in range(0, cols, top):
            _, leaves = best(r, begin, top)
            for length, scale, q in leaves:
                lengths.append(int(np.log2(length//16)))
                scales.append(scale)
                codes.append(q)
    record.meta.update(kind="adaptive", bits=bits, leaves=len(lengths), partition_lambda=config.get("partition_lambda", 0.05))
    record.arrays.update(codes=pack_codes(np.concatenate(codes), bits), scales=np.asarray(scales, np.float16),
        lengths=pack_codes(np.asarray(lengths, np.uint8), 2), levels=table.astype(np.float16))


def decode(record):
    rows, cols = record.meta["shape"]
    lengths = 16 << unpack_codes(record.arrays["lengths"], 2, record.meta["leaves"])
    if int(lengths.sum()) != rows*cols or len(record.arrays["scales"]) != len(lengths):
        raise ValueError("invalid adaptive partition")
    codes = unpack_codes(record.arrays["codes"], record.meta["bits"], rows*cols)
    table = record.arrays["levels"].astype(np.float32)
    result = table[codes] * np.repeat(record.arrays["scales"].astype(np.float32), lengths)
    return result.reshape(rows, cols)
