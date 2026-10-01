"""Real packed scalar codecs, covariance feedback, and coupled rounding."""
import numpy as np
from .records import pack_codes, unpack_codes
from .transforms import padded


def alphabet(bits, kind="full"):
    if kind == "ternary":
        if bits != 2:
            raise ValueError("ternary requires two packed bits")
        return np.array([-1, 0, 1], np.float32)
    if kind == "nonuniform4":
        if bits != 2:
            raise ValueError("four-level alphabet requires two bits")
        return np.array([-1, -0.25, 0.25, 1], np.float32)
    return np.linspace(-1, 1, 2**bits, dtype=np.float32)


def nearest(values, table):
    # Source values must already be divided by their stored scales.
    cuts = (table[:-1] + table[1:]) * np.float32(0.5)
    return np.searchsorted(cuts, values).astype(np.uint8)


def block_geometry(inputs, cols, group, damping=0.01):
    groups = (cols + group - 1) // group
    if inputs is None:
        h = np.broadcast_to(np.eye(group), (groups, group, group)).copy()
    else:
        x = np.pad(inputs, ((0, 0), (0, groups * group - cols))).astype(np.float64)
        x = x.reshape(len(x), groups, group)
        h = np.einsum("sgi,sgj->gij", x, x, optimize=True) / len(x)
    diagonal = np.diagonal(h, axis1=1, axis2=2).copy()
    floor = np.maximum(diagonal.mean(axis=1), 1e-12) * damping
    h += floor[:, None, None] * np.eye(group)[None]
    return h.astype(np.float32), diagonal.astype(np.float32)


def encode(values, inputs, config, record):
    bits, group = int(config.get("bits", 4)), int(config.get("group", 32))
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError("unsupported scalar bitwidth")
    rows, cols = values.shape
    w = padded(values, group)
    table = alphabet(bits, config.get("alphabet", "full")).astype(np.float16).astype(np.float32)
    h, importance = block_geometry(inputs, cols, group, config.get("damping", 0.01))
    if inputs is None:
        importance.fill(1)
    importance = np.maximum(importance, 1e-12)
    base = np.maximum(np.max(np.abs(w), axis=-1), 1e-12)
    precision = config.get("scale_dtype", "f16")
    dtype = np.float16 if precision == "f16" else np.float32
    def stored(s):
        return np.maximum(s, np.nextafter(dtype(0), dtype(1))).astype(dtype).astype(np.float32)
    scales = stored(base)
    best = np.sum(importance[None] * (w - scales[:, :, None] * table[nearest(w / scales[:, :, None], table)])**2, axis=-1)
    if config.get("clip", True):
        for fraction in [0.9, 0.8, 0.65, 0.5]:
            s = stored(base * fraction)
            for _ in range(3):
                q = table[nearest(w / s[:, :, None], table)]
                numerator = np.sum(importance[None] * w * q, axis=-1, dtype=np.float64)
                denominator = np.sum(importance[None] * q * q, axis=-1, dtype=np.float64)
                s = stored(np.maximum(numerator / np.maximum(denominator, 1e-30), 1e-12))
            error = np.sum(importance[None] * (w - s[:, :, None] * table[nearest(w / s[:, :, None], table)])**2, axis=-1)
            select = error < best
            scales[select], best[select] = s[select], error[select]
    rounding = config.get("rounding", "nearest")
    codes = nearest(w / scales[:, :, None], table)
    if rounding in ("stochastic", "paired"):
        if config.get("alphabet", "full") != "full":
            raise ValueError("stochastic rounding requires the full uniform grid")
        normalized = w / scales[:, :, None]
        hi = np.searchsorted(table, normalized).clip(1, len(table)-1)
        lo = hi - 1
        probability = np.clip((normalized-table[lo])/(table[hi]-table[lo]), 0, 1)
        rng = np.random.default_rng(config.get("seed", 0))
        u = rng.random(w.shape, dtype=np.float32)
        if rounding == "paired":
            u[..., 1::2] = 1 - u[..., 0::2]
        codes = np.where(u < probability, hi, lo).astype(np.uint8)
    elif rounding == "feedback":
        # Block GPTQ-style triangular error compensation. All calibration state
        # is encode-only; only final codes/scales are required for decode.
        factor = np.linalg.cholesky(np.linalg.inv(h.astype(np.float64))).swapaxes(-1, -2).astype(np.float32)
        work = w.copy()
        for i in range(group):
            codes[:, :, i] = nearest(work[:, :, i] / scales, table)
            q = scales * table[codes[:, :, i]]
            error = (work[:, :, i] - q) / factor[None, :, i, i]
            work[:, :, i:] -= error[:, :, None] * factor[None, :, i, i:]
    elif rounding == "adjacent":
        residual = np.zeros(w.shape[:2], np.float32)
        for i in range(group):
            target = w[:, :, i] + residual
            codes[:, :, i] = nearest(target / scales, table)
            residual = target - scales * table[codes[:, :, i]]
    elif rounding == "joint":
        recon = scales[:, :, None] * table[codes]
        gradient = np.einsum("rgj,gji->rgi", recon-w, h, optimize=True)
        for _ in range(config.get("rounds", 3)):
            for i in range(group):
                current = codes[:, :, i].astype(np.int16)
                up, down = np.minimum(current+1, len(table)-1), np.maximum(current-1, 0)
                du = scales * (table[up] - table[current])
                dd = scales * (table[down] - table[current])
                lu = 2*du*gradient[:, :, i] + du*du*h[None, :, i, i]
                ld = 2*dd*gradient[:, :, i] + dd*dd*h[None, :, i, i]
                choice = np.where((lu < ld) & (lu < 0), up, np.where(ld < 0, down, current))
                delta = scales * (table[choice] - table[current])
                codes[:, :, i] = choice.astype(np.uint8)
                gradient += delta[:, :, None] * h[None, :, i, :]
    elif rounding != "nearest":
        raise ValueError(f"unknown rounding {rounding}")

    if config.get("posterior", False):
        # Freeze codes and estimate conditional normalized cell means. Training
        # rows and held-out rows can be separated by the experiment caller.
        prior_rows = max(1, rows // 2)
        normalized = w[:prior_rows] / scales[:prior_rows, :, None]
        fixed = codes[:prior_rows].reshape(-1)
        weights = np.broadcast_to(importance, normalized.shape).reshape(-1)
        numerator = np.bincount(fixed, weights=(normalized.reshape(-1)*weights), minlength=len(table))
        denominator = np.bincount(fixed, weights=weights, minlength=len(table))
        means = np.divide(numerator, denominator, out=table.astype(np.float64).copy(), where=denominator > 0)
        # Conditional means remain in the cells used by the frozen encoder.
        cuts = (table[:-1]+table[1:])/2
        table = np.clip(means, np.r_[-np.inf, cuts], np.r_[cuts, np.inf]).astype(np.float16).astype(np.float32)
    record.meta.update(kind="scalar", bits=bits, group=group, padded_cols=w.shape[1]*group, alphabet=config.get("alphabet", "full"))
    record.arrays["codes"] = pack_codes(codes, bits)
    record.arrays["scales"] = scales.astype(dtype)
    record.arrays["levels"] = table.astype(np.float16)


def decode(record):
    rows, cols = record.meta["shape"]
    group, padded_cols, bits = record.meta["group"], record.meta["padded_cols"], record.meta["bits"]
    codes = unpack_codes(record.arrays["codes"], bits, rows*padded_cols).reshape(rows, padded_cols // group, group)
    scales = record.arrays["scales"].astype(np.float32)
    table = record.arrays["levels"].astype(np.float32)
    if np.any(codes >= len(table)) or scales.shape != codes.shape[:2] or np.any(scales <= 0):
        raise ValueError("invalid scalar decoder state")
    return (scales[:, :, None] * table[codes]).reshape(rows, padded_cols)[:, :cols]
