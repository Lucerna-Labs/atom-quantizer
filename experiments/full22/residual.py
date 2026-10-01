"""Explicit primitive-toolkit factorization of functional residual directions."""
import numpy as np


def functional_factors(error, inputs, rank, shuffled=False, seed=0):
    if type(rank) is not int or rank <= 0:
        raise ValueError("functional residual rank must be a positive integer")
    if inputs is None or np.iscomplexobj(error) or np.iscomplexobj(inputs):
        raise ValueError("functional residual needs real calibration inputs")
    error, inputs = np.asarray(error, np.float64), np.asarray(inputs, np.float64)
    if (error.ndim != 2 or inputs.ndim != 2 or not error.size or not inputs.size
            or error.shape[1] != inputs.shape[1] or not np.isfinite(error).all()
            or not np.isfinite(inputs).all()):
        raise ValueError("functional residual inputs must be finite, nonempty and aligned")
    geometry = inputs
    if shuffled:
        geometry = inputs[:, np.random.default_rng(seed).permutation(inputs.shape[1])]
    outputs = error @ geometry.T
    if not np.isfinite(outputs).all():
        raise ValueError("nonfinite functional residual outputs")
    u, singular, _ = np.linalg.svd(outputs, full_matrices=False)
    tolerance = singular[0] * max(outputs.shape) * np.finfo(np.float64).eps
    effective = min(rank, min(error.shape), int(np.count_nonzero(singular > tolerance)))
    left = u[:, :effective]
    right = left.T @ error
    with np.errstate(over="ignore", invalid="ignore"):
        left, right = left.astype(np.float16), right.astype(np.float16)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("functional residual factors exceed f16 storage")
    return left, right


def add_stored(values, arrays, metadata):
    left, right = arrays.get("lowrank_u"), arrays.get("lowrank_v")
    if (left is None or right is None or left.ndim != 2 or right.ndim != 2
            or left.shape[0] != values.shape[0] or right.shape[1] != values.shape[1]
            or left.shape[1] != right.shape[0]
            or metadata.get("lowrank_effective_rank", left.shape[1]) != left.shape[1]
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("invalid stored low-rank factors")
    return values + left.astype(np.float32) @ right.astype(np.float32)
