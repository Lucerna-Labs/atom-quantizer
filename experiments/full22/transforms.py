import numpy as np
from .records import pack_codes, unpack_codes


def checked_native_scale(values, cols):
    scale = np.asarray(values, dtype=np.float32)
    if (scale.shape != (cols,) or not np.isfinite(scale).all()
            or np.any(scale < 0.25) or np.any(scale > 4.0)):
        raise ValueError("invalid stored native resource scale")
    return scale


def basis_apply(values, basis, inverse=False):
    """Apply stored native-discovery transport; never regenerate it during decode."""
    basis = np.asarray(basis, dtype=np.float32)
    if basis.shape != (32, 32) or not np.isfinite(basis).all():
        raise ValueError("invalid stored basis")
    if values.ndim != 2 or values.shape[1] % 32:
        raise ValueError("stored basis requires rows divisible by 32")
    error = np.max(np.abs(basis.astype(np.float64) @ basis.astype(np.float64).T - np.eye(32)))
    if error > 2e-6:
        raise ValueError("stored basis is not orthogonal within f32 tolerance")
    return (values.reshape(-1, 32) @ (basis if inverse else basis.T)).reshape(values.shape)


def hadamard(values, block):
    if block == 0:
        return np.array(values, dtype=np.float32, copy=True)
    if block & (block - 1) or values.shape[-1] % block:
        raise ValueError("Hadamard requires a power-of-two block dividing the row")
    shape = values.shape
    data = np.array(values, dtype=np.float32, copy=True).reshape(-1, block)
    step = 1
    while step < block:
        view = data.reshape(-1, block // (2 * step), 2, step)
        a, b = view[:, :, 0].copy(), view[:, :, 1].copy()
        view[:, :, 0], view[:, :, 1] = a + b, a - b
        step *= 2
    data *= np.float32(1 / np.sqrt(block))
    return data.reshape(shape)


def forward(weight, inputs, config, record):
    n = weight.shape[1]
    data = np.array(weight, dtype=np.float32, copy=True)
    if config.get("center", False):
        mean = data.mean(axis=1, dtype=np.float64).astype(np.float16)
        record.arrays["pre_mean"] = mean
        data -= mean.astype(np.float32)[:, None]
    if config.get("diagonal", 0):
        if inputs is None:
            raise ValueError("diagonal preconditioning needs linear operator inputs")
        act = np.sqrt(np.mean(np.square(inputs, dtype=np.float64), axis=0))
        magnitude = np.maximum(np.max(np.abs(weight), axis=0), 1e-8)
        d = np.power(np.maximum(act, 1e-8) / magnitude, config["diagonal"])
        d /= np.exp(np.mean(np.log(d)))
        d = np.clip(d, 1/16, 16).astype(np.float16)
        record.arrays["pre_diagonal"] = d
        data *= d.astype(np.float32)
    if "native_scale_path" in config:
        scale = checked_native_scale(np.load(config["native_scale_path"], allow_pickle=False), n)
        record.arrays["pre_native_scale"] = scale
        data *= scale
    block = int(config.get("rotation", 0))
    record.meta["rotation"] = block
    if config.get("signed", False):
        rng = np.random.default_rng(config.get("seed", 0))
        signs = rng.integers(0, 2, size=n, dtype=np.uint8)
        record.arrays["pre_signs"] = pack_codes(signs, 1)
        data *= (signs.astype(np.float32) * 2 - 1)
    data = hadamard(data, block)
    if "basis_path" in config:
        basis = np.asarray(np.load(config["basis_path"], allow_pickle=False), dtype=np.float32)
        data = basis_apply(data, basis)
        record.arrays["pre_basis"] = basis
    return data


def inverse(values, record):
    data = basis_apply(values, record.arrays["pre_basis"], True) if "pre_basis" in record.arrays else values
    data = hadamard(data, record.meta.get("rotation", 0))
    n = data.shape[1]
    if "pre_signs" in record.arrays:
        data *= (unpack_codes(record.arrays["pre_signs"], 1, n).astype(np.float32) * 2 - 1)
    if "pre_native_scale" in record.arrays:
        data /= checked_native_scale(record.arrays["pre_native_scale"], n)
    if "pre_diagonal" in record.arrays:
        data /= record.arrays["pre_diagonal"].astype(np.float32)
    if "pre_mean" in record.arrays:
        data += record.arrays["pre_mean"].astype(np.float32)[:, None]
    return data


def transform_inputs(inputs, record):
    if inputs is None:
        return None
    x = np.array(inputs, dtype=np.float32, copy=True)
    if "pre_diagonal" in record.arrays:
        x /= record.arrays["pre_diagonal"].astype(np.float32)
    if "pre_native_scale" in record.arrays:
        x /= checked_native_scale(record.arrays["pre_native_scale"], x.shape[1])
    if "pre_signs" in record.arrays:
        x *= (unpack_codes(record.arrays["pre_signs"], 1, x.shape[1]).astype(np.float32) * 2 - 1)
    x = hadamard(x, record.meta.get("rotation", 0))
    return basis_apply(x, record.arrays["pre_basis"]) if "pre_basis" in record.arrays else x


def padded(values, group):
    rows, cols = values.shape
    width = (cols + group - 1) // group * group
    return np.pad(values, ((0, 0), (0, width-cols))).reshape(rows, width // group, group)
