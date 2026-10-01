"""Composable experimental codecs with fully stored, blind decoder recipes."""
import numpy as np
from .records import Record, raw_record
from . import scalar, lattice, trellis, additive, adaptive, transforms, residual

FAMILIES = {"scalar": scalar, "e8": lattice, "trellis": trellis, "additive": additive, "adaptive": adaptive}
_USE_BASE_INPUTS = object()


def graph_apply(values, record):
    mode = record.meta.get("repair", "none")
    strength = float(record.meta.get("repair_strength", 0))
    if mode == "none" or strength == 0:
        return values
    if mode in ("flat", "row"):
        kernel = [0.1, 0.2, 0.4, 0.2, 0.1]
        data = values.reshape(1, -1) if mode == "flat" else values
        padded = np.pad(data, ((0, 0), (2, 2)))
        smooth = sum(np.float32(c)*padded[:, i:i+data.shape[1]] for i, c in enumerate(kernel))
        return (data + np.float32(strength)*(smooth-data)).reshape(values.shape)
    if mode != "graph":
        raise ValueError("unknown repair geometry")
    block = record.meta["graph_block"]
    rows, cols = values.shape
    width = (cols+block-1)//block*block
    indices = record.arrays["graph_indices"]
    weights = record.arrays["graph_weights"].astype(np.float32)
    adjacency = np.zeros((width//block, block, block), np.float32)
    for g in range(len(adjacency)):
        for i in range(block):
            adjacency[g, i, indices[g, i]] = weights[g, i]
    adjacency = (adjacency+adjacency.swapaxes(1, 2))*np.float32(0.5)
    laplacian = -adjacency
    index = np.arange(block)
    laplacian[:, index, index] += adjacency.sum(axis=-1)
    filters = np.linalg.inv(np.eye(block)[None] + strength*laplacian).astype(np.float32)
    data = np.pad(values, ((0, 0), (0, width-cols))).reshape(rows, width//block, block)
    return np.einsum("rgi,gij->rgj", data, filters, optimize=True).reshape(rows, width)[:, :cols]


def decode(record, context=None):
    m, arrays = record.meta, record.arrays
    kind = m["kind"]
    if kind == "raw_bf16":
        result = (arrays["values"].astype(np.uint32) << 16).view(np.float32)
    elif kind == "raw_f32":
        result = arrays["values"].astype(np.float32, copy=True)
    elif kind == "constant":
        result = np.full(m["shape"], arrays["value"][0], np.float32)
    else:
        if kind not in FAMILIES:
            raise ValueError("unknown experimental payload")
        result = FAMILIES[kind].decode(record)
        if "escape_indices" in arrays:
            flat = result.reshape(-1).copy()
            index = arrays["escape_indices"]
            if np.any(index >= len(flat)):
                raise ValueError("invalid exception index")
            flat[index] = arrays["escape_values"].astype(np.float32)
            result = flat.reshape(result.shape)
        result = transforms.inverse(result, record)
        if "anchor" in m:
            if context is None or m["anchor"] not in context:
                raise ValueError("missing decoded predictor anchor")
            anchor = context[m["anchor"]]
            if anchor.shape != result.shape:
                raise ValueError("predictor anchor shape mismatch")
            result += arrays["anchor_gain"].astype(np.float32)[0]*anchor
        result = graph_apply(result, record)
        if "lowrank_u" in arrays or "lowrank_v" in arrays:
            result = residual.add_stored(result, arrays, m)
        if "row_gain" in arrays:
            result *= arrays["row_gain"].astype(np.float32)[:, None]
        if "row_offset" in arrays:
            result += arrays["row_offset"].astype(np.float32)[:, None]
    if list(result.shape) != m["shape"] or not np.isfinite(result).all():
        raise ValueError("invalid reconstructed tensor")
    return np.ascontiguousarray(result, dtype=np.float32)


def _add_graph(record, inputs, config):
    mode = config.get("repair", "none")
    record.meta.update(repair=mode, repair_strength=float(config.get("repair_strength", 0.05)))
    if mode != "graph":
        return
    if inputs is None:
        raise ValueError("graph repair needs real input relationships")
    block = 32
    x = transforms.padded(inputs, block).astype(np.float64)
    cov = np.einsum("sgi,sgj->gij", x, x, optimize=True)/len(x)
    standard = np.sqrt(np.maximum(np.diagonal(cov, axis1=-2, axis2=-1), 1e-20))
    correlation = cov/(standard[:, :, None]*standard[:, None, :])
    correlation[:, np.arange(block), np.arange(block)] = 0
    correlation = np.maximum(correlation, 0)
    indices = np.argsort(correlation, axis=-1)[:, :, -2:].astype(np.uint8)
    weights = np.take_along_axis(correlation, indices, axis=-1).astype(np.float16)
    if config.get("shuffled_graph", False):
        rng = np.random.default_rng(config.get("seed", 0))
        for g in range(len(indices)):
            indices[g] = rng.permutation(block)[indices[g]]
    record.meta["graph_block"] = block
    record.arrays.update(graph_indices=indices, graph_weights=weights)


def _add_lowrank(record, source, inputs, config, context):
    requested = config.get("rank", 4)
    if type(requested) is not int or requested <= 0:
        raise ValueError("residual rank must be a positive integer")
    rank = min(requested, min(source.shape))
    geometry = config.get("rank_geometry", "diagonal")
    if geometry in ("activation", "activation-shuffled"):
        error = source.astype(np.float64) - decode(record, context).astype(np.float64)
        left, right = residual.functional_factors(error, inputs, rank,
            shuffled=geometry == "activation-shuffled", seed=config.get("seed", 0))
        record.arrays.update(lowrank_u=left, lowrank_v=right)
        record.meta["lowrank_effective_rank"] = left.shape[1]
        return
    if geometry != "diagonal":
        raise ValueError("unknown residual geometry")
    weight_residual = source-decode(record, context)
    importance = np.ones(source.shape[1], np.float32) if inputs is None else np.sqrt(np.mean(inputs*inputs, axis=0))
    importance = np.maximum(importance, 1e-6)
    matrix = weight_residual*importance[None]
    rng = np.random.default_rng(config.get("seed", 0))
    width = min(rank+8, min(source.shape))
    omega = rng.standard_normal((matrix.shape[1], width), dtype=np.float32)
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    # One power iteration improves the residual subspace while keeping cost bounded.
    q, _ = np.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    u, s, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    u = (q @ u[:, :rank])*s[:rank][None]
    vt = vt[:rank]/importance[None]
    record.arrays.update(lowrank_u=u.astype(np.float16), lowrank_v=vt.astype(np.float16))


def _add_gain(record, source, inputs, mode, context):
    reconstruction = decode(record, context)
    if mode == "norm":
        gain = np.linalg.norm(source.astype(np.float64), axis=1)/np.maximum(np.linalg.norm(reconstruction.astype(np.float64), axis=1), 1e-20)
    else:
        y = source if inputs is None else source @ inputs.T
        q = reconstruction if inputs is None else reconstruction @ inputs.T
        if mode == "gain_dc":
            # A weight-row offset contributes offset*sum(x), not a constant
            # output bias. Fit exactly that legal weight-space correction.
            z = np.ones(source.shape[1], np.float32) if inputs is None else inputs.sum(axis=1)
            aa = np.sum(q*q, axis=1, dtype=np.float64)
            ab = q.astype(np.float64) @ z.astype(np.float64)
            bb = float(np.sum(z*z, dtype=np.float64))
            ay = np.sum(q*y, axis=1, dtype=np.float64)
            by = y.astype(np.float64) @ z.astype(np.float64)
            det = aa*bb-ab*ab
            good = det > 1e-12*np.maximum(aa*bb, 1e-20)
            gain = np.divide(ay*bb-by*ab, det, out=np.ones_like(aa), where=good)
            offset = np.divide(by*aa-ay*ab, det, out=np.zeros_like(aa), where=good)
            record.arrays["row_offset"] = offset.astype(np.float16)
        else:
            numerator = np.sum(y*q, axis=1, dtype=np.float64)
            denominator = np.sum(q*q, axis=1, dtype=np.float64)
            gain = np.divide(numerator, denominator, out=np.ones_like(numerator), where=denominator > 1e-20)
    record.arrays["row_gain"] = np.clip(gain, 0, 4).astype(np.float16)


def encode(weight, inputs=None, config=None, anchor=None, *, residual_inputs=_USE_BASE_INPUTS):
    config = dict(config or {})
    source = np.ascontiguousarray(weight, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("nonfinite source")
    if residual_inputs is not _USE_BASE_INPUTS:
        if (not config.get("rank", 0) or source.ndim != 2 or residual_inputs is None
                or np.iscomplexobj(residual_inputs) or np.asarray(residual_inputs).ndim != 2
                or not np.asarray(residual_inputs).size
                or np.asarray(residual_inputs).shape[1] != source.shape[1]
                or not np.isfinite(residual_inputs).all()):
            raise ValueError("explicit residual inputs require a rank and finite, nonempty aligned observations")
    if source.ndim != 2 or config.get("family") == "lossless":
        return raw_record(source)
    if np.all(source == source.flat[0]):
        return Record({"kind": "constant", "shape": list(source.shape)}, {"value": source.reshape(-1)[:1]})
    record = Record({"shape": list(source.shape), "config": config})
    context = None
    target = source
    if anchor is not None:
        name, decoded_anchor = anchor
        if decoded_anchor.shape != source.shape:
            raise ValueError("predictor shapes differ")
        gain = np.sum(source*decoded_anchor, dtype=np.float64)/max(np.sum(decoded_anchor*decoded_anchor, dtype=np.float64), 1e-20)
        gain = np.array([gain], np.float16)
        record.meta["anchor"] = name
        record.arrays["anchor_gain"] = gain
        context = {name: decoded_anchor}
        target = source-gain.astype(np.float32)[0]*decoded_anchor
    transformed = transforms.forward(target, inputs, config, record)
    x = transforms.transform_inputs(inputs, record)
    family = config.get("family", "scalar")
    if family not in FAMILIES:
        raise ValueError("unknown encoder family")
    fraction = config.get("exceptions", 0)
    if fraction:
        # Allocate a fixed actual exception count for fair saliency controls.
        preliminary = Record(dict(record.meta), dict(record.arrays))
        scalar.encode(transformed, x, config, preliminary)
        baseline = scalar.decode(preliminary)
        score = np.abs(transformed) if config.get("exception_rule") == "magnitude" else (transformed-baseline)**2
        if config.get("exception_rule") != "magnitude" and x is not None:
            score *= np.mean(x*x, axis=0)[None]
        count = max(1, int(transformed.size*fraction))
        indices = np.argpartition(score.reshape(-1), -count)[-count:]
        indices = np.sort(indices).astype(np.uint32)
        record.arrays["escape_indices"] = indices
        record.arrays["escape_values"] = transformed.reshape(-1)[indices].astype(np.float16)
        transformed = transformed.copy()
        transformed.reshape(-1)[indices] = 0
    FAMILIES[family].encode(transformed, x, config, record)
    if config.get("repair", "none") != "none":
        _add_graph(record, inputs, config)
    if config.get("rank", 0):
        correction_inputs = inputs if residual_inputs is _USE_BASE_INPUTS else residual_inputs
        _add_lowrank(record, source, correction_inputs, config, context)
    if config.get("post", "none") != "none":
        _add_gain(record, source, inputs, config["post"], context)
    # Exercise the exact stored representation, including all f16 side data.
    record = record.roundtrip()
    decode(record, context)
    return record


def distortion(reference, reconstruction, inputs=None):
    w = np.asarray(reference, np.float32)
    q = np.asarray(reconstruction, np.float32)
    error = w.astype(np.float64)-q.astype(np.float64)
    energy = np.sum(w.astype(np.float64)**2)
    weight_mse = np.sum(error**2)/max(energy, 1e-30)
    dot = np.sum(w.astype(np.float64)*q.astype(np.float64))
    cosine = dot/max(np.sqrt(energy*np.sum(q.astype(np.float64)**2)), 1e-30)
    result = {"weight_nmse": float(weight_mse), "cosine": float(cosine), "max_error": float(np.max(np.abs(error)))}
    if inputs is not None:
        y = w @ inputs.T
        yy = q @ inputs.T
        result["output_nmse"] = float(np.sum((y.astype(np.float64)-yy)**2)/max(np.sum(y.astype(np.float64)**2), 1e-30))
    return result
