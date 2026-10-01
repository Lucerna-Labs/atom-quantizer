"""Stored multi-codebook vector quantization, including residual refinement."""
import numpy as np
from .records import pack_codes, unpack_codes
from .transforms import padded


def assign(values, centers):
    result = np.empty(len(values), np.uint16)
    norms = np.sum(centers*centers, axis=1)
    for start in range(0, len(values), 4096):
        x = values[start:start+4096]
        distances = norms[None] - 2*(x @ centers.T)
        result[start:start+len(x)] = np.argmin(distances, axis=1)
    return result


def train(values, count, seed, iterations=8):
    rng = np.random.default_rng(seed)
    sample = values[rng.choice(len(values), min(len(values), 16384), replace=False)]
    centers = sample[rng.choice(len(sample), count, replace=len(sample) < count)].copy()
    for _ in range(iterations):
        ids = assign(sample, centers)
        sums = np.zeros(centers.shape, np.float64)
        populations = np.bincount(ids, minlength=count)
        np.add.at(sums, ids, sample)
        used = populations > 0
        centers[used] = sums[used] / populations[used, None]
    return centers.astype(np.float16).astype(np.float32)


def encode(values, inputs, config, record):
    group = int(config.get("group", 32))
    dim = int(config.get("vector_dim", 8))
    books = int(config.get("books", config.get("bits", 2)))
    index_bits = int(config.get("index_bits", 8))
    if group % dim or not 1 <= books <= 8 or not 2 <= index_bits <= 8:
        raise ValueError("invalid additive codebook configuration")
    w = padded(values, group)
    scales = np.maximum(np.sqrt(np.mean(w*w, axis=-1)), np.nextafter(np.float16(0), np.float16(1))).astype(np.float16)
    target = (w/scales.astype(np.float32)[:, :, None]).reshape(-1, dim)
    rng = np.random.default_rng(config.get("seed", 0))
    training = target[rng.choice(len(target), min(len(target), 16384), replace=False)]
    residual = training.copy()
    tables = []
    for book in range(books):
        centers = train(residual, 2**index_bits, config.get("seed", 0)+book)
        tables.append(centers)
        residual -= centers[assign(residual, centers)]
    tables = np.stack(tables)
    codes = np.empty((len(target), books), np.uint16)
    reconstruction = np.zeros_like(target)
    for book in range(books):
        codes[:, book] = assign(target-reconstruction, tables[book])
        reconstruction += tables[book][codes[:, book]]
    for _ in range(config.get("refinements", 1)):
        for book in range(books):
            reconstruction -= tables[book][codes[:, book]]
            codes[:, book] = assign(target-reconstruction, tables[book])
            reconstruction += tables[book][codes[:, book]]
    record.meta.update(kind="additive", group=group, vector_dim=dim, books=books,
        index_bits=index_bits, padded_cols=w.shape[1]*group)
    record.arrays.update(codes=pack_codes(codes, index_bits), scales=scales, codebooks=tables.astype(np.float16))


def decode(record):
    rows, cols = record.meta["shape"]
    width, group = record.meta["padded_cols"], record.meta["group"]
    dim, books, bits = record.meta["vector_dim"], record.meta["books"], record.meta["index_bits"]
    codes = unpack_codes(record.arrays["codes"], bits, rows*width//dim*books).reshape(-1, books)
    tables = record.arrays["codebooks"].astype(np.float32)
    if tables.shape != (books, 2**bits, dim):
        raise ValueError("invalid additive codebook table")
    result = np.zeros((len(codes), dim), np.float32)
    for book in range(books):
        result += tables[book][codes[:, book]]
    result = result.reshape(rows, width//group, group)
    result *= record.arrays["scales"].astype(np.float32)[:, :, None]
    return result.reshape(rows, width)[:, :cols]
