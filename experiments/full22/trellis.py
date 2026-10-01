"""Restartable finite-state source quantization with actual Viterbi codes."""
import numpy as np
from scipy.special import ndtri
from .records import pack_codes, unpack_codes
from .transforms import padded


def encode(values, inputs, config, record):
    bits = int(config.get("bits", 2))
    states, branches, group = 16, 2**bits, int(config.get("group", 32))
    if branches > states or states % branches:
        raise ValueError("trellis supports at most four input bits per step")
    rng = np.random.default_rng(config.get("seed", 811))
    labels = ndtri((np.arange(states*branches)+0.5)/(states*branches))
    labels = labels[rng.permutation(len(labels))].reshape(states, branches).astype(np.float16)
    w = padded(values, group)
    scales = np.maximum(np.sqrt(np.mean(w*w, axis=-1)), np.nextafter(np.float16(0), np.float16(1))).astype(np.float16)
    targets = (w/scales.astype(np.float32)[:, :, None]).reshape(-1, group)
    if inputs is None:
        importance = np.ones(w.shape[1:], np.float32)
    else:
        importance = np.pad(np.mean(inputs*inputs, axis=0), (0, w.shape[1]*group-values.shape[1])).reshape(w.shape[1], group)
        importance = np.maximum(importance, 1e-12)
    destination = np.arange(states)
    symbols = destination % branches
    parents = destination[:, None]//branches + np.arange(branches)[None, :] * (states//branches)
    outputs = labels[parents, symbols[:, None]].astype(np.float32)
    all_codes = np.empty(targets.shape, np.uint8)
    for start in range(0, len(targets), 2048):
        x = targets[start:start+2048]
        count = len(x)
        weights = importance[(np.arange(start, start+count) % w.shape[1])]
        cost = np.full((count, states), np.inf, np.float32); cost[:, 0] = 0
        trace = np.empty((group, count, states), np.uint8)
        for position in range(group):
            choices = cost[:, parents] + weights[:, position, None, None] * (x[:, position, None, None]-outputs[None])**2
            pick = np.argmin(choices, axis=-1)
            trace[position] = pick
            cost = np.take_along_axis(choices, pick[:, :, None], axis=-1)[:, :, 0]
        state = np.argmin(cost, axis=-1)
        ids = np.arange(count)
        for position in range(group-1, -1, -1):
            all_codes[start:start+count, position] = state % branches
            state = parents[state, trace[position, ids, state]]
        if np.any(state != 0):
            raise ValueError("trellis did not backtrack to its stored initial state")
    record.meta.update(kind="trellis", bits=bits, group=group, states=states, initial_state=0, padded_cols=w.shape[1]*group)
    record.arrays.update(codes=pack_codes(all_codes, bits), scales=scales, labels=labels)


def decode(record):
    rows, cols = record.meta["shape"]
    width, group, bits = record.meta["padded_cols"], record.meta["group"], record.meta["bits"]
    states, branches = record.meta["states"], 2**bits
    codes = unpack_codes(record.arrays["codes"], bits, rows*width).reshape(-1, group)
    labels = record.arrays["labels"].astype(np.float32)
    if labels.shape != (states, branches):
        raise ValueError("trellis label shape mismatch")
    state = np.full(len(codes), record.meta["initial_state"], np.uint16)
    result = np.empty(codes.shape, np.float32)
    for position in range(group):
        symbol = codes[:, position]
        result[:, position] = labels[state, symbol]
        state = ((state << bits) | symbol) & (states-1)
    result = result.reshape(rows, width//group, group)
    result *= record.arrays["scales"].astype(np.float32)[:, :, None]
    return result.reshape(rows, width)[:, :cols]
