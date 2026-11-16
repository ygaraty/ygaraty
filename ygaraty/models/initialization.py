"""Deterministic bounded block initialization, independent of graph recipes."""

import hashlib

import numpy as np


def initial_block(config, name, spec, region):
    i, ei, j, ej = region
    shape = (ei - i, ej - j)
    if spec["kind"] == "forget_bias":
        return np.broadcast_to(
            np.where(np.arange(j, ej) < spec["hidden"], 2.0, 0.0), shape
        ).copy()
    if spec["kind"] == "zeros":
        return np.zeros(shape, dtype=np.float64)
    if spec["kind"] == "ones":
        return np.ones(shape, dtype=np.float64)
    if spec["kind"] == "constant":
        return np.asarray(spec["data"], dtype=np.float64)[i:ei, j:ej].copy()
    if spec["kind"] not in ("normal", "uniform"):
        raise ValueError("Unsupported initializer")
    # The tile layout is part of this reproducible model recipe.
    key = f"{config['seed']}:{name}:{i}:{j}:{config['tile_size']}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    rng = np.random.default_rng(seed)
    a = (
        rng.uniform(-spec["bound"], spec["bound"], size=shape)
        if spec["kind"] == "uniform"
        else rng.normal(0, spec["scale"], size=shape)
    )
    if spec.get("zero_row") is not None and i <= spec["zero_row"] < ei:
        a[spec["zero_row"] - i] = 0
    return a
