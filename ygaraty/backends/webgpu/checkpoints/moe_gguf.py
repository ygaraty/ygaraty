"""Shared mixture-of-experts GGUF helpers: serving stacked expert tensors as chunks of whole experts, and the
NumPy reference pieces both MoE families use (Kolibri-1, Qwen3-MoE).

A stacked expert tensor ([experts, rows, columns] in NumPy order) is served flat ([experts * rows, columns]) and
as chunks of whole experts (`<kind>.K.weight`), each a byte range of the stored tensor of at most `chunk_bytes`
(WebGPU's default storage binding size), so bundles hard-link the GGUF as for any other tensor.
"""

import numpy as np

from ygaraty.backends.webgpu.checkpoints.qwen35_gguf import Tensor


def expert_chunks(experts, expert_bytes, limit):
    """[(first expert, count)]: as few chunks as fit `limit` bytes each, as even as possible."""
    per = max(1, limit // expert_bytes)
    n = -(-experts // per)
    per = -(-experts // n)
    return [
        (first, min(per, experts - first)) for first in range(0, experts, per)
    ]


def read_tensors(reader, chunk_bytes):
    """(tensors, experts) of a GGUF reader: every tensor by name (2-D in NumPy order), each stacked expert tensor
    as chunks `blk.N.<kind>.K.weight`, and `experts[(layer, kind)]` the flat view of each stacked tensor."""
    tensors, experts = {}, {}
    for t in reader.tensors:
        shape = tuple(int(n) for n in reversed(t.shape.tolist()))
        data = t.data.view(np.uint8).reshape(-1)
        if len(shape) == 3:
            index, kind = int(t.name.split(".")[1]), t.name.split(".")[2]
            E, R, C = shape
            flat = Tensor(t.name, t.tensor_type.name, (E * R, C), data)
            experts[(index, kind)] = flat
            expert_bytes = R * flat.row_bytes
            for k, (first, count) in enumerate(
                expert_chunks(E, expert_bytes, chunk_bytes)
            ):
                name = f"blk.{index}.{kind}.{k}.weight"
                tensors[name] = Tensor(
                    name,
                    flat.kind,
                    (count * R, C),
                    data[
                        first * expert_bytes : (first + count) * expert_bytes
                    ],
                    base=t.name,
                    first_row=first * R,
                )
            continue
        tensors[t.name] = Tensor(t.name, t.tensor_type.name, shape, data)
    return tensors, experts


def chunk_names(layer_tensors, kind):
    """A layer's chunks of one expert tensor, in expert order."""
    names = [n for n in layer_tensors if n.startswith(kind + ".")]
    return sorted(names, key=lambda n: int(n.split(".")[1]))


# --- NumPy reference pieces ---------------------------------------------------


def rms_norm(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_halves(x, positions, theta):
    """RoPE over halves of the last axis of x[tokens, heads, d], in place (transformers' rotate_half)."""
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angle = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    c, s = (
        np.cos(angle).astype(np.float32)[:, None, :],
        np.sin(angle).astype(np.float32)[:, None, :],
    )
    a, b = x[..., : d // 2].copy(), x[..., d // 2 :].copy()
    x[..., : d // 2] = a * c - b * s
    x[..., d // 2 :] = b * c + a * s
    return x


def swiglu(a, b):
    return (a / (1.0 + np.exp(-a))) * b


def grouped_attention(heads, kv_heads, scale, q, K, V, positions, window=None):
    """Grouped-query attention of q[T, heads, d] over the cache K, V [S, kv_heads, d] (float64 softmax), causal,
    and limited to the last `window` keys when given."""
    S = K.shape[0]
    keys = np.arange(S)[None, :]
    hidden = keys > positions[:, None]
    if window:
        hidden |= keys <= positions[:, None] - window
    group = heads // kv_heads
    out = np.empty(q.shape, dtype=np.float32)
    for hh in range(heads):
        g = hh // group
        scores = (
            q[:, hh, :].astype(np.float64) @ K[:, g, :].T.astype(np.float64)
        ) * scale
        scores[hidden] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        out[:, hh, :] = (p @ V[:, g, :].astype(np.float64)).astype(np.float32)
    return out
