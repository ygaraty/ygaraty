"""Phi-4 forward kernels in NumPy (WS-9): one decoder block, the embedding and
the output head, with a per-session KV cache.

Phi-4 is transformers' Phi3 architecture: pre-norm blocks of RMSNorm, rotary
grouped-query causal attention (fused q/k/v projection), a residual, RMSNorm,
a SwiGLU feed-forward block (fused gate/up projection) and a residual; no
biases. Weights keep the Hugging Face layout (a Linear's weight is
(out, in)) and names (`LAYER_WEIGHTS`). Arithmetic runs in the dtype of the
arrays given (float64 for parity tests, float32 otherwise).

Llama and Granite checkpoints run through the same kernels (phi4_checkpoint
fuses their matrices on read). Granite's multipliers are optional config
entries: attention_scale (instead of 1/sqrt(head_dim)), embedding_scale,
residual_scale (on each block's two branches) and logits_scale.

A chunk of T tokens at absolute positions p..p+T-1 attends to the cache's
first p positions and to itself, causally. Rotary embedding is applied to the
keys before they are cached, as transformers does.
"""

import numpy as np

LAYER_WEIGHTS = (
    "input_layernorm.weight",
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
)
MODEL_WEIGHTS = (
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
)


def head_dim(config):
    return config["hidden"] // config["heads"]


def rms_norm(x, weight, eps):
    variance = (x * x).mean(axis=-1, keepdims=True)
    return weight * (x / np.sqrt(variance + eps))


def rotary(positions, dim, theta, dtype=np.float64):
    """cos and sin, (T, dim), for absolute positions: transformers' default
    rotary embedding (frequencies repeated over both halves)."""
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    freqs = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rotary(x, cos, sin, rotary_dim=None):
    """x: (heads, T, head_dim); a partial rotary factor rotates the first
    rotary_dim dimensions only."""
    d = rotary_dim or x.shape[-1]
    rot, rest = x[..., :d], x[..., d:]
    out = rot * cos[None] + rotate_half(rot) * sin[None]
    return np.concatenate([out, rest], axis=-1) if d < x.shape[-1] else out


class LayerCache:
    """One layer's keys and values for one session, (kv_heads, length,
    head_dim). `append` takes a step's keys at `position`: the cache's end,
    or, for a retried step, the start of the last step (which it replaces),
    so a retry never appends twice."""

    def __init__(self):
        self.k = self.v = None
        self.length, self.last = 0, None

    def append(self, position, k, v):
        tokens = k.shape[1]
        if position == self.length:
            pass
        elif self.last is not None and (position, tokens) == self.last:
            self.length = position  # a retry of the last step: replace it
            self.k, self.v = self.k[:, :position], self.v[:, :position]
        else:
            raise ValueError(
                f"KV position {position} out of order: the cache holds "
                f"{self.length} positions (last step {self.last})"
            )
        self.k = k if self.k is None else np.concatenate([self.k, k], axis=1)
        self.v = v if self.v is None else np.concatenate([self.v, v], axis=1)
        self.length, self.last = position + tokens, (position, tokens)
        return self.k, self.v


def attention(x, w, config, position, cache=None):
    """Rotary grouped-query causal self-attention for a chunk x (T, hidden)
    at absolute positions position..position+T-1. Without a cache the chunk
    must start at position 0 (the uncached full-prefix reference)."""
    T = x.shape[0]
    H, KV, hd = config["heads"], config["kv_heads"], head_dim(config)
    qkv = x @ w["self_attn.qkv_proj.weight"].T
    q = qkv[:, : H * hd].reshape(T, H, hd).transpose(1, 0, 2)
    k = qkv[:, H * hd : (H + KV) * hd].reshape(T, KV, hd).transpose(1, 0, 2)
    v = qkv[:, (H + KV) * hd :].reshape(T, KV, hd).transpose(1, 0, 2)
    dim = int(hd * config.get("partial_rotary_factor", 1.0))
    cos, sin = rotary(
        np.arange(position, position + T), dim, config["rope_theta"], x.dtype
    )
    q, k = apply_rotary(q, cos, sin, dim), apply_rotary(k, cos, sin, dim)
    if cache is not None:
        k, v = cache.append(position, k, v)
    elif position != 0:
        raise ValueError("Without a KV cache a chunk must start at position 0")
    L = k.shape[1]
    g = H // KV  # query head h reads key/value head h // g
    qg = q.reshape(KV, g, T, hd)
    scores = (qg @ k[:, None].transpose(0, 1, 3, 2)) * config.get(
        "attention_scale", hd**-0.5
    )  # (KV, g, T, L)
    allowed = np.arange(L)[None, :] <= (position + np.arange(T))[:, None]
    scores = np.where(allowed, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    out = (
        (p @ v[:, None])
        .reshape(H, T, hd)
        .transpose(1, 0, 2)
        .reshape(T, H * hd)
    )
    return out @ w["self_attn.o_proj.weight"].T


def mlp(x, w, config):
    gate_up = x @ w["mlp.gate_up_proj.weight"].T
    gate, up = np.split(gate_up, 2, axis=-1)
    silu = gate / (1.0 + np.exp(-gate))
    return (up * silu) @ w["mlp.down_proj.weight"].T


def block(x, w, config, position, cache=None):
    """One Phi-4 decoder block over a chunk (T, hidden)."""
    eps, scale = config["rms_epsilon"], config.get("residual_scale")
    a = attention(
        rms_norm(x, w["input_layernorm.weight"], eps),
        w,
        config,
        position,
        cache,
    )
    h = x + (a if scale is None else a * scale)
    m = mlp(rms_norm(h, w["post_attention_layernorm.weight"], eps), w, config)
    return h + (m if scale is None else m * scale)


def embed(tokens, table, scale=None):
    tokens = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if tokens.size == 0 or tokens.min() < 0 or tokens.max() >= len(table):
        raise ValueError("Token IDs outside the vocabulary")
    return table[tokens] if scale is None else table[tokens] * scale


def head(x, norm, lm_head, eps, rows=None, scale=None):
    """Logits of the given rows (default: all) after the final RMSNorm."""
    x = x if rows is None else x[rows]
    logits = rms_norm(x, norm, eps) @ lm_head.T
    return logits if scale is None else logits * scale


def greedy(logits):
    """The highest logit's token; ties go to the lowest ID, as argmax does."""
    return int(np.argmax(logits))


def forward(tokens, weights, config, position=0, caches=None):
    """A whole model over a chunk: logits (T, vocab). `weights`: the model
    weights by MODEL_WEIGHTS name and per layer by LAYER_WEIGHTS name
    (`weights["layers"][i]`). `caches`: one LayerCache per layer, or None for
    the uncached full-prefix forward (position 0)."""
    x = embed(
        tokens,
        weights["model.embed_tokens.weight"],
        config.get("embedding_scale"),
    )
    for i, w in enumerate(weights["layers"]):
        x = block(
            x, w, config, position, caches[i] if caches is not None else None
        )
    return head(
        x,
        weights["model.norm.weight"],
        weights["lm_head.weight"],
        config["rms_epsilon"],
        scale=config.get("logits_scale"),
    )
