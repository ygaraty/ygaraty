"""Qwen3-MoE (`qwen3moe`, e.g. Qwen3-Coder-30B-A3B) from llama.cpp's GGUF: the `qwen3moe` model family (DO-80).

The NumPy reference here computes in the GGUF's own layout, without Torch, following transformers'
`modeling_qwen3_moe.py`:

- Decoder layers of full causal grouped-query attention (Qwen3-Coder-30B-A3B: 48 layers, hidden 2,048; 32 query
  heads and 4 KV heads of 128, so the query width 4,096 differs from the hidden width), Q/K/V/O without bias,
  per-head RMSNorm on Q and K before RoPE; RoPE over halves of the whole head (theta 10,000,000). llama.cpp does
  not permute Qwen2/Qwen3 Q/K rows (its NEOX rotation), so the GGUF's rows are transformers' rows.
- Every layer's MLP is a mixture of experts: router logits (`ffn_gate_inp`, float32), a softmax over all experts,
  the top k (8 of 128) by probability, renormalized to sum 1 when `norm_topk_prob` (true for Qwen3-Coder); each
  expert a SwiGLU of width 768 with separate gate and up tensors. No shared expert, no post-branch norms.
- Two RMSNorms per layer (eps 1e-6): `attn_norm` before attention, `ffn_norm` before the MLP; then `output_norm`
  and an untied `output` head.

GGUF tensors (NumPy shapes, rows are output features; `blk.N.` per layer):

    token_embd.weight [vocab, hidden]   output.weight [vocab, hidden]   output_norm.weight [hidden] F32
    attn_norm, ffn_norm .weight [hidden] F32   attn_q_norm, attn_k_norm .weight [head_dim] F32
    attn_q.weight [heads*D, hidden]   attn_k/attn_v.weight [kv*D, hidden]   attn_output.weight [hidden, heads*D]
    ffn_gate_inp.weight [experts, hidden] F32 (the router)
    ffn_gate_exps.weight, ffn_up_exps.weight [experts, F, hidden]   ffn_down_exps.weight [experts, hidden, F]

`Model` serves each expert tensor as chunks of whole experts (`moe_gguf.read_tensors`), byte ranges of the
stored tensor, as Kolibri-1's loader does.

Routing ties: the reference orders experts by descending probability with ties to the lower index (a stable
sort); transformers' `torch.topk` does not define an order for exact ties. Fixtures must not depend on one.
"""

import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.checkpoints import families, gguf_remote, moe_gguf
from ygaraty.backends.webgpu.checkpoints.moe_gguf import rms_norm, rope_halves, swiglu

ARCHITECTURE = "qwen3moe"
# YGARATY_QWEN3MOE_EXPERT_CHUNK_BYTES: smaller chunks for tests on tiny models (several chunks per layer)
EXPERT_CHUNK_BYTES = int(
    os.environ.get("YGARATY_QWEN3MOE_EXPERT_CHUNK_BYTES", 128 * 2**20)
)
EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


@dataclass(frozen=True)
class Qwen3MoeConfig:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    experts: int
    top_k: int
    expert_ffn: int  # each expert's SwiGLU width
    norm_topk: bool  # renormalize the top-k probabilities to sum 1
    rope_theta: float
    eps: float
    attention_scale: float
    tied_embeddings: bool = False

    @property
    def query_width(self):
        return self.heads * self.head_dim

    @property
    def intermediate(
        self,
    ):  # the per-token MLP scratch width the plan reserves (the routed experts' gate/up rows)
        return self.top_k * 2 * self.expert_ffn

    def manifest(self):
        return dict(
            asdict(self), family=ARCHITECTURE, intermediate=self.intermediate
        )


def layer_type(config, index):
    return "full"


class Model:
    """A Qwen3-MoE GGUF: its tensors (memory-mapped, packed) and checked configuration."""

    family = ARCHITECTURE

    def layer_type(self, index):
        return layer_type(self.config, index)

    def __init__(self, path, config_path, *, tensor_only=False):
        import gguf

        self.path = Path(path)
        document = json.loads(Path(config_path).read_text())
        header = gguf_remote.read_header(self.path)
        family = families.FAMILIES[ARCHITECTURE]
        errors = [
            f["message"]
            for f in family.check(document, header)
            if f["level"] == "error"
        ]
        if errors:
            raise ValueError(
                "Unsupported Qwen3-MoE checkpoint: " + "; ".join(errors)
            )
        g = family.geometry(document, header)
        self.config = Qwen3MoeConfig(
            g["layers"],
            g["hidden"],
            g["heads"],
            g["kv_heads"],
            g["head_dim"],
            g["vocab"],
            g["experts"],
            g["top_k"],
            g["expert_ffn"],
            g["norm_topk"],
            g["rope_theta"],
            float(g["eps"]),
            float(g["attention_scale"]),
        )
        if tensor_only:
            from ygaraty.backends.webgpu.checkpoints.gguf_metadata import tensor_reader

            self.reader = tensor_reader(self.path)
        else:
            self.reader = gguf.GGUFReader(self.path)
        self.tensors, self.experts = moe_gguf.read_tensors(
            self.reader, EXPERT_CHUNK_BYTES
        )

    def __getitem__(self, name):
        return self.tensors[name]

    def expert_rows(self, index, kind, expert):
        """Float32 weights of one expert: gate or up [expert_ffn, hidden], down [hidden, expert_ffn]."""
        flat = self.experts[(index, kind)]
        rows = (
            self.config.hidden
            if kind == "ffn_down_exps"
            else self.config.expert_ffn
        )
        return flat.dequantize(
            rows=np.arange(expert * rows, (expert + 1) * rows)
        )

    def validate_webgpu(self):
        kinds = {}
        for t in self.tensors.values():
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        return kinds

    def layer_tensors(self, index):
        prefix = f"blk.{index}."
        return {
            name[len(prefix) :]: t
            for name, t in self.tensors.items()
            if name.startswith(prefix)
        }

    def expert_chunk_names(self, index, kind):
        return moe_gguf.chunk_names(self.layer_tensors(index), kind)


# --- NumPy reference, in the GGUF's layout ------------------------------------


def route(c, router, h):
    """Top-k experts of each row of h: (indices [T, k] by descending probability, ties to the lower index;
    weights [T, k]: the softmax probabilities of the chosen experts, renormalized to sum 1 when `norm_topk`;
    margins [T]: the k-th minus the (k+1)-th probability, how near the selection is to changing)."""
    logits = h.astype(np.float64) @ router.astype(np.float64).T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1, kind="stable")
    chosen = order[:, : c.top_k]
    weights = np.take_along_axis(p, chosen, axis=1)
    kth = weights[:, -1]
    nxt = (
        np.take_along_axis(p, order[:, c.top_k : c.top_k + 1], axis=1)[:, 0]
        if c.experts > c.top_k
        else -np.inf
    )
    if c.norm_topk:
        weights = weights / weights.sum(axis=1, keepdims=True)
    return chosen, weights, kth - nxt


class LayerWeights:
    """One layer's dense weights (dequantized), and its experts dequantized on demand (a few kept)."""

    def __init__(self, model, index, keep=64):
        self.model, self.index, self.keep = model, index, keep
        t = model.layer_tensors(index)
        get = lambda name: t[name].dequantize()  # noqa: E731
        self.attn_norm, self.ffn_norm = (
            get("attn_norm.weight"),
            get("ffn_norm.weight"),
        )
        self.q_norm, self.k_norm = (
            get("attn_q_norm.weight"),
            get("attn_k_norm.weight"),
        )
        self.q, self.k, self.v, self.o = (
            get("attn_q.weight"),
            get("attn_k.weight"),
            get("attn_v.weight"),
            get("attn_output.weight"),
        )
        self.router = get("ffn_gate_inp.weight")
        self.cache = OrderedDict()

    def expert(self, e):
        if e in self.cache:
            self.cache.move_to_end(e)
            return self.cache[e]
        m = self.model
        value = (
            m.expert_rows(self.index, "ffn_gate_exps", e),
            m.expert_rows(self.index, "ffn_up_exps", e),
            m.expert_rows(self.index, "ffn_down_exps", e),
        )
        self.cache[e] = value
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return value


def forced_route(c, router, h, chosen):
    """The weights the router gives a given expert choice (DO-80: a reference that follows the GPU's own routing):
    the softmax probabilities of `chosen` [T, k], renormalized to sum 1 when `norm_topk`."""
    logits = h.astype(np.float64) @ router.astype(np.float64).T
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    weights = np.take_along_axis(p, chosen, axis=1)
    return (
        weights / weights.sum(axis=1, keepdims=True)
        if c.norm_topk
        else weights
    )


def moe(c, w, h, routes=None, forced=None):
    """The MLP of h[T, hidden]: the routed experts, weighted. `routes`: appended with (indices, weights, margins).
    `forced` [T, k]: use these experts (with the router's own weights for them) instead of choosing."""
    chosen, weights, margins = route(c, w.router, h)
    if (
        routes is not None
    ):  # the reference's own choice, even when it follows `forced`
        routes.append((chosen, weights, margins))
    if forced is not None:
        chosen = np.asarray(forced, dtype=np.int64)
        weights = forced_route(c, w.router, h, chosen)
    out = np.zeros((h.shape[0], c.hidden), dtype=np.float32)
    for e in np.unique(chosen):
        rows, slots = np.nonzero(chosen == e)
        gate, up, down = w.expert(int(e))
        y = swiglu(h[rows] @ gate.T, h[rows] @ up.T) @ down.T
        out[rows] += (weights[rows, slots][:, None] * y).astype(np.float32)
    return out


def layer_forward(
    c, w, x, cache, position, kv="f32", routes=None, forced=None
):
    """Layer `w.index` over x[tokens, hidden] at `position`, appending to cache = {"k": [], "v": []}."""
    T = x.shape[0]
    h = rms_norm(x, w.attn_norm, c.eps)
    q = rms_norm((h @ w.q.T).reshape(T, c.heads, c.head_dim), w.q_norm, c.eps)
    k = rms_norm(
        (h @ w.k.T).reshape(T, c.kv_heads, c.head_dim), w.k_norm, c.eps
    )
    v = (h @ w.v.T).reshape(T, c.kv_heads, c.head_dim)
    positions = np.arange(position, position + T)
    rope_halves(q, positions, c.rope_theta)
    rope_halves(k, positions, c.rope_theta)
    if (
        kv == "f16"
    ):  # as the cache stores them (the current tokens' too: attention reads them back)
        k, v = (
            k.astype(np.float16).astype(np.float32),
            v.astype(np.float16).astype(np.float32),
        )
    cache["k"].append(k)
    cache["v"].append(v)
    out = moe_gguf.grouped_attention(
        c.heads,
        c.kv_heads,
        c.attention_scale,
        q,
        np.concatenate(cache["k"]),
        np.concatenate(cache["v"]),
        positions,
    )
    x = x + out.reshape(T, -1) @ w.o.T
    return x + moe(c, w, rms_norm(x, w.ffn_norm, c.eps), routes, forced)


class StageReference:
    """Layers first..last-1 with their KV cache, for prefill then cached steps. `routes[layer]`: each call's
    (indices, weights, margins)."""

    def __init__(self, model, first, last, kv="f32"):
        self.c, self.kv, self.position = model.config, kv, 0
        self.layers = [
            (i, LayerWeights(model, i), {"k": [], "v": []})
            for i in range(first, last)
        ]
        self.routes = {i: [] for i in range(first, last)}
        self.forced = None  # DO-80: {layer: expert ids [T, k]} for the next call, or None to choose

    def __call__(self, hidden):
        x = np.ascontiguousarray(hidden, dtype=np.float32)
        for i, w, cache in self.layers:
            forced = self.forced.get(i) if self.forced else None
            x = layer_forward(
                self.c,
                w,
                x,
                cache,
                self.position,
                self.kv,
                self.routes[i],
                forced,
            )
        self.position += x.shape[0]
        return x


def embed(model, tokens):
    return model["token_embd.weight"].dequantize(rows=tokens)


def run_layers(model, x, first, last, kv="f32", routes=None):
    """Layers first..last-1 over the whole sequence in one causal pass, one layer's weights at a time."""
    for i in range(first, last):
        log = [] if routes is not None else None
        x = layer_forward(
            model.config,
            LayerWeights(model, i),
            x,
            {"k": [], "v": []},
            0,
            kv,
            log,
        )
        if routes is not None:
            routes[i] = log[0]
    return x


def logits(model, hidden):
    """Final norm and output projection of hidden[positions, hidden] (float64)."""
    h = rms_norm(
        hidden, model["output_norm.weight"].dequantize(), model.config.eps
    ).astype(np.float64)
    W = model["output.weight"]
    out = np.empty((h.shape[0], W.shape[0]))
    for r in range(0, W.shape[0], 16384):
        rows = np.arange(r, min(r + 16384, W.shape[0]))
        out[:, rows] = h @ W.dequantize(rows=rows).astype(np.float64).T
    return out


# --- transformers cross-check (optional Torch) --------------------------------


def hf_layer_state(model, index):
    """Layer `index` as transformers' Qwen3MoeDecoderLayer state dict (float32 NumPy), from the GGUF tensors.
    transformers 5 stores the experts fused: gate_up_proj [experts, 2F, hidden] (gate rows, then up), down_proj
    [experts, hidden, F]."""
    c, w = model.config, LayerWeights(model, index, keep=0)
    gate = [
        model.expert_rows(index, "ffn_gate_exps", e) for e in range(c.experts)
    ]
    up = [model.expert_rows(index, "ffn_up_exps", e) for e in range(c.experts)]
    down = [
        model.expert_rows(index, "ffn_down_exps", e) for e in range(c.experts)
    ]
    return {
        "input_layernorm.weight": w.attn_norm,
        "post_attention_layernorm.weight": w.ffn_norm,
        "self_attn.q_proj.weight": w.q,
        "self_attn.k_proj.weight": w.k,
        "self_attn.v_proj.weight": w.v,
        "self_attn.o_proj.weight": w.o,
        "self_attn.q_norm.weight": w.q_norm,
        "self_attn.k_norm.weight": w.k_norm,
        "mlp.gate.weight": w.router,
        "mlp.experts.gate_up_proj": np.stack(
            [np.concatenate([g, u]) for g, u in zip(gate, up)]
        ),
        "mlp.experts.down_proj": np.stack(down),
    }
