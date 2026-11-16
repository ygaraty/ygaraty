"""Kolibri-1 (Aleph Alpha, `kolibri1`) from our own GGUF: the `kolibri1` model family.

Upstream llama.cpp has no `kolibri1` architecture, so `kolibri_convert.py` writes the GGUF from the official FP8
checkpoint itself (tensor names and layout below). The browser stage (`kolibri_gpu.mjs`) reads its bytes
unchanged; the NumPy reference here computes in the same layout, without Torch, following transformers'
`modeling_kolibri1.py` (5.22):

- 50 decoder layers, hidden 2,560; attention: 48 query heads and 4 KV heads of 128, Q/K/V/O without bias,
  per-head RMSNorm on Q and K (before RoPE). `layer_types`: 4 of every 5 layers are sliding-window attention
  (key k visible to query q when q - window < k <= q, window 513: 512 earlier tokens and the current one) with
  RoPE over halves (theta 10,000, the whole head); every fifth layer is global causal attention without
  position encoding ("NoPE").
- Every layer's MLP is a mixture of experts: a router (float32 logits from `ffn_gate_inp`), the top 6 of 384
  experts by logit + `exp_probs_b` (the selection bias, not in the weights), weighted by sigmoid(logit) without
  renormalization; each expert a SwiGLU of width 512 (`ffn_gate_up_exps`: gate rows then up rows), plus one
  shared SwiGLU expert of width 512 added unweighted.
- Four norms per layer (all RMSNorm, eps 1e-6, weights as they are): `attn_norm` before attention,
  `attn_post_norm` on the attention output before the residual add, `ffn_norm` before the MLP, `ffn_post_norm`
  on the MLP output before the residual add. Then `output_norm` and an untied `output` head.

GGUF tensors (NumPy shapes, rows are output features; `blk.N.` per layer):

    token_embd.weight [vocab, hidden]   output.weight [vocab, hidden]   output_norm.weight [hidden] F32
    attn_norm, attn_post_norm, ffn_norm, ffn_post_norm .weight [hidden] F32
    attn_q.weight [heads*128, hidden]   attn_k/attn_v.weight [kv*128, hidden]   attn_output.weight [hidden, heads*128]
    attn_q_norm, attn_k_norm .weight [128] F32
    ffn_gate_inp.weight [experts, hidden] F32 (the router)   exp_probs_b.bias [experts] F32
    ffn_gate_up_exps.weight [experts, 2*512, hidden]   ffn_down_exps.weight [experts, hidden, 512]
    ffn_gate_shexp, ffn_up_shexp .weight [512, hidden]   ffn_down_shexp.weight [hidden, 512]

`Model` serves each expert tensor as chunks of whole experts (`ffn_gate_up_exps.K.weight`, 2-D: experts x rows
by columns), each at most `EXPERT_CHUNK_BYTES` (WebGPU's default storage binding size); a chunk is a byte range
of the stored tensor, so bundles hard-link the GGUF as for any tensor.
"""

import os
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.checkpoints import families, gguf_remote, moe_gguf
from ygaraty.backends.webgpu.checkpoints.moe_gguf import rms_norm, rope_halves, swiglu  # noqa: F401 (shared; re-exported)

ARCHITECTURE = "kolibri1"
# YGARATY_KOLIBRI_EXPERT_CHUNK_BYTES: smaller chunks for tests on tiny models (several chunks per layer)
EXPERT_CHUNK_BYTES = int(
    os.environ.get("YGARATY_KOLIBRI_EXPERT_CHUNK_BYTES", 128 * 2**20)
)
EXPERT_TENSORS = ("ffn_gate_up_exps", "ffn_down_exps")
LAYER_MATRICES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
)
LAYER_FLOATS = (
    "attn_norm.weight",
    "attn_post_norm.weight",
    "ffn_norm.weight",
    "ffn_post_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "ffn_gate_inp.weight",
    "exp_probs_b.bias",
)


@dataclass(frozen=True)
class KolibriConfig:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    experts: int
    top_k: int
    expert_ffn: int  # each routed expert's SwiGLU width
    shared_ffn: int  # the shared expert's
    window: int  # keys a sliding layer sees, the current one included
    rope_theta: float
    eps: float
    attention_scale: float
    sliding: tuple  # per layer: True for sliding-window attention with RoPE, False for global attention without
    tied_embeddings: bool = False

    @property
    def intermediate(
        self,
    ):  # the per-token MLP scratch width the plan reserves (routed experts' gate/up rows)
        return self.top_k * 2 * self.expert_ffn

    def manifest(self):
        out = asdict(self)
        out["sliding"] = list(self.sliding)
        return dict(out, family=ARCHITECTURE, intermediate=self.intermediate)


def layer_type(config, index):
    return "sliding" if config.sliding[index] else "full"


def expert_chunks(experts, expert_bytes, limit=EXPERT_CHUNK_BYTES):
    """[(first expert, count)]: as few chunks as fit `limit` bytes each, as even as possible."""
    return moe_gguf.expert_chunks(experts, expert_bytes, limit)


class Model:
    """A Kolibri-1 GGUF: its tensors (memory-mapped, packed) and checked configuration."""

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
                "Unsupported Kolibri checkpoint: " + "; ".join(errors)
            )
        g = family.geometry(document, header)
        self.config = KolibriConfig(
            g["layers"],
            g["hidden"],
            g["heads"],
            g["kv_heads"],
            g["head_dim"],
            g["vocab"],
            g["experts"],
            g["top_k"],
            g["expert_ffn"],
            g["shared_ffn"],
            g["window"],
            g["rope_theta"],
            float(g["eps"]),
            float(g["attention_scale"]),
            tuple(g["sliding"]),
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
        """Float32 weights of one expert: gate/up [2*expert_ffn, hidden] or down [hidden, expert_ffn]."""
        flat = self.experts[(index, kind)]
        rows = (
            2 * self.config.expert_ffn
            if kind == "ffn_gate_up_exps"
            else self.config.hidden
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


def route(c, router, bias, h):
    """Top-k experts of each row of h: (indices [T, k] by descending biased score, ties to the lower index;
    weights [T, k] = sigmoid of the unbiased logits; margins [T]: the k-th minus the (k+1)-th biased score,
    how near the selection is to changing)."""
    logits = h.astype(np.float64) @ router.astype(np.float64).T
    scores = logits + bias.astype(np.float64)
    order = np.argsort(-scores, axis=1, kind="stable")
    chosen = order[:, : c.top_k]
    kth = np.take_along_axis(scores, chosen[:, -1:], axis=1)[:, 0]
    nxt = (
        np.take_along_axis(scores, order[:, c.top_k : c.top_k + 1], axis=1)[
            :, 0
        ]
        if c.experts > c.top_k
        else np.inf
    )
    weights = 1.0 / (1.0 + np.exp(-np.take_along_axis(logits, chosen, axis=1)))
    return chosen, weights, kth - nxt


class LayerWeights:
    """One layer's dense weights (dequantized), and its experts dequantized on demand (a few kept)."""

    def __init__(self, model, index, keep=64):
        self.model, self.index, self.keep = model, index, keep
        t = model.layer_tensors(index)
        get = lambda name: t[name].dequantize()  # noqa: E731
        self.attn_norm, self.attn_post_norm = (
            get("attn_norm.weight"),
            get("attn_post_norm.weight"),
        )
        self.ffn_norm, self.ffn_post_norm = (
            get("ffn_norm.weight"),
            get("ffn_post_norm.weight"),
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
        self.router, self.bias = (
            get("ffn_gate_inp.weight"),
            get("exp_probs_b.bias"),
        )
        self.gate_s, self.up_s, self.down_s = (
            get("ffn_gate_shexp.weight"),
            get("ffn_up_shexp.weight"),
            get("ffn_down_shexp.weight"),
        )
        self.cache = OrderedDict()

    def expert(self, e):
        if e in self.cache:
            self.cache.move_to_end(e)
            return self.cache[e]
        value = (
            self.model.expert_rows(self.index, "ffn_gate_up_exps", e),
            self.model.expert_rows(self.index, "ffn_down_exps", e),
        )
        self.cache[e] = value
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return value


def moe(c, w, h, routes=None):
    """The MLP of h[T, hidden]: routed experts (weighted) plus the shared expert. `routes`: appended with
    (indices, weights, margins)."""
    chosen, weights, margins = route(c, w.router, w.bias, h)
    if routes is not None:
        routes.append((chosen, weights, margins))
    out = swiglu(h @ w.gate_s.T, h @ w.up_s.T) @ w.down_s.T
    for e in np.unique(chosen):
        rows, slots = np.nonzero(chosen == e)
        gate_up, down = w.expert(int(e))
        g = h[rows] @ gate_up.T
        y = swiglu(g[:, : c.expert_ffn], g[:, c.expert_ffn :]) @ down.T
        out[rows] += (weights[rows, slots][:, None] * y).astype(np.float32)
    return out


def attention(c, sliding, q, K, V, positions):
    """Grouped-query attention of q[T, heads, d] over the cache K, V [S, kv_heads, d] (float64 softmax)."""
    return moe_gguf.grouped_attention(
        c.heads,
        c.kv_heads,
        c.attention_scale,
        q,
        K,
        V,
        positions,
        c.window if sliding else None,
    )


def layer_forward(c, w, x, cache, position, kv="f32", routes=None):
    """Layer `w.index` over x[tokens, hidden] at `position`, appending to cache = {"k": [], "v": []}."""
    T = x.shape[0]
    sliding = c.sliding[w.index]
    h = rms_norm(x, w.attn_norm, c.eps)
    q = rms_norm((h @ w.q.T).reshape(T, c.heads, c.head_dim), w.q_norm, c.eps)
    k = rms_norm(
        (h @ w.k.T).reshape(T, c.kv_heads, c.head_dim), w.k_norm, c.eps
    )
    v = (h @ w.v.T).reshape(T, c.kv_heads, c.head_dim)
    positions = np.arange(position, position + T)
    if sliding:
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
    out = attention(
        c,
        sliding,
        q,
        np.concatenate(cache["k"]),
        np.concatenate(cache["v"]),
        positions,
    )
    x = x + rms_norm(out.reshape(T, -1) @ w.o.T, w.attn_post_norm, c.eps)
    h = rms_norm(x, w.ffn_norm, c.eps)
    return x + rms_norm(moe(c, w, h, routes), w.ffn_post_norm, c.eps)


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

    def __call__(self, hidden):
        x = np.ascontiguousarray(hidden, dtype=np.float32)
        for i, w, cache in self.layers:
            x = layer_forward(
                self.c, w, x, cache, self.position, self.kv, self.routes[i]
            )
        self.position += x.shape[0]
        return x


def embed(model, tokens):
    return model["token_embd.weight"].dequantize(rows=tokens)


def run_layers(model, x, first, last, kv="f32", routes=None):
    """Layers first..last-1 over the whole sequence in one causal pass, one layer's weights at a time.
    `routes`: a dict filled with each layer's (indices, weights, margins)."""
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
    """Layer `index` as transformers' Kolibri1DecoderLayer state dict (float32 NumPy), from the GGUF tensors."""
    c, w = model.config, LayerWeights(model, index, keep=0)
    experts = [
        model.expert_rows(index, "ffn_gate_up_exps", e)
        for e in range(c.experts)
    ]
    downs = [
        model.expert_rows(index, "ffn_down_exps", e) for e in range(c.experts)
    ]
    return {
        "input_layernorm.weight": w.attn_norm,
        "post_self_attn_layernorm.weight": w.attn_post_norm,
        "post_attention_layernorm.weight": w.ffn_norm,
        "post_mlp_layernorm.weight": w.ffn_post_norm,
        "self_attn.q_proj.weight": w.q,
        "self_attn.k_proj.weight": w.k,
        "self_attn.v_proj.weight": w.v,
        "self_attn.o_proj.weight": w.o,
        "self_attn.q_norm.weight": w.q_norm,
        "self_attn.k_norm.weight": w.k_norm,
        "mlp.gate.weight": w.router,
        "mlp.gate.e_score_correction_bias": w.bias,
        "mlp.experts.gate_up_proj": np.stack(experts),
        "mlp.experts.down_proj": np.stack(downs),
        "mlp.shared_experts.gate_proj.weight": w.gate_s,
        "mlp.shared_experts.up_proj.weight": w.up_s,
        "mlp.shared_experts.down_proj.weight": w.down_s,
    }


def transformers_layer(model, config_path, index, x):
    """The same layer, from position 0 over x[T, hidden], through transformers' own Kolibri1DecoderLayer
    (eager attention and experts, float32): an independent check of the reference's norms, routing, RoPE,
    window and NoPE layers."""
    import torch
    from transformers import AutoConfig
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.models.kolibri1 import modeling_kolibri1 as m

    config = AutoConfig.from_pretrained(
        str(Path(config_path).parent), local_files_only=True
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.quantization_config = None
    layer = m.Kolibri1DecoderLayer(config, index).to(torch.float32).eval()
    layer.load_state_dict(
        {
            k: torch.from_numpy(np.ascontiguousarray(v))
            for k, v in hf_layer_state(model, index).items()
        },
        strict=True,
    )
    T = x.shape[0]
    with torch.no_grad():
        h = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        positions = torch.arange(T)[None]
        embeddings = m.Kolibri1RotaryEmbedding(config=config)(h, positions)
        kwargs = dict(
            config=config,
            inputs_embeds=h,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        try:
            mask = (
                create_sliding_window_causal_mask
                if config.layer_types[index] == "sliding_attention"
                else create_causal_mask
            )(**kwargs)
        except TypeError:  # older signature: cache_position
            kwargs["cache_position"] = torch.arange(T)
            mask = (
                create_sliding_window_causal_mask
                if config.layer_types[index] == "sliding_attention"
                else create_causal_mask
            )(**kwargs)
        out = layer(
            h,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=embeddings,
        )
        out = out[0] if isinstance(out, tuple) else out
    return out[0].numpy()
