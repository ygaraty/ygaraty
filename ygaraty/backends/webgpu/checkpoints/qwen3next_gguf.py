"""Qwen3-Next (`qwen3next`, Qwen3-Coder-Next) from llama.cpp's GGUF: the loader and a NumPy reference that
computes in the GGUF's own (legacy) layout, without Torch, following transformers' `modeling_qwen3_next.py`.
The family's admission and the layout are described in `families.Qwen3Next`; in short, per layer:

- `attn_norm` (RMSNorm), then the token mixer, added to the residual; `post_attention_norm`, then the MLP, added.
- Token mixer of a `linear_attention` layer, a Gated DeltaNet: `ssm_in` gives per key head [q, k, v, z] and
  `ssm_ba` per key head [b, a] (grouped value heads: value head j belongs to key head j div r, r = Vh / Kh).
  The channels [q of all key heads, k of all key heads, v of all value heads] pass a causal depthwise convolution
  (`ssm_conv1d`, kernel 4, no bias) and SiLU. q and k are L2-normalized (eps 1e-6) and repeated to the value
  heads, q scaled by Dk^-1/2; beta = sigmoid(b), decay g = `ssm_a` * softplus(a + `ssm_dt.bias`) (ssm_a is
  -exp(A_log)). Per value head the state S [Dk, Dv] is decayed by exp(g), updated by the delta rule
  S += k (beta (v - S^T k))^T, and read by q: o = S^T q. o is RMS-normalized per head (`ssm_norm`, eps), times
  SiLU(z), then `ssm_out`.
- Token mixer of a `full_attention` layer: `attn_q` gives per head [query, gate]; per-head RMSNorm on Q and K;
  RoPE over halves of the first `rope_dims` dims of each head; causal grouped-query attention; the output times
  sigmoid(gate), then `attn_output`.
- MLP: softmax router over all experts, top k renormalized (`qwen3moe_gguf.route`), SwiGLU experts; plus the
  shared expert SwiGLU times sigmoid(`ffn_gate_inp_shexp` . h).

RMSNorm weights in the GGUF already include transformers' +1 (checked on the real Coder-Next artifact: norms near
1, `ssm_norm` near 0.01, `ssm_a` negative). Per-sequence state: each DeltaNet layer's S [Vh, Dk, Dv] and its last
kernel-1 convolution inputs; each attention layer's K and V.

`Model` reads an unsplit GGUF or every shard of a split one (name the first, `X-00001-of-0000N.gguf`) and serves
expert tensors as chunks of whole experts (`moe_gguf.read_tensors`). Routing ties follow `qwen3moe_gguf.route`.
"""

import json
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.checkpoints import (
    families,
    gguf_remote,
    moe_gguf,
    qwen3moe_gguf,
)
from ygaraty.backends.webgpu.checkpoints.moe_gguf import rms_norm, rope_halves, swiglu

ARCHITECTURE = "qwen3next"
# YGARATY_QWEN3NEXT_EXPERT_CHUNK_BYTES: smaller chunks for tests on tiny models (several chunks per layer)
EXPERT_CHUNK_BYTES = int(
    os.environ.get("YGARATY_QWEN3NEXT_EXPERT_CHUNK_BYTES", 128 * 2**20)
)
EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
L2_EPS = 1e-6  # transformers' l2norm (as the FLA kernels)


@dataclass(frozen=True)
class Qwen3NextConfig:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    rope_theta: float
    rope_dims: int
    eps: float
    attention_scale: float
    layer_types: tuple
    linear_key_heads: int
    linear_value_heads: int
    linear_key_dim: int
    linear_value_dim: int
    conv_kernel: int
    experts: int
    top_k: int
    expert_ffn: int
    shared_ffn: int
    norm_topk: bool
    tied_embeddings: bool = False

    @property
    def query_width(self):
        return self.heads * self.head_dim

    @property
    def intermediate(
        self,
    ):  # the per-token MLP scratch width the plan reserves (the routed experts' gate/up rows)
        return self.top_k * 2 * self.expert_ffn

    @property
    def value_group(self):  # value heads per key head (r)
        return self.linear_value_heads // self.linear_key_heads

    def manifest(self):
        return dict(
            asdict(self), family=ARCHITECTURE, intermediate=self.intermediate
        )


def layer_type(config, index):
    return (
        "full" if config.layer_types[index] == "full_attention" else "linear"
    )


def shard_paths(path):
    """The local files of a GGUF: itself, or every shard of a split GGUF named by its first."""
    path = Path(path)
    return [
        path.with_name(name) for name in gguf_remote.split_names(path.name)
    ]


class Model:
    """A Qwen3-Next GGUF: its tensors (memory-mapped, packed) and checked configuration."""

    family = ARCHITECTURE

    def layer_type(self, index):
        return layer_type(self.config, index)

    def __init__(self, path, config_path, *, tensor_only=False):
        import gguf

        self.path = Path(path)
        self.paths = shard_paths(self.path)
        document = json.loads(Path(config_path).read_text())
        header = gguf_remote.read_split_headers(self.paths)
        family = families.FAMILIES[ARCHITECTURE]
        errors = [
            f["message"]
            for f in family.admit(document, header)
            if f["level"] == "error"
        ]
        if errors:
            raise ValueError(
                "Unsupported Qwen3-Next checkpoint: " + "; ".join(errors)
            )
        g = family.geometry(document, header)
        self.config = Qwen3NextConfig(
            g["layers"],
            g["hidden"],
            g["heads"],
            g["kv_heads"],
            g["head_dim"],
            g["vocab"],
            g["rope_theta"],
            g["rope_dims"],
            float(g["eps"]),
            float(g["attention_scale"]),
            tuple(g["layer_types"]),
            g["linear_key_heads"],
            g["linear_value_heads"],
            g["linear_key_dim"],
            g["linear_value_dim"],
            g["conv_kernel"],
            g["experts"],
            g["top_k"],
            g["expert_ffn"],
            g["shared_ffn"],
            g["norm_topk"],
        )
        self.tensors, self.experts, self.readers = {}, {}, []
        for shard in self.paths:
            if tensor_only:
                from ygaraty.backends.webgpu.checkpoints.gguf_metadata import (
                    tensor_reader,
                )

                reader = tensor_reader(shard)
            else:
                reader = gguf.GGUFReader(shard)
            self.readers.append(reader)
            tensors, experts = moe_gguf.read_tensors(
                reader, EXPERT_CHUNK_BYTES
            )
            self.tensors.update(tensors)
            self.experts.update(experts)

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

    def layer_tensors(self, index):
        prefix = f"blk.{index}."
        return {
            name[len(prefix) :]: t
            for name, t in self.tensors.items()
            if name.startswith(prefix)
        }

    def expert_chunk_names(self, index, kind):
        return moe_gguf.chunk_names(self.layer_tensors(index), kind)

    def validate_webgpu(self):
        kinds = {}
        for t in self.tensors.values():
            kinds[t.kind] = kinds.get(t.kind, 0) + 1
        return kinds


# --- NumPy reference, in the GGUF's layout ------------------------------------


class LayerWeights:
    """One layer's dense weights (dequantized), and its experts dequantized on demand (a few kept)."""

    def __init__(self, model, index, keep=64):
        self.model, self.index, self.keep = model, index, keep
        self.kind = model.layer_type(index)
        t = model.layer_tensors(index)
        get = lambda name: t[name].dequantize().astype(np.float32)  # noqa: E731
        self.attn_norm, self.ffn_norm = (
            get("attn_norm.weight"),
            get("post_attention_norm.weight"),
        )
        if self.kind == "full":
            self.q_norm, self.k_norm = (
                get("attn_q_norm.weight"),
                get("attn_k_norm.weight"),
            )
            self.q, self.k, self.v, self.o = (
                get(f"attn_{n}.weight") for n in ("q", "k", "v", "output")
            )
        else:
            self.qkvz, self.ba = get("ssm_in.weight"), get("ssm_ba.weight")
            self.conv, self.a, self.dt_bias = (
                get("ssm_conv1d.weight"),
                get("ssm_a"),
                get("ssm_dt.bias"),
            )
            self.ssm_norm, self.out = (
                get("ssm_norm.weight"),
                get("ssm_out.weight"),
            )
        self.router = get("ffn_gate_inp.weight")
        self.shared_gate = get("ffn_gate_inp_shexp.weight")
        self.shared = (
            get("ffn_gate_shexp.weight"),
            get("ffn_up_shexp.weight"),
            get("ffn_down_shexp.weight"),
        )
        self.cache = OrderedDict()

    def expert(self, e):
        if e in self.cache:
            self.cache.move_to_end(e)
            return self.cache[e]
        m = self.model
        value = tuple(
            m.expert_rows(self.index, kind, e) for kind in EXPERT_TENSORS
        )
        self.cache[e] = value
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return value


def new_state(c, kind):
    """A sequence's empty state for one layer: KV cache lists, or the DeltaNet's S and convolution history."""
    if kind == "full":
        return {"k": [], "v": []}
    channels = (
        2 * c.linear_key_heads * c.linear_key_dim
        + c.linear_value_heads * c.linear_value_dim
    )
    return {
        "S": np.zeros(
            (c.linear_value_heads, c.linear_key_dim, c.linear_value_dim),
            np.float32,
        ),
        "conv": np.zeros((c.conv_kernel - 1, channels), np.float32),
    }


def _l2norm(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + L2_EPS)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def deltanet(c, w, h, state):
    """The Gated DeltaNet of h[T, hidden], continuing `state` (updated in place); returns [T, hidden]."""
    T, Kh, Vh, Dk, Dv, r = (
        h.shape[0],
        c.linear_key_heads,
        c.linear_value_heads,
        c.linear_key_dim,
        c.linear_value_dim,
        c.value_group,
    )
    p = (h @ w.qkvz.T).reshape(T, Kh, 2 * Dk + 2 * r * Dv)
    q, k = p[..., :Dk], p[..., Dk : 2 * Dk]
    v = p[..., 2 * Dk : 2 * Dk + r * Dv].reshape(
        T, Vh, Dv
    )  # grouped: value head kh*r + i
    z = p[..., 2 * Dk + r * Dv :].reshape(T, Vh, Dv)
    ba = (h @ w.ba.T).reshape(T, Kh, 2 * r)
    b, a = ba[..., :r].reshape(T, Vh), ba[..., r:].reshape(T, Vh)
    mixed = np.concatenate(
        [q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], axis=1
    ).astype(np.float32)
    history = np.concatenate(
        [state["conv"], mixed]
    )  # [kernel - 1 + T, channels]
    K = c.conv_kernel
    conv = sum(history[j : j + T] * w.conv[:, j] for j in range(K))
    state["conv"] = history[-(K - 1) :].copy()
    conv = conv * _sigmoid(conv)  # SiLU
    q = conv[:, : Kh * Dk].reshape(T, Kh, Dk)
    k = conv[:, Kh * Dk : 2 * Kh * Dk].reshape(T, Kh, Dk)
    v = conv[:, 2 * Kh * Dk :].reshape(T, Vh, Dv)
    q = (
        np.repeat(_l2norm(q), r, axis=1) * Dk**-0.5
    )  # value head j reads key head j div r
    k = np.repeat(_l2norm(k), r, axis=1)
    beta = _sigmoid(b)
    decay = np.exp(w.a * _softplus(a + w.dt_bias))  # [T, Vh]
    S = state["S"].astype(np.float64)
    out = np.empty((T, Vh, Dv), np.float64)
    for t in range(T):
        S *= decay[t][:, None, None]
        memory = np.einsum("hkv,hk->hv", S, k[t])
        delta = (v[t] - memory) * beta[t][:, None]
        S += k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", S, q[t])
    state["S"] = S.astype(np.float32)
    o = rms_norm(out.astype(np.float32), w.ssm_norm, c.eps) * (z * _sigmoid(z))
    return o.reshape(T, Vh * Dv) @ w.out.T


def attention(c, w, h, state, position, kv="f32"):
    """Gated full attention of h[T, hidden] at `position`, appending to state = {"k": [], "v": []}."""
    T, D = h.shape[0], c.head_dim
    qg = (h @ w.q.T).reshape(T, c.heads, 2 * D)
    q, gate = (
        rms_norm(qg[..., :D], w.q_norm, c.eps),
        qg[..., D:].reshape(T, -1),
    )
    k = rms_norm((h @ w.k.T).reshape(T, c.kv_heads, D), w.k_norm, c.eps)
    v = (h @ w.v.T).reshape(T, c.kv_heads, D)
    positions = np.arange(position, position + T)
    q, k = (
        np.ascontiguousarray(q, np.float32),
        np.ascontiguousarray(k, np.float32),
    )
    rope_halves(
        q[..., : c.rope_dims], positions, c.rope_theta
    )  # a view: rotated in place
    rope_halves(k[..., : c.rope_dims], positions, c.rope_theta)
    if (
        kv == "f16"
    ):  # as the cache stores them (the current tokens' too: attention reads them back)
        k, v = (
            k.astype(np.float16).astype(np.float32),
            v.astype(np.float16).astype(np.float32),
        )
    state["k"].append(k)
    state["v"].append(v)
    out = moe_gguf.grouped_attention(
        c.heads,
        c.kv_heads,
        c.attention_scale,
        q,
        np.concatenate(state["k"]),
        np.concatenate(state["v"]),
        positions,
    )
    return (out.reshape(T, -1) * _sigmoid(gate)) @ w.o.T


def mlp(c, w, h, routes=None, forced=None):
    """The routed experts (`qwen3moe_gguf.moe`) plus the gated shared expert."""
    gate, up, down = w.shared
    shared = swiglu(h @ gate.T, h @ up.T) @ down.T
    return (
        qwen3moe_gguf.moe(c, w, h, routes, forced)
        + _sigmoid(h @ w.shared_gate)[:, None] * shared
    )


def layer_forward(
    c, w, x, state, position, kv="f32", routes=None, forced=None
):
    """Layer `w.index` over x[tokens, hidden] at `position`, continuing its `state` (`new_state`)."""
    h = rms_norm(x, w.attn_norm, c.eps)
    mixer = (
        attention(c, w, h, state, position, kv)
        if w.kind == "full"
        else deltanet(c, w, h, state)
    )
    x = (x + mixer).astype(np.float32)
    return (
        x + mlp(c, w, rms_norm(x, w.ffn_norm, c.eps), routes, forced)
    ).astype(np.float32)


class StageReference:
    """Layers first..last-1 with their state, for prefill then cached steps. `routes[layer]`: each call's
    (indices, weights, margins)."""

    def __init__(self, model, first, last, kv="f32"):
        self.c, self.kv, self.position = model.config, kv, 0
        self.layers = [
            (
                i,
                LayerWeights(model, i),
                new_state(model.config, model.layer_type(i)),
            )
            for i in range(first, last)
        ]
        self.routes = {i: [] for i in range(first, last)}
        self.forced = None  # {layer: expert ids [T, k]} for the next call, or None to choose

    def __call__(self, hidden):
        x = np.ascontiguousarray(hidden, dtype=np.float32)
        for i, w, state in self.layers:
            forced = self.forced.get(i) if self.forced else None
            x = layer_forward(
                self.c,
                w,
                x,
                state,
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
            new_state(model.config, model.layer_type(i)),
            0,
            kv,
            log,
        )
        if routes is not None:
            routes[i] = log[0]
    return x


def logits(model, hidden):
    """Final norm and output projection of hidden[positions, hidden] (float64)."""
    return qwen3moe_gguf.logits(model, hidden)


# --- transformers cross-check (optional Torch) --------------------------------


def hf_layer_state(model, index):
    """Layer `index` as transformers' Qwen3NextDecoderLayer state dict (float32 NumPy), from the GGUF tensors:
    undoes the converter's +1 on RMSNorm weights (not `ssm_norm`), `ssm_a` = -exp(A_log) and the squeezed
    convolution; the fused DeltaNet projections are transformers' own. Experts fused as transformers 5 stores
    them: gate_up_proj [experts, 2F, hidden] (gate rows, then up), down_proj [experts, hidden, F]."""
    c, w = model.config, LayerWeights(model, index, keep=0)
    rows = {
        kind: [model.expert_rows(index, kind, e) for e in range(c.experts)]
        for kind in EXPERT_TENSORS
    }
    gate, up, down = w.shared
    out = {
        "input_layernorm.weight": w.attn_norm - 1,
        "post_attention_layernorm.weight": w.ffn_norm - 1,
        "mlp.gate.weight": w.router,
        "mlp.experts.gate_up_proj": np.stack(
            [
                np.concatenate([g, u])
                for g, u in zip(rows["ffn_gate_exps"], rows["ffn_up_exps"])
            ]
        ),
        "mlp.experts.down_proj": np.stack(rows["ffn_down_exps"]),
        "mlp.shared_expert.gate_proj.weight": gate,
        "mlp.shared_expert.up_proj.weight": up,
        "mlp.shared_expert.down_proj.weight": down,
        "mlp.shared_expert_gate.weight": w.shared_gate[None, :],
    }
    if w.kind == "full":
        out.update(
            {
                "self_attn.q_proj.weight": w.q,
                "self_attn.k_proj.weight": w.k,
                "self_attn.v_proj.weight": w.v,
                "self_attn.o_proj.weight": w.o,
                "self_attn.q_norm.weight": w.q_norm - 1,
                "self_attn.k_norm.weight": w.k_norm - 1,
            }
        )
    else:
        out.update(
            {
                "linear_attn.in_proj_qkvz.weight": w.qkvz,
                "linear_attn.in_proj_ba.weight": w.ba,
                "linear_attn.conv1d.weight": w.conv[:, None, :],
                "linear_attn.dt_bias": w.dt_bias,
                "linear_attn.A_log": np.log(-w.a),
                "linear_attn.norm.weight": w.ssm_norm,
                "linear_attn.out_proj.weight": w.out,
            }
        )
    return out


def hf_state(model):
    """The whole model as transformers' Qwen3NextForCausalLM state dict (float32 NumPy)."""
    out = {
        "model.embed_tokens.weight": model["token_embd.weight"].dequantize(),
        "model.norm.weight": model["output_norm.weight"].dequantize() - 1,
        "lm_head.weight": model["output.weight"].dequantize(),
    }
    for i in range(model.config.layers):
        out.update(
            {
                f"model.layers.{i}.{k}": v
                for k, v in hf_layer_state(model, i).items()
            }
        )
    return out
