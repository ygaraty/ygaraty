"""Dense decoder-only transformers (Llama shape) from their GGUF file: the `dense` model family.

Covers llama.cpp's `llama`, `granite`, `mistral` and `phi3` (Phi-4) architectures: RMSNorm, grouped-query
attention with RoPE, SwiGLU FFN, no biases. The browser stage (`dense_gpu.mjs`) reads the GGUF bytes unchanged; the NumPy reference
here computes in the GGUF's own layout, without Torch:

- llama.cpp's converter permutes the rows of `attn_q` and `attn_k` per head so that RoPE rotates adjacent
  pairs (2i, 2i+1) with frequency theta^(-2i/d) ("normal" RoPE); transformers rotates halves (i, i+d/2). The
  reference and the stage use the adjacent pairs; `hf_tensor` undoes the permutation for a transformers
  cross-check (`transformers_layer`).
- `phi3` is converted without that permutation, so its RoPE rotates halves (`rope_pairs="halves"`). It stores
  transformers' fused `qkv_proj` as `attn_qkv` ([q; k; v] rows) and `gate_up_proj` as `ffn_up` ([gate; up]
  rows); `Model` serves their row ranges as `attn_q`/`attn_k`/`attn_v` and `ffn_gate`/`ffn_up` (K-quant rows
  are whole, so a slice is a byte range of the stored tensor: bundles hard-link the GGUF as for any tensor).
- RMSNorm weights are stored as they are (no +1 offset, unlike Qwen3.5's GGUF).
- Granite's attention multiplier replaces 1/sqrt(head_dim); its embedding, residual and logit multipliers must
  be 1 (`families.Dense.check`).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.checkpoints import families, gguf_remote
from ygaraty.backends.webgpu.checkpoints.qwen35_gguf import Tensor


@dataclass(frozen=True)
class DenseConfig:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    rope_theta: float
    eps: float
    attention_scale: float
    tied_embeddings: bool
    rope_pairs: str = "adjacent"  # "adjacent" (2i, 2i+1): llama.cpp's permuted Q/K rows; "halves" (i, i + d/2)

    def manifest(self):
        return dict(asdict(self), family="dense")


def layer_type(index):
    return "dense"


def row_slice(t, name, first, rows):
    """Rows first..first+rows-1 of a stored tensor, as a tensor of its own named `name`."""
    width = t.row_bytes
    return Tensor(
        name,
        t.kind,
        (rows,) + t.shape[1:],
        t.data[first * width : (first + rows) * width],
        base=t.name,
        first_row=first,
    )


class Model:
    """A dense GGUF checkpoint: its tensors (memory-mapped, packed) and checked configuration."""

    family = "dense"

    def layer_type(self, index):
        return layer_type(index)

    def __init__(self, path, config_path, *, tensor_only=False):
        import gguf

        self.path = Path(path)
        document = json.loads(Path(config_path).read_text())
        header = gguf_remote.read_header(self.path)
        findings = families.FAMILIES["dense"].check(document, header)
        errors = [f["message"] for f in findings if f["level"] == "error"]
        if errors:
            raise ValueError(
                "Unsupported dense checkpoint: " + "; ".join(errors)
            )
        g = families.FAMILIES["dense"].geometry(document, header)
        self.config = DenseConfig(
            g["layers"],
            g["hidden"],
            g["intermediate"],
            g["heads"],
            g["kv_heads"],
            g["head_dim"],
            header.tensors["token_embd.weight"].shape[0],
            g["rope_theta"],
            float(g["eps"]),
            float(g["attention_scale"]),
            "output.weight" not in header.tensors,
            g["rope_pairs"],
        )
        if tensor_only:
            from ygaraty.backends.webgpu.checkpoints.gguf_metadata import tensor_reader

            self.reader = tensor_reader(self.path)
        else:
            self.reader = gguf.GGUFReader(self.path)
        self.tensors = {}
        for t in self.reader.tensors:
            shape = tuple(int(n) for n in reversed(t.shape.tolist()))
            self.tensors[t.name] = Tensor(
                t.name,
                t.tensor_type.name,
                shape,
                t.data.view(np.uint8).reshape(-1),
            )
        c = self.config
        q, kv, ffn = (
            c.heads * c.head_dim,
            c.kv_heads * c.head_dim,
            c.intermediate,
        )
        for i in range(c.layers):
            p = f"blk.{i}."
            if p + "attn_qkv.weight" in self.tensors:
                fused = self.tensors.pop(p + "attn_qkv.weight")
                for name, first, rows in (
                    ("attn_q", 0, q),
                    ("attn_k", q, kv),
                    ("attn_v", q + kv, kv),
                ):
                    self.tensors[p + name + ".weight"] = row_slice(
                        fused, p + name + ".weight", first, rows
                    )
            if (
                p + "ffn_gate.weight" not in self.tensors
            ):  # fused [gate; up], checked by the family
                fused = self.tensors.pop(p + "ffn_up.weight")
                for name, first in (("ffn_gate", 0), ("ffn_up", ffn)):
                    self.tensors[p + name + ".weight"] = row_slice(
                        fused, p + name + ".weight", first, ffn
                    )

    def __getitem__(self, name):
        if (
            name == "output.weight"
            and name not in self.tensors
            and self.config.tied_embeddings
        ):
            name = "token_embd.weight"
        return self.tensors[name]

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


# --- NumPy reference, in the GGUF's layout ------------------------------------


def rms_norm(x, w, eps):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions, theta, pairs="adjacent"):
    """RoPE over the last axis (head_dim) of x[tokens, heads, head_dim], in place: pair i is (2i, 2i+1)
    ("adjacent") or (i, i + d/2) ("halves"), rotated by position * theta^(-2i/d)."""
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    angle = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    c, s = (
        np.cos(angle).astype(np.float32)[:, None, :],
        np.sin(angle).astype(np.float32)[:, None, :],
    )
    first, second = (
        (np.s_[..., 0::2], np.s_[..., 1::2])
        if pairs == "adjacent"
        else (np.s_[..., : d // 2], np.s_[..., d // 2 :])
    )
    a, b = x[first].copy(), x[second].copy()
    x[first] = a * c - b * s
    x[second] = b * c + a * s
    return x


class LayerWeights:
    def __init__(self, model, index):
        t = model.layer_tensors(index)
        get = lambda name: t[name].dequantize()  # noqa: E731
        self.attn_norm, self.ffn_norm = (
            get("attn_norm.weight"),
            get("ffn_norm.weight"),
        )
        self.q, self.k, self.v, self.o = (
            get("attn_q.weight"),
            get("attn_k.weight"),
            get("attn_v.weight"),
            get("attn_output.weight"),
        )
        self.gate, self.up, self.down = (
            get("ffn_gate.weight"),
            get("ffn_up.weight"),
            get("ffn_down.weight"),
        )


def layer_forward(c, w, x, cache, position, kv="f32"):
    """One layer over x[tokens, hidden] at `position`, appending to cache = {"k": [], "v": []}."""
    T = x.shape[0]
    h = rms_norm(x, w.attn_norm, c.eps)
    q = (h @ w.q.T).reshape(T, c.heads, c.head_dim)
    k = (h @ w.k.T).reshape(T, c.kv_heads, c.head_dim)
    v = (h @ w.v.T).reshape(T, c.kv_heads, c.head_dim)
    positions = np.arange(position, position + T)
    rope(q, positions, c.rope_theta, c.rope_pairs)
    rope(k, positions, c.rope_theta, c.rope_pairs)
    if (
        kv == "f16"
    ):  # as the cache stores them (the current tokens' too: attention reads them back)
        k, v = (
            k.astype(np.float16).astype(np.float32),
            v.astype(np.float16).astype(np.float32),
        )
    cache["k"].append(k)
    cache["v"].append(v)
    K, V = (
        np.concatenate(cache["k"]),
        np.concatenate(cache["v"]),
    )  # [S, kv_heads, d]
    group = c.heads // c.kv_heads
    out = np.empty((T, c.heads, c.head_dim), dtype=np.float32)
    S = K.shape[0]
    mask = (
        np.arange(S)[None, :] > (positions[:, None])
    )  # causal: key s visible to token t if s <= position+t
    for hh in range(c.heads):
        g = hh // group
        scores = (
            q[:, hh, :].astype(np.float64) @ K[:, g, :].T.astype(np.float64)
        ) * c.attention_scale
        scores[mask] = -np.inf
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        out[:, hh, :] = (p @ V[:, g, :].astype(np.float64)).astype(np.float32)
    x = x + out.reshape(T, -1) @ w.o.T
    h = rms_norm(x, w.ffn_norm, c.eps)
    a, b = h @ w.gate.T, h @ w.up.T
    return x + ((a / (1.0 + np.exp(-a))) * b) @ w.down.T


class StageReference:
    """Layers first..last-1 with their KV cache, for prefill then cached steps (weights kept dequantized:
    meant for a few layers)."""

    def __init__(self, model, first, last, kv="f32"):
        self.c, self.kv, self.position = model.config, kv, 0
        self.layers = [
            (i, LayerWeights(model, i), {"k": [], "v": []})
            for i in range(first, last)
        ]

    def __call__(self, hidden):
        x = np.ascontiguousarray(hidden, dtype=np.float32)
        for _, w, cache in self.layers:
            x = layer_forward(self.c, w, x, cache, self.position, self.kv)
        self.position += x.shape[0]
        return x


def embed(model, tokens):
    return model["token_embd.weight"].dequantize(rows=tokens)


def run_layers(model, x, first, last, kv="f32"):
    """Layers first..last-1 over the whole sequence in one causal pass, one layer's weights at a time."""
    for i in range(first, last):
        x = layer_forward(
            model.config, LayerWeights(model, i), x, {"k": [], "v": []}, 0, kv
        )
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


def unpermute(w, heads):
    """Undo llama.cpp's per-head Q/K row permutation (adjacent pairs back to rotate-half halves)."""
    rows = w.shape[0]
    return (
        w.reshape(heads, rows // heads // 2, 2, *w.shape[1:])
        .swapaxes(1, 2)
        .reshape(w.shape)
    )


def transformers_layer(model, config_path, index, x, kv="f32"):
    """The same layer through transformers' own decoder layer (Granite, Llama or Phi-3) on un-permuted
    (re-fused, for Phi-3) weights: an independent check of the reference's RoPE convention, multipliers,
    tensor splits and attention."""
    import torch
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )  # noqa: F401
    import importlib

    config = AutoConfig.from_pretrained(
        str(Path(config_path).parent), local_files_only=True
    )
    config._attn_implementation = "eager"
    kind = config.model_type
    module = importlib.import_module(
        f"transformers.models.{kind}.modeling_{kind}"
    )
    Layer = getattr(module, f"{kind.capitalize()}DecoderLayer")
    Rotary = getattr(module, f"{kind.capitalize()}RotaryEmbedding")
    c, w = model.config, LayerWeights(model, index)
    layer = Layer(config, index).to(torch.float32).eval()
    state = {
        "input_layernorm.weight": w.attn_norm,
        "post_attention_layernorm.weight": w.ffn_norm,
        "self_attn.o_proj.weight": w.o,
        "mlp.down_proj.weight": w.down,
    }
    if kind == "phi3":
        state.update(
            {
                "self_attn.qkv_proj.weight": np.concatenate([w.q, w.k, w.v]),
                "mlp.gate_up_proj.weight": np.concatenate([w.gate, w.up]),
            }
        )
    else:
        state.update(
            {
                "self_attn.q_proj.weight": unpermute(w.q, c.heads),
                "self_attn.k_proj.weight": unpermute(w.k, c.kv_heads),
                "self_attn.v_proj.weight": w.v,
                "mlp.gate_proj.weight": w.gate,
                "mlp.up_proj.weight": w.up,
            }
        )
    layer.load_state_dict(
        {
            k: torch.from_numpy(np.ascontiguousarray(v))
            for k, v in state.items()
        },
        strict=True,
    )
    T = x.shape[0]
    with torch.no_grad():
        h = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]
        positions = torch.arange(T)[None]
        embeddings = Rotary(config=config)(h, positions)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[
            None, None
        ]
        out = layer(
            h,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=embeddings,
        )
        out = out[0] if isinstance(out, tuple) else out
    return out[0].numpy()
