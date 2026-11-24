"""Qwen3.5 (9B, S5-Nano, 0.8B) from its GGUF file: packed tensors for WebGPU stages, and the
transformers reference on the same quantized weights (DO-36).

The browser kernels read the GGUF bytes unchanged. llama.cpp's converter
(conversion/qwen.py, `Qwen3NextModel` and `_LinearAttentionVReorderBase`)
changed five things on the way from the Hugging Face checkpoint; `hf_tensor`
undoes them so the unmodified transformers model can run the reference:

- `A_log` is stored as `ssm_a = -exp(A_log)`;
- RMSNorm weights are stored plus one, except the DeltaNet's own norm
  (`ssm_norm`, transformers' `linear_attn.norm`);
- the convolution kernel is squeezed to [channels, 4];
- `dt_bias` is renamed `ssm_dt.bias`;
- when the DeltaNet has more value heads than key heads (9B: 32 and 16),
  the value heads are reordered from grouped (by key head) to tiled order: the V rows of `attn_qkv`, the rows of `attn_gate` (z),
  `ssm_alpha` (a), `ssm_beta` (b), `ssm_a`, `ssm_dt.bias`, the V channels of
  `ssm_conv1d` and the columns of `ssm_out`. In tiled order value head j reads
  key head j mod K (grouped: j div (V/K)); the kernels use that directly.
  With as many value heads as key heads (0.8B: 16 and 16) the order is the
  identity.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

LAYERS = 32
HIDDEN = 4096
K_HEADS, V_HEADS, HEAD_K, HEAD_V = 16, 32, 128, 128  # 9B defaults
HEAD = 256  # attention head width
FULL_EVERY = 4


@dataclass(frozen=True)
class WebGPUConfig:
    """Dimensions supported by the shared Qwen3.5 text shaders.

    Head widths (attention 256, DeltaNet 128), the conv kernel (4), partial
    rotary (64 dims) and the schedule (every fourth layer full attention) are
    fixed by the shaders; query heads must be a whole multiple of KV heads
    (4 for 9B and 0.8B, 6 for Qwen3.8-27B). Layer and head
    counts and the widths vary per model; the defaults are the 9B's, which
    bundles prepared before these fields existed assume.
    """

    hidden: int = 4096
    intermediate: int = 12288
    layers: int = 32
    tied_embeddings: bool = False
    attention_heads: int = 16
    attention_kv_heads: int = 4
    linear_k_heads: int = 16
    linear_v_heads: int = 32

    @property
    def conv_dim(self) -> int:
        """DeltaNet q, k and v channels (the qkv projection and conv)."""
        return 2 * self.linear_k_heads * HEAD_K + self.value_dim

    @property
    def value_dim(self) -> int:
        """DeltaNet value channels (z gate and ssm_out input)."""
        return self.linear_v_heads * HEAD_V

    @property
    def attention_dim(self) -> int:
        """Query (and attention output) channels."""
        return self.attention_heads * HEAD

    @property
    def kv_dim(self) -> int:
        return self.attention_kv_heads * HEAD

    @classmethod
    def from_file(cls, path: str | Path) -> "WebGPUConfig":
        document = json.loads(Path(path).read_text())
        text = document.get("text_config", document)
        expected = {
            "head_dim": HEAD,
            "linear_key_head_dim": HEAD_K,
            "linear_value_head_dim": HEAD_V,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
        }
        for key, value in expected.items():
            if text.get(key) != value:
                raise ValueError(f"Unsupported Qwen WebGPU {key}")
        counts = {
            key: text.get(key)
            for key in (
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "linear_num_key_heads",
                "linear_num_value_heads",
            )
        }
        for key, value in counts.items():
            if type(value) is not int or not 1 <= value <= 256:
                raise ValueError(f"Unsupported Qwen WebGPU {key}")
        layers = counts["num_hidden_layers"]
        if layers % FULL_EVERY:
            raise ValueError("Unsupported Qwen WebGPU num_hidden_layers")
        if counts["num_attention_heads"] % counts["num_key_value_heads"]:
            raise ValueError("Unsupported Qwen WebGPU num_attention_heads")
        if counts["linear_num_value_heads"] % counts["linear_num_key_heads"]:
            raise ValueError("Unsupported Qwen WebGPU linear_num_value_heads")
        kinds = [
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(layers)
        ]
        if text.get("layer_types") != kinds:
            raise ValueError("Unsupported Qwen attention schedule")
        rope = text.get("rope_parameters", {})
        if (
            rope.get("rope_theta") != 10000000
            or rope.get("partial_rotary_factor") != 0.25
            or rope.get("rope_type", "default") != "default"
            or text.get("attention_bias", False)
            or text.get("hidden_act") != "silu"
            or not text.get("attn_output_gate", True)
        ):
            raise ValueError("Unsupported Qwen attention/activation settings")
        hidden, intermediate = text["hidden_size"], text["intermediate_size"]
        for value in (hidden, intermediate):
            if type(value) is not int or value <= 0 or value % 256:
                raise ValueError(
                    "Qwen widths must be positive multiples of 256"
                )
        tied = text.get(
            "tie_word_embeddings", document.get("tie_word_embeddings", False)
        )
        if type(tied) is not bool:
            raise ValueError("tie_word_embeddings must be boolean")
        return cls(
            hidden,
            intermediate,
            layers,
            tied,
            counts["num_attention_heads"],
            counts["num_key_value_heads"],
            counts["linear_num_key_heads"],
            counts["linear_num_value_heads"],
        )

    def manifest(self) -> dict[str, int | bool]:
        return asdict(self)


def layer_type(index):
    return "full" if index % FULL_EVERY == FULL_EVERY - 1 else "linear"


@dataclass
class Tensor:
    name: str
    kind: str  # GGML type name: F32, Q4_K, Q5_K, Q6_K, Q8_0
    shape: tuple  # NumPy order: (rows, columns); rows are output features
    data: np.ndarray  # raw bytes (uint8 view of the memory map)
    base: str | None = (
        None  # a row slice of this stored tensor (a fused GGUF tensor), else itself
    )
    first_row: int = 0  # the slice's first row in `base`

    @property
    def row_bytes(self):
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

        if self.kind == "F32":
            return 4 * self.shape[-1]
        block, size = GGML_QUANT_SIZES[GGMLQuantizationType[self.kind]]
        return self.shape[-1] // block * size

    def dequantize(self, rows=None):
        """Float32 values of all rows, or of the listed rows only."""
        import gguf
        from gguf.constants import GGMLQuantizationType

        qtype = GGMLQuantizationType[self.kind]
        raw = (
            self.data.reshape(-1, self.row_bytes)
            if len(self.shape) > 1
            else self.data
        )
        if rows is not None:
            raw = raw[np.asarray(rows)]
        out = gguf.quants.dequantize(np.ascontiguousarray(raw), qtype)
        return out.astype(np.float32).reshape(
            (len(rows),) + self.shape[1:] if rows is not None else self.shape
        )


class Model:
    family = "qwen35"

    def layer_type(self, index):
        return layer_type(index)

    def __init__(self, path, config_path=None, *, tensor_only=False):
        import gguf

        self.config = (
            WebGPUConfig.from_file(config_path)
            if config_path
            else WebGPUConfig()
        )
        self.path = Path(path)
        if tensor_only:
            from .gguf_metadata import tensor_reader

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

        embedding = self.tensors["token_embd.weight"]
        if embedding.shape[1] != self.config.hidden:
            raise ValueError("Embedding width disagrees with model config")
        for index in range(self.config.layers):
            tensor = self.tensors[f"blk.{index}.ffn_up.weight"]
            if tensor.shape != (self.config.intermediate, self.config.hidden):
                raise ValueError(
                    f"FFN shape disagrees with config: layer {index}"
                )
        if self["output.weight"].shape != embedding.shape:
            raise ValueError("Output head shape disagrees with embedding")

    def __getitem__(self, name):
        if (
            name == "output.weight"
            and name not in self.tensors
            and self.config.tied_embeddings
        ):
            name = "token_embd.weight"
        return self.tensors[name]

    def validate_webgpu(self) -> dict[str, int]:
        """Validate tensors consumed by the current text shaders, without math.

        Extra vision/MTP tensors are intentionally ignored. Float metadata is
        consumed directly as F32; projection matrices use packed decoders.
        """
        hidden, ffn = self.config.hidden, self.config.intermediate
        kinds: dict[str, int] = {}

        def check(name, shape, *, floating=False):
            tensor = self[name]
            allowed = (
                {"F32"}
                if floating
                else {"F32", "F16", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"}
            )
            if tensor.shape != shape or tensor.kind not in allowed:
                raise ValueError(
                    f"Unsupported tensor {name}: "
                    f"{tensor.shape} {tensor.kind}; "
                    f"expected {shape}, one of {sorted(allowed)}"
                )
            kinds[tensor.kind] = kinds.get(tensor.kind, 0) + 1

        vocab = self["token_embd.weight"].shape[0]
        check("token_embd.weight", (vocab, hidden))
        check("output.weight", (vocab, hidden))
        check("output_norm.weight", (hidden,), floating=True)
        for index in range(self.config.layers):
            prefix = f"blk.{index}."
            matrices = {
                "ffn_gate.weight": (ffn, hidden),
                "ffn_up.weight": (ffn, hidden),
                "ffn_down.weight": (hidden, ffn),
            }
            floats = {
                "attn_norm.weight": (hidden,),
                "post_attention_norm.weight": (hidden,),
            }
            c = self.config
            if layer_type(index) == "linear":
                matrices.update(
                    {
                        "attn_qkv.weight": (c.conv_dim, hidden),
                        "attn_gate.weight": (c.value_dim, hidden),
                        "ssm_alpha.weight": (c.linear_v_heads, hidden),
                        "ssm_beta.weight": (c.linear_v_heads, hidden),
                        "ssm_out.weight": (hidden, c.value_dim),
                    }
                )
                floats.update(
                    {
                        "ssm_a": (c.linear_v_heads,),
                        "ssm_dt.bias": (c.linear_v_heads,),
                        "ssm_norm.weight": (HEAD_V,),
                        "ssm_conv1d.weight": (c.conv_dim, 4),
                    }
                )
            else:
                matrices.update(
                    {
                        "attn_q.weight": (2 * c.attention_dim, hidden),
                        "attn_k.weight": (c.kv_dim, hidden),
                        "attn_v.weight": (c.kv_dim, hidden),
                        "attn_output.weight": (hidden, c.attention_dim),
                    }
                )
                floats.update(
                    {
                        "attn_q_norm.weight": (256,),
                        "attn_k_norm.weight": (256,),
                    }
                )
            for name, shape in matrices.items():
                check(prefix + name, shape)
            for name, shape in floats.items():
                check(prefix + name, shape, floating=True)
        return kinds

    def layer_tensors(self, index):
        prefix = f"blk.{index}."
        return {
            name[len(prefix) :]: t
            for name, t in self.tensors.items()
            if name.startswith(prefix)
        }


# --- reference: undo the converter ---------------------------------------


def tiled_to_grouped(n_heads_k, per_k, head_dim):
    """Index map: grouped position -> tiled position, along one axis."""
    tiled = np.arange(n_heads_k * per_k * head_dim).reshape(
        per_k, n_heads_k, head_dim
    )
    return tiled.transpose(1, 0, 2).reshape(-1)


def value_orders(config):
    """(channel order, head order) undoing the converter's V reordering."""
    per_k = config.linear_v_heads // config.linear_k_heads
    return (
        tiled_to_grouped(config.linear_k_heads, per_k, HEAD_V),
        tiled_to_grouped(config.linear_k_heads, per_k, 1),
    )


V_ORDER, H_ORDER = value_orders(WebGPUConfig())  # the 9B's 4096 and 32


def hf_layer(model, index):
    """The layer's transformers state dict (float32), from the GGUF tensors."""
    t = model.layer_tensors(index)
    get = lambda name: t[name].dequantize()  # noqa: E731
    sd = {
        "input_layernorm.weight": get("attn_norm.weight") - 1,
        "post_attention_layernorm.weight": get("post_attention_norm.weight")
        - 1,
        "mlp.gate_proj.weight": get("ffn_gate.weight"),
        "mlp.up_proj.weight": get("ffn_up.weight"),
        "mlp.down_proj.weight": get("ffn_down.weight"),
    }
    if layer_type(index) == "full":
        sd.update(
            {
                "self_attn.q_proj.weight": get("attn_q.weight"),
                "self_attn.k_proj.weight": get("attn_k.weight"),
                "self_attn.v_proj.weight": get("attn_v.weight"),
                "self_attn.o_proj.weight": get("attn_output.weight"),
                "self_attn.q_norm.weight": get("attn_q_norm.weight") - 1,
                "self_attn.k_norm.weight": get("attn_k_norm.weight") - 1,
            }
        )
        return sd
    qk = 2 * model.config.linear_k_heads * HEAD_K
    V_ORDER, H_ORDER = value_orders(model.config)  # noqa: N806
    qkv = get("attn_qkv.weight")
    qkv = np.concatenate([qkv[:qk], qkv[qk:][V_ORDER]])
    conv = get(
        "ssm_conv1d.weight"
    )  # GGUF dims [4, channels]: NumPy (channels, 4)
    conv = np.concatenate([conv[:qk], conv[qk:][V_ORDER]])
    sd.update(
        {
            "linear_attn.in_proj_qkv.weight": qkv,
            "linear_attn.in_proj_z.weight": get("attn_gate.weight")[V_ORDER],
            "linear_attn.in_proj_b.weight": get("ssm_beta.weight")[H_ORDER],
            "linear_attn.in_proj_a.weight": get("ssm_alpha.weight")[H_ORDER],
            "linear_attn.conv1d.weight": conv[:, None, :],
            "linear_attn.A_log": np.log(-get("ssm_a"))[H_ORDER],
            "linear_attn.dt_bias": get("ssm_dt.bias")[H_ORDER],
            "linear_attn.norm.weight": get("ssm_norm.weight"),
            "linear_attn.out_proj.weight": get("ssm_out.weight")[:, V_ORDER],
        }
    )
    return sd


def text_config(config_path):
    import json

    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )

    text = json.loads(Path(config_path).read_text())["text_config"]
    config = Qwen3_5TextConfig(**text)
    config._attn_implementation = "eager"
    return config


def kv_cache(config, kv="f32"):
    """Transformers' DynamicCache; with kv="f16" the attention keys and
    values are rounded to float16 as they are stored, as the WebGPU cache
    stores them (the current tokens' too: attention reads them back)."""
    import torch
    from transformers.cache_utils import DynamicCache

    if kv == "f32":
        return DynamicCache(config=config)
    if kv != "f16":
        raise ValueError(kv)

    class Float16KV(DynamicCache):
        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            round16 = lambda t: t.to(torch.float16).to(t.dtype)  # noqa: E731
            return super().update(
                round16(key_states),
                round16(value_states),
                layer_idx,
                *args,
                **kwargs,
            )

    return Float16KV(config=config)


class StageReference:
    """Transformers' decoder layers `first..last-1` in float32 on the GGUF
    weights, with their cache, for prefill then cached one-token steps."""

    def __init__(self, model, config, first, last, kv="f32"):
        import torch
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m

        self.torch, self.m = torch, m
        self.config, self.first, self.last = config, first, last
        self.head_order = value_orders(model.config)[1]
        self.layers = []
        for index in range(first, last):
            layer = (
                m.Qwen3_5DecoderLayer(config, index).to(torch.float32).eval()
            )
            state = {
                k: torch.from_numpy(np.ascontiguousarray(v))
                for k, v in hf_layer(model, index).items()
            }
            layer.load_state_dict(state, strict=True)
            self.layers.append((index, layer))
        self.rotary = m.Qwen3_5TextRotaryEmbedding(config=config)
        self.cache = kv_cache(config, kv)
        self.position = 0

    def __call__(self, hidden):
        """Accept (tokens, hidden_size) float32 entering layer `first`.

        Return hidden states leaving layer `last-1`.
        """
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(
                np.ascontiguousarray(hidden, dtype=np.float32)
            )[None]
            n = x.shape[1]
            positions = torch.arange(self.position, self.position + n)
            ids = positions.view(1, 1, -1).expand(3, 1, -1)
            embeddings = self.rotary(x, ids)
            mask = None
            if n > 1:
                total = self.position + n
                mask = torch.full((n, total), float("-inf"))
                mask = torch.triu(mask, diagonal=self.position + 1)[None, None]
            for index, layer in self.layers:
                x = layer(
                    x,
                    position_embeddings=embeddings,
                    attention_mask=mask
                    if layer_type(index) == "full"
                    else None,
                    position_ids=positions[None],
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.position += n
            return x[0].numpy()

    def recurrent_states(self):
        """Per linear layer in the stage: (conv state, recurrent state), the
        recurrent one with heads back in tiled (GGUF) order."""
        out = {}
        tiled = np.argsort(self.head_order)
        for index in range(self.first, self.last):
            if layer_type(index) != "linear":
                continue
            conv = self.cache.layers[index].conv_states[0].numpy()[0]
            state = self.cache.layers[index].recurrent_states[0].numpy()[0]
            out[index] = (conv, state[tiled])  # (V heads, 128 k, 128 v)
        return out


def embed(model, tokens):
    return model["token_embd.weight"].dequantize(rows=tokens)
