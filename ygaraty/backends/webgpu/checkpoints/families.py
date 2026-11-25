"""Model families: which execution path runs a checkpoint, and what it checks first.

A family owns one WebGPU stage implementation, its GGUF loader and reference, its device self-test cases and
its chat format. Onboarding a checkpoint means finding its family (`detect`) and passing that family's checks
(`check`); a new architecture is a new family, not a copy of an existing one.

- `qwen35`: Qwen3.5 text models (3 DeltaNet layers per full-attention layer): Qwen3.5-9B, S5-nano.
  Stage `qwen35_gpu.mjs`, loader `qwen35_gguf.py`.
- `dense`: dense decoder-only transformers of the Llama shape (RMSNorm, grouped-query attention with RoPE,
  SwiGLU FFN, no biases), as llama.cpp stores them (`llama`, `granite`, `mistral`, `phi3` GGUF architectures).
  Stage `dense_gpu.mjs`, loader `dense_gguf.py`. Granite's attention multiplier is supported; its
  embedding, residual and logit multipliers must be 1 (true of Granite 4). `phi3` (Phi-4) stores Q/K/V as one
  `attn_qkv` and gate/up as one `ffn_up` (the loader serves their row ranges as separate tensors) and its Q/K
  rows unpermuted, so RoPE rotates halves (`rope_pairs`).
- `kolibri1`: Aleph Alpha's Kolibri-1, a mixture of experts in every layer (top-k of many experts by a biased
  sigmoid router, plus a shared expert), per-head Q/K norms, norms after attention and after the MLP, and
  sliding-window attention with RoPE alternating with global attention without it. Our own GGUF
  (`kolibri_convert.py`; llama.cpp has no such architecture). Stage `kolibri_gpu.mjs`, loader `kolibri_gguf.py`.
- `qwen3moe`: Qwen3 mixture-of-experts text models (Qwen3-Coder-30B-A3B, DO-80): full causal grouped-query
  attention with per-head Q/K norms and RoPE over halves, and in every layer the top k of many experts by a
  softmax router, renormalized (`norm_topk_prob`), without a shared expert or post-branch norms; hidden and
  query widths differ. llama.cpp's GGUF (`qwen3moe`). Loader `qwen3moe_gguf.py`; stage `kolibri_gpu.mjs` (its qwen3moe profile).
- `qwen3next`: Qwen3-Next hybrid MoE text models (Qwen3-Coder-Next): Gated DeltaNet layers with one gated
  full-attention layer every `full_attention_interval`, and in every layer softmax-routed experts plus a shared
  expert scaled by a sigmoid gate. llama.cpp's legacy layout (see `Qwen3Next`). Loader and NumPy reference
  `qwen3next_gguf.py`; **no stage yet**: `check` (= `admit` + the stage's finding) always reports that, so
  `inspect` never calls it compatible and `load_model` refuses it.
- `inkling2`: Thinking Machines' Inkling 2 Mini (`inkling2_loop` GGUF from ggml-org): a looped decoder (24 stored
  layers, 48 visits per token), block attention residuals, attention without RoPE (relative-position bias, sink,
  output gate, Q/K clips) with short convolutions, and sigmoid-routed experts in every layer (DO-95). Loader and
  NumPy reference `inkling2_gguf.py`; **no stage yet**, admission only as for `qwen3next`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PACKED = {
    "Q4_0",
    "IQ4_NL",
    "F32",
    "F16",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
}  # what the WebGPU matrix kernels decode


def finding(level, message):
    return dict(level=level, message=message)


@dataclass(frozen=True)
class Family:
    name: str
    stage_module: str

    def check(self, config, header):  # -> list of findings
        raise NotImplementedError

    def geometry(self, config, header):
        raise NotImplementedError

    def layer_bytes(self, header):
        layers = int(header.get("block_count"))
        sizes = [0] * layers
        for name, t in header.tensors.items():
            if name.startswith("blk."):
                index = int(name.split(".")[1])
                if index < layers:
                    sizes[index] += t.nbytes
        return sizes

    def endpoint_bytes(self, header):
        t = header.tensors
        embedding = t["token_embd.weight"].nbytes
        output = (
            t["output.weight"].nbytes if "output.weight" in t else embedding
        )  # tied: stored again on the last stage
        return dict(
            embedding=embedding,
            output=output,
            output_norm=t["output_norm.weight"].nbytes,
        )


class Qwen35(Family):
    def check(self, config, header):
        import tempfile
        from ygaraty.backends.webgpu.checkpoints.qwen35_gguf import WebGPUConfig

        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(config, f)
            f.flush()
            try:
                WebGPUConfig.from_file(f.name)
            except ValueError as error:
                return [finding("error", str(error))]
        return []

    def geometry(self, config, header):
        text = config.get("text_config", config)
        layers = text["num_hidden_layers"]
        conv = (
            2 * text["linear_num_key_heads"] * text["linear_key_head_dim"]
            + text["linear_num_value_heads"] * text["linear_value_head_dim"]
        )
        return dict(
            layers=layers,
            hidden=text["hidden_size"],
            intermediate=text["intermediate_size"],
            heads=text["num_attention_heads"],
            kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            vocab=header.tensors["token_embd.weight"].shape[0],
            context=header.get("context_length"),
            tied_embeddings="output.weight" not in header.tensors,
            # only the full-attention layers keep a KV cache; the DeltaNet layers a fixed state
            kv_layers=[i for i in range(layers) if i % 4 == 3],
            state_bytes=4
            * (
                text["linear_num_value_heads"]
                * text["linear_key_head_dim"]
                * text["linear_value_head_dim"]
                + 3 * conv
            ),
            scratch_width=2 * conv,
        )


DENSE_ARCHITECTURES = {
    "llama": "llama",
    "granite": "granite",
    "mistral": "mistral",
    "phi3": "phi3",
}  # GGUF arch -> config model_type
FUSED_ARCHITECTURES = {
    "phi3"
}  # attn_qkv = [q; k; v] and ffn_up = [gate; up] rows (transformers' qkv_proj, gate_up_proj)
HALVES_ARCHITECTURES = {
    "phi3"
}  # Q/K rows stored as transformers has them: RoPE rotates (i, i + d/2), not (2i, 2i+1)


class Dense(Family):
    def geometry(self, config, header):
        hidden, heads = config["hidden_size"], config["num_attention_heads"]
        return dict(
            layers=config["num_hidden_layers"],
            hidden=hidden,
            intermediate=config["intermediate_size"],
            heads=heads,
            kv_heads=config.get("num_key_value_heads", heads),
            head_dim=config.get("head_dim") or hidden // heads,
            vocab=config["vocab_size"],
            context=config.get("max_position_embeddings"),
            tied_embeddings=bool(config.get("tie_word_embeddings", False)),
            rope_theta=_rope_theta(config),
            eps=config.get("rms_norm_eps", 1e-5),
            attention_scale=header.get("attention.scale")
            or (config.get("head_dim") or hidden // heads) ** -0.5,
            rope_pairs="halves"
            if header.architecture() in HALVES_ARCHITECTURES
            else "adjacent",
        )

    def check(self, config, header):
        out = []
        arch = header.architecture()
        if config.get("model_type") != DENSE_ARCHITECTURES.get(arch):
            out.append(
                finding(
                    "error",
                    f"config model_type {config.get('model_type')!r} does not match GGUF architecture {arch!r}",
                )
            )
        g = self.geometry(config, header)
        if g["head_dim"] not in (64, 128):
            out.append(
                finding(
                    "error",
                    f"head_dim {g['head_dim']}: the dense stage supports 64 or 128",
                )
            )
        if g["heads"] % g["kv_heads"] or g["heads"] // g["kv_heads"] > 8:
            out.append(
                finding(
                    "error",
                    f"{g['heads']} query heads over {g['kv_heads']} KV heads: need a whole group of at most 8",
                )
            )
        if g["hidden"] != g["heads"] * g["head_dim"]:
            out.append(
                finding(
                    "error",
                    "attention width (heads x head_dim) differs from the hidden width",
                )
            )
        for key in ("hidden", "intermediate"):
            if g[key] % 256:
                out.append(
                    finding(
                        "error",
                        f"{key} {g[key]} is not a multiple of 256 (K-quant blocks)",
                    )
                )
        if config.get("attention_bias") or config.get("mlp_bias"):
            out.append(
                finding("error", "attention/MLP biases are not supported")
            )
        if config.get("hidden_act", "silu") != "silu":
            out.append(
                finding(
                    "error",
                    f"activation {config.get('hidden_act')}: only SiLU (SwiGLU) is supported",
                )
            )
        rope = (
            config.get("rope_parameters") or config.get("rope_scaling") or {}
        )
        if (
            rope.get("rope_type") or rope.get("type") or "default"
        ) != "default":
            out.append(
                finding(
                    "error",
                    f"RoPE scaling {rope}: only default RoPE is supported",
                )
            )
        if config.get("sliding_window"):
            out.append(
                finding("error", "sliding-window attention is not supported")
            )
        for key in ("embedding_scale", "residual_scale", "logit_scale"):
            value = header.get(key)
            if value is not None and abs(value - 1.0) > 1e-6:
                out.append(
                    finding(
                        "error",
                        f"GGUF {arch}.{key} = {value}: the dense stage requires 1",
                    )
                )
        if header.get("rope.dimension_count", g["head_dim"]) != g["head_dim"]:
            out.append(finding("error", "partial RoPE is not supported"))
        if int(header.get("block_count")) != g["layers"]:
            out.append(
                finding(
                    "error",
                    "GGUF block count differs from config num_hidden_layers",
                )
            )
        out += _tensor_findings(header, g, fused=arch in FUSED_ARCHITECTURES)
        if g["context"] and g["context"] > 32768:
            out.append(
                finding(
                    "note",
                    f"context {g['context']}: serving capacity is set per bundle (prompt + generated), "
                    "and KV memory grows with it",
                )
            )
        return out


def _rope_theta(config):
    rope = config.get("rope_parameters") or {}
    return float(rope.get("rope_theta", config.get("rope_theta", 10000.0)))


def _tensor_findings(header, g, fused=False):
    out, t = [], header.tensors
    hidden, ffn, kv = (
        g["hidden"],
        g["intermediate"],
        g["kv_heads"] * g["head_dim"],
    )
    matrices = (
        (
            ("attn_qkv.weight", (hidden + 2 * kv, hidden)),
            ("ffn_up.weight", (2 * ffn, hidden)),
        )
        if fused
        else (
            ("attn_q.weight", (hidden, hidden)),
            ("attn_k.weight", (kv, hidden)),
            ("attn_v.weight", (kv, hidden)),
            ("ffn_gate.weight", (ffn, hidden)),
            ("ffn_up.weight", (ffn, hidden)),
        )
    )
    matrices += (
        ("attn_output.weight", (hidden, hidden)),
        ("ffn_down.weight", (hidden, ffn)),
    )

    def need(name, shape, floats=False):
        if name not in t:
            out.append(finding("error", f"missing tensor {name}"))
            return
        if t[name].shape != shape:
            out.append(
                finding(
                    "error", f"{name}: shape {t[name].shape}, expected {shape}"
                )
            )
        allowed = {"F32"} if floats else PACKED
        if t[name].kind not in allowed:
            out.append(
                finding(
                    "error",
                    f"{name}: type {t[name].kind} is not decoded by the WebGPU kernels ({sorted(allowed)})",
                )
            )

    vocab = (
        t["token_embd.weight"].shape[0]
        if "token_embd.weight" in t
        else g["vocab"]
    )
    need("token_embd.weight", (vocab, hidden))
    if "output.weight" in t:
        need("output.weight", (vocab, hidden))
    elif not g["tied_embeddings"]:
        out.append(
            finding("error", "missing output.weight in an untied model")
        )
    need("output_norm.weight", (hidden,), floats=True)
    for i in range(g["layers"]):
        p = f"blk.{i}."
        for name, shape in matrices:
            need(p + name, shape)
        for name in ("attn_norm.weight", "ffn_norm.weight"):
            need(p + name, (hidden,), floats=True)
    extra = sorted(
        {n.split(".", 2)[-1] for n in t if n.startswith("blk.")}
        - {name for name, _ in matrices}
        - {"attn_norm.weight", "ffn_norm.weight"}
    )
    if extra:
        out.append(
            finding(
                "error", f"layer tensors the dense stage would ignore: {extra}"
            )
        )
    return out


FAST = {
    "Q4_K",
    "Q5_K",
    "Q6_K",
}  # what the expert kernel decodes (the matrix-vector kernel's word-wise decoders)
KOLIBRI_LAYER_TYPES = {"sliding_attention", "full_attention"}


class Kolibri(Family):
    """Kolibri-1 (`kolibri1`): see `kolibri_gguf` for the layer and the GGUF layout our converter writes."""

    def geometry(self, config, header):
        hidden, heads = config["hidden_size"], config["num_attention_heads"]
        head_dim = config.get("head_dim") or hidden // heads
        kv, experts, top_k = (
            config["num_key_value_heads"],
            config["num_experts"],
            config["num_experts_per_tok"],
        )
        expert_ffn, shared_ffn = (
            config["moe_intermediate_size"],
            config["shared_expert_intermediate_size"],
        )
        t = header.tensors if header is not None else {}
        vocab = (
            t["token_embd.weight"].shape[0]
            if "token_embd.weight" in t
            else config["vocab_size"]
        )
        return dict(
            layers=config["num_hidden_layers"],
            hidden=hidden,
            heads=heads,
            kv_heads=kv,
            head_dim=head_dim,
            vocab=vocab,
            experts=experts,
            top_k=top_k,
            expert_ffn=expert_ffn,
            shared_ffn=shared_ffn,
            window=config.get("sliding_window"),
            rope_theta=_rope_theta(config),
            eps=config.get("rms_norm_eps", 1e-6),
            attention_scale=head_dim**-0.5,
            sliding=[
                kind == "sliding_attention"
                for kind in config.get("layer_types") or []
            ],
            context=config.get("max_position_embeddings"),
            tied_embeddings=False,
            # per-token MLP scratch: the routed experts' gate/up rows (stage_memory's `intermediate`), and
            # their outputs, the router logits, the shared expert and the normalized Q/K
            intermediate=top_k * 2 * expert_ffn,
            scratch_width=top_k * hidden
            + experts
            + 2 * shared_ffn
            + heads * head_dim
            + 3 * kv * head_dim,
        )

    def check(self, config, header):
        out = []
        arch = header.architecture()
        if arch != "kolibri1" or config.get("model_type") != "kolibri1":
            out.append(
                finding(
                    "error",
                    f"GGUF architecture {arch!r} / config model_type {config.get('model_type')!r}: "
                    "need kolibri1 (our converter's GGUF)",
                )
            )
            return out
        g = self.geometry(config, header)
        if g["head_dim"] not in (64, 128):
            out.append(
                finding(
                    "error",
                    f"head_dim {g['head_dim']}: the stage supports 64 or 128",
                )
            )
        if g["heads"] % g["kv_heads"] or g["heads"] // g["kv_heads"] > 16:
            out.append(
                finding(
                    "error",
                    f"{g['heads']} query heads over {g['kv_heads']} KV heads: need a whole group of at most 16",
                )
            )
        for key in ("hidden", "expert_ffn", "shared_ffn"):
            if g[key] % 256:
                out.append(
                    finding(
                        "error",
                        f"{key} {g[key]} is not a multiple of 256 (K-quant blocks)",
                    )
                )
        if (g["heads"] * g["head_dim"]) % 256:
            out.append(
                finding(
                    "error",
                    "attention width (heads x head_dim) is not a multiple of 256",
                )
            )
        if not 1 <= g["top_k"] <= 16 or g["experts"] < g["top_k"]:
            out.append(
                finding(
                    "error",
                    f"top {g['top_k']} of {g['experts']} experts: the router supports 1-16 of at least as many",
                )
            )
        if config.get("norm_topk_prob"):
            out.append(
                finding(
                    "error",
                    "norm_topk_prob: renormalized routing weights are not supported",
                )
            )
        if config.get("attention_bias") or config.get("mlp_bias"):
            out.append(
                finding("error", "attention/MLP biases are not supported")
            )
        if config.get("hidden_act", "silu") != "silu":
            out.append(
                finding(
                    "error",
                    f"activation {config.get('hidden_act')}: only SiLU (SwiGLU) is supported",
                )
            )
        if config.get("tie_word_embeddings"):
            out.append(finding("error", "tied embeddings are not supported"))
        rope = (
            config.get("rope_parameters") or config.get("rope_scaling") or {}
        )
        if (
            rope.get("rope_type") or rope.get("type") or "default"
        ) != "default" or rope.get("partial_rotary_factor", 1.0) != 1.0:
            out.append(
                finding(
                    "error",
                    f"RoPE {rope}: only default RoPE over the whole head is supported",
                )
            )
        types = config.get("layer_types") or []
        if len(types) != g["layers"] or set(types) - KOLIBRI_LAYER_TYPES:
            out.append(
                finding(
                    "error",
                    f"layer_types must give each of the {g['layers']} layers one of {sorted(KOLIBRI_LAYER_TYPES)}",
                )
            )
        if "sliding_attention" in types and not (
            config.get("use_sliding_window", True)
            and isinstance(g["window"], int)
            and g["window"] > 0
        ):
            out.append(
                finding(
                    "error",
                    "sliding layers need a positive integer sliding_window",
                )
            )
        if (
            header.get("block_count") is not None
            and int(header.get("block_count")) != g["layers"]
        ):
            out.append(
                finding(
                    "error",
                    "GGUF block count differs from config num_hidden_layers",
                )
            )
        pattern = header.get("attention.sliding_window_pattern")
        if pattern is not None and [bool(v) for v in pattern] != g["sliding"]:
            out.append(
                finding(
                    "error",
                    "the GGUF's sliding-window layers differ from config layer_types",
                )
            )
        out += _kolibri_tensor_findings(header, g)
        if g["context"] and g["context"] > 32768:
            out.append(
                finding(
                    "note",
                    f"context {g['context']}: serving capacity is set per bundle (prompt + generated), "
                    "and KV memory grows with it (every layer keeps its keys and values)",
                )
            )
        return out


def _kolibri_tensor_findings(header, g):
    out, t = [], header.tensors
    H, D, E = g["hidden"], g["head_dim"], g["experts"]
    q, kv, F, S = (
        g["heads"] * D,
        g["kv_heads"] * D,
        g["expert_ffn"],
        g["shared_ffn"],
    )

    def need(name, shape, allowed=PACKED):
        if name not in t:
            out.append(finding("error", f"missing tensor {name}"))
            return
        if t[name].shape != shape:
            out.append(
                finding(
                    "error", f"{name}: shape {t[name].shape}, expected {shape}"
                )
            )
        if t[name].kind not in allowed:
            out.append(
                finding(
                    "error",
                    f"{name}: type {t[name].kind} is not one of {sorted(allowed)}",
                )
            )

    vocab = g["vocab"]
    need("token_embd.weight", (vocab, H))
    need("output.weight", (vocab, H))
    need("output_norm.weight", (H,), {"F32"})
    layer = {
        "attn_q.weight": ((q, H), PACKED),
        "attn_k.weight": ((kv, H), PACKED),
        "attn_v.weight": ((kv, H), PACKED),
        "attn_output.weight": ((H, q), PACKED),
        "ffn_gate_shexp.weight": ((S, H), PACKED),
        "ffn_up_shexp.weight": ((S, H), PACKED),
        "ffn_down_shexp.weight": ((H, S), PACKED),
        "ffn_gate_up_exps.weight": ((E, 2 * F, H), FAST),
        "ffn_down_exps.weight": ((E, H, F), FAST),
        "ffn_gate_inp.weight": ((E, H), {"F32"}),
        "exp_probs_b.bias": ((E,), {"F32"}),
        "attn_q_norm.weight": ((D,), {"F32"}),
        "attn_k_norm.weight": ((D,), {"F32"}),
        **{
            f"{n}.weight": ((H,), {"F32"})
            for n in (
                "attn_norm",
                "attn_post_norm",
                "ffn_norm",
                "ffn_post_norm",
            )
        },
    }
    for i in range(g["layers"]):
        for name, (shape, allowed) in layer.items():
            need(f"blk.{i}.{name}", shape, allowed)
    extra = sorted(
        {n.split(".", 2)[-1] for n in t if n.startswith("blk.")} - set(layer)
    )
    if extra:
        out.append(
            finding("error", f"layer tensors the stage would ignore: {extra}")
        )
    return out


QWEN3MOE_ARCHITECTURES = {"qwen3moe": "qwen3_moe"}


class Qwen3Moe(Family):
    """Qwen3-MoE (`qwen3moe`): see `qwen3moe_gguf` for the layer and llama.cpp's GGUF layout."""

    def geometry(self, config, header):
        hidden, heads = config["hidden_size"], config["num_attention_heads"]
        head_dim = config.get("head_dim") or hidden // heads
        kv, experts, top_k = (
            config["num_key_value_heads"],
            config["num_experts"],
            config["num_experts_per_tok"],
        )
        expert_ffn = config["moe_intermediate_size"]
        t = header.tensors if header is not None else {}
        vocab = (
            t["token_embd.weight"].shape[0]
            if "token_embd.weight" in t
            else config["vocab_size"]
        )
        return dict(
            layers=config["num_hidden_layers"],
            hidden=hidden,
            heads=heads,
            kv_heads=kv,
            head_dim=head_dim,
            vocab=vocab,
            experts=experts,
            top_k=top_k,
            expert_ffn=expert_ffn,
            norm_topk=bool(config.get("norm_topk_prob", False)),
            rope_theta=_rope_theta(config),
            eps=config.get("rms_norm_eps", 1e-6),
            attention_scale=head_dim**-0.5,
            context=config.get("max_position_embeddings"),
            tied_embeddings=False,
            # per-token MLP scratch: the routed experts' gate and up rows, their outputs, the router logits
            # and the normalized Q/K
            intermediate=top_k * 2 * expert_ffn,
            scratch_width=top_k * hidden
            + experts
            + heads * head_dim
            + 3 * kv * head_dim,
        )

    def check(self, config, header):
        out = []
        arch = header.architecture()
        if (
            arch not in QWEN3MOE_ARCHITECTURES
            or config.get("model_type") != QWEN3MOE_ARCHITECTURES[arch]
        ):
            out.append(
                finding(
                    "error",
                    f"GGUF architecture {arch!r} / config model_type {config.get('model_type')!r}: "
                    "need qwen3moe / qwen3_moe",
                )
            )
            return out
        g = self.geometry(config, header)
        if g["head_dim"] not in (64, 128):
            out.append(
                finding(
                    "error",
                    f"head_dim {g['head_dim']}: the stage supports 64 or 128",
                )
            )
        if g["heads"] % g["kv_heads"] or g["heads"] // g["kv_heads"] > 16:
            out.append(
                finding(
                    "error",
                    f"{g['heads']} query heads over {g['kv_heads']} KV heads: need a whole group of at most 16",
                )
            )
        for key in ("hidden", "expert_ffn"):
            if g[key] % 256:
                out.append(
                    finding(
                        "error",
                        f"{key} {g[key]} is not a multiple of 256 (K-quant blocks)",
                    )
                )
        if (g["heads"] * g["head_dim"]) % 256:
            out.append(
                finding(
                    "error",
                    "attention width (heads x head_dim) is not a multiple of 256",
                )
            )
        if not 1 <= g["top_k"] <= 16 or g["experts"] < g["top_k"]:
            out.append(
                finding(
                    "error",
                    f"top {g['top_k']} of {g['experts']} experts: the router supports 1-16 of at least as many",
                )
            )
        if config.get("decoder_sparse_step", 1) != 1 or config.get(
            "mlp_only_layers"
        ):
            out.append(
                finding(
                    "error",
                    "dense MLP layers (decoder_sparse_step > 1 or mlp_only_layers) are not supported",
                )
            )
        if config.get("shared_expert_intermediate_size"):
            out.append(
                finding(
                    "error",
                    "a shared expert is not part of qwen3moe (that is qwen2moe)",
                )
            )
        if config.get("attention_bias") or config.get("mlp_bias"):
            out.append(
                finding("error", "attention/MLP biases are not supported")
            )
        if config.get("hidden_act", "silu") != "silu":
            out.append(
                finding(
                    "error",
                    f"activation {config.get('hidden_act')}: only SiLU (SwiGLU) is supported",
                )
            )
        if config.get("tie_word_embeddings"):
            out.append(finding("error", "tied embeddings are not supported"))
        if config.get("use_sliding_window"):
            out.append(
                finding(
                    "error",
                    "sliding-window attention is not supported for qwen3moe",
                )
            )
        rope = (
            config.get("rope_parameters") or config.get("rope_scaling") or {}
        )
        if (
            rope.get("rope_type") or rope.get("type") or "default"
        ) != "default" or rope.get("partial_rotary_factor", 1.0) != 1.0:
            out.append(
                finding(
                    "error",
                    f"RoPE {rope}: only default RoPE over the whole head is supported",
                )
            )
        if (
            header.get("block_count") is not None
            and int(header.get("block_count")) != g["layers"]
        ):
            out.append(
                finding(
                    "error",
                    "GGUF block count differs from config num_hidden_layers",
                )
            )
        for key, want in (
            ("expert_count", g["experts"]),
            ("expert_used_count", g["top_k"]),
            ("expert_feed_forward_length", g["expert_ffn"]),
        ):
            if header.get(key) is not None and int(header.get(key)) != want:
                out.append(
                    finding(
                        "error",
                        f"GGUF {key} {header.get(key)} differs from the config's {want}",
                    )
                )
        out += _qwen3moe_tensor_findings(header, g)
        if g["context"] and g["context"] > 32768:
            out.append(
                finding(
                    "note",
                    f"context {g['context']}: serving capacity is set per bundle (prompt + generated), "
                    "and KV memory grows with it (every layer keeps its keys and values)",
                )
            )
        return out


def _qwen3moe_tensor_findings(header, g):
    out, t = [], header.tensors
    H, D, E = g["hidden"], g["head_dim"], g["experts"]
    q, kv, F = g["heads"] * D, g["kv_heads"] * D, g["expert_ffn"]

    def need(name, shape, allowed=PACKED):
        if name not in t:
            out.append(finding("error", f"missing tensor {name}"))
            return
        if t[name].shape != shape:
            out.append(
                finding(
                    "error", f"{name}: shape {t[name].shape}, expected {shape}"
                )
            )
        if t[name].kind not in allowed:
            out.append(
                finding(
                    "error",
                    f"{name}: type {t[name].kind} is not one of {sorted(allowed)}",
                )
            )

    need("token_embd.weight", (g["vocab"], H))
    need("output.weight", (g["vocab"], H))
    need("output_norm.weight", (H,), {"F32"})
    layer = {
        "attn_q.weight": ((q, H), PACKED),
        "attn_k.weight": ((kv, H), PACKED),
        "attn_v.weight": ((kv, H), PACKED),
        "attn_output.weight": ((H, q), PACKED),
        "ffn_gate_exps.weight": ((E, F, H), FAST),
        "ffn_up_exps.weight": ((E, F, H), FAST),
        "ffn_down_exps.weight": ((E, H, F), FAST),
        "ffn_gate_inp.weight": ((E, H), {"F32"}),
        "attn_q_norm.weight": ((D,), {"F32"}),
        "attn_k_norm.weight": ((D,), {"F32"}),
        "attn_norm.weight": ((H,), {"F32"}),
        "ffn_norm.weight": ((H,), {"F32"}),
    }
    for i in range(g["layers"]):
        for name, (shape, allowed) in layer.items():
            need(f"blk.{i}.{name}", shape, allowed)
    extra = sorted(
        {n.split(".", 2)[-1] for n in t if n.startswith("blk.")} - set(layer)
    )
    if extra:
        out.append(
            finding("error", f"layer tensors the stage would ignore: {extra}")
        )
    return out


QWEN3NEXT_STAGE = None  # no WebGPU stage runs qwen3next yet: admission reports it as an error


class Qwen3Next(Family):
    """Qwen3-Next (`qwen3next` GGUF, `qwen3_next` config), as in Qwen/Qwen3-Coder-Next-GGUF (Q4_K_M, 4 shards).

    That GGUF has llama.cpp's *legacy* DeltaNet layout, which differs from our `qwen35` GGUFs (and from newer
    llama.cpp converters, which write split `attn_qkv` / `attn_gate`):

    - `ssm_in` [2*Kh*Dk + 2*Vh*Dv, hidden] is transformers' `in_proj_qkvz` unchanged: rows grouped by key head,
      each group [q (Dk), k (Dk), v (Vh/Kh*Dv), z (Vh/Kh*Dv)]; `ssm_ba` [2*Vh, hidden] likewise [b, a] per key head.
    - Value heads stay in grouped order (value head j reads key head j div Vh/Kh), also in `ssm_conv1d`'s V
      channels (its channels are [q all heads, k all heads, v]), `ssm_a`, `ssm_dt.bias` and `ssm_out`'s columns.
      `ssm_out` is K-quantized over blocks of 256 columns and a value head is 128 wide, so the qwen35 kernels'
      tiled order cannot be produced by permuting packed bytes: a stage must index grouped order itself.
    - Full attention: `attn_q` [2*heads*D, hidden] interleaves per head [query (D), output gate (D)];
      attention output is multiplied by sigmoid(gate). Per-head Q/K RMSNorm; RoPE over halves of the first
      `rope.dimension_count` (partial_rotary_factor * D) dims.
    - The norm before the MLP is `post_attention_norm` (not `ffn_norm`). RMSNorm weights are stored plus one
      (the converter adds transformers' zero-centred offset), except `ssm_norm`; `ssm_a` = -exp(A_log).
    - MoE in every layer: router `ffn_gate_inp` (F32), softmax, top k renormalized (`norm_topk_prob`), experts
      split gate/up/down (`_exps`), plus a shared expert (`_shexp`) times sigmoid(`ffn_gate_inp_shexp` . x),
      whose gate vector is [hidden] (F16 in the Q4_K_M artifact).
    """

    def geometry(self, config, header):
        c = config.get("text_config", config)
        heads, head_dim = c["num_attention_heads"], c["head_dim"]
        layers, interval = (
            c["num_hidden_layers"],
            c.get("full_attention_interval", 4),
        )
        Kh, Vh = c["linear_num_key_heads"], c["linear_num_value_heads"]
        Dk, Dv, conv_kernel = (
            c["linear_key_head_dim"],
            c["linear_value_head_dim"],
            c["linear_conv_kernel_dim"],
        )
        experts, top_k = c["num_experts"], c["num_experts_per_tok"]
        expert_ffn, shared_ffn = (
            c["moe_intermediate_size"],
            c["shared_expert_intermediate_size"],
        )
        conv = 2 * Kh * Dk + Vh * Dv
        types = c.get("layer_types") or [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ]
        t = header.tensors if header is not None else {}
        vocab = (
            t["token_embd.weight"].shape[0]
            if "token_embd.weight" in t
            else c["vocab_size"]
        )
        rope_parameters = c.get("rope_parameters") or {}
        rope_dims = int(
            head_dim
            * rope_parameters.get(
                "partial_rotary_factor", c.get("partial_rotary_factor", 1.0)
            )
        )
        return dict(
            layers=layers,
            hidden=c["hidden_size"],
            heads=heads,
            kv_heads=c["num_key_value_heads"],
            head_dim=head_dim,
            vocab=vocab,
            context=c.get("max_position_embeddings"),
            tied_embeddings=False,
            rope_theta=_rope_theta(c),
            rope_dims=rope_dims,
            eps=c.get("rms_norm_eps", 1e-6),
            attention_scale=head_dim**-0.5,
            layer_types=types,
            kv_layers=[
                i for i, kind in enumerate(types) if kind == "full_attention"
            ],
            linear_key_heads=Kh,
            linear_value_heads=Vh,
            linear_key_dim=Dk,
            linear_value_dim=Dv,
            conv_kernel=conv_kernel,
            conv_channels=conv,
            experts=experts,
            top_k=top_k,
            expert_ffn=expert_ffn,
            shared_ffn=shared_ffn,
            norm_topk=bool(c.get("norm_topk_prob", False)),
            value_order="grouped",
            layout="legacy_qkvz",
            # per DeltaNet layer and sequence: the recurrent state (float32) and the convolution's history
            state_bytes=4 * (Vh * Dk * Dv + (conv_kernel - 1) * conv),
            # per-token MLP scratch: the routed experts' gate/up rows (stage_memory's `intermediate`)
            intermediate=top_k * 2 * expert_ffn,
            # the widest per-token buffers besides it: the fused DeltaNet projection, the gated query, the
            # routed outputs, router logits and the shared expert's gate/up rows
            scratch_width=max(2 * Kh * Dk + 2 * Vh * Dv, 2 * heads * head_dim)
            + top_k * c["hidden_size"]
            + experts
            + 2 * shared_ffn,
        )

    def check(self, config, header):
        out = self.admit(config, header)
        if QWEN3NEXT_STAGE is None:
            out.append(
                finding(
                    "error",
                    "no WebGPU stage runs qwen3next yet (admission only: the artifact was checked, "
                    "not executed)",
                )
            )
        return out

    def admit(self, config, header):
        """The artifact's findings without the stage's: what the loader and the NumPy reference require."""
        out = []
        arch, c = header.architecture(), config.get("text_config", config)
        if arch != "qwen3next" or c.get("model_type") != "qwen3_next":
            out.append(
                finding(
                    "error",
                    f"GGUF architecture {arch!r} / config model_type {c.get('model_type')!r}: "
                    "need qwen3next / qwen3_next",
                )
            )
            return out
        g = self.geometry(config, header)
        if len(g["layer_types"]) != g["layers"] or set(g["layer_types"]) - {
            "linear_attention",
            "full_attention",
        }:
            out.append(
                finding(
                    "error",
                    f"layer types must give each of the {g['layers']} layers linear_attention "
                    "or full_attention",
                )
            )
        if not g["kv_layers"]:
            out.append(finding("error", "no full-attention layer"))
        if g["heads"] % g["kv_heads"] or g["heads"] // g["kv_heads"] > 16:
            out.append(
                finding(
                    "error",
                    f"{g['heads']} query heads over {g['kv_heads']} KV heads: need a whole group of at most 16",
                )
            )
        if g["linear_value_heads"] % g["linear_key_heads"]:
            out.append(
                finding(
                    "error",
                    f"{g['linear_value_heads']} DeltaNet value heads over {g['linear_key_heads']} key "
                    "heads: need a whole number per key head",
                )
            )
        for key in ("hidden", "expert_ffn", "shared_ffn"):
            if g[key] % 256:
                out.append(
                    finding(
                        "error",
                        f"{key} {g[key]} is not a multiple of 256 (K-quant blocks)",
                    )
                )
        if (g["heads"] * g["head_dim"]) % 256 or (
            g["linear_value_heads"] * g["linear_value_dim"]
        ) % 256:
            out.append(
                finding(
                    "error",
                    "attention or DeltaNet value width is not a multiple of 256 (K-quant blocks)",
                )
            )
        if not 1 <= g["top_k"] <= 16 or g["experts"] < g["top_k"]:
            out.append(
                finding(
                    "error",
                    f"top {g['top_k']} of {g['experts']} experts: the router supports 1-16 of at least as many",
                )
            )
        if not g["norm_topk"]:
            out.append(
                finding(
                    "error",
                    "norm_topk_prob false: only renormalized top-k routing is admitted",
                )
            )
        if c.get("decoder_sparse_step", 1) != 1 or c.get("mlp_only_layers"):
            out.append(
                finding(
                    "error",
                    "dense MLP layers (decoder_sparse_step > 1 or mlp_only_layers) are not supported",
                )
            )
        if not g["shared_ffn"]:
            out.append(
                finding("error", "no shared expert: qwen3next layers have one")
            )
        if c.get("attention_bias") or c.get("mlp_bias"):
            out.append(
                finding("error", "attention/MLP biases are not supported")
            )
        if c.get("hidden_act", "silu") != "silu":
            out.append(
                finding(
                    "error",
                    f"activation {c.get('hidden_act')}: only SiLU (SwiGLU) is supported",
                )
            )
        if c.get("tie_word_embeddings", config.get("tie_word_embeddings")):
            out.append(finding("error", "tied embeddings are not supported"))
        if c.get("use_sliding_window"):
            out.append(
                finding("error", "sliding-window attention is not supported")
            )
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        if (
            rope.get("rope_type") or rope.get("type") or "default"
        ) != "default":
            out.append(
                finding(
                    "error",
                    f"RoPE {rope}: only default (partial) RoPE is supported",
                )
            )
        if not 0 < g["rope_dims"] <= g["head_dim"] or g["rope_dims"] % 2:
            out.append(
                finding(
                    "error",
                    f"RoPE over {g['rope_dims']} of {g['head_dim']} dims: need an even count",
                )
            )
        for key, want in (
            ("block_count", g["layers"]),
            ("expert_count", g["experts"]),
            ("expert_used_count", g["top_k"]),
            ("expert_feed_forward_length", g["expert_ffn"]),
            ("expert_shared_feed_forward_length", g["shared_ffn"]),
            ("attention.head_count", g["heads"]),
            ("attention.head_count_kv", g["kv_heads"]),
            ("attention.key_length", g["head_dim"]),
            ("attention.value_length", g["head_dim"]),
            ("rope.dimension_count", g["rope_dims"]),
            ("ssm.conv_kernel", g["conv_kernel"]),
            ("ssm.state_size", g["linear_key_dim"]),
            ("ssm.group_count", g["linear_key_heads"]),
            ("ssm.time_step_rank", g["linear_value_heads"]),
            (
                "ssm.inner_size",
                g["linear_value_heads"] * g["linear_value_dim"],
            ),
        ):
            if header.get(key) is not None and int(header.get(key)) != want:
                out.append(
                    finding(
                        "error",
                        f"GGUF {key} {header.get(key)} differs from the config's {want}",
                    )
                )
        out += _qwen3next_tensor_findings(header, g)
        if g["context"] and g["context"] > 32768:
            out.append(
                finding(
                    "note",
                    f"context {g['context']}: serving capacity is set per bundle; only the "
                    f"{len(g['kv_layers'])} full-attention layers keep a KV cache",
                )
            )
        return out


def _qwen3next_tensor_findings(header, g):
    out, t = [], header.tensors
    H, D, E = g["hidden"], g["head_dim"], g["experts"]
    Kh, Vh, Dk, Dv = (
        g["linear_key_heads"],
        g["linear_value_heads"],
        g["linear_key_dim"],
        g["linear_value_dim"],
    )
    F, S = g["expert_ffn"], g["shared_ffn"]

    def need(name, shape, allowed=PACKED):
        if name not in t:
            out.append(finding("error", f"missing tensor {name}"))
            return
        if t[name].shape != shape:
            out.append(
                finding(
                    "error", f"{name}: shape {t[name].shape}, expected {shape}"
                )
            )
        if t[name].kind not in allowed:
            out.append(
                finding(
                    "error",
                    f"{name}: type {t[name].kind} is not one of {sorted(allowed)}",
                )
            )

    need("token_embd.weight", (g["vocab"], H))
    need("output.weight", (g["vocab"], H))
    need("output_norm.weight", (H,), {"F32"})
    common = {
        "attn_norm.weight": ((H,), {"F32"}),
        "post_attention_norm.weight": ((H,), {"F32"}),
        "ffn_gate_inp.weight": ((E, H), {"F32"}),
        "ffn_gate_exps.weight": ((E, F, H), FAST),
        "ffn_up_exps.weight": ((E, F, H), FAST),
        "ffn_down_exps.weight": ((E, H, F), FAST),
        "ffn_gate_shexp.weight": ((S, H), PACKED),
        "ffn_up_shexp.weight": ((S, H), PACKED),
        "ffn_down_shexp.weight": ((H, S), PACKED),
        "ffn_gate_inp_shexp.weight": ((H,), {"F32", "F16"}),
    }
    linear = {
        "ssm_in.weight": ((2 * Kh * Dk + 2 * Vh * Dv, H), PACKED),
        "ssm_ba.weight": ((2 * Vh, H), PACKED),
        "ssm_conv1d.weight": ((g["conv_channels"], g["conv_kernel"]), {"F32"}),
        "ssm_a": ((Vh,), {"F32"}),
        "ssm_dt.bias": ((Vh,), {"F32"}),
        "ssm_norm.weight": ((Dv,), {"F32"}),
        "ssm_out.weight": ((H, Vh * Dv), PACKED),
    }
    full = {
        "attn_q.weight": ((2 * g["heads"] * D, H), PACKED),
        "attn_k.weight": ((g["kv_heads"] * D, H), PACKED),
        "attn_v.weight": ((g["kv_heads"] * D, H), PACKED),
        "attn_output.weight": ((H, g["heads"] * D), PACKED),
        "attn_q_norm.weight": ((D,), {"F32"}),
        "attn_k_norm.weight": ((D,), {"F32"}),
    }
    split = sorted(
        {n.split(".", 2)[-1] for n in t if n.startswith("blk.")}
        & {"attn_qkv.weight", "attn_gate.weight"}
    )
    if split:
        out.append(
            finding(
                "error",
                f"split DeltaNet tensors {split} (newer llama.cpp layout): only the legacy fused "
                "ssm_in / ssm_ba layout is admitted",
            )
        )
    for i, kind in enumerate(g["layer_types"]):
        layer = dict(common, **(full if kind == "full_attention" else linear))
        for name, (shape, allowed) in layer.items():
            need(f"blk.{i}.{name}", shape, allowed)
        extra = sorted(
            n.split(".", 2)[-1]
            for n in t
            if n.startswith(f"blk.{i}.") and n.split(".", 2)[-1] not in layer
        )
        if extra:
            out.append(
                finding(
                    "error",
                    f"layer {i} ({kind}) tensors the stage would ignore: {extra}",
                )
            )
    return out


INKLING2_STAGE = "inkling2_gpu.mjs"  # DO-95: Inkling2Stage / Inkling2ContributorStage
INKLING2_KINDS = {"F32", "Q8_0", "IQ4_NL"}  # what inkling2_gguf's NumPy reference reads


class Inkling2(Family):
    """Inkling 2 Mini (`inkling2_loop` GGUF), as in ggml-org/Inkling-2-Mini-GGUF. The configuration comes from the
    GGUF header (`inkling2_gguf.config_from_header`); a Hugging Face config, when given, must name the same model.
    The layout and formulas are in `inkling2_gguf` and research/DO-95.md."""

    def geometry(self, config, header):
        from ygaraty.backends.webgpu.checkpoints import inkling2_gguf

        c = inkling2_gguf.config_from_header(header)
        visits = c.visits()
        full = sum(1 for il, _ in visits if not c.sliding[il])
        return dict(
            layers=c.layers,
            visits=len(visits),
            hidden=c.hidden,
            heads=c.heads,
            kv_heads=c.kv_heads,
            head_dim=c.head_dim,
            vocab=c.vocab,
            vocab_unpadded=c.vocab_unpadded,
            context=header.get("context_length"),
            tied_embeddings=False,
            eps=c.eps,
            attention_scale=c.head_dim**-0.5,
            sliding=list(c.sliding),
            window=c.window,
            experts=c.experts,
            top_k=c.top_k,
            expert_ffn=c.expert_ffn,
            loop=dict(start=c.loop_start, end=c.loop_end, passes=c.passes, gate_rank=c.gate_rank),
            # per visit and sequence: the convolution history (float32) of K, V and both branch outputs
            state_bytes=4 * (c.conv - 1) * (2 * c.kv_heads * c.head_dim + 2 * c.hidden),
            # per token: K and V of every visit (float32); only `full` visits grow past the window
            kv_bytes_per_token=4 * 2 * c.kv_heads * c.head_dim * len(visits),
            full_attention_visits=full,
            intermediate=c.top_k * 2 * c.expert_ffn,
        )

    def placement(self, header, workers, *, capacity=4096, chunk=512, budgets=None, wire_bytes=2):
        """Stages for `workers` contributors with the looped core on one of them (see `loop_placement`)."""
        from ygaraty.backends.webgpu.checkpoints import inkling2_gguf

        c = inkling2_gguf.config_from_header(header)
        passes = {il: (c.passes if c.loop_start <= il < c.loop_end else 1) for il in range(c.layers)}
        kv_row = 4 * 2 * c.kv_heads * c.head_dim
        conv = 4 * (c.conv - 1) * (2 * c.kv_heads * c.head_dim + 2 * c.hidden)
        # as Inkling2Stage allocates: sliding visits keep a ring of window + chunk rows, global ones `capacity`
        rows = lambda il: min(capacity, c.window + chunk) if c.sliding[il] else capacity
        visit = [passes[il] * (conv + rows(il) * kv_row) for il in range(c.layers)]
        F, H, K = c.expert_ffn, c.hidden, c.top_k
        scratch = 4 * chunk * (48 * H + K * (3 * F + H))  # streams, branch buffers and expert slots of one chunk
        ends = self.endpoint_bytes(header)
        t = header.tensors
        head = ends["output"] + ends["output_norm"] + t["output_res_query.weight"].nbytes
        embedding = ends["embedding"] + t["token_embd_norm.weight"].nbytes
        return loop_placement(self.layer_bytes(header), loop=(c.loop_start, c.loop_end), res_block=c.res_block,
                              hidden=H, workers=workers, visit_bytes=visit, embedding_bytes=embedding,
                              head_bytes=head, scratch_bytes=scratch, budgets=budgets, wire_bytes=wire_bytes)

    def check_splits(self, header, splits):
        from ygaraty.backends.webgpu.checkpoints import inkling2_gguf

        c = inkling2_gguf.config_from_header(header)
        return check_loop_splits(splits, (c.loop_start, c.loop_end))

    def check(self, config, header):
        out = self.admit(config, header)
        if INKLING2_STAGE is None:
            out.append(
                finding(
                    "error",
                    "no WebGPU stage runs inkling2 yet (admission only: the artifact was checked, not executed)",
                )
            )
        return out

    def admit(self, config, header):
        from ygaraty.backends.webgpu.checkpoints import inkling2_gguf

        out = []
        arch = header.architecture()
        if arch != inkling2_gguf.ARCHITECTURE:
            return [finding("error", f"GGUF architecture {arch!r}: need {inkling2_gguf.ARCHITECTURE}")]
        model_type = (config or {}).get("model_type")
        if model_type not in (None, "inkling2_loop"):
            out.append(finding("error", f"config model_type {model_type!r}: need inkling2_loop"))
        try:
            c = inkling2_gguf.config_from_header(header)
        except ValueError as e:
            return out + [finding("error", str(e))]
        if not (0 < c.loop_start < c.loop_end <= c.layers) or c.loop_start % c.res_block or c.loop_end % c.res_block:
            out.append(finding("error", "the loop core must be whole attention-residual blocks inside the layers"))
        if c.heads % c.kv_heads or len(c.sliding) != c.layers:
            out.append(finding("error", "inconsistent heads or sliding-window pattern"))
        kinds = {t.kind for t in header.tensors.values()}
        if kinds - INKLING2_KINDS:
            out.append(finding("error", f"tensor types {sorted(kinds - INKLING2_KINDS)} are not read"))
        if "IQ4_NL" in kinds and "IQ4_NL" not in PACKED:
            out.append(finding("info", "IQ4_NL experts need the IQ4_NL matrix kernel (not in the WebGPU set yet)"))
        t = header.tensors
        for i in range(c.layers):
            need = ["attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel", "attn_sink_proj",
                    "ffn_gate_inp", "ffn_gate_up_exps", "ffn_down_exps", "ffn_norm", "ffn_res_query"]
            need += ["attn_norm"] if c.sliding[i] else ["attn_q_inp_norm", "attn_k_conv_proj"]
            if c.loop_start <= i < c.loop_end:
                need += ["loop_msg_query", "loop_q_ctrl_down"]
            missing = [n for n in need if f"blk.{i}.{n}.weight" not in t]
            if missing:
                out.append(finding("error", f"layer {i} lacks {', '.join(missing)}"))
        return out


def loop_placement(
    layer_bytes,
    *,
    loop,
    res_block,
    hidden,
    workers,
    visit_bytes,
    embedding_bytes=0,
    head_bytes=0,
    scratch_bytes=0,
    budgets=None,
    wire_bytes=2,
):
    """Contiguous layer stages for a looped model, with the loop core [start, end) on one worker.

    The loop's passes restart from the prelude output and read the previous pass's block streams, so splitting the
    core would cost a round trip per pass; only boundaries outside (start, end) are allowed. Among those, the
    stages minimize the largest memory (or, with `budgets` in bytes, the largest fraction of a worker's budget),
    then the next largest and so on, then the vectors crossing between stages. A cut before layer c carries every attention-residual block stream
    started before c (layers 0, res_block, ...) plus the current stream: those vectors per token, in `wire_bytes`.

    `layer_bytes[i]`: weights of layer i; `visit_bytes[i]`: per-sequence state of one visit of layer i (KV cache
    and convolution history; the core's layers are visited `passes` times, counted by the caller in the core's
    entries); `scratch_bytes`: per-stage activations. Returns splits, stages and transfers."""
    import itertools

    layers = len(layer_bytes)
    start, end = loop
    if not (0 <= start < end <= layers) or start % res_block or end % res_block:
        raise ValueError("The loop core must be whole residual blocks inside the layers")
    cuts = [c for c in range(1, layers) if not start < c < end]
    if not 1 <= workers <= len(cuts) + 1:
        raise ValueError(f"{workers} workers: a looped model of {layers} layers takes 1 to {len(cuts) + 1}")
    if budgets is not None and len(budgets) != workers:
        raise ValueError("One memory budget per worker")

    def vectors(c):
        return 1 + sum(1 for il in range(c) if il % res_block == 0)

    def stage(a, b, index):
        weights = sum(layer_bytes[a:b]) + (embedding_bytes if index == 0 else 0) + (head_bytes if b == layers else 0)
        state = sum(visit_bytes[a:b])
        return dict(layers=[a, b], loop=a <= start and end <= b, weights=weights, state=state, scratch=scratch_bytes,
                    total=weights + state + scratch_bytes)

    best = None
    for chosen in itertools.combinations(cuts, workers - 1):
        bounds = [0, *chosen, layers]
        stages = [stage(a, b, i) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
        loads = sorted((s["total"] / (budgets[i] if budgets else 1) for i, s in enumerate(stages)), reverse=True)
        key = (tuple(loads), sum(vectors(c) for c in chosen))  # fullest worker first, then the next fullest, ...
        if best is None or key < best[0]:
            best = (key, bounds, stages)
    (loads, _), bounds, stages = best
    load = loads[0]
    if budgets is not None:
        for s, budget in zip(stages, budgets):
            s["budget"] = budget
        if load > 1:
            raise ValueError(f"No placement fits: the fullest worker needs {load:.0%} of its budget")
    transfers = [dict(before_layer=c, vectors=vectors(c), bytes_per_token=vectors(c) * hidden * wire_bytes) for c in bounds[1:-1]]
    return dict(splits=bounds, stages=stages, transfers=transfers)


def check_loop_splits(splits, loop):
    """Errors for stage boundaries that cut a looped core."""
    start, end = loop
    return [f"boundary {c} splits the loop core {start}-{end - 1}: it must stay on one worker" for c in splits if start < c < end]


FAMILIES = {
    "qwen35": Qwen35("qwen35", "qwen35_gpu.mjs"),
    "dense": Dense("dense", "dense_gpu.mjs"),
    "kolibri1": Kolibri("kolibri1", "kolibri_gpu.mjs"),
    "qwen3moe": Qwen3Moe("qwen3moe", "kolibri_gpu.mjs"),
    "qwen3next": Qwen3Next("qwen3next", None),
    "inkling2": Inkling2("inkling2", INKLING2_STAGE),
}


def detect(config, header=None):
    """The family that runs this checkpoint (by its config, and its GGUF architecture when given)."""
    arch = header.architecture() if header is not None else None
    text = config.get("text_config", config)
    if arch == "inkling2_loop" or config.get("model_type") == "inkling2_loop":
        return FAMILIES["inkling2"]
    if (
        arch == "qwen3next" or text.get("model_type") == "qwen3_next"
    ):  # before qwen35: both have DeltaNet keys
        return FAMILIES["qwen3next"]
    if (
        arch == "qwen35"
        or text.get("model_type", "").startswith("qwen3_5")
        or "linear_num_value_heads" in text
    ):
        return FAMILIES["qwen35"]
    if arch == "kolibri1" or config.get("model_type") == "kolibri1":
        return FAMILIES["kolibri1"]
    if (
        arch in QWEN3MOE_ARCHITECTURES
        or config.get("model_type") in QWEN3MOE_ARCHITECTURES.values()
    ):
        return FAMILIES["qwen3moe"]
    if (
        arch in DENSE_ARCHITECTURES
        or config.get("model_type") in DENSE_ARCHITECTURES.values()
    ):
        return FAMILIES["dense"]
    raise ValueError(
        f"No model family runs architecture {arch or config.get('model_type')!r} yet: "
        "add one (see ygaraty/backends/webgpu/checkpoints/README.md)"
    )


def load_model(gguf, config_path, *, tensor_only=False):
    """The checkpoint's loader for its family: `qwen35_gguf.Model`, `dense_gguf.Model` or `kolibri_gguf.Model`
    (all expose `family`, `config.layers`, `config.manifest()`, `layer_type(i)`, `layer_tensors(i)`,
    `validate_webgpu()`)."""
    config = json.loads(Path(config_path).read_text())
    name = detect(config).name
    if name == "inkling2":
        raise ValueError(
            "inkling2 stages are prepared from GGUF headers (preparation.prepare_inkling2_session); its loader and "
            "NumPy reference are inkling2_gguf.Model (DO-95)"
        )
    if name == "qwen3next":
        if QWEN3NEXT_STAGE is None:
            raise ValueError(
                "qwen3next has no WebGPU stage yet (admission only): its loader and NumPy reference are "
                "qwen3next_gguf.Model; see ygaraty/backends/webgpu/checkpoints/README.md"
            )
        from ygaraty.backends.webgpu.checkpoints import qwen3next_gguf

        return qwen3next_gguf.Model(gguf, config_path, tensor_only=tensor_only)
    if name == "kolibri1":
        from ygaraty.backends.webgpu.checkpoints import kolibri_gguf

        return kolibri_gguf.Model(gguf, config_path, tensor_only=tensor_only)
    if name == "qwen3moe":
        from ygaraty.backends.webgpu.checkpoints import qwen3moe_gguf

        return qwen3moe_gguf.Model(gguf, config_path, tensor_only=tensor_only)
    if name == "dense":
        from ygaraty.backends.webgpu.checkpoints import dense_gguf

        return dense_gguf.Model(gguf, config_path, tensor_only=tensor_only)
    from ygaraty.backends.webgpu.checkpoints import qwen35_gguf

    return qwen35_gguf.Model(gguf, config_path, tensor_only=tensor_only)
