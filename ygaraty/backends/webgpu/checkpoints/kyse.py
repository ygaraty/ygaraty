"""Import Kyse QW35 checkpoints without requantizing their Q4/Q8 weights.

Format mapping based on Kyse (MIT, Copyright 2026 Felipe Cruz). Runtime
execution uses Ygaraty's existing Qwen3.5 stages, with no Kyse dependency.
"""

import base64
import hashlib
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from ygaraty.inference_support.chat import chat_format
from ygaraty.inference_support.kyse_tokenizer import KyseTokenizer

from .qwen35_gguf import WebGPUConfig


def repack_matrix(raw, kind, rows, cols):
    """Exact byte-layout conversion: Kyse separate scales -> GGUF blocks."""
    if kind == 0:
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(
            np.float32
        ).reshape(rows, cols), "F32"
    if kind not in (1, 2) or cols % 32:
        raise ValueError("Unsupported Kyse matrix format or width")
    count = rows * cols
    payload = count if kind == 1 else count // 2
    scales = np.frombuffer(raw, np.uint8, offset=payload).reshape(-1, 2)
    codes = np.frombuffer(raw, np.uint8, count=payload)
    size = 34 if kind == 1 else 18
    packed = np.empty((count // 32, size), np.uint8)
    packed[:, :2] = scales
    if kind == 1:
        packed[:, 2:] = codes.reshape(-1, 32)
    else:
        original = codes.reshape(-1, 16)
        values = np.empty((count // 32, 32), np.uint8)
        values[:, ::2] = original & 15
        values[:, 1::2] = original >> 4
        packed[:, 2:] = values[:, :16] | (values[:, 16:] << 4)
    return packed.reshape(rows, cols // 32 * size), (
        "Q8_0" if kind == 1 else "Q4_0"
    )


class Reader:
    def __init__(self, path):
        self.data = np.memmap(path, mode="r", dtype=np.uint8)
        self.offset = 0

    def take(self, size, *, aligned=True):
        if aligned:
            self.offset = (self.offset + 63) // 64 * 64
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ValueError("Truncated Kyse checkpoint")
        value = self.data[self.offset : end]
        self.offset = end
        return value

    def integers(self, count):
        return struct.unpack(f"<{count}i", self.take(4 * count, aligned=False))


def read_tokenizer(path):
    path = Path(path)
    if path.stat().st_size > 64 << 20:
        raise ValueError("Kyse tokenizer exceeds 64 MiB")
    reader = Reader(path)
    count, special_count, max_length = reader.integers(3)
    if not 256 <= count <= 1_000_000 or not 1 <= special_count <= count:
        raise ValueError("Invalid Kyse vocabulary size")
    specials = list(reader.integers(special_count))
    tokens, ranks = [], []
    for _ in range(count):
        rank, length = reader.integers(2)
        if not 0 <= length <= max_length:
            raise ValueError("Invalid Kyse token length")
        tokens.append(
            base64.b64encode(reader.take(length, aligned=False)).decode(
                "ascii"
            )
        )
        ranks.append(rank)
    if reader.offset != len(reader.data):
        raise ValueError("Trailing Kyse tokenizer data")
    document = dict(
        type="ygaraty-kyse-bpe",
        version=1,
        tokens=tokens,
        ranks=ranks,
        special_ids=specials,
    )
    tokenizer = KyseTokenizer(document)
    required = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    if any(tokenizer.token_to_id(t) is None for t in required):
        raise ValueError("Kyse tokenizer is missing chat/stop tokens")
    return document, tokenizer


def convert_checkpoint(source, output):
    """Create an immutable GGUF/config/tokenizer directory for one worker."""
    import gguf

    source, output = Path(source), Path(output).absolute()
    if output.exists():
        raise FileExistsError("Use a new Kyse import output directory")
    reader = Reader(source / "model.bin")
    if bytes(reader.take(4, aligned=False)) != b"QW35":
        raise ValueError("Expected a Kyse QW35 model.bin")
    (version,) = reader.integers(1)
    if version not in (1, 2, 3, 4):
        raise ValueError("Unsupported Kyse model version")
    d, f, layers, vocab, heads, kv, hd, rot, lk, lv, dk, dv, conv = (
        reader.integers(13)
    )
    eps, theta = struct.unpack("<2f", reader.take(8, aligned=False))
    if not 1 <= layers <= 256 or not 256 <= vocab <= 1_000_000:
        raise ValueError("Invalid Kyse model dimensions")
    kinds = reader.integers(layers)
    matrix_kind, embed_kind = reader.integers(2) if version >= 2 else (0, 0)
    (untied,) = reader.integers(1) if version >= 3 else (0,)
    if (
        matrix_kind not in (0, 1, 2)
        or embed_kind not in (0, 1, 2)
        or untied not in (0, 1)
    ):
        raise ValueError("Invalid Kyse format flags")
    if lk != lv or rot != 64 or not np.isclose(eps, 1e-6, rtol=0, atol=1e-12):
        raise ValueError(
            "Kyse import currently requires the 0.8B head layout and RMSNorm epsilon"
        )
    config = dict(
        model_type="qwen3_5_text",
        hidden_size=d,
        intermediate_size=f,
        num_hidden_layers=layers,
        vocab_size=vocab,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=hd,
        linear_num_key_heads=lk,
        linear_num_value_heads=lv,
        linear_key_head_dim=dk,
        linear_value_head_dim=dv,
        linear_conv_kernel_dim=conv,
        rms_norm_eps=1e-6,
        rope_parameters=dict(
            rope_theta=theta, partial_rotary_factor=0.25, rope_type="default"
        ),
        layer_types=[
            "full_attention" if k else "linear_attention" for k in kinds
        ],
        hidden_act="silu",
        tie_word_embeddings=not bool(untied),
    )
    if any(k not in (0, 1) for k in kinds):
        raise ValueError("Invalid Kyse layer type")
    count = 1 + 7 * sum(kinds) + 8 * (layers - sum(kinds)) + untied
    formats = None
    if version == 4:
        (declared,) = reader.integers(1)
        if declared != count:
            raise ValueError("Incorrect Kyse matrix count")
        formats = iter(reader.integers(count))
    document, tokenizer = read_tokenizer(source / "tokenizer.bin")
    if len(tokenizer.tokens) != vocab:
        raise ValueError("Kyse model/tokenizer vocabulary mismatch")
    config["eos_token_id"] = [
        tokenizer.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".kyse-", dir=output.parent
    ) as temp:
        temp = Path(temp)
        (temp / "config.json").write_text(json.dumps(config, indent=2))
        WebGPUConfig.from_file(temp / "config.json")
        writer = gguf.GGUFWriter(
            temp / "model.gguf", "qwen35", use_temp_file=True
        )
        writer.add_block_count(layers)
        writer.add_context_length(4096)
        quantization = {}

        def matrix(name, rows, cols, default=matrix_kind):
            entry = next(formats) if formats is not None else default
            if entry not in (0, 1, 2):
                raise ValueError(
                    "Kyse low-rank corrections are not supported; use en-q8 or en-q4-mix-down"
                )
            if cols <= 0 or rows <= 0 or (entry and cols % 32):
                raise ValueError("Invalid Kyse matrix dimensions")
            n = rows * cols
            size = (
                2 * n
                if entry == 0
                else (n if entry == 1 else n // 2) + n // 32 * 2
            )
            packed, kind = repack_matrix(reader.take(size), entry, rows, cols)
            writer.add_tensor(
                name, packed, raw_dtype=gguf.GGMLQuantizationType[kind]
            )
            quantization[kind] = quantization.get(kind, 0) + 1

        def vector(name, shape, transform=None):
            values = (
                reader.take(int(np.prod(shape)) * 4).view("<f4").reshape(shape)
            )
            if not np.isfinite(values).all():
                raise ValueError("Non-finite Kyse vector: " + name)
            if transform is not None:
                values = transform(values)
            writer.add_tensor(name, values)

        try:
            matrix("token_embd.weight", vocab, d, embed_kind)
            width = 2 * lk * dk + lv * dv
            for layer, full in enumerate(kinds):
                p = f"blk.{layer}."
                vector(p + "attn_norm.weight", (d,))
                vector(p + "post_attention_norm.weight", (d,))
                if full:
                    for name, rows, cols in [
                        ("attn_q", 2 * heads * hd, d),
                        ("attn_k", kv * hd, d),
                        ("attn_v", kv * hd, d),
                        ("attn_output", d, heads * hd),
                    ]:
                        matrix(p + name + ".weight", rows, cols)
                    vector(p + "attn_q_norm.weight", (hd,))
                    vector(p + "attn_k_norm.weight", (hd,))
                else:
                    for name, rows, cols in [
                        ("attn_qkv", width, d),
                        ("attn_gate", lv * dv, d),
                        ("ssm_alpha", lv, d),
                        ("ssm_beta", lv, d),
                        ("ssm_out", d, lv * dv),
                    ]:
                        matrix(p + name + ".weight", rows, cols)
                    vector(p + "ssm_conv1d.weight", (width, conv))
                    vector(p + "ssm_a", (lv,), lambda x: -np.exp(x))
                    vector(p + "ssm_dt.bias", (lv,))
                    vector(p + "ssm_norm.weight", (dv,))
                matrix(p + "ffn_gate.weight", f, d)
                matrix(p + "ffn_up.weight", f, d)
                matrix(p + "ffn_down.weight", d, f)
            vector("output_norm.weight", (d,))
            if untied:
                matrix("output.weight", vocab, d, embed_kind)
            trailing = reader.data[reader.offset :]
            if len(trailing) >= 64 or np.any(trailing):
                raise ValueError("Unexpected trailing Kyse model data")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        (temp / "tokenizer.json").write_text(json.dumps(document))
        (temp / "chat-format.json").write_text(
            json.dumps(chat_format(), indent=2)
        )
        # Bootstrap discovers this local template without a custom chat flag.
        template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{% endif %}"
        (temp / "tokenizer_config.json").write_text(
            json.dumps(dict(chat_template=template))
        )
        hashes = {}
        for filename in ("model.bin", "tokenizer.bin"):
            with (source / filename).open("rb") as handle:
                hashes[filename] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
        record = dict(
            format="kyse-qw35",
            version=version,
            vocab_size=vocab,
            layers=layers,
            quantization=quantization,
            source_sha256=hashes,
            quantization_preserved=True,
            max_workers=1,
        )
        (temp / "import.json").write_text(json.dumps(record, indent=2))
        temp.rename(output)
    return dict(record, output=str(output))


def launch_checkpoint(source, workspace, workers):
    """Recognize raw Kyse files and cache an import outside bootstrap state."""
    source = Path(source)
    if not (source / "model.bin").is_file():
        return source
    with (source / "model.bin").open("rb") as handle:
        if handle.read(4) != b"QW35":
            return source
    if workers != 1:
        raise ValueError("Kyse launch currently supports --num-workers 1 only")
    hashes = {}
    for name in ("model.bin", "tokenizer.bin"):
        with (source / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    target = Path(workspace).parent / ".kyse-imports" / identity
    if not target.exists():
        print("Importing Kyse weights and English tokenizer…", flush=True)
        convert_checkpoint(source, target)
    elif (
        json.loads((target / "import.json").read_text())["source_sha256"]
        != hashes
    ):
        raise ValueError("Kyse import cache does not match its source")
    return target
