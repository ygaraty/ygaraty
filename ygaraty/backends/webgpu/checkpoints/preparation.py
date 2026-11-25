"""Shared packed-weight manifests and device self-test preparation."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EMBED_CHUNK_BYTES = 128 * 2**20


def source(model, tensor, rows=None):
    """(path, offset, length) of a tensor's bytes, or of a range of rows; a
    row slice of a fused tensor (`Tensor.base`) is a range of the stored one.
    """
    stored = tensor.base or tensor.name
    t = model.reader.tensors[
        [x.name for x in model.reader.tensors].index(stored)
    ]
    offset = int(t.data_offset) + tensor.first_row * tensor.row_bytes
    length = tensor.data.nbytes
    if rows is not None:
        first, count = rows
        offset += first * tensor.row_bytes
        length = count * tensor.row_bytes
    return (model.path, offset, length)


def describe(tensor, file):
    return dict(
        name=tensor.name,
        kind=tensor.kind,
        shape=list(tensor.shape),
        row_bytes=tensor.row_bytes,
        file=file,
    )


def f16_values(n=1 << 18, seed=2026):
    """float32 test values for the float16 conversion: normal, subnormal and
    overflowing magnitudes, exact ties of both parities, and specials."""
    rng = np.random.default_rng(seed)
    scales = 10.0 ** rng.uniform(-9, 5.5, n)
    x = (rng.standard_normal(n) * scales).astype(np.float32)
    halves = (
        rng.integers(0, 0x7BFF, 4096, dtype=np.uint16)
        .view(np.float16)
        .astype(np.float32)
    )
    ties = (
        halves + np.spacing(halves.astype(np.float16)).astype(np.float32) / 2
    ).astype(np.float32)
    specials = np.array(
        [
            0.0,
            -0.0,
            np.inf,
            -np.inf,
            np.nan,
            65504,
            65519.996,
            65520,
            6.1035156e-05,
            5.9604645e-08,
            2.9802322e-08,
            2.9802326e-08,
            -1.0,
        ],
        dtype=np.float32,
    )
    x[: len(ties)] = ties
    x[len(ties) : len(ties) + len(specials)] = specials
    return x


def embed_chunks(model):
    t = model["token_embd.weight"]
    rows = EMBED_CHUNK_BYTES // t.row_bytes
    chunks, files = [], {}
    for k, first in enumerate(range(0, t.shape[0], rows)):
        count = min(rows, t.shape[0] - first)
        files[f"embed{k}"] = source(model, t, (first, count))
        chunks.append(
            dict(
                kind=t.kind,
                shape=[count, t.shape[1]],
                row_bytes=t.row_bytes,
                file=f"embed{k}",
                first=first,
            )
        )
    return chunks, files


def head_chunks(model):
    t = model["output.weight"]
    rows = EMBED_CHUNK_BYTES // t.row_bytes
    chunks, files = [], {}
    for k, first in enumerate(range(0, t.shape[0], rows)):
        count = min(rows, t.shape[0] - first)
        files[f"output{k}"] = source(model, t, (first, count))
        chunks.append(
            dict(
                kind=t.kind,
                shape=[count, t.shape[1]],
                row_bytes=t.row_bytes,
                file=f"output{k}",
                first=first,
            )
        )
    return chunks, files


def stage_manifest(model, first, last, head=False):
    """Manifest and files of one stage: the embedding when it starts at layer
    0, layers first..last-1, the output norm and projection when it ends at
    the last layer (or when `head`: a shortened model for measurements, WS-33)."""
    if not 0 <= first < last <= model.config.layers:
        raise ValueError("Invalid stage layer range")
    model.validate_webgpu()
    files, layers = {}, []
    for index in range(first, last):
        tensors = {}
        for name, t in model.layer_tensors(index).items():
            key = f"L{index}.{name}"
            files[key] = source(model, t)
            tensors[name] = describe(t, key)
        layers.append(
            dict(index=index, type=model.layer_type(index), tensors=tensors)
        )
    manifest = dict(
        config=model.config.manifest(), layers=layers, embed=None, head=None
    )
    if first == 0:
        chunks, more = embed_chunks(model)
        files.update(more)
        manifest["embed"] = dict(chunks=chunks)
    if last == model.config.layers or head:
        chunks, more = head_chunks(model)
        files.update(more)
        files["output_norm"] = source(model, model["output_norm.weight"])
        manifest["head"] = dict(
            shared_embedding=model["output.weight"].name
            == "token_embd.weight",
            chunks=chunks,
            vocab=model["output.weight"].shape[0],
            norm=describe(model["output_norm.weight"], "output_norm"),
        )
    from ..distribution import compile_stage

    manifest["plan"] = compile_stage(manifest)
    return manifest, files


def selftest_fixture(self, model):
    rng = np.random.default_rng(36)
    self.test_files, cases, self.expected = {}, [], []
    from ygaraty.backends.webgpu._legacy.qwen_protocol import (
        SELFTEST_CASES,
        dense_selftest_cases,
    )

    for i, (name, rows, tokens, *kernel) in enumerate(
        dense_selftest_cases(model)
        if model.family in ("dense", "kolibri1", "qwen3moe")
        else SELFTEST_CASES
    ):
        t = model[name]
        # Smaller layouts have fewer rows (0.8B: 16 DeltaNet value heads, so
        # 16 ssm_alpha rows); the 9B's ranges all fit and are unchanged.
        rows = (rows[0], min(rows[1], t.shape[0] - rows[0]))
        if rows[1] < 1:
            raise ValueError(f"Self-test rows outside {name}")
        self.test_files[f"t{i}"] = read_range(source(model, t, rows))
        x = rng.standard_normal((tokens, t.shape[1])).astype(np.float32)
        self.test_files[f"tx{i}"] = x.tobytes()
        W = t.dequantize(rows=np.arange(rows[0], rows[0] + rows[1]))
        cases.append(
            dict(
                name=name,
                kind=t.kind,
                shape=list(W.shape),
                row_bytes=t.row_bytes,
                file=f"t{i}",
                x=f"tx{i}",
                tokens=tokens,
                kernel=kernel[0] if kernel else None,
            )
        )
        self.expected.append(x.astype(np.float64) @ W.astype(np.float64).T)
    self.f16_x = f16_values()
    self.test_files["f16"] = self.f16_x.tobytes()
    with np.errstate(over="ignore"):
        self.f16_want = self.f16_x.astype(np.float16)
    self.selftest = dict(cases=cases, f16="f16")


def read_range(source):
    path, offset, length = source
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise ValueError("Truncated weight range")
    return data


def _hash_files(session):
    for name, (path, offset, length) in session.files.items():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            handle.seek(offset)
            remaining = length
            while remaining:
                block = handle.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("Truncated weight range")
                digest.update(block)
                remaining -= len(block)
        session.hashes[name] = digest.hexdigest()


def inkling2_selftest(session, path, header):
    """DO-95: the device self-test for Inkling 2 stages: the generic matrix kernel over the first rows of the
    first tensor of each packed kind (Q8_0 attention, IQ4_NL experts), at 1, 3, 21 and 64 tokens, against NumPy
    dequantization of the same bytes; and the float16 conversion check every bundle carries."""
    import gguf

    rng = np.random.default_rng(36)
    session.test_files, cases, session.expected = {}, [], []
    first = {}
    for name in sorted((n for n in header.tensors if n.startswith("blk.")), key=lambda n: (int(n.split(".")[1]), n)):
        first.setdefault(header.tensors[name].kind, name)
    first.pop("F32", None)
    i = 0
    for kind, name in sorted(first.items()):
        t = header.tensors[name]
        cols, row_bytes = t.shape[-1], t.row_bytes
        rows = min(256, t.nbytes // row_bytes)
        with open(path, "rb") as handle:
            handle.seek(t.offset)
            raw = handle.read(rows * row_bytes)
        W = gguf.quants.dequantize(np.frombuffer(raw, np.uint8).reshape(rows, row_bytes),
                                   gguf.GGMLQuantizationType[kind]).astype(np.float32)
        for tokens in (1, 3, 21, 64):
            session.test_files[f"t{i}"] = raw
            x = rng.standard_normal((tokens, cols)).astype(np.float32)
            session.test_files[f"tx{i}"] = x.tobytes()
            cases.append(dict(name=name, kind=kind, shape=[rows, cols], row_bytes=row_bytes, file=f"t{i}", x=f"tx{i}",
                              tokens=tokens, kernel=None))
            session.expected.append(x.astype(np.float64) @ W.astype(np.float64).T)
            i += 1
    session.f16_x = f16_values()
    session.test_files["f16"] = session.f16_x.tobytes()
    with np.errstate(over="ignore"):
        session.f16_want = session.f16_x.astype(np.float16)
    session.selftest = dict(cases=cases, f16="f16")


def prepare_inkling2_session(args, header, config):
    """DO-95: Inkling 2 stages (`inkling2_gguf.stage_manifest`, a file per tensor) on the float32 wire."""
    from ..distribution import compile_distribution
    from .families import FAMILIES
    from .inkling2_gguf import config_from_header, stage_manifest

    c = config_from_header(header)
    bounds = [int(value) for value in args.splits.split(",")]
    if bounds[0] != 0 or bounds[-1] != c.layers or any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError("Splits must cover all model layers in order")
    errors = FAMILIES["inkling2"].check_splits(header, bounds)
    if errors:
        raise ValueError("; ".join(errors))
    eos = config.get("eos_token_id")
    settings = SimpleNamespace(capacity=args.capacity, generate=args.generate, chunk=64, kv="f32")
    session = SimpleNamespace(args=settings, files={}, hashes={}, stages=[],
                              eos=[eos] if isinstance(eos, int) else list(eos or []), chat_style=None)
    for index, (first, last) in enumerate(zip(bounds, bounds[1:])):
        manifest, files = stage_manifest(args.gguf, first, last, header=header)
        session.files.update(files)
        fixture = dict(chunks=[], prefill_calls=0, max_tokens=min(64, args.capacity), max_positions=args.capacity,
                       kv="f32", wire="f32", prompts=None, conversations=1, batch=False, route="relay", ice=[],
                       ice_policy="all", stages=len(bounds) - 1, generate=args.generate, stop_ids=sorted(session.eos),
                       draft=None, digests=False, matvec=None)
        session.stages.append(dict(stage=index, layers=[first, last], manifest=manifest, fixture=fixture, names=sorted(files)))
    session.distribution_plan = compile_distribution(session.stages)
    _hash_files(session)
    inkling2_selftest(session, args.gguf, header)
    return session


def prepare_session(args):
    """Build export inputs without launching a session, browser or listener."""
    import json

    from ygaraty.backends.webgpu._legacy.qwen_protocol import chat_format

    from .families import load_model
    from .gguf_remote import read_header

    header = read_header(args.gguf)
    if header.architecture() == "inkling2_loop":
        return prepare_inkling2_session(args, header, json.loads(Path(args.config).read_text()))

    model = load_model(args.gguf, args.config, tensor_only=True)
    config = json.loads(Path(args.config).read_text())
    eos = config.get("eos_token_id") or config.get("text_config", {}).get(
        "eos_token_id"
    )
    bounds = [int(value) for value in args.splits.split(",")]
    if (
        bounds[0] != 0
        or bounds[-1] != model.config.layers
        or any(a >= b for a, b in zip(bounds, bounds[1:]))
    ):
        raise ValueError("Splits must cover all model layers in order")
    settings = SimpleNamespace(
        capacity=args.capacity, generate=args.generate, chunk=64, kv="f16"
    )
    session = SimpleNamespace(
        args=settings,
        files={},
        hashes={},
        stages=[],
        eos=[eos] if isinstance(eos, int) else list(eos or []),
        chat_style=chat_format(json.loads(Path(args.chat_format).read_text()))
        if args.chat_format
        else None,
    )
    for index, (first, last) in enumerate(zip(bounds, bounds[1:])):
        manifest, files = stage_manifest(model, first, last)
        session.files.update(files)
        fixture = dict(
            chunks=[],
            prefill_calls=0,
            max_tokens=min(64, args.capacity),
            max_positions=args.capacity,
            kv="f16",
            wire="f16",
            prompts=None,
            conversations=1,
            batch=False,
            route="relay",
            ice=[],
            ice_policy="all",
            stages=len(bounds) - 1,
            generate=args.generate,
            stop_ids=sorted(session.eos),
            draft=None,
            digests=False,
            matvec=None,
        )
        session.stages.append(
            dict(
                stage=index,
                layers=[first, last],
                manifest=manifest,
                fixture=fixture,
                names=sorted(files),
            )
        )
    from ..distribution import compile_distribution

    for stage in session.stages:
        stage["manifest"].pop("plan", None)
    session.distribution_plan = compile_distribution(session.stages)
    _hash_files(session)
    selftest_fixture(session, model)
    return session
