"""Shared Qwen3.5 WebGPU checks (DO-36 / DO-42), on an authorized GPU host.

inspect reads local metadata only; other modes execute actual GPU workloads.
Reference comparisons use optional Torch/transformers on the same weights.

    python -m ygaraty.experiments.qwen35_webgpu kernels --gguf FILE --out DIR
    python -m ygaraty.experiments.qwen35_webgpu stage --gguf FILE --config config.json \
        --tokenizer tokenizer.json --out DIR [--first 0 --last 8] [--chunks 9,1,1,1]
    python -m ygaraty.experiments.qwen35_webgpu pipeline --gguf FILE --config config.json \
        --tokenizer tokenizer.json --out DIR [--splits 0,8,16,24,32] [--generate 16]
    python -m ygaraty.experiments.qwen35_webgpu contributors --gguf FILE --config config.json \
        --tokenizer tokenizer.json --text FILE --out DIR [--prompt-tokens 8192] [--generate 128]

kernels: one matrix of each block type (Q4_K, Q5_K, Q6_K, Q8_0) times random
inputs, against the `gguf` package's NumPy dequantization in float64.
stage: the embedding (first stage) and layers first..last-1, prefill then
cached one-token steps (`--chunks`: tokens per call), against transformers'
decoder layers in float32 (`qwen35_gguf.StageReference`); also the DeltaNet
states after the last call. A later stage's input is the reference's hidden
states from the layers before it.
pipeline: all stages in one browser, prefill then greedy tokens chosen on the
GPU; the reference then runs the prompt plus the generated tokens in one
causal pass, one stage at a time, and the final norm and output projection at
every generated position (teacher forced: the logits cached decoding should
have produced). Each step compares the logits and the greedy token.
contributors: each stage alone in its own browser, one after another, as the
four contributors run them: prefill a long prompt (tokens of `--text`) in
chunks, then single-token steps; each stage's inputs are the previous one's
wire outputs. No reference: this measures memory (application allocations,
browser processes, the driver's per-client GPU memory) and time per call.

Precision (`--kv`, `--wire`, default f16 both): the KV cache and the hidden
states between stages are float16; the reference applies the same roundings
(keys and values as the cache stores them, hidden states at each boundary).

These are development checks, not a frozen correctness gate (see
public/research/092-qwen35-webgpu-four-contributors.md).
"""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.browser_binary import run_page
from ygaraty.backends.webgpu.checkpoints import qwen35_gguf as q
from ygaraty.backends.webgpu.checkpoints.families import load_model
from ygaraty.backends.webgpu.checkpoints.preparation import (
    source,
    describe,
    f16_values,
    embed_chunks,
    head_chunks,
    stage_manifest,
)

MODULES = (
    "qwen35_gpu.mjs",
    "dense_gpu.mjs",
    "kolibri_gpu.mjs",
    "inference_plan.mjs",
    "inference_schema.mjs",
    "inference_contracts.mjs",
    "inference_layout.mjs",
    "webgpu_kernels.mjs",
)
EMBED_CHUNK_BYTES = 128 * 2**20  # the default maxStorageBufferBindingSize
KERNEL_CASES = (  # (tensor, tokens, rows or None): <= 4 tokens use the
    # matrix-vector kernel, 5-15 the tiled one, 16 or more the register-tiled
    # one (gemm); Q8_0 the generic one
    ("blk.0.attn_qkv.weight", 1, None),  # Q5_K
    ("blk.0.attn_qkv.weight", 21, None),
    ("blk.0.attn_gate.weight", 3, None),  # Q4_K
    ("blk.0.attn_gate.weight", 16, None),
    ("blk.0.ffn_up.weight", 1, None),  # Q4_K, 12288 rows
    (
        "blk.0.ffn_down.weight",
        1,
        None,
    ),  # Q6_K, 12288 columns (odd blocks unaligned)
    ("blk.0.ffn_down.weight", 37, None),
    ("blk.0.ffn_up.weight", 64, None),  # a prefill chunk
    ("blk.0.ffn_down.weight", 64, None),
    ("blk.0.attn_qkv.weight", 64, None),
    ("blk.3.attn_v.weight", 64, None),
    ("blk.0.ssm_alpha.weight", 3, None),  # Q8_0
    ("blk.3.attn_v.weight", 9, None),  # Q6_K, 1024 rows
    ("output.weight", 5, (1000, 1001)),  # Q6_K, rows not a multiple of 16
    (
        "blk.0.ffn_up.weight",
        2,
        None,
    ),  # DO-38: a decode batch of 2, 4 and 8 tokens
    ("blk.0.ffn_up.weight", 4, None),
    ("blk.0.ffn_up.weight", 8, None),
)
# WS-15: decode-shaped products of each quantization the Qwen3.5-9B stages use, at 1-16 tokens.
DECODE_KERNEL_CASES = tuple(
    (name, tokens, None)
    for name in (
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_qkv.weight",
    )
    for tokens in (1, 2, 4, 5, 8, 9, 16)
)
# WS-34: one decode token through each weight format at full size (Qwen3.5-9B: Q4_K, Q5_K, Q6_K; the head).
DECODE1_KERNEL_CASES = tuple(
    (name, 1, None)
    for name in (
        "blk.0.ffn_up.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_qkv.weight",
        "blk.3.attn_q.weight",
    )
) + (("output.weight", 1, (0, 32768)),)
# WS-46: the Q8_0 matrices (Qwen3.5's ssm_alpha and ssm_beta, 32 x 4096) through every kernel: one token
# (coalesced), batches (batched), 9 (tiled) and 64 (the GEMM's tiled fallback).
Q8_KERNEL_CASES = tuple(
    (name, tokens, None)
    for name in ("blk.0.ssm_alpha.weight", "blk.5.ssm_beta.weight")
    for tokens in (1, 2, 3, 8, 9, 32, 64)
)
# The dense family's matrices (llama.cpp Q4_K_M: Q4_K, with Q6_K for half the attn_v/ffn_down and the head).
DENSE_KERNEL_CASES = (
    ("blk.0.attn_q.weight", 1, None),
    ("blk.0.attn_q.weight", 21, None),
    ("blk.0.attn_k.weight", 3, None),
    ("blk.0.ffn_up.weight", 1, None),
    ("blk.0.ffn_down.weight", 1, None),
    ("blk.0.ffn_down.weight", 37, None),
    ("blk.1.ffn_down.weight", 9, None),
    ("blk.0.ffn_up.weight", 64, None),
    ("blk.0.attn_v.weight", 64, None),
    ("blk.1.attn_v.weight", 5, None),
    ("output.weight", 5, (1000, 1001)),
    ("blk.0.ffn_up.weight", 4, None),
)
# Kolibri-1 (our converter: Q4_K experts, Q6_K attention/shared expert/head, float32 router): every matrix
# kind, the 6,144-column attention output, the 512-column expert down projection, one expert chunk's rows.
KOLIBRI_KERNEL_CASES = (
    ("blk.0.attn_q.weight", 1, None),
    ("blk.0.attn_q.weight", 21, None),
    ("blk.0.attn_output.weight", 1, None),
    ("blk.0.attn_output.weight", 64, None),
    ("blk.0.ffn_gate_up_exps.0.weight", 1, (0, 1024)),
    ("blk.0.ffn_gate_up_exps.0.weight", 64, (1024, 1024)),
    ("blk.0.ffn_down_exps.0.weight", 1, (0, 2560)),
    ("blk.0.ffn_down_exps.0.weight", 9, (0, 2560)),
    ("blk.0.ffn_gate_inp.weight", 1, None),
    ("blk.0.ffn_gate_inp.weight", 64, None),
    ("blk.0.ffn_down_shexp.weight", 3, None),
    ("output.weight", 5, (1000, 1001)),
)


def parse_matvec_c(text):
    """WS-34/DO-74: "TPR,L[,WG]" for every format, then optional ";KIND=TPR,L[,WG]" overrides (e.g.
    "16,4;Q4_K=8,2,256")."""

    def one(spec):
        v = [int(x) for x in spec.split(",")]
        if (
            len(v) not in (2, 3, 4)
            or v[0] not in (4, 8, 16, 32, 64)
            or v[1] not in (1, 2, 4, 8)
            or v[1] > v[0]
            or (len(v) >= 3 and (v[2] not in (64, 128, 256) or v[2] < v[0]))
            or (len(v) == 4 and v[3] not in (1, 2, 4, 8))
        ):
            raise SystemExit(f"--matvec-c: bad TPR,L[,WG[,R]] {spec!r}")
        return dict(
            zip(("tpr", "lanes", "wg", "rows"), v)
        )  # DO-76: R rows per thread

    first, *rest = text.split(";")
    out = one(first)
    if rest:
        out["kinds"] = {k: one(v) for k, v in (r.split("=") for r in rest)}
    return out


def sweep_shapes(text):
    """DO-74: "all" = every valid coalesced shape (TPR 4..64, L 1..8 with L <= TPR, workgroups 64..256 of at least
    TPR threads), else shapes separated by "/" in --matvec-c's form."""
    if text != "all":
        return [parse_matvec_c(t) for t in text.split("/")]
    return [
        dict(tpr=t, lanes=l, wg=w)
        for t in (4, 8, 16, 32, 64)
        for l in (1, 2, 4, 8)
        for w in (64, 128, 256)
        if l <= t and w >= t
    ]


def shape_name(cfg):
    return f"{cfg['tpr']},{cfg['lanes']},{cfg.get('wg', 64)}"


def kernel_options(args):
    """Diagnostic overrides of the many-token matmul kernel for the page."""
    out = {}
    if args.prefill_kernel:
        out["prefill_kernel"] = args.prefill_kernel
    if args.gemm:
        parts = args.gemm.split(
            ","
        )  # WS-50: a fifth field "v" stages the tiles as vec4
        out["gemm"] = dict(
            zip(("GR", "GT", "RR", "RT"), (int(v) for v in parts[:4])),
            **({"V": True} if parts[4:] == ["v"] else {}),
        )
    if getattr(args, "matvec", None):
        v = [
            int(x) for x in args.matvec.split(",")
        ]  # R,VT or SINGLE,ROWS,TOKENS (the sessions' 2,4,8)
        out["matvec"] = (
            dict(zip(("rows", "tokens"), v))
            if len(v) == 2
            else dict(single=v[0], rows=v[1], tokens=v[2])
        )
    if getattr(
        args, "attention_split", None
    ):  # WS-35: keys per attention split for decode steps
        out["attention_split"] = args.attention_split
    if getattr(
        args, "matvec_c", None
    ):  # WS-34: TPR,L for the coalesced one-token kernel
        out["matvec_c"] = parse_matvec_c(args.matvec_c)
    if getattr(
        args, "q8_fast_up_to", None
    ):  # WS-47: Q8_0 vectorized for calls of at most N tokens
        out["q8_fast_up_to"] = args.q8_fast_up_to
    if getattr(
        args, "q8_fast", False
    ):  # WS-46 (experiment 130, opt-in): Q8_0 through the vectorized decoders
        out["q8_fast"] = True
    if getattr(
        args, "no_wgsl_memo", False
    ):  # WS-33: the arm without the shader-source cache
        out["no_wgsl_memo"] = True
    if getattr(
        args, "repack", None
    ):  # DO-80: K-quant formats repacked at load (exact), e.g. Q4_K,Q5_K,Q6_K
        out["repack"] = [k for k in args.repack.split(",") if k]
    if getattr(
        args, "repack_q4k", False
    ):  # WS-40: layer Q4_K matrices repacked at load (exact)
        out["repack_q4k"] = True
    if getattr(
        args, "norm_fold", False
    ):  # WS-49: a one-token norm folded into the grouped product that reads it
        out["norm_fold"] = True
    if getattr(
        args, "group_matmul", False
    ):  # WS-36: one-token products of the same input in one dispatch
        out["group_matmul"] = True
    if getattr(
        args, "matvec_b", None
    ):  # DO-69: tokens per pass of the batched coalesced kernel
        out["matvec_b"] = args.matvec_b
    return out


def commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def inspect(args):
    """Check local GGUF metadata without a browser or reference framework."""
    model = load_model(args.gguf, args.config)
    report = {
        "family": model.family,
        "mode": "inspect",
        "config": model.config.manifest(),
        "tensor_types": model.validate_webgpu(),
        "output_tensor": model["output.weight"].name,
        "status": "metadata compatible; GPU parity not established",
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write(args, report)


def kernels(args):
    if getattr(
        args, "calibrate_repack", False
    ):  # DO-83: only the repack measurement (synthetic matrices, any model)
        files = {
            "manifest.json": json.dumps(
                dict(
                    mode="kernels",
                    cases=[],
                    calibrate_repack=True,
                    **kernel_options(args),
                )
            ).encode()
        }
        result = run_page(
            "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
        )
        if result.get("error"):
            raise SystemExit(result["error"])
        print("repack calibration:", json.dumps(result["repackCalibration"]))
        write(
            args,
            dict(
                mode="kernels",
                adapter=result["adapter"],
                repack_calibration=result["repackCalibration"],
            ),
        )
        return
    model = (
        load_model(args.gguf, args.config)
        if getattr(args, "config", None)
        else q.Model(args.gguf)
    )
    rng = np.random.default_rng(2026)
    files, cases, expected, W_bytes = {}, [], [], []
    kernel_cases = dict(
        dense=DENSE_KERNEL_CASES, kolibri1=KOLIBRI_KERNEL_CASES
    ).get(model.family, KERNEL_CASES)
    if getattr(args, "cases", "default") == "decode":
        kernel_cases = DECODE_KERNEL_CASES
    elif getattr(args, "cases", "default") == "decode1":
        kernel_cases = DECODE1_KERNEL_CASES
    elif getattr(args, "cases", "default") == "q8":
        kernel_cases = Q8_KERNEL_CASES
    for i, (name, tokens, rows) in enumerate(kernel_cases):
        t = model[name]
        x = rng.standard_normal((tokens, t.shape[1])).astype(np.float32)
        files[f"w{i}"] = source(model, t, rows)
        files[f"x{i}"] = x.tobytes()
        case = dict(describe(t, f"w{i}"), x=f"x{i}", tokens=tokens)
        W = (
            t.dequantize()
            if rows is None
            else t.dequantize(rows=np.arange(rows[0], rows[0] + rows[1]))
        )
        case["shape"] = list(W.shape)
        cases.append(case)
        expected.append(x.astype(np.float64) @ W.astype(np.float64).T)
        W_bytes.append(files[f"w{i}"][2])
    if getattr(
        args, "roof_mb", None
    ):  # DO-78: enough distinct copies of each matrix to exceed the GPU's L2
        for case, nbytes in zip(cases, W_bytes):
            case["copies"] = max(1, -(-int(args.roof_mb * 2**20) // nbytes))
    sweep = (
        sweep_shapes(args.sweep_matvec_c)
        if getattr(args, "sweep_matvec_c", None)
        else None
    )
    files["manifest.json"] = json.dumps(
        dict(
            mode="kernels",
            cases=cases,
            repeat=args.repeat,
            **kernel_options(args),
            **(dict(sweep=sweep, rounds=args.rounds) if sweep else {}),
            **(
                dict(
                    roof=True,
                    rounds=args.rounds,
                    roof_dispatches=args.roof_dispatches,
                )
                if getattr(args, "roof_mb", None)
                else {}
            ),
        )
    ).encode()
    result = run_page(
        "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    report = []
    for i, ((name, tokens, _), ref) in enumerate(zip(kernel_cases, expected)):
        got = np.fromfile(
            Path(args.out) / f"y{i}.f32", dtype=np.float32
        ).reshape(ref.shape)
        err = np.abs(got - ref)
        report.append(
            dict(
                tensor=name,
                kind=model[name].kind,
                tokens=tokens,
                max_abs=float(err.max()),
                max_rel=float(err.max() / np.abs(ref).max()),
                ms=result["timings"][i],
                repeat=args.repeat,
                weight_gb_per_s=W_bytes[i] / result["timings"][i] / 1e6
                if args.repeat
                else None,
                gflops=2
                * ref.size
                * cases[i]["shape"][1]
                / result["timings"][i]
                / 1e6
                if args.repeat
                else None,
            )
        )
        print(
            f"{name:26} {model[name].kind:5} {tokens:3} tokens {result['timings'][i]:7.2f} ms {W_bytes[i] / result['timings'][i] / 1e6:6.1f} GB/s {report[-1]['gflops'] or 0:6.0f} GFLOPS  max |err| {err.max():.3e}  relative to max |y| {err.max() / np.abs(ref).max():.3e}"
        )
    if sweep:  # DO-74: each shape's error and median GB/s per one-token case
        for i, (name, tokens, _) in enumerate(kernel_cases):
            times = result.get("sweeps", {}).get(str(i))
            if not times:
                continue
            ref = expected[i]
            rows = report[i]["sweep"] = []
            for s, cfg in enumerate(sweep):
                got = np.fromfile(
                    Path(args.out) / f"y{i}_s{s}.f32", dtype=np.float32
                ).reshape(ref.shape)
                ms = float(np.median(times[s]))
                rows.append(
                    dict(
                        shape=cfg,
                        ms=ms,
                        times=times[s],
                        weight_gb_per_s=W_bytes[i] / ms / 1e6,
                        max_rel=float(
                            np.abs(got - ref).max() / np.abs(ref).max()
                        ),
                    )
                )
            best = max(rows, key=lambda r: r["weight_gb_per_s"])
            print(
                f"{name:26} sweep of {len(rows)}: best {shape_name(best['shape'])} {best['weight_gb_per_s']:.1f} GB/s; "
                f"worst error {max(r['max_rel'] for r in rows):.2e}"
            )
    if getattr(
        args, "roof_mb", None
    ):  # DO-78: A/B/C GB/s, and A's and B's checksums against the file's bytes
        for i, (name, tokens, rows) in enumerate(kernel_cases):
            got = result.get("roofs", {}).get(str(i))
            if not got:
                continue
            path, offset, length = files[f"w{i}"]
            with open(path, "rb") as f:
                f.seek(offset)
                raw = np.frombuffer(f.read(length), dtype=np.uint8)
            n4 = length // 16
            want_a = int(np.bitwise_xor.reduce(raw[: n4 * 16].view("<u4")))
            got_a = int(
                np.bitwise_xor.reduce(
                    np.fromfile(Path(args.out) / f"roofA{i}.u32", dtype="<u4")
                )
            )
            t = model[name]
            block = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}.get(t.kind)
            span = {
                "Q4_K": (16, 144),
                "Q5_K": (48, 176),
                "Q6_K": (0, 128),
            }.get(t.kind)
            b_ok = None
            if block:
                nrows = length // t.row_bytes
                blocks = raw.reshape(nrows, t.row_bytes // block, block)[
                    :, :, span[0] : span[1]
                ]
                want_b = np.bitwise_xor.reduce(
                    np.ascontiguousarray(blocks)
                    .reshape(nrows, -1)
                    .view("<u4"),
                    axis=1,
                )
                got_b = np.fromfile(
                    Path(args.out) / f"roofB{i}.u32", dtype="<u4"
                )[:nrows]
                b_ok = bool(np.array_equal(want_b, got_b))
            med = {k: float(np.median(v)) for k, v in got["times"].items()}
            gbs = {k: length / v / 1e6 for k, v in med.items()}
            report[i]["roof"] = dict(
                copies=got["copies"],
                dispatches=got["dispatches"],
                times=got["times"],
                ms=med,
                gb_per_s=gbs,
                checksum_read=want_a == got_a,
                checksum_pattern=b_ok,
            )
            print(
                f"{name:26} {t.kind:5} roof x{got['copies']}: A read {gbs['A']:6.1f}  B pattern {gbs['B']:6.1f}  "
                f"C kernel {gbs['C']:6.1f} GB/s  (checksums A {want_a == got_a}, B {b_ok})"
            )
    write(
        args,
        dict(
            mode="kernels",
            adapter=result["adapter"],
            cases=report,
            **(dict(sweep=sweep, rounds=args.rounds) if sweep else {}),
        ),
    )


def f16(args):
    """The cache's float32 -> float16 conversion and back against NumPy (round
    to nearest even); WGSL's pack2x16float reported for comparison."""
    x = f16_values()
    files = {
        "x": x.tobytes(),
        "manifest.json": json.dumps(dict(mode="pack")).encode(),
    }
    result = run_page(
        "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    out = Path(args.out)
    with np.errstate(over="ignore"):
        want = x.astype(np.float16)
    same = lambda got: float(  # noqa: E731
        np.mean(
            (got.view(np.uint16) == want.view(np.uint16))
            | (np.isnan(got) & np.isnan(want))
        )
    )  # noqa: E731
    bits = np.fromfile(out / "bits.f16", dtype=np.float16)
    pack = np.fromfile(out / "pack.f16", dtype=np.float16)
    back = np.fromfile(out / "back.f32", dtype=np.float32)
    exact_back = want.astype(np.float32)
    back_same = float(
        np.mean(
            (back.view(np.uint32) == exact_back.view(np.uint32))
            | (np.isnan(back) & np.isnan(exact_back))
        )
    )
    report = dict(
        mode="f16",
        adapter=result["adapter"],
        values=len(x),
        f16_bits_equal_numpy=same(bits),
        pack2x16float_equal_numpy=same(pack),
        f16_value_exact=back_same,
    )
    print(
        f"f16_bits equal to NumPy: {report['f16_bits_equal_numpy']:.6f}; f16_value exact: {back_same:.6f}; "
        f"pack2x16float equal to NumPy: {report['pack2x16float_equal_numpy']:.6f}"
    )
    write(args, report)


def capacity(args):
    """How much one page's WebGPU device holds (gpu_capacity.mjs): buffers of --chunk-mb filled whole by a
    compute pass until --gb or the first error, then each checked; for single-contributor models on large
    GPUs, which need most of the card in one browser tab."""
    target = int(args.gb * 2**30)
    files = {
        "manifest.json": json.dumps(
            dict(target=target, chunk=args.chunk_mb * 2**20)
        ).encode()
    }
    result = run_page(
        "gpu_capacity.mjs",
        files,
        args.out,
        (),
        timeout=args.timeout,
        memory=True,
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    got = result.get("allocated_bytes", 0) / 2**30
    report = dict(
        result,
        target_gib=args.gb,
        allocated_gib=round(got, 2),
        reached=got + result.get("chunk", 0) / 2**30 > args.gb
        and not result.get("corrupt_samples")
        and not result.get("lost")
        and not result.get("first_error"),
    )
    print(
        f"{result.get('adapter', {}).get('description') or result.get('adapter')}: {got:.1f} of {args.gb} GiB "
        f"in {result.get('verified_buffers')} buffers of {result.get('chunk', 0) / 2**20:.0f} MiB; "
        f"corrupt samples {result.get('corrupt_samples')}; fill {result.get('fill_gib_per_s') or 0:.0f} GiB/s; "
        f"first error: {result.get('first_error')}; lost: {result.get('lost')}; "
        f"maxBufferSize {result.get('limits', {}).get('maxBufferSize')}, "
        f"maxStorageBufferBindingSize {result.get('limits', {}).get('maxStorageBufferBindingSize')}"
    )
    write(args, report)
    if not report["reached"]:
        raise SystemExit(1)


def profile_summary(steps):
    """Per decode token (median over steps): JavaScript encode time, GPU pass time (timestamp queries, None
    without them), wall time, dispatches and submissions, and the dispatch counts by name."""
    import statistics

    rows = []
    for st in steps:
        recs = st.get("profile") or []
        gpu = [r["gpuMs"] for r in recs]
        names = {}
        for r in recs:
            for k, v in r["byName"].items():
                names[k] = names.get(k, 0) + v
        rows.append(
            dict(
                encode=sum(r["encodeMs"] for r in recs),
                wall=sum(r["wallMs"] for r in recs),
                expand=sum(r.get("expandMs") or 0 for r in recs),
                gpu=None if not recs or None in gpu else sum(gpu),
                dispatches=sum(r["dispatches"] for r in recs),
                submissions=len(recs),
                names=names,
            )
        )
    med = lambda key: statistics.median(r[key] for r in rows)  # noqa: E731
    gpu = [r["gpu"] for r in rows]
    return dict(
        steps=len(rows),
        encode_ms=med("encode"),
        wall_ms=med("wall"),
        expand_ms=med("expand"),
        gpu_ms=None if None in gpu else statistics.median(gpu),
        dispatches=med("dispatches"),
        submissions=med("submissions"),
        by_name=sorted(rows[-1]["names"].items(), key=lambda kv: -kv[1]),
    )


BENCH_VARIANTS = [  # DO-63: decode variants on one loaded stage; matvec as WS-15/16 ({rows, tokens, single})
    dict(name="base", tokens=24),
    dict(name="detail", tokens=6, profile="detail"),
    dict(name="matvec-r1", tokens=16, matvec=dict(rows=1, tokens=4, single=1)),
    dict(name="matvec-r2", tokens=16, matvec=dict(rows=2, tokens=4, single=2)),
    dict(name="matvec-r4", tokens=16, matvec=dict(rows=4, tokens=4, single=4)),
    dict(name="batch-2", tokens=12, batch=2),
    dict(name="batch-4", tokens=12, batch=4),
    dict(name="batch-8", tokens=12, batch=8),
    dict(name="base-again", tokens=24),
]


def roofline_variants(contexts, tokens=16, detail=4):
    """WS-31: per context a fresh slot prefilled to it, `tokens` profiled decode steps then `detail` steps with a
    timestamped pass per dispatch (same slot, about `tokens` positions later); then base again at the first
    context in a new slot (steadiness check)."""
    out = []
    for i, c in enumerate(contexts):
        out += [
            dict(name=f"base-{c}", tokens=tokens, context=c, slot=i + 1),
            dict(
                name=f"detail-{c}", tokens=detail, profile="detail", slot=i + 1
            ),
        ]
    out.append(
        dict(
            name=f"base-again-{contexts[0]}",
            tokens=tokens,
            context=contexts[0],
            slot=len(contexts) + 1,
        )
    )
    return out


def resident_variants(contexts, tokens=32):
    """WS-32: per context B R R B, each in a fresh slot prefilled to it: B the usual loop (each token read back
    before the next step), R the GPU-resident loop; no profiling in either (timestamps would add read-backs)."""
    out, slot = [], 1
    for c in contexts:
        for k, arm in enumerate("BRRB"):
            out.append(
                dict(
                    name=f"{arm}{k + 1}-{c}",
                    tokens=tokens,
                    context=c,
                    slot=slot,
                    profile=False,
                    resident=arm == "R",
                )
            )
            slot += 1
    return out


def layer_breakdown(sequence, end_marker="moe_combine", tail=2):
    """The detail sequence ([name, ms] per dispatch) cut into layers (each ends `tail` dispatches after
    `end_marker`), then per layer shape (the tuple of names): each position's median ms and the layers' count."""
    import statistics

    ends = [
        i + tail for i, (name, _) in enumerate(sequence) if name == end_marker
    ]
    layers = []
    for k, end in enumerate(ends):
        begin = (ends[k - 1] + 1) if k else None
        if (
            begin is None
        ):  # the first layer starts after the embedding: as long as the next layer of its shape
            continue
        layers.append(sequence[begin : end + 1])
    if (
        ends
    ):  # the first layer: the same length as the second, ending at ends[0]
        n = len(layers[0]) if layers else 0
        if n and ends[0] + 1 - n >= 0:
            layers.insert(0, sequence[ends[0] + 1 - n : ends[0] + 1])
    shapes = {}
    for layer in layers:
        shapes.setdefault(tuple(n for n, _ in layer), []).append(
            [ms for _, ms in layer]
        )
    out = []
    for names, rows in shapes.items():
        out.append(
            dict(
                layers=len(rows),
                ops=[
                    [n, statistics.median(r[i] for r in rows)]
                    for i, n in enumerate(names)
                ],
                total_ms=statistics.median(sum(r) for r in rows),
            )
        )
    outside = sum(ms for _, ms in sequence) - sum(
        sum(r) for rows in shapes.values() for r in rows
    )
    return dict(shapes=out, outside_layers_ms=outside)


def bench(args):
    """DO-63: one stage of the whole model loaded once, decode variants in turn (BENCH_VARIANTS), no reference."""
    import statistics
    from tokenizers import Tokenizer

    model = load_model(args.gguf, args.config)
    prompt = Tokenizer.from_file(str(args.tokenizer)).encode(args.prompt).ids
    chunks = [
        min(args.chunk, len(prompt) - i)
        for i in range(0, len(prompt), args.chunk)
    ]
    # WS-33 `--layers N`: the first N layers with the embedding and head (host time per layer of models too large
    # for this GPU; the tokens are not the model's).
    last = (
        min(args.layers, model.config.layers)
        if getattr(args, "layers", None)
        else model.config.layers
    )
    m, files = stage_manifest(model, 0, last, head=last < model.config.layers)
    contexts = (
        [int(c) for c in args.contexts.split(",")] if args.contexts else None
    )
    variants = (
        (
            resident_variants(contexts)
            if args.resident
            else roofline_variants(contexts)
        )
        if contexts
        else BENCH_VARIANTS
    )
    if args.batches:  # DO-67: N conversations decoding together (runBatch), 16 steps each, no profiling
        variants = [
            dict(
                name=f"conv-{n}",
                tokens=16,
                batch=n,
                profile="detail" if args.batch_profile else False,
            )
            for n in (int(v) for v in args.batches.split(","))
        ]
        if args.batch_check:  # DO-69: each batch size against one conversation alone, after the timing runs
            variants += [
                dict(name=f"check-{n}", tokens=16, batch=n, check=True)
                for n in (int(v) for v in args.batches.split(","))
            ]
    decode = sum(v["tokens"] for v in variants)
    positions = (
        max(contexts) + 2 * decode + 8
        if contexts
        else len(prompt) + decode + 8
    )
    fixture = dict(
        tokens=prompt,
        chunks=chunks,
        kv=args.kv,
        slots=max(v.get("slot", 0) for v in variants) + 1
        if contexts
        else max(
            [8]
            + [
                v.get("batch", 1) + (1 if v.get("check") else 0)
                for v in variants
            ]
        ),
        max_tokens=max(
            [max(chunks), 8] + [v.get("batch", 1) for v in variants]
        ),
        max_positions=positions,
        variants=variants,
        merge_experts=not args.no_merge_experts,
        host_cache=not args.no_host_cache,
        fuse=not args.no_fuse,
        copy_mb=args.copy_mb if contexts and not args.resident else 0,
        **{
            k: v
            for k, v in kernel_options(args).items()
            if k
            in (
                "matvec",
                "matvec_c",
                "matvec_b",
                "group_matmul",
                "norm_fold",
                "repack_q4k",
                "repack",
                "no_wgsl_memo",
                "attention_split",
                "gemm",
                "q8_fast",
                "q8_fast_up_to",
            )
        },
        batch_kernel=args.batch_kernel,
    )
    files["manifest.json"] = json.dumps(
        dict(mode="bench", fixture=fixture, stages=[m])
    ).encode()
    result = run_page(
        "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    rows = []
    for v in result["variants"]:
        if v.get("check"):  # DO-69: batch rows against one conversation alone
            rows.append(
                dict(
                    name=v["name"],
                    batch=v["batch"],
                    check=True,
                    tokens_equal=v["tokens_equal"],
                    logits_max_abs=v["logits_max_abs"],
                    logits_scale=v["logits_scale"],
                )
            )
            print(
                f"{v['name']:12} batch {v['batch']}: rows with the lone conversation's tokens {sum(v['tokens_equal'])}/{v['batch']}, "
                f"last logits max |diff| {max(v['logits_max_abs']):.3g} (largest {v['logits_scale']:.3g})"
            )
            continue
        steps = (
            v["steps"][1:] or v["steps"]
        )  # the first step of a variant may compile new kernels
        recs = [st["profile"] for st in steps]
        enc = [sum(r["encodeMs"] for r in rec) for rec in recs]
        expand = [sum(r.get("expandMs") or 0 for r in rec) for rec in recs]
        gpu = [
            sum(r["gpuMs"] for r in rec)
            if rec and all(r["gpuMs"] is not None for r in rec)
            else None
            for rec in recs
        ]
        wall = statistics.median(st["ms"] for st in steps)
        row = dict(
            name=v["name"],
            batch=v["batch"],
            context=v.get("context"),
            tokens=v.get("tokens"),
            wall_ms=wall,
            tokens_per_s=v["batch"] * 1000 / wall,
            encode_ms=statistics.median(enc),
            expand_ms=statistics.median(expand),
            gpu_ms=None if None in gpu else statistics.median(gpu),
            dispatches=statistics.median(
                sum(r["dispatches"] for r in rec) for rec in recs
            ),
        )
        if all(
            st.get("host") for st in steps
        ):  # DO-73: host ms per step in submit, onSubmittedWorkDone, mapAsync
            row["host"] = {
                k: statistics.median(st["host"][k] for st in steps)
                for k in ("submit", "wait", "map", "submits", "maps")
            }
        if v.get("sequence"):
            names = {}
            for rec in recs:
                for r in rec:
                    for k, ms in (r.get("gpuByName") or {}).items():
                        names.setdefault(k, []).append(ms)
            row["gpu_by_name"] = sorted(
                ((k, sum(x) / len(recs)) for k, x in names.items()),
                key=lambda kv: -kv[1],
            )
            sizes = {}
            for rec in recs:
                for r in rec:
                    for k, b in (r.get("bytesByName") or {}).items():
                        sizes[k] = sizes.get(k, 0) + b / len(recs)
            if sizes:  # WS-31: bound bytes (an upper bound on traffic: caches are bound whole) per op and GB/s
                roof = (result.get("copy") or {}).get("median")
                row["ops"] = [
                    dict(
                        op=k,
                        ms=ms,
                        bound_mb=sizes.get(k, 0) / 2**20,
                        gbs=sizes.get(k, 0) / ms / 1e6 if ms else None,
                        share=(sizes.get(k, 0) / ms / 1e6 / roof)
                        if ms and roof
                        else None,
                    )
                    for k, ms in row["gpu_by_name"]
                ]
            row["layers"] = layer_breakdown(v["sequence"])
        rows.append(row)
    if (
        args.resident
    ):  # WS-32: every variant of a context must give the same tokens
        for c in contexts:
            seqs = [r["tokens"] for r in rows if r["context"] == c]
            n = min(map(len, seqs))
            for r in rows:
                if r["context"] == c:
                    r["tokens_equal"] = (
                        r["tokens"][:n] == seqs[0][:n] and n > 0
                    )
    report = dict(
        mode="bench",
        adapter=result["adapter"],
        load_ms=result["loadMs"],
        allocated=result["allocated"],
        peak=result["peak"],
        copy=result.get("copy"),
        contexts=contexts,
        resident=bool(args.resident),
        prompt_tokens=len(prompt),
        merge_experts=fixture["merge_experts"],
        variants=rows,
    )
    for r in rows:
        if r.get("check"):
            continue
        g = "-" if r["gpu_ms"] is None else f"{r['gpu_ms']:.1f}"
        if args.resident:
            print(
                f"{r['name']:12s}: {r['wall_ms']:7.2f} ms/token  {r['tokens_per_s']:6.1f} tok/s  tokens equal {r.get('tokens_equal')}"
            )
            continue
        print(
            f"{r['name']:12s} batch {r['batch']}: wall {r['wall_ms']:7.1f} ms/step  {r['tokens_per_s']:6.1f} tok/s  "
            f"expand {r['expand_ms']:5.1f} ms  encode {r['encode_ms']:6.1f} ms  gpu {g:>6s} ms  {r['dispatches']:.0f} dispatches"
        )
        if r.get("host"):
            h = r["host"]
            print(
                f"   host: submit {h['submit']:.2f} ms ({h['submits']:.0f}), wait {h['wait']:.2f} ms, map {h['map']:.2f} ms ({h['maps']:.0f})"
            )
        if r.get("gpu_by_name"):
            print(
                "   gpu by op: "
                + ", ".join(f"{k} {ms:.2f}" for k, ms in r["gpu_by_name"][:12])
            )
            for o in r.get("ops", [])[:12]:
                share = (
                    ""
                    if o["share"] is None
                    else f" ({100 * o['share']:.0f}% of copy)"
                )
                print(
                    f"     {o['op']:24s} {o['ms']:7.2f} ms  {o['bound_mb']:9.1f} MB bound  {o['gbs'] or 0:7.1f} GB/s{share}"
                )
            for sh in r["layers"]["shapes"]:
                print(
                    f"   layer shape x{sh['layers']} ({sh['total_ms']:.2f} ms): "
                    + ", ".join(f"{n} {ms:.3f}" for n, ms in sh["ops"])
                )
            print(
                f"   outside layers: {r['layers']['outside_layers_ms']:.2f} ms"
            )
    write(args, report)


def reference_logits(
    model, config, tokens, positions, splits, kv="f32", wire="f32"
):
    """Transformers' layers, one stage at a time (each loaded once), over the
    whole token sequence in one causal pass; then the final norm and the
    output projection at `positions`. Returns float64 logits (positions,
    vocab) and the hidden states leaving each stage (before the wire's
    rounding)."""
    import gc

    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    x = q.embed(model, tokens)
    boundaries = []
    for first, last in zip(splits, splits[1:]):
        print(f"reference: layers {first}..{last - 1}", flush=True)
        reference = q.StageReference(model, config, first, last, kv=kv)
        x = reference(x)
        boundaries.append(x[positions])
        if wire == "f16" and last < q.LAYERS:
            x = x.astype(np.float16).astype(np.float32)
        del reference
        gc.collect()
    norm = Qwen3_5RMSNorm(model.config.hidden, eps=config.rms_norm_eps)
    norm.load_state_dict(
        {
            "weight": torch.from_numpy(
                model["output_norm.weight"].dequantize() - 1
            )
        }
    )
    with torch.no_grad():
        h = norm(torch.from_numpy(x[positions])).numpy().astype(np.float64)
    W = model["output.weight"]
    logits = np.empty((len(positions), W.shape[0]))
    step = 16384
    for r in range(0, W.shape[0], step):
        rows = np.arange(r, min(r + step, W.shape[0]))
        # float32 products as the model would compute them; float64 sum
        logits[:, rows] = h @ W.dequantize(rows=rows).astype(np.float64).T
    return logits, boundaries


def numpy_reference(model):
    """The NumPy reference module of a family that has one (dense, kolibri1), else None (Qwen3.5: transformers)."""
    if model.family == "dense":
        from ygaraty.backends.webgpu.checkpoints import dense_gguf

        return dense_gguf
    if model.family == "kolibri1":
        from ygaraty.backends.webgpu.checkpoints import kolibri_gguf

        return kolibri_gguf
    if model.family == "qwen3moe":  # DO-80
        from ygaraty.backends.webgpu.checkpoints import qwen3moe_gguf

        return qwen3moe_gguf
    return None


def dense_reference_logits(
    model, tokens, positions, splits, kv="f32", wire="f32"
):
    """The dense or kolibri1 family's NumPy reference (`dense_gguf`, `kolibri_gguf`), as `reference_logits`: one
    causal pass per stage, the wire's rounding at each boundary, then the final norm and head at `positions`."""
    d = numpy_reference(model)

    x = d.embed(model, tokens)
    boundaries = []
    for first, last in zip(splits, splits[1:]):
        print(f"reference: layers {first}..{last - 1}", flush=True)
        x = d.run_layers(model, x, first, last, kv=kv)
        boundaries.append(x[positions])
        if wire == "f16" and last < model.config.layers:
            x = x.astype(np.float16).astype(np.float32)
    return d.logits(model, x[positions]), boundaries


def pipeline(args):
    from tokenizers import Tokenizer

    model = load_model(args.gguf, args.config)
    dense = numpy_reference(model) is not None
    config = None if dense else q.text_config(args.config)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    prompt = tokenizer.encode(args.prompt).ids
    splits = [int(v) for v in args.splits.split(",")]
    if splits[0] != 0 or splits[-1] != model.config.layers:
        raise SystemExit(f"splits must run from 0 to {model.config.layers}")
    chunk = args.chunk
    chunks = [
        min(chunk, len(prompt) - i) for i in range(0, len(prompt), chunk)
    ]
    files, stages = {}, []
    for first, last in zip(splits, splits[1:]):
        m, more = stage_manifest(model, first, last)
        files.update(more)
        stages.append(m)
    eos_id = (
        json.loads(Path(args.config).read_text()).get("eos_token_id")
        if dense
        else config.eos_token_id
    )
    eos = [eos_id] if isinstance(eos_id, int) else list(eos_id or [])
    fixture = dict(
        tokens=prompt,
        chunks=chunks,
        generate=args.generate,
        eos=eos,
        stop_at_eos=True,
        max_tokens=max(chunks),
        max_positions=len(prompt) + args.generate,
        kv=args.kv,
        wire=args.wire,
        profile=args.profile,
        merge_experts=not args.no_merge_experts,
        host_cache=not args.no_host_cache,
        fuse=not args.no_fuse,
    )
    files["manifest.json"] = json.dumps(
        dict(
            mode="pipeline",
            stages=stages,
            fixture=fixture,
            **kernel_options(args),
        )
    ).encode()
    print(
        f"browser: {len(prompt)} prompt tokens in chunks of {chunk}, up to {args.generate} generated, stages {splits}, "
        f"kv {args.kv}, wire {args.wire}",
        flush=True,
    )
    result = run_page(
        "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    generated = result["generated"]
    print("generated:", repr(tokenizer.decode(generated)), flush=True)

    sequence = prompt + generated[:-1]
    positions = list(range(len(prompt) - 1, len(sequence)))
    reference = (
        (
            lambda **kw: dense_reference_logits(
                model, sequence, positions, splits, **kw
            )
        )
        if dense
        else (
            lambda **kw: reference_logits(
                model, config, sequence, positions, splits, **kw
            )
        )
    )
    logits, _ = reference(kv=args.kv, wire=args.wire)
    out = Path(args.out)
    steps = []
    for i, ref in enumerate(logits):
        got = np.fromfile(out / f"logits{i}.f32", dtype=np.float32).astype(
            np.float64
        )
        top2 = np.argsort(ref)[-2:][::-1]
        steps.append(
            dict(
                token=generated[i],
                reference_token=int(top2[0]),
                match=generated[i] == int(top2[0]),
                reference_margin=float(ref[top2[0]] - ref[top2[1]]),
                max_abs=float(np.abs(got - ref).max()),
                max_rel=float(np.abs(got - ref).max() / np.abs(ref).max()),
            )
        )
        print(
            f"step {i:2}: token {generated[i]:6} ref {top2[0]:6} {'=' if steps[-1]['match'] else 'DIFFERS'}  "
            f"margin {steps[-1]['reference_margin']:.3f}  logits max |err| {steps[-1]['max_abs']:.2e} ({steps[-1]['max_rel']:.1e} of max)"
        )
    policy = None
    if args.compare_f32 and (args.kv, args.wire) != ("f32", "f32"):
        # The precision policy's own effect: the reference with and without it.
        logits32, _ = reference()
        policy = []
        for ref, ref32 in zip(logits, logits32):
            top = np.argsort(ref32)[-2:][::-1]
            policy.append(
                dict(
                    max_abs=float(np.abs(ref - ref32).max()),
                    same_token=int(np.argmax(ref)) == int(top[0]),
                    f32_margin=float(ref32[top[0]] - ref32[top[1]]),
                )
            )
        print(
            f"precision policy vs float32 reference: logits max |diff| {max(p['max_abs'] for p in policy):.2e}, "
            f"same greedy token at {sum(p['same_token'] for p in policy)}/{len(policy)} steps"
        )
    report = dict(
        mode="pipeline",
        adapter=result["adapter"],
        splits=splits,
        kv=args.kv,
        wire=args.wire,
        policy_vs_f32=policy,
        boundary_max_abs=result["boundaryMax"],
        prompt=args.prompt,
        prompt_tokens=prompt,
        generated=generated,
        text=tokenizer.decode(generated),
        chunk=chunk,
        load_ms=result["loadMs"],
        allocated=result["allocated"],
        peak=result["peak"],
        prefill=result["prefill"],
        decode=result["steps"],
        steps=steps,
        all_match=all(s["match"] for s in steps),
    )
    print(
        f"tokens identical to the reference's greedy choice: {report['all_match']} ({sum(s['match'] for s in steps)}/{len(steps)})"
    )
    decode = sorted(sum(d["stageMs"]) for d in result["steps"])
    if decode:
        print(
            f"decode: median {decode[len(decode) // 2]:.0f} ms per token; largest |hidden| at the boundaries {result['boundaryMax']}"
        )
    if args.profile and result["steps"]:
        report["profile"] = profile_summary(result["steps"])
        p = report["profile"]
        print(
            f"profile (decode median per token): plan expansion {p['expand_ms']:.1f} ms; submission wall {p['wall_ms']:.1f} ms = JavaScript encode {p['encode_ms']:.1f} ms "
            f"+ the rest; GPU pass {p['gpu_ms'] if p['gpu_ms'] is None else round(p['gpu_ms'], 1)} ms; "
            f"{p['dispatches']} dispatches in {p['submissions']} submissions; top: "
            + ", ".join(f"{k} {v}" for k, v in p["by_name"][:8])
        )
    write(args, report)


def stage(args):
    from tokenizers import Tokenizer

    model = load_model(args.gguf, args.config)
    dense = numpy_reference(model) is not None
    config = None if dense else q.text_config(args.config)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokens = tokenizer.encode(args.prompt).ids
    chunks = (
        [int(c) for c in args.chunks.split(",")]
        if args.chunks
        else [len(tokens) - 3, 1, 1, 1]
    )
    if sum(chunks) > len(tokens):
        raise SystemExit(
            f"chunks {chunks} exceed the prompt's {len(tokens)} tokens"
        )
    tokens = tokens[: sum(chunks)]
    manifest, files = stage_manifest(model, args.first, args.last)
    manifest["mode"] = "stage"
    d = numpy_reference(model)
    x = d.embed(model, tokens) if dense else q.embed(model, tokens)
    if args.first > 0 and dense:
        print(f"reference input: layers 0..{args.first - 1}", flush=True)
        x = d.run_layers(model, x, 0, args.first)
        files["hidden.f32"] = np.ascontiguousarray(
            x, dtype=np.float32
        ).tobytes()
    elif args.first > 0:
        # The stage's input: the reference's hidden states leaving layer first-1.
        import gc

        for first in range(0, args.first, 8):
            last = min(first + 8, args.first)
            print(f"reference input: layers {first}..{last - 1}", flush=True)
            before = q.StageReference(model, config, first, last)
            x = before(x)
            del before
            gc.collect()
        files["hidden.f32"] = np.ascontiguousarray(
            x, dtype=np.float32
        ).tobytes()
    manifest["fixture"] = dict(
        tokens=tokens,
        chunks=chunks,
        max_tokens=max(chunks),
        max_positions=sum(chunks),
        hidden="hidden.f32" if args.first > 0 else None,
        kv=args.kv,
    )
    capture = bool(getattr(args, "capture_routes", False))
    files["manifest.json"] = json.dumps(
        dict(
            manifest,
            **kernel_options(args),
            **({"capture_routes": True} if capture else {}),
        )
    ).encode()

    print(
        f"reference: layers {args.first}..{args.last - 1}, {len(tokens)} tokens, chunks {chunks}",
        flush=True,
    )
    reference = (
        d.StageReference(model, args.first, args.last, kv=args.kv)
        if dense
        else q.StageReference(model, config, args.first, args.last, kv=args.kv)
    )
    expected, at = [], 0
    for n in chunks:
        expected.append(reference(x[at : at + n]))
        at += n
    states = {} if dense else reference.recurrent_states()
    # Kolibri-1: how near each call's expert choice came to changing (the k-th minus the next biased router
    # score); a near-tie can choose a different expert on the GPU and move that row by an expert's output.
    routing = {
        i: [float(np.min(margins)) for _, _, margins in calls]
        for i, calls in getattr(reference, "routes", {}).items()
    }
    del reference

    print("browser ...", flush=True)
    result = run_page(
        "qwen35_parity.mjs", files, args.out, MODULES, timeout=args.timeout
    )
    if result.get("error"):
        raise SystemExit(result["error"])
    out = Path(args.out)
    report = dict(
        mode="stage",
        adapter=result["adapter"],
        layers=[args.first, args.last],
        kv=args.kv,
        tokens=tokens,
        chunks=chunks,
        load_ms=result["loadMs"],
        runs=result["runs"],
        allocated=result["allocated"],
        peak=result["peak"],
        calls=[],
        routing_margin_min=routing or None,
    )
    if routing:
        print(
            "smallest routing margin per layer and call: "
            + "; ".join(
                f"{i}: {', '.join(f'{m:.2e}' for m in margins)}"
                for i, margins in routing.items()
            )
        )
    for i, (n, ref) in enumerate(zip(chunks, expected)):
        got = np.fromfile(out / f"out{i}.f32", dtype=np.float32).reshape(
            n, model.config.hidden
        )
        err = np.abs(got.astype(np.float64) - ref)
        scale = np.abs(ref).max(axis=1)
        rel = (err.max(axis=1) / scale).max()
        cos = float(
            np.min(
                np.sum(got * ref, axis=1)
                / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
            )
        )
        report["calls"].append(
            dict(
                tokens=n,
                max_abs=float(err.max()),
                max_rel_row=float(rel),
                min_cosine=cos,
            )
        )
        print(
            f"call {i} ({n} tokens): max |err| {err.max():.3e}, per-row max |err| / max |ref| {rel:.3e}, min cosine {cos:.8f}"
        )
    if (
        capture and model.family == "qwen3moe"
    ):  # DO-80: the reference again, following the GPU's expert choices
        k, layers = model.config.top_k, list(range(args.first, args.last))
        forced_ref = d.StageReference(model, args.first, args.last, kv=args.kv)
        at, report["routes"] = 0, []
        for i, n in enumerate(chunks):
            ids = (
                np.fromfile(out / f"routes{i}.u32", dtype="<u4")
                .reshape(len(layers), n, k)
                .astype(np.int64)
            )
            forced_ref.forced = {L: ids[j] for j, L in enumerate(layers)}
            ref = forced_ref(x[at : at + n]).astype(np.float64)
            got = np.fromfile(out / f"out{i}.f32", dtype=np.float32).reshape(
                n, model.config.hidden
            )
            rel = float(
                (np.abs(got - ref).max(axis=1) / np.abs(ref).max(axis=1)).max()
            )
            flips = [
                int(
                    (
                        np.sort(ids[j], 1)
                        != np.sort(forced_ref.routes[L][-1][0], 1)
                    )
                    .any(1)
                    .sum()
                )
                for j, L in enumerate(layers)
            ]
            report["routes"].append(
                dict(
                    tokens=n,
                    max_rel_row_forced=rel,
                    own_choice_differs_rows=dict(zip(layers, flips)),
                )
            )
            print(
                f"call {i} ({n} tokens), reference following the GPU's experts: per-row max |err| / max |ref| {rel:.3e}; "
                f"rows where the reference's own choice differs: {sum(flips)} in {sum(1 for f in flips if f)} layers"
            )
            at += n
    report["states"] = {}
    for index, (conv, state) in states.items():
        got = np.fromfile(out / f"S{index}.f32", dtype=np.float32).reshape(
            state.shape
        )
        ref = state
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        report["states"][index] = dict(state_rel=err)
        print(
            f"layer {index} recurrent state: max |err| / max |ref| {err:.3e}"
        )
    write(args, report)


def contributors(args):
    import hashlib

    from tokenizers import Tokenizer

    model = load_model(args.gguf, args.config)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text = Path(args.text).read_bytes()
    tokens = tokenizer.encode(text.decode()).ids
    total = args.prompt_tokens + args.generate
    if len(tokens) < total:
        raise SystemExit(f"{args.text} has {len(tokens)} tokens; need {total}")
    tokens = tokens[:total]
    splits = [int(v) for v in args.splits.split(",")]
    prefill = [
        min(args.chunk, args.prompt_tokens - i)
        for i in range(0, args.prompt_tokens, args.chunk)
    ]
    chunks = prefill + [1] * args.generate
    base = Path(args.out)
    report = dict(
        mode="contributors",
        splits=splits,
        kv=args.kv,
        wire=args.wire,
        chunk=args.chunk,
        prompt_tokens=args.prompt_tokens,
        generate=args.generate,
        capacity=total,
        text=str(args.text),
        text_sha256=hashlib.sha256(text).hexdigest(),
        stages=[],
    )
    previous = None
    for k, (first, last) in enumerate(zip(splits, splits[1:])):
        manifest, files = stage_manifest(model, first, last)
        manifest["mode"] = "contributor"
        manifest["fixture"] = dict(
            tokens=tokens if first == 0 else None,
            chunks=chunks,
            prefill_calls=len(prefill),
            max_tokens=max(chunks),
            max_positions=total,
            kv=args.kv,
            wire=args.wire,
        )
        if previous is not None:
            files.update(
                {f"in{i}": previous / f"out{i}" for i in range(len(chunks))}
            )
        files["manifest.json"] = json.dumps(
            dict(manifest, **kernel_options(args))
        ).encode()
        out = base / f"stage{k}"
        print(
            f"contributor {k}: layers {first}..{last - 1}, {args.prompt_tokens} prompt tokens in chunks of {args.chunk}, "
            f"{args.generate} steps, kv {args.kv}, wire {args.wire}",
            flush=True,
        )
        result = run_page(
            "qwen35_parity.mjs",
            files,
            out,
            MODULES,
            timeout=args.timeout,
            memory=True,
        )
        if result.get("error"):
            raise SystemExit(result["error"])
        calls = result["calls"]
        prefill_ms = sum(c["ms"] for c in calls[: len(prefill)])
        steps = sorted(c["ms"] for c in calls[len(prefill) :])
        mem = result["memory"]
        stage_report = dict(
            layers=[first, last],
            adapter=result["adapter"],
            load_ms=result["loadMs"],
            allocated=result["allocated"],
            allocated_peak=result["peak"],
            prefill_ms=prefill_ms,
            first_chunk_ms=calls[0]["ms"],
            last_chunk_ms=calls[len(prefill) - 1]["ms"],
            step_ms_median=steps[len(steps) // 2] if steps else None,
            step_ms_p95=steps[int(0.95 * (len(steps) - 1))] if steps else None,
            tokens=result["tokens"],
            memory=mem,
            calls=calls,
        )
        report["stages"].append(stage_report)
        peak = mem["peak"]
        gb = lambda v: f"{(v or 0) / 1e9:.2f} GB"  # noqa: E731
        print(
            f"  app allocations {gb(result['allocated'])} (peak {gb(result['peak'])}); driver: GPU process "
            f"vram {gb(peak.get('drm.gpu-process:drm-total-vram'))} + gtt {gb(peak.get('drm.gpu-process:drm-total-gtt'))}, "
            f"all clients {gb(peak.get('drm.drm-total-vram', 0) + peak.get('drm.drm-total-gtt', 0))} (peaks); "
            f"PSS total {gb(peak.get('pss.total'))}, GPU process {gb(peak.get('pss.gpu-process'))}, renderer {gb(peak.get('pss.renderer'))}",
            flush=True,
        )
        print(
            f"  load {result['loadMs'] / 1e3:.1f} s, prefill {prefill_ms / 1e3:.1f} s "
            f"(first chunk {calls[0]['ms']:.0f} ms, last {calls[len(prefill) - 1]['ms']:.0f} ms), "
            f"step median {stage_report['step_ms_median']:.1f} ms p95 {stage_report['step_ms_p95']:.1f} ms",
            flush=True,
        )
        previous = out
    last_tokens = report["stages"][-1]["tokens"]
    report["greedy_after_prompt"] = tokenizer.decode(
        last_tokens[len(prefill) - 1 :]
    )
    report["text_continuation"] = tokenizer.decode(
        tokens[args.prompt_tokens :]
    )
    write(args, report)


def write(args, report):
    report.update(
        date=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=commit(),
        host=platform.node(),
        gguf=str(args.gguf),
    )
    path = Path(args.out) / "report.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=(
            "inspect",
            "kernels",
            "f16",
            "capacity",
            "stage",
            "pipeline",
            "contributors",
            "bench",
        ),
    )
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=8)
    parser.add_argument("--chunks", default=None)
    parser.add_argument(
        "--prompt",
        default="The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is",
    )
    parser.add_argument(
        "--splits",
        default="0,8,16,24,32",
        help="pipeline: layer boundaries of the stages",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="pipeline: prefill tokens per call",
    )
    parser.add_argument(
        "--generate", type=int, default=16, help="pipeline: greedy tokens"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="kernels: dispatches timed per case",
    )
    parser.add_argument(
        "--attention-split",
        type=int,
        help="WS-35: keys per attention split in decode steps (dense and "
        "kolibri1 stages; e.g. 64; default the prefill's 512)",
    )
    parser.add_argument(
        "--matvec-c",
        help="WS-34 kernels: TPR,L[,WG] = the coalesced one-token kernel (threads per row, threads per item, "
        "threads per workgroup), then optional ;KIND=TPR,L[,WG] per format (DO-74)",
    )
    parser.add_argument(
        "--group-matmul",
        action="store_true",
        help="WS-36: consecutive one-token products of the same "
        "input in one dispatch (gate/up with silu_mul fused); needs --matvec-c",
    )
    parser.add_argument(
        "--roof-mb",
        type=float,
        help="kernels, DO-78: time a plain read (A), matvecC's access pattern "
        "without decoding (B) and matvecC (C) over enough copies of each matrix to exceed this many MiB",
    )
    parser.add_argument(
        "--roof-dispatches",
        type=int,
        default=30,
        help="kernels --roof-mb: dispatches per timing",
    )
    parser.add_argument(
        "--no-wgsl-memo",
        action="store_true",
        help="WS-33: regenerate every shader source per dispatch "
        "(the arm without the cache)",
    )
    parser.add_argument(
        "--capture-routes",
        action="store_true",
        help="stage (qwen3moe), DO-80: capture the GPU's "
        "expert ids per layer and also compare with a reference that follows them",
    )
    parser.add_argument(
        "--layers",
        type=int,
        help="bench, WS-33: only the first N layers (with the embedding and "
        "head): host time per layer of a model too large for this GPU",
    )
    parser.add_argument(
        "--calibrate-repack",
        action="store_true",
        help="kernels, DO-83: run the contributor's "
        "load-time repack measurement on this GPU and report it",
    )
    parser.add_argument(
        "--repack",
        help="DO-80: comma-separated K-quant formats to repack exactly at load (Q4_K,Q5_K,Q6_K)",
    )
    parser.add_argument(
        "--repack-q4k",
        action="store_true",
        help="WS-40: repack layer Q4_K matrices at load into Q4_KR "
        "(scales unpacked to float32; exact, 4/3 the bytes)",
    )
    parser.add_argument(
        "--norm-fold",
        action="store_true",
        help="WS-49: fold a one-token RMS norm into the grouped "
        "product that reads it (needs --group-matmul)",
    )
    parser.add_argument(
        "--sweep-matvec-c",
        help="kernels, DO-74: also time these coalesced shapes on each one-token "
        'case ("all", or TPR,L[,WG] separated by /)',
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="kernels --sweep-matvec-c: timing rounds (median)",
    )
    parser.add_argument(
        "--cases",
        choices=("default", "decode", "decode1", "q8"),
        default="default",
        help="kernels: decode = WS-15's ffn_up/ffn_down/attn_qkv at 1-16 tokens",
    )
    parser.add_argument(
        "--matvec",
        help="WS-15: R,VT = the matrix-vector kernel with R rows per thread (1, 2, 4) "
        "and up to VT tokens (4, 8) per dispatch row",
    )
    parser.add_argument(
        "--kv",
        choices=("f16", "f32"),
        default="f16",
        help="attention KV cache precision",
    )
    parser.add_argument(
        "--wire",
        choices=("f16", "f32"),
        default="f16",
        help="hidden states between stages",
    )
    parser.add_argument(
        "--compare-f32",
        action="store_true",
        help="pipeline: also the float32 reference (the policy's effect)",
    )
    parser.add_argument(
        "--text",
        type=Path,
        help="contributors: text whose tokens are the prompt",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=8192,
        help="contributors: prompt length",
    )
    parser.add_argument(
        "--prefill-kernel",
        choices=("tiled", "gemm", "matvec"),
        help="force the many-token matmul kernel "
        "(matvec: the matrix-vector kernel in groups of 4 tokens, as DO-38's batched decode)",
    )
    parser.add_argument(
        "--gemm",
        help="gemm tiling GR,GT,RR,RT[,v] (default 64,64,8,8; v: vec4 staging, WS-50)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="pipeline: per decode token, JavaScript encode time, GPU "
        "pass time (timestamp queries), dispatches by name",
    )
    parser.add_argument(
        "--no-merge-experts",
        action="store_true",
        help="pipeline (Kolibri): keep each layer's expert "
        "chunks in separate buffers (before fusion step 1)",
    )
    parser.add_argument(
        "--no-host-cache",
        action="store_true",
        help="pipeline, bench: check every plan task's contract "
        "and create every bind group each token (before DO-65)",
    )
    parser.add_argument(
        "--contexts",
        help="bench, WS-31: comma-separated contexts (e.g. 64,1024,4096): each in a fresh "
        "slot, profiled and detail decode, then base again; GB/s per op against the copy bandwidth",
    )
    parser.add_argument(
        "--matvec-b",
        type=int,
        help="DO-69: decode batches through the coalesced kernel (needs --matvec-c), tokens per pass 2, 4 or 8",
    )
    parser.add_argument(
        "--batch-check",
        action="store_true",
        help="bench --batches, DO-69: also check each batch size's tokens and logits against one conversation alone",
    )
    parser.add_argument(
        "--q8-fast-up-to",
        type=int,
        help="WS-47: Q8_0 through the vectorized decoders for calls of at most N tokens",
    )
    parser.add_argument(
        "--q8-fast",
        action="store_true",
        help="WS-46 (experiment 130, opt-in): Q8_0 matrices through the vectorized decoders",
    )
    parser.add_argument(
        "--batch-kernel",
        choices=("tiled", "gemm"),
        help="bench --batches, WS-45: decode batches through this many-token kernel (--gemm sets its tile)",
    )
    parser.add_argument(
        "--batch-profile",
        action="store_true",
        help="bench --batches, WS-44: GPU time per dispatch (detail profile) of each batch size's steps",
    )
    parser.add_argument(
        "--batches",
        help="bench, DO-67: comma-separated conversation counts decoded together, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--resident",
        action="store_true",
        help="bench --contexts, WS-32: B R R B per context, the usual "
        "loop against the GPU-resident one (stages with KV caches only: Kolibri, dense)",
    )
    parser.add_argument(
        "--copy-mb",
        type=int,
        default=256,
        help="bench --contexts: the copy probe's buffer, MiB",
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="pipeline, bench: keep each norm and its residual add as "
        "two dispatches (before DO-65's fusion)",
    )
    parser.add_argument(
        "--gb",
        type=float,
        default=8,
        help="capacity: GiB to allocate and fill",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=1024,
        help="capacity: buffer size (capped by the device's limits)",
    )
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    if args.mode not in ("f16", "capacity") and not args.gguf:
        parser.error("This mode requires --gguf")
    needs_config = args.mode in (
        "inspect",
        "stage",
        "pipeline",
        "contributors",
        "bench",
    )
    if needs_config and not args.config:
        parser.error("This mode requires --config")
    if (
        args.mode in ("stage", "pipeline", "contributors", "bench")
        and not args.tokenizer
    ):
        parser.error("This mode requires --tokenizer")
    if args.chunk < 1 or args.generate < 1:
        parser.error("--chunk and --generate must be positive")
    dict(
        inspect=inspect,
        kernels=kernels,
        f16=f16,
        capacity=capacity,
        bench=bench,
        stage=stage,
        pipeline=pipeline,
        contributors=contributors,
    )[args.mode](args)


if __name__ == "__main__":
    main()
