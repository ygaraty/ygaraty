"""Qwen3.5 WebGPU allocation estimates from prepared stage manifests.

Mirrors Stage.load in browser/qwen35_gpu.mjs, plus resident snapshots and
20% + 256 MiB headroom. This is a planning estimate, not a measured peak.
"""

import math

from .distribution import gate_stride

GIB = 1024**3
HEAD, LINEAR_HEAD = 256, 128  # attention / DeltaNet head widths


def qwen_geometry(config):
    """Head counts and derived widths; defaults are the 9B's (bundles
    prepared before these fields existed carry none of them)."""
    heads = config.get("attention_heads", 16)
    kv_heads = config.get("attention_kv_heads", 4)
    linear_k = config.get("linear_k_heads", 16)
    linear_v = config.get("linear_v_heads", 32)
    value = linear_v * LINEAR_HEAD
    return dict(
        heads=heads,
        kv_heads=kv_heads,
        linear_v=linear_v,
        conv=2 * linear_k * LINEAR_HEAD + value,
        value=value,
        attention=heads * HEAD,
        kv=kv_heads * HEAD,
    )


def estimate_memory(data, context=None, slots=1):
    if not 1 <= slots <= 16:
        raise ValueError("Require slots 1–16")
    reports = []
    for stage in data["stages"]:
        m, f = stage["manifest"], stage["fixture"]
        config = m.get("config", {})
        if (
            config.get("family", "qwen35") != "qwen35"
            or config.get("layers", 32) % 4
        ) or any(
            layer.get("type") not in {"linear", "full"}
            for layer in m["layers"]
        ):
            raise ValueError(
                "Automatic memory estimation requires a Qwen3.5 bundle"
            )
        if f.get("repack") or f.get("draft"):
            raise ValueError(
                "Automatic memory estimation does not support "
                "repack/draft bundles"
            )
        capacity = context if context is not None else f["max_positions"]
        if not f["max_tokens"] <= capacity <= f["max_positions"]:
            raise ValueError(
                "Context must fit prepared capacity and be at least "
                "the prefill chunk size"
            )
        tokens = f["max_tokens"]
        hidden, intermediate = config["hidden"], config["intermediate"]
        g = qwen_geometry(config)
        scratch_width = max(hidden, g["attention"], g["value"])

        def align(n):
            return max(16, (n + 15) // 16 * 16)

        def size(t):
            return align(data["files"][t["file"]]["length"])

        weights = sum(
            size(t) for layer in m["layers"] for t in layer["tensors"].values()
        )
        linear = [layer for layer in m["layers"] if layer["type"] == "linear"]
        full = len(m["layers"]) - len(linear)
        weights += sum(
            size(layer["tensors"]["ssm_conv1d.weight"])
            + align(2 * g["linear_v"] * 4)  # ssm_a and dt per head
            for layer in linear
        )
        if m.get("embed"):
            weights += sum(size(t) for t in m["embed"]["chunks"])
        head = m.get("head")
        if head:
            weights += size(head["norm"])
            if not (head.get("shared_embedding") and m.get("embed")):
                weights += sum(size(t) for t in head["chunks"])
        recurrent = (
            len(linear)
            * slots
            * (
                3 * g["conv"] * 4
                + g["linear_v"] * LINEAR_HEAD * LINEAR_HEAD * 4
            )
        )
        state = recurrent + full * slots * 2 * align(
            capacity * g["kv"] * (2 if f["kv"] == "f16" else 4)
        )
        # Stage.load's activation buffers, in order: ids, x, h, big, big2,
        # mid, z, ab, k, kn, v, f1, f2.
        widths = [
            1,
            hidden,
            scratch_width,
            max(g["conv"], 2 * g["attention"]),
            g["conv"],
            scratch_width,
            g["value"],
            gate_stride(g["linear_v"]),
            g["kv"],
            g["kv"],
            g["kv"],
            intermediate,
            intermediate,
        ]
        scratch = 128 + sum(align(tokens * width * 4) for width in widths)
        if full:
            splits = math.ceil(capacity / 512)
            scratch += align(tokens * splits * g["attention"] * 4) + align(
                tokens * splits * g["heads"] * 2 * 4
            )
        if head:
            scratch += align(min(slots, tokens) * head["vocab"] * 4) + align(
                min(slots, tokens) * 256
            )
        allocated = weights + state + scratch
        # Resident decoding can snapshot each conversation’s recurrent state.
        estimated = allocated + recurrent
        recommended = math.ceil(estimated * 1.2) + 256 * 1024**2
        reports.append(
            dict(
                worker=len(reports) + 1,
                layers=stage["layers"],
                weights=weights,
                state=state,
                scratch=scratch,
                snapshots=recurrent,
                allocated=allocated,
                estimated=estimated,
                recommended=recommended,
                context=capacity,
            )
        )
    if not reports:
        raise ValueError("Bundle has no workers")
    return reports


def report_memory(
    data,
    workers,
    context=None,
    slots=1,
    memory_gb=None,
    *,
    verbose=True,
    reports_out=None,
):
    """Check the budget; print the estimate (verbose) and/or hand the rows
    to the caller (reports_out, a list extended in place)."""
    reports = estimate_memory(data, context, slots)
    if len(reports) != workers:
        raise ValueError("Bundle worker count does not match --num-workers")
    required = math.ceil(max(row["recommended"] for row in reports) / GIB)
    if memory_gb is not None and memory_gb < required:
        raise ValueError(
            "--memory-gb is below the calculated recommendation "
            f"({required} GiB)"
        )
    if reports_out is not None:
        reports_out[:] = reports
    if not verbose:
        return memory_gb or required
    print(
        f"Configuration: {workers} workers, "
        f"context {reports[0]['context']}, {slots} chat slot(s).",
        flush=True,
    )
    for row in reports:
        print(
            f"Worker {row['worker']}: weights "
            f"{row['weights'] / GIB:.2f} GiB; estimated GPU RAM "
            f"{row['estimated'] / GIB:.2f} GiB; recommended "
            f"{row['recommended'] / 1e9:.2f} GB "
            f"({row['recommended'] / GIB:.2f} GiB).",
            flush=True,
        )
    print(
        "Calculated from bundle tensors, KV/recurrent state, scratch and "
        "snapshots; recommendation adds 20% + 256 MiB headroom. Estimates "
        "are not measured peaks or server RAM requirements; device "
        "self-tests must pass.",
        flush=True,
    )
    total = sum(row["recommended"] for row in reports)
    print(
        f"Recommended GPU RAM across workers: {total / 1e9:.2f} GB "
        f"({total / GIB:.2f} GiB).",
        flush=True,
    )
    return memory_gb or required
