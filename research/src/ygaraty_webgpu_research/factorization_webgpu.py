"""Experiment 48 stage A: the factorization tile on WebGPU against NumPy, one
tile at a time.

    ygaraty experiment factorization_webgpu

Headless Chromium runs ygaraty/backends/webgpu/browser/factorization_gpu.mjs on each case: X is
generated on the GPU
from the tile's seeded spec (the latent rows travel, not X), in float32 with
compensated reductions.
The harness compares the packed output (dU, dV, loss) with
runtime/factorization_kernels.py in
float64, and the GPU's 64-bit noise hash with
models/factorization_sharded.noise_words bit for bit.

Gates, fixed before the first run: every hash word equal; per case, the largest
error in dU and
in dV at most 1e-5 of that block's largest reference magnitude, and the loss
within 1e-6 relative.
GPU_FLAGS (the default --chrome-arg list) give headless Chromium the machine's
GPU on Linux (Vulkan;
checked on an AMD Radeon 680M); without them it falls back to SwiftShader
(software), which the
report records and the run refuses unless --allow-software. Timings: gpu_ms is
the three passes and
the readback, X's generation included; upload_ms is pipeline creation and the
input upload. The
NumPy time is the kernel with X given (not generated) and multi-threaded BLAS:
not a like-for-like
comparison.
"""

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.host import component as _host_component

SDK_ASSETS = _host_component("browser_assets").SDK_ASSETS
from ygaraty_webgpu_research.webgpu_parity import run_browser
from ygaraty.models.factorization_sharded import (
    factor_rows,
    hashed_noise,
    noise_words,
    x_values,
)
from ygaraty.runtime.factorization_kernels import execute

GRADIENT_TOLERANCE = 1e-5
LOSS_TOLERANCE = 1e-6
GPU_FLAGS = (
    "--enable-features=Vulkan",
    "--use-webgpu-adapter=default",
    "--ignore-gpu-blocklist",
    "--enable-gpu",
    "--use-gl=angle",
)

ASSETS = (
    "factorization_stage_a.mjs",
    "factorization_gpu.mjs",
    "sdk.mjs",
    "budget.mjs",
    "convolution.mjs",
) + SDK_ASSETS

# name: tile rows and columns, rank, latent, noise, the tile's offsets, the
# whole matrix's size.
CASES = {
    "exp46-tile": dict(
        shape=[2048, 2048],
        rank=64,
        latent=8,
        noise=0.05,
        row_offset=2048,
        column_offset=4096,
        rows=122880,
        columns=122880,
    ),
    "offset-1e6": dict(
        shape=[512, 512],
        rank=64,
        latent=8,
        noise=0.05,
        row_offset=999424,
        column_offset=998912,
        rows=1000000,
        columns=1000000,
    ),
    "ragged": dict(
        shape=[1000, 300],
        rank=1,
        latent=1,
        noise=0.0,
        row_offset=7,
        column_offset=13,
        rows=3000,
        columns=900,
    ),
    "large-tile": dict(
        shape=[4096, 4096],
        rank=64,
        latent=8,
        noise=0.05,
        row_offset=0,
        column_offset=4096,
        rows=122880,
        columns=122880,
    ),
}


def tile_x(spec, left, right):
    """x_values for the whole tile, from its own latent rows: the same
    expression, without
    generating the latent factor for the whole matrix."""
    values = left[:, :1] * right[:, 0][None, :]
    for k in range(1, spec["latent"]):
        values += left[:, k : k + 1] * right[:, k][None, :]
    values /= math.sqrt(spec["latent"])
    if spec["noise"]:
        rows, cols = spec["shape"]
        values = values + spec["noise"] * hashed_noise(
            spec["seed"],
            spec["row_offset"],
            spec["row_offset"] + rows,
            spec["column_offset"],
            spec["column_offset"] + cols,
            spec["columns"],
        )
    return values


def make_case(name, seed):
    c = CASES[name]
    rows, cols = c["shape"]
    spec = dict(
        kind="lowrank_x",
        shape=c["shape"],
        row_offset=c["row_offset"],
        column_offset=c["column_offset"],
        rows=c["rows"],
        columns=c["columns"],
        latent=c["latent"],
        noise=c["noise"],
        seed=seed,
    )
    left = factor_rows(
        seed,
        "X-left",
        c["row_offset"],
        c["row_offset"] + rows,
        c["latent"],
        1.0,
    )
    right = factor_rows(
        seed,
        "X-right",
        c["column_offset"],
        c["column_offset"] + cols,
        c["latent"],
        1.0,
    )
    u = factor_rows(
        seed, "U", c["row_offset"], c["row_offset"] + rows, c["rank"], 0.1
    )
    v = factor_rows(
        seed,
        "V",
        c["column_offset"],
        c["column_offset"] + cols,
        c["rank"],
        0.1,
    )
    attrs = dict(x_spec=spec, total_elements=c["rows"] * c["columns"])
    started = time.perf_counter()
    reference = execute(
        "factorization_tile",
        [tile_x(spec, left, right), u, v],
        dict(total_elements=attrs["total_elements"]),
    )
    cpu_seconds = time.perf_counter() - started
    case = dict(
        name=name,
        attrs=attrs,
        shape=[rows + cols + 1, c["rank"]],
        **{
            k: dict(shape=list(a.shape), data=a.ravel().tolist())
            for k, a in (("L", left), ("R", right), ("U", u), ("V", v))
        },
    )
    return case, reference, cpu_seconds


def make_probes(seed):
    """Entries whose hash words the GPU must reproduce: corners, word
    boundaries, random."""
    rng = np.random.default_rng(seed)
    probes = []
    for columns in (122880, 1000000, 4294967295):
        pairs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [columns - 1 if columns < 2**32 else 0, columns - 1],
            [65535, 65536],
            [999999, 999999],
            *rng.integers(0, min(columns, 2**31), size=(58, 2)).tolist(),
        ]
        for s in (
            seed,
            0,
            (1 << 53) - 1,
        ):  # JSON numbers are doubles in a browser: seeds stay below 2^53
            probes.append(
                dict(
                    columns=columns,
                    seed=s,
                    pairs=[[int(r), int(c)] for r, c in pairs],
                )
            )
    return probes


def compare(case, reference, output):
    rows, cols = case["attrs"]["x_spec"]["shape"]
    got = np.asarray(output, dtype=np.float64).reshape(case["shape"])
    out = {}
    for block, sl in (
        ("dU", slice(0, rows)),
        ("dV", slice(rows, rows + cols)),
    ):
        ref, error = reference[sl], np.abs(got[sl] - reference[sl])
        scale = float(np.abs(ref).max())
        out[block] = dict(
            max_error=float(error.max()),
            scale=scale,
            relative=float(error.max() / scale),
        )
    loss, ref_loss = (
        float(got[rows + cols, 0]),
        float(reference[rows + cols, 0]),
    )
    out["loss"] = dict(
        gpu=loss,
        reference=ref_loss,
        relative=abs(loss - ref_loss) / abs(ref_loss),
    )
    out["passed"] = (
        out["dU"]["relative"] <= GRADIENT_TOLERANCE
        and out["dV"]["relative"] <= GRADIENT_TOLERANCE
        and out["loss"]["relative"] <= LOSS_TOLERANCE
        and bool(np.all(got[rows + cols, 1:] == 0))
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cases", nargs="+", choices=sorted(CASES), default=list(CASES)
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="GPU runs per case; timings of every run are kept",
    )
    parser.add_argument("--chrome")
    parser.add_argument(
        "--chrome-arg",
        action="append",
        help=f"Chromium flags (repeat); default: {' '.join(GPU_FLAGS)}",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--allow-software",
        action="store_true",
        help="Accept a software (SwiftShader) adapter: a smoke test only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.chrome_arg = (
        list(GPU_FLAGS) if args.chrome_arg is None else args.chrome_arg
    )
    output = args.output or Path(
        "runs/experiments/048/stage-a"
    ) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)

    # The fast path (tile_x) must equal the generated X the CPU workers use.
    check = CASES["ragged"]
    spec = dict(
        kind="lowrank_x",
        shape=check["shape"],
        row_offset=check["row_offset"],
        column_offset=check["column_offset"],
        rows=check["rows"],
        columns=check["columns"],
        latent=check["latent"],
        noise=0.05,
        seed=args.seed,
    )
    left = factor_rows(
        args.seed,
        "X-left",
        spec["row_offset"],
        spec["row_offset"] + spec["shape"][0],
        spec["latent"],
        1.0,
    )
    right = factor_rows(
        args.seed,
        "X-right",
        spec["column_offset"],
        spec["column_offset"] + spec["shape"][1],
        spec["latent"],
        1.0,
    )
    if not np.array_equal(tile_x(spec, left, right), x_values(spec)):
        raise AssertionError("tile_x differs from x_values")

    cases, references, cpu_seconds = [], {}, {}
    for name in args.cases:
        case, references[name], cpu_seconds[name] = make_case(name, args.seed)
        cases.append(case)
    probes = make_probes(args.seed)
    fixture = dict(cases=cases, probes=probes, repeats=args.repeats)
    browser = run_browser(
        fixture,
        output,
        args.chrome,
        args.timeout,
        args.chrome_arg,
        page="factorization_stage_a.mjs",
        assets=ASSETS,
    )

    adapter = browser["adapter"]
    software = (
        "swiftshader" in json.dumps(adapter).lower()
        or adapter.get("isFallbackAdapter") is True
    )
    hash_mismatches = 0
    for probe, got in zip(probes, browser["probes"]):
        pairs = np.asarray(probe["pairs"], dtype=np.uint64)
        words = noise_words(
            probe["seed"], pairs[:, 0], pairs[:, 1], probe["columns"]
        )
        expected = [
            [int(w & 0xFFFFFFFF), int(w >> 32)] for w in words.tolist()
        ]
        hash_mismatches += sum(a != b for a, b in zip(expected, got["words"]))
    results = {}
    for case, got in zip(cases, browser["cases"]):
        results[case["name"]] = dict(
            compare(case, references[case["name"]], got["output"]),
            tile=case["attrs"]["x_spec"]["shape"],
            rank=case["shape"][1],
            gpu_timings_ms=got["timings"],
            numpy_seconds=cpu_seconds[case["name"]],
        )
    passed = (
        hash_mismatches == 0
        and all(r["passed"] for r in results.values())
        and (not software or args.allow_software)
    )
    report = dict(
        experiment=48,
        stage="A",
        passed=passed,
        software_adapter=software,
        adapter=adapter,
        user_agent=browser["userAgent"],
        hash=dict(
            entries=sum(len(p["pairs"]) for p in probes),
            mismatches=hash_mismatches,
        ),
        cases=results,
        gates=dict(
            gradient_relative=GRADIENT_TOLERANCE,
            loss_relative=LOSS_TOLERANCE,
            hash="bit-exact",
        ),
        chrome_args=args.chrome_arg,
        seed=args.seed,
        machine=platform.platform(),
        revision=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        dirty=bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        date=datetime.now(UTC).isoformat(),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"adapter {adapter.get('vendor')} {adapter.get('architecture')}{' (SOFTWARE)' if software else ''}; "  # noqa: E501 -- preserve literal/report layout
        f"hash {report['hash']['entries'] - hash_mismatches}/{report['hash']['entries']} words equal"  # noqa: E501 -- preserve literal/report layout
    )
    for name, r in results.items():
        steady = [t["gpu_ms"] for t in r["gpu_timings_ms"][1:]] or [
            r["gpu_timings_ms"][0]["gpu_ms"]
        ]
        print(
            f"{name:12s} {r['tile'][0]}x{r['tile'][1]} rank {r['rank']}: dU {r['dU']['relative']:.1e}, dV {r['dV']['relative']:.1e}, "  # noqa: E501 -- preserve literal/report layout
            f"loss {r['loss']['relative']:.1e} {'PASS' if r['passed'] else 'FAIL'}; GPU {min(steady):.1f} ms, NumPy {r['numpy_seconds'] * 1e3:.0f} ms"  # noqa: E501 -- preserve literal/report layout
        )
    print(
        f"stage A {'PASSED' if passed else 'FAILED'}: {output / 'report.json'}"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
