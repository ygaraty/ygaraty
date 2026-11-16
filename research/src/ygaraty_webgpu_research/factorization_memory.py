"""Experiment 48 stage C: memory budgets hold against oversized and malformed
tasks.

    ygaraty experiment factorization_memory

Headless Chromium on the machine's GPU (stage A's flags) runs each case through
the browser
worker's admission (sdk.mjs admitTask: its own count from budget.mjs must equal
the task's
memory_bytes and fit the budget) and, when admitted, its execution, counting
every GPU buffer the
device creates. Cases: a real tile of the 8,192 plan at a budget it exactly
fits and one byte
under it; a reduction under a 1 KiB budget; a coordinator that understates a
task's size; a tile
far larger than any budget; malformed shapes, specs, input lists and ops; a
seed a browser cannot
hold; and a task from a job that does not enforce budgets (admitted as before).

Gates, fixed before the first run: every case expected to be refused is refused
and creates no
GPU buffer; every admitted case's GPU bytes stay within its bound's GPU share
(runtime/budget.py);
the exactly fitting tile's result matches NumPy float64 to stage A's 1e-5.
"""

import argparse
import copy
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.host import component as _host_component

SDK_ASSETS = _host_component("browser_assets").SDK_ASSETS
from ygaraty_webgpu_research.factorization_webgpu import GPU_FLAGS
from ygaraty_webgpu_research.webgpu_parity import run_browser
from ygaraty.models.factorization_sharded import (
    build_sharded_plan,
    shard_values,
)
from ygaraty_webgpu_research.host import component as _host_component

task_bytes = _host_component("runtime.budget").task_bytes
from ygaraty.runtime.factorization_kernels import execute

ASSETS = (
    "factorization_stage_c.mjs",
    "factorization_gpu.mjs",
    "sdk.mjs",
    "budget.mjs",
    "convolution.mjs",
) + SDK_ASSETS
TOLERANCE = 1e-5


def gpu_bound(task):
    """The GPU share of runtime/budget.py's bound for a GPU tile: float32
    inputs, residual, output
    and its readback, parameters."""
    rows, cols = task["attrs"]["x_spec"]["shape"]
    rank = task["shape"][1]
    inputs = sum(r * c for r, c in task["input_shapes"])
    return (
        4 * inputs
        + 4 * rows * cols
        + 2 * 4 * ((rows + cols) * rank + rows)
        + 64
    )


def described(task, plan_shapes):
    shapes = [
        [x.region[1] - x.region[0], x.region[3] - x.region[2]]
        if x.region
        else list(plan_shapes[x.value])
        for x in task.inputs
    ]
    return dict(
        id=task.id,
        op=task.op,
        shape=list(task.shape),
        attrs=copy.deepcopy(task.attrs),
        input_shapes=shapes,
        memory_bytes=task_bytes(task.op, shapes, list(task.shape), task.attrs),
    )


def make_cases(seed):
    plan = build_sharded_plan(
        8192,
        8192,
        64,
        block_rows=2048,
        block_columns=2048,
        learning_rate=500.0,
        seed=seed,
        generated_x=True,
        latent_inputs=True,
    )
    shapes = {
        **{n: v["shape"] for n, v in plan.leaves.items()},
        **{t.id: list(t.shape) for t in plan.tasks},
    }
    tile_task = next(t for t in plan.tasks if t.id == "tile/1/2")
    tile = described(tile_task, shapes)
    arrays = [
        shard_values(plan.initializers[x.value]) for x in tile_task.inputs
    ]
    inputs = [
        dict(shape=list(a.shape), data=a.ravel().tolist()) for a in arrays
    ]
    shapes_only = [dict(shape=list(a.shape)) for a in arrays]
    reduction_task = next(t for t in plan.tasks if t.op == "add_many")
    reduction = described(reduction_task, shapes)
    reduction_inputs = [dict(shape=s) for s in reduction["input_shapes"]]
    need = tile["memory_bytes"]
    big_spec = dict(tile["attrs"]["x_spec"], shape=[65536, 65536])
    big = dict(
        tile,
        attrs=dict(tile["attrs"], x_spec=big_spec),
        shape=[65536 * 2 + 1, 64],
        input_shapes=[[65536, 8], [65536, 8], [65536, 64], [65536, 64]],
    )
    big["memory_bytes"] = task_bytes(
        "factorization_tile", big["input_shapes"], big["shape"], big["attrs"]
    )

    def task(base, **changes):
        out = {
            k: v
            for k, v in dict(base, **changes).items()
            if k != "input_shapes"
        }
        return out

    cases = [
        # name, task, inputs, budget, expected: 'run' or 'refuse'
        ("tile fits exactly", task(tile), inputs, need, "run"),
        (
            "tile one byte over budget",
            task(tile),
            shapes_only,
            need - 1,
            "refuse",
        ),
        (
            "reduction under a 1 KiB budget",
            task(reduction),
            reduction_inputs,
            1024,
            "refuse",
        ),
        (
            "coordinator understates the size",
            task(tile, memory_bytes=need // 100),
            shapes_only,
            need,
            "refuse",
        ),
        (
            "tile far larger than any budget",
            task(big),
            [dict(shape=s) for s in big["input_shapes"]],
            1 << 30,
            "refuse",
        ),
        (
            "negative shape",
            task(tile, shape=[-1, 64]),
            shapes_only,
            1 << 30,
            "refuse",
        ),
        (
            "fractional shape",
            task(tile, shape=[4097.5, 64]),
            shapes_only,
            1 << 30,
            "refuse",
        ),
        (
            "missing input shape",
            task(tile),
            shapes_only[:3] + [dict()],
            1 << 30,
            "refuse",
        ),
        ("three inputs", task(tile), shapes_only[:3], 1 << 30, "refuse"),
        (
            "no x_spec",
            task(
                tile,
                attrs=dict(total_elements=tile["attrs"]["total_elements"]),
            ),
            shapes_only,
            1 << 30,
            "refuse",
        ),
        (
            "op without a bound",
            task(tile, op="matmul"),
            shapes_only,
            1 << 30,
            "refuse",
        ),
        ("budget missing", task(tile), shapes_only, None, "refuse"),
        (
            "seed above 2^53",
            task(
                tile,
                attrs=dict(
                    tile["attrs"],
                    x_spec=dict(tile["attrs"]["x_spec"], seed=2**60),
                ),
            ),
            inputs,
            1 << 30,
            "refuse",
        ),
        (
            "non-enforcing job (no memory_bytes)",
            {k: v for k, v in task(tile).items() if k != "memory_bytes"},
            inputs,
            None,
            "run",
        ),
    ]
    reference = execute("factorization_tile", arrays, tile_task.attrs)
    return [
        dict(
            name=n,
            task=t,
            inputs=i,
            budget=b,
            expected=e,
            return_output=n == "tile fits exactly",
            bound=t.get("memory_bytes"),
            gpu_bound=gpu_bound(tile) if e == "run" else None,
        )
        for n, t, i, b, e in cases
    ], reference


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chrome")
    parser.add_argument(
        "--chrome-arg",
        action="append",
        help=f"Chromium flags (repeat); default: {' '.join(GPU_FLAGS)}",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-software", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.chrome_arg = (
        list(GPU_FLAGS) if args.chrome_arg is None else args.chrome_arg
    )
    output = args.output or Path(
        "runs/experiments/048/stage-c"
    ) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    cases, reference = make_cases(args.seed)
    browser = run_browser(
        dict(cases=cases),
        output,
        args.chrome,
        args.timeout,
        args.chrome_arg,
        page="factorization_stage_c.mjs",
        assets=ASSETS,
    )
    adapter = browser["adapter"]
    software = (
        "swiftshader" in json.dumps(adapter).lower()
        or adapter.get("isFallbackAdapter") is True
    )
    rows = []
    for case, got in zip(cases, browser["results"]):
        ok = True
        if case["expected"] == "refuse":
            stopped = got["refused"] is not None or got["error"] is not None
            ok = stopped and got["buffers"] == 0
        else:
            ok = (
                got["refused"] is None
                and got["error"] is None
                and got["gpu_bytes"] <= case["gpu_bound"]
            )
        accuracy = None
        if got["output"] is not None:
            out = np.asarray(got["output"]).reshape(reference.shape)
            rows_, cols = case["task"]["attrs"]["x_spec"]["shape"]
            accuracy = max(
                float(
                    np.abs(out[s] - reference[s]).max()
                    / np.abs(reference[s]).max()
                )
                for s in (slice(0, rows_), slice(rows_, rows_ + cols))
            )
            ok = ok and accuracy <= TOLERANCE
        rows.append(
            dict(
                name=case["name"],
                expected=case["expected"],
                passed=ok,
                refused=got["refused"],
                error=got["error"],
                gpu_buffers=got["buffers"],
                gpu_bytes=got["gpu_bytes"],
                bound=case["bound"],
                gpu_bound=case["gpu_bound"],
                budget=case["budget"],
                relative_error=accuracy,
            )
        )
    passed = all(r["passed"] for r in rows) and (
        not software or args.allow_software
    )
    report = dict(
        experiment=48,
        stage="C",
        passed=passed,
        adapter=adapter,
        software_adapter=software,
        default_budget=browser["default_budget"],
        cases=rows,
        chrome_args=args.chrome_arg,
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
        f"adapter {adapter.get('vendor')} "
        f"{adapter.get('architecture')}{' (SOFTWARE)' if software else ''}; default budget {browser['default_budget']} bytes"  # noqa: E501 -- preserve literal/report layout
    )
    for r in rows:
        why = (r["refused"] or r["error"] or "")[:90]
        bound = f" (GPU bound {r['gpu_bound']})" if r["gpu_bound"] else ""
        error = (
            f", error {r['relative_error']:.1e}"
            if r["relative_error"] is not None
            else ""
        )
        print(
            f"{'PASS' if r['passed'] else 'FAIL'} {r['name']:38s} {r['expected']:6s} buffers {r['gpu_buffers']:2d}, "  # noqa: E501 -- preserve literal/report layout
            f"{r['gpu_bytes']:>10d} GPU bytes{bound}{error}  {why}"
        )
    print(
        f"stage C {'PASSED' if passed else 'FAILED'}: {output / 'report.json'}"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
