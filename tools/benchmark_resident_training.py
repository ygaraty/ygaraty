"""Paired local ConvNet comparison: per-step versus full-round GPU residency.

Both arms retain the earlier GPU-resident Muon subgraph. This measures the added
benefit of retaining all tensors and state across local steps, on one local GPU.
Multiple contributors here share that GPU; this is not physical GPU scaling.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from ygaraty.training.session import implementation_hash


def compare(directory, seeds):
    rows = []
    for seed in seeds:
        a, b = [
            json.loads((directory / f"{seed}-{arm}.json").read_text())
            for arm in ["tasks", "resident"]
        ]
        for key in ["config", "dataset", "samples", "source_sha256"]:
            if a[key] != b[key]:
                raise ValueError(f"Mismatched {key}")
        losses = []
        for left, right in zip(a["metrics"], b["metrics"], strict=True):
            if left.get("evaluation"):
                losses.append(
                    abs(
                        left["evaluation"]["loss"]
                        - right["evaluation"]["loss"]
                    )
                )
        error = max(losses)
        if error > 5e-5:
            raise ValueError(
                f"Learning-curve parity failed for seed {seed}: {error}"
            )
        stats_a = [
            s for m in a["metrics"] for s in m.get("worker_stats", {}).values()
        ]
        stats_b = [
            s for m in b["metrics"] for s in m.get("worker_stats", {}).values()
        ]
        if not stats_b or any(
            s.get("resident_rounds") != 1
            or s.get("readbacks") != 1
            or s["cpu_tasks"] != 0
            for s in stats_b
        ):
            raise ValueError("Resident arm did not use GPU-resident rounds")
        if any(s.get("resident_rounds", 0) for s in stats_a):
            raise ValueError("Baseline unexpectedly used resident rounds")
        rows.append(
            dict(
                seed=seed,
                tasks_seconds=a["seconds"],
                resident_seconds=b["seconds"],
                speedup=a["seconds"] / b["seconds"],
                tasks_worker_seconds=sum(s["seconds"] for s in stats_a),
                resident_worker_seconds=sum(s["seconds"] for s in stats_b),
                tasks_loss=a["metrics"][-1]["evaluation"]["loss"],
                resident_loss=b["metrics"][-1]["evaluation"]["loss"],
                max_evaluation_loss_difference=error,
                tasks_accuracy=a["metrics"][-1]["evaluation"]["accuracy"],
                resident_accuracy=b["metrics"][-1]["evaluation"]["accuracy"],
                readbacks=sum(s["readbacks"] for s in stats_b),
                max_gpu_buffer_bytes=max(
                    s["gpu_buffer_bytes"] for s in stats_b
                ),
            )
        )
    return dict(
        rows=rows,
        median_tasks_seconds=statistics.median(
            r["tasks_seconds"] for r in rows
        ),
        median_resident_seconds=statistics.median(
            r["resident_seconds"] for r in rows
        ),
        speedup=statistics.median(r["tasks_seconds"] for r in rows)
        / statistics.median(r["resident_seconds"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=768)
    parser.add_argument(
        "--num-workers", type=int, choices=[1, 2, 3, 4], default=1
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.samples < 128 * args.num_workers or args.samples % (
        128 * args.num_workers
    ):
        parser.error("samples must be a positive multiple of workers * 128")
    if (
        args.timeout < 1
        or len(set(args.seeds)) != len(args.seeds)
        or any(not 0 <= s < 2**32 for s in args.seeds)
    ):
        parser.error("invalid timeout or seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    order = [
        (seed, arm)
        for i, seed in enumerate(args.seeds)
        for arm in (
            ["tasks", "resident"] if i % 2 == 0 else ["resident", "tasks"]
        )
    ]
    source = implementation_hash()
    protocol = dict(
        source_sha256=source,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset="fashion-mnist",
        samples=args.samples,
        workers=args.num_workers,
        topology="same-host default adapter; no independent physical GPU count",
        channels=[8, 16, 16],
        batch_size=16,
        local_steps=8,
        rounds=args.samples // (128 * args.num_workers),
        optimizer="muon",
        muon_execution="resident",
        eval_examples=256,
        eval_interval=2,
        loss_parity_tolerance=5e-5,
        trial_order=order,
        timeout_seconds=args.timeout,
        cpu_library_threads=1,
        runtime=subprocess.check_output(
            ["deno", "--version"], text=True
        ).strip(),
        notes="Sequential same-GPU trials; includes startup and evaluation; no thermal/cache reset; initial coordinator evaluation and dataset download excluded.",
    )
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(root),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for seed, arm in order:
        name = f"{seed}-{arm}"
        command = [
            sys.executable,
            str(root / "tools/training_e2e.py"),
            "--runtime",
            "deno",
            "--precision",
            "webgpu",
            "--num-workers",
            str(args.num_workers),
            "--optimizer",
            "muon",
            "--channels",
            "8",
            "16",
            "16",
            "--batch-size",
            "16",
            "--local-steps",
            "8",
            "--rounds",
            str(protocol["rounds"]),
            "--dataset",
            "fashion-mnist",
            "--eval-examples",
            "256",
            "--eval-interval",
            "2",
            "--training-run",
            "--seed",
            str(seed),
            "--training-execution",
            arm,
            "--muon-execution",
            "resident",
            "--output",
            str(args.output / f"{name}.json"),
            "--timeout",
            str(args.timeout),
        ]
        with (args.output / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout + 60,
            )
        report = json.loads((args.output / f"{name}.json").read_text())
        if (
            report["source_sha256"] != source
            or implementation_hash() != source
        ):
            raise ValueError("Source changed during benchmark")
        print(
            f"{name}: {report['seconds']:.2f}s, loss {report['metrics'][-1]['evaluation']['loss']:.8f}",
            flush=True,
        )
    summary = compare(args.output, args.seeds)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Median end-to-end speedup: {summary['speedup']:.2f}x")


if __name__ == "__main__":
    main()
