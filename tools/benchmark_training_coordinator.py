"""Compare frozen baseline and candidate coordinator paths on one local GPU.

Each pair uses identical training inputs and full GPU residency. One/four worker
processes share the default GPU: this measures coordinator changes, not scaling.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def source_hash(root):
    package = root / "ygaraty"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in (".py", ".mjs"):
            digest.update(str(path.relative_to(package)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def compare(output, workers, seeds):
    pairs = []
    for count in workers:
        for seed in seeds:
            reports = [
                json.loads(
                    (output / f"{seed}-{count}w-{arm}.json").read_text()
                )
                for arm in ("baseline", "candidate")
            ]
            baseline, candidate = reports
            for key in ("config", "dataset", "samples"):
                if baseline[key] != candidate[key]:
                    raise ValueError(
                        f"Mismatched {key}: {count} workers, seed {seed}"
                    )
            errors = []
            for a, b in zip(
                baseline["metrics"], candidate["metrics"], strict=True
            ):
                if a.get("evaluation"):
                    errors.append(
                        abs(a["evaluation"]["loss"] - b["evaluation"]["loss"])
                    )
                    if (
                        a["evaluation"]["accuracy"]
                        != b["evaluation"]["accuracy"]
                    ):
                        raise ValueError("Accuracy differs")
                if a.get("losses"):
                    for worker in a["losses"]:
                        errors.extend(
                            abs(x - y)
                            for x, y in zip(
                                a["losses"][worker],
                                b["losses"][worker],
                                strict=True,
                            )
                        )
            error = max(errors)
            if error > 1e-10:
                raise ValueError(f"Learning-curve parity failed: {error}")
            for report in reports:
                stats = [
                    s
                    for m in report["metrics"]
                    for s in m.get("worker_stats", {}).values()
                ]
                if not stats or any(
                    s.get("resident_rounds") != 1
                    or s.get("readbacks") != 1
                    or s["cpu_tasks"] != 0
                    for s in stats
                ):
                    raise ValueError("GPU residency counter gate failed")
            pairs.append(
                dict(
                    workers=count,
                    seed=seed,
                    baseline_seconds=baseline["seconds"],
                    candidate_seconds=candidate["seconds"],
                    speedup=baseline["seconds"] / candidate["seconds"],
                    max_loss_difference=error,
                    final_loss=candidate["metrics"][-1]["evaluation"]["loss"],
                    accuracy=candidate["metrics"][-1]["evaluation"][
                        "accuracy"
                    ],
                )
            )
    medians = []
    for count in workers:
        rows = [r for r in pairs if r["workers"] == count]
        old = statistics.median(r["baseline_seconds"] for r in rows)
        new = statistics.median(r["candidate_seconds"] for r in rows)
        medians.append(
            dict(
                workers=count,
                baseline_seconds=old,
                candidate_seconds=new,
                speedup=old / new,
            )
        )
    return dict(pairs=pairs, medians=medians)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3072)
    parser.add_argument(
        "--workers", type=int, nargs="+", choices=(1, 2, 4), default=[1, 4]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2026, 2027, 2028]
    )
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.samples < 1536 or args.samples % 1536:
        parser.error("--samples must be a positive multiple of 1536")
    roots = dict(
        baseline=args.baseline_root.resolve(),
        candidate=args.candidate_root.resolve(),
    )
    for root in roots.values():
        if not (root / "tools/training_e2e.py").is_file():
            parser.error("Each root must contain the public training harness")
    if len(set(args.workers)) != len(args.workers) or len(
        set(args.seeds)
    ) != len(args.seeds):
        parser.error("Workers and seeds must not repeat")
    if roots["baseline"] == roots["candidate"]:
        parser.error("Baseline and candidate roots must be distinct")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {arm: source_hash(root) for arm, root in roots.items()}
    order = []
    for index, seed in enumerate(args.seeds):
        workers = args.workers if index % 2 == 0 else args.workers[::-1]
        arms = (
            ["baseline", "candidate"]
            if index % 2 == 0
            else ["candidate", "baseline"]
        )
        order.extend([seed, n, arm] for n in workers for arm in arms)
    protocol = dict(
        source_sha256=hashes,
        trial_order=order,
        samples=args.samples,
        dataset="fashion-mnist",
        channels=[8, 16, 16],
        optimizer="muon",
        batch_size=16,
        local_steps=8,
        evaluation_examples=256,
        evaluation_every_samples=1536,
        loss_tolerance=1e-10,
        topology="same-host default GPU; processes are not independent physical GPUs",
        cpu_library_threads=1,
        platform=platform.platform(),
        runtime=subprocess.check_output(
            ["deno", "--version"], text=True
        ).strip(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing="Unprofiled; includes contributor startup, training and evaluation. Excludes initial coordinator evaluation and dataset loading. No thermal/cache reset.",
    )
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    for seed, count, arm in order:
        root = roots[arm]
        if source_hash(root) != hashes[arm]:
            raise RuntimeError("Source changed during paired benchmark")
        name = f"{seed}-{count}w-{arm}"
        report = args.output / (name + ".json")
        command = [
            sys.executable,
            str(root / "tools/training_e2e.py"),
            "--runtime",
            "deno",
            "--precision",
            "webgpu",
            "--training-execution",
            "resident",
            "--num-workers",
            str(count),
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
            str(args.samples // (128 * count)),
            "--dataset",
            "fashion-mnist",
            "--eval-examples",
            "256",
            "--eval-interval",
            str(12 // count),
            "--training-run",
            "--seed",
            str(seed),
            "--timeout",
            str(args.timeout),
            "--output",
            str(report),
        ]
        print("Running", name, flush=True)
        env = dict(
            os.environ,
            PYTHONPATH=str(root),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout + 30,
                check=True,
            )
        result = json.loads(report.read_text())
        if (
            result["source_sha256"] != hashes[arm]
            or source_hash(root) != hashes[arm]
        ):
            raise RuntimeError("Report/source hash mismatch")
        print(name, round(result["seconds"], 3), "seconds", flush=True)
    summary = compare(args.output, args.workers, args.seeds)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary["medians"], indent=2))


if __name__ == "__main__":
    main()
