"""Sequential, bounded Adam/Muon DiLoCo pilot on one physical GPU.

Writes the protocol before launching any trials. Each arm uses the same sample
budget; worker-count arms have different shards and numbers of outer updates.
Uses the public training harness, closes its browsers, and retains every trial.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from ygaraty.training.session import implementation_hash


def summarize(reports, targets):
    rows = []
    for filename in reports:
        d = json.loads(filename.read_text())
        cfg = d["config"]
        stats = [
            s for m in d["metrics"][1:] for s in m["worker_stats"].values()
        ]
        row = dict(
            file=filename.name,
            optimizer=cfg["optimizer"],
            workers=cfg["num_workers"],
            seed=cfg["seed"],
            seconds=d["seconds"],
            samples=d["samples"],
            final_loss=d["metrics"][-1]["evaluation"]["loss"],
            accuracy=d["metrics"][-1]["evaluation"]["accuracy"],
            worker_compute_seconds=sum(s["seconds"] for s in stats),
            optimizer_seconds=sum(s["optimizer_seconds"] for s in stats)
            if all("optimizer_seconds" in s for s in stats)
            else None,
            traffic=d["traffic"],
            time_to_loss={},
        )
        for target in targets:
            crossing = next(
                (
                    o
                    for o in d["observations"]
                    if o.get("evaluation") is not None
                    and o["evaluation"]["loss"] <= target
                ),
                None,
            )
            row["time_to_loss"][str(target)] = (
                None if crossing is None else crossing["seconds_since_launch"]
            )
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027])
    p.add_argument("--samples", type=int, default=15360)
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()
    if args.samples < 768 or args.samples % 768:
        p.error("samples must be a positive multiple of 768")
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        p.error("seeds must be unique uint32 values")
    if args.timeout <= 0:
        p.error("timeout must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    runs = []
    for i, seed in enumerate(args.seeds):
        order = ["adam", "muon"] if i % 2 == 0 else ["muon", "adam"]
        for workers in [1, 3]:
            for optimizer in order:
                name = f"seed-{seed}-{workers}w-{optimizer}"
                command = [
                    sys.executable,
                    str(root / "tools/training_e2e.py"),
                    "--training-run",
                    "--dataset",
                    "fashion-mnist",
                    "--precision",
                    "webgpu",
                    "--num-workers",
                    str(workers),
                    "--optimizer",
                    optimizer,
                    "--learning-rate",
                    "0.001",
                    "--muon-learning-rate",
                    "0.02",
                    "--rounds",
                    str(args.samples // (workers * 128)),
                    "--local-steps",
                    "8",
                    "--batch-size",
                    "16",
                    "--channels",
                    "8",
                    "16",
                    "16",
                    "--seed",
                    str(seed),
                    "--eval-examples",
                    "1024",
                    "--eval-interval",
                    str(6 // workers),
                    "--timeout",
                    str(args.timeout),
                    "--output",
                    str((args.output / (name + ".json")).resolve()),
                ]
                runs.append(dict(name=name, command=command))
    protocol = dict(
        kind="same-host Muon/Adam DiLoCo pilot",
        source_sha256=implementation_hash(),
        targets=[1.5, 1.0],
        samples=args.samples,
        seeds=args.seeds,
        optimizer_tuning="Fixed rates, no tuning or trial selection",
        outer="Nesterov lr=0.7 momentum=0.9",
        weight_decay=0,
        muon=dict(momentum=0.95, ns_steps=5, learning_rate=0.02),
        adam=dict(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8),
        evaluation="First observed crossing, no interpolation; 1024 held-out test examples every 768 training samples",
        limitations=[
            "one physical GPU",
            "worker count changes shards and global update count",
            "two seeds, short runs",
            "uncontrolled shader cache and thermals",
        ],
        platform=platform.platform(),
        runs=[dict(name=x["name"], arguments=x["command"][2:]) for x in runs],
    )
    # Public protocol records portable arguments, not local interpreter/repository paths.
    for run in protocol["runs"]:
        run["arguments"][-1] = run["name"] + ".json"
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )
    reports = []
    for run in runs:
        print("START " + run["name"], flush=True)
        with (args.output / (run["name"] + ".log")).open("w") as log:
            completed = subprocess.run(
                run["command"],
                cwd=root,
                env=dict(os.environ, PYTHONPATH=str(root)),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        log_path = args.output / (run["name"] + ".log")
        log_path.write_text(
            log_path.read_text()
            .replace(str(root), "<checkout>")
            .replace(sys.prefix, "<python-env>")
            .replace(str(Path.home()), "<home>")
        )
        if completed.returncode:
            raise RuntimeError("Trial failed; retained log: " + run["name"])
        report = args.output / (run["name"] + ".json")
        if (
            json.loads(report.read_text())["source_sha256"]
            != protocol["source_sha256"]
        ):
            raise RuntimeError("Source changed between trials")
        reports.append(report)
        rows = summarize(reports, protocol["targets"])
        (args.output / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        print(json.dumps(rows[-1]), flush=True)
    print(
        "All trials completed; each harness stopped its browsers/server.",
        flush=True,
    )


if __name__ == "__main__":
    main()
