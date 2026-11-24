"""Validate and summarize a completed Muon pilot; optionally plot its curves."""

import argparse
import json
import statistics
from pathlib import Path

from muon_experiment import summarize


def compare(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    paths = [directory / (run["name"] + ".json") for run in protocol["runs"]]
    reports = [json.loads(p.read_text()) for p in paths]
    first = reports[0]
    for d in reports:
        if (
            d["kind"] != "same-host-browser-training"
            or d["source_sha256"] != protocol["source_sha256"]
            or d["dataset"] != first["dataset"]
            or d["samples"] != protocol["samples"]
        ):
            raise ValueError("Incompatible trial identity")
        final = d["observations"][-1]
        if (
            final["version"] != d["config"]["rounds"]
            or final["evaluation"] != d["metrics"][-1]["evaluation"]
        ):
            raise ValueError("Missing final committed evaluation")
        for m in d["metrics"][1:]:
            if len(m["worker_stats"]) != d["config"]["num_workers"]:
                raise ValueError("Incomplete worker membership")
            for s in m["worker_stats"].values():
                if (
                    s["gpu_tasks"] <= 0
                    or s["adapter"]["isFallbackAdapter"] is not False
                ):
                    raise ValueError("Hardware GPU execution not established")
    for seed in protocol["seeds"]:
        for workers in [1, 3]:
            pair = [
                d
                for d in reports
                if d["config"]["seed"] == seed
                and d["config"]["num_workers"] == workers
            ]
            if len(pair) != 2 or {d["config"]["optimizer"] for d in pair} != {
                "adam",
                "muon",
            }:
                raise ValueError("Incomplete optimizer pair")
            configs = [
                {k: v for k, v in d["config"].items() if k != "optimizer"}
                for d in pair
            ]
            if (
                configs[0] != configs[1]
                or pair[0]["metrics"][0]["evaluation"]
                != pair[1]["metrics"][0]["evaluation"]
            ):
                raise ValueError(
                    "Optimizer pair initialization or configuration mismatch"
                )
    rows = summarize(paths, protocol["targets"])
    aggregates = []
    for workers in [1, 3]:
        for optimizer in ["adam", "muon"]:
            group = [
                x
                for x in rows
                if x["workers"] == workers and x["optimizer"] == optimizer
            ]
            targets = {}
            for target in protocol["targets"]:
                hits = [
                    x["time_to_loss"][str(target)]
                    for x in group
                    if x["time_to_loss"][str(target)] is not None
                ]
                targets[str(target)] = {
                    "reached": len(hits),
                    "trials": len(group),
                    "median_seconds": statistics.median(hits)
                    if len(hits) == len(group)
                    else None,
                }
            aggregates.append(
                dict(
                    workers=workers,
                    optimizer=optimizer,
                    targets=targets,
                    median_seconds=statistics.median(
                        x["seconds"] for x in group
                    ),
                    median_final_loss=statistics.median(
                        x["final_loss"] for x in group
                    ),
                    median_accuracy=statistics.median(
                        x["accuracy"] for x in group
                    ),
                    optimizer_share_of_worker_execution=sum(
                        x["optimizer_seconds"] for x in group
                    )
                    / sum(x["worker_compute_seconds"] for x in group)
                    if all(x["optimizer_seconds"] is not None for x in group)
                    else None,
                )
            )
    return dict(trials=rows, aggregates=aggregates), reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result, reports = compare(args.directory)
    (args.directory / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["aggregates"], indent=2))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seeds = sorted({d["config"]["seed"] for d in reports})
        fig, axes = plt.subplots(
            len(seeds),
            2,
            figsize=(11, 4 * len(seeds)),
            squeeze=False,
            sharey=True,
        )
        for row, seed in enumerate(seeds):
            for d in reports:
                cfg = d["config"]
                if cfg["seed"] != seed:
                    continue
                curve = [
                    o
                    for o in d["observations"]
                    if o.get("evaluation") is not None
                ]
                y = [o["evaluation"]["loss"] for o in curve]
                style = dict(
                    color="#2563eb"
                    if cfg["optimizer"] == "adam"
                    else "#c2410c",
                    linestyle="-" if cfg["num_workers"] == 1 else "--",
                    linewidth=1.7,
                    label=f"{cfg['optimizer'].title()}, {cfg['num_workers']} worker(s)",
                )
                axes[row, 0].plot(
                    [o["seconds_since_launch"] for o in curve], y, **style
                )
                axes[row, 1].plot(
                    [
                        o["version"]
                        * cfg["local_steps"]
                        * cfg["batch_size"]
                        * cfg["num_workers"]
                        for o in curve
                    ],
                    y,
                    **style,
                )
            for col, ax in enumerate(axes[row]):
                for target in [1.5, 1.0]:
                    ax.axhline(
                        target, color="#64748b", linewidth=0.8, alpha=0.6
                    )
                ax.set_title(
                    f"Seed {seed} — {'elapsed time' if col == 0 else 'training samples'}"
                )
                ax.set_xlabel("Seconds" if col == 0 else "Examples processed")
                ax.set_ylabel("Held-out cross-entropy")
                ax.grid(alpha=0.15)
                ax.legend(fontsize=8)
        fig.suptitle(
            "Muon vs Adam inside DiLoCo · one Apple M1 GPU", fontsize=14
        )
        fig.tight_layout()
        fig.savefig(args.directory / "learning-curves.png", dpi=160)
        fig.savefig(args.directory / "learning-curves.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
