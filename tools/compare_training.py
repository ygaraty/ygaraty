"""Compare recorded contributor training curves at matched sample budgets.

Times are first observed evaluation crossings, with no interpolation. HTTP
body bytes include assets and protocol traffic, but exclude headers and TLS.
"""

import argparse
import json
from pathlib import Path


def summarize(report, target):
    if report["kind"] not in {
        "same-host-browser-training",
        "same-host-native-training",
    }:
        raise ValueError("Expected a successful measured training run")
    curve = [p for p in report["observations"] if p["evaluation"] is not None]
    final = curve[-1]
    if (
        final["version"] != report["config"]["rounds"]
        or final["evaluation"] != report["metrics"][-1]["evaluation"]
    ):
        raise ValueError(
            "Final committed evaluation is missing from observations"
        )
    crossing = next(
        (p for p in curve if p["evaluation"]["loss"] <= target), None
    )
    return dict(
        workers=report["config"]["num_workers"],
        samples=report["samples"],
        seconds=report["seconds"],
        final_loss=final["evaluation"]["loss"],
        final_accuracy=final["evaluation"]["accuracy"],
        target_loss=target,
        seconds_to_target=None
        if crossing is None
        else crossing["seconds_since_launch"],
        round_work_seconds=sum(m["seconds"] for m in report["metrics"][1:]),
        http_body_bytes=sum(report["traffic"].values()),
        curve=curve,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--target-loss", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    first = reports[0]
    for report in reports[1:]:
        if (
            report["samples"] != first["samples"]
            or report["dataset"] != first["dataset"]
            or report["source_sha256"] != first["source_sha256"]
        ):
            raise ValueError(
                "Comparison needs matching sample budgets, "
                "dataset and implementation"
            )
        for name in (
            "channels",
            "batch_size",
            "local_steps",
            "learning_rate",
            "optimizer",
            "muon_learning_rate",
            "muon_momentum",
            "muon_steps",
            "outer_learning_rate",
            "momentum",
            "seed",
            "eval_examples",
            "precision",
        ):
            if report["config"].get(name) != first["config"].get(name):
                raise ValueError(f"Comparison differs in {name}")
    summaries = [summarize(r, args.target_loss) for r in reports]
    for summary in summaries:
        summary["seconds_to_final_losses"] = {
            str(other["workers"]): next(
                (
                    point["seconds_since_launch"]
                    for point in summary["curve"]
                    if point["evaluation"]["loss"] <= other["final_loss"]
                ),
                None,
            )
            for other in summaries
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for report, summary in zip(reports, summaries):
            curve = summary["curve"]
            losses = [p["evaluation"]["loss"] for p in curve]
            runtime = report.get("runtime", "chromium")
            label = f"{summary['workers']} {runtime} worker(s)"
            axes[0].plot(
                [p["seconds_since_launch"] for p in curve],
                losses,
                marker=".",
                label=label,
            )
            samples_per_round = (
                report["config"]["num_workers"]
                * report["config"]["local_steps"]
                * report["config"]["batch_size"]
            )
            axes[1].plot(
                [p["version"] * samples_per_round for p in curve],
                losses,
                marker=".",
                label=label,
            )
        for axis in axes:
            axis.axhline(
                args.target_loss,
                color="gray",
                linestyle="--",
                label=f"Target {args.target_loss}",
            )
            axis.set_ylabel(
                "Held-out cross-entropy "
                f"({first['config']['eval_examples']:,} images)"
            )
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0].set_xlabel("Wall seconds from contributor launch")
        axes[1].set_xlabel("Aggregate training examples processed")
        fig.suptitle(
            "DiLoCo training · see experiment record for physical GPU topology"
        )
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)
    for summary in summaries:
        print(json.dumps({k: v for k, v in summary.items() if k != "curve"}))


if __name__ == "__main__":
    main()
