"""Plot saved local predictions; requires matplotlib and runs no simulation."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path("runs/factorization-comparison")
    )
    args = parser.parse_args()
    comparison = json.loads((args.directory / "comparison.json").read_text())
    larger = json.loads((args.directory / "fused-4096.json").read_text())
    reference = comparison["reports"]["reference_batch16"]
    variants = [
        ("reference_batch16", "Reference, claim 16", "#777777"),
        ("fused_batch16", "Fused, claim 16", "#2166ac"),
        ("fused_batch1", "Fused, claim 1", "#13816e"),
    ]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for index, scenario in enumerate(("loopback", "home")):
        latency, scaling = axes[index]
        for key, label, color in variants:
            rows = next(
                s["rows"]
                for s in comparison["reports"][key]["scenarios"]
                if s["name"] == scenario
            )
            x = [row["workers"] for row in rows]
            y = [row["seconds_median"] for row in rows]
            latency.errorbar(
                x,
                y,
                yerr=[
                    [
                        row["seconds_median"] - row["seconds_min"]
                        for row in rows
                    ],
                    [
                        row["seconds_max"] - row["seconds_median"]
                        for row in rows
                    ],
                ],
                marker="o",
                capsize=3,
                label=label,
                color=color,
            )
        latency.set_yscale("log")
        latency.set_ylabel("Predicted seconds / update (log scale)")
        small = reference["workload"]
        latency.set_title(
            f"{scenario.title()}: {small['rows']:,} × "
            f"{small['columns']:,}, rank {small['rank']}"
        )
        rows = next(
            s["rows"] for s in larger["scenarios"] if s["name"] == scenario
        )
        x = [row["workers"] for row in rows]
        scaling.plot(x, x, "--", color="#777777", label="Ideal linear")
        scaling.plot(
            x,
            [n * 0.8 for n in x],
            ":",
            color="#999999",
            label="80% efficiency",
        )
        scaling.plot(
            x,
            [row["speedup"] for row in rows],
            marker="o",
            color="#13816e",
            label="Fused, claim 1",
        )
        big = larger["workload"]
        scaling.set_title(
            f"{scenario.title()}: {big['rows']:,} × {big['columns']:,}, "
            f"rank {big['rank']}"
        )
        scaling.set_ylabel("Predicted speedup over own one-slot time")
        for ax in (latency, scaling):
            ax.set_xscale("log", base=2)
            ax.set_xticks(x, labels=x)
            ax.set_xlabel("Modeled worker slots")
            ax.grid(True, alpha=0.18)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Fused matrix factorization: modeled predictions",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        f"Median update {reference['reported_update']} across {len(reference['seeds'])} seeds · "  # noqa: E501 -- preserve fixture or report layout
        "Fixed problem size within each panel",
        ha="center",
    )
    fig.text(
        0.5,
        0.025,
        "Not hardware measurements. Costs are uncalibrated for "
        "this workload. Left error bars show seed min/max.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    for suffix in ("png", "svg"):
        fig.savefig(args.directory / ("comparison." + suffix), dpi=160)


if __name__ == "__main__":
    main()
