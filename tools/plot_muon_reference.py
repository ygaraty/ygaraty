"""Plot sanitized one/four-GPU Ygaraty and upstream Torch reference reports."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    identities = set()
    for engine, arm, workers, color in (
        ("ygaraty", "muon", 1, "#2458b4"),
        ("ygaraty", "muon", 4, "#db6826"),
        ("torch", "ygaraty-math", 1, "#29865b"),
        ("torch", "upstream", 1, "#8753a6"),
    ):
        for seed in (2026, 2027):
            name = (
                f"muon-{workers}w-seed{seed}.json"
                if engine == "ygaraty"
                else f"torch-{arm}-seed{seed}.json"
            )
            report = json.loads((args.directory / name).read_text())
            identities.add(
                (
                    report["dataset"]["sha256"],
                    report["source_sha256"],
                    report["samples"],
                )
            )
            if engine == "ygaraty":
                curve = [
                    x for x in report["observations"] if x.get("evaluation")
                ]
                times = [x["seconds_since_launch"] for x in curve]
                samples = [x["version"] * workers * 128 for x in curve]
                label = f"Ygaraty {workers} GPU, seed {seed}"
            else:
                curve = report["curve"]
                times = [x["seconds"] for x in curve]
                samples = [x["samples"] for x in curve]
                label = f"Torch {arm}, seed {seed}"
            style = dict(
                label=label,
                color=color,
                linestyle="-" if seed == 2026 else "--",
                linewidth=1.3,
            )
            losses = [x["evaluation"]["loss"] for x in curve]
            axes[0].plot(times, losses, **style)
            axes[1].plot(samples, losses, **style)
    if len(identities) != 1:
        raise ValueError("Reports differ in dataset, source or sample budget")
    for axis in axes:
        axis.set_ylabel("Held-out cross-entropy (1,024 images)")
        axis.grid(alpha=0.2)
        axis.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel(
        "Wall seconds (log scale after 1 s; see timing protocol)"
    )
    axes[1].set_xlabel("Aggregate training examples")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Same Fashion-MNIST ConvNet · optimizer and engine comparisons"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(args.directory / f"reference-curves.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
