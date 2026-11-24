"""Compare matched global-batch gradient runs; reject different sample streams."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(paths, target=1.5):
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    for report in reports:
        if (
            report["status"] != "passed"
            or report["metrics"][-1]["step"] != report["config"]["steps"]
        ):
            raise ValueError("Incomplete gradient experiment")
        for key in (
            "source_sha256",
            "sample_sequence_sha256",
            "dataset",
            "model",
            "optimizer",
            "learning_rate",
        ):
            if report[key] != first[key]:
                raise ValueError(f"Mismatch: {key}")
        for key in (
            "steps",
            "global_batch",
            "seed",
            "eval_every",
            "eval_examples",
            "verify_steps",
        ):
            if report["config"][key] != first["config"][key]:
                raise ValueError(f"Configuration mismatch: {key}")
    summaries = []
    for path, report in zip(paths, reports):
        curve = [m for m in report["metrics"] if "evaluation" in m]
        crossing = next(
            (m for m in curve if m["evaluation"]["loss"] <= target), None
        )
        final = curve[-1]["evaluation"]
        summaries.append(
            dict(
                workers=report["config"]["num_workers"],
                samples=report["metrics"][-1]["samples"],
                steps=report["config"]["steps"],
                seconds=report["seconds"],
                final_loss=final["loss"],
                accuracy=final["accuracy"],
                seconds_to_target=None
                if crossing is None
                else crossing["seconds"],
                target_loss=target,
                http_body_bytes=sum(report["traffic"].values()),
                curve=[
                    {
                        k: m[k]
                        for k in ("step", "samples", "seconds", "evaluation")
                    }
                    for m in curve
                ],
            )
        )
    deltas = []
    with np.load(paths[0].with_suffix(".npz"), allow_pickle=False) as baseline:
        for path in paths[1:]:
            with np.load(
                path.with_suffix(".npz"), allow_pickle=False
            ) as candidate:
                differences = {}
                for prefix in (
                    "first_gradient_",
                    "first_weight_",
                    "final_weight_",
                    "final_m_",
                    "final_v_",
                ):
                    keys = [k for k in baseline.files if k.startswith(prefix)]
                    differences[prefix.rstrip("_")] = max(
                        float(np.abs(baseline[k] - candidate[k]).max())
                        for k in keys
                    )
                deltas.append(differences)
    return dict(
        config=first["config"],
        dataset=first["dataset"],
        source_sha256=first["source_sha256"],
        sample_sequence_sha256=first["sample_sequence_sha256"],
        runs=summaries,
        max_absolute_differences=deltas,
        final_loss_agreement=all(
            abs(r["final_loss"] - summaries[0]["final_loss"]) <= 0.05
            for r in summaries
        ),
        final_accuracy_agreement=all(
            abs(r["accuracy"] - summaries[0]["accuracy"]) <= 0.02
            for r in summaries
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    result = compare(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for row in result["runs"]:
            curve = row["curve"]
            label = f"{row['workers']} worker(s)"
            for axis, field in zip(axes, ("seconds", "samples")):
                axis.plot(
                    [p[field] for p in curve],
                    [p["evaluation"]["loss"] for p in curve],
                    label=label,
                    marker=".",
                    alpha=0.8,
                )
        for axis in axes:
            axis.axhline(1.5, linestyle="--", color="gray", label="Target 1.5")
            axis.grid(alpha=0.2)
            axis.set_ylabel("Held-out cross-entropy")
            axis.legend()
        axes[0].set_xlabel("End-to-end seconds")
        axes[1].set_xlabel("Identical training examples processed")
        fig.suptitle(
            "Tree gradient reduction · shared Adam · one physical GPU"
        )
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)
    print(
        json.dumps(
            {
                **result,
                "runs": [
                    {k: v for k, v in r.items() if k != "curve"}
                    for r in result["runs"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
