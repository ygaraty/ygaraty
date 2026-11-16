"""Audit and summarize repeated three-mode browser Transformer trials."""

import argparse
import json
from pathlib import Path
from statistics import median


def summarize_benchmark(source):
    data = json.loads(Path(source).read_text())
    if not data["complete"] or data["error"]:
        raise ValueError("Benchmark incomplete or failed")
    results = data["results"]
    if [(r["round"], r["mode"]) for r in results] != [
        (r["round"], r["mode"]) for r in data["order"]
    ]:
        raise ValueError("Trial order mismatch")
    reference = None
    gpu_thresholds = None
    check_keys = None
    for r in results:
        report = json.loads(Path(r["report_path"]).read_text())
        ref = {
            k: report["reference"][k]
            for k in (
                "checkpoint_sha256",
                "tokenizer_sha256",
                "fixture_sha256",
                "config",
                "optimizer",
                "label_smoothing",
            )
        }
        if reference is None:
            reference = ref
        if reference != ref:
            raise ValueError("Different reference")
        keys = set(report["checks"])
        if check_keys is None:
            check_keys = keys
        if keys != check_keys or len(keys) != 511:
            raise ValueError("Different verification coverage")
        if r["mode"] != "float64":
            thresholds = {
                k: (c["rtol"], c["atol"], c["comparison"])
                for k, c in report["checks"].items()
            }
            if gpu_thresholds is None:
                gpu_thresholds = thresholds
            if gpu_thresholds != thresholds:
                raise ValueError("GPU tolerances changed")
        if (
            len(report["workers"]) != 2
            or report["retries"]
            or report["rejected"]
        ):
            raise ValueError("Invalid worker conditions")
        if any(w["completed"] == 0 for w in report["workers"].values()):
            raise ValueError("Idle contributor")
        plan = json.loads(
            (Path(r["report_path"]).parent / "plan.json").read_text()
        )
        matmuls = sum(t["op"] == "matmul" for t in plan)
        if report["gpu_operations"] != (
            0 if r["mode"] == "float64" else matmuls
        ):
            raise ValueError("GPU coverage changed")
        if r["mode"] != "float32" and (
            not report["passed"]
            or not all(
                c["strict_float64_passed"] for c in report["checks"].values()
            )
        ):
            raise ValueError("Precision regression")
    modes = {}
    for mode in ("float64", "float32", "compensated"):
        trials = [r for r in results if r["mode"] == mode]
        metrics = {}
        for key in (
            "active_seconds",
            "worker_compute_seconds",
            "transferred_bytes",
        ):
            values = [r[key] for r in trials]
            metrics[key] = dict(
                median=median(values),
                min=min(values),
                max=max(values),
                trials=values,
            )
        modes[mode] = dict(
            runs=len(trials),
            passed_runs=sum(r["passed"] for r in trials),
            metrics=metrics,
            max_abs_error={
                k: max(r["groups"][k]["max_abs_error"] for r in trials)
                for k in trials[0]["groups"]
            },
        )
    comparisons = {}
    for baseline in ("float32", "float64"):
        paired = {}
        for metric in ("active_seconds", "worker_compute_seconds"):
            ratios = []
            for rnd in sorted({r["round"] for r in results}):
                bymode = {r["mode"]: r for r in results if r["round"] == rnd}
                ratios.append(
                    bymode["compensated"][metric] / bymode[baseline][metric]
                )
            ratio = (
                modes["compensated"]["metrics"][metric]["median"]
                / modes[baseline]["metrics"][metric]["median"]
            )
            paired[metric] = dict(
                ratio_of_medians=ratio,
                percent_cost=100 * (ratio - 1),
                paired_ratios=ratios,
                median_paired_ratio=median(ratios),
            )
        comparisons["compensated_vs_" + baseline] = paired
    return dict(
        modes=modes,
        comparisons=comparisons,
        reference=reference,
        trials=[
            {
                k: r[k]
                for k in (
                    "round",
                    "mode",
                    "report_path",
                    "active_seconds",
                    "worker_compute_seconds",
                    "transferred_bytes",
                    "passed",
                )
            }
            for r in results
        ],
        methodology=[
            "Two real browser workers on the same host, sharing its "
            "GPU; 3-mode order rotated each round.",
            "Same checkpoint, minibatch, optimizer and tile size 512. "
            "Fresh SDK and reset checkpoint per trial.",
            "Active elapsed time excludes job construction and worker "
            "startup; includes scheduling, HTTP transport, "
            "verification and checkpoint updates.",
            "Worker compute is summed across both workers, including "
            "JS kernels, GPU conversion/allocation, pipeline "
            "creation, dispatch and readback; not isolated shader "
            "time.",
            "All 1,344 matmuls use GPU in both GPU modes; remaining "
            "kernels and HTTP tensor payloads use float64.",
            "No separate warm-up trials. Fresh worker SDK each trial; "
            "browser/driver caches may stay warm.",
            "Small local sample; ranges and per-round ratios are "
            "reported, not statistical significance or remote scaling "
            "claims.",
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summary = summarize_benchmark(args.source)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({k: summary[k] for k in ("modes", "comparisons")}, indent=2)
    )


if __name__ == "__main__":
    main()
