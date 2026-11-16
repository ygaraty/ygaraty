"""Summarize two completed, independently verified browser precision trials."""

import argparse
import json
import math
from pathlib import Path


def summarize(path):
    report = json.loads(Path(path).read_text())
    groups = {}
    for label, check in report["checks"].items():
        kind = check["kind"]
        g = groups.setdefault(
            kind,
            dict(
                tensors=0,
                failed_tensors=0,
                strict_failed_tensors=0,
                elements=0,
                violations=0,
                max_abs_error=0.0,
                error_squared=0.0,
                reference_squared=0.0,
                actual_squared=0.0,
                dot=0.0,
            ),
        )
        g["tensors"] += 1
        g["failed_tensors"] += int(not check["passed"])
        g["strict_failed_tensors"] += int(not check["strict_float64_passed"])
        g["max_abs_error"] = max(g["max_abs_error"], check["max_abs_error"])
        for key in (
            "elements",
            "violations",
            "error_squared",
            "reference_squared",
            "actual_squared",
            "dot",
        ):
            g[key] += check[key]
    for g in groups.values():
        g["relative_l2"] = (
            math.sqrt(g["error_squared"] / g["reference_squared"])
            if g["reference_squared"]
            else None
        )
        norm = math.sqrt(g["actual_squared"] * g["reference_squared"])
        g["cosine_similarity"] = g["dot"] / norm if norm else None
    return dict(
        report_path=str(Path(path).resolve()),
        precision=report["precision"],
        gpu_arithmetic=report.get("gpu_arithmetic", "float32"),
        passed=report["passed"],
        active_seconds=report["active_seconds"],
        worker_compute_seconds=sum(
            w["compute_ms"] for w in report["workers"].values()
        )
        / 1000,
        transferred_bytes=report["sent_bytes"] + report["received_bytes"],
        gpu_operations=report["gpu_operations"],
        browser_operations=report["browser_operations"],
        workers=len(report["workers"]),
        retries=report["retries"],
        rejected=report["rejected"],
        loss=report["losses"][0],
        groups=groups,
        failed_checks={
            k: v for k, v in report["checks"].items() if not v["passed"]
        },
    ), report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--float64", required=True)
    p.add_argument("--webgpu", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cpu, cr = summarize(args.float64)
    gpu, gr = summarize(args.webgpu)
    for key in (
        "checkpoint_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
        "config",
        "optimizer",
        "label_smoothing",
    ):
        if cr["reference"][key] != gr["reference"][key]:
            raise ValueError("Mismatched reference " + key)
    if cpu["workers"] != gpu["workers"]:
        raise ValueError("Mismatched worker conditions")
    gpu_plan = json.loads((Path(args.webgpu).parent / "plan.json").read_text())
    if gpu["gpu_operations"] != sum(t["op"] == "matmul" for t in gpu_plan):
        raise ValueError("Not every matmul ran on GPU")
    if cpu["gpu_operations"] or not gpu["gpu_operations"]:
        raise ValueError("Missing actual GPU/CPU contrast")
    result = dict(
        float64=cpu,
        webgpu=gpu,
        webgpu_wall_speedup=cpu["active_seconds"] / gpu["active_seconds"],
        webgpu_compute_speedup=cpu["worker_compute_seconds"]
        / gpu["worker_compute_seconds"],
        caveats=[
            "One local trial per mode; not a statistically stable benchmark",
            "Two browser workers share this machine and its GPU",
            "Active time includes transport, scheduling, "
            "full numerical verification and report work; "
            "excludes creation/waiting for contributors",
            "Worker compute includes GPU buffer transfer, "
            "dispatch and readback, not isolated shader time",
            "WebGPU mode uses float32 matmul; other kernels "
            "and task transport remain float64",
            "Gradients and Adam update are compared with the "
            "same independent PyTorch float64 oracle",
        ],
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("float64", "webgpu")
            },
            indent=2,
        )
    )
    for mode, value in [("float64", cpu), ("webgpu", gpu)]:
        print(
            mode,
            value["active_seconds"],
            value["worker_compute_seconds"],
            {
                k: (v["max_abs_error"], v["failed_tensors"])
                for k, v in value["groups"].items()
            },
        )


if __name__ == "__main__":
    main()
