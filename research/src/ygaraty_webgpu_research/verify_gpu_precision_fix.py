"""Audit the precision fix against the original budgets and unchanged
oracle."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    for key in (
        "checkpoint_sha256",
        "tokenizer_sha256",
        "fixture_sha256",
        "config",
        "optimizer",
        "label_smoothing",
    ):
        assert before["reference"][key] == after["reference"][key], key
    assert before["checks"].keys() == after["checks"].keys()
    for name in before["checks"]:
        for key in ("rtol", "atol", "shape", "comparison"):
            assert before["checks"][name][key] == after["checks"][name][key], (
                name,
                key,
            )
    plan = json.loads((Path(args.after).parent / "plan.json").read_text())
    assert after["gpu_operations"] == sum(t["op"] == "matmul" for t in plan)
    assert after["gpu_arithmetic"] == "compensated"
    assert after["passed"] and all(
        c["passed"] for c in after["checks"].values()
    )
    assert not after["retries"] and not after["rejected"]
    groups = {}
    for kind in ("logits", "loss", "gradient", "m", "v", "updated"):
        groups[kind] = {}
        for mode, report in [("before", before), ("after", after)]:
            checks = [
                c for c in report["checks"].values() if c["kind"] == kind
            ]
            groups[kind][mode] = dict(
                max_abs_error=max(c["max_abs_error"] for c in checks),
                failed_tensors=sum(not c["passed"] for c in checks),
                violations=sum(c["violations"] for c in checks),
                strict_float64_failures=sum(
                    not c["strict_float64_passed"] for c in checks
                ),
            )
    worst = max(
        (c for c in before["checks"].values() if c["kind"] == "updated"),
        key=lambda c: c["max_abs_error"],
    )
    result = dict(
        passed=True,
        tolerances_unchanged=True,
        reference_unchanged=True,
        gpu_operations=after["gpu_operations"],
        browser_operations=after["browser_operations"],
        total_checks=len(after["checks"]),
        strict_float64_passes=sum(
            c["strict_float64_passed"] for c in after["checks"].values()
        ),
        groups=groups,
        diagnostic_example=worst["worst_element"],
        before_adam_replay_max_abs_error=max(
            c["optimizer_replay_max_abs_error"]
            for c in before["checks"].values()
            if c["kind"] == "updated"
        ),
        before_report=str(Path(args.before).resolve()),
        after_report=str(Path(args.after).resolve()),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
