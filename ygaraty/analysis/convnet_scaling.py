"""Offline comparison of experiment-45 reports; no model or execution
imports."""


def comparison_lines(cases):
    """Validate completed, matching workloads and format the existing CLI
    table.

    Reports retain their original timing boundaries. This does not run a
    benchmark
    or infer hardware equivalence; comparison policy is unchanged from
    experiment 45.
    """
    if any(c["state"] != "completed" for c in cases):
        raise ValueError("Only completed cases can be compared")
    keys = (
        "local_steps",
        "measured_rounds",
        "warmup_rounds",
        "batch",
        "seed",
        "tiny",
        "model",
        "data",
        "allow_shared_host",
    )

    def identity(c):
        return (
            {k: c["config"][k] for k in keys},
            c["metadata"]["data_sha256"],
            c["metadata"].get("source_sha256"),
        )

    if any(identity(c) != identity(cases[0]) for c in cases):
        raise ValueError(
            "Cases have different workloads; compare each "
            "H/batch/dataset separately"
        )
    lines = [
        "backend/device/dtype | replicas | images/sec | measured "
        "seconds | scaling vs same-backend 1 host"
    ]
    for c in cases:
        m = next(iter(c["members"].values()))
        label = "/".join(
            (
                c["config"]["backend"],
                m.get("device", "GPU"),
                m.get("dtype", "mixed"),
            )
        )
        baseline = next(
            (
                b
                for b in cases
                if b["config"]["replicas"] == 1
                and b["config"]["backend"] == c["config"]["backend"]
                and all(
                    next(iter(b["members"].values())).get(k) == m.get(k)
                    for k in ("device", "dtype")
                )
            ),
            None,
        )
        scaling = (
            f"{c['images_per_second'] / baseline['images_per_second']:.2f}x"
            if baseline
            else "n/a"
        )
        lines.append(
            f"{label} | {c['config']['replicas']} | "
            f"{c['images_per_second']:.2f} | "
            f"{c['measured_seconds']:.3f} | {scaling}"
        )
    return lines
