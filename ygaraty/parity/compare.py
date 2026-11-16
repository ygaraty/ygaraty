"""One comparison primitive, with metrics that stay meaningful across runs.

`ratio` is the worst |error| / (atol + rtol*|expected|). Below 1 passes; its
trend over steps and over code changes is the headroom we track.
"""

import numpy as np


def check(name, kind, actual, expected, tolerance, before=None):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        actual = actual.reshape(expected.shape)
    if before is not None:
        # Both sides subtract the state each one started from.
        actual = actual - before[0]
        expected = expected - before[1]
    rtol, atol = tolerance["rtol"], tolerance["atol"]
    error = np.abs(actual - expected)
    allowed = atol + rtol * np.abs(expected)
    finite = bool(np.isfinite(actual).all())
    ratios = error / allowed
    worst = int(np.argmax(ratios)) if ratios.size else 0
    reference_norm = float(np.linalg.norm(expected))
    actual_norm = float(np.linalg.norm(actual))
    dot = float(np.sum(actual * expected))
    return dict(
        name=name,
        kind=kind,
        shape=list(expected.shape),
        elements=int(expected.size),
        passed=finite and bool(np.all(error <= allowed)),
        violations=int(np.count_nonzero(~(error <= allowed))),
        ratio=float(ratios.max()) if ratios.size else 0.0,
        max_abs=float(error.max()) if error.size else 0.0,
        relative_l2=float(np.linalg.norm(error) / reference_norm)
        if reference_norm
        else None,
        cosine=dot / (actual_norm * reference_norm)
        if actual_norm and reference_norm
        else None,
        worst=dict(
            index=[int(i) for i in np.unravel_index(worst, expected.shape)],
            actual=float(actual.flat[worst]),
            expected=float(expected.flat[worst]),
        )
        if ratios.size
        else None,
        rtol=rtol,
        atol=atol,
    )


def summarize(checks):
    """Aggregate per kind: worst ratio, its tensor, and totals."""
    kinds = {}
    for c in checks:
        k = kinds.setdefault(
            c["kind"],
            dict(
                tensors=0,
                failed=0,
                elements=0,
                violations=0,
                ratio=0.0,
                max_abs=0.0,
                worst_tensor=None,
            ),
        )
        k["tensors"] += 1
        k["failed"] += not c["passed"]
        k["elements"] += c["elements"]
        k["violations"] += c["violations"]
        k["max_abs"] = max(k["max_abs"], c["max_abs"])
        if k["worst_tensor"] is None or c["ratio"] > k["ratio"]:
            k["ratio"], k["worst_tensor"] = c["ratio"], c["name"]
    return kinds
