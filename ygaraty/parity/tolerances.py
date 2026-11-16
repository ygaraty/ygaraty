"""Named acceptance profiles, fixed before a run. Never loosen one to pass a
run.

Each kind maps to (rtol, atol) with numpy.allclose semantics:
|actual - expected| <= atol + rtol * |expected|.
`update_delta` compares (after - before) so large weights cannot hide a poor
step.
"""

KINDS = (
    "diagnostic",
    "logits",
    "loss",
    "gradient",
    "update_delta",
    "weight",
    "m",
    "v",
)

PROFILES = {
    # The documented SQL/browser and 100-update acceptance tolerance.
    "float64": {kind: (2e-7, 2e-9) for kind in KINDS},
    # Thresholds of the architecture unit tests (test_lstm_translation). Those
    # tests
    # check Adam only as a delta, so `weight` keeps the documented float64
    # budget.
    "float64-tight": dict(
        diagnostic=(1e-10, 1e-12),
        logits=(1e-10, 1e-12),
        loss=(1e-10, 1e-12),
        gradient=(1e-9, 1e-12),
        update_delta=(2e-7, 2e-9),
        weight=(2e-7, 2e-9),
        m=(1e-9, 1e-12),
        v=(1e-9, 1e-12),
    ),
    # Budgets fixed for the WebGPU float32-matmul experiment
    # (webapp/translation.py).
    "webgpu": dict(
        diagnostic=(2e-4, 2e-4),
        logits=(2e-4, 2e-4),
        loss=(2e-5, 2e-5),
        gradient=(2e-3, 2e-6),
        update_delta=(1e-2, 1e-6),
        weight=(1e-2, 1e-6),
        m=(2e-3, 2e-7),
        v=(1e-2, 2e-8),
    ),
}


def profile(name):
    if name not in PROFILES:
        raise ValueError(
            f"Unknown tolerance profile {name!r}; choose from "
            f"{sorted(PROFILES)}"
        )
    return {
        kind: dict(rtol=r, atol=a) for kind, (r, a) in PROFILES[name].items()
    }
