"""CIFAR replica configuration and initial model state.

The MLP and legacy ConvNet use the existing recipes unchanged. Importing this
module does not initialize a model, import PyTorch or execute any workload.
"""

import math


def validate_spec(spec):
    """Copy and validate a job specification before allocating its model.

    This adapter also serves research drivers: one-replica Adam controls,
    smaller MLPs and ConvNet remain accepted. Release readiness is separate
    from accepting a recipe; no durability or resource guarantee is implied.
    """
    spec = dict(spec)
    model = spec.get("model", "mlp")
    if model not in ("mlp", "convnet"):
        raise ValueError("CIFAR model must be mlp or convnet")
    for key in ("replicas", "rounds", "batch", "local_steps", "tile"):
        value = spec[key]
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(spec["seed"]) is not int or spec["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    every = spec.get("eval_every", 1)
    if type(every) is not int or every < 1:
        raise ValueError("eval_every must be a positive integer")
    rate = spec["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if model == "mlp":
        hidden = spec["hidden"]
        if not isinstance(hidden, (list, tuple)) or any(
            type(width) is not int or width < 1 for width in hidden
        ):
            raise ValueError("hidden must contain positive integer widths")
        spec["hidden"] = list(hidden)
    return spec


def initial_state(spec):
    """Return the worker recipe and initial weights, without training."""
    # Pixels travel as bytes, eight per float64 of the binary frame.
    if spec.get("model") == "convnet":
        from ygaraty.training.convnet_native import initial_weights

        weights = initial_weights(int(spec["seed"]))
        recipe = dict(
            kind="convnet",
            learning_rate=float(spec["learning_rate"]),
            pixels="packed-u8",
        )
    else:
        from ygaraty.training.mlp import MLPRecipe

        model = dict(inputs=3072, hidden=list(spec["hidden"]), classes=10)
        recipe = dict(
            kind="mlp",
            config=model,
            learning_rate=float(spec["learning_rate"]),
            tile_size=int(spec["tile"]),
            pixels="packed-u8",
        )
        mlp = MLPRecipe(
            model,
            learning_rate=recipe["learning_rate"],
            tile_size=recipe["tile_size"],
        )
        weights = mlp.initial_weights(seed=int(spec["seed"]))
    return recipe, weights
