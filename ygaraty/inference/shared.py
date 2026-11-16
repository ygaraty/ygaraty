"""Bounded forward-only evaluation of validated inline Shared artifacts."""

import math

import numpy as np

from ..compiler.plan import Input, Task
from .forward import select_tasks


def ancestors(tasks, output):
    return select_tasks(tasks, [output])[1]


def forward_tasks(bundle):
    if bundle["recipe"]["model"] == "tape":
        return [
            Task(
                n["id"],
                n["op"],
                [Input(i) for i in n["inputs"]],
                tuple(n["shape"]),
                n["attrs"],
            )
            for n in bundle["recipe"]["tape"]["nodes"]
        ]
    return [
        Task(
            t["id"],
            t["op"],
            [Input(**i) for i in t["inputs"]],
            tuple(t["shape"]),
            t["attrs"],
        )
        for t in bundle["tasks"]
    ]


def forward_plan(bundle, output):
    if any(s["kind"] == "external" for s in bundle["initializers"].values()):
        raise ValueError("Inference currently requires inline Shared inputs")
    tasks = forward_tasks(bundle)
    forward = ancestors(tasks, bundle["loss"])
    if not isinstance(output, str) or output not in forward:
        raise ValueError(
            "Output must name a forward tensor contributing to the loss"
        )
    selected, needed = select_tasks(tasks, [output])
    elements = sum(math.prod(t.shape) for t in selected)
    elements += sum(
        math.prod(s["shape"])
        for n, s in bundle["initializers"].items()
        if n in needed
    )
    if len(selected) > 4096 or elements * 8 > 64 << 20:
        raise ValueError(
            "Inference exceeds the 4,096-task or 64 MiB tensor budget"
        )
    return selected, needed


def input_values(bundle, weights, inputs, needed):
    from ..models.synthetic import initial_block, regions

    specs = bundle["initializers"]
    if not isinstance(inputs, dict) or set(inputs) - (
        (set(specs) & needed) - set(weights)
    ):
        raise ValueError("Inputs must name non-parameter initializers")
    values = {}
    config = bundle["recipe"]
    for name in needed & specs.keys():
        spec = specs[name]
        if name in weights or name in inputs:
            value = np.asarray(
                weights[name] if name in weights else inputs[name],
                dtype=np.float64,
            )
        else:
            value = np.empty(spec["shape"], dtype=np.float64)
            for region in regions(spec["shape"], config["tile_size"]):
                i, ei, j, ej = region
                value[i:ei, j:ej] = initial_block(config, name, spec, region)
        if value.shape != tuple(spec["shape"]) or not np.isfinite(value).all():
            raise ValueError("Invalid inference input: " + name)
        values[name] = value.copy()
    return values


def predict(bundle, weights, inputs, output):
    from ..runtime.local import execute_plan

    tasks, needed = forward_plan(bundle, output)
    values = input_values(bundle, weights, inputs, needed)
    result = execute_plan(tasks, values, {output})[output]
    if not np.isfinite(result).all():
        raise ValueError("Inference produced nonfinite values")
    return result.copy()
