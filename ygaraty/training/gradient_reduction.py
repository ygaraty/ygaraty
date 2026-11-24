"""Experimental synchronous gradient reduction before a single shared Adam step.

Leaves compute gradients of mean loss. Pairwise reduction carries sample counts
so unequal partitions remain equivalent to the unsplit global batch. No worker
performs a local optimizer update. Reduction topology is independent of transport.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from ygaraty.compiler.optimizer import with_optimizer
from ygaraty.inference.shared import ancestors
from ygaraty.runtime.local import execute_plan
from .coordinator import copy_tensors


@dataclass(frozen=True)
class GradientPart:
    samples: int
    loss: float
    gradients: dict


def gradient_plan(recipe, batch):
    """Keep only Tape forward/backward tasks; updates names the gradient outputs."""
    plan = recipe.compile(batch, 1)
    outputs = {name: plan.gradients[p] for name, p in plan.parameters.items()}
    needed = set().union(
        *(
            ancestors(plan.tasks, output)
            for output in [plan.loss, *outputs.values()]
        )
    )
    return replace(
        plan, tasks=[t for t in plan.tasks if t.id in needed], updates=outputs
    )


def compute_gradients(recipe, weights, batch):
    plan = gradient_plan(recipe, batch)
    inputs = dict(plan.constants)
    inputs.update({plan.parameters[n]: w for n, w in weights.items()})
    values = execute_plan(
        plan.tasks, inputs, [plan.loss, *plan.updates.values()]
    )
    return GradientPart(
        len(batch.images),
        float(values[plan.loss][0, 0]),
        {n: values[ref] for n, ref in plan.updates.items()},
    )


def reduce_gradients(parts):
    """Deterministic adjacent-pair tree, carrying an odd tail unchanged.

    All leaves must be evaluated at the same model version. The transport owns
    membership/version checks. Reduction validates every tensor before merging.
    """
    level = []
    expected = None
    for part in parts:
        if (
            type(part.samples) is not int
            or part.samples < 1
            or not math.isfinite(part.loss)
        ):
            raise ValueError("Invalid gradient sample count or loss")
        gradients = copy_tensors(part.gradients, expected)
        expected = gradients
        level.append(GradientPart(part.samples, float(part.loss), gradients))
    if not level:
        raise ValueError("Cannot reduce an empty gradient tree")
    with np.errstate(over="raise", invalid="raise"):
        while len(level) > 1:
            merged = []
            for index in range(0, len(level), 2):
                left = level[index]
                if index + 1 == len(level):
                    merged.append(left)
                    continue
                right = level[index + 1]
                count = left.samples + right.samples
                a, b = left.samples / count, right.samples / count
                gradients = copy_tensors(
                    {
                        n: a * g + b * right.gradients[n]
                        for n, g in left.gradients.items()
                    }
                )
                loss = a * left.loss + b * right.loss
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite reduced loss")
                merged.append(GradientPart(count, loss, gradients))
            level = merged
    return level[0]


class SharedAdam:
    """One optimizer clock and moments, using the existing compiler/runtime."""

    def __init__(self, recipe, weights):
        self.recipe = recipe
        self.weights = copy_tensors(weights)
        self.step = 0
        self.moments = {
            kind: {n: np.zeros_like(w) for n, w in self.weights.items()}
            for kind in ("m", "v")
        }

    def apply(self, gradients):
        gradients = copy_tensors(gradients, self.weights)
        inputs = dict(self.weights)
        refs = {n: f"gradient_{i}" for i, n in enumerate(self.weights)}
        inputs.update({refs[n]: g for n, g in gradients.items()})
        for kind, values in self.moments.items():
            inputs.update(
                {f"adam_{kind}_{n}": value for n, value in values.items()}
            )
        tasks, updates = with_optimizer(
            [],
            {n: w.shape for n, w in inputs.items()},
            list(self.weights),
            refs,
            self.recipe.tile_size,
            self.recipe.learning_rate,
            "adam",
            self.step + 1,
            self.recipe.optimizer,
        )
        values = execute_plan(tasks, inputs, updates.values())
        weights = copy_tensors(
            {n: values[updates[n]] for n in self.weights}, self.weights
        )
        moments = {
            kind: copy_tensors(
                {n: values[updates[f"adam_{kind}_{n}"]] for n in self.weights},
                self.weights,
            )
            for kind in ("m", "v")
        }
        self.weights, self.moments = weights, moments
        self.step += 1
