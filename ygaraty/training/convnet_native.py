"""Torch-free adapter for the legacy CIFAR ConvNet wire layout.

Uses Ygaraty's existing portable ConvNet graph and NumPy task kernels. Wire
names stay compatible with historical checkpoints; head weights are transposed
when crossing into the portable graph. Optimizer moments stay in Trainer.
"""

from dataclasses import replace

import numpy as np

from ygaraty.runtime.local import execute_plan
from ygaraty.training.convnet_webgpu import ConvBatch, ConvNetRecipe
from ygaraty.training.coordinator import copy_tensors
from ygaraty.training.trainer import Trainer

NAMES = {
    "0.weight": "conv0.weight",
    "0.bias": "conv0.bias",
    "3.weight": "conv1.weight",
    "3.bias": "conv1.bias",
    "6.weight": "conv2.weight",
    "6.bias": "conv2.bias",
    "10.weight": "head.weight",
    "10.bias": "head.bias",
}


def to_portable(weights, recipe):
    weights = copy_tensors(weights)
    if weights.keys() != NAMES.keys():
        raise ValueError("Unexpected ConvNet weight names")
    result = {}
    for old, new in NAMES.items():
        shape = recipe.specs[new]["shape"]
        if weights[old].shape != (1, int(np.prod(shape))):
            raise ValueError(f"Invalid ConvNet shape: {old}")
        value = weights[old]
        result[new] = (
            value.reshape(shape[::-1]).T.copy()
            if new == "head.weight"
            else value.reshape(shape).copy()
        )
    return result


def to_wire(weights):
    return {
        old: (weights[new].T if new == "head.weight" else weights[new])
        .reshape(1, -1)
        .copy()
        for old, new in NAMES.items()
    }


def initial_weights(seed):
    """NumPy-seeded initialization, not bitwise Torch RNG reproduction."""
    return to_wire(ConvNetRecipe().initial_weights(seed=seed))


class Replica:
    def __init__(self, worker, weights, learning_rate):
        self.recipe = ConvNetRecipe(learning_rate=learning_rate, max_batch=256)
        to_portable(weights, self.recipe)
        self.trainer = Trainer(worker, self.recipe)

    def run_round(self, assignment, batches):
        local = replace(
            assignment, weights=to_portable(assignment.weights, self.recipe)
        )
        result = self.trainer.run_round(local, batches)
        return replace(result, pseudo_gradient=to_wire(result.pseudo_gradient))


def logits(weights, images):
    """Forward-only evaluation; optimizer/backward tasks are never executed."""
    from ygaraty.inference.shared import ancestors

    recipe = ConvNetRecipe()
    state = to_portable(weights, recipe)
    images = np.asarray(images).reshape(-1, 3, 32, 32)
    batch = ConvBatch(images, np.zeros(len(images), dtype=np.int64))
    plan = recipe.compile(batch, 1)
    needed = ancestors(plan.tasks, plan.logits)
    tasks = [task for task in plan.tasks if task.id in needed]
    inputs = dict(plan.constants)
    inputs.update(
        {plan.parameters[name]: value for name, value in state.items()}
    )
    return execute_plan(tasks, inputs, [plan.logits])[plan.logits]
