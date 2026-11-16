"""A full local trainer for one DiLoCo participant; no per-task
communication."""

import numpy as np

from ygaraty.runtime.local import execute_plan
from ygaraty.training.coordinator import RoundResult, copy_tensors


class Trainer:
    def __init__(self, worker, recipe):
        self.worker, self.recipe = worker, recipe
        self.step = 0
        self.weights = None
        self.moments = None
        self._seen_rounds = set()
        self._version = -1

    def run_round(self, assignment, batches):
        """Rebase weights, retain Adam moments/clock, train H steps, return
        θ₀−θH.

        A failed local round leaves this trainer's previous state intact. The
        batches are supplied by the caller, so SQL/sharding policies stay
        outside
        the optimizer. An iterator need only hold one batch at a time.
        """
        if (
            assignment.worker != self.worker
            or assignment.round_id in self._seen_rounds
            or assignment.version < self._version
        ):
            raise ValueError("Foreign or already executed assignment")
        if type(assignment.steps) is not int or assignment.steps < 1:
            raise ValueError("A round requires a positive step count")
        weights = copy_tensors(assignment.weights)
        if weights.keys() != self.recipe.specs.keys() or any(
            w.shape != tuple(self.recipe.specs[n]["shape"])
            for n, w in weights.items()
        ):
            raise ValueError(
                "Assignment model does not match the local recipe"
            )
        base = weights
        moments = (
            self.moments
            if self.moments is not None
            else {
                kind: {name: np.zeros_like(w) for name, w in weights.items()}
                for kind in ("m", "v")
            }
        )
        iterator, losses = iter(batches), []
        for offset in range(1, assignment.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                raise ValueError(
                    "Not enough batches for the assigned round"
                ) from None
            plan = self.recipe.compile(batch, self.step + offset)
            inputs = dict(plan.constants)
            for name, p in plan.parameters.items():
                inputs[p] = weights[name]
                for kind in ("m", "v"):
                    inputs[f"adam_{kind}_{p}"] = moments[kind][name]
            values = execute_plan(
                plan.tasks, inputs, [plan.loss, *plan.updates.values()]
            )
            losses.append(float(values[plan.loss][0, 0]))
            weights = {
                name: values[plan.updates[p]]
                for name, p in plan.parameters.items()
            }
            moments = {
                kind: {
                    name: values[plan.updates[f"adam_{kind}_{p}"]]
                    for name, p in plan.parameters.items()
                }
                for kind in ("m", "v")
            }
        gradient = {name: base[name] - weights[name] for name in base}
        if not all(np.isfinite(a).all() for a in gradient.values()):
            raise ValueError("Nonfinite pseudo-gradient")
        self.weights, self.moments = weights, moments
        self.step += assignment.steps
        self._seen_rounds.add(assignment.round_id)
        self._version = assignment.version
        return RoundResult(
            assignment.round_id,
            assignment.version,
            self.worker,
            assignment.steps,
            gradient,
            tuple(losses),
        )
