"""A full local trainer for one DiLoCo participant; no per-task
communication."""

import numpy as np

from ygaraty.runtime.local import execute_plan
from ygaraty.training.coordinator import (
    RoundResult,
    copy_tensors,
    describe_tensors,
)


def optimizer_state(state, expected, step, moment_names=None):
    """Validate complete portable optimizer state at a round boundary."""
    if (
        not isinstance(state, dict)
        or type(state.get("step")) is not int
        or state["step"] < 0
        or state["step"] != step
    ):
        raise ValueError("Wrong optimizer checkpoint step")
    moments = state.get("moments")
    if step == 0 and moments is None:
        return None
    if not isinstance(moments, dict) or set(moments) != {"m", "v"}:
        raise ValueError("Incomplete optimizer checkpoint")
    decoded = {}
    for kind in ("m", "v"):
        if not isinstance(moments[kind], dict) or any(
            not isinstance(v, dict) for v in moments[kind].values()
        ):
            raise ValueError("Malformed optimizer moments")
        decoded[kind] = copy_tensors(
            {
                n: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"])
                for n, v in moments[kind].items()
            },
            {n: expected[n] for n in moment_names[kind]}
            if moment_names
            else expected,
        )
    if any(np.any(v < 0) for v in decoded["v"].values()):
        raise ValueError("Negative Adam variance")
    return decoded


class Trainer:
    def __init__(self, worker, recipe):
        self.worker, self.recipe = worker, recipe
        self.step = 0
        self.weights = None
        self.moments = None
        self._seen_rounds = set()
        self._version = -1

    def checkpoint(self):
        return dict(
            step=self.step,
            moments=None
            if self.moments is None
            else {
                kind: describe_tensors(values)
                for kind, values in self.moments.items()
            },
        )

    def restore(self, state, weights):
        """Restore a committed optimizer; the next assignment rebases weights."""
        moments = optimizer_state(
            state,
            weights,
            state["step"],
            self.recipe.moment_names()
            if hasattr(self.recipe, "moment_names")
            else None,
        )
        self.step, self.moments = state["step"], moments
        self.weights = None
        self._seen_rounds = set()
        self._version = -1

    def run_round(self, assignment, batches):
        """Rebase weights, retain optimizer moments/clock, train H steps, return
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
                kind: {
                    name: np.zeros_like(weights[name])
                    for name in (
                        self.recipe.moment_names()[kind]
                        if hasattr(self.recipe, "moment_names")
                        else weights
                    )
                }
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
            state_parameters = plan.state_parameters or {
                kind: {
                    n: f"adam_{kind}_{p}" for n, p in plan.parameters.items()
                }
                for kind in ("m", "v")
            }
            for kind, mapping in state_parameters.items():
                for name, ref in mapping.items():
                    inputs[ref] = moments[kind][name]
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
                    name: values[plan.updates[ref]]
                    for name, ref in state_parameters[kind].items()
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
