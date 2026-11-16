"""Synchronous DiLoCo rounds, independent of models, execution and
transport."""

import copy
import math
import secrets
import threading
from dataclasses import dataclass

import numpy as np


def copy_tensors(tensors, expected=None):
    """Validate a complete named state before accepting any of it."""
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("Expected a nonempty tensor dictionary")
    if expected is not None and tensors.keys() != expected.keys():
        raise ValueError("Tensor names do not match the model")
    if any(not isinstance(name, str) or not name for name in tensors):
        raise ValueError("Tensor names must be nonempty strings")
    result = {}
    for name in sorted(tensors):
        value = np.asarray(tensors[name], dtype=np.float64)
        if value.ndim != 2 or not value.size or not np.isfinite(value).all():
            raise ValueError(
                f"Expected a finite, nonempty rank-two tensor: {name}"
            )
        if expected is not None and value.shape != expected[name].shape:
            raise ValueError(f"Tensor shape mismatch: {name}")
        result[name] = value.copy()
    return result


def describe_tensors(tensors):
    return {
        name: dict(shape=list(w.shape), data=w.ravel().tolist())
        for name, w in tensors.items()
    }


@dataclass(frozen=True)
class DiLoCoConfig:
    local_steps: int = 100
    outer_learning_rate: float = 0.7
    momentum: float = 0.9
    nesterov: bool = True

    def __post_init__(self):
        if type(self.local_steps) is not int or self.local_steps < 1:
            raise ValueError("local_steps must be a positive integer")
        if any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in (self.outer_learning_rate, self.momentum)
        ):
            raise ValueError("Outer optimizer settings must be finite numbers")
        if self.outer_learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid outer learning rate or momentum")
        if type(self.nesterov) is not bool or (
            self.nesterov and not self.momentum
        ):
            raise ValueError("Nesterov requires positive momentum")


@dataclass(frozen=True)
class Assignment:
    round_id: str
    version: int
    worker: str
    steps: int
    weights: dict

    def describe(self):
        return dict(
            round_id=self.round_id,
            version=self.version,
            worker=self.worker,
            steps=self.steps,
            weights=describe_tensors(self.weights),
        )


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    version: int
    worker: str
    steps: int
    pseudo_gradient: dict
    losses: tuple

    def describe(self):
        return dict(
            round_id=self.round_id,
            version=self.version,
            worker=self.worker,
            steps=self.steps,
            losses=list(self.losses),
            pseudo_gradient=describe_tensors(self.pseudo_gradient),
        )

    @classmethod
    def from_description(cls, data):
        """Decode a browser result; submit() performs model/protocol
        validation."""
        gradient = {
            name: np.asarray(value["data"], dtype=np.float64).reshape(
                value["shape"]
            )
            for name, value in data["pseudo_gradient"].items()
        }
        return cls(
            data["round_id"],
            data["version"],
            data["worker"],
            data["steps"],
            gradient,
            tuple(data["losses"]),
        )


class Coordinator:
    """Full replicas synchronize weights only; inner moments stay at the
    worker.

    Membership is fixed for an open round. Results may arrive in any order, but
    are averaged in sorted worker order. A commit publishes all weights and the
    outer momentum together. This is an in-memory, trusted-participant backend;
    result validation here checks the protocol, not the claimed computation.
    """

    def __init__(self, weights, config=None):
        self.config = config or DiLoCoConfig()
        self._weights = copy_tensors(weights)
        self._velocity = {
            name: np.zeros_like(w) for name, w in self._weights.items()
        }
        self.version = 0
        self.history = []
        self._round_id = None
        self._workers = ()
        self._results = {}
        self._lock = threading.RLock()

    @property
    def weights(self):
        with self._lock:
            return copy_tensors(self._weights)

    def checkpoint(self):
        """Snapshot committed global weights and outer optimizer state."""
        with self._lock:
            if self._round_id is not None:
                raise ValueError("Cannot checkpoint an open round")
            return dict(
                weights=self.weights,
                velocity=copy_tensors(self._velocity),
                version=self.version,
                history=copy.deepcopy(self.history),
            )

    def restore(self, state):
        with self._lock:
            if self._round_id is not None:
                raise ValueError("Cannot restore an open round")
            weights = copy_tensors(state["weights"], self._weights)
            velocity = copy_tensors(state["velocity"], weights)
            if type(state["version"]) is not int or state["version"] < 0:
                raise ValueError("Invalid checkpoint version")
            self._weights, self._velocity = weights, velocity
            self.version = state["version"]
            self.history = copy.deepcopy(state["history"])

    @property
    def pending(self):
        with self._lock:
            return tuple(w for w in self._workers if w not in self._results)

    def begin_round(self, workers):
        workers = tuple(workers)
        if (
            not workers
            or any(not isinstance(w, str) or not w for w in workers)
            or len(set(workers)) != len(workers)
        ):
            raise ValueError("A round needs distinct, nonempty worker IDs")
        with self._lock:
            if self._round_id is not None:
                raise ValueError("Commit or abort the open round first")
            self._workers = tuple(sorted(workers))
            self._round_id = secrets.token_hex(16)
            self._results = {}

    def assignment(self, worker):
        with self._lock:
            if self._round_id is None or worker not in self._workers:
                raise ValueError("Worker is not a member of an open round")
            return Assignment(
                self._round_id,
                self.version,
                worker,
                self.config.local_steps,
                copy_tensors(self._weights),
            )

    def submit(self, result):
        with self._lock:
            if (
                self._round_id is None
                or result.round_id != self._round_id
                or result.version != self.version
            ):
                raise ValueError("Stale or foreign round result")
            if (
                result.worker not in self._workers
                or result.worker in self._results
            ):
                raise ValueError("Unknown worker or duplicate round result")
            if (
                type(result.steps) is not int
                or result.steps != self.config.local_steps
            ):
                raise ValueError(
                    "Result must complete exactly local_steps updates"
                )
            losses = np.asarray(result.losses, dtype=np.float64)
            if (
                losses.shape != (result.steps,)
                or not np.isfinite(losses).all()
            ):
                raise ValueError("Expected one finite loss per local step")
            gradient = copy_tensors(result.pseudo_gradient, self._weights)
            self._results[result.worker] = RoundResult(
                result.round_id,
                result.version,
                result.worker,
                result.steps,
                gradient,
                tuple(map(float, losses)),
            )

    def commit_round(self):
        with self._lock:
            if self._round_id is None or self.pending:
                raise ValueError(
                    "Every member must finish before committing a round"
                )
            weights, velocity = {}, {}
            cfg = self.config
            # Compute into fresh dictionaries: overflow or invalid data cannot
            # leave a partially updated model or momentum buffer.
            with np.errstate(over="raise", invalid="raise"):
                for name, before in self._weights.items():
                    mean = np.zeros_like(before)
                    for worker in self._workers:
                        mean += self._results[worker].pseudo_gradient[
                            name
                        ] / len(self._workers)
                    v = cfg.momentum * self._velocity[name] + mean
                    direction = mean + cfg.momentum * v if cfg.nesterov else v
                    weights[name] = (
                        before - cfg.outer_learning_rate * direction
                    )
                    velocity[name] = v
            # Keep the membership and per-worker losses, never graph
            # intermediates.
            record = dict(
                version=self.version + 1,
                round_id=self._round_id,
                workers=list(self._workers),
                local_steps=cfg.local_steps,
                losses={
                    w: list(self._results[w].losses) for w in self._workers
                },
            )
            self._weights, self._velocity = weights, velocity
            self.version += 1
            self.history.append(record)
            self._round_id, self._workers, self._results = None, (), {}
            return dict(record)

    def abort_round(self):
        """Drop submissions without changing global weights, momentum or
        version.

        Workers that already trained have advanced their local moments. Their
        owners must decide whether to retain those or restore a local
        checkpoint.
        """
        with self._lock:
            self._round_id, self._workers, self._results = None, (), {}

    def extend_parameters(self, weights):
        """Add parameters between rounds, preserving old weights and
        momentum."""
        with self._lock:
            if self._round_id is not None:
                raise ValueError(
                    "Commit or abort the open round before growth"
                )
            candidate = copy_tensors(weights)
            if not self._weights.keys() < candidate.keys():
                raise ValueError(
                    "Growth must strictly extend the parameter schema"
                )
            for name, before in self._weights.items():
                if candidate[name].shape != before.shape or not np.array_equal(
                    candidate[name], before
                ):
                    raise ValueError(
                        f"Growth changed existing parameter: {name}"
                    )
            added = sorted(candidate.keys() - self._weights.keys())
            velocity = {
                name: self._velocity[name].copy()
                if name in self._velocity
                else np.zeros_like(w)
                for name, w in candidate.items()
            }
            self._weights, self._velocity = candidate, velocity
            self.version += 1
            record = dict(
                event="parameter_growth", version=self.version, added=added
            )
            self.history.append(record)
            return dict(record)
