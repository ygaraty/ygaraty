"""Fixed-membership network DiLoCo with durable, transactional round commits.

The service trusts admitted contributors' numerical results. Authentication and
shape checks prevent protocol mistakes; they do not prove correct computation.
"""

import copy
import hashlib
import json
import math
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import recovery
from .convnet_webgpu import ConvNetRecipe
from .coordinator import (
    Coordinator,
    DiLoCoConfig,
    RoundResult,
    describe_tensors,
)
from .trainer import optimizer_state


@dataclass(frozen=True)
class TrainingConfig:
    num_workers: int
    rounds: int = 10
    local_steps: int = 5
    batch_size: int = 8
    channels: tuple = (8, 16, 16)
    learning_rate: float = 0.001
    optimizer: str = "adam"
    muon_learning_rate: float = 0.02
    muon_momentum: float = 0.95
    muon_steps: int = 5
    outer_learning_rate: float = 0.7
    momentum: float = 0.9
    seed: int = 2026
    eval_examples: int = 256
    eval_interval: int = 1
    round_timeout: float = 600
    lease_seconds: float = 45
    max_retries: int = 3
    precision: str = "webgpu"

    def __post_init__(self):
        for name, low, high in (
            ("num_workers", 1, 4),
            ("rounds", 1, 100000),
            ("local_steps", 1, 100),
            ("batch_size", 1, 32),
            ("seed", 0, 2**32 - 1),
            ("eval_examples", 1, 10000),
            ("eval_interval", 1, 100000),
            ("max_retries", 0, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        for name in ("round_timeout", "lease_seconds"):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if self.precision not in ("webgpu", "float64"):
            raise ValueError("Unknown training precision")
        if (
            not isinstance(self.channels, (tuple, list))
            or len(self.channels) != 3
            or any(
                type(n) is not int or not 1 <= n <= 32 for n in self.channels
            )
        ):
            raise ValueError(
                "Browser training supports three channel counts in 1..32"
            )
        if self.batch_size * self.local_steps > 256:
            raise ValueError("Use at most 256 samples per worker per round")
        self.recipe()  # Validate model/inner optimizer without allocating weights.
        self.outer()

    def recipe(self):
        return ConvNetRecipe(
            dict(
                in_channels=1,
                height=28,
                width=28,
                channels=list(self.channels),
                residual=True,
            ),
            learning_rate=self.learning_rate,
            optimizer_kind=self.optimizer,
            muon_learning_rate=self.muon_learning_rate,
            muon_momentum=self.muon_momentum,
            muon_steps=self.muon_steps,
            max_batch=self.batch_size,
        )

    def outer(self):
        return DiLoCoConfig(
            self.local_steps,
            self.outer_learning_rate,
            self.momentum,
            bool(self.momentum),
        )


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def implementation_hash():
    """Bind recovery and experiment reports to the actual installed sources."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in (".py", ".mjs"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


class TrainingSession:
    def __init__(
        self, config, dataset, directory, *, resume=False, clock=time.monotonic
    ):
        self.config, self.dataset = config, dataset
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "checkpoint.json"
        self.recipe = config.recipe()
        self.coordinator = Coordinator(
            self.recipe.initial_weights(config.seed), config.outer()
        )
        self.members = tuple(f"worker-{i}" for i in range(config.num_workers))
        if len(dataset.train_images) < config.num_workers:
            raise ValueError("Not enough training samples for membership")
        self.identity = dict(
            format="ygaraty-network-diloco/v1",
            config=asdict(config),
            dataset=dict(name=dataset.name, sha256=dataset.sha256),
            implementation_sha256=implementation_hash(),
        )
        self.identity = json.loads(json.dumps(self.identity))
        self.states = {w: dict(step=0, moments=None) for w in self.members}
        self.receipts, self.metrics = {}, []
        self.clock, self.lock = clock, threading.RLock()
        self.join_token = secrets.token_urlsafe(32)
        self.clients, self.staged, self.bundles = {}, {}, {}
        self.started = None
        self.retries = 0
        self.error = None
        if resume:
            self._restore(recovery.load(self.path))
        elif self.path.exists():
            raise ValueError(
                "Checkpoint exists; use --resume or a new --run-dir"
            )
        else:
            self.metrics.append(
                dict(
                    version=0,
                    evaluation=dataset.evaluate(
                        self.recipe,
                        self.coordinator.weights,
                        config.eval_examples,
                    ),
                )
            )
            self._save()

    def _snapshot(self):
        return dict(
            identity=self.identity,
            coordinator=recovery.pack(self.coordinator.checkpoint()),
            states=self.states,
            receipts=self.receipts,
            metrics=self.metrics,
        )

    def _save(self):
        recovery.save(self.path, self._snapshot())

    def _restore(self, saved):
        if saved["identity"] != self.identity:
            raise ValueError(
                "Checkpoint dataset or training configuration changed"
            )
        self.coordinator.restore(recovery.unpack(saved["coordinator"]))
        version = self.coordinator.version
        if (
            not 0 <= version <= self.config.rounds
            or set(saved["states"]) != set(self.members)
            or len(saved["metrics"]) != version + 1
            or len(self.coordinator.history) != version
        ):
            raise ValueError("Incomplete committed training checkpoint")
        for state in saved["states"].values():
            optimizer_state(
                state,
                self.coordinator.weights,
                version * self.config.local_steps,
                self.recipe.moment_names(),
            )
        self.states, self.receipts, self.metrics = (
            saved["states"],
            saved["receipts"],
            saved["metrics"],
        )

    def _abort(self, reason):
        self.coordinator.abort_round()
        self.staged, self.bundles, self.started = {}, {}, None
        self.retries += 1
        if self.retries > self.config.max_retries:
            self.error = (
                reason + "; retry limit reached; restart with --resume"
            )

    def tick(self):
        """Called by the server timer as well as requests, even with no clients."""
        with self.lock:
            now = self.clock()
            expired = {
                w
                for w, c in self.clients.items()
                if now - c["seen"] >= self.config.lease_seconds
            }
            overdue = (
                self.started is not None
                and now - self.started >= self.config.round_timeout
            )
            if self.started is not None and (expired or overdue):
                # Completed workers can stay. Pending slots become replaceable;
                # all tentative moments roll back to the committed generation.
                evicted = expired | (
                    set(self.coordinator.pending) if overdue else set()
                )
                self._abort(
                    "Round deadline exceeded"
                    if overdue
                    else "Contributor disconnected"
                )
                expired |= evicted
            for worker in expired:
                self.clients.pop(worker, None)

    def join(self, token, client_id):
        with self.lock:
            if not secrets.compare_digest(token, self.join_token):
                raise PermissionError("Invalid contributor link")
            if (
                not isinstance(client_id, str)
                or not 16 <= len(client_id) <= 128
            ):
                raise ValueError("Invalid client identity")
            self.tick()
            if self.error or self.coordinator.version == self.config.rounds:
                return self.status()
            for worker, client in self.clients.items():
                if client["id"] == client_id:
                    client["seen"] = self.clock()
                    return dict(
                        worker=worker,
                        token=client["token"],
                        precision=self.config.precision,
                        heartbeat_seconds=min(
                            10, self.config.lease_seconds / 3
                        ),
                    )
            for worker in self.members:
                if worker not in self.clients:
                    self.clients[worker] = dict(
                        id=client_id,
                        token=secrets.token_urlsafe(32),
                        seen=self.clock(),
                    )
                    return dict(
                        worker=worker,
                        token=self.clients[worker]["token"],
                        precision=self.config.precision,
                        heartbeat_seconds=min(
                            10, self.config.lease_seconds / 3
                        ),
                    )
            raise ValueError("All worker slots are occupied")

    def authenticate(self, worker, token):
        self.tick()
        client = self.clients.get(worker)
        if not client or not secrets.compare_digest(client["token"], token):
            raise PermissionError(
                "Worker lease expired; rejoin with the contributor link"
            )
        client["seen"] = self.clock()

    def status(self):
        with self.lock:
            self.tick()
            return dict(
                state="failed"
                if self.error
                else "complete"
                if self.coordinator.version == self.config.rounds
                else "training"
                if self.started is not None
                else "waiting",
                version=self.coordinator.version,
                rounds=self.config.rounds,
                connected=len(self.clients),
                required=len(self.members),
                pending=list(self.coordinator.pending),
                retries=self.retries,
                error=self.error,
                metrics=copy.deepcopy(self.metrics),
            )

    def assignment(self, worker, token):
        with self.lock:
            self.authenticate(worker, token)
            status = self.status()
            if status["state"] in ("failed", "complete") or len(
                self.clients
            ) != len(self.members):
                return status
            if self.started is None:
                self.coordinator.begin_round(self.members)
                self.started = self.clock()
            if worker in self.staged:
                return dict(
                    state="submitted", version=self.coordinator.version
                )
            if worker not in self.bundles:
                step = self.states[worker]["step"]
                batches = self.dataset.batches(
                    self.members.index(worker),
                    len(self.members),
                    step,
                    self.config.local_steps,
                    self.config.batch_size,
                    self.config.seed,
                )
                self.bundles[worker] = dict(
                    state="assignment",
                    optimizer=copy.deepcopy(self.states[worker]),
                    **self.recipe.round_bundle(
                        self.coordinator.assignment(worker),
                        batches,
                        start_step=step,
                    ),
                )
            return copy.deepcopy(self.bundles[worker])

    def submit(self, worker, token, payload):
        with self.lock:
            self.authenticate(worker, token)
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("result"), dict)
                or not isinstance(
                    payload["result"].get("pseudo_gradient"), dict
                )
                or any(
                    not isinstance(v, dict)
                    for v in payload["result"]["pseudo_gradient"].values()
                )
            ):
                raise ValueError("Malformed round result")
            if (
                not isinstance(payload.get("stats", {}), dict)
                or len(json.dumps(payload.get("stats", {}))) > 8192
            ):
                raise ValueError("Worker statistics exceed 8 KiB")
            digest = fingerprint(payload)
            result = RoundResult.from_description(payload["result"])
            if result.worker != worker:
                raise ValueError("Foreign worker result")
            for receipt in (
                self.receipts.get(worker),
                self.staged.get(worker),
            ):
                if receipt and receipt["round_id"] == result.round_id:
                    if receipt["digest"] != digest:
                        raise ValueError(
                            "Conflicting retry of a submitted result"
                        )
                    return dict(
                        accepted=True, version=self.coordinator.version
                    )
            expected_step = (
                self.coordinator.version + 1
            ) * self.config.local_steps
            moments = optimizer_state(
                payload["optimizer"],
                self.coordinator.weights,
                expected_step,
                self.recipe.moment_names(),
            )
            self.coordinator.submit(result)
            self.staged[worker] = dict(
                round_id=result.round_id,
                digest=digest,
                optimizer=dict(
                    step=expected_step,
                    moments={
                        kind: describe_tensors(values)
                        for kind, values in moments.items()
                    },
                ),
                stats=copy.deepcopy(payload.get("stats", {})),
            )
            if not self.coordinator.pending:
                self._commit()
            return dict(accepted=True, version=self.coordinator.version)

    def _commit(self):
        # Only the durable checkpoint constitutes a commit. Never ACK a final
        # result while its global weights and every worker's moments are split.
        before = recovery.load(self.path)
        try:
            record = self.coordinator.commit_round()
            record["seconds"] = self.clock() - self.started
            if (
                self.coordinator.version % self.config.eval_interval == 0
                or self.coordinator.version == self.config.rounds
            ):
                record["evaluation"] = self.dataset.evaluate(
                    self.recipe,
                    self.coordinator.weights,
                    self.config.eval_examples,
                )
            record["worker_stats"] = {
                w: s["stats"] for w, s in self.staged.items()
            }
            self.states = {w: s["optimizer"] for w, s in self.staged.items()}
            self.receipts = {
                w: {k: s[k] for k in ("round_id", "digest")}
                for w, s in self.staged.items()
            }
            self.metrics.append(record)
            self._save()
            # Evaluation/checkpoint I/O holds the session lock. Heartbeats
            # queued behind that work must not expire healthy contributors.
            for client in self.clients.values():
                client["seen"] = self.clock()
        except Exception:
            self.coordinator.abort_round()
            self._restore(before)
            self.error = "Round commit failed; inspect server error and restart with --resume"
            raise
        finally:
            self.staged, self.bundles, self.started = {}, {}, None
        self.retries = 0
        print(json.dumps(record, allow_nan=False), flush=True)
