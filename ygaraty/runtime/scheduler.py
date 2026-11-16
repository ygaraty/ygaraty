"""In-memory coordinator state: leases, dependencies, and single result
commits."""

import time
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from ygaraty.runtime import capture


def owned(array):
    """A float64 array nobody else can modify. Read-only arrays (received
    frames) are kept as they are."""
    if array.dtype == np.float64 and not array.flags.writeable:
        return array
    return np.array(array, dtype=np.float64, copy=True)


@dataclass
class Lease:
    worker: bytes
    token: str
    deadline: float


class Scheduler:
    def __init__(
        self, tasks, leaves, lease_seconds=30.0, max_attempts=5, event=None
    ):
        self.tasks = {t.id: t for t in tasks}
        if len(self.tasks) != len(tasks) or set(self.tasks) & set(leaves):
            from collections import Counter

            repeated = [
                i for i, n in Counter(t.id for t in tasks).items() if n > 1
            ]
            raise ValueError(
                "Task IDs must be unique and distinct from leaves: "
                f"repeated {repeated[:5]}, leaves "
                f"{sorted(set(self.tasks) & set(leaves))[:5]}"
            )
        self.store = {k: v.copy() for k, v in leaves.items()}
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.event = event or (lambda *args, **kwargs: None)
        self.ready = deque()
        # a task that just became ready runs before the ones waiting: results
        # are consumed (and freed) sooner
        self.consumers_first = False
        self.waiting = {}
        self.dependents = {}
        self.running = {}
        self.committed = set()
        self.attempts = dict.fromkeys(self.tasks, 0)
        self.retries = 0
        # Handoffs (coordination/handoff.py): tasks leased before their inputs
        # were committed, as
        # {task: {relay name: (input task, its attempt)}}, and the committed
        # attempt and block
        # hashes of each relayed result, which such a task's result is checked
        # against.
        self.preleased, self.handoff_commits = {}, {}
        available = set(self.store) | set(self.tasks)
        for task in tasks:
            if not task.dependencies <= available:
                raise ValueError(f"Missing dependency for {task.id}")
            self.waiting[task.id] = len(task.dependencies - self.store.keys())
            for dep in task.dependencies:
                self.dependents.setdefault(dep, []).append(task.id)
            if not self.waiting[task.id]:
                self.ready.append(task.id)
                capture.emit(
                    "scheduler.ready", task=task.id, attempt_hash=None
                )
        # Validate acyclicity before any network execution.
        counts = dict(self.waiting)
        queue = deque(self.ready)
        seen = 0
        while queue:
            ident = queue.popleft()
            seen += 1
            for child in self.dependents.get(ident, []):
                counts[child] -= 1
                if counts[child] == 0:
                    queue.append(child)
        if seen != len(tasks):
            raise ValueError("Task graph contains a dependency cycle")

    @property
    def done(self):
        return len(self.committed) == len(self.tasks)

    def busy(self, worker):
        return any(lease.worker == worker for lease in self.running.values())

    def load(self, worker):
        return sum(lease.worker == worker for lease in self.running.values())

    def claim(self, worker, now=None, limit=1, choose=None):
        """Lease the next ready task to `worker`, if it holds fewer than
        `limit` tasks. `choose`,
        when given, picks the task from the ready queue (an index, or None for
        none it may take):
        owned tasks go only to their owner."""
        if not self.ready or self.load(worker) >= limit:
            return None
        now = time.monotonic() if now is None else now
        if choose is None:
            ident = self.ready.popleft()
        else:
            index = choose(self.ready, worker)
            if index is None:
                return None
            ident = self.ready[index]
            del self.ready[index]
        if self.attempts[ident] >= self.max_attempts:
            raise RuntimeError(
                f"Task {ident} exceeded {self.max_attempts} attempts"
            )
        self.attempts[ident] += 1
        lease = Lease(worker, uuid.uuid4().hex, now + self.lease_seconds)
        self.running[ident] = lease
        self.event(
            "assigned",
            task=ident,
            attempt=lease.token,
            worker=worker.hex(),
            number=self.attempts[ident],
        )
        return self.tasks[ident], lease

    def prelease(self, ident, worker, names, now=None):
        """Lease a task whose uncommitted inputs are all being computed (a
        handoff): it is in no
        queue, and its inputs' commits will not queue it. `names` as in
        self.preleased."""
        now = time.monotonic() if now is None else now
        if self.attempts[ident] >= self.max_attempts:
            raise RuntimeError(
                f"Task {ident} exceeded {self.max_attempts} attempts"
            )
        self.attempts[ident] += 1
        lease = Lease(worker, uuid.uuid4().hex, now + self.lease_seconds)
        self.running[ident] = lease
        self.preleased[ident] = names
        self.event(
            "assigned",
            task=ident,
            attempt=lease.token,
            worker=worker.hex(),
            number=self.attempts[ident],
        )
        return self.tasks[ident], lease

    def retry(self, ident, reason):
        lease = self.running.pop(ident)
        self.preleased.pop(ident, None)
        if not self.waiting[
            ident
        ]:  # a handed-off task goes back only once its inputs exist
            self.ready.append(ident)
            capture.emit(
                "scheduler.ready", task=ident, attempt_hash=None, retry=True
            )
        self.retries += 1
        self.event("retry", task=ident, attempt=lease.token, reason=reason)
        # Tasks handed this attempt's result would wait for a name that is
        # never published.
        for child, names in list(self.preleased.items()):
            if child in self.running and any(
                source == ident and token == lease.token
                for source, token in names.values()
            ):
                self.retry(child, f"input {ident} retried")

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        for ident, lease in list(self.running.items()):
            if lease.deadline <= now:
                self.retry(ident, "lease expired")

    def matches(self, ident, token, worker):
        lease = self.running.get(ident)
        return (
            lease is not None
            and lease.token == token
            and lease.worker == worker
        )

    def results(self, ident, token, worker, arrays):
        if not self.matches(ident, token, worker):
            return self.result(ident, token, worker, None)
        task = self.tasks[ident]
        specs = (
            task.attrs["nodes"]
            if task.op == "group"
            else [dict(id=ident, shape=task.shape)]
        )
        if len(arrays) != len(specs) or any(
            not isinstance(a, np.ndarray)
            or tuple(a.shape) != tuple(n["shape"])
            or not np.isfinite(a).all()
            for a, n in zip(arrays, specs)
        ):
            self.retry(ident, "invalid grouped or single result")
            return False
        # Everything result() would check again is checked above.
        for spec, array in zip(specs[:-1], arrays[:-1]):
            self.store[spec["id"]] = owned(array)
        return self._commit(ident, worker, owned(arrays[-1]))

    def result(self, ident, token, worker, array):
        if not self.matches(ident, token, worker):
            self.event(
                "ignored",
                task=ident,
                attempt=token,
                reason="stale or duplicate result",
            )
            return False
        task = self.tasks[ident]
        if (
            not isinstance(array, np.ndarray)
            or array.shape != task.shape
            or not np.isfinite(array).all()
        ):
            self.retry(ident, "invalid result shape or nonfinite data")
            return False
        return self._commit(ident, worker, owned(array))

    def result_reference(self, ident, token, worker, reference):
        """Commit a descriptor already checked against the external store."""
        if not self.matches(ident, token, worker):
            self.event(
                "ignored", task=ident, attempt=token, reason="stale reference"
            )
            return False
        if tuple(reference["shape"]) != self.tasks[ident].shape:
            self.retry(ident, "invalid reference shape")
            return False
        return self._commit(ident, worker, dict(reference))

    def _commit(self, ident, worker, value):
        self.store[ident] = value
        self.committed.add(ident)
        token = self.running[ident].token
        del self.running[ident]
        for child in self.dependents.get(ident, []):
            self.waiting[child] -= 1
            if not self.waiting[child]:
                if (
                    child not in self.running
                ):  # a handed-off task is leased already
                    (
                        self.ready.appendleft
                        if self.consumers_first
                        else self.ready.append
                    )(child)
                capture.emit("scheduler.ready", task=child, attempt_hash=None)
        self.event("completed", task=ident, attempt=token, worker=worker.hex())
        return True
