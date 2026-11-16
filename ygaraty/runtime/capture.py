"""Opt-in local calibration traces. No sockets, tensor values, credentials or
argv.

Set YGARATY_CAPTURE_DIR on the processes to record. A bounded queue keeps disk
I/O
out of request/kernel threads; drops/errors and a closing record make loss
visible.
Durations use monotonic clocks; UTC anchors are only for approximate host
alignment.
"""

import atexit
import hashlib
import json
import os
import platform
import queue
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

try:
    import resource
except ImportError:  # Windows workers can still capture spans.
    resource = None
import sys
import threading
import time
import uuid

_context = ContextVar("calibration_context", default={})
_instance = None
_init_lock = threading.Lock()


def attempt_id(value):
    return hashlib.sha256(str(value).encode()).hexdigest() if value else None


def task_fields(work):
    task = work["task"]
    return dict(
        task=task["id"],
        attempt_hash=attempt_id(work.get("attempt")),
        op=task["op"],
        owner=task.get("attrs", {}).get("owner"),
        layer=task.get("attrs", {}).get("layer"),
        band=task.get("attrs", {}).get("band"),
    )


class Recorder:
    def __init__(self, directory, capacity=32768):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.identity = uuid.uuid4().hex
        self.path = (
            root / f"{platform.node()}-{os.getpid()}-{self.identity}.jsonl"
        )
        self.file = self.path.open("x", buffering=1 << 20)
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.failed = False
        self.closed = False
        self.guard = threading.Lock()
        self.thread = threading.Thread(
            target=self._write, name="calibration-writer", daemon=True
        )
        package = Path(__file__).resolve().parents[1]
        source = hashlib.sha256()
        for p in sorted(package.rglob("*.py")):
            source.update(str(p.relative_to(package)).encode())
            source.update(p.read_bytes())
        self.emit(
            "capture_start",
            python=platform.python_version(),
            system=platform.platform(),
            machine=platform.machine(),
            cpu_count=os.cpu_count(),
            package_sha256=source.hexdigest(),
            revision=os.environ.get("YGARATY_CAPTURE_REVISION", ""),
            thread_settings={
                k: os.environ.get(k)
                for k in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        )
        try:
            import numpy as np

            self.emit("numpy", version=np.__version__)
            from threadpoolctl import threadpool_info

            self.emit(
                "libraries",
                numpy=np.__version__,
                blas=[
                    {k: v for k, v in p.items() if k != "filepath"}
                    for p in threadpool_info()
                ],
            )
        except ImportError:
            self.emit(
                "libraries",
                warning="threadpoolctl or numpy unavailable; retain BLAS "
                "identity separately",
            )
        self.thread.start()

    def emit(self, event, **fields):
        with self.guard:
            if self.closed or self.failed:
                return
            row = dict(
                schema_version=1,
                capture=self.identity,
                event=event,
                mono_ns=time.perf_counter_ns(),
                utc_ns=time.time_ns(),
                pid=os.getpid(),
                thread=threading.get_ident(),
                **dict(_context.get(), **fields),
            )
            try:
                self.queue.put_nowait(row)
            except queue.Full:
                self.dropped += 1

    def _write(self):
        try:
            next_sample = 0
            while True:
                try:
                    row = self.queue.get(timeout=0.5)
                except queue.Empty:
                    row = {}
                if row is None:
                    break
                if row:
                    if row["event"] == "plan":
                        started = time.perf_counter_ns()
                        row["graph_sha256"] = hashlib.sha256(
                            json.dumps(
                                row["graph"],
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode()
                        ).hexdigest()
                        row["writer_hash_ns"] = (
                            time.perf_counter_ns() - started
                        )
                    self.file.write(json.dumps(row, allow_nan=False) + "\n")
                if resource is not None and time.monotonic() >= next_sample:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    self.file.write(
                        json.dumps(
                            dict(
                                schema_version=1,
                                capture=self.identity,
                                event="process_sample",
                                mono_ns=time.perf_counter_ns(),
                                utc_ns=time.time_ns(),
                                user_s=usage.ru_utime,
                                system_s=usage.ru_stime,
                                max_rss=usage.ru_maxrss,
                                max_rss_unit="bytes"
                                if sys.platform == "darwin"
                                else "KiB",
                                minor_faults=usage.ru_minflt,
                                major_faults=usage.ru_majflt,
                                voluntary_switches=usage.ru_nvcsw,
                                involuntary_switches=usage.ru_nivcsw,
                            )
                        )
                        + "\n"
                    )
                    self.file.flush()
                    next_sample = time.monotonic() + 1
            self.file.write(
                json.dumps(
                    dict(
                        schema_version=1,
                        capture=self.identity,
                        event="capture_end",
                        dropped=self.dropped,
                        failed=self.failed,
                        mono_ns=time.perf_counter_ns(),
                        utc_ns=time.time_ns(),
                    )
                )
                + "\n"
            )
            self.file.flush()
        except Exception as error:
            self.failed = True
            print(
                f"Calibration capture failed ({type(error).__name__}); "
                f"trace is incomplete",
                file=sys.stderr,
            )
        finally:
            self.file.close()

    def close(self, timeout=10):
        with self.guard:
            if self.closed:
                return
            self.closed = True
        deadline = time.monotonic() + timeout
        # Bound the entire close, including making room for the sentinel.
        while self.thread.is_alive() and time.monotonic() < deadline:
            try:
                self.queue.put(
                    None, timeout=min(0.1, max(0, deadline - time.monotonic()))
                )
                break
            except queue.Full:
                pass
        self.thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.thread.is_alive():
            self.failed = True
            print(
                "Calibration capture did not finish; trace is incomplete",
                file=sys.stderr,
            )


def recorder():
    global _instance
    if not os.environ.get("YGARATY_CAPTURE_DIR"):
        return None
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = Recorder(os.environ["YGARATY_CAPTURE_DIR"])
                atexit.register(_instance.close)
    return _instance


def emit(event, **fields):
    r = recorder()
    if r:
        r.emit(event, **fields)


@contextmanager
def context(**fields):
    token = _context.set(dict(_context.get(), **fields))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def post_context(worker, records):
    if recorder() is None:
        yield
        return
    with context(
        worker=worker, attempts=[attempt_id(r["attempt"]) for r in records]
    ):
        yield


@contextmanager
def span(event, **fields):
    r = recorder()
    if r is None:
        yield {}
        return
    start, cpu = time.perf_counter_ns(), time.thread_time_ns()
    extra = {}
    try:
        yield extra
    except BaseException as error:
        extra["error_type"] = type(error).__name__
        raise
    finally:
        end, cpu_end = time.perf_counter_ns(), time.thread_time_ns()
        r.emit(
            event,
            start_ns=start,
            end_ns=end,
            wall_ns=end - start,
            thread_cpu_ns=cpu_end - cpu,
            **fields,
            **extra,
        )


def traced(event, job=False):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if recorder() is None:
                return fn(*args, **kwargs)
            self = args[0] if args else kwargs.get("self")
            rest = args[1:]
            fields = dict(job=self.id, update=self.step + 1) if job else {}
            if fn.__name__ in ("make_step", "finish_step_io"):
                fields.update(task=None, attempt_hash=None, worker=None)
            if event.startswith("worker.") and hasattr(self, "id"):
                fields["worker"] = self.id
            if job and fn.__name__ in ("record_result", "commit"):
                if fn.__name__ == "record_result":
                    worker = rest[0] if rest else kwargs["worker"]
                    message = rest[1] if len(rest) > 1 else kwargs["message"]
                    fields.update(
                        worker=worker,
                        task=message.get("task_id"),
                        attempt_hash=attempt_id(message.get("attempt")),
                    )
                else:
                    task = rest[0] if rest else kwargs["task"]
                    lease = rest[1] if len(rest) > 1 else kwargs["lease"]
                    fields.update(
                        task=task.id, attempt_hash=attempt_id(lease.token)
                    )
            if fn.__name__ == "finish_step_io":
                fields.update(
                    job=self.id, update=rest[0] if rest else kwargs["step"]
                )
            with context(**fields), span(event):
                result = fn(*args, **kwargs)
                if job and fn.__name__ == "make_step":
                    emit("update_ready", tasks=len(self.tasks))
                return result

        return wrapped

    return decorate


def lock(value):
    # Disabled path uses the original lock context manager directly.
    r = recorder()
    return value if r is None else _timed_lock(value, r)


@contextmanager
def _timed_lock(value, r):
    start = time.perf_counter_ns()
    value.acquire()
    acquired = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        value.release()
        # Release this acquisition before logging; an outer reentrant
        # acquisition may remain.
        r.emit(
            "coordinator.lock_wait",
            start_ns=start,
            end_ns=acquired,
            wall_ns=acquired - start,
        )
        r.emit(
            "coordinator.lock_hold",
            start_ns=acquired,
            end_ns=end,
            wall_ns=end - acquired,
        )


def instrument_store(store):
    """Measure public store calls, including DB/client work, not pure DB
    execution time."""
    if recorder() is None or getattr(store, "_calibration_wrapped", False):
        return
    store._calibration_wrapped = True
    for name in (
        "read",
        "put",
        "put_remote",
        "put_block",
        "publish",
        "lease",
        "release",
        "locate",
        "assemble",
        "transpose",
        "cleanup_attempt",
    ):
        method = getattr(store, name, None)
        if method is None:
            continue

        def wrapped(*args, _method=method, _name=name, **kwargs):
            with span("store." + _name):
                return _method(*args, **kwargs)

        setattr(store, name, wrapped)


def exchange(connection, method, path, body, headers, stage):
    """Application-body counts; send/wait/read include socket/server delays,
    not pure wire time."""
    if recorder() is None:
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response, response.read()
    request_id = uuid.uuid4().hex
    with (
        context(request=request_id),
        span(
            stage, method=method, request_bytes=len(body) if body else 0
        ) as details,
    ):
        with span("http.send"):
            connection.request(
                method,
                path,
                body,
                dict(headers, **{"X-Ygaraty-Trace": request_id}),
            )
        with span("http.wait_headers"):
            response = connection.getresponse()
        with span("http.read"):
            raw = response.read()
        details.update(
            status=response.status,
            response_bytes=len(raw),
            body_byte_scope="encoded application body; excludes "
            "HTTP/TLS headers",
            request_compressed=bool(headers.get("Content-Encoding")),
            response_compressed=bool(response.getheader("Content-Encoding"))
            if hasattr(response, "getheader")
            else None,
        )
        return response, raw


def capture_plan(job):
    """Snapshot metadata only. Leaf descriptors omit
    versions/blocks/credentials.

    Task attributes may contain token IDs: keep traces local, like dataset
    artifacts.
    """
    if recorder() is None:
        return
    import copy

    with (
        context(job=job.id, update=job.step + 1),
        span("capture.plan_overhead"),
    ):
        graph = dict(
            tasks=[t.describe() for t in job.tasks],
            leaves={
                k: dict(shape=list(v["shape"])) for k, v in job.leaves.items()
            },
            reusable=sorted(job.reusable_ids()),
            owner_keys=list(getattr(job, "owner_keys", [])),
            owned=bool(getattr(job, "owned_mode", False)),
            replicated=False,
            tokens=copy.deepcopy(job.current_source.get("tokens", {})),
            model=copy.deepcopy(job.model_config),
        )
        # Task.describe() already deep-copies attributes. Hash/serialize on the
        # writer.
        # Copy the remaining metadata once so later updates cannot mutate
        # queued evidence.
        tokenizer = getattr(job, "tokenizer_record", None)
        model = tokenizer.get("model") if tokenizer else None
        emit(
            "plan",
            graph=graph,
            provenance=dict(
                dataset_sha256=job.dataset_hash,
                tokenizer_sha256=hashlib.sha256(model).hexdigest()
                if isinstance(model, (bytes, bytearray, memoryview))
                else None,
                batch=copy.deepcopy(job.current_source),
                config=copy.deepcopy(job.config),
                publishing=getattr(job, "publishing", False),
                runtime={
                    k: getattr(job.runtime, k, None)
                    for k in (
                        "max_task_bytes",
                        "layer_task_bytes",
                        "output_task_bytes",
                        "activation_wire",
                        "float32_wire",
                        "owners",
                        "owned_layers",
                        "owned_vocab",
                        "batch_tokens",
                        "length_window",
                        "band",
                        "placement",
                        "placement_pairs",
                        "prune_plan",
                        "tile_size",
                    )
                },
                checkpoint_every=job.checkpoint_every,
                lease_seconds=job.lease_seconds,
                owners={
                    k: v.decode() if isinstance(v, bytes) else v
                    for k, v in getattr(job, "owners", {}).items()
                },
                epoch=job.epoch,
                cursor=job.cursor,
            ),
        )
