"""ASGI transport for existing synchronous workload protocol services.

No sockets or HTTP server are created here. Responses flow through a bounded
queue; ASGI owns HTTP framing, disconnects, and transfer encoding.
"""

import asyncio
from email.message import Message
import io
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException
from starlette.responses import StreamingResponse


class Transport:
    # Concurrent synchronous requests over all sessions; above it, 503. A vLLM contributor alone sends one output
    # event per running chat (32) plus its job poll (DO-88). YGARATY_COORDINATOR_REQUESTS overrides.
    def __init__(self, workers=None):
        import os

        workers = workers or int(
            os.environ.get("YGARATY_COORDINATOR_REQUESTS", "128")
        )
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="coordinator"
        )
        self.slots = threading.BoundedSemaphore(workers)
        # A 32-chat burst acknowledges 32 jobs at once beside polls and heartbeats; 16 slots overflowed (DO-91).
        control_workers = int(
            os.environ.get("YGARATY_WEBGPU_CONTROL_REQUESTS", "64")
        )
        self.control_executor = ThreadPoolExecutor(
            max_workers=control_workers, thread_name_prefix="webgpu-control"
        )
        self.control_slots = threading.BoundedSemaphore(control_workers)
        self.closed = threading.Event()
        self.requests = set()
        self.lock = threading.Lock()

    def close(self):
        self.closed.set()
        with self.lock:
            for stop in self.requests:
                stop.set()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.control_executor.shutdown(wait=True, cancel_futures=True)

    async def dispatch(
        self, request, protocol, path=None, transform=None, control=False
    ):
        slots = self.control_slots if control else self.slots
        executor = self.control_executor if control else self.executor
        if self.closed.is_set() or not slots.acquire(blocking=False):
            raise HTTPException(503, "Coordinator request capacity reached")
        stop = threading.Event()
        with self.lock:
            self.requests.add(stop)
        submitted = False
        try:
            # Bound both length-delimited and chunked bodies before protocol decoding.
            lengths = request.headers.getlist("content-length")
            if len(lengths) > 1 or (lengths and not lengths[0].isdigit()):
                raise HTTPException(400, "Invalid Content-Length")
            if lengths and int(lengths[0]) > 16 * 1024 * 1024:
                raise HTTPException(413, "Request exceeds 16 MiB")
            body_started = time.monotonic()
            body = bytearray()
            async with asyncio.timeout(30):
                async for part in request.stream():
                    body.extend(part)
                    if len(body) > 16 * 1024 * 1024:
                        raise HTTPException(413, "Request exceeds 16 MiB")
            if lengths and int(lengths[0]) != len(body):
                raise HTTPException(400, "Truncated request body")
            headers = Message()
            for name, value in request.headers.items():
                if name.lower() != "content-length":
                    headers[name] = value
            headers["Content-Length"] = str(len(body))
            loop = asyncio.get_running_loop()
            ready = loop.create_future()
            chunks = queue.Queue(maxsize=8)

            def put(value):
                while not stop.is_set() and not self.closed.is_set():
                    try:
                        chunks.put(value, timeout=0.1)
                        return
                    except queue.Full:
                        pass
                raise BrokenPipeError("ASGI client disconnected")

            def resolve(value):
                if not ready.done():
                    ready.set_result(value)

            class Writer:
                def write(self, value):
                    view = memoryview(value).cast("B")
                    for start in range(0, len(view), 65536):
                        put(bytes(view[start : start + 65536]))
                    return len(view)

                def flush(self):
                    pass

            class Context(protocol):
                def send_response(self, code, message=None):
                    self.code, self.output_headers = code, {}

                def send_header(self, name, value):
                    if name.lower() not in {"transfer-encoding", "connection"}:
                        self.output_headers[name.lower()] = str(value)

                def end_headers(self):
                    self.started = True
                    loop.call_soon_threadsafe(
                        resolve, (self.code, self.output_headers)
                    )

                def stream(self, events):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        for event in events:
                            self.wfile.write(
                                (json.dumps(event) + "\n").encode()
                            )
                    finally:
                        if hasattr(events, "close"):
                            events.close()

                send_stream = stream

                def reply(
                    self, code, body=b"", mime="application/json", extra=()
                ):
                    if code >= 500:
                        body = {"error": "Contributor protocol failure"}
                    if transform and isinstance(body, (str, bytes, dict, list)):
                        body = transform(body, mime)
                    if isinstance(body, (dict, list)):
                        body = json.dumps(body).encode()
                    elif isinstance(body, str):
                        body = body.encode()
                    self.send_response(code)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    for name, value in extra:
                        self.send_header(name, value)
                    self.end_headers()
                    self.wfile.write(body)

            context = object.__new__(Context)
            context.path = path or request.url.path
            if request.url.query:
                context.path += "?" + request.url.query
            context.headers, context.rfile = headers, io.BytesIO(body)
            context.wfile, context.started = Writer(), False
            context.cancelled = stop
            context.received_body_seconds = time.monotonic() - body_started
            context.client_address = (
                request.client.host if request.client else "unknown",
                0,
            )
            # Upload protocol compatibility: the ASGI body deadline already ran.
            context.connection = type(
                "ReadDeadline",
                (),
                {"gettimeout": lambda s: 30, "settimeout": lambda s, t: None},
            )()

            def run():
                try:
                    (
                        context.do_POST
                        if request.method == "POST"
                        else context.do_GET
                    )()
                except BrokenPipeError:
                    pass
                except Exception:
                    if not context.started:
                        loop.call_soon_threadsafe(
                            resolve,
                            (500, {"content-type": "application/json"}),
                        )
                        put(b'{"error":"Coordinator protocol failure"}')
                finally:
                    if not context.started:
                        loop.call_soon_threadsafe(resolve, (500, {}))
                    try:
                        put(None)
                    except BrokenPipeError:
                        pass
                    context.rfile.close()
                    slots.release()
                    with self.lock:
                        self.requests.discard(stop)

            executor.submit(run)
            submitted = True
            code, output_headers = await ready

            async def content():
                try:
                    while True:
                        try:
                            part = await asyncio.to_thread(
                                chunks.get, True, 0.1
                            )
                        except queue.Empty:
                            if stop.is_set() or self.closed.is_set():
                                break
                            continue
                        if part is None:
                            break
                        yield part
                finally:
                    stop.set()

            return StreamingResponse(
                content(), status_code=code, headers=output_headers
            )
        except BaseException as error:
            stop.set()
            if not submitted:
                slots.release()
                with self.lock:
                    self.requests.discard(stop)
            if isinstance(error, TimeoutError):
                raise HTTPException(
                    408, "Request body deadline exceeded"
                ) from error
            raise
