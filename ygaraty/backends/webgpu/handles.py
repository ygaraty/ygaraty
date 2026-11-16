"""Bounded asynchronous events over the unchanged synchronous chat producer."""

import asyncio
import queue
import threading
from collections.abc import AsyncIterator

from ygaraty.backend_contracts.chat import (
    ErrorCode,
    EventKind,
    Failure,
    FinishReason,
    GenerationEvent,
    Usage,
)
from ygaraty.backend_contracts.lifecycle import (
    CancelReason,
    CancelReceipt,
    CancelStatus,
)
from ygaraty.backend_contracts.persistence import RequestOutcome


class WebGPUHandle:
    def __init__(self, session, request, grant, key, arguments):
        self.session, self.request, self.grant = session, request, grant
        self.key, self.arguments = key, arguments
        self._loop = asyncio.get_running_loop()
        self._buffer = queue.Queue(maxsize=32)
        self._available = asyncio.Event()
        self._cancelled = threading.Event()
        self._reason = CancelReason.CLIENT_REQUEST
        self._status = CancelStatus.REQUESTED
        self._failure = None
        self._consumed = self._closed = False
        self._task = asyncio.create_task(self._drive())
        self._watcher = asyncio.create_task(self._watch())

    @property
    def request_id(self):
        return self.request.request_id

    def _receipt(self):
        return CancelReceipt(self.request_id, self._reason, self._status)

    def _put(self, text):
        for offset in range(0, len(text), 16384):
            while not self._cancelled.is_set():
                try:
                    self._buffer.put(
                        text[offset : offset + 16384], timeout=0.05
                    )
                    self._loop.call_soon_threadsafe(self._available.set)
                    break
                except queue.Full:
                    continue
            if self._cancelled.is_set():
                return

    def _produce(self):
        stream = self.session.engine.chat(
            cancel_event=self._cancelled, **self.arguments
        )
        terminal = dict(
            kind=EventKind.FAILED,
            failure=Failure(ErrorCode.EXECUTION_FAILED, "Incomplete response"),
        )
        try:
            for event in stream:
                if self._cancelled.is_set():
                    break
                if "error" in event:
                    terminal = dict(
                        kind=EventKind.FAILED,
                        failure=Failure(
                            ErrorCode.EXECUTION_FAILED,
                            "Contributor response failed",
                        ),
                    )
                    break
                if event.get("text"):
                    self._put(event["text"])
                if event.get("done"):
                    terminal = dict(
                        kind=EventKind.COMPLETED,
                        finish_reason=(
                            FinishReason.LENGTH
                            if event.get("stopped") == "length"
                            else FinishReason.STOP
                        ),
                        usage=Usage(
                            event.get("prompt_tokens", 0),
                            event.get("generated", 0),
                        ),
                    )
                    break
        finally:
            stream.close()
        return terminal

    async def _drive(self):
        try:
            terminal = await self._loop.run_in_executor(
                self.session.executor, self._produce
            )
        except Exception:
            terminal = dict(
                kind=EventKind.FAILED,
                failure=Failure(
                    ErrorCode.EXECUTION_FAILED, "Execution failed"
                ),
            )
        if self._cancelled.is_set():
            terminal = dict(
                kind=EventKind.CANCELLED, cancellation=self._receipt()
            )
        if self._failure:
            terminal = dict(kind=EventKind.FAILED, failure=self._failure)
        outcome = {
            EventKind.COMPLETED: RequestOutcome.COMPLETED,
            EventKind.CANCELLED: RequestOutcome.CANCELLED,
            EventKind.FAILED: RequestOutcome.FAILED,
        }[terminal["kind"]]
        try:
            await asyncio.wait_for(
                self.session.services.requests.finish(self.key, outcome), 2
            )
        except Exception:
            self.session.record_failure(self.key, outcome)
            terminal = dict(
                kind=EventKind.FAILED,
                failure=Failure(
                    ErrorCode.EXECUTION_FAILED, "Outcome recording failed"
                ),
            )
        self._available.set()
        return terminal

    async def _watch(self):
        clock = self.session.services.clock
        deadline = clock.monotonic_seconds() + self.session.request_timeout
        try:
            while not self._closed:
                await asyncio.sleep(self.session.revalidate_interval)
                if clock.monotonic_seconds() >= deadline:
                    self._failure = Failure(
                        ErrorCode.DEADLINE_EXCEEDED,
                        "Request deadline exceeded",
                    )
                    await self.cancel(CancelReason.DEADLINE)
                    await self.aclose()
                    return
                try:
                    await asyncio.wait_for(
                        self.session.authorize(self.grant), 2
                    )
                except Exception:
                    self._failure = Failure(
                        ErrorCode.ACCESS_REVOKED,
                        "Access is no longer valid",
                    )
                    await self.cancel(CancelReason.REVOKED)
        except asyncio.CancelledError:
            pass

    async def events(self) -> AsyncIterator[GenerationEvent]:
        if self._consumed:
            raise RuntimeError("A request stream has one consumer")
        self._consumed = True
        sequence = 0
        try:
            yield GenerationEvent(
                self.request_id, sequence, EventKind.ACCEPTED
            )
            while True:
                self._available.clear()
                try:
                    text = self._buffer.get_nowait()
                except queue.Empty:
                    if self._task.done():
                        break
                    await self._available.wait()
                    continue
                if not self._cancelled.is_set():
                    sequence += 1
                    yield GenerationEvent(
                        self.request_id, sequence, EventKind.TEXT, text=text
                    )
            terminal = await self._task
            if self._failure:
                terminal = dict(kind=EventKind.FAILED, failure=self._failure)
            yield GenerationEvent(self.request_id, sequence + 1, **terminal)
        finally:
            await asyncio.shield(self.aclose())

    async def cancel(self, reason: CancelReason) -> CancelReceipt:
        if not isinstance(reason, CancelReason):
            raise ValueError("Expected CancelReason")
        if not self._cancelled.is_set():
            self._reason = reason
            self._cancelled.set()
        self._available.set()
        # Legacy reset packets are generation-fenced, but have no GPU stop ACK.
        return self._receipt()

    async def aclose(self) -> None:
        if self._closed:
            return
        if not self._task.done():
            await self.cancel(CancelReason.DISCONNECTED)
        try:
            await asyncio.wait_for(
                asyncio.shield(self._task), self.session.close_timeout
            )
        except TimeoutError:
            self._status = CancelStatus.TIMED_OUT
            self.session.quarantined = True
            # Keep the reservation while an uncooperative producer still runs.
            self._task.add_done_callback(lambda _: self.session.release(self))
        else:
            self.session.release(self)
        self._closed = True
        if asyncio.current_task() is not self._watcher:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
