"""Admission, streaming and failure handling using a fake producer only."""

import asyncio
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

from session_fakes import OWNER, collect, request, setup
from ygaraty.backend_contracts.chat import (
    BackendError,
    ErrorCode,
    EventKind,
    EventSequence,
)
from ygaraty.backend_contracts.lifecycle import (
    CancelReason,
    CancelStatus,
    Revocation,
)
from ygaraty.backend_contracts.models import HistoryMode, SessionState
from ygaraty.backend_contracts.persistence import (
    RequestKey,
    RequestOutcome,
    RequestRecord,
)

from ygaraty.backends.webgpu.buffers import TokenMailbox


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = setup()
        self.session = self.f.session
        self.grant = self.f.access.grant(OWNER)

    async def asyncTearDown(self):
        self.f.engine.release.set()
        await self.session.close()

    async def test_encoding_leaves_event_loop_free_and_cancellation_bounded(
        self,
    ):
        entered, release = threading.Event(), threading.Event()
        main_thread = threading.get_ident()

        def encode_batch(texts):
            self.assertNotEqual(threading.get_ident(), main_thread)
            entered.set()
            if not release.wait(3):
                raise AssertionError(
                    "Encoding test did not release its thread"
                )
            return [SimpleNamespace(ids=[1, 2])]

        self.f.engine.tokenizer.encode_batch = encode_batch
        pending = asyncio.create_task(
            self.session.admit(request(), self.grant)
        )
        try:
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.005)
            self.assertTrue(entered.is_set())
            self.assertEqual(self.session.snapshot().state, SessionState.READY)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            with self.assertRaises(BackendError) as caught:
                await self.session.admit(request("second"), self.grant)
            self.assertEqual(
                caught.exception.failure.code, ErrorCode.OVERLOADED
            )
            self.assertFalse(self.session.handles)
            self.assertFalse(self.f.engine.calls)
        finally:
            release.set()
            await asyncio.wait_for(asyncio.shield(self.session._encoding), 2)
        handle = await self.session.admit(request("after"), self.grant)
        await collect(handle)

    async def test_completion_usage_and_reservation_cleanup(self):
        handle = await self.session.admit(request(), self.grant)
        events = await collect(handle)
        order = EventSequence("r")
        for event in events:
            order.accept(event)
        order.finish()
        self.assertEqual("".join(e.text for e in events), "hello world")
        self.assertEqual(events[-1].usage.output_tokens, 2)
        self.assertFalse(self.session.handles)
        self.assertEqual(
            self.f.ledger.records[handle.key].outcome, RequestOutcome.COMPLETED
        )
        with self.assertRaises(RuntimeError):
            await collect(handle)

    async def test_unready_and_full_context_fail_before_dispatch(self):
        self.f.engine.state = "waiting"
        with self.assertRaises(BackendError) as caught:
            await self.session.admit(request(), self.grant)
        self.assertEqual(caught.exception.failure.code, ErrorCode.NOT_READY)
        self.f.engine.state = "running"
        self.f.engine.tokenizer.encode = lambda text: SimpleNamespace(
            ids=list(range(128))
        )
        with self.assertRaises(BackendError) as caught:
            await self.session.admit(request(), self.grant)
        self.assertEqual(
            caught.exception.failure.code, ErrorCode.CONTEXT_EXCEEDED
        )
        self.assertFalse(self.f.engine.calls)

    async def test_webgpu_output_budget_still_clamps(self):
        handle = await self.session.admit(
            replace(request(), max_output_tokens=999), self.grant
        )
        await collect(handle)
        self.assertEqual(self.f.engine.calls[0]["max_tokens"], 126)

    async def test_per_owner_and_conversation_admission_limits(self):
        self.f.engine.mode = "blocked"
        first = await self.session.admit(request(), self.grant)
        for value, code in (
            (request("duplicate"), ErrorCode.CONVERSATION_BUSY),
            (request(), ErrorCode.INVALID_REQUEST),
        ):
            with self.assertRaises(BackendError) as caught:
                await self.session.admit(value, self.grant)
            self.assertEqual(caught.exception.failure.code, code)
        second = await self.session.admit(request("two", "two"), self.grant)
        with self.assertRaises(BackendError) as caught:
            await self.session.admit(request("three", "three"), self.grant)
        self.assertEqual(caught.exception.failure.code, ErrorCode.OVERLOADED)
        await first.aclose()
        await second.aclose()

    async def test_policy_ceiling_is_independent_of_backend_slots(self):
        limited = replace(self.grant, max_active=1, max_outstanding=1)
        handle = await self.session.admit(request(), limited)
        with self.assertRaises(BackendError):
            await self.session.admit(request("two", "two"), limited)
        await handle.aclose()

    async def test_access_revocation_stops_running_producer(self):
        self.f.engine.mode = "blocked"
        handle = await self.session.admit(request(), self.grant)
        self.f.access.version += 1
        events = await asyncio.wait_for(collect(handle), 1)
        self.assertEqual(events[-1].failure.code, ErrorCode.ACCESS_REVOKED)
        self.assertTrue(self.f.engine.finished.is_set())

    async def test_cancel_and_revoke_are_scoped_and_idempotent(self):
        self.f.engine.mode = "blocked"
        handle = await self.session.admit(request(), self.grant)
        await self.session.revoke(Revocation("session", "other"))
        self.assertFalse(handle._cancelled.is_set())
        a = await handle.cancel(CancelReason.CLIENT_REQUEST)
        b = await handle.cancel(CancelReason.REVOKED)
        self.assertEqual(a, b)
        self.assertEqual(a.status, CancelStatus.REQUESTED)
        events = await collect(handle)
        self.assertEqual(events[-1].kind, EventKind.CANCELLED)

    async def test_slow_consumer_is_bounded_and_close_stops_producer(self):
        self.f.engine.mode = "flood"
        handle = await self.session.admit(request(), self.grant)
        for _ in range(100):
            if handle._buffer.full():
                break
            await asyncio.sleep(0.005)
        self.assertEqual(handle._buffer.qsize(), 32)
        await asyncio.wait_for(handle.aclose(), 1)
        self.assertTrue(self.f.engine.finished.is_set())
        self.assertFalse(self.session.handles)

    async def test_uncooperative_producer_quarantines_until_cleanup(self):
        self.f.engine.mode = "uncooperative"
        self.session.close_timeout = 0.01
        handle = await self.session.admit(request(), self.grant)
        await asyncio.to_thread(self.f.engine.started.wait, 0.5)
        await handle.aclose()
        self.assertEqual(handle._receipt().status, CancelStatus.TIMED_OUT)
        self.assertEqual(self.session.snapshot().state, SessionState.DRAINING)
        self.assertIn("r", self.session.handles)
        self.f.engine.release.set()
        await handle._task
        await asyncio.sleep(0)
        self.assertFalse(self.session.handles)

    async def test_record_failure_does_not_dispatch_or_leak_resources(self):
        self.f.ledger.fail_admit = True
        with self.assertRaises(BackendError):
            await self.session.admit(request(), self.grant)
        self.assertFalse(self.f.engine.calls)
        self.f.ledger.fail_admit = False
        self.f.ledger.fail_finish = True
        handle = await self.session.admit(request(), self.grant)
        events = await collect(handle)
        self.assertEqual(events[-1].kind, EventKind.FAILED)
        self.assertIn(handle.key, self.session.unrecorded)
        self.assertFalse(self.session.handles)
        self.assertEqual(self.session.snapshot().state, SessionState.DRAINING)

    async def test_closed_session_and_stale_grant_cannot_admit(self):
        with self.assertRaises(BackendError):
            await self.session.admit(
                request(), replace(self.grant, epoch="old")
            )
        await self.session.close()
        with self.assertRaises(BackendError):
            await self.session.admit(request(), self.grant)

    async def test_append_and_reset_use_scoped_history(self):
        value = replace(request(), history_mode=HistoryMode.APPEND)
        handle = await self.session.admit(value, self.grant)
        await collect(handle)
        identity = self.f.engine.calls[0]["conversation_id"]
        self.assertEqual(len(identity), 64)
        self.assertNotEqual(identity, value.conversation_id)
        reset = replace(
            value,
            request_id="reset",
            history_mode=HistoryMode.RESET,
            messages=(),
        )
        handle = await self.session.admit(reset, self.grant)
        await collect(handle)
        self.assertTrue(self.f.engine.calls[-1]["reset"])
        self.assertEqual(self.f.engine.calls[-1]["conversation_id"], identity)

    async def test_revoke_between_recording_and_dispatch(self):
        original = self.f.ledger.admit

        async def revoke_after_write(record):
            await original(record)
            self.f.access.version += 1

        self.f.ledger.admit = revoke_after_write
        with self.assertRaises(BackendError) as caught:
            await self.session.admit(request(), self.grant)
        self.assertEqual(
            caught.exception.failure.code, ErrorCode.ACCESS_REVOKED
        )
        self.assertFalse(self.f.engine.calls)
        self.assertEqual(
            next(iter(self.f.ledger.records.values())).outcome,
            RequestOutcome.FAILED,
        )

    async def test_cancellation_after_recording_finalizes_without_dispatch(
        self,
    ):
        original = self.f.access.validate
        reached = asyncio.Event()
        calls = 0

        async def pause(grant, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                reached.set()
                await asyncio.Event().wait()
            await original(grant, **kwargs)

        self.f.access.validate = pause
        task = asyncio.create_task(self.session.admit(request(), self.grant))
        await reached.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            next(iter(self.f.ledger.records.values())).outcome,
            RequestOutcome.FAILED,
        )
        self.assertFalse(self.session.handles)
        self.assertFalse(self.f.engine.calls)

    async def test_revoked_grant_discards_buffer_after_producer_finishes(self):
        handle = await self.session.admit(request(), self.grant)
        await handle._task
        self.f.access.version += 1
        for _ in range(100):
            if handle._cancelled.is_set():
                break
            await asyncio.sleep(0.005)
        events = await collect(handle)
        self.assertFalse(any(e.kind == EventKind.TEXT for e in events))
        self.assertEqual(events[-1].failure.code, ErrorCode.ACCESS_REVOKED)

    async def test_duplicate_ledger_record_is_never_finalized_by_new_request(
        self,
    ):
        key = RequestKey(self.f.spec.session_id, self.f.spec.epoch, "r")
        existing = RequestRecord(key, "different-principal", "other-chat")
        await self.f.ledger.admit(existing)
        with self.assertRaises(BackendError):
            await self.session.admit(request(), self.grant)
        self.assertEqual(self.f.ledger.records[key], existing)
        self.assertFalse(self.f.engine.calls)

    async def test_cancel_during_ledger_admission_finalizes_record(self):
        original = self.f.ledger.admit
        reached = asyncio.Event()

        async def write_then_pause(record):
            await original(record)
            reached.set()
            await asyncio.Event().wait()

        self.f.ledger.admit = write_then_pause
        task = asyncio.create_task(self.session.admit(request(), self.grant))
        await reached.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(
            next(iter(self.f.ledger.records.values())).outcome,
            RequestOutcome.FAILED,
        )
        self.assertFalse(self.session.handles)
        self.assertFalse(self.f.engine.calls)

    async def test_deadline_and_stream_disconnect_release_reservations(self):
        self.f.engine.mode = "blocked"
        self.session.request_timeout = 0.02
        handle = await self.session.admit(request(), self.grant)
        events = await asyncio.wait_for(collect(handle), 1)
        self.assertEqual(events[-1].failure.code, ErrorCode.DEADLINE_EXCEEDED)
        self.session.request_timeout = 120
        handle = await self.session.admit(request("again"), self.grant)
        stream = handle.events()
        await anext(stream)
        await stream.aclose()
        self.assertFalse(self.session.handles)

    async def test_deadline_counts_from_the_last_text(self):
        # Long answers take minutes: only a pause without text fails them.
        self.f.engine.mode = "slow"
        self.session.request_timeout = 0.06  # the answer takes over 0.15 s
        handle = await self.session.admit(request(), self.grant)
        events = await asyncio.wait_for(collect(handle), 2)
        self.assertEqual(events[-1].kind, EventKind.COMPLETED)

    async def test_same_conversation_name_is_isolated_between_principals(self):
        from unittest.mock import AsyncMock

        self.session.services = replace(
            self.session.services,
            access=SimpleNamespace(validate=AsyncMock()),
        )
        self.f.engine.mode = "blocked"
        first = await self.session.admit(request(), self.grant)
        second = await self.session.admit(
            request("two"), replace(self.grant, principal_id="other")
        )
        self.assertNotEqual(
            first.arguments["conversation_id"],
            second.arguments["conversation_id"],
        )
        await first.aclose()
        await second.aclose()


class MailboxTests(unittest.TestCase):
    def test_overflow_is_terminal_and_never_blocks_a_control_sender(self):
        mailbox = TokenMailbox(2)
        for value in range(100):
            mailbox.put({"token": value})
        self.assertIn("error", mailbox.get(timeout=0.01))


def test_engine_turn_and_session_share_the_idle_timeout():
    # A long prompt sends no text while it is read; both limits must allow it.
    import inspect

    from ygaraty.backends.webgpu._legacy import prepared
    from ygaraty.backends.webgpu.configuration import IDLE_TIMEOUT_S
    from ygaraty.backends.webgpu.session import WebGPUSession

    assert WebGPUSession.request_timeout == IDLE_TIMEOUT_S >= 600
    assert "turn_timeout=IDLE_TIMEOUT_S" in inspect.getsource(prepared)
