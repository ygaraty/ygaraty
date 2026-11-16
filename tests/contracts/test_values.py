"""Metadata-only validation, with no backend or workload imports."""

import unittest
from dataclasses import FrozenInstanceError, replace

from ygaraty.backend_contracts.access import AccessGrant, Action
from ygaraty.backend_contracts.chat import ChatRequest, Message, Role
from ygaraty.backend_contracts.models import (
    BackendConfiguration,
    BackendDescriptor,
    EffectiveCapabilities,
    HistoryMode,
    ModelIdentity,
    Route,
    SessionSnapshot,
    SessionSpec,
    SessionState,
)
from ygaraty.backend_contracts.persistence import (
    RequestKey,
    RequestOutcome,
    RequestRecord,
    SessionRecord,
    StoreConflict,
)


def session_spec() -> SessionSpec:
    return SessionSpec(
        "session-1",
        "epoch-1",
        ModelIdentity("fixture", "sha256:a", "sha256:b", "sha256:c", "f32"),
        BackendConfiguration("fixture", 1, b'{"context_tokens":128}'),
    )


def grant() -> AccessGrant:
    return AccessGrant(
        "session-1",
        "epoch-1",
        "alice",
        "grant-1",
        1,
        100,
        frozenset({Action.GENERATE}),
        1,
        2,
    )


class ValueTests(unittest.TestCase):
    def test_nested_values_are_immutable(self):
        spec = session_spec()
        with self.assertRaises(FrozenInstanceError):
            spec.model.precision = "f16"
        with self.assertRaises(FrozenInstanceError):
            spec.configuration.payload = b"changed"
        with self.assertRaises(ValueError):
            replace(spec.configuration, payload=bytearray(b"mutable"))

    def test_configuration_requires_bounded_versioned_payload(self):
        for changes in (
            {"schema_version": True},
            {"schema_version": 0},
            {"payload": b""},
            {"payload": b"x" * ((1 << 20) + 1)},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    replace(session_spec().configuration, **changes)

    def test_descriptor_rejects_invalid_versions(self):
        for versions in ((), (True,), (0,), (1, 1), [1]):
            with self.subTest(versions=versions):
                with self.assertRaises(ValueError):
                    BackendDescriptor("fixture", "dev", versions)

    def test_effective_capacity_is_strict_and_bounded(self):
        caps = EffectiveCapabilities(
            128, 32, 1, 0, (HistoryMode.FULL,), (Route.RELAY,), True
        )
        for changes in (
            {"context_tokens": True},
            {"output_tokens": 129},
            {"active_requests": 0},
            {"queued_requests": -1},
            {"routes": ()},
            {"history_modes": [HistoryMode.FULL]},
            {"routes": (Route.RELAY, Route.RELAY)},
            {"browser_contribution": 1},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(caps, **changes)

    def test_ready_requires_effective_capabilities(self):
        snapshot = SessionSnapshot("s", "e", 0, SessionState.PREPARING)
        with self.assertRaises(ValueError):
            replace(snapshot, state=SessionState.READY)
        with self.assertRaises(ValueError):
            replace(snapshot, state="ready")

    def test_history_mode_and_conversation_are_explicit(self):
        request = ChatRequest(
            "r",
            "chat-1",
            "fixture",
            HistoryMode.FULL,
            (Message(Role.USER, "Hello"),),
        )
        for changes in (
            {"messages": []},
            {"messages": ()},
            {"messages": (Message(Role.ASSISTANT, "Hi"),)},
            {"conversation_id": "../other"},
            {"max_output_tokens": True},
            {"history_mode": "append"},
            {"messages": request.messages * 129},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    replace(request, **changes)
        reset = replace(request, history_mode=HistoryMode.RESET, messages=())
        self.assertEqual(reset.messages, ())


class AccessTests(unittest.TestCase):
    def test_scope_and_expiry_boundaries(self):
        args = dict(
            session_id="session-1",
            epoch="epoch-1",
            action=Action.GENERATE,
            now=99,
        )
        grant().require(**args)
        for changes in (
            {"session_id": "other"},
            {"epoch": "old"},
            {"action": Action.MANAGE},
            {"now": 100},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(PermissionError):
                    grant().require(**(args | changes))

    def test_access_ceilings_and_scopes_are_validated(self):
        for changes in (
            {"max_active": 3},
            {"version": 0},
            {"expires_at": True},
            {"actions": {Action.GENERATE}},
            {"actions": frozenset()},
            {"actions": frozenset({"generate"})},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(grant(), **changes)


class PersistenceValueTests(unittest.TestCase):
    def setUp(self):
        self.record = RequestRecord(
            RequestKey("s", "e", "r"), "alice", "chat-1"
        )

    def test_terminal_transition_is_immutable_and_idempotent(self):
        completed = self.record.finish(RequestOutcome.COMPLETED)
        self.assertIsNone(self.record.outcome)
        self.assertEqual(completed.finish(RequestOutcome.COMPLETED), completed)
        with self.assertRaises(StoreConflict):
            completed.finish(RequestOutcome.FAILED)

    def test_interrupted_work_cannot_later_be_marked_successful(self):
        interrupted = self.record.finish(RequestOutcome.INTERRUPTED)
        with self.assertRaises(StoreConflict):
            interrupted.finish(RequestOutcome.COMPLETED)

    def test_records_reject_invalid_state(self):
        with self.assertRaises(ValueError):
            replace(self.record, outcome="completed")
        with self.assertRaises(ValueError):
            SessionRecord(session_spec(), revision=True)
        with self.assertRaises(ValueError):
            SessionRecord(session_spec(), closed="yes")


if __name__ == "__main__":
    unittest.main()
