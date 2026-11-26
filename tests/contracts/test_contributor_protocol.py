"""Contributor protocol values, plus the conformance suite run on a fake backend."""

import json
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from test_values import session_spec

from ygaraty.backend_contracts.contributors import (
    ContributorError,
    ContributorFault,
    ContributorState,
    JoinReply,
    JoinTicket,
    RosterEntry,
    contributor_port,
    decode,
    encode,
    operation,
)
from ygaraty.backend_contracts.models import (
    BackendDescriptor,
    EffectiveCapabilities,
    HistoryMode,
    Route,
    SessionSnapshot,
    SessionState,
)
from ygaraty.backend_contracts.protocols import RuntimeServices
from ygaraty.backend_contracts.testing import BackendConformance


class ValueTests(unittest.TestCase):
    def test_messages_are_bounded_json_objects(self):
        self.assertEqual(decode(encode({"a": 1})), {"a": 1})
        for bad in (b"[]", b"nope", "{}", b"\xff", b"{" + b" " * (1 << 20) + b"}"):
            with self.assertRaises(ValueError):
                JoinTicket(0, "192.0.2.1", 1, bad)

    def test_ticket_reply_and_roster_validation(self):
        JoinTicket(1, "192.0.2.1", 1, b"{}", previous_id="abc")
        with self.assertRaises(ValueError):
            JoinTicket(-1, "192.0.2.1", 1, b"{}")
        with self.assertRaises(ValueError):
            JoinTicket(0, "", 1, b"{}")
        with self.assertRaises(ValueError):
            JoinReply("id", "yes", b"{}")
        RosterEntry("id", 0, ContributorState.ACTIVE, progress=b'{"phase":"ready"}')
        with self.assertRaises(ValueError):
            RosterEntry("id", 0, "active")

    def test_operation_names(self):
        operation("claim")
        for bad in ("join", "heartbeat", "leave", "Claim", "a/b", "", "x" * 40):
            with self.assertRaises(ContributorError) as caught:
                operation(bad)
            self.assertEqual(caught.exception.fault, ContributorFault.UNSUPPORTED)

    def test_snapshot_detail(self):
        SessionSnapshot("s", "e", 0, SessionState.FAILED, detail="Contributor left")
        with self.assertRaises(ValueError):
            SessionSnapshot("s", "e", 0, SessionState.FAILED, detail="")

    def test_port_lookup(self):
        self.assertIsNone(contributor_port(object()))
        port = Mock()
        self.assertIs(contributor_port(Mock(contributors=lambda: port)), port)


class FakePort:
    """The smallest correct ContributorPort: one slot, one `ready` operation."""

    def __init__(self):
        self.entries = {}

    def operations(self):
        return frozenset({"ready"})

    def join(self, ticket):
        if decode(ticket.payload).get("backend") != "fixture":
            raise ContributorError(ContributorFault.INVALID, "Wrong backend")
        cid = f"c{len(self.entries)}"
        self.entries[cid] = RosterEntry(cid, ticket.slot, ContributorState.PREPARING)
        return JoinReply(cid, True, b"{}")

    def heartbeat(self, cid, payload):
        entry = self.entries.get(cid)
        return entry is not None and entry.state != ContributorState.RETIRED

    def call(self, cid, name, payload):
        operation(name)
        if name not in self.operations():
            raise ContributorError(ContributorFault.UNSUPPORTED, "Unknown operation")
        if not self.heartbeat(cid, b"{}"):
            raise ContributorError(ContributorFault.GONE, "Join again")
        entry = self.entries[cid]
        self.entries[cid] = RosterEntry(cid, entry.slot, ContributorState.ACTIVE)
        return b"{}"

    def leave(self, cid, reason):
        if cid in self.entries:
            entry = self.entries[cid]
            self.entries[cid] = RosterEntry(cid, entry.slot, ContributorState.RETIRED)

    def roster(self):
        return tuple(self.entries.values())

    def expire(self):
        return ()


class FakeSession:
    def __init__(self, spec):
        self.spec, self.port, self.closed = spec, FakePort(), False

    def contributors(self):
        return self.port

    def snapshot(self):
        if self.closed:
            return SessionSnapshot(self.spec.session_id, self.spec.epoch, 1, SessionState.CLOSED)
        ready = any(e.state == ContributorState.ACTIVE for e in self.port.roster())
        capabilities = EffectiveCapabilities(
            128, 32, 1, 0, (HistoryMode.FULL,), (Route.RELAY,), False
        )
        return SessionSnapshot(
            self.spec.session_id,
            self.spec.epoch,
            0,
            SessionState.READY if ready else SessionState.PREPARING,
            capabilities if ready else None,
        )

    async def close(self):
        self.closed = True


class FakeFactory:
    def describe(self):
        return BackendDescriptor("fixture", "test")

    def validate(self, spec):
        config = spec.configuration
        if config.backend_id != "fixture" or config.schema_version != 1:
            raise ValueError("Unsupported configuration")
        try:
            data = json.loads(config.payload)
        except ValueError:
            raise ValueError("Configuration must be JSON") from None
        if set(data) != {"context_tokens"}:
            raise ValueError("Unknown configuration fields")

    async def open(self, spec, services):
        self.validate(spec)
        return FakeSession(spec)


class FakeBackendConformance(BackendConformance, IsolatedAsyncioTestCase):
    def make_factory(self):
        return FakeFactory()

    def make_spec(self):
        return session_spec()

    def make_services(self):
        return RuntimeServices(Mock(), Mock(), Mock())

    async def start_contributors(self, session):
        port = session.contributors()
        reply = port.join(JoinTicket(0, "192.0.2.1", 1 << 30, encode({"backend": "fixture"})))
        port.call(reply.contributor_id, "ready", b"{}")
        self.assertEqual(session.snapshot().state, SessionState.READY)


if __name__ == "__main__":
    unittest.main()
