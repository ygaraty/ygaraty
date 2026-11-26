"""Conformance checks every backend runs against its own factory.

Mix `BackendConformance` into a `unittest.IsolatedAsyncioTestCase` and supply
fixtures; the backend may use fakes (no GPU, model or network is required):

    class VllmConformance(BackendConformance, IsolatedAsyncioTestCase):
        def make_factory(self): ...
        def make_spec(self): ...            # valid for that factory
        def make_services(self): ...
        async def start_contributors(self, session): ...  # optional

The checks cover the boundary only: discovery without resources, strict
configuration validation, snapshots, idempotent close, and the contributor
port's message rules. Chat behaviour stays with each backend's own tests,
which should pass every stream through `assert_stream`.
"""

from dataclasses import replace

from .chat import EventSequence
from .contributors import (
    RESERVED,
    ContributorError,
    ContributorFault,
    JoinReply,
    JoinTicket,
    RosterEntry,
    contributor_port,
    encode,
)
from .models import (
    CONTRACT_MAJOR,
    BackendDescriptor,
    SessionSnapshot,
    SessionState,
)
from .registry import BackendRegistry


async def assert_stream(handle) -> list:
    """Consume a request handle; require one valid, terminated sequence."""
    sequence = EventSequence(handle.request_id)
    events = []
    async for event in handle.events():
        sequence.accept(event)
        events.append(event)
    sequence.finish()
    await handle.aclose()
    await handle.aclose()  # idempotent
    return events


class BackendConformance:
    def make_factory(self):
        raise NotImplementedError

    def make_spec(self):
        raise NotImplementedError

    def make_services(self):
        raise NotImplementedError

    async def start_contributors(self, session):
        """Optional: join and ready fake contributors so the session is READY."""

    # Discovery and validation -------------------------------------------

    def test_describe_needs_no_resources(self):
        descriptor = self.make_factory().describe()
        self.assertIsInstance(descriptor, BackendDescriptor)
        self.assertIn(CONTRACT_MAJOR, descriptor.contract_majors)
        self.assertEqual(
            descriptor.backend_id, self.make_spec().configuration.backend_id
        )

    def test_registers_once(self):
        registry = BackendRegistry()
        registry.register(self.make_factory())
        with self.assertRaises(ValueError):
            registry.register(self.make_factory())

    def test_validate_accepts_its_spec(self):
        self.make_factory().validate(self.make_spec())

    def test_validate_rejects_foreign_or_unknown_configuration(self):
        factory, spec = self.make_factory(), self.make_spec()
        config = spec.configuration
        for bad in (
            replace(config, backend_id=config.backend_id + "-other"),
            replace(config, schema_version=config.schema_version + 1000),
            replace(config, payload=b"not json"),
            replace(config, payload=b'{"unknown_option_xyz": 1}'),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    factory.validate(replace(spec, configuration=bad))

    # Session lifecycle --------------------------------------------------

    async def open_session(self):
        session = await self.make_factory().open(
            self.make_spec(), self.make_services()
        )
        self.addAsyncCleanup(session.close)
        return session

    async def test_snapshot_and_idempotent_close(self):
        session = await self.open_session()
        snapshot = session.snapshot()
        self.assertIsInstance(snapshot, SessionSnapshot)
        self.assertEqual(snapshot.session_id, self.make_spec().session_id)
        await self.start_contributors(session)
        snapshot = session.snapshot()
        if snapshot.state == SessionState.READY:
            capabilities = snapshot.capabilities
            self.assertLessEqual(
                capabilities.output_tokens, capabilities.context_tokens
            )
        await session.close()
        await session.close()
        self.assertIn(
            session.snapshot().state,
            {SessionState.CLOSED, SessionState.FAILED, SessionState.DRAINING},
        )

    # Contributor port (skipped for self-contained backends) ----------------

    async def port(self):
        session = await self.open_session()
        port = contributor_port(session)
        if port is None:
            self.skipTest("backend has no contributors")
        return port

    async def test_contributor_operations_are_named_and_unreserved(self):
        port = await self.port()
        operations = port.operations()
        self.assertIsInstance(operations, frozenset)
        self.assertFalse(operations & RESERVED)
        for name in operations:
            self.assertRegex(name, r"^[a-z][a-z0-9_]{0,31}$")

    async def test_contributor_rejects_unknown_operation_and_stranger(self):
        port = await self.port()
        with self.assertRaises(ContributorError) as caught:
            port.call("no-such-contributor", "no_such_operation", b"{}")
        self.assertIn(
            caught.exception.fault,
            {ContributorFault.UNSUPPORTED, ContributorFault.GONE},
        )
        for name in port.operations():
            with self.assertRaises(ContributorError):
                port.call("no-such-contributor", name, b"{}")
        self.assertFalse(port.heartbeat("no-such-contributor", b"{}"))
        port.leave("no-such-contributor", "test")  # idempotent, no error

    async def test_contributor_join_rejects_foreign_payload(self):
        port = await self.port()
        ticket = JoinTicket(0, "192.0.2.1", 1 << 30, encode({"backend": "x"}))
        try:
            reply = port.join(ticket)
        except ContributorError as error:
            self.assertEqual(error.fault, ContributorFault.INVALID)
        else:
            self.assertIsInstance(reply, JoinReply)
            self.assertFalse(reply.accepted)

    async def test_roster_and_expire_types(self):
        port = await self.port()
        for entry in port.roster():
            self.assertIsInstance(entry, RosterEntry)
        self.assertIsInstance(port.expire(), tuple)


__all__ = ["BackendConformance", "assert_stream"]
