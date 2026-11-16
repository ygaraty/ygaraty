"""Composition tests use fakes, never a real engine or service."""

import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

from test_values import session_spec

from ygaraty.backend_contracts.models import BackendDescriptor
from ygaraty.backend_contracts.persistence import RequestLedger
from ygaraty.backend_contracts.protocols import (
    AccessPolicy,
    BackendSession,
    Clock,
    RuntimeServices,
)
from ygaraty.backend_contracts.registry import BackendRegistry


class FixtureFactory:
    """A strict metadata decoder, not an inference implementation."""

    def __init__(self, backend_id="fixture", versions=(1,)):
        self.descriptor = BackendDescriptor(backend_id, "test", versions)
        self.open = AsyncMock(return_value=Mock(spec=BackendSession))
        self.validations = 0

    def describe(self):
        return self.descriptor

    def validate(self, spec):
        self.validations += 1
        if spec.configuration.schema_version != 1:
            raise ValueError("Unsupported fixture schema")
        options = json.loads(spec.configuration.payload)
        if options != {"context_tokens": 128}:
            raise ValueError("Invalid fixture configuration")


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = BackendRegistry()
        self.factory = FixtureFactory()
        self.registry.register(self.factory)
        self.services = RuntimeServices(
            Mock(spec=Clock), Mock(spec=AccessPolicy), Mock(spec=RequestLedger)
        )

    def test_discovery_does_not_open_or_validate(self):
        self.assertEqual(self.registry.describe(), (self.factory.descriptor,))
        self.assertEqual(self.factory.validations, 0)
        self.factory.open.assert_not_called()

    def test_duplicates_and_unsupported_versions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register(FixtureFactory())
        with self.assertRaises(ValueError):
            self.registry.register(FixtureFactory("other", (2,)))
        self.assertEqual(len(self.registry.describe()), 1)

    async def test_only_selected_factory_opens(self):
        other = FixtureFactory("other")
        self.registry.register(other)
        spec = session_spec()
        session = await self.registry.open(spec, self.services)
        self.assertIs(session, self.factory.open.return_value)
        self.factory.open.assert_awaited_once_with(spec, self.services)
        self.assertEqual(self.factory.validations, 1)
        self.assertEqual(other.validations, 0)
        other.open.assert_not_called()

    async def test_missing_backend_does_not_fall_back(self):
        spec = session_spec()
        spec = replace(
            spec,
            configuration=replace(spec.configuration, backend_id="missing"),
        )
        with self.assertRaises(ValueError):
            await self.registry.open(spec, self.services)
        self.factory.open.assert_not_called()

    async def test_invalid_configuration_cannot_acquire_resources(self):
        spec = session_spec()
        for changes in (
            {"schema_version": 2},
            {"payload": b'{"unknown":true}'},
            {"payload": b"not-json"},
        ):
            with self.subTest(changes=changes):
                invalid = replace(
                    spec, configuration=replace(spec.configuration, **changes)
                )
                with self.assertRaises(ValueError):
                    await self.registry.open(invalid, self.services)
                self.factory.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
