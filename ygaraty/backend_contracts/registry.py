"""Explicit composition without plugin scanning or implicit private imports."""

from .models import CONTRACT_MAJOR, BackendDescriptor, SessionSpec
from .protocols import BackendFactory, BackendSession, RuntimeServices


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}
        self._descriptors: dict[str, BackendDescriptor] = {}

    def register(self, factory: BackendFactory) -> None:
        descriptor = factory.describe()
        if not isinstance(descriptor, BackendDescriptor):
            raise ValueError("Expected BackendDescriptor")
        if CONTRACT_MAJOR not in descriptor.contract_majors:
            raise ValueError("Incompatible backend contract major")
        if descriptor.backend_id in self._factories:
            raise ValueError("Duplicate backend ID")
        self._factories[descriptor.backend_id] = factory
        self._descriptors[descriptor.backend_id] = descriptor

    def describe(self) -> tuple[BackendDescriptor, ...]:
        return tuple(self._descriptors.values())

    async def open(
        self, spec: SessionSpec, services: RuntimeServices
    ) -> BackendSession:
        try:
            factory = self._factories[spec.configuration.backend_id]
        except KeyError:
            raise ValueError("Backend is not registered") from None
        factory.validate(spec)
        return await factory.open(spec, services)
