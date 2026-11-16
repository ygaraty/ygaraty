"""Asynchronous execution and host ports, independent of web frameworks."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .access import AccessGrant, Action
from .chat import ChatRequest, GenerationEvent
from .lifecycle import CancelReason, CancelReceipt, Revocation
from .models import BackendDescriptor, SessionSnapshot, SessionSpec
from .persistence import RequestLedger


class Clock(Protocol):
    def unix_seconds(self) -> int: ...

    def monotonic_seconds(self) -> float: ...


class AccessPolicy(Protocol):
    async def validate(
        self,
        grant: AccessGrant,
        *,
        session_id: str,
        epoch: str,
        action: Action,
    ) -> None:
        """Check scope, expiry AND live membership/key/grant versions.

        Fail closed on rejection or unavailability. Check at admission, before
        queued dispatch and periodically during execution. Grants originate
        in the host's authenticated context, never directly from HTTP bodies.
        """
        ...


@dataclass(frozen=True, slots=True)
class RuntimeServices:
    clock: Clock
    access: AccessPolicy
    requests: RequestLedger


class RequestHandle(Protocol):
    @property
    def request_id(self) -> str: ...

    def events(self) -> AsyncIterator[GenerationEvent]:
        """Single consumer, bounded buffering, one logical terminal outcome."""
        ...

    async def cancel(self, reason: CancelReason) -> CancelReceipt:
        """Idempotent signal, with a separate termination acknowledgement.

        Host authenticates the handle owner before exposing this operation.
        Keep cleanup usable when the authorization store is unavailable.
        """
        ...

    async def aclose(self) -> None:
        """Idempotently drain and release, or quarantine uncertain state."""
        ...


class BackendSession(Protocol):
    def snapshot(self) -> SessionSnapshot: ...

    async def admit(
        self, request: ChatRequest, grant: AccessGrant
    ) -> RequestHandle:
        """Validate, authorize, reserve bounded capacity and persist admission.

        Return before streaming headers; unwind reservations on failure.
        Tokenization/context/state ownership remain backend responsibilities.
        """
        ...

    async def revoke(self, scope: Revocation) -> None:
        """Trusted host signal, affecting matching queued and running work."""
        ...

    async def close(self) -> None:
        """Idempotently stop admission, drain and release with a deadline."""
        ...


class BackendFactory(Protocol):
    def describe(self) -> BackendDescriptor:
        """Discover metadata without loading models or starting resources."""
        ...

    def validate(self, spec: SessionSpec) -> None:
        """Reject unsupported schemas, unknown options and invalid bounds.

        No resource acquisition. Own the decoder for configuration.payload;
        the contracts package validates only its immutable envelope.
        """
        ...

    async def open(
        self, spec: SessionSpec, services: RuntimeServices
    ) -> BackendSession:
        """Prepare resources, never rent them; unwind partial failure.

        Open does not imply readiness. Factories must validate even when called
        directly, outside BackendRegistry.
        """
        ...
