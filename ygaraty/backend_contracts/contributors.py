"""Contributors: the machines that run a session's model work.

A host (a self-hosted server, or a commercial platform such as Kaa) owns the
transport, authentication and reservations: it decides which contributor may
hold which slot, and passes each authenticated contributor call to the
backend's ContributorPort. The backend owns everything inside a call: its
message formats, model identity checks, work dispatch and results.

Messages cross the boundary as immutable JSON-object bytes. The host never
decodes or edits a backend's payload; the backend never sees the host's
accounts, keys or database. Every backend that uses contributors (vLLM,
llama.cpp, WebGPU, ...) implements the same port, so a host needs one set of
contributor routes for all of them.

The port is synchronous: hosts usually call it from a request thread. Each
method must return within a few seconds; long polls are bounded by the backend.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ._validation import integer, member, text

MESSAGE_LIMIT = 1 << 20
_OPERATION = re.compile(r"[a-z][a-z0-9_]{0,31}")
# Lifecycle steps every port has; `call` carries only backend operations.
RESERVED = frozenset({"join", "heartbeat", "leave"})


def message(payload: bytes, name: str = "payload") -> None:
    """Require a UTF-8 JSON object of at most MESSAGE_LIMIT bytes."""
    if type(payload) is not bytes or len(payload) > MESSAGE_LIMIT:
        raise ValueError(f"{name} must be at most {MESSAGE_LIMIT} bytes")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, ValueError):
        raise ValueError(f"{name} must be a UTF-8 JSON object") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a UTF-8 JSON object")


def encode(value: dict) -> bytes:
    """Encode a JSON object as a contributor message."""
    payload = json.dumps(value, separators=(",", ":")).encode()
    message(payload)
    return payload


def decode(payload: bytes) -> dict:
    message(payload)
    return json.loads(payload)


class ContributorState(StrEnum):
    PREPARING = "preparing"  # joined, loading its model share
    ACTIVE = "active"  # ready for work
    RETIRED = "retired"  # left, failed or expired; never reused


class ContributorFault(StrEnum):
    INVALID = "invalid"  # malformed message or wrong model identity
    UNSUPPORTED = "unsupported"  # unknown operation for this backend
    NOT_READY = "not_ready"  # valid, but the session or contributor isn't ready
    GONE = "gone"  # reservation ended: the contributor must join again


class ContributorError(Exception):
    """A rejected contributor call; hosts map faults to transport errors."""

    def __init__(self, fault: ContributorFault, message: str):
        member(fault, ContributorFault)
        text(message, "message", 500)
        super().__init__(message)
        self.fault = fault
        self.message = message


@dataclass(frozen=True, slots=True)
class JoinTicket:
    """A host-authenticated contributor asking to take one reserved slot.

    `previous_id` is the contributor ID the host recorded for this slot, if
    any: a backend that still knows it may return the same ID (a reconnect).
    `peer` is the network address the host observed, never one the
    contributor claimed.
    """

    slot: int
    peer: str
    memory_budget_bytes: int
    payload: bytes
    previous_id: str | None = None

    def __post_init__(self) -> None:
        integer(self.slot, "slot")
        text(self.peer, "peer")
        integer(self.memory_budget_bytes, "memory_budget_bytes", 1)
        message(self.payload)
        if self.previous_id is not None:
            text(self.previous_id, "previous_id", 64)


@dataclass(frozen=True, slots=True)
class JoinReply:
    contributor_id: str
    accepted: bool
    payload: bytes

    def __post_init__(self) -> None:
        text(self.contributor_id, "contributor_id", 64)
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be boolean")
        message(self.payload)


@dataclass(frozen=True, slots=True)
class RosterEntry:
    """What a host may show and persist about one contributor.

    `progress` and `connectivity` are backend-written JSON objects for display
    (loading phase, link routes); hosts store them but don't interpret them.
    """

    contributor_id: str
    slot: int
    state: ContributorState
    progress: bytes = b"{}"
    connectivity: bytes = b"{}"

    def __post_init__(self) -> None:
        text(self.contributor_id, "contributor_id", 64)
        integer(self.slot, "slot")
        member(self.state, ContributorState)
        message(self.progress, "progress")
        message(self.connectivity, "connectivity")


class ContributorPort(Protocol):
    def operations(self) -> frozenset[str]:
        """Backend operation names accepted by `call` (lowercase, no
        RESERVED names). Discoverable without a contributor."""
        ...

    def join(self, ticket: JoinTicket) -> JoinReply:
        """Take the ticket's slot, or reply accepted=False when it can't.

        Raise ContributorError(INVALID) for a malformed or foreign payload.
        """
        ...

    def heartbeat(self, contributor_id: str, payload: bytes) -> bool:
        """Keep the reservation alive; payload may carry loading progress.

        False means the reservation ended and the contributor must join again.
        """
        ...

    def call(self, contributor_id: str, operation: str, payload: bytes) -> bytes:
        """Run one backend operation (e.g. ready, claim work, send a result).

        Raise ContributorError: UNSUPPORTED for unknown operations, GONE for
        retired contributors, NOT_READY or INVALID otherwise.
        """
        ...

    def leave(self, contributor_id: str, reason: str) -> None:
        """Retire the contributor (idempotent); fail or requeue its work."""
        ...

    def roster(self) -> tuple[RosterEntry, ...]:
        """Current contributors, for the host's slot records and status."""
        ...

    def expire(self) -> tuple[str, ...]:
        """Retire contributors whose heartbeats stopped; return their IDs.

        Hosts call this periodically (every few seconds).
        """
        ...


class ContributorSession(Protocol):
    """A BackendSession whose work runs on contributors."""

    def contributors(self) -> ContributorPort: ...


def contributor_port(session: object) -> ContributorPort | None:
    """The session's ContributorPort, or None for a self-contained backend."""
    method = getattr(session, "contributors", None)
    return method() if callable(method) else None


def operation(name: str) -> None:
    if not isinstance(name, str) or not _OPERATION.fullmatch(name) or name in RESERVED:
        raise ContributorError(
            ContributorFault.UNSUPPORTED, "Unsupported contributor operation"
        )
