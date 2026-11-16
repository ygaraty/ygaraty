"""Host-owned durable metadata; never persist live execution objects here."""

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from ._validation import integer, member, text
from .models import SessionSpec


class StoreConflict(Exception):
    """Duplicate ID, stale epoch/revision or conflicting terminal write."""


class RequestOutcome(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RequestKey:
    session_id: str
    epoch: str
    request_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class RequestRecord:
    key: RequestKey
    principal_id: str
    conversation_id: str
    outcome: RequestOutcome | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, RequestKey):
            raise ValueError("Expected RequestKey")
        text(self.principal_id, "principal_id")
        text(self.conversation_id, "conversation_id", 64)
        if self.outcome is not None:
            member(self.outcome, RequestOutcome)

    def finish(self, outcome: RequestOutcome) -> "RequestRecord":
        """Deterministic terminal transition; replaying the outcome is safe."""
        member(outcome, RequestOutcome)
        if self.outcome is not None and self.outcome != outcome:
            raise StoreConflict("Request already has a different outcome")
        return replace(self, outcome=outcome)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Desired configuration, not a claim that runtime resources are ready."""

    spec: SessionSpec
    revision: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.spec, SessionSpec):
            raise ValueError("Expected SessionSpec")
        integer(self.revision, "revision")
        if type(self.closed) is not bool:
            raise ValueError("closed must be boolean")


class RequestLedger(Protocol):
    async def admit(self, record: RequestRecord) -> None:
        """Atomically insert unfinished record for an open, current epoch.

        Duplicate key, stale/closed/missing session or terminal input raises
        StoreConflict. A successful return acknowledges configured durability.
        No dispatch is permitted before this succeeds.
        """
        ...

    async def finish(
        self, key: RequestKey, outcome: RequestOutcome
    ) -> RequestRecord:
        """Atomically finish an existing record; same outcome is idempotent.

        Missing keys or conflicting outcomes raise StoreConflict. This must
        never create a record or resurrect an old-epoch interrupted request.
        """
        ...


class SessionStore(RequestLedger, Protocol):
    """Optional host adapter implemented by Coopy, SQL or another local store.

    Implementations serialize each mutation atomically, including across the
    session and request records during restart. Storage failures propagate;
    they must not silently degrade into successful in-memory-only writes.
    """

    async def create(self, record: SessionRecord) -> None:
        """Insert revision-zero open session, rejecting existing IDs."""
        ...

    async def get(self, session_id: str) -> SessionRecord | None: ...

    async def close(
        self, session_id: str, *, expected_revision: int
    ) -> SessionRecord:
        """CAS close and increment revision; reject stale/missing records.

        Stop new admissions; existing work is drained by the host separately.
        Already closed with matching revision returns the unchanged record.
        """
        ...

    async def restart(
        self, session_id: str, *, expected_revision: int, new_epoch: str
    ) -> SessionRecord:
        """CAS rotate an open session epoch and increment its revision.

        Atomically mark unfinished old-epoch requests INTERRUPTED. Reject a
        missing/closed session, stale revision or previously used epoch. Never
        replay generation. The host opens fresh runtime resources afterwards.
        """
        ...
