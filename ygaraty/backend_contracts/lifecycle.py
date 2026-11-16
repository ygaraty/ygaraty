"""Control outcomes distinguish accepted cancellation from stopped compute."""

from dataclasses import dataclass
from enum import StrEnum

from ._validation import member, text


class CancelReason(StrEnum):
    CLIENT_REQUEST = "client_request"
    DISCONNECTED = "disconnected"
    REVOKED = "revoked"
    DEADLINE = "deadline"
    SESSION_CLOSED = "session_closed"


class CancelStatus(StrEnum):
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class CancelReceipt:
    request_id: str
    reason: CancelReason
    status: CancelStatus

    def __post_init__(self) -> None:
        text(self.request_id, "request_id")
        member(self.reason, CancelReason)
        member(self.status, CancelStatus)


@dataclass(frozen=True, slots=True)
class Revocation:
    """Trusted host control: supplied filters intersect; none targets epoch."""

    session_id: str
    epoch: str
    principal_id: str | None = None
    grant_id: str | None = None
    conversation_id: str | None = None

    def __post_init__(self) -> None:
        text(self.session_id, "session_id")
        text(self.epoch, "epoch")
        for name in ("principal_id", "grant_id", "conversation_id"):
            value = getattr(self, name)
            if value is not None:
                text(value, name)
