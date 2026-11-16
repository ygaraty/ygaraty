"""Host-issued access ceilings; never trust grants supplied by HTTP clients."""

from dataclasses import dataclass
from enum import StrEnum

from ._validation import integer, text


class Action(StrEnum):
    GENERATE = "generate"
    CANCEL = "cancel"
    CONTRIBUTE = "contribute"
    MANAGE = "manage"


@dataclass(frozen=True, slots=True)
class AccessGrant:
    session_id: str
    epoch: str
    principal_id: str
    grant_id: str
    version: int
    expires_at: int
    actions: frozenset[Action]
    max_active: int
    max_outstanding: int

    def __post_init__(self) -> None:
        for name in ("session_id", "epoch", "principal_id", "grant_id"):
            text(getattr(self, name), name)
        for name in ("version", "expires_at", "max_active", "max_outstanding"):
            integer(getattr(self, name), name, 1)
        if self.max_active > self.max_outstanding:
            raise ValueError("Active ceiling exceeds outstanding ceiling")
        if (
            not isinstance(self.actions, frozenset)
            or not self.actions
            or any(not isinstance(action, Action) for action in self.actions)
        ):
            raise ValueError("actions must be a nonempty frozenset of Action")

    def require(
        self, *, session_id: str, epoch: str, action: Action, now: int
    ) -> None:
        """Check local scope/expiry, NOT authenticity or live revocation.

        The host AccessPolicy must also validate current grant/key/membership
        versions. Time is explicit so checks can be tested deterministically.
        """
        integer(now, "now")
        if not isinstance(action, Action):
            raise ValueError("Expected Action")
        if (
            self.session_id != session_id
            or self.epoch != epoch
            or action not in self.actions
            or now >= self.expires_at
        ):
            raise PermissionError("Access scope or expiry rejected")
