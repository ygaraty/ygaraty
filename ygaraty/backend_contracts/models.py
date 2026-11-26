"""Validated metadata without model loading, services or mutable payloads."""

from dataclasses import dataclass
from enum import StrEnum

from ._validation import immutable_items, integer, member, text

CONTRACT_MAJOR = 1


class Route(StrEnum):
    RELAY = "relay"
    DIRECT = "direct"
    PRIVATE_TURN = "private_turn"


class HistoryMode(StrEnum):
    FULL = "full"
    APPEND = "append"
    RESET = "reset"


class SessionState(StrEnum):
    PREPARING = "preparing"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DRAINING = "draining"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_id: str
    checkpoint_digest: str
    tokenizer_digest: str
    template_digest: str
    precision: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    backend_id: str
    implementation_version: str
    contract_majors: tuple[int, ...] = (CONTRACT_MAJOR,)

    def __post_init__(self) -> None:
        text(self.backend_id, "backend_id")
        text(self.implementation_version, "implementation_version")
        immutable_items(self.contract_majors, int, "contract_majors")
        if not self.contract_majors:
            raise ValueError("At least one contract major is required")
        for version in self.contract_majors:
            integer(version, "contract major", 1)
        if len(set(self.contract_majors)) != len(self.contract_majors):
            raise ValueError("Duplicate contract majors")


@dataclass(frozen=True, slots=True)
class BackendConfiguration:
    """Opaque immutable bytes. The selected factory MUST validate its schema.

    These are not credentials or permission to load arbitrary Python code.
    No generic decoder may treat this envelope as validated backend options.
    """

    backend_id: str
    schema_version: int
    payload: bytes

    def __post_init__(self) -> None:
        text(self.backend_id, "backend_id")
        integer(self.schema_version, "schema_version", 1)
        if (
            type(self.payload) is not bytes
            or not 0 < len(self.payload) <= 1 << 20
        ):
            raise ValueError(
                "Configuration must be 1..1048576 immutable bytes"
            )


@dataclass(frozen=True, slots=True)
class SessionSpec:
    session_id: str
    epoch: str
    model: ModelIdentity
    configuration: BackendConfiguration

    def __post_init__(self) -> None:
        text(self.session_id, "session_id")
        text(self.epoch, "epoch")
        if not isinstance(self.model, ModelIdentity):
            raise ValueError("Expected ModelIdentity")
        if not isinstance(self.configuration, BackendConfiguration):
            raise ValueError("Expected BackendConfiguration")


@dataclass(frozen=True, slots=True)
class EffectiveCapabilities:
    context_tokens: int
    output_tokens: int
    active_requests: int
    queued_requests: int
    history_modes: tuple[HistoryMode, ...]
    routes: tuple[Route, ...]
    browser_contribution: bool

    def __post_init__(self) -> None:
        for name in ("context_tokens", "output_tokens", "active_requests"):
            integer(getattr(self, name), name, 1)
        integer(self.queued_requests, "queued_requests")
        if self.output_tokens > self.context_tokens:
            raise ValueError("Output limit exceeds context capacity")
        for name, kind in (("history_modes", HistoryMode), ("routes", Route)):
            value = getattr(self, name)
            immutable_items(value, kind, name)
            if not value or len(set(value)) != len(value):
                raise ValueError(f"{name} must be nonempty and unique")
        if type(self.browser_contribution) is not bool:
            raise ValueError("browser_contribution must be boolean")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    session_id: str
    epoch: str
    revision: int
    state: SessionState
    capabilities: EffectiveCapabilities | None = None
    # Short human-readable reason for UNAVAILABLE/FAILED, safe to show users.
    detail: str | None = None

    def __post_init__(self) -> None:
        text(self.session_id, "session_id")
        text(self.epoch, "epoch")
        integer(self.revision, "revision")
        member(self.state, SessionState)
        if self.detail is not None:
            text(self.detail, "detail", 500)
        if self.capabilities is not None and not isinstance(
            self.capabilities, EffectiveCapabilities
        ):
            raise ValueError("Expected EffectiveCapabilities")
        if self.state == SessionState.READY and self.capabilities is None:
            raise ValueError(
                "Ready sessions must report effective capabilities"
            )
