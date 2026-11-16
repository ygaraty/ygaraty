"""Backend-neutral text requests and ordered stream events."""

import re
from dataclasses import dataclass
from enum import StrEnum

from ._validation import immutable_items, integer, member, text
from .lifecycle import CancelReceipt
from .models import HistoryMode


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        member(self.role, Role)
        text(self.content, "content", 100000)


@dataclass(frozen=True, slots=True)
class ChatRequest:
    request_id: str
    conversation_id: str
    model_id: str
    history_mode: HistoryMode
    messages: tuple[Message, ...]
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        text(self.request_id, "request_id")
        text(self.model_id, "model_id")
        text(self.conversation_id, "conversation_id", 64)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.conversation_id):
            raise ValueError("Invalid conversation ID")
        member(self.history_mode, HistoryMode)
        immutable_items(self.messages, Message, "messages")
        if len(self.messages) > 128:
            raise ValueError("At most 128 messages")
        if not self.messages and self.history_mode != HistoryMode.RESET:
            raise ValueError("Messages are required outside reset")
        if self.messages and self.messages[-1].role != Role.USER:
            raise ValueError("The final message must be from the user")
        if self.max_output_tokens is not None:
            integer(self.max_output_tokens, "max_output_tokens", 1)


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONTEXT_EXCEEDED = "context_exceeded"
    NOT_READY = "not_ready"
    CONVERSATION_BUSY = "conversation_busy"
    OVERLOADED = "overloaded"
    ACCESS_REVOKED = "access_revoked"
    STALE_EPOCH = "stale_epoch"
    CONTRIBUTOR_LOST = "contributor_lost"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    EXECUTION_FAILED = "execution_failed"
    STATE_RESET_REQUIRED = "state_reset_required"


@dataclass(frozen=True, slots=True)
class Failure:
    code: ErrorCode
    message: str
    retry_after_seconds: int | None = None

    def __post_init__(self) -> None:
        member(self.code, ErrorCode)
        text(self.message, "safe error message", 1024)
        if self.retry_after_seconds is not None:
            integer(self.retry_after_seconds, "retry_after_seconds")


class BackendError(Exception):
    def __init__(self, failure: Failure):
        if not isinstance(failure, Failure):
            raise ValueError("Expected Failure")
        self.failure = failure
        super().__init__(failure.message)


class EventKind(StrEnum):
    ACCEPTED = "accepted"
    TEXT = "text"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        integer(self.prompt_tokens, "prompt_tokens")
        integer(self.output_tokens, "output_tokens")


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    request_id: str
    sequence: int
    kind: EventKind
    text: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    failure: Failure | None = None
    cancellation: CancelReceipt | None = None

    def __post_init__(self) -> None:
        text(self.request_id, "request_id")
        integer(self.sequence, "sequence")
        member(self.kind, EventKind)
        if self.kind == EventKind.TEXT:
            text(self.text, "text delta", 100000)
        elif self.text != "":
            raise ValueError("Only text events carry text")
        if self.kind == EventKind.COMPLETED:
            member(self.finish_reason, FinishReason)
            if not isinstance(self.usage, Usage):
                raise ValueError("Completed events require usage")
        elif self.finish_reason is not None or self.usage is not None:
            raise ValueError("Only completed events carry final usage/reason")
        if self.kind == EventKind.FAILED:
            if not isinstance(self.failure, Failure):
                raise ValueError("Failed events require Failure")
        elif self.failure is not None:
            raise ValueError("Only failed events carry Failure")
        if self.kind == EventKind.CANCELLED:
            if (
                not isinstance(self.cancellation, CancelReceipt)
                or self.cancellation.request_id != self.request_id
            ):
                raise ValueError("Cancelled events require matching receipt")
        elif self.cancellation is not None:
            raise ValueError("Only cancelled events carry a cancel receipt")


class EventSequence:
    """Per-handle stream validator; not a durable delivery guarantee."""

    def __init__(self, request_id: str):
        text(request_id, "request_id")
        self.request_id = request_id
        self._next = 0
        self._terminal = False

    def accept(self, event: GenerationEvent) -> None:
        if (
            not isinstance(event, GenerationEvent)
            or self._terminal
            or event.request_id != self.request_id
            or event.sequence != self._next
            or (event.kind == EventKind.ACCEPTED) != (self._next == 0)
        ):
            raise ValueError("Invalid event identity, ordering or lifecycle")
        self._next += 1
        self._terminal = event.kind in {
            EventKind.COMPLETED,
            EventKind.CANCELLED,
            EventKind.FAILED,
        }

    def finish(self) -> None:
        if not self._terminal:
            raise ValueError("Stream ended without a terminal event")
