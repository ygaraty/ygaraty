"""Host-authorized admission around the existing batched WebGPU session."""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor

from ygaraty.backend_contracts.access import AccessGrant, Action
from ygaraty.backend_contracts.chat import (
    BackendError,
    ChatRequest,
    ErrorCode,
    Failure,
    Role,
)
from ygaraty.backend_contracts.lifecycle import Revocation
from ygaraty.backend_contracts.models import (
    EffectiveCapabilities,
    HistoryMode,
    SessionSnapshot,
    SessionState,
)
from ygaraty.backend_contracts.persistence import (
    RequestKey,
    RequestOutcome,
    RequestRecord,
    StoreConflict,
)

from .handles import WebGPUHandle


def reject(code, message):
    raise BackendError(Failure(code, message))


class WebGPUSession:
    request_timeout = 120
    revalidate_interval = 1
    close_timeout = 2

    def __init__(self, spec, config, services, engine):
        self.spec, self.config, self.services, self.engine = (
            spec,
            config,
            services,
            engine,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=config.conversation_slots,
            thread_name_prefix="webgpu-chat",
        )
        self.handles = {}
        self.unrecorded = {}
        self.quarantined = self.closed = False
        self._lock = asyncio.Lock()
        self._encoding = None
        self._revision = 0
        self._last_state = None

    def snapshot(self) -> SessionSnapshot:
        state = (
            SessionState.READY
            if self.engine.state == "running"
            else SessionState.UNAVAILABLE
        )
        if self.closed:
            state = SessionState.CLOSED
        elif self.quarantined or self.unrecorded:
            state = SessionState.DRAINING
        if state != self._last_state:
            self._revision += 1
            self._last_state = state
        capabilities = None
        if state == SessionState.READY:
            capabilities = EffectiveCapabilities(
                self.engine.capacity,
                self.engine.capacity,
                self.config.conversation_slots,
                0,
                tuple(HistoryMode),
                (self.config.route,),
                True,
            )
        return SessionSnapshot(
            self.spec.session_id,
            self.spec.epoch,
            self._revision,
            state,
            capabilities,
        )

    async def authorize(self, grant):
        grant.require(
            session_id=self.spec.session_id,
            epoch=self.spec.epoch,
            action=Action.GENERATE,
            now=self.services.clock.unix_seconds(),
        )
        await self.services.access.validate(
            grant,
            session_id=self.spec.session_id,
            epoch=self.spec.epoch,
            action=Action.GENERATE,
        )

    def _identity(self, request, grant):
        import json

        parts = [
            self.spec.session_id,
            self.spec.epoch,
            grant.principal_id,
            request.conversation_id,
        ]
        return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

    def _arguments(self, request, identity):
        from ._legacy.chat_contract import chat_prompt, encode_ids

        if request.model_id != self.spec.model.model_id:
            reject(ErrorCode.INVALID_REQUEST, "Model does not match session")
        arguments = dict(
            message="",
            conversation_id=identity,
            reset=request.history_mode == HistoryMode.RESET,
        )
        if request.history_mode == HistoryMode.FULL:
            messages = [(m.role.value, m.content) for m in request.messages]
            tokens = encode_ids(
                self.engine.tokenizer,
                chat_prompt(messages, self.engine.chat_style),
            )
            arguments["prompt_tokens"] = tokens
        else:
            if request.messages and (
                len(request.messages) != 1
                or request.messages[0].role != Role.USER
            ):
                reject(
                    ErrorCode.INVALID_REQUEST,
                    "Append/reset accepts one user message",
                )
            if not request.messages:
                return arguments
            arguments["message"] = request.messages[0].content
            pool = self.engine.concurrent_chat
            with pool.lock:
                history = (
                    []
                    if arguments["reset"]
                    else list(pool.histories.get(identity, []))
                )
            tokens = encode_ids(
                self.engine.tokenizer,
                chat_prompt(
                    history + [("user", arguments["message"])],
                    self.engine.chat_style,
                ),
            )
        remaining = self.engine.capacity - len(tokens)
        if remaining <= 0:
            reject(
                ErrorCode.CONTEXT_EXCEEDED, "Prompt fills the context window"
            )
        arguments["max_tokens"] = min(
            request.max_output_tokens or remaining, remaining
        )
        return arguments

    async def admit(
        self, request: ChatRequest, grant: AccessGrant
    ) -> WebGPUHandle:
        try:
            await asyncio.wait_for(self.authorize(grant), 2)
        except Exception:
            reject(ErrorCode.ACCESS_REVOKED, "Access is no longer valid")
        async with self._lock:
            if self.snapshot().state != SessionState.READY:
                reject(ErrorCode.NOT_READY, "Session is not ready")
            identity = self._identity(request, grant)
            if request.request_id in self.handles:
                reject(
                    ErrorCode.INVALID_REQUEST, "Duplicate active request ID"
                )
            if any(
                h.arguments["conversation_id"] == identity
                for h in self.handles.values()
            ):
                reject(ErrorCode.CONVERSATION_BUSY, "Conversation is busy")
            owned = sum(
                h.grant.principal_id == grant.principal_id
                for h in self.handles.values()
            )
            if len(
                self.handles
            ) >= self.config.conversation_slots or owned >= min(
                grant.max_active, grant.max_outstanding
            ):
                reject(
                    ErrorCode.OVERLOADED, "Session request capacity reached"
                )
            # Encoding must leave both the GIL and event loop free.
            # Retain one encoding reservation per session.
            if self._encoding is not None and not self._encoding.done():
                reject(ErrorCode.OVERLOADED, "Prompt encoding is still active")
            self._encoding = asyncio.get_running_loop().run_in_executor(
                self.executor, self._arguments, request, identity
            )
            # Cancellation cannot stop Rust encoding. Retain its reservation
            # until it finishes, and retrieve errors even if the caller left.
            self._encoding.add_done_callback(
                lambda future: None
                if future.cancelled()
                else future.exception()
            )
            arguments = await asyncio.shield(self._encoding)
            if self.snapshot().state != SessionState.READY:
                reject(ErrorCode.NOT_READY, "Session is not ready")
            key = RequestKey(
                self.spec.session_id, self.spec.epoch, request.request_id
            )
            record = RequestRecord(
                key, grant.principal_id, request.conversation_id
            )
            try:
                await asyncio.wait_for(self.services.requests.admit(record), 2)
            except StoreConflict:
                reject(
                    ErrorCode.INVALID_REQUEST, "Duplicate or stale admission"
                )
            except BaseException as error:
                try:
                    await asyncio.shield(
                        asyncio.wait_for(
                            self.services.requests.finish(
                                key, RequestOutcome.FAILED
                            ),
                            2,
                        )
                    )
                except StoreConflict:
                    pass  # No record was inserted, or an earlier outcome won.
                except Exception:
                    self.record_failure(key, RequestOutcome.FAILED)
                if isinstance(error, asyncio.CancelledError):
                    raise
                reject(
                    ErrorCode.EXECUTION_FAILED, "Admission recording failed"
                )
            # Revalidate after the awaited store write, before dispatch.
            try:
                await asyncio.wait_for(self.authorize(grant), 2)
            except BaseException as error:
                try:
                    await asyncio.wait_for(
                        self.services.requests.finish(
                            key, RequestOutcome.FAILED
                        ),
                        2,
                    )
                except Exception:
                    self.record_failure(key, RequestOutcome.FAILED)
                if isinstance(error, asyncio.CancelledError):
                    raise
                reject(ErrorCode.ACCESS_REVOKED, "Access is no longer valid")
            handle = WebGPUHandle(self, request, grant, key, arguments)
            self.handles[request.request_id] = handle
            return handle

    def record_failure(self, key, outcome):
        self.unrecorded[key] = outcome

    def release(self, handle):
        if self.handles.get(handle.request_id) is handle:
            self.handles.pop(handle.request_id)

    async def revoke(self, scope: Revocation) -> None:
        if (scope.session_id, scope.epoch) != (
            self.spec.session_id,
            self.spec.epoch,
        ):
            return
        from ygaraty.backend_contracts.lifecycle import CancelReason

        for handle in list(self.handles.values()):
            if (
                (
                    scope.principal_id is None
                    or scope.principal_id == handle.grant.principal_id
                )
                and (
                    scope.grant_id is None
                    or scope.grant_id == handle.grant.grant_id
                )
                and (
                    scope.conversation_id is None
                    or scope.conversation_id == handle.request.conversation_id
                )
            ):
                await handle.cancel(CancelReason.REVOKED)

    async def close(self) -> None:
        async with self._lock:
            if self.closed:
                return
            self.closed = True
        await asyncio.gather(
            *(h.aclose() for h in list(self.handles.values()))
        )
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.engine.close), self.close_timeout
            )
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
