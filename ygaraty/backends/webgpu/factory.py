"""Local-bundle factory without platform imports or provider provisioning."""

import asyncio
from pathlib import Path

from ygaraty.backend_contracts.models import (
    BackendDescriptor,
    Route,
    SessionSpec,
)
from ygaraty.backend_contracts.protocols import RuntimeServices

from .bundle_validation import remote_base
from .configuration import WebGPUConfig, json_object
from .preparation import prepare_session
from .session import WebGPUSession


def load_engine(path, output, config, metadata, *, turn=None):
    from ._legacy.prepared import PreparedSession

    # __new__ lets failed initialization unwind handles acquired before errors.
    engine = PreparedSession.__new__(PreparedSession)
    try:
        engine.__init__(
            path,
            output,
            [{"layers": list(bounds)} for bounds in config.stage_layers],
            config.memory_budget_bytes / 1024**3,
            route={
                Route.RELAY: "relay",
                Route.DIRECT: "webrtc",
                Route.PRIVATE_TURN: "private",
            }[config.route],
            turn=turn,
            context_window=config.context_tokens,
            conversation_slots=config.conversation_slots,
            metadata=metadata,
        )
        # Chat/API ingress stays on HTTP; only contributor links use WebRTC.
        for stage in engine.stages:
            stage["fixture"]["chat_direct"] = False
    except BaseException:
        log = getattr(engine, "log", None)
        if log is not None:
            log.close()
        raise
    return engine


class WebGPUFactory:
    def __init__(
        self,
        bundle: Path,
        output: Path,
        *,
        engine_loader=load_engine,
        turn=None,
    ):
        self.bundle = Path(bundle).resolve()
        self.output = Path(output).resolve()
        self._loader = engine_loader
        self.turn = turn

    def describe(self) -> BackendDescriptor:
        return BackendDescriptor("webgpu", "0.1.0.dev0")

    def validate(self, spec: SessionSpec) -> None:
        config = WebGPUConfig.from_contract(spec.configuration)
        if config.route == Route.PRIVATE_TURN and self.turn is None:
            raise ValueError(
                "TURN-only routes require a configured TURN server"
            )

    async def open(
        self, spec: SessionSpec, services: RuntimeServices
    ) -> WebGPUSession:
        self.validate(spec)
        config = WebGPUConfig.from_contract(spec.configuration)
        if self.bundle.stat().st_size > 16 << 20:
            raise ValueError("Bundle exceeds 16 MiB")
        raw = self.bundle.read_bytes()
        prepare_session(spec, bundle=raw)
        metadata = json_object(raw)
        remote_base(metadata)  # local, or public HTTPS weight pieces
        # Retain the worker until initialization completes, even on cancel.
        loading = asyncio.create_task(
            asyncio.to_thread(
                self._loader,
                self.bundle,
                self.output,
                config,
                metadata,
                **({"turn": self.turn} if config.route != Route.RELAY else {}),
            )
        )
        try:
            engine = await asyncio.shield(loading)
        except asyncio.CancelledError:
            engine = await loading
            await asyncio.to_thread(engine.close)
            raise
        return WebGPUSession(spec, config, services, engine)
