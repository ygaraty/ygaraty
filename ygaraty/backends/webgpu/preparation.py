"""Compile metadata without loading tensors, services or runtime resources."""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass

from ygaraty.backend_contracts.models import (
    SessionSnapshot,
    SessionSpec,
    SessionState,
)

from .configuration import WebGPUConfig, integer, json_object
from .distribution import compile_distribution
from .operations import legacy_stage


def validate_distribution_plan(stored: dict, compiled: dict) -> None:
    """Retain exact current and pre-contract v2 compatibility."""
    if stored == compiled:
        return
    if any("tasks" not in stage for stage in compiled["stages"]):  # DO-95: inkling2 plans have no v2 form
        raise ValueError("Stored distribution plan differs from stages")
    legacy = dict(
        compiled,
        version=2,
        stages=[legacy_stage(stage) for stage in compiled["stages"]],
    )
    if stored != legacy:
        raise ValueError("Stored distribution plan differs from stages")


def _validate_bundle(data: dict, config: WebGPUConfig) -> list[dict]:
    if data.get("format") != "ygaraty-webgpu-v1":
        raise ValueError("Unsupported WebGPU bundle format")
    options = data["options"]
    capacity = options["capacity"]
    integer(capacity, "bundle context capacity")
    integer(options["chunk"], "prefill chunk")
    if config.context_tokens > capacity:
        raise ValueError("Requested context exceeds bundle capacity")
    stages = data["stages"]
    if not isinstance(stages, list) or len(stages) != len(config.stage_layers):
        raise ValueError("Stage count differs from configuration")
    for index, (stage, bounds) in enumerate(zip(stages, config.stage_layers)):
        fixture = stage["fixture"]
        if (
            type(stage["stage"]) is not int
            or stage["stage"] != index
            or stage["layers"] != list(bounds)
            or fixture.get("route") != "relay"
            or type(fixture.get("conversations")) is not int
            or fixture["conversations"] != 1
            or fixture.get("ice")
        ):
            raise ValueError("Invalid stored stage assignment or route")
        limit = fixture.get("max_positions", capacity)
        integer(limit, "stage capacity")
        if limit < capacity:
            raise ValueError("Bundle context exceeds stage capacity")
        lengths = [data["files"][name]["length"] for name in stage["names"]]
        for length in lengths:
            integer(length, "weight range length", 2**53 - 1)
        if sum(lengths) > config.memory_budget_bytes:
            raise ValueError("Stage weights exceed contribution budget")
        # Same metadata overrides as PreparedSession; no kernel policy changes.
        fixture["memory_budget_bytes"] = config.memory_budget_bytes
        fixture["max_positions"] = config.context_tokens
        tokens = fixture.get("max_tokens", options["chunk"])
        integer(tokens, "prefill tokens")
        fixture["max_tokens"] = min(tokens, config.context_tokens)
    return stages


@dataclass(frozen=True, slots=True, init=False)
class PreparedWebGPU:
    """Immutable preparation result; exported dictionaries are independent."""

    spec: SessionSpec
    config: WebGPUConfig
    _payload: bytes

    def __new__(cls):
        raise TypeError("Use prepare_session(spec, bundle=...)")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self._payload).hexdigest()

    def to_dict(self) -> dict:
        return json.loads(self._payload)

    def snapshot(self) -> SessionSnapshot:
        """Only an executing adapter may attest to readiness/capacity."""
        return SessionSnapshot(
            self.spec.session_id, self.spec.epoch, 0, SessionState.PREPARING
        )


def prepare_session(spec: SessionSpec, *, bundle: bytes) -> PreparedWebGPU:
    """Validate an approved bundle and freshly compile its distribution.

    The trusted host supplies model identity and approved manifest bytes. The
    digest pins those bytes; it does not verify weights or tokenizer contents.
    Route/slot policy is retained as a request for the future runtime adapter.
    """
    if not isinstance(spec, SessionSpec):
        raise ValueError("Expected SessionSpec")
    config = WebGPUConfig.from_contract(spec.configuration)
    if type(bundle) is not bytes or not 0 < len(bundle) <= 16 << 20:
        raise ValueError("Bundle must be 1..16777216 immutable bytes")
    if hashlib.sha256(bundle).hexdigest() != config.bundle_sha256:
        raise ValueError("Bundle digest mismatch")
    try:
        data = json_object(bundle)
        stages = _validate_bundle(data, config)
        compiled = compile_distribution(stages)
        if "distribution_plan" in data:
            try:
                validate_distribution_plan(data["distribution_plan"], compiled)
            except ValueError:
                # Prepared before the gates rows were padded; the fresh
                # (padded) plan is what runs.
                validate_distribution_plan(
                    data["distribution_plan"],
                    compile_distribution(
                        deepcopy(stages), aligned_gates=False
                    ),
                )
        payload = json.dumps(
            {"stages": stages, "distribution_plan": compiled},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    except (KeyError, TypeError, AttributeError, RecursionError) as error:
        raise ValueError("Malformed WebGPU bundle metadata") from error
    prepared = object.__new__(PreparedWebGPU)
    object.__setattr__(prepared, "spec", spec)
    object.__setattr__(prepared, "config", config)
    object.__setattr__(prepared, "_payload", payload)
    return prepared
