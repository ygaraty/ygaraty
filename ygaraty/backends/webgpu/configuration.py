"""Strict version-one WebGPU configuration, carried by the public contract."""

import json
import re
from dataclasses import asdict, dataclass

from ygaraty.backend_contracts.models import BackendConfiguration, Route

BACKEND_ID = "webgpu"
CONFIG_VERSION = 1
# Seconds without new text before a request fails, in the session and in the
# engine's turn. Reading a prompt sends none: an 11K-token prompt took 89 s on
# a Radeon 680M (S5-Nano) and over 120 s on an RTX 3090 (Qwen3.8-27B, DO-108).
# Lost workers are caught sooner by their heartbeats.
IDLE_TIMEOUT_S = 600


def integer(value: int, name: str, maximum: int = 2**31 - 1) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"Invalid {name}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def json_object(payload: bytes) -> dict:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid JSON document") from error
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


@dataclass(frozen=True, slots=True)
class WebGPUConfig:
    bundle_sha256: str
    stage_layers: tuple[tuple[int, int], ...]
    context_tokens: int
    memory_budget_bytes: int
    conversation_slots: int = 8
    route: Route = Route.RELAY

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.bundle_sha256
        ):
            raise ValueError("Expected SHA-256 bundle digest")
        integer(self.context_tokens, "context_tokens")
        integer(self.memory_budget_bytes, "memory_budget_bytes", 2**53 - 1)
        integer(self.conversation_slots, "conversation_slots", 16)
        if not isinstance(self.route, Route):
            raise ValueError("Expected Route")
        if (
            not isinstance(self.stage_layers, tuple)
            or not 1 <= len(self.stage_layers) <= 26
        ):
            raise ValueError("Expected immutable stage assignments")
        previous = 0
        for bounds in self.stage_layers:
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(type(n) is not int for n in bounds)
                or bounds[0] != previous
                or bounds[1] <= bounds[0]
            ):
                raise ValueError(
                    "Stages must cover contiguous layers from zero"
                )
            previous = bounds[1]

    def to_contract(self) -> BackendConfiguration:
        return BackendConfiguration(
            BACKEND_ID,
            CONFIG_VERSION,
            json.dumps(asdict(self), sort_keys=True, allow_nan=False).encode(),
        )

    @classmethod
    def from_contract(cls, value: BackendConfiguration) -> "WebGPUConfig":
        if not isinstance(value, BackendConfiguration) or (
            value.backend_id != BACKEND_ID
            or value.schema_version != CONFIG_VERSION
        ):
            raise ValueError("Unsupported WebGPU configuration")
        data = json_object(value.payload)
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Missing or unknown WebGPU configuration fields")
        layers = data["stage_layers"]
        if not isinstance(layers, list) or any(
            not isinstance(bounds, list) for bounds in layers
        ):
            raise ValueError("Expected stage assignment arrays")
        data["stage_layers"] = tuple(tuple(bounds) for bounds in layers)
        data["route"] = Route(data["route"])
        return cls(**data)
