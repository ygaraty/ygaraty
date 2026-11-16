"""Bounded JSON checkpoints: no pickle or executable checkpoint payloads."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

LIMIT = 16 * 1024 * 1024


def pack(value):
    if isinstance(value, np.ndarray):
        return {
            "type": "array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.ravel().tolist(),
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [pack(v) for v in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [pack(v) for v in value]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [[pack(k), pack(v)] for k, v in value.items()],
        }
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f"Unsupported checkpoint value: {type(value).__name__}")


def unpack(value):
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "array":
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Unsupported checkpoint array dtype")
        return np.asarray(value["data"], dtype=dtype).reshape(value["shape"])
    items = value["items"]
    if kind == "tuple":
        return tuple(unpack(v) for v in items)
    if kind == "list":
        return [unpack(v) for v in items]
    if kind == "dict":
        return {unpack(k): unpack(v) for k, v in items}
    raise ValueError("Unknown checkpoint encoding")


def bounded(value):
    raw = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > LIMIT - 4096:
        raise ValueError("Replica checkpoint exceeds 16 MiB transport bound")
    return raw


def save(path, payload):
    """Publish one complete generation only after its contents are durable."""
    path = Path(path)
    raw = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    envelope = dict(
        format="ygaraty.training-checkpoint/v1",
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=raw.decode(),
    )
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        json.dump(envelope, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(path):
    envelope = json.loads(Path(path).read_text())
    raw = envelope["payload"].encode()
    if (
        envelope["format"] != "ygaraty.training-checkpoint/v1"
        or hashlib.sha256(raw).hexdigest() != envelope["sha256"]
    ):
        raise ValueError("Invalid checkpoint format or checksum")
    return json.loads(raw)


def common_state(checkpoint, identity, spec, dataset):
    """Validate a common snapshot without constructing a backend."""
    import re

    from ygaraty.training.coordinator import copy_tensors

    if checkpoint["identity"] != identity:
        raise ValueError("Checkpoint inputs or configuration changed")
    state = unpack(checkpoint["state"])
    members, version = state["members"], state["round_index"]
    if (
        not isinstance(members, list)
        or any(
            not isinstance(w, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", w)
            for w in members
        )
        or len(set(members)) != spec["replicas"]
        or len(members) != spec["replicas"]
        or set(members) != set(state["replicas"])
        or type(version) is not int
        or not 1 <= version <= spec["rounds"]
        or state["coordinator"]["version"] != version
        or len(state["history"]) != version
        or any(state["replicas"][w] is None for w in members)
    ):
        raise ValueError("Incomplete common checkpoint")
    epoch, cursor = state["dataset"]
    if (
        type(epoch) is not int
        or epoch < 0
        or type(cursor) is not int
        or not 0 <= cursor <= dataset.rows
        or state["cursor"]
        != version
        * spec["replicas"]
        * spec["local_steps"]
        * spec["batch_size"]
        or epoch * dataset.rows + cursor != state["cursor"]
    ):
        raise ValueError("Invalid dataset checkpoint cursor")
    weights = copy_tensors(state["coordinator"]["weights"])
    copy_tensors(state["coordinator"]["velocity"], weights)
    for worker in members:
        bounded(state["replicas"][worker])
    return state
