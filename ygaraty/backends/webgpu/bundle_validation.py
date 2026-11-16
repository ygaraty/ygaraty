"""Verify local bundle artifacts and freshly compile their inference plans."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from ygaraty.backend_contracts.models import ModelIdentity, SessionSpec

from ._legacy.model_artifacts import local_file
from .bundle_io import file_digest, local_input, range_digest, read_json
from .configuration import WebGPUConfig
from .preparation import prepare_session


@dataclass(frozen=True, slots=True)
class BundleReport:
    path: Path
    sha256: str
    stages: int
    context_tokens: int
    weight_bytes: int
    artifacts_pinned: bool


def remote_base(data):
    """Return the public HTTPS weight base URL, or None for local weights.

    A ``url`` backend means contributors fetch each hash-pinned piece from
    ``base_url + path`` themselves; the server holds no weight bytes.
    """
    storage = data.get("storage", {})
    backend = storage.get("backend", "local")
    if backend == "local":
        return None
    if backend != "url":
        raise ValueError("Unsupported bundle storage backend")
    base = storage.get("base_url")
    parsed = urlsplit(base) if isinstance(base, str) else None
    if (
        parsed is None
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not base.endswith("/")
    ):
        raise ValueError("Remote weights need a public HTTPS base URL")
    return base


def remote_piece_path(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Unsafe remote weight path")
    return value


def validate_bundle(path) -> BundleReport:
    """Check files and hashes without constructing an engine or executing it.

    Legacy bundles without artifact_digests remain readable. Weight ranges are
    verified, but tokenizer/self-test content is explicitly unpinned. Remote
    (``url`` storage) weights are checked for shape only; contributors verify
    each piece's SHA-256 when they download it.
    """
    path = local_input(path, limit=16 << 20)
    data = read_json(path)
    remote = remote_base(data) is not None
    if not data.get("files"):
        raise ValueError("Bundle has no weight ranges")
    total = 0
    for name, value in data["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("Invalid weight name")
        if remote:
            remote_piece_path(value["path"])
            if (
                value["offset"] != 0
                or type(value["length"]) is not int
                or value["length"] <= 0
                or not re.fullmatch(r"[a-f0-9]{64}", str(value["sha256"]))
            ):
                raise ValueError(
                    "Remote weights must be complete, hash-pinned pieces"
                )
        else:
            source = local_file(path.parent, value["path"])
            actual = range_digest(source, value["offset"], value["length"])
            if actual != value["sha256"]:
                raise ValueError(f"Weight hash mismatch: {name}")
        total += value["length"]
    tokenizer = local_file(path.parent, data["tokenizer"])
    read_json(tokenizer, limit=64 << 20)
    referenced = {data["tokenizer"]}
    for relative in data["test_files"].values():
        local_input(local_file(path.parent, relative), limit=16 << 20)
        referenced.add(relative)
    cases = data["selftest"]["cases"]
    if not cases or len(cases) != len(data["expected"]):
        raise ValueError("Incomplete self-test references")
    for case, relative in zip(cases, data["expected"]):
        import numpy as np

        source = local_file(path.parent, relative)
        local_input(source, limit=16 << 20)
        expected = np.load(source, allow_pickle=False)
        if not isinstance(expected, np.ndarray):
            expected.close()
            raise ValueError("Expected a NumPy array, not an archive")
        if expected.dtype.kind != "f" or expected.shape != (
            case["tokens"],
            case["shape"][0],
        ):
            raise ValueError("Invalid self-test reference shape or dtype")
        referenced.add(relative)
        for key in ("file", "x"):
            if case[key] not in data["test_files"]:
                raise ValueError("Missing self-test input")
    if data["selftest"]["f16"] not in data["test_files"]:
        raise ValueError("Missing float16 self-test input")
    pins = data.get("artifact_digests")
    if pins is not None:
        if not isinstance(pins, dict) or pins.keys() != referenced:
            raise ValueError("Incomplete artifact digest inventory")
        for relative, digest in pins.items():
            if file_digest(local_file(path.parent, relative)) != digest:
                raise ValueError(f"Artifact hash mismatch: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = WebGPUConfig(
        digest,
        tuple(tuple(stage["layers"]) for stage in data["stages"]),
        data["options"]["capacity"],
        max(total, 1),
    )
    identity = ModelIdentity("validation", digest, "local", "local", "bundle")
    spec = SessionSpec(
        "validation", "validation", identity, config.to_contract()
    )
    prepare_session(spec, bundle=path.read_bytes())
    return BundleReport(
        path,
        digest,
        len(data["stages"]),
        config.context_tokens,
        total,
        pins is not None,
    )
