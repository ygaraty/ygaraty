"""Bounded local artifact I/O shared by preparation and validation."""

import hashlib
from pathlib import Path

from .configuration import json_object

BLOCK_BYTES = 1 << 20


def local_input(value, *, limit=None):
    if "://" in str(value):
        raise ValueError("Use a local file, not a URL")
    try:
        path = Path(value).resolve(strict=True)
    except FileNotFoundError as error:
        requested = Path(value).absolute()
        raise ValueError(
            f"Local file not found: {requested}. "
            "Provide your existing model files; /path/to/... in examples "
            "is a placeholder, not a bundled checkpoint."
        ) from error
    if not path.is_file():
        raise ValueError(f"Expected a local file: {path.name}")
    if limit is not None and path.stat().st_size > limit:
        raise ValueError(f"{path.name} exceeds the {limit}-byte limit")
    return path


def read_json(path, *, limit=16 << 20):
    return json_object(local_input(path, limit=limit).read_bytes())


def range_digest(path, offset, length, *, output=None):
    """Hash or copy exactly one range using a bounded buffer."""
    path = local_input(path)
    if (
        type(offset) is not int
        or type(length) is not int
        or offset < 0
        or length <= 0
        or offset + length > path.stat().st_size
    ):
        raise ValueError("Invalid artifact byte range")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining:
            block = source.read(min(remaining, BLOCK_BYTES))
            if not block:
                raise ValueError("Truncated artifact byte range")
            digest.update(block)
            if output is not None:
                output.write(block)
            remaining -= len(block)
    return digest.hexdigest()


def file_digest(path):
    path = local_input(path)
    return range_digest(path, 0, path.stat().st_size)
