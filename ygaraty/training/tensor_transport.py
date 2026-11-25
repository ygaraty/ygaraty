"""Bounded YGT1 frames: JSON header and little-endian f64 tensors.

No pickle, base64 or JSON tensor arrays. SHA-256 covers the tensor payload;
authentication belongs to the private-LAN HTTP transport.
"""

import hashlib
import json
import struct

import numpy as np

MAX_FRAME = 64 * 1024 * 1024
MAX_HEADER = 4 * 1024 * 1024
CONTENT_TYPE = "application/vnd.ygaraty.tensor"


def encode(metadata, tensors=None):
    chunks, entries = [], []
    for name, value in (tensors or {}).items():
        array = np.asarray(value, dtype="<f8", order="C")
        if not isinstance(name, str) or not name or array.ndim != 2:
            raise ValueError("Expected named rank-two tensors")
        if min(array.shape) < 1 or not np.isfinite(array).all():
            raise ValueError("Invalid tensor shape or values")
        chunks.append(array.tobytes())
        entries.append(dict(name=name, shape=list(array.shape), dtype="f64le"))
    payload = b"".join(chunks)
    header = json.dumps(
        dict(
            metadata=metadata,
            tensors=entries,
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(header) > MAX_HEADER or 8 + len(header) + len(payload) > MAX_FRAME:
        raise ValueError("Tensor frame too large")
    return b"YGT1" + struct.pack("<I", len(header)) + header + payload


def decode(frame):
    if not 8 <= len(frame) <= MAX_FRAME or frame[:4] != b"YGT1":
        raise ValueError("Invalid tensor frame")
    size = struct.unpack_from("<I", frame, 4)[0]
    if not 1 <= size <= MAX_HEADER or 8 + size > len(frame):
        raise ValueError("Invalid tensor header")
    header = json.loads(frame[8 : 8 + size])
    if not isinstance(header, dict):
        raise ValueError("Invalid tensor header")
    payload = memoryview(frame)[8 + size :]
    if hashlib.sha256(payload).hexdigest() != header.get("sha256"):
        raise ValueError("Tensor checksum mismatch")
    if not isinstance(header.get("metadata"), dict):
        raise ValueError("Invalid metadata")
    entries = header.get("tensors")
    if not isinstance(entries, list):
        raise ValueError("Invalid tensor index")
    tensors, offset = {}, 0
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Invalid tensor descriptor")
        name, shape = item.get("name"), item.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or name in tensors
            or item.get("dtype") != "f64le"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
        ):
            raise ValueError("Invalid tensor descriptor")
        count = shape[0] * shape[1]
        end = offset + count * 8
        if end > len(payload):
            raise ValueError("Truncated tensor")
        value = (
            np.frombuffer(payload[offset:end], dtype="<f8")
            .reshape(shape)
            .copy()
        )
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite tensor")
        tensors[name], offset = value, end
    if offset != len(payload):
        raise ValueError("Trailing tensor bytes")
    return header["metadata"], tensors
