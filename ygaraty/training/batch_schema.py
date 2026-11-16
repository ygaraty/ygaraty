"""Pure numeric batch-schema validation shared by data and transport code."""

import math

import numpy as np

DTYPES = {"float64", "float32", "int64", "int32", "uint8"}


def schema_fields(schema):
    if (
        not isinstance(schema, dict)
        or type(schema.get("version")) is not int
        or schema["version"] != 1
    ):
        raise ValueError("Expected batch schema version 1")
    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Expected nonempty batch fields")
    names = []
    for field in fields:
        if (
            not isinstance(field, dict)
            or not isinstance(field.get("name"), str)
            or not field["name"]
            or field.get("dtype") not in DTYPES
        ):
            raise ValueError("Invalid batch field")
        names.append(field["name"])
    if len(set(names)) != len(names):
        raise ValueError("Duplicate batch field")
    return fields


def decode_field(block, shape, dtype):
    """Version 1 stores every numeric field in float64 wire slots."""
    if (
        not isinstance(shape, (list, tuple))
        or any(type(n) is not int or n < 1 for n in shape)
        or math.prod(shape) != block.size
        or not np.isfinite(block).all()
    ):
        raise ValueError("Invalid numeric batch field")
    if dtype.startswith("int") or dtype == "uint8":
        info = np.iinfo(dtype)
        low, high = max(info.min, -(2**53 - 1)), min(info.max, 2**53 - 1)
        if (
            np.any(block != np.floor(block))
            or np.any(block < low)
            or np.any(block > high)
        ):
            raise ValueError("Integer batch field exceeds exact wire range")
    with np.errstate(over="ignore", invalid="ignore"):
        out = block.astype(dtype).reshape(shape)
    if not np.isfinite(out).all():
        raise ValueError("Batch field overflows dtype")
    return out
