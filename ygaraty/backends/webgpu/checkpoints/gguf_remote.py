"""Local GGUF header and tensor metadata.

The historical module name is retained for loader compatibility.
This extraction has no HTTP or repository-download helpers.
"""

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# GGUF value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
# GGML tensor types: (name, elements per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


@dataclass
class TensorInfo:
    name: str
    kind: str
    shape: tuple  # NumPy order: (rows, columns); rows are output features
    offset: (
        int  # from the start of the file (its shard's file in a split GGUF)
    )
    nbytes: int
    shard: int = 0  # which file of a split GGUF holds it

    @property
    def row_bytes(self):
        _, block, size = next(
            v for v in GGML_TYPES.values() if v[0] == self.kind
        )
        return self.shape[-1] // block * size


@dataclass
class Header:
    version: int
    metadata: dict
    tensors: dict = field(default_factory=dict)
    data_start: int = 0
    size: int | None = (
        None  # bytes of the file (of all shards for a split GGUF)
    )
    shards: list = field(
        default_factory=list
    )  # split GGUF: [{"size", "data_start", "tensors"}] per shard

    def architecture(self):
        return self.metadata.get("general.architecture")

    def get(self, key, default=None):
        """A metadata value under the architecture's prefix (`llama.block_count`) or as given."""
        arch = self.architecture()
        return self.metadata.get(
            f"{arch}.{key}", self.metadata.get(key, default)
        )


class _Source:
    """Read bounded local GGUF header bytes; never resolve a URL."""

    def __init__(self, where, first=8 << 20):
        if "://" in str(where):
            raise ValueError("Checkpoint preparation accepts local files only")
        self.where = Path(where).resolve(strict=True)
        self.size = self.where.stat().st_size
        self.data = b""
        self.ensure(first)

    def ensure(self, end):
        if end <= len(self.data):
            return
        if end > 64 << 20:
            raise ValueError("GGUF header exceeds 64 MiB")
        if len(self.data) >= self.size:
            raise ValueError("Truncated GGUF header")
        with self.where.open("rb") as handle:
            self.data = handle.read(
                min(64 << 20, max(end, 2 * len(self.data)))
            )


def read_header(where, first=8 << 20) -> Header:
    """The GGUF header (metadata and tensor table) of a local file."""
    src = _Source(where, first)
    pos = 0

    def take(fmt):
        nonlocal pos
        size = struct.calcsize("<" + fmt)
        src.ensure(pos + size)
        values = struct.unpack_from("<" + fmt, src.data, pos)
        pos += size
        return values if len(values) > 1 else values[0]

    def string():
        nonlocal pos
        n = take("Q")
        src.ensure(pos + n)
        s = src.data[pos : pos + n].decode("utf-8", "replace")
        pos += n
        return s

    def value(kind, summarize=False):
        nonlocal pos
        if kind in SCALARS:
            return take(SCALARS[kind])
        if kind == 8:
            return string()
        if kind == 9:
            element, count = take("I"), take("Q")
            if summarize:  # tokenizer arrays: skipped, only their length kept
                for _ in range(count):
                    value(element, True)
                return {"array_of": element, "length": count}
            return [value(element) for _ in range(count)]
        raise ValueError(f"Unknown GGUF value type {kind}")

    if take("4s") != b"GGUF":
        raise ValueError("Not a GGUF file")
    version = take("I")
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version {version}")
    n_tensors, n_kv = take("Q"), take("Q")
    metadata = {}
    for _ in range(n_kv):
        key = string()
        kind = take("I")
        metadata[key] = value(
            kind, summarize=key.startswith("tokenizer.ggml.") and kind == 9
        )
    tensors = {}
    for _ in range(n_tensors):
        name = string()
        dims = [take("Q") for _ in range(take("I"))]
        kind, offset = take("I"), take("Q")
        if kind not in GGML_TYPES:
            raise ValueError(f"Unknown GGML type {kind} for {name}")
        type_name, block, size = GGML_TYPES[kind]
        elements = 1
        for d in dims:
            elements *= d
        tensors[name] = TensorInfo(
            name,
            type_name,
            tuple(reversed(dims)),
            offset,
            elements // block * size,
        )
    alignment = int(metadata.get("general.alignment", 32))
    data_start = (pos + alignment - 1) // alignment * alignment
    for t in tensors.values():
        t.offset += data_start
    return Header(version, metadata, tensors, data_start, src.size)


SPLIT_NAME = re.compile(
    r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$"
)


def split_names(first):
    """All shard names of a split GGUF given its first (`X-00001-of-00004.gguf` -> 4 names), or [first] for a
    single file. Raises for a later shard: the first one holds the model's metadata."""
    m = SPLIT_NAME.match(first)
    if not m:
        return [first]
    count = int(m["count"])
    if int(m["no"]) != 1 or count < 1:
        raise ValueError(
            f"{first}: name the first shard (-00001-of-{m['count']}.gguf)"
        )
    return [
        f"{m['stem']}-{i:05d}-of-{count:05d}.gguf" for i in range(1, count + 1)
    ]


def read_split_headers(locations, first=8 << 20) -> Header:
    """One logical header for a split GGUF, from its shards in order (local paths; one location is an unsplit
    file). Metadata comes from the first shard (later shards carry only `split.*`). Checked: every shard's
    `split.no` and `split.count`, one `split.tensors.count` equal to the tensors found, no tensor in two shards,
    each tensor's bytes inside its shard, and no two tensors of a shard overlapping."""
    headers = [read_header(where, first) for where in locations]
    if len(headers) == 1 and "split.count" not in headers[0].metadata:
        return headers[0]
    count = len(headers)
    expected_tensors = headers[0].metadata.get("split.tensors.count")
    errors = []
    for i, h in enumerate(headers):
        if (
            h.metadata.get("split.no") != i
            or h.metadata.get("split.count") != count
        ):
            errors.append(
                f"shard {i + 1}: split.no {h.metadata.get('split.no')} / split.count "
                f"{h.metadata.get('split.count')}, expected {i} / {count}"
            )
        if h.metadata.get("split.tensors.count") != expected_tensors:
            errors.append(
                f"shard {i + 1}: split.tensors.count {h.metadata.get('split.tensors.count')} differs "
                f"from the first shard's {expected_tensors}"
            )
    if headers[0].architecture() is None:
        errors.append("the first shard has no general.architecture")
    tensors, shards = {}, []
    for i, h in enumerate(headers):
        spans = []
        for name, t in h.tensors.items():
            if name in tensors:
                errors.append(
                    f"tensor {name} is in shards {tensors[name].shard + 1} and {i + 1}"
                )
                continue
            if h.size is not None and t.offset + t.nbytes > h.size:
                errors.append(
                    f"tensor {name} ends at byte {t.offset + t.nbytes}, past shard {i + 1}'s {h.size}"
                )
            t.shard = i
            tensors[name] = t
            spans.append((t.offset, t.offset + t.nbytes, name))
        spans.sort()
        for (_, end, a), (start, _, b) in zip(spans, spans[1:]):
            if start < end:
                errors.append(f"tensors {a} and {b} overlap in shard {i + 1}")
        shards.append(
            dict(size=h.size, data_start=h.data_start, tensors=len(h.tensors))
        )
    if expected_tensors is None or len(tensors) != expected_tensors:
        errors.append(
            f"{len(tensors)} tensors across {count} shards, split.tensors.count says {expected_tensors}"
        )
    if errors:
        raise ValueError("Invalid split GGUF: " + "; ".join(errors))
    sizes = [s["size"] for s in shards]
    return Header(
        headers[0].version,
        dict(headers[0].metadata),
        tensors,
        headers[0].data_start,
        sum(sizes) if None not in sizes else None,
        shards,
    )
