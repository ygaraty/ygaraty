"""Skip unused GGUF tokenizer metadata without allocating an object per token.

Only bundle preparation opts in. Tensor parsing/dequantization remains gguf's;
all non-tokenizer metadata (including alignment) retains the normal reader path.
"""

import struct

SCALAR_BYTES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def skip_value(data, offset, kind, endian="<", depth=0):
    """Return the next byte offset of a GGUF value, checking bounds at every step."""
    if depth > 32:
        raise ValueError("GGUF metadata nesting exceeds 32 levels")

    def advance(position, size):
        if position < 0 or size < 0 or position + size > len(data):
            raise ValueError("Truncated GGUF metadata")
        return position + size

    if kind in SCALAR_BYTES:
        return advance(offset, SCALAR_BYTES[kind])
    if kind == 8:  # STRING: uint64 length, then UTF-8 bytes
        start = advance(offset, 8)
        length = struct.unpack_from(endian + "Q", data, offset)[0]
        return advance(start, length)
    if (
        kind == 9
    ):  # ARRAY: element type, count, then values (without repeated type tags)
        start = advance(offset, 12)
        element, count = struct.unpack_from(endian + "IQ", data, offset)
        if element in SCALAR_BYTES:
            return advance(start, count * SCALAR_BYTES[element])
        if element not in (8, 9):
            raise ValueError("Unknown GGUF array element type")
        minimum = 8 if element == 8 else 12
        advance(start, count * minimum)
        for _ in range(count):
            start = skip_value(data, start, element, endian, depth + 1)
        return start
    raise ValueError("Unknown GGUF metadata type")


def tensor_reader(path):
    import gguf

    class TensorReader(gguf.GGUFReader):
        def _build_fields(self, offset, count):
            endian = "<" if self.endianess == gguf.GGUFEndian.LITTLE else ">"
            for _ in range(count):
                end = skip_value(self.data, offset, 8, endian)
                name = bytes(self.data[offset + 8 : end]).decode("utf-8")
                type_end = skip_value(self.data, end, 4, endian)
                kind = struct.unpack_from(endian + "I", self.data, end)[0]
                if name.startswith("tokenizer."):
                    offset = skip_value(self.data, type_end, kind, endian)
                else:
                    offset = super()._build_fields(offset, 1)
            return offset

    return TensorReader(path)
