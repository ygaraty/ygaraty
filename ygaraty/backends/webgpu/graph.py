"""Shape-only Tape graph used by WebGPU lowering; no numerical values."""

import re
from dataclasses import asdict, dataclass, field


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    shape: tuple[int, int]
    attrs: dict = field(default_factory=dict)


class ShapeTape:
    """Structural subset of compiler.Tape; no training or execution methods."""

    def __init__(self):
        self.shapes = {}
        self.nodes = []
        self._counter = 0

    def _name(self, name=None):
        if name is None:
            while f"v{self._counter}" in self.shapes:
                self._counter += 1
            name = f"v{self._counter}"
            self._counter += 1
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name in self.shapes
        ):
            raise ValueError(f"Invalid or duplicate tensor ID: {name!r}")
        return name

    def _op(self, op, inputs, shape, attrs=None, name=None):
        name = self._name(name)
        self.shapes[name] = shape
        self.nodes.append(Node(name, op, list(inputs), shape, attrs or {}))
        return name


@dataclass
class Input:
    value: str
    region: list[int] | None = None


@dataclass
class Task:
    id: str
    op: str
    inputs: list[Input]
    shape: tuple[int, int]
    attrs: dict = field(default_factory=dict)

    def describe(self):
        return asdict(self)
