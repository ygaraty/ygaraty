"""Bounded, data-only forward tapes. Rebuild through public shape rules."""

import math
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from .tape import Tape

MAX_NODES = 128
MAX_ELEMENTS = 65536
OPERATIONS = {
    "matmul": (2, set()),
    "add": (2, set()),
    "mul": (2, set()),
    "add_bias": (2, set()),
    "relu": (1, set()),
    "sigmoid": (1, set()),
    "tanh": (1, set()),
    "transpose": (1, set()),
    "sum": (1, set()),
    "scale": (1, {"factor"}),
    "slice_columns": (1, {"start", "width"}),
    "cross_entropy": (1, {"targets", "ignore_index", "label_smoothing"}),
}


def describe(tape: Tape, loss: str) -> dict[str, Any]:
    """Snapshot a forward tape; differentiation is the compiler's
    responsibility.
    """
    if not isinstance(tape, Tape) or any(
        not hasattr(v, "tolist") for v in tape.values.values()
    ):
        raise ValueError(
            "Model expects a numeric Tape with explicit input values"
        )
    if tape.loss is not None or tape.owned_layers or tape.owned_vocab:
        raise ValueError(
            "Model expects a forward tape, before backward(), "
            "without ownership"
        )
    spec = dict(
        inputs={name: value.tolist() for name, value in tape.values.items()},
        parameters=list(tape.parameters),
        nodes=[asdict(n) for n in tape.nodes],
        loss=loss,
    )
    rebuild(spec)
    return deepcopy(spec)


class TapeSpecBuilder:
    """Rebuild bounded forward graphs through validated public Tape
    operations.

    Each build starts a fresh tape; no numerical kernels are evaluated.
    """

    def build(self, spec: dict[str, Any]) -> Tape:
        """Validate dependencies and dimensions without evaluating a
        numerical kernel.
        """
        if not isinstance(spec, dict) or set(spec) != {
            "inputs",
            "parameters",
            "nodes",
            "loss",
        }:
            raise ValueError("Invalid tape specification")
        nodes = spec["nodes"]
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= MAX_NODES:
            raise ValueError("A tape needs 1..128 forward operations")
        tape = self._tape = Tape()
        self._bind_inputs(spec)
        for node in nodes:
            self._append_node(node)
        loss = spec["loss"]
        if not isinstance(loss, str) or tape.shapes.get(loss) != (1, 1):
            raise ValueError("The model loss must name a scalar tensor")
        return tape

    def build_forward(self, spec: dict[str, Any]) -> Tape:
        """Validate named forward outputs and optional parameters."""
        if not isinstance(spec, dict) or set(spec) != {
            "inputs",
            "parameters",
            "nodes",
            "outputs",
        }:
            raise ValueError("Invalid forward specification")
        if (
            not isinstance(spec["nodes"], list)
            or len(spec["nodes"]) > MAX_NODES
        ):
            raise ValueError("A forward tape supports at most 128 operations")
        tape = self._tape = Tape()
        self._bind_inputs(spec, require_parameters=False)
        for node in spec["nodes"]:
            self._append_node(node)
        outputs = spec["outputs"]
        if (
            not isinstance(outputs, dict)
            or not outputs
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or value not in tape.shapes
                for name, value in outputs.items()
            )
        ):
            raise ValueError(
                "Outputs must map public names to forward tensors"
            )
        return tape

    def _bind_inputs(
        self, spec: dict[str, Any], *, require_parameters: bool = True
    ) -> None:
        tape = self._tape
        values, parameters = spec["inputs"], spec["parameters"]
        if not isinstance(values, dict) or not 1 <= len(values) <= MAX_NODES:
            raise ValueError("A tape needs 1..128 input tensors")
        if (
            not isinstance(parameters, list)
            or (require_parameters and not parameters)
            or any(not isinstance(n, str) for n in parameters)
            or len(set(parameters)) != len(parameters)
            or not set(parameters) <= values.keys()
        ):
            raise ValueError("Parameters must be unique named input tensors")
        elements = 0
        for name, value in values.items():
            if not isinstance(name, str):
                raise ValueError("Input names must be strings")
            if (
                not isinstance(value, list)
                or not 1 <= len(value) <= 128
                or any(
                    not isinstance(row, list) or not 1 <= len(row) <= 128
                    for row in value
                )
            ):
                raise ValueError("Input dimensions must be 1..128")
            elements += sum(map(len, value))
            if elements > MAX_ELEMENTS:
                raise ValueError("Tape inputs exceed 65,536 elements")
            if name in parameters:
                tape.parameter(name, value)
            else:
                tape.input(name, value)
        tape.parameters = list(parameters)

    def _append_node(self, node: dict[str, Any]) -> None:
        tape = self._tape
        if not isinstance(node, dict) or set(node) != {
            "id",
            "op",
            "inputs",
            "shape",
            "attrs",
        }:
            raise ValueError("Invalid forward operation")
        op, inputs, attrs = node["op"], node["inputs"], node["attrs"]
        if (
            not isinstance(node["id"], str)
            or not isinstance(node["shape"], (list, tuple))
            or len(node["shape"]) != 2
            or any(type(n) is not int or n < 1 for n in node["shape"])
        ):
            raise ValueError(
                "Operations need a string ID and "
                "two positive integer dimensions"
            )
        if not isinstance(op, str) or op not in OPERATIONS:
            raise ValueError(f"Unsupported public tape operation: {op!r}")
        count, fields = OPERATIONS[op]
        if (
            not isinstance(inputs, list)
            or len(inputs) != count
            or any(
                not isinstance(n, str) or n not in tape.shapes for n in inputs
            )
            or not isinstance(attrs, dict)
            or set(attrs) != fields
        ):
            raise ValueError("Invalid operation inputs or attributes")
        if op == "scale" and (
            type(attrs["factor"]) not in (int, float)
            or not math.isfinite(attrs["factor"])
        ):
            raise ValueError("Scale factor must be finite")
        if op == "slice_columns" and any(
            type(attrs[k]) is not int for k in fields
        ):
            raise ValueError("Slice bounds must be integers")
        if op == "cross_entropy":
            if (
                attrs["ignore_index"] is not None
                and type(attrs["ignore_index"]) is not int
                or type(attrs["label_smoothing"]) not in (int, float)
            ):
                raise ValueError("Invalid cross-entropy attributes")
        ident = getattr(tape, op)(*inputs, **attrs, name=node["id"])
        if list(tape.shapes[ident]) != list(node["shape"]):
            raise ValueError("Declared shape differs from the operation shape")
        if math.prod(tape.shapes[ident]) > MAX_ELEMENTS:
            raise ValueError("Operation exceeds the public tape element limit")


def rebuild(spec: dict[str, Any]) -> Tape:
    """Compatibility entry point for bounded, data-only forward tapes."""
    return TapeSpecBuilder().build(spec)
