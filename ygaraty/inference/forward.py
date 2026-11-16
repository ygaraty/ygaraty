"""Immutable admission boundary for public, forward-only Tape computations."""

import json
from dataclasses import asdict, dataclass
from typing import Any

from ..compiler.plan import Input, Task
from ..compiler.spec import TapeSpecBuilder
from ..compiler.tape import Tape


def select_tasks(tasks, outputs):
    """Select a forward dependency closure, preserving execution order."""
    needed = set(outputs)
    for task in reversed(tasks):
        if task.id in needed:
            needed.update(task.dependencies)
    return [task for task in tasks if task.id in needed], needed


@dataclass(frozen=True, slots=True, init=False)
class ForwardPlan:
    """Validated JSON snapshot; exported dictionaries are independent copies.

    The local target uses F64 rank-two tensors and the bounded public Tape
    operation set. It does not imply a generic WebGPU lowering.
    """

    _payload: str

    def __new__(cls):
        raise TypeError("Use Tape.forward() or ForwardPlan.from_dict()")

    @classmethod
    def from_tape(
        cls, tape: Tape, *, outputs: dict[str, str]
    ) -> "ForwardPlan":
        if not isinstance(tape, Tape) or tape.loss is not None:
            raise ValueError("Expected a forward Tape before backward()")
        if tape.owned_layers or tape.owned_vocab:
            raise ValueError("Owned training operations are not forward plans")
        if any(not hasattr(value, "tolist") for value in tape.values.values()):
            raise ValueError("Forward inputs need explicit numeric values")
        return cls.from_dict(
            {
                "version": 1,
                "target": "local",
                "graph": {
                    "inputs": {k: v.tolist() for k, v in tape.values.items()},
                    "parameters": list(tape.parameters),
                    "nodes": [asdict(node) for node in tape.nodes],
                    "outputs": outputs,
                },
            }
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ForwardPlan":
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "target", "graph"}
            or type(value["version"]) is not int
            or value["version"] != 1
            or value["target"] != "local"
        ):
            raise ValueError("Unsupported forward plan")
        # Serialization severs caller aliases, rejects nonfinite JSON values,
        # and normalizes shape tuples before validation.
        payload = json.dumps(value, allow_nan=False, sort_keys=True)
        snapshot = json.loads(payload)
        tape = TapeSpecBuilder().build_forward(snapshot["graph"])
        if sum(r * c for r, c in tape.shapes.values()) * 8 > 64 << 20:
            raise ValueError("Forward plan exceeds the 64 MiB tensor budget")
        plan = object.__new__(cls)
        object.__setattr__(plan, "_payload", payload)
        return plan

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, serializable copy of the admitted graph."""
        return json.loads(self._payload)

    def run(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate outputs using the captured parameters."""
        import numpy as np

        from ..runtime.local import execute_plan

        spec = self.to_dict()["graph"]
        tasks = [
            Task(
                n["id"],
                n["op"],
                [Input(i) for i in n["inputs"]],
                tuple(n["shape"]),
                n["attrs"],
            )
            for n in spec["nodes"]
        ]
        tasks, needed = select_tasks(tasks, spec["outputs"].values())
        parameters = set(spec["parameters"])
        required = (needed & spec["inputs"].keys()) - parameters
        if not isinstance(inputs, dict) or set(inputs) != required:
            raise ValueError(f"Expected forward inputs: {sorted(required)}")
        values = {}
        for name in needed & spec["inputs"].keys():
            default = np.asarray(spec["inputs"][name], dtype=np.float64)
            value = np.asarray(
                default if name in parameters else inputs[name],
                dtype=np.float64,
            )
            if value.shape != default.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid forward input: {name}")
            values[name] = value.copy()
        result = execute_plan(tasks, values, set(spec["outputs"].values()))
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Forward execution produced nonfinite values")
        return {
            name: result[tensor].copy()
            for name, tensor in spec["outputs"].items()
        }
