"""Bind model parameter names to one tape; no execution or initialization
I/O.
"""

from collections.abc import Mapping
from types import MappingProxyType

from .shapes import ShapeTape


class Parameters(Mapping):
    """Read-only parameter bindings with tape-local, lazy transpose reuse.

    Numeric tapes receive arrays; ShapeTape receives initializer
    specifications. Model names retain their dots here and use
    underscores only in tensor IDs. Bindings and cached graph nodes
    belong to this tape, never to a global model.
    """

    def __init__(self, tape, state):
        self._tape = tape
        self._ids = MappingProxyType(
            {name: self._bind(name, value) for name, value in state.items()}
        )
        self._transposes = {}

    def _bind(self, name, value):
        tensor_id = name.replace(".", "_")
        if isinstance(self._tape, ShapeTape):
            initializer = {
                k: v
                for k, v in value.items()
                if k not in ("shape", "original_shape")
            }
            return self._tape.declare(
                tensor_id, value["shape"], initializer, True
            )
        array = value.reshape(1, -1) if value.ndim == 1 else value
        return self._tape.parameter(tensor_id, array)

    def __getitem__(self, name):
        return self._ids[name]

    def __iter__(self):
        return iter(self._ids)

    def __len__(self):
        return len(self._ids)

    def transpose(self, name):
        """Create a transpose at first use, preserving graph construction
        order.
        """
        if name not in self._transposes:
            self._transposes[name] = self._tape.transpose(self[name])
        return self._transposes[name]

    def linear(self, value, prefix):
        """Apply a named weight (output × input) and its row bias."""
        return self._tape.add_bias(
            self._tape.matmul(value, self.transpose(prefix + ".weight")),
            self[prefix + ".bias"],
        )

    def layer_norm(self, value, prefix):
        return self._tape.layer_norm(
            value, self[prefix + ".weight"], self[prefix + ".bias"]
        )
