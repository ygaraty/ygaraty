"""Operation work estimates for fitted timing models: seconds = a + b *
work.

These are operation-specific work units, not measured durations or FLOP
counts. The caller separately accounts for any constituent kernels in
``extra_work``.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

Shape = Sequence[int]
_FACTORIZATION = frozenset(
    {
        "factorization_block",
        "factorization_tile",
        "factorization_tiles",
    }
)
_TRANSFORMER_FORWARD = frozenset({"tf_layer", "tf_owned_layer"})
_TRANSFORMER_BACKWARD = frozenset({"tf_layer_grad", "tf_owned_grad"})
_OWNER_TRANSFER = frozenset(
    {
        "tf_owned_step",
        "tf_vocab_step",
        "tf_owned_load",
        "tf_vocab_load",
    }
)


def _transformer_work(rows: int, width: int, attrs: Mapping[str, Any]):
    """Projection/feed-forward work plus attention over sequence positions."""
    projections = 8 if attrs.get("decoder") else 4
    return rows * width * width * projections + rows * width * attrs.get(
        "steps", 1
    )


def task_work(
    op: str,
    attrs: Mapping[str, Any],
    shape: Shape,
    inputs: Sequence[Shape],
) -> int | float:
    """Estimate work for an output shape and its input-region shapes.

    Shapes and attributes are supplied by validated compiler plans. Unknown
    operations retain the generic input-plus-output element estimate.
    """
    output_elements = math.prod(shape)
    input_elements = sum(math.prod(dims) for dims in inputs)
    rows, width = shape[0], shape[1]
    if op == "matmul":
        (rows, inner), (_, columns) = inputs[0], inputs[1]
        return rows * inner * columns
    if op in _FACTORIZATION:
        # extra_work already describes the constituent kernels. Retain only
        # the fitted dispatch intercept here; do not double-count their work.
        return 0
    if op in _TRANSFORMER_FORWARD:
        return _transformer_work(rows, width, attrs)
    if op in _TRANSFORMER_BACKWARD:
        return 3 * _transformer_work(inputs[0][0], width, attrs)
    if op == "ce_fused":
        return inputs[0][0] * inputs[0][1] * inputs[1][1]
    if op in ("tf_vocab_stats", "tf_vocab_grad"):
        start, stop = attrs["rows"]
        passes = 1 if op == "tf_vocab_stats" else 3
        return inputs[0][0] * inputs[0][1] * (stop - start) * passes
    if op in _OWNER_TRANSFER:
        return input_elements + attrs.get("elements", 0)
    return input_elements + output_elements
