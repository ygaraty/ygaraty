"""The bytes one task may make a browser worker allocate: experiment 48's
memory budget.

A worker advertises a budget; a job that enforces budgets offers it only tasks
whose working set
fits (coordinator side), and the worker recomputes the same number from the
task it received and
refuses one that does not fit before allocating anything (worker side).
ygaraty/backends/webgpu/browser/budget.mjs is the JavaScript counterpart.

The number is an upper bound on what the browser code allocates for the task,
GPU and JavaScript
together, given that its inputs are float64 once received (a float32 wire is
widened on arrival):
  every op: the inputs as received (8 bytes per value) and the float64 result;
  elementwise and reduction ops (add, add_many, scale, sgd, sum_squares): a second result-sized
    float64 array of scratch;
  factorization_tile on the GPU (four inputs and an x_spec): float32 copies of the inputs in
    JavaScript and on the GPU, the float32 residual (rows x columns), the float32 output on the
    GPU, its mapped readback and its JavaScript copy, and 64 bytes of parameters;
  factorization_tiles (a group of p x q tiles, 2 (p + q) inputs): the group's inputs and result,
    plus one tile at a time: the largest tile's GPU share above and its float64 result before it
    is added to the group's.
An op not in this table has no bound: a worker enforcing budgets refuses it.
"""  # noqa: E501 -- preserve documentation layout

import math

ELEMENTWISE = ("add", "add_many", "scale", "sgd", "sum_squares")
OPS = ELEMENTWISE + ("factorization_tile", "factorization_tiles")
PARAMS_BYTES = 64


def _shape(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(type(n) is int and 0 < n <= 1 << 31 for n in value)
    ):
        raise ValueError("shapes must be two positive integers")
    return value


def task_bytes(op, input_shapes, shape, attrs=None):
    """Working-set bytes of one task from its op, input shapes ([rows, cols]
    each), output shape
    and attributes. ValueError for an op without a bound or a malformed
    description."""
    if op not in OPS:
        raise ValueError(f"no memory bound for {op}")
    inputs = sum(math.prod(_shape(s)) for s in input_shapes)
    out = math.prod(_shape(shape))
    total = 8 * inputs + 8 * out
    if op in ELEMENTWISE:
        return total + 8 * out
    rank = _shape(shape)[1]
    if op == "factorization_tiles":
        return total + max(
            _tile_extra(rows, cols, latent, rank)
            + 8 * (rows + cols + 1) * rank
            for rows, cols, latent in _group_tiles(input_shapes, attrs)
        )
    spec = (attrs or {}).get("x_spec")
    if len(input_shapes) != 4 or not isinstance(spec, dict):
        raise ValueError(
            "a factorization tile is bounded only on the GPU path: "
            "four inputs and an x_spec"
        )
    rows, cols = _shape(spec.get("shape"))
    latent = _shape(input_shapes[0])[1]
    return total + _tile_extra(rows, cols, latent, rank)


def _tile_extra(rows, cols, latent, rank):
    """The GPU tile's own allocations beyond its float64 inputs and result."""
    gpu_out = 4 * ((rows + cols) * rank + rows)
    inputs = (rows + cols) * (latent + rank)
    return 2 * 4 * inputs + 4 * rows * cols + 3 * gpu_out + PARAMS_BYTES


def _group_tiles(input_shapes, attrs):
    """(rows, cols, latent) of every tile of a factorization_tiles task, from
    its shards."""
    attrs = attrs or {}
    row_shards, column_shards = (
        attrs.get("row_shards"),
        attrs.get("column_shards"),
    )

    def bounds(value):
        return (
            isinstance(value, (list, tuple))
            and 1 <= len(value) <= 64
            and all(
                isinstance(b, (list, tuple))
                and len(b) == 2
                and all(type(n) is int for n in b)
                and 0 <= b[0] < b[1]
                for b in value
            )
        )

    if (
        not bounds(row_shards)
        or not bounds(column_shards)
        or not isinstance(attrs.get("x_spec"), dict)
    ):
        raise ValueError(
            "a tile group needs row_shards, column_shards and an x_spec"
        )
    p, q = len(row_shards), len(column_shards)
    if len(input_shapes) != 2 * (p + q):
        raise ValueError(
            "a tile group is bounded only on the GPU path: latent "
            "rows, U and V for each shard"
        )
    latent = _shape(input_shapes[0])[1]
    for k, b in enumerate(row_shards + column_shards):
        if (
            _shape(input_shapes[k])[0] != b[1] - b[0]
            or _shape(input_shapes[p + q + k])[0] != b[1] - b[0]
        ):
            raise ValueError("a tile group's inputs must match its shards")
    return [
        (r[1] - r[0], c[1] - c[0], latent)
        for r in row_shards
        for c in column_shards
    ]
