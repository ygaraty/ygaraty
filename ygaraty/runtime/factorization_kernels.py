"""Numerical kernels for fully observed dense matrix factorization.

Two operations share one computation: ``factorization_block`` (a row shard of X
with the whole of V,
the fused plan of experiments/factorization-scaling) and ``factorization_tile``
(one tile X_ij with
only U_i and V_j, the two-dimensional plan of models/factorization_sharded.py).
They differ only in
what the attributes must satisfy.
"""

from collections.abc import Mapping

import numpy as np

OPS = {"factorization_block", "factorization_tile", "factorization_tiles"}


MAX_GENERATED_BYTES = 1 << 30


def _generated_x(spec):
    """One X tile from its seeded spec (models/factorization_sharded.py),
    bounded before it is allocated."""
    _generated_x_check(spec)
    from ygaraty.models.factorization_sharded import shard_values

    return shard_values(dict(spec))


def _generated_x_check(spec):
    shape = spec.get("shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or not all(type(n) is int and n > 0 for n in shape)
        or shape[0] * shape[1] * 8 > MAX_GENERATED_BYTES
        or spec.get("kind") != "lowrank_x"
    ):
        raise ValueError("x_spec must describe one bounded low-rank X tile")
    for name, limit in (
        ("rows", 1 << 21),
        ("columns", 1 << 21),
        ("latent", 64),
        ("row_offset", 1 << 21),
        ("column_offset", 1 << 21),
        ("seed", 1 << 62),
    ):
        if type(spec.get(name)) is not int or not 0 <= spec[name] <= limit:
            raise ValueError(f"x_spec {name} is out of range")


def _latent_x(spec, left, right):
    """One X tile from its latent rows, bounded and shape-checked like
    _generated_x."""
    _generated_x_check(spec)
    from ygaraty.models.factorization_sharded import x_from_latent

    left, right = (np.asarray(a, dtype=np.float64) for a in (left, right))
    rows, columns = spec["shape"]
    if (
        left.shape != (rows, spec["latent"])
        or right.shape != (columns, spec["latent"])
        or not (np.isfinite(left).all() and np.isfinite(right).all())
    ):
        raise ValueError(
            "latent rows must be finite and match the x_spec tile"
        )
    return x_from_latent(dict(spec), left, right)


def _tile_group(inputs, attrs):
    """`factorization_tiles` (a plan's `group`): the p x q tiles of row shards
    `row_shards` and column
    shards `column_shards`, one at a time, their gradients summed per shard in
    a fixed order. The
    result packs one band per row shard, one per column shard, then the loss
    (the CPU kernel's
    layout for one tile, with p + q bands). Inputs: the latent rows of the
    shards (p + q, with
    latent_inputs) then U_i (p) and V_j (q)."""
    rows, columns, spec, count = (
        attrs.get("row_shards"),
        attrs.get("column_shards"),
        attrs.get("x_spec"),
        attrs.get("total_elements"),
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
        not bounds(rows)
        or not bounds(columns)
        or not isinstance(spec, Mapping)
        or type(count) is not int
    ):
        raise ValueError(
            "factorization_tiles needs row_shards, column_shards, an "
            "x_spec and total_elements"
        )
    p, q = len(rows), len(columns)
    if len(inputs) == 2 * (p + q):
        left, right, inputs = inputs[:p], inputs[p : p + q], inputs[p + q :]
    elif len(inputs) == p + q:
        left = right = None
    else:
        raise ValueError(
            "factorization_tiles needs one input per shard (U then "
            "V), with their latent rows first"
        )
    u, v = inputs[:p], inputs[p:]
    rank = np.asarray(u[0]).shape[1] if np.asarray(u[0]).ndim == 2 else 0
    sizes = [b[1] - b[0] for b in rows] + [b[1] - b[0] for b in columns]
    out = np.zeros((sum(sizes) + 1, rank), dtype=np.float64)
    row_offsets = np.cumsum([0] + sizes[:p]).tolist()
    column_offsets = (sum(sizes[:p]) + np.cumsum([0] + sizes[p:])).tolist()
    for a, (r0, r1) in enumerate(rows):
        for b, (c0, c1) in enumerate(columns):
            tile_spec = dict(
                spec, shape=[r1 - r0, c1 - c0], row_offset=r0, column_offset=c0
            )
            tile_inputs = (
                [u[a], v[b]]
                if left is None
                else [left[a], right[b], u[a], v[b]]
            )
            tile = execute(
                "factorization_tile",
                tile_inputs,
                dict(x_spec=tile_spec, total_elements=count),
            )
            size, width = r1 - r0, c1 - c0
            out[row_offsets[a] : row_offsets[a] + size] += tile[:size]
            out[column_offsets[b] : column_offsets[b] + width] += tile[
                size : size + width
            ]
            out[-1, 0] += tile[-1, 0]
    return out


def execute(op, inputs, attrs):
    """Return packed data gradients and loss for one immutable shard or tile.

    The result has rows + columns + 1 rows of `rank` columns: the gradient of
    the loss with respect
    to this block's U rows, then the gradient with respect to its V rows, then
    the block's share of
    the loss in the first column of the last row. All blocks in an update read
    the same U/V
    snapshot; the gradients are normalized by the whole matrix's element count
    (`total_elements`),
    so blocks add up to the full gradient. Regularization and optimizer updates
    belong to the
    surrounding graph, not this kernel.

    With `x_spec` in the attributes of a tile, X is not an input: it is
    generated from the spec, from
    inputs (U, V), or from (left, right, U, V) given the tile's rows of X's
    latent factors (a plan's
    `latent_inputs`, which WebGPU workers need); both give the same values.

    A block's total_elements must be a multiple of its width (the row shards'
    whole-matrix count
    always is); a tile's only has to cover the tile, since its width need not
    divide the matrix's.
    """
    if op not in OPS:
        raise ValueError(f"Unsupported factorization operation: {op}")
    if not isinstance(attrs, Mapping):
        raise ValueError(f"{op} requires attributes")
    if op == "factorization_tiles":
        return _tile_group(inputs, attrs)
    if (
        len(inputs) == 2
        and op == "factorization_tile"
        and isinstance(attrs.get("x_spec"), Mapping)
    ):
        inputs = [
            _generated_x(attrs["x_spec"]),
            *inputs,
        ]  # the worker builds its own X tile from the seeded spec
    elif (
        len(inputs) == 4
        and op == "factorization_tile"
        and isinstance(attrs.get("x_spec"), Mapping)
    ):
        inputs = [
            _latent_x(attrs["x_spec"], inputs[0], inputs[1]),
            *inputs[2:],
        ]
    elif len(inputs) != 3:
        raise ValueError(f"{op} requires X, U, V and attributes")
    arrays = []
    for name, value in zip(("X", "U", "V"), inputs):
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or raw.ndim != 2 or 0 in raw.shape:
            raise ValueError(f"{name} must be a nonempty real matrix")
        array = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite values")
        arrays.append(array)
    x, u, v = arrays
    rows, columns = x.shape
    if u.shape[0] != rows or v.shape != (columns, u.shape[1]):
        raise ValueError("incompatible factorization block shapes")
    count = attrs.get("total_elements")
    if (
        type(count) is not int
        or count < x.size
        or (op == "factorization_block" and count % columns)
    ):
        raise ValueError(
            "total_elements must be the positive whole-matrix element count"
        )
    # Keep the dense residual local. Only the two gradient bands and scalar
    # loss are uploaded; input arrays are never changed in place.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = u @ v.T - x
        rank = u.shape[1]
        result = np.zeros((rows + columns + 1, rank), dtype=np.float64)
        result[:rows] = (residual @ v) / count
        result[rows : rows + columns] = (residual.T @ u) / count
        result[-1, 0] = np.sum(residual * residual) / (2 * count)
    if not np.isfinite(result).all():
        raise ValueError("Kernel produced invalid/nonfinite output")
    return result
