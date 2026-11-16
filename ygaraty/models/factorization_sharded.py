"""Two-dimensional task-distributed SGD factorization: X_ij, U_i and V_j are
all a task ever reads.

X (rows x columns) is cut into tiles by `block_rows` and `block_columns`; U
(rows x rank) into row
shards that line up with X's tile rows and V (columns x rank) into shards that
line up with X's tile
columns. Every shard is a leaf of its own (`X/i/j`, `U/i`, `V/j`), so no task,
worker or tensor ever
names a complete factor:

* `tile/i/j` (`factorization_tile`) reads X/i/j, U/i and V/j, forms the
residual locally and returns
  its share of dU_i, dV_j and the loss, packed. The dense residual never leaves the worker.
* dU_i is the sum over j of the tiles' U bands and dV_j the sum over i of their
V bands: bounded
  `add_many` trees (fan-in `fan_in`; `bounded_reductions` also bounds by the task budget).
* L2 regularization is added once per shard, to the shard's own gradient.
* `update/U/i` and `update/V/j` are SGD steps on one shard with that shard's
gradient.
* The loss is a bounded sum of the tiles' scalars plus the shards' squared
norms.

An update reads U and V only through the leaves named in `plan.updates`' keys;
the update tasks write
new immutable versions, which a job installs as the leaves for the next update
after the last task
of this one. So every tile of an update sees the same factor snapshot, whatever
order the scheduler
runs the tasks in. The graph has no `assemble` task: a full U or V exists only
if a diagnostic joins
the shards (`join_shards`), never in the graph or on a worker.

Communication is what capacity costs. A tile reads b_r*rank + b_c*rank +
b_r*b_c elements and
returns (b_r + b_c + 1)*rank, so one update moves about (columns/b_c)*|U| + (rows/b_r)*|V| bytes of
partial gradients on top of the factor and X reads; smaller tiles fit smaller
workers and move more.

Graph construction is separate from execution and holds no arrays.
`initializers` and `shard_values`
generate the seeded problem (a low-rank X plus noise, small random factors)
shard by shard, so any
shard layout sees the same values and no process needs a whole matrix.
"""  # noqa: E501 -- preserve documentation layout

import hashlib
import math
from functools import lru_cache
from numbers import Real
from types import SimpleNamespace

import numpy as np

from ygaraty.compiler.plan import (
    Input,
    Task,
    admit_plan,
    bounded_reductions,
)

MASK64 = (1 << 64) - 1


def shard_bounds(total, block):
    """[(start, stop), ...]: consecutive shards of `block` entries, the last
    one possibly shorter."""
    return [
        (start, min(total, start + block)) for start in range(0, total, block)
    ]


def x_leaf(i, j):
    return f"X/{i}/{j}"


def xl_leaf(i):
    return f"XL/{i}"


def xr_leaf(j):
    return f"XR/{j}"


def u_leaf(i):
    return f"U/{i}"


def v_leaf(j):
    return f"V/{j}"


# Seeded problem, independent of the shard layout
# -------------------------------------------------


def _row_generator(seed, name, row):
    key = f"{seed}:{name}:{row}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    )


def factor_rows(seed, name, start, stop, width, scale):
    """Rows start..stop of a normal(0, scale) matrix of `width` columns; row r
    depends only on (seed, name, r)."""
    return np.stack(
        [
            _row_generator(seed, name, row).normal(0.0, scale, width)
            for row in range(start, stop)
        ]
    )


@lru_cache(maxsize=16)
def _latent(seed, name, count, width):
    """The whole latent factor of X (count x width, small): shards slice it.
    Cached per process."""
    return factor_rows(seed, name, 0, count, width, 1.0)


def noise_words(seed, rows, cols, columns):
    """hashed_noise's 64-bit splitmix64 word for entries (rows, cols) of a
    matrix `columns` wide
    (uint64 arrays that broadcast). WebGPU workers compute the same words
    (experiment 48)."""
    salt = np.uint64((seed * 0x9E3779B97F4A7C15) & MASK64)
    with np.errstate(over="ignore"):
        z = (
            np.asarray(rows, dtype=np.uint64) * np.uint64(columns)
            + np.asarray(cols, dtype=np.uint64)
            + salt
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def hashed_noise(
    seed, row_start, row_stop, column_start, column_stop, columns
):
    """Zero-mean, unit-variance noise whose (r, c) entry depends only on (seed,
    r, c, columns).

    A splitmix64 hash of the entry's index feeds two 26-bit uniforms, summed
    (triangular), so a tile
    is generated without generating the rows or columns around it."""
    z = noise_words(
        seed,
        np.arange(row_start, row_stop, dtype=np.uint64)[:, None],
        np.arange(column_start, column_stop, dtype=np.uint64)[None, :],
        columns,
    )
    first = (z >> np.uint64(38)).astype(np.float64) / (1 << 26)
    second = ((z >> np.uint64(12)) & np.uint64((1 << 26) - 1)).astype(
        np.float64
    ) / (1 << 26)
    return (first + second - 1.0) * math.sqrt(6.0)


def x_values(spec, region=None):
    """X (or a region of it, [i, ei, j, ej] relative to the spec's tile): a
    low-rank product plus noise."""
    shape = spec["shape"]
    i, ei, j, ej = region or [0, shape[0], 0, shape[1]]
    row0, col0 = spec["row_offset"] + i, spec["column_offset"] + j
    row1, col1 = spec["row_offset"] + ei, spec["column_offset"] + ej
    latent = spec["latent"]
    left = _latent(spec["seed"], "X-left", spec["rows"], latent)[row0:row1]
    right = _latent(spec["seed"], "X-right", spec["columns"], latent)[
        col0:col1
    ]
    return x_from_latent(spec, left, right, row0, col0)


def x_from_latent(spec, left, right, row0=None, col0=None):
    """X entries from their rows of the two latent factors (left: rows x
    latent, right: columns x
    latent), the tile's own by default. A worker given those rows
    (`latent_inputs`) builds the
    same values as x_values without generating the whole latent factor."""
    row0 = spec["row_offset"] if row0 is None else row0
    col0 = spec["column_offset"] if col0 is None else col0
    latent = spec["latent"]
    if left.shape[1] != latent or right.shape[1] != latent:
        raise ValueError("latent rows must have x_spec latent columns")
    # A fixed-order sum over the latent dimension (not a BLAS product, whose
    # rounding depends on the
    # slice's shape): each entry is the same floating-point expression in every
    # shard layout.
    values = left[:, :1] * right[:, 0][None, :]
    for k in range(1, latent):
        values += left[:, k : k + 1] * right[:, k][None, :]
    values /= math.sqrt(latent)
    if spec["noise"]:
        values = values + spec["noise"] * hashed_noise(
            spec["seed"],
            row0,
            row0 + left.shape[0],
            col0,
            col0 + right.shape[0],
            spec["columns"],
        )
    return values


def shard_values(spec, region=None):
    """The values of a leaf described by `initializers`' spec (or a region of
    it, relative to the leaf)."""
    shape = spec["shape"]
    i, ei, j, ej = region or [0, shape[0], 0, shape[1]]
    if spec["kind"] == "factor_rows":
        return factor_rows(
            spec["seed"],
            spec["factor"],
            spec["row_offset"] + i,
            spec["row_offset"] + ei,
            shape[1],
            spec["scale"],
        )[:, j:ej]
    if spec["kind"] == "lowrank_x":
        return x_values(spec, [i, ei, j, ej])
    raise ValueError(f"Unsupported factorization initializer: {spec['kind']}")


def initializers(
    rows,
    columns,
    rank,
    block_rows,
    block_columns,
    *,
    seed=2026,
    scale=0.1,
    latent=8,
    noise=0.05,
):
    """Leaf name -> initializer spec (data only, a few hundred bytes per leaf)
    for every shard and tile."""
    specs = {}
    row_shards, column_shards = (
        shard_bounds(rows, block_rows),
        shard_bounds(columns, block_columns),
    )
    for i, (start, stop) in enumerate(row_shards):
        specs[u_leaf(i)] = dict(
            kind="factor_rows",
            factor="U",
            shape=[stop - start, rank],
            row_offset=start,
            scale=scale,
            seed=seed,
        )
        for j, (cstart, cstop) in enumerate(column_shards):
            specs[x_leaf(i, j)] = dict(
                kind="lowrank_x",
                shape=[stop - start, cstop - cstart],
                row_offset=start,
                column_offset=cstart,
                rows=rows,
                columns=columns,
                latent=latent,
                noise=noise,
                seed=seed,
            )
    for j, (start, stop) in enumerate(column_shards):
        specs[v_leaf(j)] = dict(
            kind="factor_rows",
            factor="V",
            shape=[stop - start, rank],
            row_offset=start,
            scale=scale,
            seed=seed,
        )
    return specs


def join_shards(leaves, names):
    """A diagnostic's whole matrix from its shards, in order. Not part of any
    plan or worker."""
    return np.vstack([leaves[name] for name in names])


# The plan
# ----------------------------------------------------------------------------------------  # noqa: E501 -- preserve literal/report layout


def _joins(count, fan_in):
    total = 0
    while count > fan_in:
        count = -(-count // fan_in)
        total += count
    return total


class ShardedFactorizationPlan:
    """One plan assembly: validation, declarations, tasks, reductions and
    admission."""

    def __init__(
        self,
        rows,
        columns,
        rank,
        *,
        block_rows=256,
        block_columns=256,
        learning_rate=0.01,
        regularization=0.001,
        max_task_bytes=512 << 20,
        max_plan_tasks=100000,
        fan_in=8,
        seed=2026,
        scale=0.1,
        latent=8,
        noise=0.05,
        generated_x=False,
        latent_inputs=False,
        group=None,
        max_result_bytes=12 << 20,
        wire_bytes=8,
    ):
        self.rows = rows
        self.columns = columns
        self.rank = rank
        self.block_rows = block_rows
        self.block_columns = block_columns
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_task_bytes = max_task_bytes
        self.max_plan_tasks = max_plan_tasks
        self.fan_in = fan_in
        self.seed = seed
        self.scale = scale
        self.latent = latent
        self.noise = noise
        self.generated_x = generated_x
        self.latent_inputs = latent_inputs
        self.group = group
        self.max_result_bytes = max_result_bytes
        self.wire_bytes = wire_bytes

    def build(self):
        self._validate()
        self._declare()
        self._tiles()
        self._updates()
        self._loss()
        self._admit()
        return self._result()

    def _validate(self):
        """Validate options and count the worst-case graph before allocating
        tasks."""
        for name, value in (
            ("rows", self.rows),
            ("columns", self.columns),
            ("rank", self.rank),
            ("block_rows", self.block_rows),
            ("block_columns", self.block_columns),
            ("max_task_bytes", self.max_task_bytes),
            ("max_plan_tasks", self.max_plan_tasks),
            ("fan_in", self.fan_in),
            ("latent", self.latent),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.wire_bytes not in (4, 8):
            raise ValueError("wire_bytes must be 4 or 8")
        if self.fan_in < 2:
            raise ValueError("fan_in must be at least 2")
        if self.latent_inputs and (not self.generated_x):
            raise ValueError("latent_inputs needs generated_x")
        if self.group is not None:
            self.group = (
                (self.group, self.group)
                if type(self.group) is int
                else tuple(self.group)
            )
            if (
                not self.generated_x
                or len(self.group) != 2
                or (not all(type(g) is int and g >= 1 for g in self.group))
            ):
                raise ValueError(
                    "group must be a positive integer or pair, with "
                    "generated_x"
                )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name, value, allow_zero in (
            ("learning_rate", self.learning_rate, False),
            ("regularization", self.regularization, True),
            ("scale", self.scale, False),
            ("noise", self.noise, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or (not math.isfinite(value))
                or (value < 0)
                or (not allow_zero and value == 0)
            ):
                raise ValueError(
                    f"{name} must be finite and "
                    f"{('nonnegative' if allow_zero else 'positive')}"
                )
        self.learning_rate, self.regularization, self.scale, self.noise = (
            float(self.learning_rate),
            float(self.regularization),
            float(self.scale),
            float(self.noise),
        )
        self.nr, self.nc = (
            -(-self.rows // self.block_rows),
            -(-self.columns // self.block_columns),
        )
        gr, gc = (
            (-(-self.nr // self.group[0]), -(-self.nc // self.group[1]))
            if self.group
            else (self.nr, self.nc)
        )
        estimated = (
            gr * gc
            + (self.nr + self.nc) * (2 + (3 if self.regularization else 0))
            + self.nr * _joins(gc, self.fan_in)
            + self.nc * _joins(gr, self.fan_in)
            + 2
            + _joins(gr * gc, self.fan_in)
            + (self.nr + self.nc + 4) * bool(self.regularization)
            + _joins(self.nr + self.nc, self.fan_in)
        )
        if estimated > self.max_plan_tasks:
            raise ValueError(
                "estimated graph exceeds max_plan_tasks; increase "
                "block_rows/block_columns or reduce the matrix"
            )

    def _declare(self):
        """Describe input leaves and initialize fresh per-build assembly
        state."""
        self.row_shards, self.column_shards = (
            shard_bounds(self.rows, self.block_rows),
            shard_bounds(self.columns, self.block_columns),
        )
        self.shapes = {}
        self.specs = initializers(
            self.rows,
            self.columns,
            self.rank,
            self.block_rows,
            self.block_columns,
            seed=self.seed,
            scale=self.scale,
            latent=self.latent,
            noise=self.noise,
        )
        if self.latent_inputs:
            for i, (start, stop) in enumerate(self.row_shards):
                self.specs[xl_leaf(i)] = dict(
                    kind="factor_rows",
                    factor="X-left",
                    shape=[stop - start, self.latent],
                    row_offset=start,
                    scale=1.0,
                    seed=self.seed,
                )
            for j, (start, stop) in enumerate(self.column_shards):
                self.specs[xr_leaf(j)] = dict(
                    kind="factor_rows",
                    factor="X-right",
                    shape=[stop - start, self.latent],
                    row_offset=start,
                    scale=1.0,
                    seed=self.seed,
                )
        for name, spec in self.specs.items():
            if not (self.generated_x and name.startswith("X/")):
                self.shapes[name] = tuple(spec["shape"])
        self.tasks, self.losses, self.norms = ([], [], [])
        self.template = dict(
            kind="lowrank_x",
            rows=self.rows,
            columns=self.columns,
            latent=self.latent,
            noise=self.noise,
            seed=self.seed,
        )
        self.p, self.q = self.group or (1, 1)
        self.tile_parts = {}

    def _tiles(self):
        """Visit tile groups in deterministic row-major order."""
        for gi in range(0, self.nr, self.p):
            row_block = list(range(gi, min(gi + self.p, self.nr)))
            for gj in range(0, self.nc, self.q):
                column_block = list(range(gj, min(gj + self.q, self.nc)))
                self._tile_group(row_block, column_block, gi, gj)

    def _tile_group(self, row_block, column_block, gi, gj):
        sizes = [
            self.row_shards[i][1] - self.row_shards[i][0] for i in row_block
        ] + [
            self.column_shards[j][1] - self.column_shards[j][0]
            for j in column_block
        ]
        extra_work = self._tile_work(row_block, column_block)
        attrs = dict(
            total_elements=self.rows * self.columns,
            extra_work=extra_work,
            upload_rows=sizes + [1],
        )
        if self.group is None:
            i, j = (row_block[0], column_block[0])
            ident, op = (f"tile/{i}/{j}", "factorization_tile")
            inputs = [Input(u_leaf(i)), Input(v_leaf(j))]
            if self.latent_inputs:
                inputs = [Input(xl_leaf(i)), Input(xr_leaf(j)), *inputs]
            if self.generated_x:
                attrs["x_spec"] = self.specs[x_leaf(i, j)]
            else:
                inputs.insert(0, Input(x_leaf(i, j)))
        else:
            ident, op = (
                f"tiles/{gi // self.p}/{gj // self.q}",
                "factorization_tiles",
            )
            inputs = [Input(u_leaf(i)) for i in row_block] + [
                Input(v_leaf(j)) for j in column_block
            ]
            if self.latent_inputs:
                inputs = (
                    [Input(xl_leaf(i)) for i in row_block]
                    + [Input(xr_leaf(j)) for j in column_block]
                    + inputs
                )
            attrs.update(
                x_spec=self.template,
                row_shards=[list(self.row_shards[i]) for i in row_block],
                column_shards=[
                    list(self.column_shards[j]) for j in column_block
                ],
            )
        total_rows = sum(sizes)
        if (
            self.group is not None
            and (total_rows + 1) * self.rank * self.wire_bytes
            > self.max_result_bytes
        ):
            raise MemoryError(
                f"Task {ident} returns "
                f"{(total_rows + 1) * self.rank * self.wire_bytes} "
                f"bytes; at most {self.max_result_bytes}. Reduce the "
                f"group or the tile."
            )
        self.tasks.append(
            Task(ident, op, inputs, (total_rows + 1, self.rank), attrs)
        )
        self.losses.append(Input(ident, [total_rows, total_rows + 1, 0, 1]))
        self._record_bands(ident, row_block, column_block)

    def _tile_work(self, row_block, column_block):
        """Conservative decomposed costs and generated-X scratch-space
        guard."""
        extra_work = []
        for i in row_block:
            count = self.row_shards[i][1] - self.row_shards[i][0]
            for j in column_block:
                width = self.column_shards[j][1] - self.column_shards[j][0]
                extra_work += [("matmul", count * width * self.rank)] * 3
                extra_work += [
                    ("scale", 2 * count * width),
                    ("add", 3 * count * width),
                    ("scale", 2 * count * self.rank),
                    ("scale", 2 * width * self.rank),
                    ("mul", 3 * count * width),
                    ("sum", count * width + 1),
                    ("scale", 2),
                ]
                if self.generated_x:
                    extra_work.append(("scale", count * width))
                    generated = 4 * count * width * 8
                    if generated > self.max_task_bytes:
                        raise MemoryError(
                            f"Task tile/{i}/{j} generates an X tile of "
                            f"{count * width * 8} bytes; its estimate "
                            f"exceeds the budget {self.max_task_bytes}."
                        )
        return extra_work

    def _record_bands(self, ident, row_block, column_block):
        offset = 0
        for i in row_block:
            for j in column_block:
                self.tile_parts[i, j] = (ident, offset, None)
            offset += self.row_shards[i][1] - self.row_shards[i][0]
        for j in column_block:
            for i in row_block:
                self.tile_parts[i, j] = (*self.tile_parts[i, j][:2], offset)
            offset += self.column_shards[j][1] - self.column_shards[j][0]

    def _updates(self):
        """Assemble or reduce gradients and apply regularization once per
        factor element."""
        self.gradients, self.updates = ({}, {})
        for factor, shards, leaf, band, label in (
            (
                "U",
                self.row_shards,
                u_leaf,
                lambda i, j, a, b: [0, a, 0, self.rank],
                "row",
            ),
            ("V", self.column_shards, v_leaf, None, "column"),
        ):
            for index, (start, stop) in enumerate(shards):
                size = stop - start
                if factor == "U":
                    holders = dict.fromkeys(
                        (
                            self.tile_parts[index, j][0],
                            self.tile_parts[index, j][1],
                        )
                        for j in range(self.nc)
                    )
                else:
                    holders = dict.fromkeys(
                        (
                            self.tile_parts[i, index][0],
                            self.tile_parts[i, index][2],
                        )
                        for i in range(self.nr)
                    )
                parts = [
                    Input(ident, [row, row + size, 0, self.rank])
                    for ident, row in holders
                ]
                data = (
                    f"gradient/{factor}/data/{index}"
                    if self.regularization
                    else f"gradient/{factor}/{index}"
                )
                self.tasks.append(
                    Task(data, "add_many", parts, (size, self.rank))
                )
                gradient = Input(data)
                if self.regularization:
                    self.tasks.append(
                        Task(
                            f"gradient/{factor}/regularization/{index}",
                            "scale",
                            [Input(leaf(index))],
                            (size, self.rank),
                            {"factor": self.regularization},
                        )
                    )
                    self.tasks.append(
                        Task(
                            f"gradient/{factor}/{index}",
                            "add",
                            [
                                gradient,
                                Input(
                                    f"gradient/{factor}/regularization/{index}"
                                ),
                            ],
                            (size, self.rank),
                        )
                    )
                    gradient = Input(f"gradient/{factor}/{index}")
                    self.tasks.append(
                        Task(
                            f"factor_norm/{factor}/{index}",
                            "sum_squares",
                            [Input(leaf(index))],
                            (1, 1),
                        )
                    )
                    self.norms.append(Input(f"factor_norm/{factor}/{index}"))
                self.tasks.append(
                    Task(
                        f"update/{factor}/{index}",
                        "sgd",
                        [Input(leaf(index)), gradient],
                        (size, self.rank),
                        {"learning_rate": self.learning_rate},
                    )
                )
                self.gradients[leaf(index)] = gradient.value
                self.updates[leaf(index)] = f"update/{factor}/{index}"

    def _loss(self):
        """Combine data loss and factor norms without changing reduction
        order."""
        self.tasks.append(
            Task(
                "data_loss" if self.regularization else "loss",
                "add_many",
                self.losses,
                (1, 1),
            )
        )
        if self.regularization:
            self.tasks.append(
                Task("factor_norm", "add_many", self.norms, (1, 1))
            )
            self.tasks.append(
                Task(
                    "regularization_loss",
                    "scale",
                    [Input("factor_norm")],
                    (1, 1),
                    {"factor": self.regularization / 2},
                )
            )
            self.tasks.append(
                Task(
                    "loss",
                    "add",
                    [Input("data_loss"), Input("regularization_loss")],
                    (1, 1),
                )
            )

    def _admit(self):
        """Bound reductions and enforce task-count and memory limits."""
        self.tasks = bounded_reductions(
            self.tasks, fan_in=self.fan_in, budget=self.max_task_bytes
        )
        if len(self.tasks) > self.max_plan_tasks:
            raise ValueError("compiled graph exceeds max_plan_tasks")
        self.admitted = admit_plan(
            self.tasks,
            self.shapes,
            self.max_task_bytes,
            output_budget=self.max_task_bytes,
        )
        if self.generated_x:

            def largest_tile(t):
                if t.op == "factorization_tile":
                    return math.prod(t.attrs["x_spec"]["shape"])
                return max(
                    (r[1] - r[0]) * (c[1] - c[0])
                    for r in t.attrs["row_shards"]
                    for c in t.attrs["column_shards"]
                )

            self.admitted = max(
                self.admitted,
                max(
                    4
                    * (
                        sum(math.prod(self.shapes[x.value]) for x in t.inputs)
                        + math.prod(t.shape)
                        + largest_tile(t)
                    )
                    * 8
                    for t in self.tasks
                    if t.op in ("factorization_tile", "factorization_tiles")
                ),
            )
            if self.admitted > self.max_task_bytes:
                raise MemoryError(
                    f"a generated-X tile needs an estimated "
                    f"{self.admitted} bytes; budget is "
                    f"{self.max_task_bytes}. Reduce the tile."
                )

    def _result(self):
        """Publish the existing plan metadata contract."""
        version = f"synthetic-factorization-X-v1:{self.seed}:{self.rows}x{self.columns}:{self.latent}:{self.noise}"  # noqa: E501 -- preserve literal/report layout
        persistent = (
            {}
            if self.generated_x
            else {
                x_leaf(i, j): f"{version}:{r[0]}-{r[1]}:{c[0]}-{c[1]}"
                for i, r in enumerate(self.row_shards)
                for j, c in enumerate(self.column_shards)
            }
        )
        return SimpleNamespace(
            tasks=self.tasks,
            leaves={
                name: dict(shape=list(shape))
                for name, shape in self.shapes.items()
            },
            owner_keys=[],
            owned=False,
            reusable=set(self.shapes),
            persistent_inputs=persistent,
            updates=self.updates,
            gradients=self.gradients,
            loss="loss",
            admitted_task_bytes=self.admitted,
            shards=dict(rows=self.row_shards, columns=self.column_shards),
            initializers={
                n: v for n, v in self.specs.items() if self.shapes.get(n)
            },
            model=dict(
                kind="dense_sgd_factorization",
                implementation="tiles_2d",
                rows=self.rows,
                columns=self.columns,
                rank=self.rank,
                block_rows=self.block_rows,
                block_columns=self.block_columns,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                fan_in=self.fan_in,
                seed=self.seed,
                latent=self.latent,
                noise=self.noise,
                scale=self.scale,
                generated_x=self.generated_x,
                latent_inputs=self.latent_inputs,
                group=list(self.group) if self.group else None,
            ),
        )


def build_sharded_plan(
    rows,
    columns,
    rank,
    *,
    block_rows=256,
    block_columns=256,
    learning_rate=0.01,
    regularization=0.001,
    max_task_bytes=512 << 20,
    max_plan_tasks=100000,
    fan_in=8,
    seed=2026,
    scale=0.1,
    latent=8,
    noise=0.05,
    generated_x=False,
    latent_inputs=False,
    group=None,
    max_result_bytes=12 << 20,
    wire_bytes=8,
):
    """A shape-only full-batch SGD update over 2D tiles. See the module
    docstring for the graph.

    `max_task_bytes` is the admission budget of every task (admit_plan: four
    times the bytes it
    reads and writes, and no result above 8 MiB); a tile or reduction that does
    not fit raises
    MemoryError, so a plan that builds fits the budget it was built for.
    `seed`, `scale`, `latent`
    and `noise` describe the seeded problem (`initializers`), not the graph.

    With `generated_x` no X leaf exists: a tile task reads only U_i and V_j and
    its worker generates the
    tile from the seeded spec in the task's attributes (`x_spec`), so X is
    never stored or sent, and a
    matrix larger than any machine's memory is a plan like any other. The
    tile's admission counts the
    generated block as an input.

    With `latent_inputs` (generated_x only) a tile also reads its rows of X's
    two latent factors
    (leaves XL/i and XR/j, latent columns each, never updated): workers that
    cannot run NumPy's
    generator (WebGPU browsers, experiment 48) build X from them and the hashed
    noise.

    With `group` = (p, q) (generated_x only) one task, `tiles/I/J` with op
    `factorization_tiles`,
    computes the p x q tiles of row shards I and column shards J one at a time
    and returns the
    summed gradient of each of its p + q shards: bytes follow the group's
    perimeter while compute
    follows its area, so a 5 x 5 group moves one fifth of the result bytes of
    its 25 tiles, at the
    memory of one tile. An int is a square group; the last groups along each
    side may be smaller. A
    group's result ((p + q) bands) may exceed the usual 8 MiB result cap, up to
    `max_result_bytes`
    (12 MiB: one result must fit a 16 MiB request body with its header) at
    `wire_bytes` per value
    (8; 4 on a float32 wire, which fits a group twice as long on a side).
    """
    return ShardedFactorizationPlan(
        rows,
        columns,
        rank,
        block_rows=block_rows,
        block_columns=block_columns,
        learning_rate=learning_rate,
        regularization=regularization,
        max_task_bytes=max_task_bytes,
        max_plan_tasks=max_plan_tasks,
        fan_in=fan_in,
        seed=seed,
        scale=scale,
        latent=latent,
        noise=noise,
        generated_x=generated_x,
        latent_inputs=latent_inputs,
        group=group,
        max_result_bytes=max_result_bytes,
        wire_bytes=wire_bytes,
    ).build()


# Locality
# ----------------------------------------------------------------------------------------  # noqa: E501 -- preserve literal/report layout


def audit_locality(plan):
    """What every task reads, against the 2D contract. Returns a report;
    `violations` is empty when
    no task reads a complete factor (or a leaf outside its own shard indices)
    and the graph has no
    assembly of one. A factor with one shard is its own whole by construction
    and not a violation."""
    shapes = {name: tuple(leaf["shape"]) for name, leaf in plan.leaves.items()}
    shapes.update((task.id, tuple(task.shape)) for task in plan.tasks)
    rows, columns = plan.shards["rows"], plan.shards["columns"]
    violations, widest = [], {}
    for task in plan.tasks:
        if task.op in ("assemble", "transpose"):
            violations.append(
                f"{task.id}: {task.op} builds a whole value inside the graph"
            )
            continue
        read = 0
        for ref in task.inputs:
            dims = (
                (ref.region[1] - ref.region[0], ref.region[3] - ref.region[2])
                if ref.region
                else shapes.get(ref.value, (0, 0))
            )
            read += math.prod(dims) * 8
            if ref.value in ("U", "V") or (
                ref.value.split("/")[0] in ("U", "V")
                and ref.value.count("/") == 0
            ):
                violations.append(f"{task.id}: reads the whole {ref.value}")
        family = task.op if task.op != "add_many" else "add_many"
        widest[family] = max(widest.get(family, 0), read)
        if task.op == "factorization_tile":
            i, j = (int(part) for part in task.id.split("/")[1:])
            expected = (
                [u_leaf(i), v_leaf(j)]
                if "x_spec" in task.attrs
                else [x_leaf(i, j), u_leaf(i), v_leaf(j)]
            )
            if [ref.value for ref in task.inputs] != expected or any(
                ref.region for ref in task.inputs
            ):
                violations.append(
                    f"{task.id}: must read exactly " + ", ".join(expected)
                )
            elif "x_spec" in task.attrs and (
                task.attrs["x_spec"].get("row_offset"),
                task.attrs["x_spec"].get("column_offset"),
            ) != (rows[i][0], columns[j][0]):
                violations.append(
                    f"{task.id}: generates a tile that is not its own"
                )
        elif task.op == "factorization_tiles":
            # A group reads exactly the shards it lists (contiguous, its own),
            # their latent rows first if any.
            r_idx = [
                rows.index(tuple(b)) if tuple(b) in rows else None
                for b in task.attrs.get("row_shards", [])
            ]
            c_idx = [
                columns.index(tuple(b)) if tuple(b) in columns else None
                for b in task.attrs.get("column_shards", [])
            ]

            def contiguous(idx):
                return (
                    idx
                    and None not in idx
                    and idx == list(range(idx[0], idx[0] + len(idx)))
                )

            if not contiguous(r_idx) or not contiguous(c_idx):
                violations.append(
                    f"{task.id}: its shards are not a contiguous block "
                    f"of the plan's"
                )
            else:
                expected = [u_leaf(i) for i in r_idx] + [
                    v_leaf(j) for j in c_idx
                ]
                if any(ref.value.startswith("XL/") for ref in task.inputs):
                    expected = (
                        [xl_leaf(i) for i in r_idx]
                        + [xr_leaf(j) for j in c_idx]
                        + expected
                    )
                if [ref.value for ref in task.inputs] != expected or any(
                    ref.region for ref in task.inputs
                ):
                    violations.append(
                        f"{task.id}: must read exactly " + ", ".join(expected)
                    )
        elif task.op in ("sgd", "scale", "sum_squares") and task.inputs[
            0
        ].value[:2] in ("U/", "V/"):
            leaf = task.inputs[0].value
            if any(
                ref.value not in (leaf,)
                and not ref.value.endswith("/" + leaf.split("/", 1)[1])
                for ref in task.inputs[1:]
            ):
                violations.append(
                    f"{task.id}: a shard-local step reads another shard"
                )
    for leaf, shape in shapes.items():
        if leaf[:2] in ("U/", "V/"):
            index = int(leaf[2:])
            bounds = (rows if leaf[0] == "U" else columns)[index]
            if shape[0] != bounds[1] - bounds[0]:
                violations.append(f"{leaf}: shape does not match its shard")
    return dict(
        violations=violations,
        widest_read_bytes=widest,
        tasks=len(plan.tasks),
        tiles=len(rows) * len(columns),
        tile_tasks=sum(
            t.op in ("factorization_tile", "factorization_tiles")
            for t in plan.tasks
        ),
        row_shards=len(rows),
        column_shards=len(columns),
    )
