"""Dense low-rank factorization with a differentiable squared-error objective.

The observed matrix X has shape (rows, columns), U is (rows, rank), and V is
(columns, rank). This recipe treats every entry as observed, including zeros;
it is not a sparse recommendation objective with missing observations.

Graph construction is separate from execution. With no values supplied, only
shapes and initializer metadata are retained, suitable for offline simulation.
"""

import math
from collections.abc import Mapping
from numbers import Real
from types import SimpleNamespace

import numpy as np

from ygaraty.compiler.plan import (
    Input,
    Task,
    admit_plan,
    bounded_reductions,
)
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.compiler.tape import Tape


class FactorizationGraph:
    """Dense squared-error objective and optional L2 penalty, without
    differentiation."""

    def __init__(
        self, rows, columns, rank, *, regularization=0.0, values=None
    ):
        for name, value in (
            ("rows", rows),
            ("columns", columns),
            ("rank", rank),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, Real)
            or not math.isfinite(regularization)
            or regularization < 0
        ):
            raise ValueError("regularization must be finite and nonnegative")
        regularization = float(regularization)
        self.rows, self.columns = rows, columns
        self.regularization = regularization
        shapes = {
            "X": (rows, columns),
            "U": (rows, rank),
            "V": (columns, rank),
        }
        if values is None:
            tape = ShapeTape()
            for name, shape in shapes.items():
                initializer = (
                    dict(kind="input")
                    if name == "X"
                    else dict(kind="normal", scale=0.1)
                )
                tape.declare(name, shape, initializer, parameter=name != "X")
            # Match Tape.parameter's contract for shape-only parameter
            # gradients.
            tape.parameter_like.update(("U", "V"))
        else:
            if not isinstance(values, Mapping) or set(values) != set(shapes):
                raise ValueError("values must contain exactly X, U and V")
            tape = Tape()
            for name, shape in shapes.items():
                array = np.asarray(values[name], dtype=np.float64)
                if array.shape != shape:
                    raise ValueError(f"{name} must have shape {shape}")
                (tape.input if name == "X" else tape.parameter)(name, array)
        self.tape = tape

    def training_loss(self):
        tape, rows, columns = self.tape, self.rows, self.columns
        regularization = self.regularization
        prediction = tape.matmul("U", tape.transpose("V"), name="prediction")
        residual = tape.add(prediction, tape.scale("X", -1.0), name="residual")
        squared_error = tape.mul(residual, residual, name="squared_error")
        data_loss = tape.scale(
            tape.sum(squared_error),
            1.0 / (2 * rows * columns),
            name="data_loss" if regularization else "loss",
        )
        if regularization:
            u_squared = tape.sum(tape.mul("U", "U"))
            v_squared = tape.sum(tape.mul("V", "V"))
            penalty = tape.scale(
                tape.add(u_squared, v_squared), regularization / 2.0
            )
            loss = tape.add(data_loss, penalty, name="loss")
        else:
            loss = data_loss
        return loss


def build_tape(rows, columns, rank, *, regularization=0.0, values=None):
    """Compatibility recipe for a differentiated dense factorization
    objective."""
    model = FactorizationGraph(
        rows, columns, rank, regularization=regularization, values=values
    )
    model.tape.backward(model.training_loss())
    return model.tape


class FusedFactorizationPlan:
    """One plan assembly: validation, declarations, tasks, reductions and
    admission."""

    def __init__(
        self,
        rows,
        columns,
        rank,
        *,
        block_rows=64,
        learning_rate=0.01,
        regularization=0.001,
        max_task_bytes=512 << 20,
        max_plan_tasks=100000,
    ):
        self.rows = rows
        self.columns = columns
        self.rank = rank
        self.block_rows = block_rows
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_task_bytes = max_task_bytes
        self.max_plan_tasks = max_plan_tasks

    def build(self):
        self._validate()
        self._declare()
        self._rows()
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
            ("max_task_bytes", self.max_task_bytes),
            ("max_plan_tasks", self.max_plan_tasks),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value, allow_zero in (
            ("learning_rate", self.learning_rate, False),
            ("regularization", self.regularization, True),
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
        self.learning_rate, self.regularization = (
            float(self.learning_rate),
            float(self.regularization),
        )
        blocks = (self.rows + self.block_rows - 1) // self.block_rows

        def join_tasks(count):
            total = 0
            while count > 2:
                count = (count + 1) // 2
                total += count
            return total

        estimated_tasks = 2 * blocks + 5 + 2 * join_tasks(blocks)
        if self.regularization:
            estimated_tasks += 3 * blocks + 6 + join_tasks(blocks + 1)
        if estimated_tasks > self.max_plan_tasks:
            raise ValueError(
                "estimated graph exceeds max_plan_tasks; increase "
                "block_rows or reduce rows"
            )

    def _declare(self):
        """Describe input leaves and initialize fresh per-build assembly
        state."""
        self.shapes = {
            "X": (self.rows, self.columns),
            "U": (self.rows, self.rank),
            "V": (self.columns, self.rank),
        }
        (
            self.tasks,
            self.u_gradients,
            self.u_updates,
            self.positions,
            self.v_parts,
            self.losses,
            self.norms,
        ) = ([], [], [], [], [], [], [])

    def _rows(self):
        """Build independent row gradients and U updates in deterministic
        order."""
        for start in range(0, self.rows, self.block_rows):
            stop = min(self.rows, start + self.block_rows)
            count = stop - start
            region = [start, stop, 0, self.rank]
            shape = (count, self.rank)
            ident = f"factorization/rows/{start}"
            extra_work = [("matmul", count * self.columns * self.rank)] * 3
            extra_work += [
                ("scale", 2 * count * self.columns),
                ("add", 3 * count * self.columns),
                ("scale", 2 * count * self.rank),
                ("scale", 2 * self.columns * self.rank),
                ("mul", 3 * count * self.columns),
                ("sum", count * self.columns + 1),
                ("scale", 2),
            ]
            self.tasks.append(
                Task(
                    ident,
                    "factorization_block",
                    [
                        Input("X", [start, stop, 0, self.columns]),
                        Input("U", region),
                        Input("V"),
                    ],
                    (count + self.columns + 1, self.rank),
                    dict(
                        total_elements=self.rows * self.columns,
                        extra_work=extra_work,
                    ),
                )
            )
            gradient = Input(ident, [0, count, 0, self.rank])
            if self.regularization:
                penalty = f"gradient/U/regularization/{start}"
                self.tasks.append(
                    Task(
                        penalty,
                        "scale",
                        [Input("U", region)],
                        shape,
                        {"factor": self.regularization},
                    )
                )
                grad = f"gradient/U/rows/{start}"
                self.tasks.append(
                    Task(grad, "add", [gradient, Input(penalty)], shape)
                )
                gradient = Input(grad)
                norm = f"factor_norm/U/{start}"
                self.tasks.append(
                    Task(norm, "sum_squares", [Input("U", region)], (1, 1))
                )
                self.norms.append(Input(norm))
            update = f"update/U/rows/{start}"
            self.tasks.append(
                Task(
                    update,
                    "sgd",
                    [Input("U", region), gradient],
                    shape,
                    {"learning_rate": self.learning_rate},
                )
            )
            self.u_gradients.append(gradient)
            self.u_updates.append(Input(update))
            self.positions.append(region)
            self.v_parts.append(
                Input(ident, [count, count + self.columns, 0, self.rank])
            )
            self.losses.append(
                Input(
                    ident,
                    [count + self.columns, count + self.columns + 1, 0, 1],
                )
            )

    def _updates(self):
        """Assemble or reduce gradients and apply regularization once per
        factor element."""
        for ident, refs in (
            ("gradient/U", self.u_gradients),
            ("update/U", self.u_updates),
        ):
            self.tasks.append(
                Task(
                    ident,
                    "assemble",
                    refs,
                    self.shapes["U"],
                    dict(shape=list(self.shapes["U"]), blocks=self.positions),
                )
            )
        v_data = "gradient/data/V" if self.regularization else "gradient/V"
        self.tasks.append(
            Task(v_data, "add_many", self.v_parts, self.shapes["V"])
        )
        if self.regularization:
            self.tasks.append(
                Task(
                    "gradient/V/regularization",
                    "scale",
                    [Input("V")],
                    self.shapes["V"],
                    {"factor": self.regularization},
                )
            )
            self.tasks.append(
                Task(
                    "gradient/V",
                    "add",
                    [Input(v_data), Input("gradient/V/regularization")],
                    self.shapes["V"],
                )
            )
        self.tasks.append(
            Task(
                "update/V",
                "sgd",
                [Input("V"), Input("gradient/V")],
                self.shapes["V"],
                {"learning_rate": self.learning_rate},
            )
        )

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
                Task("factor_norm/V", "sum_squares", [Input("V")], (1, 1))
            )
            self.tasks.append(
                Task(
                    "factor_norm",
                    "add_many",
                    [*self.norms, Input("factor_norm/V")],
                    (1, 1),
                )
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
            self.tasks, fan_in=8, budget=self.max_task_bytes
        )
        if len(self.tasks) > self.max_plan_tasks:
            raise ValueError("compiled graph exceeds max_plan_tasks")
        self.admitted = admit_plan(
            self.tasks, self.shapes, self.max_task_bytes
        )

    def _result(self):
        """Publish the existing plan metadata contract."""
        return SimpleNamespace(
            tasks=self.tasks,
            leaves={
                name: dict(shape=list(shape))
                for name, shape in self.shapes.items()
            },
            owner_keys=[],
            owned=False,
            reusable={"X", "U", "V"},
            persistent_inputs={
                "X": f"synthetic-factorization-X-v1:{self.rows}x{self.columns}"
            },
            updates={"U": "update/U", "V": "update/V"},
            gradients={"U": "gradient/U", "V": "gradient/V"},
            loss="loss",
            admitted_task_bytes=self.admitted,
            model=dict(
                kind="dense_sgd_factorization",
                implementation="fused_rows",
                rows=self.rows,
                columns=self.columns,
                rank=self.rank,
                block_rows=self.block_rows,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
            ),
        )


def build_fused_plan(
    rows,
    columns,
    rank,
    *,
    block_rows=64,
    learning_rate=0.01,
    regularization=0.001,
    max_task_bytes=512 << 20,
    max_plan_tasks=100000,
):
    """Build a shape-only full-batch SGD update with independent row shards.

    Each block computes data gradients and loss from the same input factors.
    U updates remain row-sharded; V partial gradients use a bounded reduction
    tree before one update. Each factor element receives L2 regularization
    exactly once. Block size is independent of the available worker count.

    The persistent X version describes a synthetic shape-only input, not its
    contents. A caller binding real data must supply its own immutable version
    before enabling a cache that persists across updates or changing X.
    """
    return FusedFactorizationPlan(
        rows,
        columns,
        rank,
        block_rows=block_rows,
        learning_rate=learning_rate,
        regularization=regularization,
        max_task_bytes=max_task_bytes,
        max_plan_tasks=max_plan_tasks,
    ).build()
