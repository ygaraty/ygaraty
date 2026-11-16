"""Append worker-side optimizer tasks without executing or storing tensors."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .plan import MAX_OUTPUT_BYTES
from .shapes import regions
from .tasks import Input, Task


def adam_options(config: Mapping[str, Any]) -> dict[str, float]:
    """Validate optimizer settings, preserving defaults and ranges."""
    options = dict(
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.999),
        epsilon=config.get("epsilon", 1e-8),
        clip_norm=config.get("clip_norm", 0.0),
    )
    if not all(
        isinstance(v, (float, int)) and math.isfinite(v)
        for v in options.values()
    ):
        raise ValueError("Optimizer settings must be finite numbers")
    if (
        not 0 <= options["beta1"] < 1
        or not 0 <= options["beta2"] < 1
        or options["epsilon"] <= 0
        or options["clip_norm"] < 0
    ):
        raise ValueError("Invalid optimizer settings")
    return options


class OptimizerPlanBuilder:
    """One append operation: clipping, update kernels and output assembly."""

    def __init__(
        self,
        tasks: Sequence[Task],
        shapes: Mapping[str, tuple[int, int]],
        tile_size: int,
        learning_rate: float,
        optimizer: str = "sgd",
        step: int = 1,
        options: Mapping[str, Any] | None = None,
        fused_bytes: int | None = None,
    ) -> None:
        if optimizer not in ("sgd", "adam") or step < 1:
            raise ValueError("Invalid optimizer or update index")
        self.options = adam_options(options or {})
        self.tasks = list(tasks)
        self.shapes, self.tile_size = shapes, tile_size
        self.learning_rate, self.optimizer = learning_rate, optimizer
        self.step, self.fused_bytes = step, fused_bytes
        self.updated: dict[str, str] = {}

    def _norm(self, parameters, gradients):
        if not self.options["clip_norm"]:
            return
        squares = []
        for parameter in parameters:
            for region in regions(self.shapes[parameter], self.tile_size):
                ident = f"norm/{parameter}/{region[0]}/{region[2]}"
                self.tasks.append(
                    Task(
                        ident,
                        "sum_squares",
                        [Input(gradients[parameter], region)],
                        (1, 1),
                    )
                )
                squares.append(Input(ident))
        self.tasks.append(
            Task("gradient_squares", "add_many", squares, (1, 1))
        )

    def _gradient(self, ident, gradient, region, shape):
        reference = Input(gradient, region)
        if self.options["clip_norm"]:
            self.tasks.append(
                Task(
                    ident + "/clip",
                    "clip_gradient",
                    [reference, Input("gradient_squares")],
                    shape,
                    self.options,
                )
            )
            reference = Input(ident + "/clip")
        return reference

    def _fused(self, parameter, gradient, parts, positions):
        rows, columns = self.shapes[parameter]
        # Admission counts four times the bytes of four inputs and three
        # outputs.
        band = max(
            1,
            min(
                rows,
                self.fused_bytes // (4 * 7 * 8 * columns),
                MAX_OUTPUT_BYTES // (3 * 8 * columns),
            ),
        )
        for start in range(0, rows, band):
            stop = min(rows, start + band)
            region, count = [start, stop, 0, columns], stop - start
            ident = f"update/{parameter}/{start}/0"
            grad = self._gradient(ident, gradient, region, (count, columns))
            inputs = [
                Input(name, region)
                for name in (
                    parameter,
                    f"adam_m_{parameter}",
                    f"adam_v_{parameter}",
                )
            ]
            self.tasks.append(
                Task(
                    ident,
                    "adam_step",
                    [*inputs, grad],
                    (3 * count, columns),
                    dict(
                        self.options,
                        step=self.step,
                        learning_rate=self.learning_rate,
                        upload_rows=[count, count, count],
                    ),
                )
            )
            for index, name in enumerate(
                (f"adam_m_{parameter}", f"adam_v_{parameter}", parameter)
            ):
                parts[name].append(
                    Input(
                        ident, [index * count, (index + 1) * count, 0, columns]
                    )
                )
            positions.append(region)

    def _tiled(self, parameter, gradient, parts, positions):
        for region in regions(self.shapes[parameter], self.tile_size):
            start, stop, left, right = region
            shape = (stop - start, right - left)
            ident = f"update/{parameter}/{start}/{left}"
            grad = self._gradient(ident, gradient, region, shape)
            if self.optimizer == "adam":
                moments = []
                for moment in ("m", "v"):
                    name, mid = (
                        f"adam_{moment}_{parameter}",
                        ident + "/" + moment,
                    )
                    self.tasks.append(
                        Task(
                            mid,
                            "adam_" + moment,
                            [Input(name, region), grad],
                            shape,
                            self.options,
                        )
                    )
                    parts[name].append(Input(mid))
                    moments.append(Input(mid))
                self.tasks.append(
                    Task(
                        ident,
                        "adam_update",
                        [Input(parameter, region), *moments],
                        shape,
                        dict(
                            self.options,
                            step=self.step,
                            learning_rate=self.learning_rate,
                        ),
                    )
                )
            else:
                self.tasks.append(
                    Task(
                        ident,
                        "sgd",
                        [Input(parameter, region), grad],
                        shape,
                        {"learning_rate": self.learning_rate},
                    )
                )
            parts[parameter].append(Input(ident))
            positions.append(region)

    def _assemble(self, parameter, parts, positions):
        for name, references in parts.items():
            ident = "updated/" + name
            shape = self.shapes[parameter]
            self.tasks.append(
                Task(
                    ident,
                    "assemble",
                    references,
                    tuple(shape),
                    {"shape": list(shape), "blocks": positions},
                )
            )
            self.updated[name] = ident

    def build(
        self,
        parameters: Sequence[str],
        gradients: Mapping[str, str],
    ) -> tuple[list[Task], dict[str, str]]:
        """Append once to this builder; use a new builder for each update
        plan.
        """
        self._norm(parameters, gradients)
        for parameter in parameters:
            names = [parameter]
            if self.optimizer == "adam":
                names += [f"adam_m_{parameter}", f"adam_v_{parameter}"]
            parts, positions = {name: [] for name in names}, []
            method = (
                self._fused
                if self.optimizer == "adam" and self.fused_bytes
                else self._tiled
            )
            method(parameter, gradients[parameter], parts, positions)
            self._assemble(parameter, parts, positions)
        return self.tasks, self.updated


def with_optimizer(
    tasks,
    shapes,
    parameters,
    gradients,
    tile_size,
    learning_rate,
    optimizer="sgd",
    step=1,
    options=None,
    fused_bytes=None,
):
    """Compatibility wrapper for clipping, tiled or fused updates and
    assembly.
    """
    builder = OptimizerPlanBuilder(
        tasks,
        shapes,
        tile_size,
        learning_rate,
        optimizer,
        step,
        options,
        fused_bytes,
    )
    return builder.build(parameters, gradients)


def decode_task(description: Mapping[str, Any]) -> Task:
    """Reconstruct the existing task description format."""
    return Task(
        description["id"],
        description["op"],
        [Input(**reference) for reference in description["inputs"]],
        tuple(description["shape"]),
        description["attrs"],
    )
