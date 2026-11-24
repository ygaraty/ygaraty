"""The existing translation Transformer, packaged as complete local steps."""

import math
from dataclasses import dataclass

import numpy as np

from ygaraty.compiler.optimizer import adam_options, with_optimizer
from ygaraty.compiler.plan import split
from ygaraty.compiler.sequence import resolve_views
from ygaraty.compiler.shapes import regions
from ygaraty.models.synthetic import initial_block
from ygaraty.models.transformer import build_transformer
from ygaraty.models.transformer_translation import (
    build_transformer_translation,
    parameter_specs,
)


@dataclass(frozen=True)
class Batch:
    source: np.ndarray
    target_input: np.ndarray
    targets: np.ndarray


@dataclass
class StepPlan:
    tasks: list
    constants: dict
    parameters: dict
    updates: dict
    loss: str
    step: int
    state_parameters: dict | None = None

    def describe(self):
        """Portable data-only plan; weights and moments are sent once per
        round."""
        return dict(
            tasks=[t.describe() for t in self.tasks],
            constants={
                k: dict(shape=list(a.shape), data=a.ravel().tolist())
                for k, a in self.constants.items()
            },
            parameters=self.parameters,
            updates=self.updates,
            loss=self.loss,
            step=self.step,
            **(
                {"state_parameters": self.state_parameters}
                if self.state_parameters is not None
                else {}
            ),
        )


class TransformerRecipe:
    """Same parameter names, masks, tied embedding and loss as SQL training.

    `portable=True` lowers to the primitive operations the browser SDK
    supports.
    Python may use the fused layer kernels instead. Both share optimizer
    lowering.
    Inner Adam intentionally matches Ygaraty's recipe (no weight
    decay/dropout),
    rather than changing the model to reproduce the paper's AdamW experiments.
    """

    def __init__(
        self,
        config,
        *,
        learning_rate=0.0005,
        optimizer=None,
        smoothing=0.1,
        tile_size=512,
        portable=False,
    ):
        self.config = dict(config)
        for name in (
            "vocab",
            "embedding",
            "hidden",
            "layers",
            "heads",
            "max_length",
        ):
            if type(config.get(name)) is not int or config[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            config["embedding"] % config["heads"]
            or config.get("dropout", 0.0) != 0.0
        ):
            raise ValueError("Require divisible heads and disabled dropout")
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(smoothing) or not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        if type(tile_size) is not int or tile_size < 1:
            raise ValueError("tile_size must be a positive integer")
        self.learning_rate, self.smoothing = learning_rate, smoothing
        self.optimizer = adam_options(
            dict({"beta1": 0.9, "beta2": 0.98}, **(optimizer or {}))
        )
        self.tile_size, self.portable = tile_size, portable
        self.specs = parameter_specs(self.config)

    def initial_weights(self, seed=2026):
        """Use the main backend's block initialization, including its tile
        layout."""
        weights = {}
        for name, spec in self.specs.items():
            array = np.empty(spec["shape"], dtype=np.float64)
            for region in regions(spec["shape"], self.tile_size):
                i, ei, j, ej = region
                array[i:ei, j:ej] = initial_block(
                    dict(seed=seed, tile_size=self.tile_size),
                    name.replace(".", "_"),
                    spec,
                    region,
                )
            weights[name] = array
        return weights

    def compile(self, batch, step):
        src, inp, labels = (
            np.asarray(a)
            for a in (batch.source, batch.target_input, batch.targets)
        )
        if (
            src.ndim != 2
            or inp.ndim != 2
            or inp.shape != labels.shape
            or not src.size
            or not inp.size
            or len(src) != len(inp)
        ):
            raise ValueError(
                "Expected nonempty padded source, input and target matrices"
            )
        for a in (src, inp, labels):
            if (
                a.dtype.kind not in "iu"
                or np.any(a < 0)
                or np.any(a >= self.config["vocab"])
            ):
                raise ValueError(
                    "Token IDs must be integers within the vocabulary"
                )
        if any(
            np.any(np.count_nonzero(a, axis=1) == 0)
            for a in (src, inp, labels)
        ):
            raise ValueError(
                "Every example needs source, input and target tokens"
            )
        if self.portable:
            builder = build_transformer
        else:
            builder = build_transformer_translation
        tape, _, parameters = builder(
            self.config,
            self.specs,
            src,
            inp,
            labels,
            self.smoothing,
            shape_only=True,
        )
        for p in tape.parameters:
            for kind in ("m", "v"):
                tape.declare(
                    f"adam_{kind}_{p}", tape.shapes[p], dict(kind="zeros")
                )
        tasks, updates = with_optimizer(
            split(tape, self.tile_size),
            tape.shapes,
            tape.parameters,
            tape.gradients,
            self.tile_size,
            self.learning_rate,
            "adam",
            step,
            self.optimizer,
        )
        # A single fused band exposes parameter gradients as views. Optimizer
        # tasks are appended after split(), so resolve their inputs as well.
        tasks = resolve_views(tape, tasks)
        state_ids = set(parameters.values()) | {
            f"adam_{k}_{p}" for p in parameters.values() for k in ("m", "v")
        }
        constants = {
            k: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"])
            for k, v in tape.values.items()
            if k not in state_ids
        }
        return StepPlan(tasks, constants, parameters, updates, tape.loss, step)

    def round_bundle(self, assignment, batches, *, start_step=0):
        """A portable round fixture for browser/Node execution, without any
        code.

        Model arrays occur once. Per-batch plans contain only constants and
        operation metadata. A future transport can stream these plans instead.
        """
        if not self.portable:
            raise ValueError("Browser rounds require portable=True")
        plans = [
            self.compile(batch, start_step + i + 1).describe()
            for i, batch in enumerate(batches)
        ]
        if len(plans) != assignment.steps:
            raise ValueError(
                "Expected exactly one batch per assigned local step"
            )
        return dict(
            format="ygaraty-diloco-round-v1",
            assignment=assignment.describe(),
            plans=plans,
        )
