"""A multi-layer perceptron image classifier (CIFAR-10), packaged as complete
local steps.

Pixels arrive as integers 0..255 in CIFAR's layout (1,024 red, then green, then
blue values per
image); the recipe normalizes them per channel. Layers are x @ W + b with ReLU
between them, the
loss is mean cross-entropy over the batch, and inner Adam is the same lowering
as the Transformer's.
"""

import math
from dataclasses import dataclass

import numpy as np

from ygaraty.compiler.optimizer import adam_options, with_optimizer
from ygaraty.compiler.plan import split
from ygaraty.compiler.sequence import resolve_views
from ygaraty.compiler.shapes import ShapeTape, regions
from ygaraty.models.synthetic import initial_block
from ygaraty.training.transformer import StepPlan

MEAN = np.array([0.4914, 0.4822, 0.4465])
STD = np.array([0.2470, 0.2435, 0.2616])


@dataclass(frozen=True)
class ImageBatch:
    images: np.ndarray  # (n, 3072) pixel values 0..255
    labels: np.ndarray  # (n,) classes


def normalize(images):
    """Integer pixels in CIFAR layout to per-channel standardized float64
    rows."""
    x = (
        np.asarray(images, dtype=np.float64).reshape(len(images), 3, -1)
        / 255.0
    )
    return ((x - MEAN[:, None]) / STD[:, None]).reshape(len(images), -1)


class MLPRecipe:
    """`config`: inputs, hidden (a list of widths) and classes. PyTorch
    Linear's default
    initialization bounds (1/sqrt(fan_in)), drawn by the main backend's block
    initializer."""

    kind = "mlp"

    def __init__(
        self, config, *, learning_rate=0.001, optimizer=None, tile_size=512
    ):
        self.config = dict(config, hidden=list(config["hidden"]))
        sizes = [
            self.config["inputs"],
            *self.config["hidden"],
            self.config["classes"],
        ]
        if any(type(s) is not int or s < 1 for s in sizes):
            raise ValueError(
                "inputs, hidden widths and classes must be positive integers"
            )
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate, self.tile_size = learning_rate, tile_size
        self.optimizer = adam_options(optimizer or {})
        self.specs = {}
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            bound = 1 / math.sqrt(a)
            self.specs[f"layer{i}.weight"] = dict(
                shape=(a, b), kind="uniform", bound=bound
            )
            self.specs[f"layer{i}.bias"] = dict(
                shape=(1, b), kind="uniform", bound=bound
            )

    def initial_weights(self, seed=2026):
        weights = {}
        for name, spec in self.specs.items():
            array = np.empty(spec["shape"], dtype=np.float64)
            for i, ei, j, ej in regions(spec["shape"], self.tile_size):
                array[i:ei, j:ej] = initial_block(
                    dict(seed=seed, tile_size=self.tile_size),
                    name.replace(".", "_"),
                    spec,
                    (i, ei, j, ej),
                )
            weights[name] = array
        return weights

    def compile(self, batch, step):
        labels = np.asarray(batch.labels).ravel()
        if len(batch.images) != len(labels) or not len(labels):
            raise ValueError("Expected one label per image")
        if (
            labels.dtype.kind not in "iu"
            or np.any(labels < 0)
            or np.any(labels >= self.config["classes"])
        ):
            raise ValueError("Labels must be integer class indices")
        images = np.asarray(batch.images)
        if (
            images.shape != (len(labels), self.config["inputs"])
            or not np.isfinite(images).all()
            or np.any(images < 0)
            or np.any(images > 255)
        ):
            raise ValueError("Invalid pixel matrix")
        tape = ShapeTape()
        x = tape.declare(
            "x", (len(labels), self.config["inputs"]), dict(kind="zeros")
        )
        parameters = {
            name: tape.declare(
                name.replace(".", "_"),
                spec["shape"],
                dict(kind="zeros"),
                parameter=True,
            )
            for name, spec in self.specs.items()
        }
        layers = len(self.config["hidden"]) + 1
        for i in range(layers):
            x = tape.add_bias(
                tape.matmul(x, parameters[f"layer{i}.weight"]),
                parameters[f"layer{i}.bias"],
            )
            if i < layers - 1:
                x = tape.relu(x)
        tape.backward(tape.cross_entropy(x, labels.astype(np.int64)))
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
        # The tape's own constants (the loss gradient's seed), and the images.
        constants = {
            k: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"])
            for k, v in tape.values.items()
            if v.get("kind") == "constant"
        }
        constants["x"] = normalize(batch.images)
        return StepPlan(
            resolve_views(tape, tasks),
            constants,
            parameters,
            updates,
            tape.loss,
            step,
        )

    def predict(self, weights, images):
        """Logits for a batch of images, in NumPy (evaluation only)."""
        x = normalize(images)
        layers = len(self.config["hidden"]) + 1
        for i in range(layers):
            x = x @ weights[f"layer{i}.weight"] + weights[f"layer{i}.bias"]
            if i < layers - 1:
                x = np.maximum(x, 0)
        return x
