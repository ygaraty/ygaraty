"""Portable three-layer ConvNet local-step recipe for DiLoCo CPU/WebGPU parity.

Images are NCHW uint8, normalized to [-1,1], matching the small ConvNet pilot.
This is a full-replica path: weights, Adam state and activations fit per
worker.
"""

import math
from dataclasses import dataclass

import numpy as np

from ygaraty.compiler.muon import append_muon, muon_options
from ygaraty.compiler.optimizer import adam_options, with_optimizer
from ygaraty.compiler.plan import admit_plan, split
from ygaraty.compiler.sequence import resolve_views
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.training.transformer import StepPlan


@dataclass(frozen=True)
class ConvBatch:
    images: np.ndarray
    labels: np.ndarray


class ConvNetRecipe:
    kind = "convnet"

    def __init__(
        self,
        config=None,
        *,
        learning_rate=0.001,
        optimizer=None,
        optimizer_kind="adam",
        muon_learning_rate=0.02,
        muon_momentum=0.95,
        muon_steps=5,
        tile_size=512,
        max_batch=16,
    ):
        if type(max_batch) is not int or not 1 <= max_batch <= 256:
            raise ValueError("max_batch must be between 1 and 256")
        self.max_batch = max_batch
        c = dict(
            in_channels=3,
            height=32,
            width=32,
            channels=[32, 64, 128],
            classes=10,
            residual=False,
        )
        if config:
            if set(config) - set(c):
                raise ValueError("Unknown ConvNet configuration")
            c.update(config)
        for name, low, high in [
            ("in_channels", 1, 3),
            ("height", 4, 32),
            ("width", 4, 32),
            ("classes", 2, 100),
        ]:
            if type(c[name]) is not int or not low <= c[name] <= high:
                raise ValueError("Invalid " + name)
        if c["height"] % 4 or c["width"] % 4:
            raise ValueError("Spatial dimensions must be divisible by four")
        if (
            not isinstance(c["channels"], (list, tuple))
            or len(c["channels"]) != 3
            or any(
                type(x) is not int or not 1 <= x <= 128 for x in c["channels"]
            )
        ):
            raise ValueError(
                "Expected three positive channel counts up to 128"
            )
        if type(c["residual"]) is not bool or (
            c["residual"] and c["channels"][1] != c["channels"][2]
        ):
            raise ValueError(
                "Residual block requires equal final channel counts"
            )
        if (
            type(learning_rate) not in (int, float)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Invalid learning rate")
        if type(tile_size) is not int or not 1 <= tile_size <= 1024:
            raise ValueError("Invalid optimizer tile size")
        self.config = dict(c, channels=list(c["channels"]))
        self.learning_rate = learning_rate
        self.optimizer = adam_options(optimizer or {})
        if optimizer_kind not in ("adam", "muon"):
            raise ValueError("Unknown local optimizer")
        if (
            type(muon_learning_rate) not in (int, float)
            or not math.isfinite(muon_learning_rate)
            or muon_learning_rate <= 0
        ):
            raise ValueError("Invalid Muon learning rate")
        self.optimizer_kind = optimizer_kind
        self.muon_learning_rate = muon_learning_rate
        self.muon_options = muon_options(muon_momentum, muon_steps)
        self.tile_size = tile_size
        self.specs = {}
        channels = c["in_channels"]
        for index, out in enumerate(c["channels"]):
            self.specs[f"conv{index}.weight"] = dict(
                shape=(out, channels * 9), bound=1 / math.sqrt(channels * 9)
            )
            self.specs[f"conv{index}.bias"] = dict(
                shape=(1, out), bound=1 / math.sqrt(channels * 9)
            )
            channels = out
        self.specs["head.weight"] = dict(
            shape=(channels, c["classes"]), bound=1 / math.sqrt(channels)
        )
        self.specs["head.bias"] = dict(
            shape=(1, c["classes"]), bound=1 / math.sqrt(channels)
        )

    def moment_names(self):
        return {
            "m": list(self.specs),
            "v": [
                n
                for n in self.specs
                if not (
                    self.optimizer_kind == "muon"
                    and n.startswith("conv")
                    and n.endswith(".weight")
                )
            ],
        }

    def initial_weights(self, seed=2026):
        rng = np.random.default_rng(seed)
        return {
            name: rng.uniform(-s["bound"], s["bound"], s["shape"])
            for name, s in self.specs.items()
        }

    def compile(self, batch, step):
        c = self.config
        images = np.asarray(batch.images)
        labels = np.asarray(batch.labels)
        if (
            images.ndim != 4
            or images.shape[1:] != (c["in_channels"], c["height"], c["width"])
            or not 1 <= len(images) <= self.max_batch
        ):
            raise ValueError(
                f"Expected 1..{self.max_batch} NCHW images "
                "with the configured shape"
            )
        if (
            images.dtype != np.uint8
            or labels.shape != (len(images),)
            or labels.dtype.kind not in "iu"
            or np.any(labels < 0)
            or np.any(labels >= c["classes"])
        ):
            raise ValueError("Expected uint8 pixels and valid integer labels")
        if type(step) is not int or step < 1:
            raise ValueError("Invalid optimizer step")
        n = len(images)
        tape = ShapeTape()
        x = tape.declare(
            "images",
            (n, c["in_channels"] * c["height"] * c["width"]),
            dict(kind="zeros"),
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
        channels, h, w = c["in_channels"], c["height"], c["width"]
        for index, out in enumerate(c["channels"]):
            skip = x
            x = tape.conv2d(
                x,
                parameters[f"conv{index}.weight"],
                parameters[f"conv{index}.bias"],
                channels=channels,
                height=h,
                width=w,
            )
            if index == 2 and c["residual"]:
                x = tape.add(x, skip)
            x = tape.relu(x)
            channels = out
            if index < 2:
                x = tape.pool2d(x, channels=channels, height=h, width=w)
                h, w = h // 2, w // 2
        x = tape.pool2d(x, channels=channels, height=h, width=w, average=True)
        logits = tape.add_bias(
            tape.matmul(x, parameters["head.weight"]), parameters["head.bias"]
        )
        tape.backward(tape.cross_entropy(logits, labels))
        muon = [
            p
            for n, p in parameters.items()
            if self.optimizer_kind == "muon"
            and n.startswith("conv")
            and n.endswith(".weight")
        ]
        aux = [p for p in tape.parameters if p not in muon]
        muon_updates = append_muon(
            tape, muon, self.muon_learning_rate, **self.muon_options
        )
        state_parameters = {"m": {}, "v": {}}
        for name, p in parameters.items():
            if p in muon:
                state_parameters["m"][name] = f"muon_m_{p}"
            else:
                for kind in ("m", "v"):
                    state = f"adam_{kind}_{p}"
                    tape.declare(state, tape.shapes[p], dict(kind="zeros"))
                    state_parameters[kind][name] = state
        tasks, updates = with_optimizer(
            split(tape, self.tile_size),
            tape.shapes,
            aux,
            tape.gradients,
            self.tile_size,
            self.learning_rate,
            "adam",
            step,
            self.optimizer,
        )
        updates.update(muon_updates)
        optimizer_nodes = {
            n.id for n in tape.nodes if n.attrs.get("optimizer")
        }
        for task in tasks:
            if task.id.split("/")[0] in optimizer_nodes or task.id.startswith(
                ("update/", "updated/")
            ):
                task.attrs["optimizer"] = True
        tasks = resolve_views(tape, tasks)
        constants = {
            k: np.asarray(v["data"], dtype=np.float64).reshape(v["shape"])
            for k, v in tape.values.items()
            if v.get("kind") == "constant"
        }
        constants["images"] = (images.astype(np.float64) / 127.5 - 1).reshape(
            n, -1
        )
        admit_plan(tasks, {k: tape.shapes[k] for k in tape.values}, 256 << 20)
        plan = StepPlan(tasks, constants, parameters, updates, tape.loss, step)
        plan.state_parameters = state_parameters
        plan.gradients = tape.gradients
        plan.logits = logits
        return plan

    def predict(self, weights, images):
        """Run just the compiled forward graph for held-out evaluation."""
        from ygaraty.inference.shared import ancestors
        from ygaraty.runtime.local import execute_plan

        plan = self.compile(
            ConvBatch(images, np.zeros(len(images), dtype=np.int64)), 1
        )
        needed = ancestors(plan.tasks, plan.logits)
        tasks = [task for task in plan.tasks if task.id in needed]
        inputs = dict(plan.constants)
        inputs.update({plan.parameters[n]: w for n, w in weights.items()})
        return execute_plan(tasks, inputs, [plan.logits])[plan.logits]

    def round_bundle(self, assignment, batches, *, start_step=0):
        plans = [
            self.compile(batch, start_step + i + 1).describe()
            for i, batch in enumerate(batches)
        ]
        if len(plans) != assignment.steps:
            raise ValueError("Expected one batch per local step")
        return dict(
            format="ygaraty-diloco-round-v1",
            assignment=dict(
                round_id=assignment.round_id,
                version=assignment.version,
                worker=assignment.worker,
                steps=assignment.steps,
                weights={
                    n: dict(shape=list(a.shape), data=a.ravel().tolist())
                    for n, a in assignment.weights.items()
                },
            ),
            plans=plans,
        )
