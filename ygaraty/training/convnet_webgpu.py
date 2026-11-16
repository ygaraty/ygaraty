"""Portable three-layer ConvNet local-step recipe for DiLoCo CPU/WebGPU parity.

Images are NCHW uint8, normalized to [-1,1], matching the small ConvNet pilot.
This is a full-replica path: weights, Adam state and activations fit per
worker.
"""

import math
from dataclasses import dataclass

import numpy as np

from ygaraty.compiler.optimizer import adam_options, with_optimizer
from ygaraty.compiler.plan import Input, Task, admit_plan
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
        parameters = {name: name.replace(".", "_") for name in self.specs}
        shapes = {
            parameters[name]: s["shape"] for name, s in self.specs.items()
        }
        constants = {
            "images": (images.astype(np.float64) / 127.5 - 1).reshape(n, -1),
            "one": np.ones((1, 1)),
        }
        shapes.update({name: a.shape for name, a in constants.items()})
        tasks = []

        def add(name, op, refs, shape, attrs=None):
            tasks.append(
                Task(
                    name, op, [Input(ref) for ref in refs], shape, attrs or {}
                )
            )
            shapes[name] = shape
            return name

        x = "images"
        channels = c["in_channels"]
        h, w = c["height"], c["width"]
        history = []
        for index, out in enumerate(c["channels"]):
            attrs = dict(
                batch=n,
                in_channels=channels,
                height=h,
                width=w,
                out_channels=out,
                kernel=3,
            )
            weight, bias = (
                parameters[f"conv{index}.weight"],
                parameters[f"conv{index}.bias"],
            )
            pre = add(
                f"conv{index}",
                "conv2d",
                [x, weight, bias],
                (n, out * h * w),
                attrs,
            )
            active = add(f"relu{index}", "relu", [pre], shapes[pre])
            poolattrs = dict(batch=n, in_channels=out, height=h, width=w)
            history.append(
                (x, pre, active, attrs, poolattrs, weight, bias, index < 2)
            )
            x = (
                add(
                    f"pool{index}",
                    "maxpool2d",
                    [active],
                    (n, out * (h // 2) * (w // 2)),
                    poolattrs,
                )
                if index < 2
                else active
            )
            if index < 2:
                h //= 2
                w //= 2
            channels = out
        avgattrs = dict(batch=n, in_channels=channels, height=h, width=w)
        features = add("features", "avgpool2d", [x], (n, channels), avgattrs)
        head, bias = parameters["head.weight"], parameters["head.bias"]
        logits = add(
            "logits",
            "add_bias",
            [
                add("linear", "matmul", [features, head], (n, c["classes"])),
                bias,
            ],
            (n, c["classes"]),
        )
        ce = dict(targets=labels.astype(int).tolist(), count=n)
        loss = add("loss", "cross_entropy", [logits], (1, 1), ce)
        dy = add(
            "d_logits", "ce_backward", [logits, "one"], shapes[logits], ce
        )
        gradients = {}
        ft = add("features_t", "transpose", [features], (channels, n))
        gradients[head] = add("grad_head", "matmul", [ft, dy], shapes[head])
        gradients[bias] = add(
            "grad_head_bias", "reduce_rows", [dy], shapes[bias]
        )
        ht = add("head_t", "transpose", [head], (c["classes"], channels))
        df = add("d_features", "matmul", [dy, ht], shapes[features])
        dy = add("d_avg", "avgpool2d_grad", [df], shapes[x], avgattrs)
        for index in range(2, -1, -1):
            before, pre, active, attrs, poolattrs, weight, bias, pooled = (
                history[index]
            )
            if pooled:
                dy = add(
                    f"d_pool{index}",
                    "maxpool2d_grad",
                    [active, dy],
                    shapes[active],
                    poolattrs,
                )
            dy = add(f"d_relu{index}", "relu_backward", [pre, dy], shapes[pre])
            gradients[weight] = add(
                f"grad_conv{index}",
                "conv2d_weight_grad",
                [before, dy],
                shapes[weight],
                attrs,
            )
            gradients[bias] = add(
                f"grad_bias{index}",
                "conv2d_bias_grad",
                [dy],
                shapes[bias],
                attrs,
            )
            if index:
                dy = add(
                    f"d_conv{index}",
                    "conv2d_input_grad",
                    [dy, weight],
                    shapes[before],
                    attrs,
                )
        leaves = {
            name: shapes[name] for name in [*constants, *parameters.values()]
        }
        for p in parameters.values():
            for kind in ("m", "v"):
                shapes[f"adam_{kind}_{p}"] = shapes[p]
                leaves[f"adam_{kind}_{p}"] = shapes[p]
        tasks, updates = with_optimizer(
            tasks,
            shapes,
            list(parameters.values()),
            gradients,
            self.tile_size,
            self.learning_rate,
            "adam",
            step,
            self.optimizer,
        )
        admit_plan(tasks, leaves, 256 << 20)
        plan = StepPlan(tasks, constants, parameters, updates, loss, step)
        plan.gradients = gradients
        plan.logits = logits
        return plan

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
