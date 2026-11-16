"""Export a tiny synthetic DiLoCo/WebGPU numerical test, without starting
services.

PyTorch supplies independent forward/backward and one-Adam-update references.
The exported browser page runs only when explicitly opened on an execution
host.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.host import component as _host_component

PROJECT_ROOT = _host_component("paths").PROJECT_ROOT
from ygaraty_webgpu_research.host import component as _host_component

ConvBatch = _host_component("diloco.convnet_webgpu").ConvBatch
ConvNetRecipe = _host_component("diloco.convnet_webgpu").ConvNetRecipe
from ygaraty_webgpu_research.host import component as _host_component

Coordinator = _host_component("diloco.coordinator").Coordinator
DiLoCoConfig = _host_component("diloco.coordinator").DiLoCoConfig
RoundResult = _host_component("diloco.coordinator").RoundResult


def pack(a):
    a = np.asarray(a, dtype=np.float64)
    return dict(shape=list(a.shape), data=a.ravel().tolist())


def torch_step(recipe, weights, batch):
    """One small independent autograd/Adam step; no training loop or
    dataset."""
    import torch
    from torch.nn import functional as F

    c = recipe.config
    parameters = {
        name: torch.tensor(a, dtype=torch.float64, requires_grad=True)
        for name, a in weights.items()
    }
    optimizer = torch.optim.Adam(
        list(parameters.values()),
        lr=recipe.learning_rate,
        betas=(recipe.optimizer["beta1"], recipe.optimizer["beta2"]),
        eps=recipe.optimizer["epsilon"],
    )
    x = torch.tensor(
        batch.images.astype(np.float64) / 127.5 - 1, dtype=torch.float64
    )
    channels = c["in_channels"]
    for index, out in enumerate(c["channels"]):
        x = F.relu(
            F.conv2d(
                x,
                parameters[f"conv{index}.weight"].reshape(out, channels, 3, 3),
                parameters[f"conv{index}.bias"].reshape(out),
                padding=1,
            )
        )
        if index < 2:
            x = F.max_pool2d(x, 2)
        channels = out
    logits = (
        x.mean(dim=(2, 3)) @ parameters["head.weight"]
        + parameters["head.bias"]
    )
    loss = F.cross_entropy(
        logits, torch.tensor(batch.labels, dtype=torch.int64)
    )
    loss.backward()
    gradients = {
        name: p.grad.detach().numpy().copy() for name, p in parameters.items()
    }
    if recipe.optimizer["clip_norm"]:
        raise ValueError("Fixture oracle does not enable gradient clipping")
    optimizer.step()
    after = {name: p.detach().numpy().copy() for name, p in parameters.items()}
    return dict(
        loss=float(loss.detach()),
        logits=logits.detach().numpy().copy(),
        gradients=gradients,
        weights=after,
    )


def kernel_cases():
    import torch
    from torch.nn import functional as F

    from ygaraty.compiler.convolution import contract

    rng = np.random.default_rng(31)
    cases = []
    for k in (1, 3):
        attrs = dict(
            batch=2, in_channels=2, height=4, width=6, out_channels=3, kernel=k
        )
        x = torch.tensor(
            rng.normal(size=(2, 2, 4, 6)),
            dtype=torch.float64,
            requires_grad=True,
        )
        w = torch.tensor(
            rng.normal(size=(3, 2, k, k)),
            dtype=torch.float64,
            requires_grad=True,
        )
        bias = torch.tensor(
            rng.normal(size=3), dtype=torch.float64, requires_grad=True
        )
        y = F.conv2d(x, w, bias, padding=k // 2)
        dy = torch.tensor(rng.normal(size=tuple(y.shape)), dtype=torch.float64)
        y.backward(dy)
        for op, inputs, expected in [
            ("conv2d", [x, w, bias], y),
            ("conv2d_input_grad", [dy, w], x.grad),
            ("conv2d_weight_grad", [x, dy], w.grad),
            ("conv2d_bias_grad", [dy], bias.grad),
        ]:
            shapes, out, _ = contract(op, attrs)
            cases.append(
                dict(
                    op=op,
                    attrs=attrs,
                    shape=list(out),
                    inputs=[
                        pack(a.detach().numpy().reshape(s))
                        for a, s in zip(inputs, shapes)
                    ],
                    expected=pack(expected.detach().numpy().reshape(out)),
                )
            )
    attrs = dict(batch=2, in_channels=2, height=4, width=6)
    for tied in (False, True):
        x = torch.tensor(
            np.ones((2, 2, 4, 6)) if tied else rng.normal(size=(2, 2, 4, 6)),
            dtype=torch.float64,
            requires_grad=True,
        )
        for forward, backward, operation in [
            ("maxpool2d", "maxpool2d_grad", lambda a: F.max_pool2d(a, 2)),
            ("avgpool2d", "avgpool2d_grad", lambda a: a.mean(dim=(2, 3))),
        ]:
            x.grad = None
            y = operation(x)
            dy = torch.tensor(
                rng.normal(size=tuple(y.shape)), dtype=torch.float64
            )
            y.backward(dy)
            for op, inputs, expected in [
                (forward, [x], y),
                (
                    backward,
                    [x, dy] if forward == "maxpool2d" else [dy],
                    x.grad,
                ),
            ]:
                shapes, out, _ = contract(op, attrs)
                cases.append(
                    dict(
                        op=op,
                        attrs=attrs,
                        shape=list(out),
                        inputs=[
                            pack(a.detach().numpy().reshape(s))
                            for a, s in zip(inputs, shapes)
                        ],
                        expected=pack(expected.detach().numpy().reshape(out)),
                    )
                )
    return cases


def make_fixture():
    import torch

    recipe = ConvNetRecipe(
        dict(in_channels=1, height=8, width=8, channels=[2, 3, 4], classes=3),
        tile_size=64,
    )
    weights = recipe.initial_weights(37)
    config = DiLoCoConfig(
        local_steps=1, outer_learning_rate=0.7, momentum=0.9, nesterov=True
    )
    coordinator = Coordinator(weights, config)
    coordinator.begin_round(["w0", "w1"])
    rng = np.random.default_rng(41)
    rounds = []
    for name in ("w0", "w1"):
        batch = ConvBatch(
            rng.integers(0, 256, (2, 1, 8, 8), dtype=np.uint8),
            np.array([0, 2]),
        )
        assignment = coordinator.assignment(name)
        reference = torch_step(recipe, weights, batch)
        pseudo = {n: weights[n] - reference["weights"][n] for n in weights}
        rounds.append(
            dict(
                bundle=recipe.round_bundle(assignment, [batch]),
                expected=dict(
                    losses=[reference["loss"]],
                    pseudo_gradient={n: pack(a) for n, a in pseudo.items()},
                ),
            )
        )
        coordinator.submit(
            RoundResult(
                assignment.round_id,
                assignment.version,
                name,
                1,
                pseudo,
                (reference["loss"],),
            )
        )
    coordinator.commit_round()
    return dict(
        format="ygaraty-convnet-webgpu-v1",
        kind="synthetic_numerical_fixture",
        torch_version=torch.__version__,
        config=recipe.config,
        parameters=sum(a.size for a in weights.values()),
        kernels=kernel_cases(),
        replicas=rounds,
        outer=dict(learning_rate=0.7, momentum=0.9, nesterov=True),
        expected_global={n: pack(a) for n, a in coordinator.weights.items()},
        tolerances=dict(
            float64=dict(rtol=1e-10, atol=1e-12),
            webgpu=dict(rtol=0.003, atol=2e-5),
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory")
    fixture = make_fixture()
    root = _host_component("browser_assets").BROWSER_ROOT
    args.output.mkdir(parents=True)
    from ygaraty_webgpu_research.host import component as _host_component

    SDK_ASSETS = _host_component("browser_assets").SDK_ASSETS

    files = [
        "contributor.css",
        "sdk.mjs",
        "budget.mjs",
        "convolution.mjs",
        "diloco.mjs",
        "convnet_test.mjs",
        "convnet_test.html",
    ]
    files += list(SDK_ASSETS)
    for name in files:
        shutil.copyfile(root / name, args.output / name)
    (args.output / "fixture.json").write_text(
        json.dumps(fixture, allow_nan=False) + "\n"
    )
    files.append("fixture.json")
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                n: hashlib.sha256((args.output / n).read_bytes()).hexdigest()
                for n in files
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output / "convnet_test.html")


if __name__ == "__main__":
    main()
