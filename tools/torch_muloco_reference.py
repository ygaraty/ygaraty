"""Benchmark pinned upstream MuLoCo-1 on Ygaraty's exact residual ConvNet.

Requires the optional training-torch dependency. Upstream code is supplied as
an explicit, hash-verified file, never imported into the production runtime.
Only one worker is supported by this upstream reference. The ygaraty-math arm
uses the same outer wrapper with our inner equations to isolate engine effects.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ygaraty.training.convnet_webgpu import ConvBatch
from ygaraty.training.coordinator import Coordinator
from ygaraty.training.datasets import load_dataset
from ygaraty.training.session import TrainingConfig, implementation_hash
from ygaraty.training.trainer import Trainer

UPSTREAM_COMMIT = "33cc33315eacd1bb5f179d7d5d9f85e012fe58f2"
UPSTREAM_SHA256 = (
    "3f78ddb772eaa06be99d219a8b93945edd2aa8ed61cf51b41113ae42f58ebed7"
)


class ConvNet(nn.Module):
    """Keep parameters in the same matrix layout used by our Tape."""

    def __init__(self, weights, device, dtype):
        super().__init__()
        self.values = nn.ParameterDict(
            {
                name.replace(".", "_"): nn.Parameter(
                    torch.tensor(w, dtype=dtype, device=device)
                )
                for name, w in weights.items()
            }
        )

    def named_weights(self):
        return [(n.replace("_", "."), p) for n, p in self.values.items()]

    def forward(self, x):
        for i in range(3):
            skip = x
            w = self.values[f"conv{i}_weight"]
            b = self.values[f"conv{i}_bias"]
            x = F.conv2d(
                x,
                w.reshape(w.shape[0], x.shape[1], 3, 3),
                b.flatten(),
                padding=1,
            )
            if i == 2:
                x = x + skip
            x = F.relu(x)
            if i < 2:
                x = F.max_pool2d(x, 2)
        x = x.mean(dim=(2, 3))
        return x @ self.values["head_weight"] + self.values["head_bias"]


class TapeMathMuon(torch.optim.Optimizer):
    """Independent Torch rendering of the current Tape optimizer equations."""

    def __init__(self, model):
        self.named_weights = model.named_weights()
        self.steps = 0
        super().__init__([p for _, p in self.named_weights], {})

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1
        for name, p in self.named_weights:
            g = p.grad
            state = self.state[p]
            if not state:
                state["m"] = torch.zeros_like(p)
                if not name.startswith("conv") or not name.endswith("weight"):
                    state["v"] = torch.zeros_like(p)
            if "v" not in state:
                m = state["m"] = 0.95 * state["m"] + 0.05 * g
                x = 0.95 * m + 0.05 * g
                x = x / (x.norm() + 1e-7)
                tall = x.shape[0] > x.shape[1]
                if tall:
                    x = x.T
                for _ in range(5):
                    a = x @ x.T
                    b = -4.7750 * a + 2.0315 * (a @ a)
                    x = 3.4445 * x + b @ x
                if tall:
                    x = x.T
                p.add_(
                    x, alpha=-0.02 * math.sqrt(max(1, p.shape[0] / p.shape[1]))
                )
            else:
                m = state["m"] = 0.9 * state["m"] + 0.1 * g
                v = state["v"] = 0.999 * state["v"] + 0.001 * g.square()
                p.addcdiv_(
                    m / (1 - 0.9**self.steps),
                    (v / (1 - 0.999**self.steps)).sqrt() + 1e-8,
                    value=-0.001,
                )


def load_upstream(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Upstream source does not match the pinned revision")
    spec = importlib.util.spec_from_file_location("muloco_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def optimizer(module, model, arm, local_steps=8):
    matrix, auxiliary = [], []
    for name, p in model.named_weights():
        (
            matrix
            if name.startswith("conv") and name.endswith("weight")
            else auxiliary
        ).append(p)
    groups = [
        dict(params=matrix, algorithm="muon", lr=0.02),
        dict(params=auxiliary, algorithm="adamw", lr=0.001),
    ]
    opt = module.MuLoCo1(
        groups,
        inner_lr=0.02,
        outer_lr=0.7,
        outer_momentum=0.9,
        sync_interval=local_steps,
        mu=0.95,
        betas=(0.9, 0.999),
        weight_decay=0,
        nesterov=False,
    )
    if arm == "ygaraty-math":
        opt.inner_optimizer = TapeMathMuon(model)
    return opt


def tensors(batch, device, dtype):
    # Normalize in float64 then cast, matching the Tape input constants.
    x = torch.tensor(
        batch.images.astype(np.float64) / 127.5 - 1, dtype=dtype, device=device
    )
    y = torch.tensor(batch.labels.astype(np.int64), device=device)
    return x, y


def verify(module, device):
    """Check architecture, backward/inner updates and two outer commits."""
    cfg = TrainingConfig(
        1,
        channels=(8, 16, 16),
        batch_size=16,
        local_steps=2,
        rounds=2,
        optimizer="muon",
    )
    recipe = cfg.recipe()
    weights = recipe.initial_weights(cfg.seed)
    rng = np.random.default_rng(41)
    batches = [
        ConvBatch(
            rng.integers(0, 256, (16, 1, 28, 28), dtype=np.uint8),
            rng.integers(0, 10, 16, dtype=np.int64),
        )
        for _ in range(4)
    ]
    dtype = torch.float64 if device == "cpu" else torch.float32
    model = ConvNet(weights, device, dtype)
    opt = optimizer(module, model, "ygaraty-math", 2)
    x, _ = tensors(batches[0], device, dtype)
    logits_error = float(
        np.max(
            np.abs(
                model(x).detach().cpu().numpy()
                - recipe.predict(weights, batches[0].images)
            )
        )
    )
    coordinator = Coordinator(weights, cfg.outer())
    trainer = Trainer("reference", recipe)
    errors = []
    for r in range(2):
        coordinator.begin_round(["reference"])
        batch_slice = batches[r * 2 : r * 2 + 2]
        result = trainer.run_round(
            coordinator.assignment("reference"), batch_slice
        )
        coordinator.submit(result)
        coordinator.commit_round()
        for batch in batch_slice:
            x, y = tensors(batch, device, dtype)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        expected = coordinator.weights
        errors.append(
            max(
                float(
                    np.max(np.abs(p.detach().cpu().numpy() - expected[name]))
                )
                for name, p in model.named_weights()
            )
        )
    tolerance = 1e-8 if device == "cpu" else 5e-5
    if max(logits_error, *errors) > tolerance:
        raise AssertionError(
            dict(
                logits_error=logits_error,
                weight_errors=errors,
                tolerance=tolerance,
            )
        )
    return dict(
        logits_error=logits_error,
        weight_errors=errors,
        tolerance=tolerance,
        device=device,
        passed=True,
    )


def evaluate(model, dataset, device, dtype, count=1024):
    loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, count, 16):
            batch = ConvBatch(
                dataset.test_images[start : start + 16],
                dataset.test_labels[start : start + 16],
            )
            x, y = tensors(batch, device, dtype)
            logits = model(x)
            loss += F.cross_entropy(logits.double(), y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
    return dict(
        split="test",
        examples=count,
        loss=loss / count,
        accuracy=correct / count,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument(
        "--arm", choices=("ygaraty-math", "upstream"), default="upstream"
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--samples", type=int, default=15360)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    args = p.parse_args()
    if args.samples < 1536 or args.samples % 1536:
        p.error("samples must be a positive multiple of 1536")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    module = load_upstream(args.upstream_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        result = verify(module, args.device)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    dataset = load_dataset("fashion-mnist", args.data_dir)
    cfg = TrainingConfig(
        1,
        optimizer="muon",
        batch_size=16,
        local_steps=8,
        rounds=args.samples // 128,
        seed=args.seed,
        eval_examples=1024,
        eval_interval=12,
    )
    start = time.monotonic()
    model = ConvNet(
        cfg.recipe().initial_weights(args.seed), args.device, torch.float32
    )
    opt = optimizer(module, model, args.arm)
    initial = evaluate(model, dataset, args.device, torch.float32)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    setup_seconds = time.monotonic() - start
    timed = time.monotonic()
    curve = [dict(samples=0, seconds=0.0, evaluation=initial)]
    for step, batch in enumerate(
        dataset.batches(0, 1, 0, args.samples // 16, 16, args.seed), start=1
    ):
        if time.monotonic() - start > args.timeout:
            raise TimeoutError("Reference trial exceeded its deadline")
        x, y = tensors(batch, args.device, torch.float32)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        opt.step()
        if step % 96 == 0:
            evaluation = evaluate(model, dataset, args.device, torch.float32)
            point = dict(
                samples=step * 16,
                seconds=time.monotonic() - timed,
                evaluation=evaluation,
            )
            curve.append(point)
            print(json.dumps(point), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    report = dict(
        kind="torch-muloco-reference",
        arm=args.arm,
        seed=args.seed,
        samples=args.samples,
        workers=1,
        seconds=time.monotonic() - timed,
        setup_seconds=setup_seconds,
        seconds_with_setup=time.monotonic() - start,
        curve=curve,
        dataset=dict(name=dataset.name, sha256=dataset.sha256),
        source_sha256=implementation_hash(),
        torch_version=torch.__version__,
        upstream_commit=UPSTREAM_COMMIT,
        upstream_sha256=UPSTREAM_SHA256,
        device=args.device,
        dtype="float32",
        arithmetic_note=(
            "Torch uses FP32 tensors on CUDA; resident Ygaraty uses FP32 "
            "convolutions and compensated FP32 pairs for other GPU ops. "
            "Outer updates use the host. Control is not bitwise exact."
        ),
        tf32=False,
        optimizer=dict(
            inner_lr=0.02,
            auxiliary_lr=0.001,
            outer_lr=0.7,
            outer_momentum=0.9,
            local_steps=8,
            weight_decay=0,
            auxiliary_betas=[0.9, 0.999],
        ),
        differences=(
            []
            if args.arm == "ygaraty-math"
            else [
                "upstream BF16 Newton-Schulz with per-iteration coefficients",
                "upstream spectral scaling sqrt(rows/cols), no clamp",
                "upstream non-Nesterov inner momentum",
            ]
        ),
        timing=("training, batch preparation/transfers and evaluation; "
                "separate setup timing"),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["arm", "seed", "seconds"]}))


if __name__ == "__main__":
    main()
