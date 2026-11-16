"""Optional historical Torch ConvNet reference for experiment comparisons.

Distributed replicas use convnet_native and Ygaraty's portable graph. Keep this
implementation for explicit verification of weights, gradients and learning.
"""

import numpy as np
import torch
from torch import nn

from ygaraty.training.coordinator import RoundResult


class SmallConvNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )


def pack(model):
    # The coordinator's transport-neutral contract is named rank-two float64
    # arrays.
    return {
        n: p.detach().cpu().numpy().astype(np.float64).reshape(1, -1).copy()
        for n, p in model.named_parameters()
    }


def rebase(model, weights):
    if set(weights) != {n for n, _ in model.named_parameters()}:
        raise ValueError("Parameter names do not match")
    with torch.no_grad():
        for name, p in model.named_parameters():
            a = weights[name]
            if a.shape != (1, p.numel()) or not np.isfinite(a).all():
                raise ValueError(f"Invalid parameter: {name}")
            p.copy_(torch.from_numpy(a.reshape(tuple(p.shape))))


class Replica:
    def __init__(
        self,
        worker,
        weights,
        learning_rate,
        dtype=torch.float64,
        *,
        model=None,
    ):
        self.worker = worker
        self.model = (SmallConvNet() if model is None else model).to(
            dtype=dtype
        )
        rebase(self.model, weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.seen = set()
        self.version = -1

    def run_round(self, assignment, batches):
        if (
            assignment.worker != self.worker
            or assignment.round_id in self.seen
            or assignment.version < self.version
        ):
            raise ValueError("Foreign, stale or duplicate assignment")
        rebase(
            self.model, assignment.weights
        )  # Preserve private Adam moments.
        losses = []
        self.model.train()
        for _ in range(assignment.steps):
            x, y = next(batches)
            self.optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
        after = pack(self.model)
        self.seen.add(assignment.round_id)
        self.version = assignment.version
        return RoundResult(
            assignment.round_id,
            assignment.version,
            self.worker,
            assignment.steps,
            {n: assignment.weights[n] - after[n] for n in after},
            tuple(losses),
        )


def tensors(x, y, dtype=torch.float64):
    return torch.from_numpy(x.copy()).to(dtype=dtype).div_(255).sub_(0.5).div_(
        0.5
    ), torch.from_numpy(y.astype(np.int64))
