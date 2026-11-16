"""M-1 B numerical implementation; use on an authorized training
workstation."""

import torch
from torch import nn

from .convnet import Replica, SmallConvNet, rebase


class GrowingConvNet(SmallConvNet):
    def __init__(self, stage=0):
        if type(stage) is not int or not 0 <= stage <= 3:
            raise ValueError("Stage must be 0..3")
        # Same seed gives every arm the same head as the ordinary model.
        super().__init__()
        self.stage = stage
        for index, minimum in ((0, 3), (3, 2), (6, 1)):
            if stage < minimum:
                del self._modules[str(index)]

    def forward(self, x):
        if self.stage < 3:
            x = nn.functional.relu(x)
            for _ in range(1 if self.stage == 2 else 2):
                x = nn.functional.max_pool2d(x, 2)
            channels = (128, 64, 32)[self.stage]
            x = x[:, torch.arange(channels, device=x.device) % 3]
        for index in range((8, 6, 3, 0)[self.stage], 11):
            x = self._modules[str(index)](x)
        return x

    def grow(self):
        if self.stage == 3:
            raise ValueError("Already at full depth")
        index, inputs, outputs = ((6, 64, 128), (3, 32, 64), (0, 3, 32))[
            self.stage
        ]
        reference = next(self.parameters())
        layer = nn.Conv2d(inputs, outputs, 3, padding=1).to(reference)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
            for channel in range(outputs):
                layer.weight[channel, channel % 3, 1, 1] = 1
        self.add_module(str(index), layer)
        self.stage += 1
        return list(layer.parameters())


class GrowthReplica(Replica):
    def __init__(
        self, worker, weights, learning_rate, stage, dtype=torch.float64
    ):
        super().__init__(
            worker, weights, learning_rate, dtype, model=GrowingConvNet(stage)
        )

    def adopt_growth(self, weights, stage, version):
        if stage != self.model.stage + 1 or version <= self.version:
            raise ValueError("Expected one new stage and a newer version")
        rebase(
            self.model,
            {n: weights[n] for n, _ in self.model.named_parameters()},
        )
        new_parameters = self.model.grow()
        rebase(self.model, weights)
        # Retain old Parameter objects/moments. Adam lazily zero-initializes
        # new moments.
        self.optimizer.add_param_group({"params": new_parameters})
        self.version = version
