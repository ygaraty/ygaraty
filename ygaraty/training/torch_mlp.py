"""CIFAR example configuration for the generic PyTorch DiLoCo adapter."""

import numpy as np

from ygaraty.training.coordinator import copy_tensors
from ygaraty.training.mlp import normalize
from ygaraty.training.torch_backend import TorchBackend


class TorchMLPBackend(TorchBackend):
    """Compatibility factory for the existing CIFAR MLP CLI example."""

    def __init__(self, worker, recipe, weights):
        import torch

        if recipe.get("kind") != "mlp" or recipe["config"]["inputs"] != 3072:
            raise ValueError("Expected the CIFAR MLP recipe")
        config = recipe["config"]
        sizes = [3072, *config["hidden"], config["classes"]]
        shapes = {}
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            shapes[f"layer{i}.weight"] = (a, b)
            shapes[f"layer{i}.bias"] = (1, b)
        weights = copy_tensors(weights)
        if weights.keys() != shapes.keys() or any(
            w.shape != shapes[n] for n, w in weights.items()
        ):
            raise ValueError("MLP parameter layout mismatch")
        parameters = {
            n: torch.nn.Parameter(torch.from_numpy(w.copy()))
            for n, w in weights.items()
        }

        def model(x):
            for i in range(len(sizes) - 1):
                x = x @ parameters[f"layer{i}.weight"]
                x = x + parameters[f"layer{i}.bias"]
                if i < len(sizes) - 2:
                    x = torch.relu(x)
            return x

        def prepare(images, labels):
            labels = np.asarray(labels)
            if (
                labels.ndim != 1
                or not len(labels)
                or images.shape != (len(labels), 3072)
                or labels.dtype.kind not in "iu"
                or np.any(labels < 0)
                or np.any(labels >= sizes[-1])
                or not np.isfinite(images).all()
                or np.any(images < 0)
                or np.any(images > 255)
            ):
                raise ValueError("Invalid CIFAR batch")
            return (
                torch.from_numpy(normalize(images)),
                torch.from_numpy(labels.astype(np.int64)),
            )

        optimizer = torch.optim.Adam(
            list(parameters.values()),
            lr=recipe["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        super().__init__(
            worker,
            model,
            optimizer,
            lambda model, batch: torch.nn.functional.cross_entropy(
                model(batch[0]), batch[1]
            ),
            prepare_batch=prepare,
            parameters=parameters,
            wire_shapes=shapes,
        )


def create_pytorch_backend(worker, recipe, weights):
    return TorchMLPBackend(worker, recipe, weights)
