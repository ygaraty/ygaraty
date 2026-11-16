"""Local training adapters, independent of HTTP framing and coordination."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ygaraty.training.coordinator import Assignment, RoundResult


class RoundBackend(Protocol):
    """One persistent replica; rebasing must retain private optimizer state."""

    def prepare_batch(self, *arrays):
        """Convert decoded arrays to a backend-native batch."""
        ...

    def run_round(self, assignment: Assignment, batches) -> RoundResult:
        """Return named starting-minus-final weights for exactly H steps."""
        ...


@dataclass
class LocalBackend:
    """Adapt existing trainers without changing their numerical operations."""

    trainer: object
    prepare_batch: Callable
    iterator_batches: bool = False

    def run_round(self, assignment, batches):
        return self.trainer.run_round(
            assignment, iter(batches) if self.iterator_batches else batches
        )


def create_backend(worker, recipe, weights) -> RoundBackend:
    """Create a built-in adapter lazily; never import code named by a peer.

    Custom callers may inject their own factory into DiLoCoReplica. This
    default supports only the existing recipes, not arbitrary model loading.
    """
    kind = recipe.get("kind", "transformer")
    if kind == "convnet":
        from ygaraty.training.convnet_native import Replica
        from ygaraty.training.convnet_webgpu import ConvBatch

        trainer = Replica(worker, weights, recipe["learning_rate"])

        def prepare(images, labels):
            return ConvBatch(images.reshape(-1, 3, 32, 32), labels)

        return LocalBackend(trainer, prepare, iterator_batches=True)
    if kind not in ("mlp", "transformer"):
        raise ValueError("Unknown recipe kind")

    from ygaraty.training.trainer import Trainer

    if kind == "mlp":
        from ygaraty.training.mlp import ImageBatch, MLPRecipe

        model = MLPRecipe(
            recipe["config"],
            learning_rate=recipe["learning_rate"],
            tile_size=recipe["tile_size"],
        )
        batch_type = ImageBatch
    else:
        from ygaraty.training.transformer import Batch, TransformerRecipe

        model = TransformerRecipe(
            recipe["config"],
            learning_rate=recipe["learning_rate"],
            tile_size=recipe["tile_size"],
            smoothing=recipe["smoothing"],
        )
        batch_type = Batch
    return LocalBackend(Trainer(worker, model), batch_type)
