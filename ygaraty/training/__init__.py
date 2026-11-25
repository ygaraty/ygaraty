"""DiLoCo backend: reusable local plans, replica state, and round
synchronization."""

from ygaraty.training.coordinator import (
    Assignment,
    Coordinator,
    DiLoCoConfig,
    RoundResult,
)
from ygaraty.training.trainer import Trainer
from ygaraty.training.transformer import Batch, TransformerRecipe

__all__ = [
    "Assignment",
    "Coordinator",
    "DiLoCoConfig",
    "RoundResult",
    "Trainer",
    "Batch",
    "TransformerRecipe",
]

from .convnet_webgpu import ConvBatch, ConvNetRecipe
from .mlp import ImageBatch, MLPRecipe
from .session import TrainingConfig, TrainingSession

__all__ += ["ImageBatch", "MLPRecipe", "ConvBatch", "ConvNetRecipe"]
__all__ += ["TrainingConfig", "TrainingSession"]
