"""Public local execution for forward, backward and optimizer plans."""

from .kernels import evaluate, execute, execute_task
from .local import execute_plan

__all__ = ["evaluate", "execute", "execute_task", "execute_plan"]
