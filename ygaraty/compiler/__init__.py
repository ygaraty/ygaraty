"""Transport-independent tensor graph construction and lowering."""

from importlib import import_module

_MODULES = {
    "Tape": "tape",
    "ShapeTape": "shapes",
    "Parameters": "parameters",
    "Input": "tasks",
    "Task": "tasks",
    "with_optimizer": "optimizer",
    **dict.fromkeys(
        ("split", "group", "bounded_reductions", "admit_plan", "prune_unused"),
        "plan",
    ),
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
