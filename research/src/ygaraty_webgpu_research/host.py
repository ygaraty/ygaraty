"""Resolve research components exclusively from public implementations."""

import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace


def component(name: str):
    if name == "paths":
        return SimpleNamespace(
            PROJECT_ROOT=Path(
                os.environ.get("YGARATY_RESEARCH_ROOT", ".")
            ).resolve()
        )
    modules = {
        "browser_assets": "ygaraty.backends.webgpu.browser_assets",
        "coordinator.qwen_protocol": "ygaraty.backends.webgpu._legacy.qwen_protocol",
        "experiments.factorization_http": "ygaraty.training.factorization",
    }
    if name in modules:
        return import_module(modules[name])
    for prefix, target in (
        ("diloco", "ygaraty.training"),
        ("runtime", "ygaraty.runtime"),
        ("compiler", "ygaraty.compiler"),
        ("parity", "ygaraty.parity"),
    ):
        if name == prefix or name.startswith(prefix + "."):
            module = import_module(target + name[len(prefix) :])
            if name == "runtime":
                import_module("ygaraty.runtime.capture")
            return module
    raise ValueError(f"Unknown public research component: {name!r}")
