"""Ygaraty's public compiler, training and execution API."""

from importlib import import_module

_EXPORTS = {
    "Tape": "ygaraty.compiler",
    "ShapeTape": "ygaraty.compiler",
    "Parameters": "ygaraty.compiler",
    "Trainer": "ygaraty.training",
    "Coordinator": "ygaraty.training",
    "DiLoCoConfig": "ygaraty.training",
    "WebGPUFactory": "ygaraty.backends.webgpu",
    "BundleBuilder": "ygaraty.backends.webgpu.bundles",
    "Checkpoint": "ygaraty.backends.webgpu.bundles",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
