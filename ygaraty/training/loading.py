"""Explicitly selected local factories; never load code from round metadata."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path


def resolve_factory(value, base):
    module, separator, name = value.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("Factory must be MODULE:CALLABLE or FILE.py:CALLABLE")
    if module.endswith(".py"):
        path = (Path(base) / module).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        module = str(path)
    elif not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("Invalid factory module")
    return f"{module}:{name}"


def load_factory(value):
    module, _, name = resolve_factory(value, Path.cwd()).partition(":")
    if module.endswith(".py"):
        path = Path(module)
        key = "_ygaraty_user_" + hashlib.sha256(str(path).encode()).hexdigest()
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, path)
            loaded = importlib.util.module_from_spec(spec)
            # Keep sibling imports available to the factory and its callbacks.
            parent = str(path.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            sys.modules[key] = loaded
            try:
                spec.loader.exec_module(loaded)
            except BaseException:
                sys.modules.pop(key, None)
                raise
        loaded = sys.modules[key]
    else:
        loaded = importlib.import_module(module)
    factory = getattr(loaded, name)
    if not callable(factory):
        raise ValueError("Selected factory is not callable")
    return factory
