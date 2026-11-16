"""Safe local artifact lookup shared by inference backends."""
from pathlib import Path


def local_file(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Model artifact paths must be relative to their bundle')
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Model bundle file is missing or outside the model directory')
    return path
