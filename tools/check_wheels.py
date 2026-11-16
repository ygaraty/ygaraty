"""Verify the runtime wheel and its optional research companion."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

PUBLIC = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ygaraty-wheel-") as temp:
        root = Path(temp)
        for module, distribution, original in (
            ("ygaraty", "ygaraty", PUBLIC / "ygaraty"),
            (
                "ygaraty_webgpu_research",
                "ygaraty_webgpu_research",
                PUBLIC / "research/src/ygaraty_webgpu_research",
            ),
        ):
            wheels = list(args.wheel_dir.glob(f"{distribution}-*.whl"))
            if len(wheels) != 1:
                raise ValueError(f"Expected one {distribution} wheel")
            with ZipFile(wheels[0]) as wheel:
                if any(
                    name.startswith(("ygaraty_cloud/", "ygaraty_public/"))
                    for name in wheel.namelist()
                ):
                    raise ValueError("Unexpected package in wheel")
                wheel.extractall(root)
            for path in original.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts:
                    continue
                target = root / module / path.relative_to(original)
                if not target.is_file() or (
                    hashlib.sha256(path.read_bytes()).digest()
                    != hashlib.sha256(target.read_bytes()).digest()
                ):
                    raise ValueError(f"Missing or stale wheel asset: {path}")
        shutil.copytree(PUBLIC / "tests", root / "tests")
        shutil.copytree(PUBLIC / "research/tests", root / "tests/research")
        script = """
import importlib, importlib.abc, importlib.metadata
import json, pkgutil, sys, unittest
from pathlib import Path
sys.path[:0] = [sys.argv[1], *json.loads(sys.argv[2])]
class NoPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ygaraty_cloud', 'vllm', 'django'}:
            raise AssertionError('Private dependency: ' + fullname)
sys.meta_path.insert(0, NoPrivate())
package = importlib.import_module('ygaraty')
for module in pkgutil.walk_packages(package.__path__, 'ygaraty.'):
    if module.name.endswith('.__main__'):
        continue
    try:
        importlib.import_module(module.name)
    except ImportError as error:
        optional = ('.training.convnet', '.training.convnet_growth',
                    '.reference.transformer', '.reference.translator',
                    '.verification.lstm_translation_oracle')
        if not module.name.endswith(optional) or error.name != 'torch':
            raise
for name in ('convnet_webgpu', 'factorization_webgpu', 'factorization_memory',
             'factorization_browser_training', 'webgpu_parity',
             'qwen35_session', 'convnet_remote_scaling'):
    importlib.import_module('ygaraty_webgpu_research.' + name)
dist = importlib.metadata.distribution('ygaraty')
assert not any(req.startswith(('ygaraty-core', 'ygaraty-webgpu',
    'ygaraty-backend-contracts', 'ygaraty-inference-support'))
    for req in (dist.requires or []))
for entry in dist.entry_points:
    if entry.group == 'console_scripts':
        assert callable(entry.load()), entry
suite = unittest.TestSuite()
for suffix in ('webgpu', 'contracts', 'research', ''):
    directory = str(Path(sys.argv[1]) / 'tests' / suffix)
    suite.addTests(unittest.TestLoader().discover(
        directory, top_level_dir=directory))
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(not result.wasSuccessful())
"""
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                script,
                str(root),
                json.dumps(
                    [p for p in sys.path if Path(p).name == "site-packages"]
                ),
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [
                "node",
                "--test",
                *map(str, (root / "tests/webgpu/browser").glob("*.test.mjs")),
            ],
            cwd=root,
            check=True,
        )
    print("Self-contained wheels passed isolation and numerical checks")


if __name__ == "__main__":
    main()
