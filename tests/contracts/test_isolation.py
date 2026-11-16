"""Import the complete public package without site-packages or the monorepo."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class IsolationTests(unittest.TestCase):
    def test_all_modules_import_without_private_or_optional_dependencies(self):
        source = Path(__file__).resolve().parents[2]
        program = """
import importlib
import importlib.abc
import pkgutil
import sys

class RejectPrivateImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'ygaraty.compiler', 'ygaraty.backends.webgpu'} or fullname.split('.')[0] in {
            'ygaraty_cloud', 'ygaraty.backends.webgpu', 'ygaraty.compiler', 'ygaraty_native', 'django', 'fastapi', 'vllm',
            'numpy', 'coopy',
        }:
            raise AssertionError('Forbidden dependency: ' + fullname)

sys.meta_path.insert(0, RejectPrivateImports())
sys.path.insert(0, sys.argv[1])
import ygaraty.backend_contracts
for module in pkgutil.walk_packages(
    ygaraty.backend_contracts.__path__, 'ygaraty.backend_contracts.'
):
    importlib.import_module(module.name)
assert ygaraty.backend_contracts.CONTRACT_MAJOR == 1
assert ygaraty.__file__.endswith("ygaraty/__init__.py")
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", program, str(source)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
