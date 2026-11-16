"""Core imports and numerical behavior without any inference backend."""
import subprocess
import sys
from pathlib import Path


def test_core_alone_supports_graphs_and_reference_kernels():
    source = Path(__file__).resolve().parents[2]
    code = '''import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'ygaraty.backends.webgpu', 'ygaraty_cloud', 'ygaraty_webgpu', 'ygaraty_core'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectBackends())
import numpy as np
import ygaraty
from ygaraty.compiler import Tape
from ygaraty.runtime.kernels import execute
assert ygaraty.__file__.endswith("ygaraty/__init__.py")
tape = Tape()
a = tape.input('a', np.array([[1., 2.]], dtype=np.float32))
b = tape.input('b', np.array([[3.], [4.]], dtype=np.float32))
out = tape.matmul(a, b)
assert tape.shapes[out] == (1, 1)
np.testing.assert_allclose(execute('matmul', [tape.values[a], tape.values[b]], {}), [[11.]])
'''
    subprocess.run([sys.executable, '-I', '-c', code, str(source)], check=True)
