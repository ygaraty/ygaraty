"""Preserved numerical/report helpers work without hosted applications."""

import ast
import hashlib
import importlib
import io
import json
import pkgutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import ygaraty_webgpu_research
from ygaraty_webgpu_research.host import component


class PublicResearchTests(unittest.TestCase):
    def test_all_modules_and_component_requests_are_public(self):
        for info in pkgutil.iter_modules(ygaraty_webgpu_research.__path__):
            module = importlib.import_module(
                "ygaraty_webgpu_research." + info.name
            )
            for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_host_component"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    component(node.args[0].value)

    def test_factorized_reference_matches_dense_reference(self):
        from ygaraty_webgpu_research.factorization_browser_training import (
            dense_reference,
            factor_space_reference,
        )

        args = SimpleNamespace(
            rows=4,
            rank=2,
            block=2,
            seed=9,
            updates=3,
            learning_rate=0.05,
            regularization=0.001,
        )
        a = np.random.default_rng(4).normal(size=(4, 3))
        x = a @ a.T / np.sqrt(a.shape[1])
        losses, zero_loss = factor_space_reference(a, args)
        np.testing.assert_allclose(
            losses, dense_reference(x, args), rtol=1e-12, atol=1e-12
        )
        self.assertAlmostEqual(zero_loss, float(np.sum(x * x) / (2 * x.size)))

    def test_weight_range_helpers_reject_truncated_and_invalid_ranges(self):
        from ygaraty_webgpu_research.qwen35_session import (
            range_hashes,
            read_range,
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.bin"
            path.write_bytes(b"0123")
            for invalid in ((0, 5), (-1, 2), (0, -1), (5, 0)):
                with self.assertRaisesRegex(ValueError, "outside"):
                    range_hashes(path, [invalid])
            with self.assertRaisesRegex(ValueError, "Truncated"):
                read_range((path, 1, 4))
            # Simulate the file shrinking between stat() and the hash read.
            with patch(
                "ygaraty_webgpu_research.qwen35_session.open",
                return_value=io.BytesIO(b""),
                create=True,
            ):
                with self.assertRaisesRegex(ValueError, "Truncated"):
                    range_hashes(path, [(0, 4)])
            self.assertEqual(
                range_hashes(path, [(4, 0)])[(4, 0)],
                hashlib.sha256(b"").hexdigest(),
            )

    def test_range_helpers_and_saved_report_cli(self):
        from ygaraty_webgpu_research.convnet_remote_scaling import main
        from ygaraty_webgpu_research.qwen35_session import (
            range_hashes,
            read_range,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "weights.bin"
            weights.write_bytes(b"0123456789")
            self.assertEqual(read_range((weights, 2, 4)), b"2345")
            hashes = range_hashes(weights, [(2, 4)])
            self.assertEqual(
                hashes[(2, 4)], hashlib.sha256(b"2345").hexdigest()
            )
            self.assertEqual(range_hashes(weights, [(2, 4)]), hashes)
            (root / "report.json").write_text(json.dumps({"state": "failed"}))
            with self.assertRaisesRegex(ValueError, "completed"):
                main(["report", str(root)])
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    main(["--help"])
            self.assertEqual(result.exception.code, 0)
