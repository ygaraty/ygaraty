"""Missing local inputs produce actionable errors before model preparation."""

import tempfile
import unittest
from pathlib import Path

from ygaraty.backends.webgpu.bundle_io import local_input


class LocalInputTests(unittest.TestCase):
    def test_missing_parent_reports_file_and_placeholder_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing/checkpoint/model.gguf"
            with self.assertRaises(ValueError) as error:
                local_input(missing)
            self.assertIn(str(missing), str(error.exception))
            self.assertIn("placeholder", str(error.exception))
            self.assertFalse(missing.parent.exists())
