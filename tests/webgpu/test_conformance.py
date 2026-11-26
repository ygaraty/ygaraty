"""WebGPU against the shared backend conformance suite (fake engine, no GPU)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from session_fakes import setup

from ygaraty.backend_contracts.testing import BackendConformance
from ygaraty.backends.webgpu.factory import WebGPUFactory


class WebGPUConformance(BackendConformance, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.f = setup()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "bundle.json").write_bytes(self.f.raw)

    def make_factory(self):
        return WebGPUFactory(
            self.root / "bundle.json",
            self.root,
            engine_loader=Mock(return_value=self.f.engine),
        )

    def make_spec(self):
        return self.f.spec

    def make_services(self):
        return self.f.services


if __name__ == "__main__":
    unittest.main()
