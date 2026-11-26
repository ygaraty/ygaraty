"""The committed models.toml: strict parsing and the derived presets."""

import tomllib
import unittest
from importlib.resources import files

from ygaraty.backends.webgpu import model_registry
from ygaraty.backends.webgpu.bootstrap import PROFILES
from ygaraty.backends.webgpu.downloads import PUBLIC_PRESETS


def raw():
    text = files(model_registry.__package__).joinpath("models.toml").read_text()
    return tomllib.loads(text)


class ModelRegistryTests(unittest.TestCase):
    def test_committed_file_parses_and_feeds_presets(self):
        registry = model_registry.load()
        self.assertEqual(set(PROFILES), set(registry.models))
        self.assertEqual(
            PUBLIC_PRESETS["qwen3.8-27b", 2].url,
            "https://ygaraty-models.fra1.digitaloceanspaces.com/"
            "qwen3.8-27b/prepared/v2/2-workers/bundle.json",
        )
        self.assertEqual(PUBLIC_PRESETS["s5-nano", 1].context, 16384)
        self.assertEqual(PUBLIC_PRESETS["s5-nano", 1].output, 4096)
        self.assertNotIn(("qwen3-coder-30b", 1), PUBLIC_PRESETS)

    def test_unknown_or_missing_keys_are_rejected(self):
        data = raw()
        data["models"]["s5-nano"]["surprise"] = 1
        with self.assertRaisesRegex(ValueError, "s5-nano"):
            model_registry.parse(data)
        data = raw()
        del data["models"]["s5-nano"]["public"]["output"]
        with self.assertRaisesRegex(ValueError, "public"):
            model_registry.parse(data)
        data = raw()
        data["kaa"] = {}
        with self.assertRaisesRegex(ValueError, "schema"):
            model_registry.parse(data)

    def test_bounds(self):
        data = raw()
        data["models"]["s5-nano"]["public"]["output"] = 16384
        with self.assertRaisesRegex(ValueError, "below context"):
            model_registry.parse(data)
        data = raw()
        data["models"]["s5-nano"]["splits"] = [0, 16]
        with self.assertRaisesRegex(ValueError, "splits"):
            model_registry.parse(data)
        data = raw()
        data["bundle_base"] = "http://example.org"
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            model_registry.parse(data)
        with self.assertRaises(KeyError):
            model_registry.load().bundle_url("qwen3-coder-30b", 1)


if __name__ == "__main__":
    unittest.main()
