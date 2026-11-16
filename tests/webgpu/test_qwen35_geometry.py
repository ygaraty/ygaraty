"""Qwen3.5 layouts the WebGPU shaders accept: 9B/S5-Nano, 0.8B and 3.8-27B."""

import json
import tempfile
import unittest
from pathlib import Path

from ygaraty.backends.webgpu.checkpoints.qwen35_gguf import (
    WebGPUConfig,
    value_orders,
)


def text_config(**overrides):
    layers = overrides.get("num_hidden_layers", 32)
    config = dict(
        num_hidden_layers=layers,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        hidden_size=4096,
        intermediate_size=12288,
        hidden_act="silu",
        attn_output_gate=True,
        rope_parameters=dict(
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            rope_type="default",
        ),
        layer_types=[
            "full_attention" if i % 4 == 3 else "linear_attention"
            for i in range(layers)
        ],
        tie_word_embeddings=False,
    )
    config.update(overrides)
    return {"text_config": config}


SMALL = dict(  # Qwen3.5-0.8B
    num_hidden_layers=24,
    num_attention_heads=8,
    num_key_value_heads=2,
    linear_num_value_heads=16,
    hidden_size=1024,
    intermediate_size=3584,
    tie_word_embeddings=True,
)

# Qwen3.8-27B: six query heads per KV head, three value heads per key head
LARGE = dict(
    num_hidden_layers=64,
    num_attention_heads=24,
    num_key_value_heads=4,
    linear_num_value_heads=48,
    hidden_size=5120,
    intermediate_size=17408,
)


class GeometryTests(unittest.TestCase):
    def load(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(text_config(**overrides)))
            return WebGPUConfig.from_file(path)

    def test_9b_defaults_are_unchanged(self):
        config = self.load()
        self.assertEqual(config, WebGPUConfig())
        self.assertEqual(
            (
                config.conv_dim,
                config.value_dim,
                config.attention_dim,
                config.kv_dim,
            ),
            (8192, 4096, 4096, 1024),
        )

    def test_08b_layout_is_accepted(self):
        config = self.load(**SMALL)
        self.assertEqual(
            (
                config.layers,
                config.attention_heads,
                config.attention_kv_heads,
                config.linear_k_heads,
                config.linear_v_heads,
            ),
            (24, 8, 2, 16, 16),
        )
        self.assertEqual(
            (
                config.conv_dim,
                config.value_dim,
                config.attention_dim,
                config.kv_dim,
            ),
            (6144, 2048, 2048, 512),
        )
        # The browser receives the head counts in the stage manifest.
        manifest = config.manifest()
        self.assertEqual(manifest["attention_heads"], 8)
        self.assertEqual(manifest["linear_v_heads"], 16)
        # As many value heads as key heads: the converter's reorder is
        # the identity.
        channels, heads = value_orders(config)
        self.assertEqual(list(heads), list(range(16)))
        self.assertEqual(list(channels), list(range(2048)))

    def test_27b_layout_is_accepted(self):
        config = self.load(**LARGE)
        self.assertEqual(
            (config.layers, config.attention_heads, config.attention_kv_heads),
            (64, 24, 4),
        )
        self.assertEqual(
            (
                config.conv_dim,
                config.value_dim,
                config.attention_dim,
                config.kv_dim,
            ),
            (10240, 6144, 6144, 1024),
        )
        _, heads = value_orders(config)
        self.assertEqual(list(heads[:4]), [0, 16, 32, 1])

    def test_9b_value_heads_are_tiled(self):
        _, heads = value_orders(WebGPUConfig())
        self.assertEqual(list(heads[:4]), [0, 16, 1, 17])

    def test_layouts_the_shaders_cannot_run_are_rejected(self):
        for overrides, field in [
            # Query heads must be a whole multiple of the KV heads.
            (dict(num_attention_heads=10), "num_attention_heads"),
            (dict(num_hidden_layers=26), "num_hidden_layers"),
            (dict(linear_num_value_heads=24), "linear_num_value_heads"),
            (dict(head_dim=128), "head_dim"),
            (dict(linear_conv_kernel_dim=3), "linear_conv_kernel_dim"),
        ]:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    self.load(**overrides)
        with self.assertRaisesRegex(ValueError, "schedule"):
            self.load(layer_types=["linear_attention"] * 32)


if __name__ == "__main__":
    unittest.main()
