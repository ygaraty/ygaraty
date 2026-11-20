"""Exact packed conversion and Kyse byte/rank tokenizer contracts."""

import base64
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ygaraty.backends.webgpu.checkpoints.kyse import (
    convert_checkpoint,
    launch_checkpoint,
    read_tokenizer,
    repack_matrix,
)
from ygaraty.inference_support.kyse_tokenizer import (
    KyseTokenizer,
    load_tokenizer,
)


def vocabulary():
    tokens = [bytes([i]) for i in range(256)]
    tokens += [b"ab", b"abc", b"bc", b"aa", b"aaa"]
    ranks = [-1] * 256 + [1, 3, 2, 4, 5]
    special = []
    for token in [b"<|im_start|>", b"<|im_end|>", b"<|endoftext|>"]:
        special.append(len(tokens))
        tokens.append(token)
        ranks.append(-1)
    return tokens, ranks, special


def tokenizer_file(path):
    tokens, ranks, special = vocabulary()
    with path.open("wb") as file:
        file.write(struct.pack("<3i", len(tokens), len(special), 32))
        file.write(struct.pack(f"<{len(special)}i", *special))
        for token, rank in zip(tokens, ranks):
            file.write(struct.pack("<2i", rank, len(token)))
            file.write(token)


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        tokens, ranks, special = vocabulary()
        self.document = dict(
            type="ygaraty-kyse-bpe",
            version=1,
            tokens=[base64.b64encode(t).decode() for t in tokens],
            ranks=ranks,
            special_ids=special,
        )
        self.tokenizer = KyseTokenizer(self.document)

    def test_rank_order_and_leftmost_ties(self):
        self.assertEqual(self.tokenizer.encode("abc").ids, [257])
        self.assertEqual(self.tokenizer.encode("aaaaa").ids, [259, 260])
        self.assertEqual(self.tokenizer.encode("1abc2").ids, [49, 257, 50])

    def test_specials_and_byte_roundtrip(self):
        for text in [
            " a\t\r\n b ",
            "café 日本語 🙂",
            "<|im_start|>abc<|im_end|>",
        ]:
            ids = self.tokenizer.encode_batch([text])[0].ids
            self.assertEqual(
                self.tokenizer.decode(ids, skip_special_tokens=False), text
            )
        self.assertEqual(
            self.tokenizer.decode(
                self.tokenizer.encode("<|im_start|>abc").ids
            ),
            "abc",
        )
        with self.assertRaises(ValueError):
            self.tokenizer.encode("bad\0text")

    def test_binary_import_and_json_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.bin"
            tokenizer_file(path)
            document, original = read_tokenizer(path)
            converted = path.with_suffix(".json")
            converted.write_text(json.dumps(document))
            self.assertEqual(
                load_tokenizer(converted).encode("abc").ids,
                original.encode("abc").ids,
            )
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                read_tokenizer(path)

    def test_rejects_missing_byte_and_invalid_special(self):
        with self.assertRaises(ValueError):
            KyseTokenizer(dict(self.document, special_ids=[]))
        duplicate = list(self.document["tokens"])
        duplicate[-1] = duplicate[-2]
        with self.assertRaises(ValueError):
            KyseTokenizer(dict(self.document, tokens=duplicate))
        bad = dict(self.document, special_ids=[999999])
        with self.assertRaises(ValueError):
            KyseTokenizer(bad)
        bad = dict(self.document, tokens=[""] + self.document["tokens"][1:])
        with self.assertRaises(ValueError):
            KyseTokenizer(bad)


@unittest.skipUnless(
    importlib.util.find_spec("gguf"), "install ygaraty[prepare]"
)
class ConversionTests(unittest.TestCase):
    def test_quantized_values_are_exact_without_requantization(self):
        import gguf

        rng = np.random.default_rng(482)
        for kind in [1, 2]:
            codes = rng.integers(
                -127 if kind == 1 else -8,
                128 if kind == 1 else 8,
                (5, 96),
                dtype=np.int16,
            )
            scales = rng.uniform(0.001, 0.2, (15, 1)).astype("<f2")
            if kind == 1:
                raw = codes.astype(np.int8).tobytes()
            else:
                biased = (codes.reshape(-1) + 8).astype(np.uint8)
                raw = (biased[::2] | (biased[1::2] << 4)).tobytes()
            raw += scales.tobytes()
            packed, name = repack_matrix(raw, kind, 5, 96)
            actual = gguf.quants.dequantize(
                packed, gguf.GGMLQuantizationType[name]
            )
            expected = (
                codes.reshape(-1, 32) * scales.astype(np.float32)
            ).reshape(5, 96)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(packed.nbytes, len(raw))

    def test_bf16_expansion_is_lossless(self):
        values = np.array([0x3F80, 0xBF80, 0x3EAB, 0x0001], dtype="<u2")
        packed, kind = repack_matrix(values.tobytes(), 0, 1, 4)
        self.assertEqual(kind, "F32")
        np.testing.assert_array_equal(
            packed.view(np.uint32).reshape(-1), values.astype(np.uint32) << 16
        )

    def test_lowrank_is_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_file(root / "tokenizer.bin")
            # Valid four-layer geometry, with unsupported rank-4 embedding.
            header = b"QW35" + struct.pack(
                "<i13i2f4i3ii",
                4,
                256,
                256,
                4,
                len(vocabulary()[0]),
                4,
                1,
                256,
                64,
                1,
                1,
                128,
                128,
                4,
                1e-6,
                1e7,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                32,
            )
            (root / "model.bin").write_bytes(
                header + struct.pack("<32i", 1025, *([1] * 31))
            )
            with self.assertRaisesRegex(ValueError, "low-rank"):
                convert_checkpoint(root, root / "out")
            self.assertFalse((root / "out").exists())

    def test_launch_rejects_multiple_workers_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.bin").write_bytes(b"QW35")
            with patch(
                "ygaraty.backends.webgpu.checkpoints.kyse.convert_checkpoint"
            ) as convert:
                with self.assertRaisesRegex(ValueError, "num-workers 1"):
                    launch_checkpoint(root, root / "session", 2)
                convert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
