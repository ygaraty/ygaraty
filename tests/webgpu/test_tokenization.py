"""Tokenization paths use the upstream batch API, without a real checkpoint."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ygaraty.backends.webgpu._legacy.chat_contract import encode_ids
from ygaraty.backends.webgpu._legacy.qwen_protocol import SessionProtocol


class TokenizationTests(unittest.TestCase):
    def test_helper_preserves_ids_and_uses_one_batch(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = AssertionError("GIL-holding path")
        tokenizer.encode_batch.return_value = [SimpleNamespace(ids=[2, 3])]
        self.assertEqual(encode_ids(tokenizer, "prompt"), [2, 3])
        tokenizer.encode_batch.assert_called_once_with(["prompt"])

    def test_serial_chat_preserves_previous_token_prefix(self):
        tokenizer = Mock()
        tokenizer.encode.side_effect = AssertionError("GIL-holding path")
        tokenizer.encode_batch.return_value = [SimpleNamespace(ids=[2, 3])]
        engine = SimpleNamespace(
            tokenizer=tokenizer, last_token=7, im_end=9, chat_style=None
        )
        self.assertEqual(
            SessionProtocol.chat_tokens(engine, "hello", True), [2, 3]
        )
        self.assertEqual(
            SessionProtocol.chat_tokens(engine, "hello", False), [7, 9, 2, 3]
        )
        engine.last_token = 9
        self.assertEqual(
            SessionProtocol.chat_tokens(engine, "hello", False), [9, 2, 3]
        )
