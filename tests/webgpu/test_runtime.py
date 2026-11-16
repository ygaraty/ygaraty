"""Compatibility engine tests with synthetic tokens, no model or sockets."""

import io
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from ygaraty.backends.webgpu._legacy.conversations import Conversations
from ygaraty.backends.webgpu._legacy.prepared import PreparedSession
from ygaraty.backends.webgpu._legacy.qwen_protocol import BoundedMailbox


def engine():
    value = SimpleNamespace(
        state="running",
        capacity=20,
        epoch=0,
        stop_ids={99},
        args=SimpleNamespace(turn_timeout=1),
        turns=[],
        chat_style=None,
        tokenizer=SimpleNamespace(
            encode=lambda text: SimpleNamespace(ids=[1, 2]),
            decode=lambda ids: "".join(chr(65 + n) for n in ids),
        ),
        inbox=[queue.Queue(), queue.Queue()],
        deliver=Mock(),
    )
    value.tokenizer.encode_batch = lambda texts: [
        value.tokenizer.encode(t) for t in texts
    ]
    value.concurrent_chat = Conversations(value)
    return value


def prompt(value):
    while True:
        kind, meta, _ = value.inbox[0].get(timeout=1)
        if kind == "prompt":
            return meta


class RuntimeTests(unittest.TestCase):
    def test_eight_slots_advance_before_any_finishes(self):
        value = engine()
        chats = value.concurrent_chat
        with ThreadPoolExecutor(8) as pool:
            jobs = [
                pool.submit(
                    lambda i=i: list(
                        chats.chat(
                            "",
                            prompt_tokens=[i],
                            conversation_id=f"user{i}",
                            max_tokens=2,
                        )
                    )
                )
                for i in range(8)
            ]
            packets = [prompt(value) for _ in range(8)]
            self.assertEqual(len({p["conv"] for p in packets}), 8)
            self.assertIn("busy", list(chats.chat("extra"))[0]["error"])
            for packet in packets:
                chats.token(packet, packet["conv"], True)
            self.assertTrue(
                all(len(t["tokens"]) == 1 for t in chats.active.values())
            )
            for packet in packets:
                chats.token(packet, 99, True)
            results = [job.result(timeout=1) for job in jobs]
        self.assertTrue(all(r[-1]["generated"] == 2 for r in results))
        self.assertFalse(chats.active)

    def test_cancelled_slot_fences_old_packets_without_resetting_other(self):
        value = engine()
        chats = value.concurrent_chat
        cancelled = threading.Event()
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(
                lambda: list(
                    chats.chat(
                        "a", cancel_event=cancelled, conversation_id="a"
                    )
                )
            )
            old = prompt(value)
            second = pool.submit(
                lambda: list(
                    chats.chat("b", conversation_id="b", max_tokens=1)
                )
            )
            other = prompt(value)
            cancelled.set()
            first.result(timeout=1)
            replacement = pool.submit(
                lambda: list(
                    chats.chat("c", conversation_id="c", max_tokens=1)
                )
            )
            current = prompt(value)
            self.assertEqual(current["conv"], old["conv"])
            chats.token(old, 7, True)
            self.assertFalse(chats.active[current["conv"]]["tokens"])
            chats.token(current, 2, True)
            chats.token(other, 1, True)
            self.assertEqual(second.result(timeout=1)[-1]["full_text"], "B")
            self.assertEqual(
                replacement.result(timeout=1)[-1]["full_text"], "C"
            )

    def test_shutdown_drains_full_mailbox_and_closes_resources(self):
        value = PreparedSession.__new__(PreparedSession)
        box = BoundedMailbox(max_items=1)
        box.put(("token", {}, b""))
        value.inbox, value.turn = [box], None
        value.throughput = Mock()
        value.concurrent_chat = Mock()
        value.log_lock, value.log = threading.Lock(), io.StringIO()
        roster = Mock()
        roster.snapshot.return_value = {"contributors": [{"id": "test"}]}
        value.contributor_roster = lambda: roster
        value.close()
        self.assertTrue(value.log.closed)
        self.assertEqual(box.get_nowait()[0], "stop")
        roster.retire.assert_called_once()
