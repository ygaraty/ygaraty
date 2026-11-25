"""WS-51: the relay holds a sender while the next stage's mailbox is full, instead of dropping the stage."""

import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from ygaraty.backends.webgpu._legacy.qwen_protocol import (
    BoundedMailbox,
    SendBusy,
    SessionProtocol,
    SignalRejected,
    mailbox_bytes,
)


def session(bound):
    value = SessionProtocol.__new__(SessionProtocol)
    value.lock = threading.RLock()
    value.epoch = 0
    value.contributors = {"a": {"stage": 0}}
    value.stages = [{"fixture": {"plan_route": {"output": "hidden", "next": 1}}}, {}]
    value.inbox = [BoundedMailbox(), BoundedMailbox(max_bytes=bound)]
    value.hop_bytes = [0, 0]
    value.delayer = None
    value.contributor_roster = lambda: SimpleNamespace(touch=lambda cid: None)
    return value


class RelayBackpressureTest(unittest.TestCase):
    def test_empty_mailbox_takes_one_oversized_call(self):
        box = BoundedMailbox(max_bytes=10)
        box.put(("hidden", {}, b"x" * 100))
        with self.assertRaises(SignalRejected):
            box.put(("hidden", {}, b"x"))

    def test_wait_room_returns_once_the_receiver_reads(self):
        box = BoundedMailbox(max_bytes=200)
        box.put(("hidden", {}, b"x" * 100))
        threading.Timer(0.1, box.get).start()
        started = time.monotonic()
        self.assertTrue(box.wait_room(150, timeout=2))
        self.assertGreater(time.monotonic() - started, 0.05)
        self.assertEqual(box.waits, 1)

    def test_wait_room_times_out_while_full(self):
        box = BoundedMailbox(max_bytes=200)
        box.put(("hidden", {}, b"x" * 100))
        self.assertFalse(box.wait_room(150, timeout=0.05))

    def test_busy_send_can_be_retried_with_its_number(self):
        value = session(bound=300)
        with mock.patch.object(BoundedMailbox, "wait_room", return_value=False):
            with self.assertRaises(SendBusy):
                value.send("a", "hidden", {"epoch": 0}, b"x" * 100, n=0)
        self.assertEqual(value.inbox[1].qsize(), 0)
        value.send("a", "hidden", {"epoch": 0}, b"x" * 100, n=0)
        self.assertEqual(value.inbox[1].qsize(), 1)
        value.send("a", "hidden", {"epoch": 0}, b"x" * 100, n=0)  # a duplicate is still dropped
        self.assertEqual(value.inbox[1].qsize(), 1)

    def test_sender_waits_for_room_instead_of_failing(self):
        value = session(bound=300)
        value.send("a", "hidden", {"epoch": 0}, b"x" * 200, n=0)
        threading.Timer(0.1, value.inbox[1].get).start()
        value.send("a", "hidden", {"epoch": 0}, b"x" * 200, n=1)
        self.assertEqual(value.inbox[1].qsize(), 1)
        self.assertLessEqual(value.inbox[1].peak_bytes, 300)

    def test_mailbox_size_is_a_few_calls_or_whole_context(self):
        stage = {
            "fixture": {"max_positions": 8192, "max_tokens": 64, "wire": "f32"},
            "manifest": {"config": {"hidden": 2304}, "plan": {"input_vectors": 4}},
        }
        call = 64 * 2304 * 4 * 4
        with mock.patch.dict(os.environ, {"YGARATY_MAILBOX_CALLS": "4"}):
            self.assertEqual(mailbox_bytes(stage), 4 * call)
        with mock.patch.dict(os.environ, {"YGARATY_MAILBOX_CALLS": "0"}):
            self.assertEqual(mailbox_bytes(stage), (128 + 8) * call)


if __name__ == "__main__":
    unittest.main()
