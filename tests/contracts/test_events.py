"""Stream lifecycle checks, independent of network delivery or inference."""

import unittest

from ygaraty.backend_contracts.chat import (
    ErrorCode,
    EventKind,
    EventSequence,
    Failure,
    FinishReason,
    GenerationEvent,
    Usage,
)
from ygaraty.backend_contracts.lifecycle import (
    CancelReason,
    CancelReceipt,
    CancelStatus,
)


class StreamTests(unittest.TestCase):
    def test_all_terminal_outcomes(self):
        endings = (
            GenerationEvent(
                "r",
                2,
                EventKind.COMPLETED,
                finish_reason=FinishReason.STOP,
                usage=Usage(2, 1),
            ),
            GenerationEvent(
                "r",
                2,
                EventKind.FAILED,
                failure=Failure(ErrorCode.CONTRIBUTOR_LOST, "Worker lost"),
            ),
            GenerationEvent(
                "r",
                2,
                EventKind.CANCELLED,
                cancellation=CancelReceipt(
                    "r", CancelReason.DISCONNECTED, CancelStatus.TIMED_OUT
                ),
            ),
        )
        for ending in endings:
            with self.subTest(kind=ending.kind):
                stream = EventSequence("r")
                stream.accept(GenerationEvent("r", 0, EventKind.ACCEPTED))
                stream.accept(
                    GenerationEvent("r", 1, EventKind.TEXT, text="é")
                )
                stream.accept(ending)
                stream.finish()
                with self.assertRaises(ValueError):
                    stream.accept(ending)

    def test_bad_order_does_not_advance_validator(self):
        stream = EventSequence("r")
        with self.assertRaises(ValueError):
            stream.accept(GenerationEvent("r", 0, EventKind.TEXT, text="x"))
        stream.accept(GenerationEvent("r", 0, EventKind.ACCEPTED))
        for event in (
            GenerationEvent("r", 1, EventKind.ACCEPTED),
            GenerationEvent("other", 1, EventKind.TEXT, text="x"),
            GenerationEvent("r", 2, EventKind.TEXT, text="x"),
        ):
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    stream.accept(event)
        stream.accept(GenerationEvent("r", 1, EventKind.TEXT, text="valid"))

    def test_truncated_stream_is_not_success(self):
        stream = EventSequence("r")
        with self.assertRaises(ValueError):
            stream.finish()
        stream.accept(GenerationEvent("r", 0, EventKind.ACCEPTED))
        with self.assertRaises(ValueError):
            stream.finish()

    def test_payloads_are_discriminated(self):
        for args in (
            {"kind": EventKind.TEXT},
            {"kind": EventKind.ACCEPTED, "text": "unexpected"},
            {"kind": EventKind.COMPLETED},
            {"kind": EventKind.FAILED},
            {"kind": EventKind.CANCELLED},
            {"kind": EventKind.ACCEPTED, "usage": Usage(1, 1)},
            {
                "kind": EventKind.CANCELLED,
                "cancellation": CancelReceipt(
                    "other", CancelReason.REVOKED, CancelStatus.REQUESTED
                ),
            },
        ):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    GenerationEvent("r", 0, **args)


if __name__ == "__main__":
    unittest.main()
