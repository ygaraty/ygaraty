"""Terminal chat against a fake streaming API; no sockets or GPU."""

import asyncio
import io
import json
import unittest
from urllib.error import HTTPError

from ygaraty.backends.webgpu.terminal_chat import (
    format_stats,
    local_base,
    network_summary,
    stream_reply,
    terminal_chat,
)


def sse(*parts, finish=True, usage=None):
    lines = [
        "data: "
        + json.dumps({"choices": [{"delta": {"content": p}}]})
        + "\n\n"
        for p in parts
    ]
    if finish:
        lines.append(
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    **({"usage": usage} if usage else {}),
                }
            )
            + "\n\n"
        )
    lines.append("data: [DONE]\n\n")
    return io.BytesIO("".join(lines).encode())


class FakeAPI:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request, timeout):
        self.requests.append(
            (request.full_url, dict(request.header_items()), request.data)
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TerminalChatTests(unittest.TestCase):
    def test_local_base_uses_loopback_for_wildcard_hosts(self):
        self.assertEqual(local_base("0.0.0.0", 8765), "http://127.0.0.1:8765")
        self.assertEqual(local_base("::1", 9), "http://[::1]:9")
        self.assertEqual(local_base("10.0.0.2", 1), "http://10.0.0.2:1")

    def test_stream_reply_collects_text_and_requires_finish(self):
        api, seen = FakeAPI(sse("Hel", "lo"), sse("cut", finish=False)), []
        answer, stats = stream_reply(
            "http://h", "owner", [], "c1", seen.append, opener=api
        )
        self.assertEqual((answer, seen), ("Hello", ["Hel", "lo"]))
        self.assertEqual(stats["tokens"], 2)  # no usage: counted chunks
        url, headers, body = api.requests[0]
        self.assertEqual(url, "http://h/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer owner")
        self.assertEqual(
            json.loads(body),
            {"messages": [], "conversation_id": "c1", "stream": True},
        )
        with self.assertRaisesRegex(ValueError, "interrupted"):
            stream_reply("http://h", "owner", [], "c1", print, opener=api)

    def test_stats_use_server_usage_and_exclude_ttft_from_rate(self):
        ticks = iter([10.0, 10.5, 12.5])  # start, first token, end
        api = FakeAPI(
            sse("a", "b", usage={"prompt_tokens": 7, "completion_tokens": 9})
        )
        _, stats = stream_reply(
            "http://h", "k", [], "c", lambda _: None, api, lambda: next(ticks)
        )
        self.assertEqual(
            (stats["ttft"], stats["tokens"], stats["prompt_tokens"]),
            (0.5, 9, 7),
        )
        self.assertAlmostEqual(stats["rate"], 4.0)  # 8 tokens after first
        self.assertEqual(
            format_stats(stats), "TTFT 0.50 s · 9 tokens · 4.0 tok/s"
        )

    def test_network_summary_relay_and_webrtc_links(self):
        self.assertIsNone(network_summary(None))
        self.assertEqual(network_summary({"network": "relay"}), "relay")
        one = {"network": "direct", "roster": {"required": 1}}
        self.assertIsNone(network_summary(one))  # no route to report
        self.assertIn("one worker", network_summary(one, detail=True))
        link = {
            "peer": 1,
            "state": "connected",
            "local": "host",
            "remote": "srflx",
            "protocol": "udp",
            "rttMs": 0.42,
            "relinks": 1,
        }
        status = {
            "network": "direct",
            "stages": [
                {"stage": "A", "links": [link]},
                {"stage": "B", "links": [dict(link, peer=0)]},
            ],
        }
        self.assertEqual(
            network_summary(status),
            "WebRTC: A→B connected host/srflx udp rtt 0.42 ms",
        )
        self.assertEqual(
            network_summary(status, detail=True),
            "Route: WebRTC\n"
            "  A→B connected host/srflx udp rtt 0.42 ms, relinks 1",
        )
        pending = {"network": "private", "stages": [{"stage": "A"}]}
        self.assertEqual(
            network_summary(pending), "WebRTC, TURN only, links pending"
        )

    def test_quit_asks_to_stop_and_eof_only_closes(self):
        def run(lines):
            it = iter(lines)

            def read(prompt):
                try:
                    return next(it)
                except StopIteration:
                    raise EOFError from None

            return asyncio.run(
                terminal_chat(
                    "http://h", "k", read=read, write=lambda *a, **k: None
                )
            )

        self.assertEqual(run(["/quit", "never read"]), "quit")
        self.assertIsNone(run([]))

    def test_help_lists_commands_and_sends_nothing(self):
        lines, written = iter(["/help", "/quit"]), []

        def read(_):
            return next(lines)

        api = FakeAPI([])
        result = asyncio.run(
            terminal_chat(
                "http://h",
                "k",
                read=read,
                write=lambda *a, **k: written.append(a[0] if a else ""),
                opener=api,
            )
        )
        self.assertEqual(result, "quit")
        self.assertIn("/reset", written[0])
        self.assertIn("Ctrl-D", written[0])
        self.assertEqual(api.requests, [])

    def test_conversation_history_reset_errors_and_eof(self):
        busy = HTTPError(
            "u", 503, "Busy", {}, io.BytesIO(b'{"detail":"No slot"}')
        )
        api = FakeAPI(sse("Paris."), sse("Lyon."), busy, sse("Hi."))
        lines = iter(
            ["capital?", "", "second?", "third?", "/net", "/reset", "hi"]
        )

        def read(prompt):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

        output = []

        def write(*parts, end="\n", flush=False):
            output.append("".join(map(str, parts)) + end)

        asyncio.run(
            terminal_chat(
                "http://h",
                "k",
                read=read,
                write=write,
                opener=api,
                status=lambda: {"network": "relay"},
            )
        )
        bodies = [json.loads(r[2]) for r in api.requests]
        self.assertEqual(len(bodies), 4)  # the empty line is skipped
        self.assertEqual(
            [m["content"] for m in bodies[1]["messages"]],
            ["capital?", "Paris.", "second?"],
        )
        # A failed turn is not added to the history.
        self.assertEqual(len(bodies[2]["messages"]), 5)
        self.assertEqual(
            bodies[3]["messages"], [{"role": "user", "content": "hi"}]
        )
        self.assertNotEqual(
            bodies[3]["conversation_id"], bodies[0]["conversation_id"]
        )
        text = "".join(output)
        self.assertIn("[error] HTTP 503: No slot", text)
        self.assertIn("New conversation.", text)
        self.assertIn("[TTFT", text)
        self.assertIn("1 token · relay]", text)
        self.assertIn("Route: relay", text)  # /net
        self.assertTrue(text.rstrip().endswith("the server keeps running."))


if __name__ == "__main__":
    unittest.main()
