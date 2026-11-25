"""Session statistics, the public status page data and the CLI additions."""

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from fixtures import bundle
from session_fakes import CONTRIBUTOR, OWNER, setup

from ygaraty.backends.webgpu.catalog import catalog
from ygaraty.backends.webgpu.cli import main
from ygaraty.backends.webgpu.factory import WebGPUFactory
from ygaraty.backends.webgpu.server import create_app
from ygaraty.backends.webgpu.stats import SessionStats
from ygaraty.backends.webgpu.terminal_chat import (
    StatusBar,
    bar_text,
    session_details,
    stats_text,
    terminal_chat,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class StatsTests(unittest.TestCase):
    def test_ttft_rate_concurrency_and_totals(self):
        clock = Clock()
        stats = SessionStats(clock)
        stats.started("a")
        stats.started("b")
        clock.now = 100.5
        stats.first_text("a")
        stats.first_text("a")  # only the first text counts
        clock.now = 102.5
        stats.finished("a", "completed", prompt=10, generated=9)
        stats.finished("b", "cancelled", prompt=3, generated=0)
        summary = stats.summary()
        self.assertEqual(summary["peak_active_requests"], 2)
        self.assertEqual(summary["active_requests"], 0)
        self.assertEqual(
            summary["requests"], dict(completed=1, failed=0, cancelled=1)
        )
        self.assertEqual(summary["total_tokens"], 22)
        self.assertEqual(summary["ttft_s"]["average"], 0.5)
        # 8 tokens after the first, in 2 s: TTFT does not dilute the rate.
        self.assertEqual(summary["tokens_per_s"]["average"], 4.0)
        self.assertEqual(summary["throughput_tokens_per_s"], 0.15)
        self.assertEqual(summary["last"]["generated_tokens"], 9)
        clock.now = 200.0  # the last-minute window has passed
        self.assertEqual(stats.summary()["throughput_tokens_per_s"], 0)

    def test_failed_requests_do_not_enter_speed_averages(self):
        stats = SessionStats(Clock())
        stats.started("a")
        stats.finished("a", "failed")
        summary = stats.summary()
        self.assertEqual(summary["requests"]["failed"], 1)
        self.assertIsNone(summary["tokens_per_s"]["average"])
        self.assertIsNone(summary["last"])


class StatusEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = setup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "bundle.json"
        path.write_bytes(self.f.raw)
        factory = WebGPUFactory(
            path,
            Path(self.temp.name),
            engine_loader=Mock(return_value=self.f.engine),
        )
        self.app = create_app(
            factory, self.f.spec, self.f.services, self.f.access
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://t"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)

    async def status(self, key=OWNER):
        return await self.client.get(
            "/v1/status", headers={"Authorization": f"Bearer {key}"}
        )

    async def test_status_needs_a_key_by_default(self):
        self.assertEqual(
            (await self.client.get("/v1/status")).status_code, 401
        )
        self.assertEqual((await self.status("wrong")).status_code, 401)
        for key in (OWNER, CONTRIBUTOR):
            self.assertEqual((await self.status(key)).status_code, 200)
        page = (await self.client.get("/")).text
        self.assertIn('data-open-status="false"', page)

    async def test_open_status_needs_no_key(self):
        app = create_app(
            WebGPUFactory(
                Path(self.temp.name) / "bundle.json",
                Path(self.temp.name),
                engine_loader=Mock(return_value=self.f.engine),
            ),
            self.f.spec,
            self.f.services,
            self.f.access,
            open_status=True,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as client:
                self.assertEqual(
                    (await client.get("/v1/status")).status_code, 200
                )
                self.assertIn(
                    'data-open-status="true"', (await client.get("/")).text
                )

    async def test_status_counts_answers_without_exposing_keys(self):
        status = (await self.status()).json()
        self.assertEqual(status["model"], self.f.spec.model.model_id)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["stats"]["requests"]["completed"], 0)
        self.assertNotIn(OWNER, json.dumps(status))
        self.assertNotIn(CONTRIBUTOR, json.dumps(status))
        response = await self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {OWNER}"},
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status_code, 200)
        status = (await self.status()).json()
        self.assertEqual(status["stats"]["requests"]["completed"], 1)
        self.assertEqual(status["stats"]["total_tokens"], 4)
        self.assertEqual(status["slots"], 2)
        session = await self.client.get(
            "/v1/session", headers={"Authorization": f"Bearer {OWNER}"}
        )
        self.assertEqual(
            session.json()["details"]["stats"]["generated_tokens"], 2
        )

    async def test_contributor_key_only_for_the_owner(self):
        for key in ("", CONTRIBUTOR):
            response = await self.client.get(
                "/v1/session/contributor-key",
                headers={"Authorization": f"Bearer {key}"},
            )
            self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/v1/session/contributor-key",
            headers={"Authorization": f"Bearer {OWNER}"},
        )
        self.assertEqual(response.json(), {"key": CONTRIBUTOR})
        self.assertEqual(response.headers["cache-control"], "no-store")

    async def test_session_page_has_status_and_contributor_controls(self):
        page = (await self.client.get("/")).text
        for marker in ('id="session-stats"', 'id="owner"', 'id="accept"'):
            self.assertIn(marker, page)
        module = await self.client.get("/session_status.mjs")
        self.assertEqual(module.status_code, 200)
        self.assertIn("/v1/status", module.text)


DETAILS = {
    "model": "s5",
    "state": "ready",
    "context_tokens": 4608,
    "slots": 1,
    "route": "relay",
    "network": None,
    "stats": {
        "total_tokens": 1234,
        "prompt_tokens": 1000,
        "generated_tokens": 234,
        "active_requests": 0,
        "peak_active_requests": 1,
        "throughput_tokens_per_s": 2.5,
        "requests": {"completed": 3, "failed": 0, "cancelled": 1},
        "tokens_per_s": {"average": 19.8, "best": 21.0},
        "ttft_s": {"average": 0.61, "best": 0.4},
    },
}


class Opener:
    def __init__(self, body):
        self.body, self.urls = body, []

    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        return io.BytesIO(json.dumps(self.body).encode())


class TerminalStatusTests(unittest.TestCase):
    def test_bar_and_stats_text(self):
        self.assertEqual(
            bar_text(DETAILS),
            "Ygaraty s5 · ready · 1,234 tokens · avg 19.8 tok/s · "
            "TTFT 0.61 s · context 4,608 · 0/1 chats · relay",
        )
        waiting = dict(DETAILS, state="unavailable", workers_required=2)
        self.assertIn("waiting for workers (0/2)", bar_text(waiting))
        self.assertIn("server unavailable", bar_text(None))
        text = stats_text(DETAILS)
        self.assertIn("1,000 prompt + 234 generated", text)
        self.assertIn("3 completed, 0 failed, 1 cancelled", text)

    def test_session_details_reads_the_owner_session(self):
        opener = Opener({"state": "ready", "details": DETAILS})
        self.assertEqual(session_details("http://h", "k", opener), DETAILS)
        self.assertEqual(opener.urls, ["http://h/v1/session"])

    def test_status_bar_pins_row_one_and_restores_the_screen(self):
        out = io.StringIO()

        def size():
            return type("S", (), {"lines": 30, "columns": 40})()

        bar = StatusBar(out, size)
        self.assertTrue(bar.start())
        bar.draw("x" * 100)  # truncated to the terminal width
        bar.stop()
        written = out.getvalue()
        self.assertIn("\033[2;30r", written)  # rows 2–30 scroll
        self.assertIn("…", written)
        self.assertTrue(written.endswith("\0337\033[r\033[1;1H\033[2K\0338"))
        bar.draw("after stop")  # inactive: nothing more is written
        self.assertEqual(out.getvalue(), written)

    def test_stats_command_in_terminal_chat(self):
        lines = iter(["/stats"])

        def read(prompt):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

        printed = []
        asyncio.run(
            terminal_chat(
                "http://h",
                "k",
                read=read,
                write=lambda *a, **k: printed.append(a[0] if a else ""),
                opener=Opener({"details": DETAILS}),
                status_bar=False,
            )
        )
        self.assertTrue(any("Decode speed" in p for p in printed))


class CliAdditionTests(unittest.TestCase):
    def test_models_list(self):
        names = [row["model"] for row in catalog()]
        self.assertIn("qwen3.5-0.8b", names)
        self.assertNotIn("qwen3-coder-30b", names)  # no public bundle
        self.assertIn("qwen3-coder-30b", [r["model"] for r in catalog(True)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["models", "list", "--json"])
        rows = json.loads(out.getvalue())
        self.assertEqual(rows[0]["public_bundle_workers"], [1, 2, 3, 4])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["models"])
        self.assertIn("ygaraty launch --model MODEL", out.getvalue())

    def write_bundle(self, path):
        data = bundle()
        data["tokenizer"] = "tokenizer.json"
        data["files"]["synthetic-weight"]["sha256"] = "a" * 64
        (path / "tokenizer.json").write_text("{}")
        (path / "bundle.json").write_text(json.dumps(data))
        return len(data["stages"])

    def test_serve_defaults_workspace_keys_and_prints_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            stages = self.write_bundle(path)
            out = io.StringIO()
            with (
                contextlib.redirect_stdout(out),
                patch.dict("os.environ", {"YGARATY_WORKSPACE_ROOT": tmp}),
                patch("uvicorn.run") as run,
            ):
                main(
                    [
                        "serve",
                        "--bundle",
                        str(path / "bundle.json"),
                        "--model",
                        "fixture",
                        "--route",
                        "relay",
                    ]
                )
            self.assertEqual(run.call_count, 1)
            keys = path / "fixture" / f"{stages}-workers" / "keys"
            contributor = (keys / "contributor.key").read_text().strip()
            text = out.getvalue()
            self.assertIn(f"/#key={contributor}", text)
            self.assertIn(
                "Session page (private): http://127.0.0.1:8765/#owner=", text
            )

    def test_serve_rejects_a_worker_count_the_bundle_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            stages = self.write_bundle(path)
            with (
                contextlib.redirect_stderr(io.StringIO()) as err,
                patch("uvicorn.run") as run,
                self.assertRaises(SystemExit),
            ):
                main(
                    [
                        "serve",
                        "--bundle",
                        str(path / "bundle.json"),
                        "--model",
                        "fixture",
                        "--workspace",
                        str(path / "ws"),
                        "--num-workers",
                        str(stages + 1),
                    ]
                )
            run.assert_not_called()
            self.assertIn("--num-workers must match", err.getvalue())


if __name__ == "__main__":
    unittest.main()
