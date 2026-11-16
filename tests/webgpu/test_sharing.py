"""Tunnel lifecycle tests using fake processes, without sockets."""

import asyncio
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ygaraty.backends.webgpu.sharing import PublicTunnel, serve_launch


def registered(address="demo-app-name.trycloudflare.com"):
    # cloudflared's quick-tunnel banner, as it logs it
    return (
        b"2026-11-16T12:00:00Z INF |  https://" + address.encode()
        + b"                    |\n"
    )


class Process:
    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self.stopped = asyncio.Event()
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stdout.feed_eof()
        self.stopped.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stopped.set()

    async def wait(self):
        await self.stopped.wait()
        return self.returncode


class TunnelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "path with spaces"
        self.tunnel = PublicTunnel(
            "0.0.0.0", 8765, self.directory, timeout=0.05
        )
        self.tunnel.wait_reachable = AsyncMock(return_value=True)
        self.process = Process()
        self.enterContext(patch.dict("os.environ", {}, clear=False))
        self.enterContext(
            patch(
                "ygaraty.backends.webgpu.sharing.shutil.which",
                return_value="/usr/bin/cloudflared",
            )
        )
        self.spawn = self.enterContext(
            patch(
                "ygaraty.backends.webgpu.sharing.asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=self.process),
            )
        )

    async def test_parse_quick_tunnel_banner_and_close(self):
        self.process.stdout.feed_data(
            b"INF Requesting new quick Tunnel on trycloudflare.com...\n"
        )
        self.process.stdout.feed_data(registered())
        self.assertEqual(
            await self.tunnel.start(),
            "https://demo-app-name.trycloudflare.com",
        )
        self.assertEqual(
            self.spawn.call_args.args,
            (
                "/usr/bin/cloudflared",
                "tunnel",
                "--no-autoupdate",
                "--url",
                "http://127.0.0.1:8765",
            ),
        )
        self.tunnel.wait_reachable.assert_awaited_once()
        await self.tunnel.close()
        self.assertTrue(self.process.terminated)
        self.assertTrue(self.tunnel.reader.done())

    async def test_startup_timeout_terminates_process(self):
        with self.assertRaisesRegex(ValueError, "Timed out"):
            await self.tunnel.start()
        self.assertTrue(self.process.terminated)

    async def test_early_exit_surfaces_actionable_error(self):
        self.process.stdout.feed_data(
            b"2026 ERR failed to request quick Tunnel: dial tcp refused\n"
        )
        self.process.stdout.feed_eof()
        with self.assertRaisesRegex(ValueError, "refused"):
            await self.tunnel.start()
        self.assertTrue(self.process.terminated)

    async def test_cancelled_start_does_not_orphan_process(self):
        task = asyncio.create_task(self.tunnel.start())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.process.terminated)

    async def test_connection_loss_and_domain_rotation_are_detected(self):
        self.process.stdout.feed_data(registered())
        await self.tunnel.start()
        self.process.stdout.feed_data(registered("other.trycloudflare.com"))
        with self.assertRaisesRegex(ValueError, "expired or changed"):
            await self.tunnel.watch()
        await self.tunnel.close()

    async def test_exit_after_registration_is_not_silently_ignored(self):
        self.process.stdout.feed_data(registered())
        await self.tunnel.start()
        self.process.stdout.feed_eof()
        with self.assertRaisesRegex(ValueError, "disconnected"):
            await self.tunnel.watch()
        await self.tunnel.close()

    async def test_missing_cloudflared_explains_install(self):
        with patch(
            "ygaraty.backends.webgpu.sharing.shutil.which", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "cloudflared.*--lan"):
                await self.tunnel.start()
        self.assertFalse(self.directory.exists())
        self.spawn.assert_not_called()

    async def test_unreachable_tunnel_still_returns_with_a_hint(self):
        self.tunnel.wait_reachable = AsyncMock(return_value=False)
        self.process.stdout.feed_data(registered())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            url = await self.tunnel.start()
        self.assertTrue(url.endswith(".trycloudflare.com"))
        self.assertIn("does not answer yet", output.getvalue())
        await self.tunnel.close()

    async def test_stubborn_process_is_killed(self):
        self.process.stdout.feed_data(registered())
        await self.tunnel.start()
        self.process.wait = AsyncMock(side_effect=[TimeoutError(), 0])
        await self.tunnel.close()
        self.assertTrue(self.process.killed)

    def test_only_quick_tunnel_addresses_are_accepted(self):
        for line in (
            b"INF Requesting new quick Tunnel on trycloudflare.com...",
            b"https://api.trycloudflare.com/tunnel",
            b"https://evil.com/a.trycloudflare.com",
            b"https://a.trycloudflare.com.evil.com",
            b"{}",
        ):
            with self.subTest(line=line):
                self.assertIsNone(PublicTunnel.registered_url(line))
        self.assertEqual(
            PublicTunnel.registered_url(registered()),
            "https://demo-app-name.trycloudflare.com",
        )


class Server:
    started = False
    should_exit = False

    async def serve(self):
        self.started = True
        while not self.should_exit:
            await asyncio.sleep(0)


class SharingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = Server()
        self.args = SimpleNamespace(
            share=True,
            host="127.0.0.1",
            port=8765,
            keys=Path("workspace/keys"),
            public_url=None,
        )
        self.app = SimpleNamespace(state=SimpleNamespace(session=object()))
        self.tunnel = SimpleNamespace(
            start=AsyncMock(return_value="https://test-app.trycloudflare.com"),
            close=AsyncMock(),
            watch=AsyncMock(),
        )
        self.factory = self.enterContext(
            patch(
                "ygaraty.backends.webgpu.sharing.PublicTunnel", return_value=self.tunnel
            )
        )
        self.announce = self.enterContext(
            patch("ygaraty.backends.webgpu.launch.announce", new=AsyncMock())
        )
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    async def test_server_binds_before_tunnel_and_links_use_public_origin(
        self,
    ):
        async def start():
            self.assertTrue(self.server.started)
            return "https://test-app.trycloudflare.com"

        self.tunnel.start.side_effect = start
        await serve_launch(self.server, self.app, self.args)
        self.assertEqual(self.args.public_url, "https://test-app.trycloudflare.com")
        self.announce.assert_awaited_once_with(
            self.app.state.session, self.args
        )
        self.tunnel.close.assert_awaited_once()
        self.assertTrue(self.server.should_exit)

    async def test_tunnel_failure_stops_server_without_announcing_bad_links(
        self,
    ):
        self.tunnel.start.side_effect = ValueError("tunnel failed")
        with self.assertRaisesRegex(ValueError, "tunnel failed"):
            await serve_launch(self.server, self.app, self.args)
        self.announce.assert_not_awaited()
        self.tunnel.close.assert_awaited_once()
        self.assertTrue(self.server.should_exit)

    async def test_local_or_explicit_url_does_not_contact_tunnel_service(self):
        self.args.share = False
        await serve_launch(self.server, self.app, self.args)
        self.factory.assert_not_called()
        self.announce.assert_awaited_once()

    async def test_failed_bind_never_exposes_another_service(self):
        self.server.serve = AsyncMock(side_effect=SystemExit(1))
        with self.assertRaisesRegex(ValueError, "could not start"):
            await serve_launch(self.server, self.app, self.args)
        self.factory.assert_not_called()

    async def test_shutdown_while_tunnel_is_starting_cancels_it(self):
        async def start():
            self.server.should_exit = True
            await asyncio.Event().wait()

        self.tunnel.start.side_effect = start
        await serve_launch(self.server, self.app, self.args)
        self.tunnel.close.assert_awaited_once()
        self.announce.assert_not_awaited()
