"""User launch flow with synthetic bundles and mocked model preparation."""

import asyncio
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from test_memory import bundle

from ygaraty.backend_contracts.models import SessionState
from ygaraty.backends.webgpu.cli import main
from ygaraty.backends.webgpu.launch import (
    announce,
    connection_details,
    public_origin,
)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "server"
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        self.prepare = self.enterContext(
            patch("ygaraty.backends.webgpu.bootstrap.Bootstrap.prepare")
        )
        self.prepared_bundle = self.root / "prepared.json"
        self.prepared_bundle.write_text(json.dumps(bundle()))
        self.prepare.return_value = SimpleNamespace(
            bundle=self.prepared_bundle
        )
        self.serve = self.enterContext(
            patch("ygaraty.backends.webgpu.cli.serve")
        )

    def command(self, *extra):
        return main(
            [
                "launch",
                "--num-workers",
                "2",
                "--checkpoint",
                str(self.root / "input"),
                "--workspace",
                str(self.workspace),
                "--model",
                "qwen3.5-9b",
                *extra,
            ]
        )

    def test_turn_configuration_selects_webrtc_with_the_same_command(self):
        with patch.dict(
            "os.environ",
            {
                "YGARATY_TURN_HOST": "turn.example.org",
                "YGARATY_TURN_SECRET": "s" * 40,
            },
        ):
            self.command("--local")
        served = self.serve.call_args.args[0]
        self.assertEqual(served.route.value, "direct")
        self.assertEqual(served.turn_config.host, "turn.example.org")
        self.assertNotIn("s" * 40, self.output.getvalue())

    def test_webrtc_without_turn_fails_before_preparation(self):
        # Public links cross networks, so WebRTC needs TURN there.
        for args in (("--route", "webrtc", "--public"), ("--route", "turn")):
            with (
                patch.dict("os.environ", {}, clear=True),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.command(*args)
        self.prepare.assert_not_called()
        self.serve.assert_not_called()

    def test_local_links_use_webrtc_without_turn(self):
        from ygaraty.backend_contracts.models import Route

        with patch.dict("os.environ", {}, clear=True):
            self.command()
            served = self.serve.call_args.args[0]
            self.assertEqual(
                (served.route, served.turn_config), (Route.DIRECT, None)
            )
            self.command("--route", "webrtc")
            self.assertEqual(self.serve.call_args.args[0].route, Route.DIRECT)
            with patch(
                "ygaraty.backends.webgpu.sharing.PublicTunnel.command",
                return_value=["cloudflared"],
            ):
                self.command("--public")
            self.assertEqual(self.serve.call_args.args[0].route, Route.RELAY)

    def test_advanced_parameters_reach_preparation_and_serving(self):
        with patch("ygaraty.backends.webgpu.launch.Bootstrap") as setup:
            setup.return_value.prepare.return_value = SimpleNamespace(
                bundle=self.prepared_bundle
            )
            self.command(
                "--splits",
                "0,16,32",
                "--context",
                "2048",
                "--generate",
                "64",
                "--memory-gb",
                "8",
                "--slots",
                "3",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--public-url",
                "https://ai.example.com",
                "--open-chat",
            )
            profile, _, workspace, gguf, context, generate, host, port = (
                setup.call_args.args
            )
            self.assertEqual(profile.splits, (0, 16, 32))
            self.assertEqual(profile.memory_gb, 8)
            self.assertEqual(
                (workspace, gguf, context, generate, host, port),
                (self.workspace, "model.gguf", 2048, 64, "0.0.0.0", 9000),
            )
        served = self.serve.call_args.args[0]
        self.assertEqual(served.slots, 3)
        self.assertEqual(served.public_url, "https://ai.example.com")
        self.assertTrue(served.open_chat)
        self.assertTrue(served.announce)

    def test_invalid_flags_fail_before_preparation(self):
        for args in (
            ("--splits", "0,16,31"),
            ("--splits", "0,16,8,32"),
            ("--slots", "17"),
            ("--memory-gb", "0"),
            ("--generate", "4096"),
            ("--context", "0"),
            ("--public-url", "http://remote.example"),
        ):
            with (
                self.subTest(args=args),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    self.command(*args)
        self.prepare.assert_not_called()
        self.serve.assert_not_called()
        self.assertFalse(self.workspace.exists())

    def test_custom_model_uses_inspected_geometry(self):
        with (
            patch(
                "ygaraty.backends.webgpu.bundles.inspect_checkpoint",
                return_value={
                    "compatible": True,
                    "family": "dense",
                    "geometry": {"layers": 12},
                },
            ),
            patch("ygaraty.backends.webgpu.launch.Bootstrap") as setup,
        ):
            self.command("--model", "my-model", "--splits", "0,6,12")
            profile = setup.call_args.args[0]
            self.assertEqual(
                (profile.model_id, profile.layers, profile.family),
                ("my-model", 12, "dense"),
            )
            self.assertEqual(profile.splits, (0, 6, 12))

    def test_lan_serves_https_on_the_network_with_local_webrtc(self):
        from ygaraty.backend_contracts.models import Route

        (self.root / "bundle.json").write_text(json.dumps(bundle()))
        cert = (self.root / "cert.pem", self.root / "key.pem")

        def launch(*extra):
            main(
                [
                    "launch",
                    "--num-workers",
                    "2",
                    "--bundle",
                    str(self.root / "bundle.json"),
                    "--workspace",
                    str(self.workspace),
                    "--model",
                    "qwen3.5-9b",
                    *extra,
                ]
            )
            return self.serve.call_args.args[0]

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("ygaraty.backends.webgpu.bundles.validate_bundle"),
            patch(
                "ygaraty.backends.webgpu.lan.lan_address",
                return_value="192.168.2.120",
            ),
            patch(
                "ygaraty.backends.webgpu.lan.lan_certificate",
                return_value=cert,
            ) as make,
        ):
            served = launch("--lan")
            self.assertEqual(served.public_url, "https://192.168.2.120:8765")
            self.assertEqual((served.host, served.tls), ("0.0.0.0", cert))
            self.assertFalse(served.share)
            self.assertEqual(served.route, Route.DIRECT)  # LAN WebRTC
            self.assertEqual(make.call_args.args[1], "192.168.2.120")
            served = launch("--lan", "10.0.0.7", "--port", "9000")
            self.assertEqual(served.public_url, "https://10.0.0.7:9000")
            self.assertIsNone(launch().tls)  # local links: plain HTTP
            for args in (("--lan", "--public"), ("--lan", "127.0.0.1")):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    launch(*args)

    def test_local_is_default_and_public_opts_into_the_tunnel(self):
        self.command()
        self.assertFalse(self.serve.call_args.args[0].share)
        self.command("--local")
        self.assertFalse(self.serve.call_args.args[0].share)
        with patch(
            "ygaraty.backends.webgpu.sharing.PublicTunnel.command",
            return_value=["cloudflared"],
        ):
            self.command("--public")
        self.assertTrue(self.serve.call_args.args[0].share)
        self.command("--public-url", "https://ai.example.com")
        self.assertFalse(self.serve.call_args.args[0].share)
        for conflict in (
            ("--local", "--public-url", "https://ai.example.com"),
            ("--public", "--local"),
            ("--public", "--public-url", "https://ai.example.com"),
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.command(*conflict)

    def test_spaces_presets_select_two_workers_and_conservative_defaults(self):
        from ygaraty.backends.webgpu.downloads import PUBLIC_PRESETS

        for model, workers in (
            (model, workers)
            for model in (
                "qwen3.5-0.8b",
                "s5-nano",
                "qwen3.5-9b",
                "qwen3.8-27b",
            )
            for workers in range(1, 5)
        ):
            with (
                self.subTest(model=model, workers=workers),
                patch(
                    "ygaraty.backends.webgpu.downloads.fetch_bundle",
                    side_effect=lambda url, output, **kw: (
                        kw["on_manifest"](
                            bundle(workers, max_positions=16384)
                        ),
                        self.prepared_bundle,
                    )[1],
                ) as fetch,
                patch("ygaraty.backends.webgpu.bundles.validate_bundle"),
                patch.dict(
                    "os.environ", {"YGARATY_WORKSPACE_ROOT": str(self.root)}
                ),
            ):
                main(
                    [
                        "launch",
                        "--num-workers",
                        str(workers),
                        "--model",
                        model,
                        "--contributors",
                        str(workers),
                        "--local",
                    ]
                )
                self.assertEqual(
                    fetch.call_args.args[0], PUBLIC_PRESETS[model, workers].url
                )
                self.assertIn(
                    f"/{workers}-workers/bundle.json", fetch.call_args.args[0]
                )
                self.assertEqual(
                    fetch.call_args.kwargs["contributors"], workers
                )
                # Contributors fetch weights from Spaces by default.
                self.assertFalse(fetch.call_args.kwargs["weights"])
                self.assertEqual(fetch.call_args.args[1].name, "remote-model")
                served = self.serve.call_args.args[0]
                from ygaraty.backends.webgpu.memory import estimate_memory

                largest = max(
                    report["recommended"]
                    for report in estimate_memory(
                        bundle(workers, max_positions=16384), 16384, 1
                    )
                )
                self.assertEqual(
                    (served.memory_gb, served.context, served.slots),
                    (math.ceil(largest / 1024**3), 16384, 1),
                )
                self.assertEqual(
                    served.keys,
                    self.root / model / f"{workers}-workers" / "keys",
                )
        self.prepare.assert_not_called()

    def test_preset_cannot_silently_select_the_wrong_worker_count(self):
        with (
            patch("ygaraty.backends.webgpu.downloads.fetch_bundle") as fetch,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main(
                [
                    "launch",
                    "--num-workers",
                    "2",
                    "--model",
                    "s5-nano",
                    "--contributors",
                    "5",
                    "--local",
                ]
            )
        fetch.assert_not_called()

    def test_worker_count_is_mandatory_before_any_work(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            main(["launch", "--model", "s5-nano"])
        self.prepare.assert_not_called()
        self.serve.assert_not_called()

    def test_prints_per_worker_and_total_memory_for_selected_config(self):
        self.command("--memory-gb", "3", "--context", "2048", "--slots", "2")
        served = self.serve.call_args.args[0]
        # The estimate reaches the launch screen instead of a wall of text.
        self.assertEqual([r["worker"] for r in served.memory_reports], [1, 2])
        self.assertEqual(served.memory_reports[0]["context"], 2048)
        self.assertNotIn("not measured peaks", self.output.getvalue())
        from ygaraty.backends.webgpu.launch_screen import header

        served.model_id, served.public_url = "model", None
        screen = io.StringIO()
        header(served, {"contribute": "http://x/#key=k"}, out=screen)
        text = screen.getvalue()
        self.assertIn(
            "2 workers · 2,048-token context · 2 chats at a time", text
        )
        self.assertIn("    http://x/#key=k", text)
        self.assertIn("GB of free GPU memory", text)
        self.assertIn("--lan (your network) or --public (internet)", text)

    def test_explicit_split_count_must_match_workers(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.command("--splits", "0,8,16,24,32")
        self.prepare.assert_not_called()

    def test_prepare_only_does_not_launch(self):
        self.command("--prepare-only")
        self.prepare.assert_called_once()
        self.serve.assert_not_called()

    def test_bundle_resume_creates_then_preserves_keys(self):
        (self.root / "bundle.json").write_text(json.dumps(bundle()))
        args = [
            "launch",
            "--num-workers",
            "2",
            "--bundle",
            str(self.root / "bundle.json"),
            "--workspace",
            str(self.workspace),
            "--model",
            "custom",
        ]
        with patch(
            "ygaraty.backends.webgpu.bundles.validate_bundle"
        ) as validate:
            main(args)
            owner = self.workspace / "keys/owner.key"
            initial = owner.read_bytes()
            main(args)
            self.assertEqual(owner.read_bytes(), initial)
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(self.serve.call_args.args[0].model_id, "custom")
            owner.unlink()
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(args)
        self.assertEqual(self.serve.call_count, 2)

    def test_invalid_bundle_never_creates_keys(self):
        with patch(
            "ygaraty.backends.webgpu.bundles.validate_bundle",
            side_effect=ValueError("bad bundle"),
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(
                    [
                        "launch",
                        "--num-workers",
                        "2",
                        "--bundle",
                        "missing.json",
                        "--workspace",
                        str(self.workspace),
                    ]
                )
        self.assertFalse(self.workspace.exists())
        self.serve.assert_not_called()

    def test_links_use_fragments_and_separate_scopes(self):
        keys = self.root / "keys"
        keys.mkdir()
        (keys / "owner.key").write_text("owner+/secret")
        (keys / "contributor.key").write_text("contributor/secret")
        links = connection_details(
            SimpleNamespace(
                keys=keys,
                public_url="https://ai.example.com/",
                host="0.0.0.0",
                port=8765,
            )
        )
        self.assertEqual(links["api"], "https://ai.example.com/v1")
        for name, key in (
            ("chat", "owner+/secret"),
            ("contribute", "contributor/secret"),
        ):
            url = urlsplit(links[name])
            self.assertEqual(url.query, "")
            self.assertEqual(parse_qs(url.fragment)["key"], [key])
        self.assertNotIn("owner", links["contribute"])

    def test_public_url_validation(self):
        for value in (
            "https://x/path",
            "https://x?key=s",
            "https://x#key=s",
            "https://user:password@x",
            "javascript:alert(1)",
            "https://x:bad",
            "http://remote.example",
            "https://x y",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                public_origin(value, "127.0.0.1", 8765)
        self.assertEqual(
            public_origin(None, "0.0.0.0", 8765), "http://127.0.0.1:8765"
        )
        self.assertEqual(public_origin(None, "::1", 8765), "http://[::1]:8765")


class AnnouncementTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_ready_and_contributor_loss_are_reported_once(self):
        session = Mock()
        states = [
            SessionState.UNAVAILABLE,
            SessionState.UNAVAILABLE,
            SessionState.READY,
            SessionState.UNAVAILABLE,
        ]
        session.snapshot.side_effect = [
            SimpleNamespace(state=s) for s in states
        ]
        sleeps = 0

        async def tick(_):
            nonlocal sleeps
            sleeps += 1
            if sleeps > len(states):
                raise asyncio.CancelledError()

        args = SimpleNamespace(
            public_url="https://ai.example.com",
            open_chat=False,
            model_id="model",
            keys=Path("keys"),
        )
        output = io.StringIO()
        with (
            patch(
                "ygaraty.backends.webgpu.launch.asyncio.sleep",
                side_effect=tick,
            ),
            patch(
                "ygaraty.backends.webgpu.launch.connection_details",
                return_value={
                    "chat": "private",
                    "contribute": "share",
                    "api": "api/v1",
                },
            ),
            contextlib.redirect_stdout(output),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await announce(session, args)
        text = output.getvalue()
        # The link: in the header, then again after the loss.
        self.assertEqual(text.count("    share"), 2)
        self.assertEqual(text.count("✓ Ready in"), 1)
        self.assertEqual(text.count("A worker left"), 1)


class LoadingProgressTests(unittest.TestCase):
    @staticmethod
    def status(loaded_a, state_b="preparing"):
        return {
            "roster": {
                "required": 2,
                "contributors": [
                    {
                        "id": "a",
                        "state": "preparing",
                        "assignment": {"stage": "A"},
                        "progress": {
                            "loaded": loaded_a,
                            "total": 1_600_000_000,
                            "phase": "loading",
                        },
                    },
                    {
                        "id": "b",
                        "state": state_b,
                        "assignment": {"stage": "B"},
                    },
                ],
            }
        }

    def test_summarizes_stages_rate_and_throttles(self):
        from ygaraty.backends.webgpu.launch import LoadingProgress

        now = [0.0]
        progress = LoadingProgress(interval=10, clock=lambda: now[0])
        self.assertIsNone(progress.update(None))
        self.assertEqual(
            progress.update(self.status(400_000_000)),
            "Loading: A 0.40/1.60 GB 25% · B joining (2 of 2 workers joined)",
        )
        now[0] = 5
        self.assertIsNone(progress.update(self.status(500_000_000)))
        now[0] = 10
        self.assertEqual(
            progress.update(self.status(800_000_000, "active")),
            "Loading: A 0.80/1.60 GB 50% 40.0 MB/s · B ready "
            "(2 of 2 workers joined)",
        )
        self.assertIsNone(
            LoadingProgress().update({"roster": {"contributors": []}})
        )


class RequestLogTests(unittest.TestCase):
    def test_launch_console_hides_requests_and_session_events(self):
        from ygaraty.backends.webgpu.cli import serve

        args = SimpleNamespace(
            host="127.0.0.1",
            port=1,
            announce=True,
            request_log=False,
            slots=1,
        )
        configs = []

        class Config:
            def __init__(self, app, **options):
                configs.append(options)

        class Server:
            def __init__(self, config):
                self.config = config

        uvicorn = SimpleNamespace(Config=Config, Server=Server, run=Mock())
        for request_log in (False, True):
            args.request_log = request_log
            with (
                patch.dict("sys.modules", {"uvicorn": uvicorn}),
                patch("ygaraty.backends.webgpu.cli.session_spec"),
                patch("ygaraty.backends.webgpu.cli.LocalAccess"),
                patch("ygaraty.backends.webgpu.cli.WebGPUFactory"),
                patch("ygaraty.backends.webgpu.server.create_app"),
                patch(
                    "ygaraty.backends.webgpu.sharing.serve_launch", new=Mock()
                ),
                patch("ygaraty.backends.webgpu.cli.asyncio", create=True),
                patch("asyncio.run"),
                patch.object(Path, "read_text", return_value="key"),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                args.keys = Path("keys")
                args.state_dir, args.bundle = Path("state"), Path("b.json")
                serve(args)
            self.assertEqual(
                (configs[-1]["access_log"], configs[-1]["log_level"]),
                (True, "info") if request_log else (False, "warning"),
            )
            self.assertEqual("Session" in output.getvalue(), request_log)


if __name__ == "__main__":
    unittest.main()
