"""Workers view, output limit, config, status, models info, exit codes."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fixtures import bundle
from session_fakes import setup
from test_memory import bundle as qwen_bundle

from ygaraty.backends.webgpu.catalog import model_info
from ygaraty.backends.webgpu.cli import main
from ygaraty.backends.webgpu.config import config_defaults, save_config
from ygaraty.backends.webgpu.session_view import workers_text, workers_view
from ygaraty.backends.webgpu.status_command import fetch_status
from ygaraty.backends.webgpu.terminal_chat import bar_text, stats_text


def roster(*records, required=2):
    return dict(required=required, contributors=list(records))


class WorkersViewTests(unittest.TestCase):
    def test_states_heartbeats_and_links_without_identities(self):
        status = dict(
            network="direct",
            roster=roster(
                dict(
                    id="secret-id",
                    state="active",
                    last_seen=90.0,
                    assignment=dict(stage="A"),
                ),
                dict(
                    id="other-id",
                    state="preparing",
                    last_seen=50.0,
                    assignment=dict(stage="B"),
                    progress=dict(loaded=45, total=100),
                ),
            ),
            stages=[
                dict(
                    stage="A",
                    links=[
                        dict(
                            peer=1,
                            state="connected",
                            local="host",
                            remote="srflx",
                            protocol="udp",
                            rttMs=0.4,
                        )
                    ],
                ),
                dict(stage="B", links=[dict(peer=0, state="connecting")]),
            ],
        )
        workers = workers_view(status, now=100.0)
        self.assertNotIn("secret-id", json.dumps(workers))
        a, b = workers
        self.assertEqual(
            (a["state"], a["coordinator"]), ("ready", "connected")
        )
        self.assertEqual(a["route"], "WebRTC")
        self.assertEqual(
            a["links"],
            [
                dict(
                    peer="B",
                    state="connected",
                    path="direct",
                    protocol="udp",
                    rtt_ms=0.4,
                )
            ],
        )
        self.assertEqual((b["state"], b["percent"]), ("downloading", 45.0))
        self.assertEqual(b["coordinator"], "late")  # 50 s since a heartbeat
        self.assertEqual(b["route"], "relay (WebRTC pending)")
        self.assertEqual(workers_text(workers), "A ready, B 45% (late)")

    def test_empty_slots_and_one_worker(self):
        waiting = workers_view(dict(roster=roster(required=2), stages=[]))
        self.assertEqual([w["state"] for w in waiting], ["waiting"] * 2)
        one = workers_view(
            dict(
                roster=roster(
                    dict(state="active", assignment=dict(stage="A")),
                    required=1,
                ),
                stages=[dict(stage="A")],
            ),
            now=1.0,
        )
        self.assertNotIn("links", one[0])  # nothing travels between workers

    def test_terminal_shows_workers(self):
        details = dict(
            model="m",
            state="unavailable",
            workers_required=2,
            workers_ready=1,
            stats={},
            workers=[
                dict(stage="A", state="ready", coordinator="connected"),
                dict(stage="B", state="waiting"),
            ],
        )
        self.assertIn("workers A ready, B waiting", bar_text(details))
        text = stats_text(details)
        self.assertIn("Worker A: ready; coordinator connected", text)
        self.assertIn("Worker B: waiting for someone to open the link", text)


class OutputLimitTests(unittest.TestCase):
    def test_bundle_limit_is_the_default_and_the_maximum(self):
        from ygaraty.backend_contracts.chat import ChatRequest, Message, Role
        from ygaraty.backend_contracts.models import HistoryMode

        f = setup()
        f.engine.max_output = 16

        def arguments(max_tokens):
            request = ChatRequest(
                "r",
                "c",
                "fixture",
                HistoryMode.FULL,
                (Message(Role.USER, "Hi"),),
                max_tokens,
            )
            return f.session._arguments(request, "id")["max_tokens"]

        self.assertEqual(arguments(None), 16)
        self.assertEqual(arguments(8), 8)
        self.assertEqual(arguments(500), 16)
        f.engine.max_output = None  # older engines: the context bounds it
        self.assertEqual(arguments(None), 126)


class ConfigTests(unittest.TestCase):
    def parser(self):
        import argparse

        from ygaraty.backends.webgpu import launch

        commands = argparse.ArgumentParser().add_subparsers()
        return launch.add_parser(commands)

    def test_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            given = SimpleNamespace(
                workspace=Path(tmp),
                model="s5-nano",
                contributors=2,
                slots=1,
                route="relay",
                lan="auto",
                open_status=True,
                public=False,
                bundle=None,
            )
            path = save_config(given, "launch")
            text = path.read_text()
            self.assertIn('model = "s5-nano"', text)
            self.assertIn("num-workers = 2", text)
            self.assertNotIn("public", text.replace("open", ""))
            defaults = config_defaults(path, self.parser())
            self.assertEqual(defaults["contributors"], 2)
            self.assertEqual(defaults["lan"], "auto")
            self.assertTrue(defaults["open_status"])
            (Path(tmp) / "bad.toml").write_text("colour = 1\n")
            with self.assertRaisesRegex(ValueError, "unknown setting"):
                config_defaults(Path(tmp) / "bad.toml", self.parser())
            (Path(tmp) / "bad.toml").write_text('route = "pigeon"\n')
            with self.assertRaisesRegex(ValueError, "must be one of"):
                config_defaults(Path(tmp) / "bad.toml", self.parser())

    def test_launch_reads_the_workspace_config_and_flags_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "ygaraty.toml").write_text(
                'model = "s5-nano"\nnum-workers = 2\nslots = 3\n'
            )
            with (
                patch("ygaraty.backends.webgpu.launch._run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main(["launch", "--workspace", tmp, "--slots", "2"])
            args = run.call_args.args[0]
            self.assertEqual(
                (args.model, args.contributors, args.slots),
                ("s5-nano", 2, 2),
            )


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CommandTests(unittest.TestCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        code = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exit:
                code = exit.code
        return code, out.getvalue(), err.getvalue()

    def test_launch_without_model_lists_supported_models(self):
        code, out, err = self.run_main("launch", "--num-workers", "1")
        self.assertEqual(code, 1)
        self.assertIn("--model is required", err)
        self.assertIn("s5-nano", err)
        code, out, err = self.run_main(
            "launch", "--model", "s5-nano", "--json"
        )
        self.assertEqual(code, 1)
        self.assertEqual(
            json.loads(out)["error"].split(":")[0], "--num-workers is required"
        )

    def test_bootstrap_is_hidden_from_help(self):
        code, out, _ = self.run_main("--help")
        self.assertEqual(code, 0)
        self.assertNotIn("bootstrap", out)
        for name in ("models", "status", "launch", "serve"):
            self.assertIn(name, out)

    def test_status_exit_codes(self):
        def opener(state):
            def open_(request, timeout, **options):
                self.seen = request
                return Response(json.dumps({"state": state}).encode())

            return open_

        from ygaraty.backends.webgpu import status_command

        args = SimpleNamespace(
            url="http://h:1",
            key="k",
            key_file=None,
            workspace=None,
            insecure=False,
            json=True,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(status_command.run(args, opener("ready")), 0)
            self.assertEqual(
                status_command.run(args, opener("unavailable")), 3
            )
        self.assertEqual(self.seen.get_header("Authorization"), "Bearer k")
        self.assertEqual(self.seen.full_url, "http://h:1/v1/status")
        with self.assertRaisesRegex(ValueError, "origin"):
            fetch_status("h:1")

    def test_models_info_from_the_manifests(self):
        data = qwen_bundle(2, max_positions=16384)  # the v2 bundles

        def opener(url):
            self.assertTrue(url.endswith("bundle.json"))
            return Response(json.dumps(data).encode())

        info = model_info("s5-nano", opener=opener)
        self.assertEqual(
            [row["workers"] for row in info["configurations"]], [1, 2, 3, 4]
        )
        row = info["configurations"][0]
        self.assertEqual(len(row["per_worker"]), 2)
        self.assertGreater(row["per_worker"][0]["recommended_gpu_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "Supported models"):
            model_info("nonexistent", opener=opener)

    def test_serve_downloads_a_supported_model_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = bundle()
            data["tokenizer"] = "tokenizer.json"
            data["files"]["synthetic-weight"]["sha256"] = "a" * 64
            (path / "tokenizer.json").write_text("{}")
            (path / "bundle.json").write_text(json.dumps(data))
            with (
                patch.dict("os.environ", {"YGARATY_WORKSPACE_ROOT": tmp}),
                patch(
                    "ygaraty.backends.webgpu.downloads.fetch_workspace_bundle",
                    return_value=path / "bundle.json",
                ) as fetch,
                patch("uvicorn.run"),
            ):
                code, out, err = self.run_main(
                    "serve",
                    "--model",
                    "s5-nano",
                    "--num-workers",
                    str(len(data["stages"])),
                    "--route",
                    "relay",
                    "--save-config",
                )
            self.assertEqual(code, 0, err)
            self.assertIn("s5-nano/prepared/v2/", fetch.call_args.args[0])
            saved = path / "s5-nano" / f"{len(data['stages'])}-workers"
            text = (saved / "ygaraty.toml").read_text()
            self.assertIn('model = "s5-nano"', text)
            self.assertNotIn("bundle =", text)  # the preset, not the cache


if __name__ == "__main__":
    unittest.main()
