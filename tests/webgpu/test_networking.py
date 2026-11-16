"""Public route selection and factory wiring without network or GPU."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_preparation import inputs
from ygaraty.backend_contracts.models import Route

from ygaraty.backends.webgpu.configuration import WebGPUConfig
from ygaraty.backends.webgpu.factory import load_engine
from ygaraty.backends.webgpu.networking import add_network_arguments, resolve_network
from ygaraty.backends.webgpu.turn import TurnConfig


class NetworkingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict("os.environ", {}, clear=True))
        self.parser = argparse.ArgumentParser()
        add_network_arguments(self.parser)

    def test_defaults_explicit_routes_and_missing_configuration(self):
        self.assertEqual(
            resolve_network(self.parser.parse_args([])), (Route.RELAY, None)
        )
        for route in ("webrtc", "turn"):
            with self.assertRaisesRegex(ValueError, "requires"):
                resolve_network(self.parser.parse_args(["--route", route]))
        # On the local network WebRTC uses host candidates, no TURN.
        local = dict(local_network=True)
        self.assertEqual(
            resolve_network(self.parser.parse_args([]), **local),
            (Route.DIRECT, None),
        )
        self.assertEqual(
            resolve_network(
                self.parser.parse_args(["--route", "webrtc"]), **local
            ),
            (Route.DIRECT, None),
        )
        with self.assertRaisesRegex(ValueError, "TURN-only links requires"):
            resolve_network(
                self.parser.parse_args(["--route", "turn"]), **local
            )
        with patch.dict(
            "os.environ", {"YGARATY_TURN_HOST": "turn.example.org"}
        ):
            with self.assertRaisesRegex(ValueError, "both"):
                resolve_network(self.parser.parse_args([]))
            self.assertEqual(
                resolve_network(self.parser.parse_args(["--route", "relay"])),
                (Route.RELAY, None),
            )

    def test_secret_file_and_environment_route_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory) / "turn.secret"
            secret.write_text("s" * 40 + "\n")
            for route, expected in [
                ("auto", Route.DIRECT),
                ("webrtc", Route.DIRECT),
                ("turn", Route.PRIVATE_TURN),
            ]:
                args = self.parser.parse_args(
                    [
                        "--route",
                        route,
                        "--turn-host",
                        "turn.example.org",
                        "--turn-secret-file",
                        str(secret),
                    ]
                )
                chosen, config = resolve_network(args)
                self.assertEqual(chosen, expected)
                self.assertEqual(config.secret, "s" * 40)

    def test_factory_sets_engine_route_and_keeps_http_chat(self):
        turn = TurnConfig("turn.example.org", "s" * 40)
        for route, legacy in [
            (Route.RELAY, "relay"),
            (Route.DIRECT, "webrtc"),
            (Route.PRIVATE_TURN, "private"),
        ]:
            spec, _ = inputs(route=route)
            config = WebGPUConfig.from_contract(spec.configuration)
            captured = {}

            def initialize(engine, *args, **kwargs):
                captured.update(kwargs)
                engine.stages = [{"fixture": {"chat_direct": True}}]

            with patch(
                "ygaraty.backends.webgpu._legacy.prepared.PreparedSession.__init__",
                initialize,
            ):
                engine = load_engine(
                    Path("bundle"), Path("out"), config, {}, turn=turn
                )
            self.assertEqual(captured["route"], legacy)
            self.assertIs(captured["turn"], turn)
            self.assertFalse(engine.stages[0]["fixture"]["chat_direct"])
            self.assertNotIn(
                turn.secret.encode(), config.to_contract().payload
            )
