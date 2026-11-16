"""Offline compatibility checks for public coturn credentials."""

import base64
import hashlib
import hmac
import json
import unittest

from ygaraty.backends.webgpu.turn import TurnConfig

SECRET = "s" * 40
TURN = TurnConfig(host="turn.example.org", secret=SECRET, ttl=600)


class TurnCredentialTests(unittest.TestCase):
    def test_credentials_follow_the_turn_rest_scheme_and_expire(self):
        issued = TURN.credentials("abc", 1, now=1000)
        stun, turn = issued["ice"]
        self.assertEqual(stun, {"urls": "stun:turn.example.org:3478"})
        self.assertEqual(turn["username"], "1600:abc:1")
        self.assertEqual(issued["expires"], 1600)
        want = base64.b64encode(
            hmac.new(SECRET.encode(), b"1600:abc:1", hashlib.sha1).digest()
        ).decode()
        self.assertEqual(turn["credential"], want)
        self.assertIn(
            "turns:turn.example.org:5349?transport=tcp", turn["urls"]
        )
        self.assertNotIn(SECRET, json.dumps(issued))
        self.assertNotIn(SECRET, repr(TURN))
        self.assertNotIn("credential", json.dumps(TURN.urls()))

    def test_configuration_from_the_environment(self):
        self.assertIsNone(TurnConfig.from_env({}))
        self.assertIsNone(TurnConfig.from_env({"YGARATY_TURN_HOST": "h"}))
        config = TurnConfig.from_env(
            {
                "YGARATY_TURN_HOST": "h.example",
                "YGARATY_TURN_SECRET": SECRET,
                "YGARATY_TURN_TTL": "120",
                "YGARATY_TURN_TLS_PORT": "",
            }
        )
        self.assertEqual((config.ttl, config.tls_port), (120, None))
        for env in (
            {"YGARATY_TURN_SECRET": "short"},
            {"YGARATY_TURN_TTL": "86400"},
            {"YGARATY_TURN_HOST": "bad host/"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                TurnConfig.from_env(
                    {
                        "YGARATY_TURN_HOST": "h.example",
                        "YGARATY_TURN_SECRET": SECRET,
                        **env,
                    }
                )

    def test_ports_and_host_are_validated(self):
        for kwargs in (
            {"port": 0},
            {"port": 65536},
            {"tls_port": -1},
            {"host": "turn.example.org\n"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TurnConfig(
                    **({"host": "turn.example.org", "secret": SECRET} | kwargs)
                )

    def test_credentials_are_scoped_by_session_slot_and_time(self):
        values = [
            TURN.credentials(session, slot, now=now)["ice"][1]["credential"]
            for session, slot, now in [
                ("a", 0, 100),
                ("b", 0, 100),
                ("a", 1, 100),
                ("a", 0, 101),
            ]
        ]
        self.assertEqual(len(set(values)), 4)
