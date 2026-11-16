"""Validate immutable contract preparation with metadata-only fixtures."""

import hashlib
import json
import unittest
from dataclasses import FrozenInstanceError, replace

from fixtures import bundle
from ygaraty.backend_contracts.models import (
    ModelIdentity,
    Route,
    SessionSpec,
    SessionState,
)

from ygaraty.backends.webgpu import WebGPUConfig, prepare_session
from ygaraty.backends.webgpu.distribution import compile_distribution
from ygaraty.backends.webgpu.operations import legacy_stage


def inputs(data=None, **options):
    data = bundle() if data is None else data
    raw = json.dumps(data).encode()
    config = WebGPUConfig(
        hashlib.sha256(raw).hexdigest(),
        tuple(tuple(stage["layers"]) for stage in data["stages"]),
        128,
        1024,
    )
    config = replace(config, **options)
    spec = SessionSpec(
        "session",
        "epoch",
        ModelIdentity("fixture", "checkpoint", "tokenizer", "template", "f16"),
        config.to_contract(),
    )
    return spec, raw


class PreparationTests(unittest.TestCase):
    def test_preparation_is_immutable_and_does_not_claim_readiness(self):
        spec, raw = inputs()
        result = prepare_session(spec, bundle=raw)
        exported = result.to_dict()
        self.assertEqual(result.snapshot().state, SessionState.PREPARING)
        self.assertIsNone(result.snapshot().capabilities)
        self.assertEqual(
            exported["stages"][0]["fixture"]["max_positions"], 128
        )
        exported["stages"].clear()
        self.assertTrue(result.to_dict()["stages"])
        self.assertEqual(
            result.digest, prepare_session(spec, bundle=raw).digest
        )
        with self.assertRaises(FrozenInstanceError):
            result.spec = spec
        self.assertEqual(
            json.loads(raw)["stages"][0]["fixture"]["max_positions"], 1024
        )

    def test_digest_and_envelope_are_checked_before_compilation(self):
        spec, raw = inputs()
        for payload in (b"{}", bytearray(raw), b"", b"x" * ((16 << 20) + 1)):
            with self.subTest(size=len(payload)):
                with self.assertRaises(ValueError):
                    prepare_session(spec, bundle=payload)

    def test_context_and_weight_budget_rejection(self):
        for options in ({"context_tokens": 2048}, {"memory_budget_bytes": 15}):
            with self.subTest(options=options):
                spec, raw = inputs(**options)
                with self.assertRaises(ValueError):
                    prepare_session(spec, bundle=raw)

    def test_stored_route_capacity_and_assignment_are_validated(self):
        for field, value in (
            ("route", "webrtc"),
            ("conversations", True),
            ("ice", [{"urls": "turn:secret"}]),
            ("max_positions", 128),
            ("max_tokens", 0),
        ):
            with self.subTest(field=field):
                data = bundle()
                data["stages"][0]["fixture"][field] = value
                spec, raw = inputs(data)
                with self.assertRaises(ValueError):
                    prepare_session(spec, bundle=raw)

    def test_current_and_exact_legacy_plan_are_accepted(self):
        for legacy in (False, True):
            data = bundle()
            plan = compile_distribution(data["stages"])
            data["distribution_plan"] = plan
            if legacy:
                data["distribution_plan"] = dict(
                    plan,
                    version=2,
                    stages=[legacy_stage(stage) for stage in plan["stages"]],
                )
            spec, raw = inputs(data)
            result = prepare_session(spec, bundle=raw)
            self.assertEqual(result.to_dict()["distribution_plan"], plan)
            data["distribution_plan"]["feedback"]["destination"] = 99
            spec, raw = inputs(data)
            with self.assertRaisesRegex(ValueError, "Stored distribution"):
                prepare_session(spec, bundle=raw)

    def test_route_and_slot_requests_do_not_activate_a_runtime(self):
        for route in Route:
            spec, raw = inputs(route=route, conversation_slots=16)
            prepared = prepare_session(spec, bundle=raw)
            self.assertEqual(prepared.config.route, route)
            self.assertIsNone(prepared.snapshot().capabilities)
            self.assertEqual(
                prepared.to_dict()["stages"][0]["fixture"]["conversations"], 1
            )


class ConfigurationTests(unittest.TestCase):
    def test_round_trip_and_strict_options(self):
        spec, _ = inputs()
        config = WebGPUConfig.from_contract(spec.configuration)
        self.assertEqual(config.to_contract(), spec.configuration)
        for change in (
            {"context_tokens": True},
            {"conversation_slots": 17},
            {"route": "relay"},
            {"stage_layers": [(0, 1)]},
            {"stage_layers": ((1, 2),)},
            {"stage_layers": ((0, 2), (3, 4))},
            {"bundle_sha256": "bad"},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    replace(config, **change)

    def test_unknown_fields_versions_duplicate_json_and_nonfinite_rejected(
        self,
    ):
        spec, _ = inputs()
        data = json.loads(spec.configuration.payload)
        for changes in (
            {"schema_version": 2},
            {"backend_id": "native"},
            {"payload": json.dumps(data | {"unknown": 1}).encode()},
            {
                "payload": json.dumps(
                    {k: v for k, v in data.items() if k != "route"}
                ).encode()
            },
            {"payload": b'{"x":1,"x":2}'},
            {"payload": b'{"x":NaN}'},
        ):
            with self.subTest(changes=list(changes)):
                with self.assertRaises(ValueError):
                    WebGPUConfig.from_contract(
                        replace(spec.configuration, **changes)
                    )
