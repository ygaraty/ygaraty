"""Authenticated WebRTC endpoints on the real ASGI host/protocol, fake GPU."""

import asyncio
import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import httpx
from session_fakes import CONTRIBUTOR, OWNER, setup
from ygaraty.backend_contracts.models import Route

from ygaraty.backends.webgpu._legacy.qwen_protocol import SessionProtocol as Protocol
from ygaraty.backends.webgpu.configuration import WebGPUConfig
from ygaraty.backends.webgpu.factory import WebGPUFactory
from ygaraty.backends.webgpu.server import create_app
from ygaraty.backends.webgpu.turn import TurnConfig


class TurnHostTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = setup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bundle.json"
        self.path.write_bytes(self.f.raw)
        config = WebGPUConfig.from_contract(self.f.spec.configuration)
        self.spec = replace(
            self.f.spec,
            configuration=replace(
                config, route=Route.PRIVATE_TURN
            ).to_contract(),
        )
        self.turn = TurnConfig("turn.example.org", "s" * 40)
        e = self.f.engine
        e.lock = threading.RLock()
        e.args = SimpleNamespace(route="webrtc", ice_policy="relay")
        e.owner = ["a", "b"]
        e.stages = [{}, {}]
        e.epoch = 1
        e.contributors = {"a": {"stage": 0}, "b": {"stage": 1}}
        self.roster = SimpleNamespace(
            lock=threading.RLock(),
            records={"a": {"state": "active"}, "b": {"state": "preparing"}},
            touch=Mock(),
        )
        e.contributor_roster = lambda: self.roster
        e.expire_contributors = Mock()
        e.ice_servers = lambda session, slot: self.turn.credentials(
            session, slot, now=1000
        )
        e.signal_boxes = {cid: queue.Queue(256) for cid in e.owner}
        e.signal_buckets = {}
        e.signal = MethodType(Protocol.signal, e)
        e.signals_for = MethodType(Protocol.signals_for, e)
        e.join = Mock(side_effect=[{"id": "a"}, {"id": "b"}])
        self.loader = Mock(return_value=e)
        factory = WebGPUFactory(
            self.path,
            Path(self.temp.name),
            engine_loader=self.loader,
            turn=self.turn,
        )
        self.app = create_app(
            factory, self.spec, self.f.services, self.f.access
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        self.keys = {}
        for cid in ("a", "b"):
            joined = await self.client.post(
                "/api/join", json={}, headers={"X-Key": CONTRIBUTOR}
            )
            self.assertEqual(joined.status_code, 200)
            self.keys[cid] = joined.json()["worker_key"]

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)

    def headers(self, cid="a"):
        return {"X-Key": CONTRIBUTOR, "X-Worker-Key": self.keys[cid]}

    async def test_ice_requires_instance_credential_and_current_assignment(
        self,
    ):
        for headers in (
            {},
            {"X-Key": OWNER},
            {"X-Key": CONTRIBUTOR},
            self.headers("b"),
        ):
            response = await self.client.get("/api/ice/a", headers=headers)
            self.assertIn(response.status_code, (401, 403))
        response = await self.client.get("/api/ice/a", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertTrue(
            response.json()["ice"][1]["username"].endswith(":epoch:0")
        )
        self.assertNotIn(self.turn.secret, response.text)
        self.roster.records["a"]["state"] = "former"
        response = await self.client.get("/api/ice/a", headers=self.headers())
        self.assertEqual(response.status_code, 403)
        self.roster.records["a"]["state"] = "active"
        self.f.engine.owner[0] = "replacement"
        response = await self.client.get("/api/ice/a", headers=self.headers())
        self.assertEqual(response.status_code, 403)

    async def test_piece_downloads_need_only_the_contributor_key(self):
        # download/<piece> names a weight piece, not a contributor instance.
        response = await self.client.get("/api/download/piece-0")
        self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/api/download/piece-0", headers={"X-Key": CONTRIBUTOR}
        )
        self.assertNotIn(response.status_code, (401, 403))

    async def test_signaling_scope_and_private_candidate_filter(
        self,
    ):
        body = {"to": 1, "data": {"sdp": {"type": "offer", "sdp": "v=0\r\n"}}}
        response = await self.client.post(
            "/api/signal/a", headers=self.headers(), json=body
        )
        self.assertEqual(response.status_code, 200)
        response = await asyncio.wait_for(
            self.client.get("/api/signal/b", headers=self.headers("b")), 2
        )
        self.assertEqual(response.json()[0]["sender"], 0)
        body["to"] = 0
        response = await self.client.post(
            "/api/signal/a", headers=self.headers(), json=body
        )
        self.assertEqual(response.status_code, 403)
        body = {
            "to": 1,
            "data": {
                "candidate": {
                    "candidate": (
                        "candidate:1 1 UDP 123 192.0.2.1 12345 typ host"
                    ),
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                }
            },
        }
        response = await self.client.post(
            "/api/signal/a", headers=self.headers(), json=body
        )
        self.assertEqual(response.status_code, 403)
        self.roster.records["a"]["state"] = "former"
        response = await self.client.post(
            "/api/signal/a", headers=self.headers(), json={}
        )
        self.assertEqual(response.status_code, 403)

    async def test_chat_api_and_route_capabilities_with_turn(self):
        self.assertIs(self.loader.call_args.kwargs["turn"], self.turn)
        response = await self.client.get(
            "/v1/session", headers={"Authorization": f"Bearer {OWNER}"}
        )
        self.assertIn("private_turn", response.text)
        for stream in (False, True):
            response = await self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {OWNER}"},
                json={
                    "model": self.spec.model.model_id,
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": stream,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("[DONE]" if stream else "hello world", response.text)
        page = await self.client.get("/")
        self.assertIn("WebRTC peer links use TURN only", page.text)
