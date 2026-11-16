"""ASGI requests against fake engines only: no sockets, model or GPU."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from session_fakes import CONTRIBUTOR, OWNER, setup
from ygaraty.backend_contracts.models import Route

from ygaraty.backends.webgpu.configuration import WebGPUConfig
from ygaraty.backends.webgpu.factory import WebGPUFactory
from ygaraty.backends.webgpu.server import create_app


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = setup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bundle.json"
        self.path.write_bytes(self.f.raw)
        self.loader = Mock(return_value=self.f.engine)
        self.factory = WebGPUFactory(
            self.path, Path(self.temp.name), engine_loader=self.loader
        )
        self.app = create_app(
            self.factory, self.f.spec, self.f.services, self.f.access
        )
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)

    async def chat(self, **data):
        return await self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {OWNER}"},
            json={"messages": [{"role": "user", "content": "Hi"}], **data},
        )

    async def test_authentication_and_static_pages(self):
        self.assertEqual((await self.client.get("/health")).status_code, 200)
        page = await self.client.get("/")
        self.assertEqual(page.status_code, 200)
        # contributor_network.mjs renders WebRTC links into this list.
        self.assertIn(
            'id="contributor-network"',
            page.text.split("data-network-links")[0],
        )
        self.assertEqual((await self.client.get("/chat")).status_code, 200)
        self.assertEqual(
            (await self.client.get("/public_chat.mjs")).status_code, 200
        )
        self.assertEqual(
            (await self.client.get("/server.py")).status_code, 404
        )
        # Pages and modules are gzip-compressed for slow public tunnels.
        module = await self.client.get(
            "/public_chat.mjs", headers={"Accept-Encoding": "gzip"}
        )
        self.assertEqual(module.headers["content-encoding"], "gzip")
        self.assertIn("Accept-Encoding", module.headers["vary"])
        plain = await self.client.get(
            "/public_chat.mjs", headers={"Accept-Encoding": "identity"}
        )
        self.assertNotIn("content-encoding", plain.headers)
        self.assertEqual(module.content, plain.content)  # decoded by client
        self.assertEqual(
            (await self.client.get("/v1/session")).status_code, 401
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {CONTRIBUTOR}"},
                    json={},
                )
            ).status_code,
            401,
        )

    async def test_model_discovery_requires_owner_even_while_waiting(self):
        self.f.engine.state = "waiting"
        for key in ("", CONTRIBUTOR):
            response = await self.client.get(
                "/v1/models", headers={"Authorization": f"Bearer {key}"}
            )
            self.assertEqual(response.status_code, 401)
        response = await self.client.get(
            "/v1/models", headers={"Authorization": f"Bearer {OWNER}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(
            response.json()["data"][0]["id"], self.f.spec.model.model_id
        )
        self.assertFalse(self.f.engine.calls)

    async def test_session_reports_route_for_chat_stats(self):
        # The chat page appends this to each answer's stats line.
        self.f.engine.status = lambda: {"network": "relay", "stages": []}
        response = await self.client.get(
            "/v1/session", headers={"Authorization": f"Bearer {OWNER}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["network"], "relay")
        self.f.engine.status = None
        response = await self.client.get(
            "/v1/session", headers={"Authorization": f"Bearer {OWNER}"}
        )
        self.assertIsNone(response.json()["network"])

    async def test_full_response_and_stream_usage(self):
        response = await self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"], "hello world"
        )
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 200)
        values = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertEqual(values[-1]["usage"]["completion_tokens"], 2)
        self.assertIn("data: [DONE]", response.text)
        self.assertFalse(self.app.state.session.handles)

    async def test_admission_fails_before_stream_headers(self):
        self.f.engine.state = "waiting"
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("text/event-stream", response.headers["content-type"])
        self.assertFalse(self.f.engine.calls)

    async def test_request_validation_and_no_shared_default_conversation(self):
        for data in (
            {"temperature": 1},
            {"n": True},
            {"extra": 1},
            {"max_tokens": -1},
            {"model": "other"},
        ):
            response = await self.chat(**data)
            self.assertEqual(response.status_code, 400)
        await self.chat()
        await self.chat()
        self.assertNotEqual(
            self.f.engine.calls[-1]["conversation_id"],
            self.f.engine.calls[-2]["conversation_id"],
        )

    async def test_worker_scope_cannot_bypass_chat_admission(self):
        headers = {"X-Key": CONTRIBUTOR}
        self.assertEqual(
            (
                await self.client.post(
                    "/api/chat", headers=headers, json={"message": "bypass"}
                )
            ).status_code,
            404,
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/api/join", headers={"X-Key": OWNER}, json={}
                )
            ).status_code,
            401,
        )
        response = await self.client.post(
            "/api/join", headers=headers, json={}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["accepted"])
        self.assertFalse(self.f.engine.calls)

    async def test_control_executor_is_available_when_data_lane_is_full(self):
        transport = self.app.state.transport
        acquired = 0
        while transport.slots.acquire(blocking=False):
            acquired += 1
        try:
            response = await asyncio.wait_for(
                self.client.get("/api/status", headers={"X-Key": CONTRIBUTOR}),
                1,
            )
            self.assertEqual(response.status_code, 200)
        finally:
            for _ in range(acquired):
                transport.slots.release()

    async def test_factory_requires_turn_before_private_route_acquisition(
        self,
    ):
        # Direct WebRTC may run without TURN on a local network; the
        # TURN-only route cannot.
        from dataclasses import replace

        config = WebGPUConfig.from_contract(self.f.spec.configuration)
        spec = replace(
            self.f.spec,
            configuration=replace(
                config, route=Route.PRIVATE_TURN
            ).to_contract(),
        )
        self.loader.reset_mock()
        with self.assertRaises(ValueError):
            await self.factory.open(spec, self.f.services)
        self.loader.assert_not_called()

    async def test_open_failure_does_not_start_transport(self):
        loader = Mock(side_effect=ValueError("bad fixture"))
        factory = WebGPUFactory(
            self.path, Path(self.temp.name), engine_loader=loader
        )
        transport = Mock()
        app = create_app(
            factory,
            self.f.spec,
            self.f.services,
            self.f.access,
            transport_factory=transport,
        )
        with self.assertRaises(ValueError):
            async with app.router.lifespan_context(app):
                self.fail("Must not start")
        transport.assert_not_called()

    async def test_failed_announcement_still_closes_session_and_transport(
        self,
    ):
        session = Mock(close=AsyncMock())
        factory = Mock(open=AsyncMock(return_value=session))
        access, transport = Mock(), Mock()
        app = create_app(
            factory,
            self.f.spec,
            self.f.services,
            access,
            transport_factory=lambda: transport,
            launch_args=Mock(),
        )
        with patch(
            "ygaraty.backends.webgpu.launch.announce",
            new=AsyncMock(side_effect=RuntimeError("announcement failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "announcement failed"):
                async with app.router.lifespan_context(app):
                    await asyncio.sleep(0)
        session.close.assert_awaited_once()
        transport.close.assert_called_once()

    async def test_announcement_cancelled_on_shutdown(self):
        session = Mock(close=AsyncMock())
        factory = Mock(open=AsyncMock(return_value=session))
        stopped = asyncio.Event()

        async def monitor(*args):
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        app = create_app(
            factory,
            self.f.spec,
            self.f.services,
            Mock(),
            transport_factory=Mock,
            launch_args=Mock(),
        )
        with patch("ygaraty.backends.webgpu.launch.announce", side_effect=monitor):
            async with app.router.lifespan_context(app):
                await asyncio.sleep(0)
        self.assertTrue(stopped.is_set())
        session.close.assert_awaited_once()

    async def test_asgi_disconnect_cancels_stream_without_leaking_slot(self):
        self.f.engine.mode = "blocked"
        sent = asyncio.Event()
        body = json.dumps(
            {"messages": [{"role": "user", "content": "Hi"}], "stream": True}
        ).encode()
        read = False

        async def receive():
            nonlocal read
            if not read:
                read = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            await sent.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                sent.set()

        scope = dict(
            type="http",
            asgi={"version": "3.0", "spec_version": "2.3"},
            method="POST",
            scheme="http",
            path="/v1/chat/completions",
            raw_path=b"/v1/chat/completions",
            query_string=b"",
            server=("test", 80),
            client=("test", 1),
            http_version="1.1",
            headers=[
                (b"authorization", f"Bearer {OWNER}".encode()),
                (b"content-type", b"application/json"),
            ],
        )
        await asyncio.wait_for(self.app(scope, receive, send), 1)
        for _ in range(100):
            if not self.app.state.session.handles:
                break
            await asyncio.sleep(0.005)
        self.assertFalse(self.app.state.session.handles)
