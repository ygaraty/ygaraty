"""Single-session public ASGI host; no sockets are opened by create_app."""

import asyncio
import gzip
import hashlib
import hmac
import secrets
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from ygaraty.backend_contracts.chat import BackendError
from ygaraty.backend_contracts.lifecycle import CancelReason
from ygaraty.backend_contracts.models import Route

from .configuration import WebGPUConfig
from .http_api import completion, error_status, read_chat, sse
from .session_view import workers_view

BROWSER = Path(__file__).with_name("browser")


@lru_cache(maxsize=128)
def gzipped(path: str, modified_ns: int) -> bytes:
    """Compressed page asset, cached until the file changes."""
    return gzip.compress(Path(path).read_bytes(), 9, mtime=0)


def page_asset(request: Request, path: Path, media_type: str):
    """Serve pages and modules gzip-compressed when the browser accepts it.

    Contributor pages load ~385 KiB of modules; compressed they are ~105 KiB,
    which matters most through slow public tunnels.
    """
    headers = {"Vary": "Accept-Encoding", "Cache-Control": "no-cache"}
    accepted = request.headers.get("accept-encoding", "").lower()
    if "gzip" in accepted:
        body = gzipped(str(path), path.stat().st_mtime_ns)
        return Response(
            body,
            media_type=media_type,
            headers={**headers, "Content-Encoding": "gzip"},
        )
    return FileResponse(path, media_type=media_type, headers=headers)


WORKER_METHODS = {
    "join",
    "ready",
    "heartbeat",
    "leave",
    "host",
    "error",
    "blob",
    "selftest",
    "send",
    "done",
    "route",
    "signal",
}


def bearer(request):
    authorization = request.headers.get("authorization", "")
    return authorization[7:] if authorization.startswith("Bearer ") else ""


def create_app(
    factory,
    spec,
    services,
    access,
    *,
    transport_factory=None,
    launch_args=None,
    open_status=False,
):
    """open_status: serve /v1/status without a key (default: owner or
    contributor key required)."""
    config = WebGPUConfig.from_contract(spec.configuration)
    direct = config.route != Route.RELAY
    worker_secret = secrets.token_bytes(32)

    def worker_token(cid):
        return hmac.new(
            worker_secret, cid.encode(), hashlib.sha256
        ).hexdigest()

    def joined(body, mime):
        if direct and isinstance(body, dict) and body.get("id"):
            return dict(body, worker_key=worker_token(body["id"]))
        return body

    if transport_factory is None:
        from ._legacy.transport import Transport

        transport_factory = Transport

    @asynccontextmanager
    async def lifespan(app):
        session = await factory.open(spec, services)
        transport = None
        announcement = None
        try:
            access.bind_contributor(session.engine)
            transport = transport_factory()
            app.state.session, app.state.transport = session, transport
            if launch_args is not None:
                from .launch import announce

                announcement = asyncio.create_task(
                    announce(session, launch_args)
                )
            yield
        finally:
            try:
                if announcement is not None:
                    announcement.cancel()
                    with suppress(asyncio.CancelledError):
                        await announcement
            finally:
                try:
                    await session.close()
                finally:
                    if transport is not None:
                        await asyncio.to_thread(transport.close)

    app = FastAPI(
        title="Ygaraty public coordinator",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def _is_owner(token):
        try:
            access.grant(token)
        except PermissionError:
            return False
        return True

    def owner(request):
        try:
            return access.grant(bearer(request))
        except PermissionError:
            raise HTTPException(
                401, "Owner bearer credential required"
            ) from None

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/session")
    async def session_status(request: Request):
        owner(request)
        from .launch import engine_status
        from .terminal_chat import network_summary

        session = app.state.session
        # Route between workers, as the terminal chat shows it after answers.
        return {
            **asdict(session.snapshot()),
            "network": network_summary(engine_status(session)),
            "details": public_status(),
        }

    def public_status():
        from .launch import engine_status
        from .terminal_chat import network_summary

        session = app.state.session
        status = engine_status(session) or {}
        roster = status.get("roster") or {}
        details = session.details()
        engine = getattr(session, "engine", None)
        try:
            now = engine.contributor_roster().clock()
        except Exception:  # fakes and engines without a roster
            now = None
        workers = workers_view(status, now)
        # Joined workers that never reported WebRTC links (a runtime without
        # WebRTC) all use the relay, whatever the route allows.
        joined = [w for w in workers if w.get("route")]
        if joined and all(
            w["route"].startswith("relay (no WebRTC") for w in joined
        ):
            network = "relay (workers reported no WebRTC links)"
        elif roster.get("required") == 1:
            network = network_summary(status) or "one worker, no links"
        else:
            network = network_summary(status)
        details.update(
            workers=workers,
            state=session.snapshot().state.value,
            workers_required=roster.get("required") or details["workers"],
            workers_ready=sum(
                record.get("state") == "active"
                for record in roster.get("contributors", [])
            ),
            workers_joined=len(roster.get("contributors", [])),
            network=network,
        )
        return details

    @app.get("/v1/status")
    async def status_page_data(request: Request):
        # Model, readiness, workers and request statistics for the session
        # page. Owner or contributor key unless the server was started with
        # an open status page. Never credentials, contributor identities or
        # network addresses.
        token = bearer(request)
        if not open_status and not (
            access.is_contributor(token) or _is_owner(token)
        ):
            raise HTTPException(401, "Owner or contributor key required")
        return JSONResponse(
            public_status(), headers={"Cache-Control": "no-store"}
        )

    @app.get("/v1/session/contributor-key")
    async def contributor_key(request: Request):
        # The owner's session page builds the contributor link from it.
        owner(request)
        credential = getattr(access, "contributor_credential", None)
        if credential is None:
            raise HTTPException(404, "Contributor key not available")
        return JSONResponse(
            {"key": credential()}, headers={"Cache-Control": "no-store"}
        )

    @app.get("/v1/models")
    async def models(request: Request):
        owner(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.model.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        grant = owner(request)
        value, streaming = await read_chat(request, spec.model.model_id)
        try:
            handle = await app.state.session.admit(value, grant)
        except BackendError as error:
            raise HTTPException(
                error_status(error.failure.code), error.failure.message
            ) from None
        if streaming:
            return StreamingResponse(
                sse(handle, value.model_id),
                media_type="text/event-stream",
                headers={
                    "X-Request-ID": handle.request_id,
                    "Cache-Control": "no-store",
                },
            )
        return await completion(handle, value.model_id)

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel(request_id: str, request: Request):
        grant = owner(request)
        handle = app.state.session.handles.get(request_id)
        if handle is None or handle.grant.principal_id != grant.principal_id:
            raise HTTPException(404, "Request not found")
        return asdict(await handle.cancel(CancelReason.CLIENT_REQUEST))

    @app.api_route("/api/{path:path}", methods=["GET", "POST"])
    @app.get("/f/{path:path}")
    async def contributor(path: str, request: Request):
        if not access.is_contributor(request.headers.get("x-key", "")):
            raise HTTPException(401, "Contributor credential required")
        actual = request.url.path
        verb = path.split("/", 1)[0]
        permitted = (
            request.method == "GET"
            and (
                actual.startswith("/f/")
                or verb in {"recv", "status", "signal", "ice", "download"}
            )
        ) or (request.method == "POST" and verb in WORKER_METHODS)
        if not permitted:
            raise HTTPException(404, "Unknown contributor endpoint")
        from ._legacy.qwen_protocol import handler

        engine = app.state.session.engine
        # download/<piece> names a public weight piece, not a contributor:
        # its descriptor needs only the contributor key checked above.
        if (
            direct
            and verb != "download"
            and not actual.startswith("/f/")
            and "/" in path
        ):
            cid = path.split("/")[1]
            token = request.headers.get("x-worker-key", "")
            if not secrets.compare_digest(token, worker_token(cid)):
                raise HTTPException(
                    403, "Contributor instance credential required"
                )
        if verb in {"ice", "signal"}:
            if not direct:
                raise HTTPException(404, "WebRTC is not enabled")
            cid = path.split("/")[1] if "/" in path else ""
            await asyncio.to_thread(engine.expire_contributors)
            roster = engine.contributor_roster()
            with engine.lock, roster.lock:
                record = roster.records.get(cid, {})
                stage = engine.contributors.get(cid, {}).get("stage")
                if (
                    record.get("state") not in {"preparing", "active"}
                    or type(stage) is not int
                    or not 0 <= stage < len(engine.owner)
                    or engine.owner[stage] != cid
                ):
                    raise HTTPException(
                        403, "Contributor no longer holds a stage"
                    )
                if verb == "ice":
                    return JSONResponse(
                        engine.ice_servers(spec.epoch, stage),
                        headers={"Cache-Control": "no-store"},
                    )
        # Contributor scope was checked; legacy authorization stays internal.
        protocol = handler(engine)
        return await app.state.transport.dispatch(
            request,
            protocol,
            transform=joined if actual == "/api/join" else None,
            control=verb not in {"recv", "send", "signal"}
            and not actual.startswith("/f/"),
        )

    @app.get("/")
    async def contribute_page():
        policy = {
            Route.RELAY: "Traffic uses the coordinator relay.",
            Route.DIRECT: "WebRTC may connect directly to other contributors, "
            "revealing your network address to them. TURN and the "
            "coordinator relay provide fallback.",
            Route.PRIVATE_TURN: "WebRTC peer links use TURN only, "
            "keeping your network address hidden from other contributors. The "
            "coordinator relay remains available as fallback.",
        }[config.route]
        page = (BROWSER / "session_contributor.html").read_text()
        page = page.replace("<!-- NETWORK_POLICY -->", policy)
        return HTMLResponse(
            page.replace(
                'data-open-status="false"',
                f'data-open-status="{str(open_status).lower()}"',
            )
        )

    @app.get("/chat")
    async def chat_page(request: Request):
        return page_asset(request, BROWSER / "session_chat.html", "text/html")

    @app.get("/{name}")
    async def asset(name: str, request: Request):
        if not name.endswith(".mjs") or name not in {
            p.name for p in BROWSER.glob("*.mjs")
        }:
            raise HTTPException(404, "Asset not found")
        return page_asset(request, BROWSER / name, "text/javascript")

    return app
