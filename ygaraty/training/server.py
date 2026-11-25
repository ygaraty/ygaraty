"""Small HTTP transport for the public training session (optional server extra)."""

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from .recovery import LIMIT


def create_app(session):
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, Response

    # Request is local to keep the server extra optional at import time.
    @asynccontextmanager
    async def lifespan(app):
        async def monitor():
            while True:
                await asyncio.sleep(1)
                await asyncio.to_thread(session.tick)

        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    browser = (
        Path(__file__).resolve().parents[1] / "backends" / "webgpu" / "browser"
    )

    @app.middleware("http")
    async def policy(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/")
    def page():
        return FileResponse(browser / "training.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        # Flat, public browser assets only; never serve the run directory.
        path = browser / name
        if (
            path.parent != browser
            or path.suffix not in (".mjs", ".js", ".css")
            or not path.is_file()
        ):
            return JSONResponse({"error": "Not found"}, status_code=404)
        return FileResponse(
            path,
            media_type="text/css"
            if path.suffix == ".css"
            else "text/javascript",
        )

    async def body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > LIMIT:
                raise ValueError("Request exceeds 16 MiB")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("Expected JSON object")
        return value

    def token(request):
        return request.headers.get("Authorization", "").removeprefix("Bearer ")

    async def respond(function, *args):
        # Session methods already return JSON-native protocol data. Returning a
        # Response skips FastAPI's recursive object conversion; render in the
        # same worker thread so large plans do not block the event loop.
        def render():
            return JSONResponse(function(*args))

        return await asyncio.to_thread(render)

    @app.post("/api/join")
    async def join(request: Request):
        try:
            data = await body(request)
            return await respond(
                session.join, token(request), data["client_id"]
            )
        except PermissionError as error:
            return JSONResponse({"error": str(error)}, status_code=401)
        except (KeyError, ValueError, TypeError) as error:
            return JSONResponse({"error": str(error)}, status_code=409)

    @app.post("/api/workers/{worker}/{action}")
    async def worker_action(worker: str, action: str, request: Request):
        try:
            auth = token(request)
            if action == "assignment":
                content = await asyncio.to_thread(
                    session.assignment_bytes, worker, auth
                )
                return Response(content, media_type="application/json")
            if action == "result":
                data = await body(request)
                return await respond(session.submit, worker, auth, data)
            if action == "heartbeat":

                def heartbeat():
                    with session.lock:
                        session.authenticate(worker, auth)
                        return {"ok": True}

                return await respond(heartbeat)
            if action == "status":

                def status():
                    with session.lock:
                        session.authenticate(worker, auth)
                        return session.status()

                return await respond(status)
            return JSONResponse({"error": "Not found"}, status_code=404)
        except PermissionError as error:
            return JSONResponse({"error": str(error)}, status_code=401)
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            return JSONResponse({"error": str(error)}, status_code=409)

    return app
