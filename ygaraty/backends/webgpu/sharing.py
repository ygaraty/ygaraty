"""Public HTTPS sharing through a Cloudflare quick tunnel (cloudflared).

`cloudflared tunnel --url http://127.0.0.1:PORT` needs no account: it prints a
random https://*.trycloudflare.com address that forwards to this server until
the process stops. Install: https://github.com/cloudflare/cloudflared
"""

import asyncio
import os
import re
import shutil
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

INSTALL = (
    "Public sharing uses Cloudflare's free quick tunnel and needs "
    "cloudflared: "
    "https://github.com/cloudflare/cloudflared#installing-cloudflared "
    "(e.g. `brew install cloudflared`, `sudo pacman -S cloudflared`, or the "
    "Linux .deb/.rpm from its releases). Or share on your network with --lan."
)
ADDRESS = re.compile(
    rb"https://([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)\.trycloudflare\.com"
    rb"(?![a-z0-9.-])"
)


class PublicTunnel:
    """Own one cloudflared quick tunnel for the life of the launch."""

    def __init__(self, host, port, directory, *, timeout=60, reachable=60):
        self.host = host
        self.port = port
        self.directory = Path(directory)
        self.timeout = timeout
        self.reachable = reachable  # seconds to wait for the name to resolve
        self.process = self.reader = None
        self.url = None
        self.last_error = ""

    def command(self):
        executable = os.environ.get("YGARATY_CLOUDFLARED") or shutil.which(
            "cloudflared"
        )
        if executable is None:
            raise ValueError(INSTALL)
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(self.host, self.host)
        if ":" in host:
            host = f"[{host}]"
        return [
            executable,
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://{host}:{self.port}",
        ]

    @staticmethod
    def registered_url(line):
        """The quick tunnel's address from a cloudflared log line, if any."""
        for match in ADDRESS.finditer(line):
            if match.group(1) != b"api":  # the service it registers with
                return "https://" + match.group(0)[8:].decode()
        return None

    async def wait_reachable(self, url):
        """A new trycloudflare name can take seconds to resolve: give the
        link out once it answers, so browsers don't cache a failed lookup."""

        def probe():
            try:
                with urllib.request.urlopen(url + "/", timeout=5) as response:
                    return response.status < 500
            except urllib.error.HTTPError as error:
                return error.code < 500
            except (OSError, ValueError):
                return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.reachable
        # Looking the name up before Cloudflare publishes it caches "no such
        # host" in this machine's resolver; give it a head start.
        await asyncio.sleep(min(5, self.reachable))
        while loop.time() < deadline:
            if await asyncio.to_thread(probe):
                return True
            await asyncio.sleep(2)
        return False

    async def _read(self, ready):
        while line := await self.process.stdout.readline():
            url = self.registered_url(line)
            if url:
                if self.url is not None and url != self.url:
                    raise ValueError(
                        "Public share link expired or changed. Restart launch "
                        "and share the new link."
                    )
                self.url = url
                if not ready.done():
                    ready.set_result(url)
            elif b" ERR " in line or b"failed" in line.lower():
                self.last_error = line.decode(errors="replace").strip()[:500]
        detail = f" ({self.last_error})" if self.last_error else ""
        raise ValueError(
            "Public tunnel disconnected" + detail + ". "
            "Check your network and retry launch, or use --lan."
        )

    async def start(self):
        command = self.command()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                limit=256 * 1024,
            )
            ready = asyncio.get_running_loop().create_future()
            self.reader = asyncio.create_task(self._read(ready))
            done, _ = await asyncio.wait(
                {ready, self.reader},
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self.reader in done:
                await self.reader
            if ready not in done:
                detail = f" ({self.last_error})" if self.last_error else ""
                raise ValueError(
                    "Timed out creating the Cloudflare tunnel" + detail + ". "
                    "Check your internet connection and retry, or use --lan."
                )
            url = ready.result()
            if not await self.wait_reachable(url):
                print(
                    f"  The tunnel {url} does not answer yet; if the link "
                    "fails, wait a minute and reload it.",
                    flush=True,
                )
            return url
        except BaseException:
            await self.close()
            raise

    async def watch(self):
        await self.reader

    async def close(self):
        if self.process is not None and self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)


async def serve_launch(server, app, args):
    """Publish only after this server has acquired its listening socket."""
    from .launch import announce

    async def run_server():
        try:
            await server.serve()
        except SystemExit as error:
            raise ValueError(
                "Server could not start; check the bind address and port"
            ) from error

    serving = asyncio.create_task(run_server())
    tunnel = None
    tasks = []
    try:
        while not server.started:
            if serving.done():
                await serving
                raise ValueError("Server stopped before startup completed")
            await asyncio.sleep(0.05)
        if args.share:
            print(
                "Creating a public link through a Cloudflare quick tunnel…",
                flush=True,
            )
            tunnel = PublicTunnel(
                args.host, args.port, args.keys.parent / ".tunnel"
            )
            starting = asyncio.create_task(tunnel.start())
            tasks.append(starting)
            done, _ = await asyncio.wait(
                {serving, starting}, return_when=asyncio.FIRST_COMPLETED
            )
            if serving in done:
                await serving
                return
            args.public_url = await starting
            tasks.remove(starting)
            tasks.append(asyncio.create_task(tunnel.watch()))
        tasks.append(asyncio.create_task(announce(app.state.session, args)))
        done, _ = await asyncio.wait(
            {serving, *tasks},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            await task
    finally:
        server.should_exit = True
        try:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if tunnel is not None:
                await tunnel.close()
        finally:
            try:
                await asyncio.wait_for(asyncio.shield(serving), 15)
            except TimeoutError:
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)
