"""Run one headless-Chromium WebGPU page over binary files (DO-36).

`webgpu_parity.run_browser` sends its fixture as JSON lists, which cannot carry
hundreds of megabytes of packed weights. This runner serves named files as raw
bytes (`GET /f/<name>`, from disk without copying them into memory) and collects
the page's results: `POST /result` (one JSON object, which ends the run),
`POST /blob/<name>` (raw bytes, saved as `<output>/<name>`) and `POST /log`
(progress lines, appended to `<output>/page.log`).

The page module and the modules it imports come from `ygaraty/backends/webgpu/browser`.

With `memory=True` a thread samples the Chromium process tree every
`interval` seconds (Linux): each process's RSS and PSS by Chromium process
type, and the GPU memory the kernel driver attributes to the browser's DRM
clients (`drm-total-vram`/`-gtt` in /proc/<pid>/fdinfo, deduplicated by
client id), plus the device-wide amdgpu counters. Samples go to
`<output>/memory.jsonl`; the result gets a `memory` summary with the peaks
and the values at each `phase <name>` line the page logs.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ygaraty_webgpu_research.host import component as _host_component

PROJECT_ROOT = _host_component("paths").PROJECT_ROOT

STATIC = _host_component("browser_assets").BROWSER_ROOT
GPU_FLAGS = (  # as factorization_webgpu: the machine's GPU through Vulkan
    "--enable-features=Vulkan",
    "--use-webgpu-adapter=default",
    "--ignore-gpu-blocklist",
    "--enable-gpu",
    "--use-gl=angle",
)
# YGARATY_CHROME_X11=1: a regular (not headless) Chrome on an X display, e.g.
# under xvfb-run. On an NVIDIA server headless Chrome cannot start Vulkan (the
# driver has no VK_EXT_headless_surface), falls back to software GL and then
# offers WebGPU only on SwiftShader (DO-36, Vast.ai RTX 3060); with X11 and
# ANGLE on Vulkan it gets the NVIDIA adapter.
X11 = os.environ.get("YGARATY_CHROME_X11") == "1"
HEADLESS = () if X11 else ("--headless=new",)
if X11:
    GPU_FLAGS = (
        "--ozone-platform=x11",
        "--enable-features=Vulkan",
        "--use-gl=angle",
        "--use-angle=vulkan",
        "--ignore-gpu-blocklist",
        "--use-webgpu-adapter=default",
    )
CHUNK = 1 << 20
# As root (a rented container), Chrome refuses to start without --no-sandbox;
# containers' /dev/shm is small.
ROOT_FLAGS = (
    ("--no-sandbox", "--disable-dev-shm-usage") if os.geteuid() == 0 else ()
)


def find_chrome(chrome=None):
    chrome = (
        chrome
        or os.environ.get("YGARATY_CHROME")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
    )
    if not chrome:
        raise RuntimeError("Chrome/Chromium required; set YGARATY_CHROME")
    return chrome


UNITS = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
DRM_KEYS = (
    "drm-total-vram",
    "drm-total-gtt",
    "drm-resident-vram",
    "drm-resident-gtt",
)


def _read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def process_tree(root):
    children = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        text = _read(stat)
        if not text:
            continue
        fields = text.rsplit(")", 1)[-1].split()
        children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo.extend(children.get(pid, []))
    return out


def process_type(pid):
    # Chromium rewrites its argv as one space-separated string (setproctitle).
    args = _read(f"/proc/{pid}/cmdline").replace("\0", " ").split()
    for arg in args:
        if arg.startswith("--type="):
            kind = arg[7:]
            if kind == "utility":
                sub = next(
                    (
                        a[len("--utility-sub-type=") :]
                        for a in args
                        if a.startswith("--utility-sub-type=")
                    ),
                    "",
                )
                return f"utility:{sub.rsplit('.', 1)[-1]}" if sub else kind
            return kind
    return "browser"


def drm_clients(pid, known):
    """{client id: {key: bytes}} for the process's DRM file descriptors;
    `known` caches which fds are DRM devices."""
    out = {}
    fds = known.setdefault(pid, {})
    try:
        names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return out
    for fd in names:
        if fd not in fds:
            try:
                fds[fd] = os.readlink(f"/proc/{pid}/fd/{fd}").startswith(
                    "/dev/dri/"
                )
            except OSError:
                fds[fd] = False
        if not fds[fd]:
            continue
        info = {}
        for line in _read(f"/proc/{pid}/fdinfo/{fd}").splitlines():
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
        if "drm-client-id" not in info:
            continue
        values = {}
        for key in DRM_KEYS:
            if key in info:
                number, _, unit = info[key].partition(" ")
                values[key] = int(number) * UNITS.get(unit or "B", 1)
        out[info["drm-client-id"]] = values
    return out


def device_memory():
    out = {}
    for card in Path("/sys/class/drm").glob("card[0-9]"):
        for name in ("mem_info_vram_used", "mem_info_gtt_used"):
            text = _read(card / "device" / name).strip()
            if text:
                out[f"{card.name}.{name}"] = int(text)
    return out


def memory_sample(root, known):
    sample = dict(
        t=time.monotonic(), rss={}, pss={}, drm={}, device=device_memory()
    )
    clients = {}
    for pid in process_tree(root):
        kind = process_type(pid)
        rss = next(
            (
                int(line.split()[1]) * 1024
                for line in _read(f"/proc/{pid}/status").splitlines()
                if line.startswith("VmRSS:")
            ),
            0,
        )
        pss = next(
            (
                int(line.split()[1]) * 1024
                for line in _read(f"/proc/{pid}/smaps_rollup").splitlines()
                if line.startswith("Pss:")
            ),
            0,
        )
        sample["rss"][kind] = sample["rss"].get(kind, 0) + rss
        sample["pss"][kind] = sample["pss"].get(kind, 0) + pss
        for client, values in drm_clients(pid, known).items():
            clients[client] = (kind, values)
    for kind, values in clients.values():
        for key, value in values.items():
            sample["drm"][key] = sample["drm"].get(key, 0) + value
            sample["drm"][f"{kind}:{key}"] = (
                sample["drm"].get(f"{kind}:{key}", 0) + value
            )
    sample["rss"]["total"] = sum(sample["rss"].values())
    sample["pss"]["total"] = sum(sample["pss"].values())
    return sample


def summarize_memory(samples, phases, baseline):
    """Peaks over the run and the first sample at or after each phase mark."""

    def flat(sample):
        out = {f"rss.{k}": v for k, v in sample["rss"].items()}
        out.update({f"pss.{k}": v for k, v in sample["pss"].items()})
        out.update({f"drm.{k}": v for k, v in sample["drm"].items()})
        out.update(
            {
                f"device.{k}": v - baseline.get(k, 0)
                for k, v in sample["device"].items()
            }
        )
        return out

    peak = {}
    for sample in samples:
        for key, value in flat(sample).items():
            peak[key] = max(peak.get(key, value), value)
    at = {}
    for t, name in phases:
        after = [s for s in samples if s["t"] >= t]
        if after:
            at[name] = flat(after[0])
    return dict(
        samples=len(samples), peak=peak, phases=at, device_baseline=baseline
    )


NATIVE_RUNNER = _host_component("browser_assets").BROWSER_ROOT / "native_runner.mjs"


def native_command(runtime, base, page):
    """DO-68: YGARATY_WEBGPU_RUNTIME=node (Dawn, the `webgpu` npm package) or deno (wgpu)."""
    if runtime in {"node", "bun"}:
        return [
            os.environ.get("YGARATY_" + runtime.upper(), runtime),
            str(NATIVE_RUNNER),
            base,
            str(page),
        ]
    if runtime in {"deno", "wgpu"}:
        return [
            os.environ.get("YGARATY_DENO", "deno"),
            "run",
            "-A",
            str(NATIVE_RUNNER),
            base,
            str(page),
        ]
    raise ValueError(
        f"YGARATY_WEBGPU_RUNTIME: chrome, node, bun, deno or wgpu, not {runtime!r}"
    )


def run_page(
    page,
    files,
    output,
    modules=(),
    chrome=None,
    timeout=600.0,
    chrome_args=GPU_FLAGS,
    query="",
    memory=False,
    interval=0.25,
):
    """Serve `files` ({name: bytes, Path or (Path, offset, length)}) to
    `page` and return its result.

    `modules` are the extra static modules the page may import.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {page, *modules}
    received, ready = [], threading.Event()
    log = (output / "page.log").open("a")
    phases = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, content, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                html = f'<script type="module" src="/{page}"></script>'
                self.send(html.encode(), "text/html")
            elif path[1:] in allowed:
                self.send((STATIC / path[1:]).read_bytes(), "text/javascript")
            elif path.startswith("/f/") and path[3:] in files:
                source = files[path[3:]]
                if isinstance(source, bytes):
                    self.send(source, "application/octet-stream")
                    return
                # A Path, or (Path, offset, length): a byte range of a file.
                source, offset, length = (
                    source
                    if isinstance(source, tuple)
                    else (source, 0, source.stat().st_size)
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with source.open("rb") as handle:
                    handle.seek(offset)
                    left = length
                    while left and (block := handle.read(min(CHUNK, left))):
                        self.wfile.write(block)
                        left -= len(block)
            else:
                self.send_error(404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/result":
                received.append(json.loads(body))
                ready.set()
            elif self.path.startswith("/blob/"):
                name = Path(self.path[6:]).name
                (output / name).write_bytes(body)
            elif self.path == "/log":
                line = body.decode()
                now = time.monotonic()
                if line.startswith("phase "):
                    phases.append((now, line[6:].strip()))
                log.write(f"{now:.3f} {line}\n")
                log.flush()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (output / "chromium.log").open("w") as chrome_log:
            command = [
                find_chrome(chrome),
                *HEADLESS,
                "--enable-unsafe-webgpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                *chrome_args,
                *ROOT_FLAGS,
                f"--user-data-dir={output / 'browser-profile'}",
                f"http://127.0.0.1:{server.server_port}/{query}",
            ]
            runtime = os.environ.get("YGARATY_WEBGPU_RUNTIME", "chrome")
            if (
                runtime != "chrome"
            ):  # DO-68: the same page module under Dawn (node) or wgpu (deno), no browser
                command = native_command(
                    runtime,
                    f"http://127.0.0.1:{server.server_port}",
                    STATIC / page,
                )
            baseline = device_memory()
            process = subprocess.Popen(
                command, stdout=chrome_log, stderr=chrome_log
            )
            samples, stop = [], threading.Event()

            def sample_loop():
                known = {}
                with (output / "memory.jsonl").open("w") as out:
                    while not stop.is_set():
                        sample = memory_sample(process.pid, known)
                        samples.append(sample)
                        out.write(json.dumps(sample) + "\n")
                        stop.wait(interval)

            sampler = threading.Thread(target=sample_loop, daemon=True)
            if memory:
                sampler.start()
            try:
                deadline = time.monotonic() + timeout
                while not ready.wait(0.2):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Chromium exited; see {output / 'chromium.log'}"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"page timed out; see {output / 'page.log'}"
                        )
            finally:
                if memory:
                    stop.set()
                    sampler.join()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        log.close()
        shutil.rmtree(output / "browser-profile", ignore_errors=True)
    result = received[0]
    if memory:
        result["memory"] = summarize_memory(samples, phases, baseline)
    return result
