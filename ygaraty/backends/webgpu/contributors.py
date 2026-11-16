"""Launch local contributor processes against a public invitation."""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent


def add_parser(commands):
    parser = commands.add_parser(
        "contribute", help="Join a session with Chromium or native WebGPU"
    )
    parser.add_argument("url", nargs="?", help="Contributor link from launch")
    parser.add_argument(
        "--runtime",
        choices=("headless", "deno", "wgpu", "node", "bun"),
        default="headless",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a native GPU compute probe without joining or downloading",
    )
    parser.add_argument("--webgpu-module", type=Path)
    parser.add_argument("--dawn-option", action="append", default=[])
    return parser


def command(args):
    if not 1 <= args.num_workers <= 99:
        raise ValueError("--num-workers must be 1–99")
    if not args.check:
        url = urlsplit(args.url or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or not parse_qs(url.fragment).get("key")
        ):
            raise ValueError("Use the contributor link with #key=...")
        if "pool" in parse_qs(url.fragment):
            raise ValueError("Use a session contributor link, not a pool link")
    runtime = "deno" if args.runtime == "wgpu" else args.runtime
    if runtime == "headless":
        if args.check or args.webgpu_module or args.dawn_option:
            raise ValueError(
                "GPU probe and Dawn options require a native runtime"
            )
        return [
            "bash",
            str(ROOT / "contributors/headless.sh"),
            args.url,
            str(args.num_workers),
        ]
    executable = shutil.which(runtime)
    if not executable:
        raise ValueError(f"{runtime} is not installed or not on PATH")
    if runtime == "deno" and (args.webgpu_module or args.dawn_option):
        raise ValueError("Deno uses built-in wgpu, not Dawn options")
    cmd = [executable]
    if runtime == "deno":
        # Only local modules; weights can redirect to the public model host.
        cmd += ["run", "--no-config", "--no-prompt", "--allow-net"]
    cmd += [str(ROOT / "browser/native_contributor.mjs")]
    cmd += ["--check"] if args.check else [args.url]
    if args.webgpu_module:
        cmd += ["--webgpu-module", str(args.webgpu_module.resolve())]
    for option in args.dawn_option:
        cmd += ["--dawn-option", option]
    return cmd


def run(args):
    cmd = command(args)
    count = 1 if args.check or args.runtime == "headless" else args.num_workers
    processes = []
    previous = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        for i in range(count):
            processes.append(subprocess.Popen(cmd, start_new_session=True))
            print(
                f"Contributor {i + 1}: {args.runtime}, "
                f"pid {processes[-1].pid}",
                flush=True,
            )
        while True:
            codes = [process.poll() for process in processes]
            if any(code is not None for code in codes):
                # One exit stops the group, including when its slot was full.
                return next(code for code in codes if code is not None)
            time.sleep(0.2)
    except KeyboardInterrupt:
        return 0
    finally:
        # Keep signal handling stable while stopping process groups, including
        # Chromium children owned by the headless helper.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
