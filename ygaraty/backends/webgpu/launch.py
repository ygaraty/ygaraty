"""User-facing setup, links and readiness for a single public model server."""

import asyncio
import os
import re
import signal
import sys
import time
import webbrowser
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from .bootstrap import PROFILES, Bootstrap, ModelProfile


def public_origin(value, host, port):
    if value:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in value)
        ):
            raise ValueError(
                "--public-url must be an HTTP(S) origin without a path"
            )
        # Accessing port also validates malformed port numbers.
        parsed.port
        if parsed.scheme != "https" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError(
                "Remote contributor links require an HTTPS public URL"
            )
        return value.rstrip("/")
    local = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in local:
        local = f"[{local}]"
    return f"http://{local}:{port}"


def connection_details(args):
    origin = public_origin(args.public_url, args.host, args.port)
    contributor = (args.keys / "contributor.key").read_text().strip()
    owner = (args.keys / "owner.key").read_text().strip()
    return {
        "contribute": origin + "/#" + urlencode({"key": contributor}),
        "chat": origin + "/chat#" + urlencode({"key": owner}),
        "api": origin + "/v1",
    }


async def announce(session, args):
    """Run after ASGI startup: the link to share, live workers, then chat."""
    from .launch_screen import Style, Workers, header, ready_lines

    await asyncio.sleep(0.5)
    links = connection_details(args)
    header(args, links)
    if args.open_chat:
        try:
            opened = await asyncio.to_thread(webbrowser.open, links["chat"])
        except webbrowser.Error:
            opened = False
        if not opened:
            print("  Open the web chat link in your browser.", flush=True)
    engine = getattr(session, "engine", None)
    if engine is not None:  # session events go to session.log only
        engine.console = getattr(args, "request_log", False)
    style = Style(sys.stdout)
    required = len(getattr(args, "memory_reports", None) or []) or getattr(
        args, "workers", 1
    )
    # In a terminal the rows are redrawn in place until the chat prompt owns
    # the screen; logs and request logging get a plain line per change.
    live = style.tty and not getattr(args, "request_log", False)
    workers = Workers(required)
    if not live:
        workers.style.tty = False
    started = time.monotonic()
    previous, chat, quitting = None, None, False
    try:
        while True:
            state = session.snapshot().state.value
            if state != previous:
                if state == "ready":
                    workers.draw(engine_status(session), force=True)
                    workers.release()
                    terminal = (
                        chat is None and not quitting and wants_chat(args)
                    )
                    summary = ready_lines(
                        args,
                        links,
                        time.monotonic() - started,
                        terminal or chat is not None,
                        style,
                    )
                    print("\n".join(summary), flush=True)
                    if terminal:
                        from .terminal_chat import (
                            local_base,
                            own_server_opener,
                            terminal_chat,
                        )

                        owner = (args.keys / "owner.key").read_text().strip()
                        secure = bool(getattr(args, "tls", None))
                        base = local_base(args.host, args.port, secure=secure)
                        chat = asyncio.create_task(
                            terminal_chat(
                                base,
                                owner,
                                status=lambda: engine_status(session),
                                **(
                                    {"opener": own_server_opener()}
                                    if secure
                                    else {}
                                ),
                            )
                        )
                        live = False  # never redraw over the prompt
                elif state == "unavailable":
                    if previous == "ready":
                        print(
                            "\n  "
                            + style.warn(
                                "! A worker left. Waiting for someone to "
                                "open the link again:"
                            )
                            + "\n    "
                            + links["contribute"]
                            + "\n",
                            flush=True,
                        )
                        started = time.monotonic()
                    workers = Workers(required)
                    if not live:
                        workers.style.tty = False
                else:
                    print(f"  Session {state}.", flush=True)
                previous = state
            if state == "unavailable":
                workers.draw(engine_status(session))
            if (
                chat is not None
                and chat.done()
                and not chat.cancelled()
                and chat.exception() is None
                and chat.result() == "quit"
            ):
                chat, quitting = None, True
                # The same graceful path as Ctrl-C.
                os.kill(os.getpid(), signal.SIGINT)
            await asyncio.sleep(1)
    finally:
        if chat is not None:
            chat.cancel()


def wants_chat(args):
    """Terminal chat needs an interactive stdin and is on by default."""
    if getattr(args, "no_terminal_chat", False):
        return False
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def engine_status(session):
    status = getattr(getattr(session, "engine", None), "status", None)
    try:
        value = status() if status else None
    except Exception:  # progress is advisory; never stop the announcer
        return None
    return value if isinstance(value, dict) else None


class LoadingProgress:
    """Summarize contributors' reported weight loading every few seconds."""

    def __init__(self, interval=10.0, clock=time.monotonic):
        self.interval, self.clock = interval, clock
        self.last_at, self.last_line, self.last_loaded = None, None, {}

    def update(self, status):
        if not status:
            return None
        now = self.clock()
        if self.last_at is not None and now - self.last_at < self.interval:
            return None
        roster = status.get("roster", {})
        parts, loaded_now = [], {}
        for record in roster.get("contributors", []):
            stage = (record.get("assignment") or {}).get("stage", "?")
            progress = record.get("progress") or {}
            loaded, total = progress.get("loaded"), progress.get("total")
            if record.get("state") == "active":
                parts.append(f"{stage} ready")
            elif isinstance(loaded, (int, float)) and total:
                text = (
                    f"{stage} {loaded / 1e9:.2f}/{total / 1e9:.2f} GB "
                    f"{100 * loaded / total:.0f}%"
                )
                before = self.last_loaded.get(record.get("id"))
                if before is not None and self.last_at is not None:
                    rate = (loaded - before) / (now - self.last_at)
                    if rate > 0:
                        text += f" {rate / 1e6:.1f} MB/s"
                loaded_now[record.get("id")] = loaded
                parts.append(text)
            else:
                parts.append(f"{stage} {progress.get('phase') or 'joining'}")
        if not parts:
            return None
        required = roster.get("required")
        joined = len(roster.get("contributors", []))
        line = "Loading: " + " · ".join(parts)
        if required:
            line += f" ({joined} of {required} workers joined)"
        self.last_at, self.last_loaded = now, loaded_now
        if line == self.last_line:
            return None
        self.last_line = line
        return line


def configure_prepared_memory(
    launch, workers, context, slots, cap=None, reports_out=None
):
    from .configuration import json_object
    from .memory import report_memory

    launch.memory_gb = report_memory(
        json_object(launch.bundle.read_bytes()),
        workers,
        context,
        slots,
        cap,
        verbose=reports_out is None,
        reports_out=reports_out,
    )


def add_parser(commands):
    parser = commands.add_parser(
        "launch", help="Set up a server, share a link and chat"
    )
    from .networking import add_network_arguments

    add_network_arguments(parser)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", type=Path, help="GGUF/config/tokenizer directory"
    )
    source.add_argument(
        "--bundle", type=Path, help="Existing prepared bundle.json"
    )
    source.add_argument(
        "--bundle-url",
        help="Public HTTPS prepared bundle; model preset used by default",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Cache and keys (default: runs/MODEL/N-workers)",
    )
    parser.add_argument(
        "--model",
        "--model-id",
        default="qwen3.5-9b",
        help="API model name; known names select preset splits and memory",
    )
    parser.add_argument(
        "--gguf",
        default="model.gguf",
        help="Checkpoint filename (first shard for split GGUF)",
    )
    parser.add_argument("--splits", help="Layer boundaries, e.g. 0,8,16,24,32")
    parser.add_argument(
        "--num-workers",
        "--workers",
        "--contributors",
        dest="contributors",
        type=int,
        required=True,
        help="Required GPU worker count (public presets support 1, 2, 3 or 4)",
    )
    parser.add_argument(
        "--context", type=int, help="Context tokens (checkpoint default: 4096)"
    )
    parser.add_argument(
        "--generate",
        type=int,
        help="Maximum output tokens (checkpoint default: 128)",
    )
    parser.add_argument(
        "--memory-gb",
        type=int,
        help="Optional GPU memory cap in GiB; must cover the recommendation",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=1,
        help="Concurrent conversation slots, 1–16",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local bind address (public sharing works on loopback)",
    )
    parser.add_argument("--port", type=int, default=8765)
    # Local by default; --lan for this network, --public for the internet.
    sharing = parser.add_mutually_exclusive_group()
    sharing.add_argument(
        "--lan",
        nargs="?",
        const="auto",
        metavar="ADDRESS",
        help="Share on the local network: HTTPS links with this machine's "
        "LAN address (or ADDRESS) and a self-signed certificate that each "
        "device accepts once",
    )
    sharing.add_argument(
        "--public",
        action="store_true",
        help="Share on the internet through a free Cloudflare quick tunnel "
        "(needs cloudflared; temporary https://*.trycloudflare.com address)",
    )
    sharing.add_argument(
        "--local",
        action="store_true",
        help="Run locally only (the default; kept for compatibility)",
    )
    sharing.add_argument(
        "--public-url",
        help="Use your own HTTPS origin instead of an automatic tunnel",
    )
    parser.add_argument(
        "--open-chat",
        action="store_true",
        help="Open private chat in this machine's browser",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare model and keys without starting",
    )
    parser.add_argument(
        "--no-terminal-chat",
        action="store_true",
        help="Do not open a chat prompt in this terminal once the model "
        "is ready",
    )
    parser.add_argument(
        "--request-log",
        action="store_true",
        help="Print HTTP requests and contributor session events "
        "(default: only links and worker loading progress)",
    )
    parser.add_argument(
        "--serve-weights",
        action="store_true",
        help="Download public weights here and serve them to contributors "
        "(default: contributors download directly from public storage)",
    )


def run(args):
    from types import SimpleNamespace

    from .bundles import inspect_checkpoint, validate_bundle
    from .cli import initialize, serve
    from .networking import resolve_network

    route, turn_config = resolve_network(
        args,
        local_network=not getattr(args, "public", False)
        and not args.public_url,
    )
    public_origin(args.public_url, args.host, args.port)
    if not 1 <= args.port <= 65535 or not 1 <= args.slots <= 16:
        raise ValueError("Require port 1–65535 and slots 1–16")
    if args.memory_gb is not None and args.memory_gb < 1:
        raise ValueError("--memory-gb must be positive")
    if args.context is not None and args.context < 1:
        raise ValueError("--context must be positive")
    if not args.model.strip():
        raise ValueError("--model must not be empty")
    contributors = args.contributors
    memory_reports = []  # the estimate per worker, for the launch screen
    if type(contributors) is not int or not 1 <= contributors <= 26:
        raise ValueError("--num-workers must be between 1 and 26")
    if args.workspace is None:
        name = (
            re.sub(r"[^A-Za-z0-9_.-]", "_", args.model).strip(".") or "model"
        )
        args.workspace = (
            Path(os.environ.get("YGARATY_WORKSPACE_ROOT", "runs"))
            / name
            / f"{contributors}-workers"
        )
    args.workspace = args.workspace.resolve()
    tls = None
    if getattr(args, "lan", None):
        from .lan import check_address, lan_address, lan_certificate

        address = (
            lan_address() if args.lan == "auto" else check_address(args.lan)
        )
        if args.host in {"127.0.0.1", "localhost", "::1"}:
            args.host = "0.0.0.0"  # reachable from the network
        tls = lan_certificate(args.workspace / ".lan", address)
        host = f"[{address}]" if ":" in address else address
        args.public_url = f"https://{host}:{args.port}"
    if getattr(args, "public", False) and not args.prepare_only:
        from .sharing import PublicTunnel

        # Check local prerequisites before preparing or downloading a model.
        PublicTunnel(
            args.host, args.port, args.workspace / ".tunnel"
        ).command()
    downloaded = args.bundle_url or (not args.bundle and not args.checkpoint)
    if downloaded:
        from .downloads import PUBLIC_PRESETS, fetch_bundle

        preset = (
            PUBLIC_PRESETS.get((args.model, contributors))
            if not args.bundle_url
            else None
        )
        url = args.bundle_url or (preset.url if preset else None)
        if preset:
            if args.context is None:
                args.context = preset.context
        if not url:
            raise ValueError(
                "Public presets support qwen3.5-0.8b/s5-nano/qwen3.5-9b/"
                "qwen3.8-27b "
                "with 1–4 workers; "
                "supply --bundle-url or --checkpoint"
            )
        if args.splits is not None or args.generate is not None:
            raise ValueError(
                "Published bundle splits/generation are fixed; "
                "use --checkpoint to rebuild"
            )
        from .memory import report_memory

        def configure_memory(data):
            args.memory_gb = report_memory(
                data,
                contributors,
                args.context,
                args.slots,
                args.memory_gb,
                verbose=False,
                reports_out=memory_reports,
            )

        serve_weights = getattr(args, "serve_weights", False)
        print(
            f"Checking the {args.model} bundle"
            + (" and downloading its weights" if serve_weights else "")
            + "…",
            flush=True,
        )
        args.bundle = fetch_bundle(
            url,
            args.workspace / ("model" if serve_weights else "remote-model"),
            contributors=contributors,
            on_manifest=configure_memory,
            weights=serve_weights,
        )
    if args.bundle:
        if args.splits is not None or args.generate is not None:
            raise ValueError(
                "--splits and --generate are preparation settings; "
                "use --checkpoint"
            )
        validate_bundle(args.bundle)
        if not downloaded:
            from .configuration import json_object
            from .memory import report_memory

            args.memory_gb = report_memory(
                json_object(args.bundle.read_bytes()),
                contributors,
                args.context,
                args.slots,
                args.memory_gb,
                verbose=False,
                reports_out=memory_reports,
            )
        launch = SimpleNamespace(
            bundle=args.bundle.resolve(),
            model_id=args.model,
            keys=args.workspace / "keys",
            state_dir=args.workspace / "state",
            context=args.context,
            memory_gb=args.memory_gb or 8,
            slots=args.slots,
            host=args.host,
            port=args.port,
        )
        paths = [
            launch.keys / name for name in ("owner.key", "contributor.key")
        ]
        if not any(path.exists() for path in paths):
            initialize(launch.keys, quiet=True)
        elif not all(
            path.is_file() and path.read_text().strip() for path in paths
        ):
            raise ValueError(
                "Incomplete credentials; refusing to replace existing keys"
            )
    else:
        profile = PROFILES.get(args.model)
        if profile is None:
            if Path(args.gguf).name != args.gguf:
                raise ValueError(
                    "--gguf must be a filename inside --checkpoint"
                )
            report = inspect_checkpoint(
                args.checkpoint / args.gguf, args.checkpoint / "config.json"
            )
            if not report["compatible"]:
                raise ValueError("Checkpoint is not supported by this backend")
            layers = report["geometry"]["layers"]
            profile = ModelProfile(
                args.model,
                report["family"],
                layers,
                (0, layers),
                8,
                "Local checkpoint",
                "Requires model/device validation",
            )
        if contributors > profile.layers:
            raise ValueError("Worker count cannot exceed model layer count")
        splits = (
            profile.splits
            if len(profile.splits) - 1 == contributors
            else tuple(
                i * profile.layers // contributors
                for i in range(contributors + 1)
            )
        )
        if args.splits is not None:
            splits = tuple(int(n) for n in args.splits.split(","))
            if (
                not 2 <= len(splits) <= 27
                or splits[0] != 0
                or splits[-1] != profile.layers
                or any(a >= b for a, b in zip(splits, splits[1:]))
            ):
                raise ValueError(
                    "--splits must increase from zero to the model's layer "
                    "count (at most 26 stages)"
                )
        if len(splits) - 1 != contributors:
            raise ValueError("--splits stage count must match --num-workers")
        profile = replace(
            profile,
            splits=splits,
            memory_gb=args.memory_gb
            or (64 if profile.family == "qwen35" else profile.memory_gb),
        )
        setup = Bootstrap(
            profile,
            args.checkpoint,
            args.workspace,
            args.gguf,
            args.context or 4096,
            args.generate if args.generate is not None else 128,
            args.host,
            args.port,
        )
        print(
            "Preparing model and credentials; "
            "completed workspaces are reused.",
            flush=True,
        )
        launch = setup.prepare()
        launch.slots = args.slots
        if profile.family == "qwen35":
            configure_prepared_memory(
                launch,
                contributors,
                args.context or 4096,
                args.slots,
                args.memory_gb,
                reports_out=memory_reports,
            )
    launch.route, launch.turn_config = route, turn_config
    label = {
        "relay": "coordinator relay",
        "direct": "WebRTC with TURN fallback"
        if turn_config
        else "WebRTC on the local network, relay fallback",
        "private_turn": "WebRTC with TURN-only peer links",
    }[route.value]
    launch.route_label = label
    launch.memory_reports = memory_reports
    launch.workers = contributors
    launch.share = getattr(args, "public", False)
    launch.public_url = args.public_url
    launch.open_chat = args.open_chat
    launch.request_log = getattr(args, "request_log", False)
    launch.tls = tls
    launch.no_terminal_chat = getattr(args, "no_terminal_chat", False)
    launch.announce = True
    if args.prepare_only:
        print(
            "Prepared. Run this command without --prepare-only to start. "
            f"Workspace: {args.workspace}"
        )
        return 0
    serve(launch)
    return 0
