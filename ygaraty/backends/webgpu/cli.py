"""Public coordinator commands; help never imports serving dependencies."""

import argparse
import hashlib
import json
import os
import secrets
import uuid
from contextlib import suppress
from pathlib import Path

from ygaraty.backend_contracts.models import ModelIdentity, SessionSpec
from ygaraty.backend_contracts.protocols import RuntimeServices

from .configuration import WebGPUConfig, json_object
from .factory import WebGPUFactory
from .hosting import LocalAccess, MemoryLedger, SystemClock
from .networking import add_network_arguments, resolve_network


def initialize(directory, quiet=False):
    directory.mkdir(parents=True, exist_ok=True)
    names = (directory / "owner.key", directory / "contributor.key")
    if any(path.exists() for path in names):
        raise ValueError(
            "Credential files already exist; refusing to overwrite"
        )
    for path in names:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as output:
            output.write(secrets.token_urlsafe(32) + "\n")
    if not quiet:
        print(f"Created owner.key and contributor.key in {directory}")


def ensure_keys(directory):
    """Create owner and contributor keys once; refuse partial key sets."""
    paths = [directory / name for name in ("owner.key", "contributor.key")]
    if not any(path.exists() for path in paths):
        initialize(directory, quiet=True)
    elif not all(
        path.is_file() and path.read_text().strip() for path in paths
    ):
        raise ValueError(
            "Incomplete credentials; refusing to replace existing keys"
        )


def prepare_serve(args):
    """serve's defaults, as launch chooses them for a prepared bundle.

    The workspace holds keys/ and state/ (default runs/MODEL/N-workers, as
    launch); --keys and --state-dir still override either. Without
    --memory-gb, the GPU memory budget is the bundle's recommendation.
    """
    from .configuration import json_object

    for name in (
        "workspace",
        "keys",
        "state_dir",
        "workers",
        "memory_gb",
        "public_url",
        "bundle_url",
    ):
        if not hasattr(args, name):
            setattr(args, name, None)
    given = dict(vars(args))
    if not 1 <= args.port <= 65535 or not 1 <= args.slots <= 16:
        raise ValueError("Require port 1–65535 and slots 1–16")
    if args.memory_gb is not None and args.memory_gb < 1:
        raise ValueError("--memory-gb must be positive")
    if args.bundle is None:
        # A supported model by name (or --bundle-url), as launch does.
        from .launch import fetch_public_bundle, require_model_and_workers

        args.model = args.model_id
        require_model_and_workers(args, "workers")
        if args.workspace is None:
            args.workspace = default_workspace(args.model_id, args.workers)
        args.workspace = Path(args.workspace).resolve()
        args.bundle = fetch_public_bundle(args, args.workers, [])
    elif not args.model_id:
        from .launch import require_model_and_workers

        args.model = None
        require_model_and_workers(args, "workers")
    data = json_object(args.bundle.read_bytes())
    stages = len(data.get("stages") or [])
    workers = getattr(args, "workers", None)
    if workers is not None and workers != stages:
        raise ValueError(
            f"This bundle has {stages} stage(s); --num-workers must match "
            "(prepare another bundle with --splits to change it)"
        )
    args.workers = stages
    if args.workspace is None and (
        args.keys is None or args.state_dir is None
    ):
        args.workspace = default_workspace(args.model_id, stages)
    if args.keys is None:
        args.keys = args.workspace / "keys"
        ensure_keys(args.keys)
    if args.state_dir is None:
        args.state_dir = args.workspace / "state"
    if args.memory_gb is None:
        from .memory import report_memory

        try:
            args.memory_gb = report_memory(
                data, stages, args.context, args.slots, None, verbose=False
            )
        except (ValueError, KeyError, TypeError):
            args.memory_gb = 8
    if getattr(args, "save_config", False):
        from types import SimpleNamespace

        from .config import save_config

        if args.workspace is None:
            raise ValueError("--save-config needs a workspace")
        given["workspace"] = args.workspace
        path = save_config(SimpleNamespace(**given), "serve")
        print(f"Saved settings to {path}", flush=True)
    return args


def default_workspace(model, workers):
    """runs/MODEL/N-workers (YGARATY_WORKSPACE_ROOT replaces runs)."""
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]", "_", model).strip(".") or "model"
    return (
        Path(os.environ.get("YGARATY_WORKSPACE_ROOT", "runs"))
        / name
        / f"{workers}-workers"
    )


def print_links(args, out=None):
    """The links serve prints at startup (launch shows its own screen)."""
    import sys

    from .launch import connection_details

    links = connection_details(args)
    lines = [
        f"Serving {args.model_id} ({args.workers} worker"
        + ("s" if args.workers != 1 else "")
        + f", {args.slots} chat"
        + ("s" if args.slots != 1 else "")
        + " at a time)",
        f"  Contributor link (share it): {links['contribute']}",
        f"  Session page (private): {links['session']}",
        f"  Web chat (private): {links['chat']}",
        f"  API: {links['api']} · model {args.model_id} · key in "
        f"{args.keys / 'owner.key'}",
    ]
    print("\n".join(lines), file=out or sys.stdout, flush=True)


def session_spec(args):
    from ._legacy.chat_contract import chat_format
    from ._legacy.model_artifacts import local_file

    if args.bundle.stat().st_size > 16 << 20:
        raise ValueError("Bundle exceeds 16 MiB")
    raw = args.bundle.read_bytes()
    data = json_object(raw)
    tokenizer = local_file(args.bundle.parent, data["tokenizer"])
    if tokenizer.stat().st_size > 64 << 20:
        raise ValueError("Tokenizer exceeds 64 MiB")

    def digest(value):
        return "sha256:" + hashlib.sha256(value).hexdigest()

    index = {name: part["sha256"] for name, part in data["files"].items()}
    model = ModelIdentity(
        args.model_id,
        digest(json.dumps(index, sort_keys=True).encode()),
        digest(tokenizer.read_bytes()),
        digest(
            json.dumps(
                chat_format(data["options"].get("chat")), sort_keys=True
            ).encode()
        ),
        "bundle-defined",
    )
    config = WebGPUConfig(
        hashlib.sha256(raw).hexdigest(),
        tuple(tuple(stage["layers"]) for stage in data["stages"]),
        data["options"]["capacity"] if args.context is None else args.context,
        args.memory_gb * 1024**3,
        conversation_slots=args.slots,
        route=resolve_network(args)[0],
    )
    return SessionSpec(
        uuid.uuid4().hex, uuid.uuid4().hex, model, config.to_contract()
    )


def serve(args):
    try:
        import uvicorn

        from .server import create_app
    except ImportError as error:
        raise ValueError(
            "Install ygaraty-webgpu[server] in this environment"
        ) from error
    if not getattr(args, "announce", False):
        prepare_serve(args)
    args.route, args.turn_config = resolve_network(
        args, local_network=not getattr(args, "public_url", None)
    )
    spec = session_spec(args)
    clock = SystemClock()
    access = LocalAccess(
        spec,
        (args.keys / "owner.key").read_text().strip(),
        (args.keys / "contributor.key").read_text().strip(),
        args.slots,
        clock,
    )
    services = RuntimeServices(clock, access, MemoryLedger(spec))
    factory = WebGPUFactory(
        args.bundle, args.state_dir / spec.epoch, turn=args.turn_config
    )
    app = create_app(
        factory,
        spec,
        services,
        access,
        open_status=getattr(args, "open_status", False),
    )
    if not getattr(args, "announce", False):
        print_links(args)
    elif getattr(args, "request_log", False):
        print(f"Session {spec.session_id}; chat at /chat, contributors at /")
    if getattr(args, "announce", False):
        import asyncio

        from .sharing import serve_launch

        # The launch console shows links and worker progress; HTTP request
        # and server-startup lines only with --request-log.
        request_log = getattr(args, "request_log", False)
        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            limit_concurrency=256,
            timeout_graceful_shutdown=10,
            access_log=request_log,
            log_level="info" if request_log else "warning",
            **(
                dict(ssl_certfile=str(tls[0]), ssl_keyfile=str(tls[1]))
                if (tls := getattr(args, "tls", None))
                else {}
            ),
        )

        class LaunchServer(uvicorn.Server):
            async def shutdown(self, sockets=None):
                # Close the session first: contributors' long polls then
                # return at once instead of holding the graceful drain
                # until its timeout.
                session = getattr(app.state, "session", None)
                if session is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(session.close(), 5)
                await super().shutdown(sockets)

        try:
            asyncio.run(serve_launch(LaunchServer(config), app, args))
        except KeyboardInterrupt:  # uvicorn re-raises Ctrl-C after shutdown
            print("\nStopped.", flush=True)
    else:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            limit_concurrency=256,
        )


def model_command(args):
    if args.operation == "import-kyse":
        from .checkpoints.kyse import convert_checkpoint

        print(
            json.dumps(convert_checkpoint(args.source, args.output), indent=2)
        )
        return 0

    from dataclasses import asdict

    from .bundles import (
        BundleBuilder,
        Checkpoint,
        inspect_checkpoint,
        validate_bundle,
    )

    if args.operation == "inspect":
        result = inspect_checkpoint(args.gguf, args.config)
        print(json.dumps(result, indent=2))
        return 0 if result["compatible"] else 1
    if args.operation == "validate":
        result = validate_bundle(args.bundle)
    else:
        splits = (
            None
            if args.splits is None
            else tuple(int(n) for n in args.splits.split(","))
        )
        checkpoint = Checkpoint(
            args.gguf, args.config, args.tokenizer, args.chat_format
        )
        result = BundleBuilder(
            checkpoint,
            splits=splits,
            context_tokens=args.context,
            max_output_tokens=args.generate,
        ).prepare(args.output)
    print(json.dumps(asdict(result), default=str, indent=2))
    return 0


def model_parser(commands):
    model = commands.add_parser(
        "model", help="Inspect, prepare and validate local model bundles"
    )
    operations = model.add_subparsers(dest="operation", required=True)
    kyse = operations.add_parser(
        "import-kyse",
        help="Import a Kyse English QW35 checkpoint for one worker",
    )
    kyse.add_argument(
        "source",
        type=Path,
        help="Directory containing model.bin and tokenizer.bin",
    )
    kyse.add_argument("--output", type=Path, required=True)
    inspect = operations.add_parser(
        "inspect", help="Read local GGUF/config metadata; no inference"
    )
    prepare = operations.add_parser(
        "prepare",
        help="Build a local bundle and numerical self-test references",
    )
    validate = operations.add_parser(
        "validate",
        help="Check local hashes and fresh plans without inference",
    )
    for parser in (inspect, prepare):
        parser.add_argument("--gguf", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument("--chat-format", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--splits", help="Layer boundaries, e.g. 0,16,32; default one stage"
    )
    prepare.add_argument("--context", type=int, default=4096)
    prepare.add_argument("--generate", type=int, default=256)
    validate.add_argument("bundle", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ygaraty",
        description="Public WebGPU coordinator: local bundles and relay",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="COMMAND"
    )
    model_parser(commands)
    from . import bootstrap, catalog, contributors, launch, status_command

    catalog.add_parser(commands)
    bootstrap.add_parser(commands)
    launch_parser = launch.add_parser(commands)
    contributors.add_parser(commands)
    init = commands.add_parser(
        "init", help="Create separate owner and contributor credentials"
    )
    init.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser(
        "serve",
        help="Run only the server for a model (no console, sharing or chat)",
    )
    add_network_arguments(run)
    source = run.add_mutually_exclusive_group()
    source.add_argument(
        "--bundle", type=Path, help="Prepared bundle.json on this machine"
    )
    source.add_argument(
        "--bundle-url", help="Public HTTPS prepared bundle (needs --model)"
    )
    run.add_argument(
        "--model",
        "--model-id",
        dest="model_id",
        help="Model: a name from 'ygaraty models list' (downloads its public "
        "bundle), or the API name for --bundle/--bundle-url",
    )
    run.add_argument(
        "--workspace",
        type=Path,
        help="Keys and state (default: runs/MODEL/N-workers, as launch)",
    )
    run.add_argument(
        "--num-workers",
        "--workers",
        "--contributors",
        dest="workers",
        type=int,
        help="GPU worker count: required with a model name or --bundle-url; "
        "with --bundle, must equal its stage count",
    )
    run.add_argument(
        "--context", type=int, help="Context tokens (default: the bundle's)"
    )
    run.add_argument(
        "--memory-gb",
        type=int,
        help="GPU memory cap in GiB (default: the bundle's recommendation)",
    )
    run.add_argument(
        "--slots",
        type=int,
        default=1,
        help="Concurrent conversation slots, 1–16",
    )
    run.add_argument("--host", default="127.0.0.1", help="Bind address")
    run.add_argument("--port", type=int, default=8765)
    run.add_argument(
        "--public-url",
        help="HTTPS origin that contributors reach this server at (for the "
        "printed links; also keeps --route auto off local-only WebRTC)",
    )
    run.add_argument(
        "--keys",
        type=Path,
        help="Credential directory (default: WORKSPACE/keys, created once)",
    )
    run.add_argument(
        "--state-dir",
        type=Path,
        help="Session state (default: WORKSPACE/state)",
    )
    launch.add_common_arguments(run)
    status_command.add_parser(commands)
    args = parser.parse_args(argv)
    if args.command in {"launch", "serve"}:
        from .config import config_defaults, find_config

        try:
            path = find_config(args)
            if path is not None:
                sub = launch_parser if args.command == "launch" else run
                sub.set_defaults(**config_defaults(path, sub))
                args = parser.parse_args(argv)
        except ValueError as error:
            fail(args, error)
    try:
        if args.command == "contribute":
            return contributors.run(args)
        if args.command == "launch":
            return launch.run(args)
        if args.command == "models":
            return catalog.run(args)
        if args.command == "status":
            return status_command.run(args)
        if args.command == "bootstrap":
            return bootstrap.run(args)
        if args.command == "model":
            return model_command(args)
        if args.command == "init":
            initialize(args.directory)
        else:
            serve(args)
    except (ValueError, OSError, KeyError) as error:
        fail(args, error)
    return 0


def fail(args, error):
    """Exit 1 for a failed command (2 stays argparse's usage error); with
    --json the message is also printed as {"error": ...} on stdout."""
    import sys

    message = (
        str(error).strip("'\"") if isinstance(error, KeyError) else str(error)
    )
    if getattr(args, "json", False):
        print(json.dumps({"error": message}))
    print(f"ygaraty {args.command}: error: {message}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
