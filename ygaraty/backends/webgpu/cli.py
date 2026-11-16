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
    args.route, args.turn_config = resolve_network(args)
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
    app = create_app(factory, spec, services, access)
    if not getattr(args, "announce", False) or getattr(
        args, "request_log", False
    ):
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
    commands = parser.add_subparsers(dest="command", required=True)
    model_parser(commands)
    from . import bootstrap, contributors, launch

    bootstrap.add_parser(commands)
    launch.add_parser(commands)
    contributors.add_parser(commands)
    init = commands.add_parser(
        "init", help="Create separate owner and contributor credentials"
    )
    init.add_argument("--directory", type=Path, required=True)
    run = commands.add_parser(
        "serve", help="Serve one prepared local WebGPU bundle"
    )
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--keys", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--context", type=int)
    run.add_argument("--memory-gb", type=int, default=8)
    run.add_argument("--slots", type=int, default=8)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8765)
    add_network_arguments(run)
    args = parser.parse_args(argv)
    try:
        if args.command == "contribute":
            return contributors.run(args)
        if args.command == "launch":
            return launch.run(args)
        if args.command == "bootstrap":
            return bootstrap.run(args)
        if args.command == "model":
            return model_command(args)
        if args.command == "init":
            initialize(args.directory)
        else:
            serve(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
