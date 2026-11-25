"""Launch a browser-contributed, full-replica DiLoCo training session."""

import argparse
import asyncio
import contextlib
from pathlib import Path
import webbrowser

from .datasets import SOURCES, load_dataset
from .session import TrainingConfig, TrainingSession


@contextlib.contextmanager
def exclusive_run(directory):
    """Prevent two CLI coordinators from publishing to one run (Linux/macOS)."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another coordinator owns this run directory"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def serve(session, host, port, share, open_browser):
    import uvicorn
    from ygaraty.backends.webgpu.sharing import PublicTunnel
    from .server import create_app

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(session), host=host, port=port, access_log=False
        )
    )
    serving = asyncio.create_task(server.serve())
    tunnel = None
    watchers = []
    try:
        while not server.started:
            if serving.done():
                await serving
                raise RuntimeError("Training server failed to start")
            await asyncio.sleep(0.05)
        url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
        if share:
            tunnel = PublicTunnel(host, port, session.directory / ".tunnel")
            url = await tunnel.start()
            watchers.append(asyncio.create_task(tunnel.watch()))
        url += "/#" + session.join_token
        print(
            f"\nShare this contributor link (you can open it yourself):\n{url}\n"
            f"Waiting for {session.config.num_workers} contributors. "
            f"Each stores a full model replica.\nCheckpoint: {session.path}\n",
            flush=True,
        )
        if open_browser:
            webbrowser.open(url)
        done, _ = await asyncio.wait(
            [serving, *watchers], return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            await task
    finally:
        server.should_exit = True
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)
        if tunnel:
            await tunnel.close()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=tuple(SOURCES), default="fashion-mnist"
    )
    parser.add_argument(
        "--num-workers", type=int, required=True, choices=(1, 2, 3, 4)
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channels", type=int, nargs=3, default=[8, 16, 16])
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--outer-learning-rate", type=float, default=0.7)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=("adam", "muon"), default="adam")
    parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-examples", type=int, default=256)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--round-timeout", type=float, default=600)
    parser.add_argument("--lease-seconds", type=float, default=45)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--precision",
        choices=("webgpu", "float64"),
        default="webgpu",
        help="float64 is JavaScript CPU validation, not a GPU benchmark",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("training-run"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public HTTPS link using cloudflared",
    )
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        config = TrainingConfig(
            **{
                name: getattr(args, name)
                for name in TrainingConfig.__dataclass_fields__
            }
        )
        if args.share:
            from ygaraty.backends.webgpu.sharing import PublicTunnel

            PublicTunnel(args.host, args.port, args.run_dir).command()
        # Fail on missing optional dependencies before downloading datasets.
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        print(f"Loading verified {args.dataset} data…", flush=True)
        dataset = load_dataset(args.dataset, args.data_dir)
        with exclusive_run(args.run_dir):
            session = TrainingSession(
                config, dataset, args.run_dir, resume=args.resume
            )
            asyncio.run(
                serve(
                    session,
                    args.host,
                    args.port,
                    args.share,
                    args.open_browser,
                )
            )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Training: {error}\n")
    except KeyboardInterrupt:
        pass
    return 0
