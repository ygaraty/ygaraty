"""Bounded, same-host contributor check; never a multi-machine benchmark.

Uses the installed public package. Run from the checkout with PYTHONPATH=. or
install its wheel first. Chromium runs in isolated temporary profiles.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ygaraty.training.datasets import ImageDataset, load_dataset
from ygaraty.training.server import create_app
from ygaraty.training.session import (
    TrainingConfig,
    TrainingSession,
    implementation_hash,
)
from ygaraty.training.trainer import Trainer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers", type=int, choices=(1, 2, 3, 4), required=True
    )
    parser.add_argument(
        "--dataset",
        choices=("fixture", "mnist", "fashion-mnist"),
        default="fixture",
    )
    parser.add_argument(
        "--precision", choices=("float64", "webgpu"), default="float64"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--runtime", choices=("chromium", "deno"), default="chromium"
    )
    parser.add_argument(
        "--gpu-selectors",
        type=Path,
        help="JSON list of unique DRI_PRIME PCI selectors, one per worker",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, nargs=3, default=[2, 4, 4])
    parser.add_argument(
        "--optimizer", choices=("adam", "muon"), default="adam"
    )
    parser.add_argument("--muon-learning-rate", type=float, default=0.02)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-examples", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument(
        "--training-run",
        action="store_true",
        help="Measure training instead of replaying every step in Python",
    )
    args = parser.parse_args(argv)
    chrome = (
        os.environ.get("YGARATY_CHROME")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
    )
    if not chrome and platform.system() == "Darwin":
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if args.runtime == "chromium" and (
        not chrome or not Path(chrome).is_file()
    ):
        parser.error("Set YGARATY_CHROME to a Chrome/Chromium executable")
    deno = shutil.which("deno")
    selectors = None
    if args.runtime == "deno" and not deno:
        parser.error("Deno must be on PATH")
    if args.gpu_selectors:
        import re

        selectors = json.loads(args.gpu_selectors.read_text())
        if (
            args.runtime != "deno"
            or not isinstance(selectors, list)
            or len(selectors) != args.num_workers
            or any(
                not isinstance(x, str)
                or not re.fullmatch(
                    r"pci-[0-9a-fA-F]{4}_[0-9a-fA-F]{2}_[0-9a-fA-F]{2}_[0-7]!",
                    x,
                )
                for x in selectors
            )
            or len(set(selectors)) != len(selectors)
        ):
            parser.error(
                "Expected one distinct DRI_PRIME=pci-DDDD_BB_DD_F! "
                "selector per Deno worker"
            )
    if args.dataset == "fixture":
        rng = np.random.default_rng(44)
        dataset = ImageDataset(
            "synthetic-protocol-fixture",
            rng.integers(0, 256, (16, 1, 28, 28), dtype=np.uint8),
            np.arange(16, dtype=np.uint8) % 10,
            rng.integers(0, 256, (4, 1, 28, 28), dtype=np.uint8),
            np.arange(4, dtype=np.uint8),
        )
    else:
        dataset = load_dataset(args.dataset, args.data_dir)
    cfg = TrainingConfig(
        args.num_workers,
        rounds=args.rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        channels=tuple(args.channels),
        seed=args.seed,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        muon_learning_rate=args.muon_learning_rate,
        muon_momentum=args.muon_momentum,
        muon_steps=args.muon_steps,
        eval_examples=args.eval_examples,
        eval_interval=args.eval_interval,
        precision=args.precision,
    )
    import uvicorn
    from fastapi import Request
    from fastapi.responses import HTMLResponse, Response

    with tempfile.TemporaryDirectory(
        prefix="ygaraty-training-e2e-"
    ) as directory:
        root = Path(directory)
        session = TrainingSession(cfg, dataset, root / "browser")
        app = create_app(session)
        errors = []
        traffic = {"request_body_bytes": 0, "response_body_bytes": 0}
        evaluated = []
        timed_start = None

        class Meter:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return await self.app(scope, receive, send)

                async def counted_receive():
                    message = await receive()
                    traffic["request_body_bytes"] += len(
                        message.get("body", b"")
                    )
                    return message

                async def counted_send(message):
                    traffic["response_body_bytes"] += len(
                        message.get("body", b"")
                    )
                    await send(message)

                await self.app(scope, counted_receive, counted_send)

        # Test-only bootstrap routes. The product page requires a deliberate
        # Start click; automation invokes exactly the same contributor client.
        @app.get("/test")
        def bootstrap():
            return HTMLResponse(
                '<script type="module" src="/test.mjs"></script>'
            )

        @app.get("/test.mjs")
        def script():
            return Response(
                """
import {contribute} from '/assets/training_client.mjs';
try { await contribute({joinToken: TOKEN}); }
catch (error) {
  await fetch('/test-error', {method:'POST',
    body:JSON.stringify({error:String(error.stack||error)})});
}
""".replace("TOKEN", json.dumps(session.join_token)),
                media_type="text/javascript",
            )

        @app.post("/test-error")
        async def report_error(request: Request):
            errors.append(await request.json())
            return {"ok": True}

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(Meter(app), log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        thread.start()
        processes, logs = [], []
        start = time.monotonic()
        try:
            while not server.started:
                if not thread.is_alive() or time.monotonic() - start > 10:
                    raise RuntimeError("Training HTTP server did not start")
                time.sleep(0.05)
            for i in range(args.num_workers):
                log = (root / f"worker-{i}.log").open("w+")
                logs.append(log)
                if args.runtime == "deno":
                    from ygaraty.backends.webgpu.browser_assets import (
                        BROWSER_ROOT,
                    )

                    config_path = root / f"worker-{i}.json"
                    origin = f"http://127.0.0.1:{sock.getsockname()[1]}"
                    config_path.write_text(
                        json.dumps(
                            dict(
                                base_url=origin,
                                join_token=session.join_token,
                                client_id=f"native-{i:020}",
                                timeout_seconds=args.timeout,
                            )
                        )
                    )
                    config_path.chmod(0o600)
                    env = dict(os.environ)
                    if selectors:
                        env.update(
                            DRI_PRIME=selectors[i],
                            DENO_WEBGPU_BACKEND="vulkan",
                            VK_LOADER_LAYERS_ENABLE="VK_LAYER_MESA_device_select",
                        )
                    command = [
                        deno,
                        "run",
                        "--unstable-webgpu",
                        f"--allow-read={config_path}",
                        f"--allow-net=127.0.0.1:{sock.getsockname()[1]}",
                        str(BROWSER_ROOT / "native_diloco.mjs"),
                        str(config_path),
                    ]
                    processes.append(
                        subprocess.Popen(
                            command, stdout=log, stderr=log, env=env
                        )
                    )
                else:
                    flags = [
                        "--headless=new",
                        "--enable-unsafe-webgpu",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-background-timer-throttling",
                        "--disable-renderer-backgrounding",
                        f"--user-data-dir={root / f'profile-{i}'}",
                    ]
                    if (
                        platform.system() == "Linux"
                        and args.precision == "webgpu"
                    ):
                        flags.extend(
                            ["--enable-features=Vulkan", "--use-angle=vulkan"]
                        )
                    processes.append(
                        subprocess.Popen(
                            [
                                chrome,
                                *flags,
                                f"http://127.0.0.1:{sock.getsockname()[1]}/test",
                            ],
                            stdout=log,
                            stderr=log,
                        )
                    )
            while True:
                if errors or session.error:
                    raise RuntimeError(str(errors or session.error))
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError(
                        "Browser training exceeded its wall-clock limit"
                    )
                complete = False
                # Version changes before evaluation and durable save.
                # Observe only after the session transaction releases its lock.
                if session.lock.acquire(blocking=False):
                    try:
                        if timed_start is None and session.started is not None:
                            timed_start = session.started
                        for metric in session.metrics[len(evaluated) :]:
                            evaluated.append(
                                dict(
                                    version=metric["version"],
                                    seconds_since_launch=time.monotonic()
                                    - start,
                                    evaluation=metric.get("evaluation"),
                                )
                            )
                        complete = (
                            session.coordinator.version == cfg.rounds
                            and session.started is None
                        )
                    finally:
                        session.lock.release()
                if complete:
                    if (
                        evaluated[-1]["version"] != cfg.rounds
                        or evaluated[-1]["evaluation"] is None
                    ):
                        raise RuntimeError(
                            "Final committed evaluation was not observed"
                        )
                    break
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError(
                        "Contributor exited before completing training"
                    )
                time.sleep(0.1)
            elapsed = time.monotonic() - start
        except Exception as failure:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        kind="failed-browser-training",
                        error=str(failure),
                        config=asdict(cfg),
                        source_sha256=session.identity[
                            "implementation_sha256"
                        ],
                        dataset=session.identity["dataset"],
                        metrics=session.metrics,
                        worker_errors=[
                            log.read() for log in logs if not log.seek(0)
                        ],
                        seconds=time.monotonic() - start,
                        traffic=traffic,
                    ),
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            raise
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            server.should_exit = True
            thread.join(10)
            sock.close()
            for log in logs:
                log.close()

        error = None
        tolerance = 5e-5 if args.precision == "webgpu" else 1e-9
        if not args.training_run:
            baseline = TrainingSession(cfg, dataset, root / "reference")
            credentials = [
                baseline.join(baseline.join_token, f"reference-{i:016}")
                for i in range(cfg.num_workers)
            ]
            for _ in range(cfg.rounds):
                for i, c in enumerate(credentials):
                    w, token = c["worker"], c["token"]
                    bundle = baseline.assignment(w, token)
                    trainer = Trainer(w, baseline.recipe)
                    trainer.restore(
                        bundle["optimizer"], baseline.coordinator.weights
                    )
                    result = trainer.run_round(
                        baseline.coordinator.assignment(w),
                        dataset.batches(
                            i,
                            cfg.num_workers,
                            trainer.step,
                            cfg.local_steps,
                            cfg.batch_size,
                            cfg.seed,
                        ),
                    )
                    baseline.submit(
                        w,
                        token,
                        dict(
                            result=result.describe(),
                            optimizer=trainer.checkpoint(),
                        ),
                    )
            error = max(
                float(np.abs(session.coordinator.weights[n] - w).max())
                for n, w in baseline.coordinator.weights.items()
            )
            tolerance = 5e-5 if args.precision == "webgpu" else 1e-9
            if error > tolerance:
                raise AssertionError(
                    f"Contributor/reference error {error} exceeds {tolerance}"
                )
        stats = [
            s for m in session.metrics[1:] for s in m["worker_stats"].values()
        ]
        if any(
            s["gpu_tasks" if args.precision == "webgpu" else "cpu_tasks"] <= 0
            for s in stats
        ):
            raise AssertionError("Expected execution backend was not used")
        if implementation_hash() != session.identity["implementation_sha256"]:
            raise RuntimeError("Implementation changed during the experiment")
        report = dict(
            runtime=args.runtime,
            gpu_selectors=selectors,
            kind=(
                "same-host-native-training"
                if args.runtime == "deno"
                else "same-host-browser-training"
            )
            if args.training_run
            else (
                "same-host-native-correctness"
                if args.runtime == "deno"
                else "same-host-browser-correctness"
            ),
            reference_checked=not args.training_run,
            samples=cfg.rounds
            * cfg.num_workers
            * cfg.local_steps
            * cfg.batch_size,
            traffic=traffic,
            observations=evaluated,
            first_assignment_seconds=None
            if timed_start is None
            else timed_start - start,
            synthetic=args.dataset == "fixture",
            source_sha256=session.identity["implementation_sha256"],
            config=asdict(cfg),
            dataset=dict(name=dataset.name, sha256=dataset.sha256),
            seconds=elapsed,
            max_weight_error=error,
            tolerance=tolerance,
            metrics=session.metrics,
            platform=platform.platform(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"Passed: {args.num_workers} {args.runtime} contributors, "
            f"{args.precision}; report: {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
