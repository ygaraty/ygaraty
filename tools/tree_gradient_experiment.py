"""Local hardware-WebGPU test of sample-weighted tree gradients and shared Adam.

The reduction tree runs at the coordinator over HTTP-collected leaf gradients.
This measures the algorithm, not a peer-to-peer reduction network. No DiLoCo
outer optimizer or local worker Adam steps are used.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from ygaraty.training.convnet_webgpu import ConvBatch, ConvNetRecipe
from ygaraty.training.coordinator import copy_tensors, describe_tensors
from ygaraty.training.datasets import load_dataset
from ygaraty.training.gradient_reduction import (
    GradientPart,
    SharedAdam,
    compute_gradients,
    gradient_plan,
    reduce_gradients,
)
from ygaraty.training.session import implementation_hash


class GradientHub:
    """Bounded experiment transport. Incomplete steps abort; results retry idempotently."""

    def __init__(self, workers):
        self.workers = workers
        self.token = secrets.token_urlsafe(32)
        self.condition = threading.Condition()
        self.ready = set()
        self.jobs, self.results, self.digests = {}, {}, {}
        self.version = 0
        self.expected = None
        self.done = False
        self.errors = []
        self.request_bytes = self.response_bytes = 0

    def install(self, recipe, weights, batch, version):
        jobs = {}
        for worker, indices in enumerate(
            np.array_split(np.arange(len(batch.images)), self.workers)
        ):
            part = ConvBatch(batch.images[indices], batch.labels[indices])
            jobs[worker] = dict(
                state="gradient",
                version=version,
                samples=len(indices),
                weights=describe_tensors(weights),
                plan=gradient_plan(recipe, part).describe(),
            )
        with self.condition:
            self.expected = weights
            self.version = version
            self.jobs, self.results = jobs, {}

    def job(self, worker):
        with self.condition:
            if not 0 <= worker < self.workers:
                raise ValueError("Unknown gradient worker")
            self.ready.add(worker)
            self.condition.notify_all()
            if self.done:
                return dict(state="complete")
            if worker in self.results or worker not in self.jobs:
                return dict(state="waiting")
            return self.jobs[worker]

    def submit(self, worker, payload):
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        with self.condition:
            version = payload["version"]
            key = (worker, version)
            if key in self.digests:
                if digest != self.digests[key]:
                    raise ValueError("Conflicting gradient retry")
                return dict(accepted=True)
            if (
                worker not in self.jobs
                or type(version) is not int
                or version != self.version
            ):
                raise ValueError("Stale or unknown gradient worker")
            samples = payload["samples"]
            if (
                type(samples) is not int
                or samples != self.jobs[worker]["samples"]
            ):
                raise ValueError("Wrong partition sample count")
            loss = float(payload["loss"])
            if not math.isfinite(loss):
                raise ValueError("Nonfinite loss")
            gradients = copy_tensors(
                {
                    n: np.asarray(t["data"], dtype=np.float64).reshape(
                        t["shape"]
                    )
                    for n, t in payload["gradients"].items()
                },
                self.expected,
            )
            stats = payload["stats"]
            if (
                stats["gpu_tasks"] <= 0
                or stats["adapter"]["isFallbackAdapter"] is not False
            ):
                raise ValueError("Hardware GPU execution required")
            self.results[worker] = (
                GradientPart(samples, loss, gradients),
                stats,
            )
            # Keep receipts for only this/previous step; covers a lost HTTP ACK.
            self.digests = {
                k: v for k, v in self.digests.items() if k[1] >= version - 1
            }
            self.digests[key] = digest
            self.condition.notify_all()
            return dict(accepted=True)

    def wait_for(self, predicate, deadline):
        with self.condition:
            while not predicate():
                if self.errors:
                    raise RuntimeError(str(self.errors))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gradient experiment deadline exceeded")
                self.condition.wait(min(remaining, 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-workers", type=int, choices=(1, 3), required=True
    )
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--global-batch", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--eval-examples", type=int, default=1024)
    parser.add_argument("--verify-steps", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        not args.num_workers <= args.global_batch <= 48
        or args.steps < 1
        or args.seed < 0
        or args.eval_every < 1
        or not 1 <= args.eval_examples <= 10000
        or not 0 <= args.verify_steps <= args.steps
        or not 0 < args.timeout <= 3600
    ):
        parser.error("Invalid experiment bounds")
    chrome = os.environ.get(
        "YGARATY_CHROME",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    )
    if not Path(chrome).is_file():
        parser.error("Set YGARATY_CHROME to the Chrome executable")
    dataset = load_dataset("fashion-mnist", args.data_dir)
    recipe = ConvNetRecipe(
        dict(
            in_channels=1,
            height=28,
            width=28,
            channels=[8, 16, 16],
            residual=True,
        ),
        max_batch=args.global_batch,
    )
    optimizer = SharedAdam(recipe, recipe.initial_weights(args.seed))
    initial = dataset.evaluate(recipe, optimizer.weights, args.eval_examples)
    hub = GradientHub(args.num_workers)
    source_hash = implementation_hash()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        kind="tree-gradient-experiment",
        status="running",
        source_sha256=source_hash,
        dataset=dict(name=dataset.name, sha256=dataset.sha256),
        config={
            k: v
            for k, v in vars(args).items()
            if k not in ("output", "data_dir")
        },
        model=recipe.config,
        learning_rate=recipe.learning_rate,
        optimizer=recipe.optimizer,
        topology="pairwise sample-weighted reduction on coordinator; HTTP star transport",
        metrics=[dict(step=0, samples=0, seconds=0, evaluation=initial)],
        parity=[],
        platform=platform.platform(),
        chrome=subprocess.check_output(
            [chrome, "--version"], text=True
        ).strip(),
    )
    start = None
    checkpoints = {}
    sample_hash = hashlib.sha256()
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, Response, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from ygaraty.backends.webgpu.browser_assets import BROWSER_ROOT

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/assets", StaticFiles(directory=BROWSER_ROOT), name="assets")

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/test/{worker}")
    def page(worker: int):
        return HTMLResponse(
            f'<script type="module" src="/script/{worker}"></script>'
        )

    @app.get("/script/{worker}")
    def script(worker: int):
        return Response(
            """
import {contributeGradients} from '/assets/gradient_client.mjs';
const token=TOKEN;
try { await contributeGradients({worker:WORKER,token}); }
catch(e) { await fetch('/failure',{method:'POST',headers:{Authorization:`Bearer ${token}`},body:JSON.stringify({error:String(e.stack||e)})}); }
""".replace("TOKEN", json.dumps(hub.token)).replace("WORKER", str(worker)),
            media_type="text/javascript",
        )

    def authorize(request):
        if not secrets.compare_digest(
            request.headers.get("Authorization", ""), "Bearer " + hub.token
        ):
            raise PermissionError("Unauthorized")

    @app.get("/gradient/{worker}/job")
    def job(worker: int, request: Request):
        try:
            authorize(request)
            return hub.job(worker)
        except PermissionError:
            return JSONResponse(dict(error="Unauthorized"), status_code=401)
        except ValueError as error:
            return JSONResponse(dict(error=str(error)), status_code=409)

    async def bounded_body(request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("Gradient body too large")
        return json.loads(data)

    @app.post("/gradient/{worker}/result")
    async def result(worker: int, request: Request):
        try:
            authorize(request)
            return hub.submit(worker, await bounded_body(request))
        except PermissionError:
            return JSONResponse(dict(error="Unauthorized"), status_code=401)
        except (ValueError, KeyError, TypeError) as error:
            return JSONResponse(dict(error=str(error)), status_code=409)

    @app.post("/failure")
    async def failure(request: Request):
        try:
            authorize(request)
            value = await bounded_body(request)
            with hub.condition:
                hub.errors.append(value)
                hub.condition.notify_all()
            return dict(ok=True)
        except PermissionError:
            return JSONResponse(dict(error="Unauthorized"), status_code=401)

    class Meter:
        async def __call__(self, scope, receive, send):
            async def read():
                message = await receive()
                hub.request_bytes += len(message.get("body", b""))
                return message

            async def write(message):
                hub.response_bytes += len(message.get("body", b""))
                await send(message)

            await app(scope, read, write)

    with tempfile.TemporaryDirectory(prefix="ygaraty-gradient-") as directory:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(Meter(), log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs=dict(sockets=[sock]), daemon=True
        )
        processes, logs = [], []
        try:
            start = time.monotonic()
            deadline = start + args.timeout
            thread.start()
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("Gradient HTTP server failed to start")
                time.sleep(0.05)
            for worker in range(args.num_workers):
                log = (Path(directory) / f"chrome-{worker}.log").open("w")
                logs.append(log)
                flags = [
                    "--headless=new",
                    "--enable-unsafe-webgpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    f"--user-data-dir={directory}/profile-{worker}",
                ]
                if platform.system() == "Linux":
                    flags += ["--enable-features=Vulkan", "--use-angle=vulkan"]
                processes.append(
                    subprocess.Popen(
                        [
                            chrome,
                            *flags,
                            f"http://127.0.0.1:{port}/test/{worker}",
                        ],
                        stdout=log,
                        stderr=log,
                    )
                )
            hub.wait_for(
                lambda: len(hub.ready) == args.num_workers,
                min(deadline, start + 60),
            )
            report["ready_seconds"] = time.monotonic() - start
            for step in range(1, args.steps + 1):
                step_start = time.monotonic()
                if step_start >= deadline:
                    raise TimeoutError("Training deadline exceeded")
                ids = dataset.indices(
                    0,
                    1,
                    (step - 1) * args.global_batch,
                    args.global_batch,
                    args.seed,
                )
                sample_hash.update(ids.astype("<i8").tobytes())
                batch = ConvBatch(
                    dataset.train_images[ids], dataset.train_labels[ids]
                )
                hub.install(recipe, optimizer.weights, batch, step)
                hub.wait_for(
                    lambda: len(hub.results) == args.num_workers,
                    min(deadline, step_start + 60),
                )
                parts = [hub.results[w][0] for w in range(args.num_workers)]
                reduced = reduce_gradients(parts)
                if step <= args.verify_steps:
                    expected = compute_gradients(
                        recipe, optimizer.weights, batch
                    )
                    difference = max(
                        float(np.abs(reduced.gradients[n] - g).max())
                        for n, g in expected.gradients.items()
                    )
                    loss_difference = abs(reduced.loss - expected.loss)
                    report["parity"].append(
                        dict(
                            step=step,
                            max_gradient_error=difference,
                            loss_error=loss_difference,
                        )
                    )
                    if difference > 2e-5 or loss_difference > 2e-5:
                        raise AssertionError(
                            f"CPU reference gradient/loss gate failed: {difference}, {loss_difference}"
                        )
                if step == 1:
                    checkpoints.update(
                        {
                            f"first_gradient_{n}": g
                            for n, g in reduced.gradients.items()
                        }
                    )
                optimizer.apply(reduced.gradients)
                if step == 1:
                    checkpoints.update(
                        {
                            f"first_weight_{n}": w
                            for n, w in optimizer.weights.items()
                        }
                    )
                metric = dict(
                    step=step,
                    samples=step * args.global_batch,
                    train_loss=reduced.loss,
                    update_seconds=time.monotonic() - step_start,
                    worker_stats={
                        str(w): hub.results[w][1]
                        for w in range(args.num_workers)
                    },
                )
                if step % args.eval_every == 0 or step == args.steps:
                    metric["evaluation"] = dataset.evaluate(
                        recipe, optimizer.weights, args.eval_examples
                    )
                metric["seconds"] = time.monotonic() - start
                report["metrics"].append(metric)
                if "evaluation" in metric:
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in metric.items()
                                if k != "worker_stats"
                            }
                        ),
                        flush=True,
                    )
            # Persist the final weights and shared optimizer moments before timing stops.
            checkpoints.update(
                {f"final_weight_{n}": w for n, w in optimizer.weights.items()}
            )
            for kind, values in optimizer.moments.items():
                checkpoints.update(
                    {f"final_{kind}_{n}": v for n, v in values.items()}
                )
            np.savez(args.output.with_suffix(".npz"), **checkpoints)
            report["seconds"] = time.monotonic() - start
            report["sample_sequence_sha256"] = sample_hash.hexdigest()
            if implementation_hash() != source_hash:
                raise RuntimeError("Implementation changed during experiment")
            report["status"] = "passed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            raise
        finally:
            hub.done = True
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            server.should_exit = True
            if thread.is_alive():
                thread.join(10)
            sock.close()
            for log in logs:
                log.close()
            report["cleanup"] = dict(
                browser_exit_codes=[p.returncode for p in processes],
                server_stopped=not thread.is_alive(),
            )
            report["traffic"] = dict(
                request_body_bytes=hub.request_bytes,
                response_body_bytes=hub.response_bytes,
            )
            args.output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
