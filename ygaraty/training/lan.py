"""Native wgpu workers and a fixed-membership, binary peer-tree experiment.

This uses synchronous data parallelism with one shared Adam optimizer.
Only control, plans, batches and root checkpoints pass through this driver.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .tensor_transport import CONTENT_TYPE, MAX_FRAME, decode, encode


class PeerClient:
    def __init__(self, config):
        self.config = config
        self.sent = self.received = 0

    def request(self, rank, path, metadata, tensors=None):
        body = encode(dict(metadata, session=self.config["session"]), tensors)
        request = urllib.request.Request(
            self.config["peers"][rank] + path,
            data=body,
            headers={
                "Authorization": "Bearer " + self.config["token"],
                "Content-Type": CONTENT_TYPE,
            },
        )
        self.sent += len(body)
        # LAN requests must not route through an ambient public HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(
                request, timeout=self.config["timeoutMs"] / 1000
            )
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read(MAX_FRAME + 1)
            self.received += len(data)
            result = decode(data)
            if response.status != 200:
                raise RuntimeError(
                    f"Peer {rank}: {result[0].get('error', response.status)}"
                )
        return result


def optimizer_plan(recipe, weights, version):
    from ygaraty.compiler.optimizer import with_optimizer

    refs = {n: f"gradient_{i}" for i, n in enumerate(weights)}
    shapes = {n: w.shape for n, w in weights.items()}
    shapes.update({refs[n]: w.shape for n, w in weights.items()})
    shapes.update(
        {
            f"adam_{kind}_{n}": w.shape
            for kind in ("m", "v")
            for n, w in weights.items()
        }
    )
    tasks, updates = with_optimizer(
        [],
        shapes,
        list(weights),
        refs,
        recipe.tile_size,
        recipe.learning_rate,
        "adam",
        version,
        recipe.optimizer,
    )
    return dict(
        step=version,
        loss=next(iter(updates.values())),
        tasks=[task.describe() for task in tasks],
        constants={},
        parameters={},
        updates=updates,
    ), refs


def initialize(directory, hosts, workers, precision="webgpu", timeout=60):
    from .session import implementation_hash

    peers = [value.rstrip("/") for value in hosts]
    if len(peers) != workers or len(set(peers)) != workers:
        raise ValueError("Provide one unique reachable origin URL per worker")
    for peer in peers:
        url = urllib.parse.urlsplit(peer)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.path
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Expected origin URLs, e.g. http://192.168.1.10:8780"
            )
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    config = dict(
        peers=peers,
        token=secrets.token_urlsafe(32),
        session=secrets.token_hex(16),
        precision=precision,
        timeoutMs=timeout * 1000,
        source_sha256=implementation_hash(),
    )
    for name, contents in [
        ("coordinator.json", config),
        *(
            (f"worker-{rank}.json", dict(config, rank=rank))
            for rank in range(workers)
        ),
    ]:
        path = directory / name
        with open(
            path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as file:
            json.dump(contents, file, indent=2)
    return config


def run(args, config):
    from .convnet_webgpu import ConvBatch, ConvNetRecipe
    from .datasets import ImageDataset, load_dataset
    from .gradient_reduction import (
        SharedAdam,
        compute_gradients,
        gradient_plan,
    )
    from .session import implementation_hash

    if args.num_workers != len(config["peers"]):
        raise ValueError(
            "--num-workers must match the generated cluster configuration"
        )
    if not args.num_workers <= args.global_batch <= 64:
        raise ValueError("global batch must be between num-workers and 64")
    source = implementation_hash()
    if source != config["source_sha256"]:
        raise ValueError("Source changed since init; generate a new session")
    if args.dataset == "fixture":
        rng = np.random.default_rng(17)
        dataset = ImageDataset(
            "fixture",
            rng.integers(0, 256, (80, 1, 28, 28), dtype=np.uint8),
            np.arange(80, dtype=np.uint8) % 10,
            rng.integers(0, 256, (16, 1, 28, 28), dtype=np.uint8),
            np.arange(16, dtype=np.uint8) % 10,
        )
    else:
        dataset = load_dataset(args.dataset, args.data_dir)
    recipe = ConvNetRecipe(
        dict(
            in_channels=1,
            height=28,
            width=28,
            channels=args.channels,
            residual=True,
        ),
        max_batch=args.global_batch,
    )
    weights = recipe.initial_weights(args.seed)
    reference = SharedAdam(recipe, weights)
    client = PeerClient(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise ValueError("Output already exists; choose a new report path")
    report = dict(
        kind="native-peer-tree",
        status="running",
        source_sha256=source,
        dataset=dict(name=dataset.name, sha256=dataset.sha256),
        config={
            key: value
            for key, value in vars(args).items()
            if key not in ("config", "data_dir", "output", "command")
        },
        topology=(
            "binary heap; peer gradient reduction and weight broadcast; "
            "root Adam"
        ),
        model=recipe.config,
        learning_rate=recipe.learning_rate,
        optimizer=recipe.optimizer,
        precision=config["precision"],
        metrics=[],
        parity=[],
        workers=[],
    )
    start = time.monotonic()
    sample_hash = hashlib.sha256()
    try:
        report["workers"] = [
            client.request(rank, "/status", {})[0]
            for rank in range(args.num_workers)
        ]
        if any(
            status["version"] != 0 or status["failed"]
            for status in report["workers"]
        ):
            raise ValueError("Start fresh workers for each experiment")
        client.request(0, "/commit", dict(version=0), weights)
        report["metrics"].append(
            dict(
                step=0,
                seconds=0,
                samples=0,
                evaluation=dataset.evaluate(
                    recipe, weights, args.eval_examples
                ),
            )
        )
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            for version in range(1, args.steps + 1):
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError("Experiment deadline reached")
                ids = dataset.indices(
                    0,
                    1,
                    (version - 1) * args.global_batch,
                    args.global_batch,
                    args.seed,
                )
                sample_hash.update(np.asarray(ids, dtype="<i8").tobytes())
                batch = ConvBatch(
                    dataset.train_images[ids], dataset.train_labels[ids]
                )
                shards = np.array_split(
                    np.arange(args.global_batch), args.num_workers
                )
                counts = [len(shard) for shard in shards]
                optimizer, refs = optimizer_plan(recipe, weights, version)
                futures = []
                for rank, shard in enumerate(shards):
                    plan = gradient_plan(
                        recipe,
                        ConvBatch(batch.images[shard], batch.labels[shard]),
                    )
                    wire = plan.describe()
                    wire["constants"] = {}
                    metadata = dict(version=version, counts=counts, plan=wire)
                    if rank == 0:
                        metadata.update(optimizer=optimizer, gradientRefs=refs)
                    futures.append(
                        pool.submit(
                            client.request,
                            rank,
                            "/prepare",
                            metadata,
                            plan.constants,
                        )
                    )
                for future in futures:
                    future.result()
                reduced, weights = client.request(
                    0, "/step", dict(version=version)
                )
                if (
                    reduced["version"] != version
                    or reduced["samples"] != args.global_batch
                ):
                    raise ValueError("Invalid collective receipt")
                nodes = reduced["nodes"]
                if sorted(node["rank"] for node in nodes) != list(
                    range(args.num_workers)
                ):
                    raise ValueError("Incomplete collective membership")
                if version <= args.verify_steps:
                    part = compute_gradients(recipe, reference.weights, batch)
                    reference.apply(part.gradients)
                    error = max(
                        float(
                            np.max(np.abs(weights[n] - reference.weights[n]))
                        )
                        for n in weights
                    )
                    report["parity"].append(
                        dict(step=version, max_weight_error=error)
                    )
                    if error > 2e-5:
                        raise ValueError(
                            f"CPU reference parity failed: {error}"
                        )
                metric = dict(
                    step=version,
                    samples=version * args.global_batch,
                    loss=reduced["loss"],
                    nodes=nodes,
                )
                if version % args.eval_interval == 0 or version == args.steps:
                    metric["evaluation"] = dataset.evaluate(
                        recipe, weights, args.eval_examples
                    )
                metric["seconds"] = time.monotonic() - start
                report["metrics"].append(metric)
                print(
                    json.dumps(
                        {k: v for k, v in metric.items() if k != "nodes"}
                    ),
                    flush=True,
                )
        report["workers"] = [
            client.request(rank, "/status", {})[0]
            for rank in range(args.num_workers)
        ]
        if implementation_hash() != source:
            raise RuntimeError("Runtime source changed during experiment")
        np.savez(args.output.with_suffix(".npz"), **weights)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(
            seconds=time.monotonic() - start,
            sample_sequence_sha256=sample_hash.hexdigest(),
            coordinator_sent_bytes=client.sent,
            coordinator_received_bytes=client.received,
        )
        args.output.write_text(
            json.dumps(report, allow_nan=False, indent=2) + "\n"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="Write private per-worker configurations"
    )
    init.add_argument(
        "--num-workers", type=int, choices=(1, 2, 4, 10), required=True
    )
    init.add_argument("--hosts", nargs="+", required=True)
    init.add_argument("--directory", type=Path, required=True)
    init.add_argument(
        "--cpu-test",
        action="store_true",
        help="CPU protocol tests only; never GPU evidence",
    )
    worker = commands.add_parser(
        "worker", help="Serve a native Deno/wgpu worker"
    )
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Private interface address to listen on",
    )
    worker.add_argument("--port", type=int, default=8780)
    experiment = commands.add_parser(
        "run", help="Run matched-batch peer-tree training"
    )
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument(
        "--num-workers", type=int, choices=(1, 2, 4, 10), required=True
    )
    experiment.add_argument(
        "--dataset",
        choices=("mnist", "fashion-mnist", "fixture"),
        default="fashion-mnist",
    )
    experiment.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--steps", type=int, default=100)
    experiment.add_argument("--global-batch", type=int, default=40)
    experiment.add_argument(
        "--channels", type=int, nargs=3, default=[8, 16, 16]
    )
    experiment.add_argument("--seed", type=int, default=2026)
    experiment.add_argument("--eval-examples", type=int, default=1024)
    experiment.add_argument("--eval-interval", type=int, default=20)
    experiment.add_argument("--verify-steps", type=int, default=0)
    experiment.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.command == "init":
        initialize(
            args.directory,
            args.hosts,
            args.num_workers,
            "float64" if args.cpu_test else "webgpu",
        )
        print(f"Private session configurations written to {args.directory}")
        return 0
    config = json.loads(args.config.read_text())
    from .session import implementation_hash

    if implementation_hash() != config["source_sha256"]:
        parser.error("Worker source must match the source used by lan init")
    if args.command == "worker":
        from ygaraty.backends.webgpu.browser_assets import BROWSER_ROOT

        deno = shutil.which("deno")
        if not deno:
            parser.error("Install Deno 2 with WebGPU support first")
        if not 1 <= args.port <= 65535:
            parser.error("Invalid port")
        network = {
            urllib.parse.urlsplit(url).netloc for url in config["peers"]
        }
        network.add(f"{args.bind}:{args.port}")
        # Replace the launcher so signals and process supervision address Deno
        # itself, leaving no orphan worker when an SSH/session process stops.
        os.execv(
            deno,
            [
                deno,
                "run",
                "--no-prompt",
                "--unstable-webgpu",
                "--allow-net=" + ",".join(sorted(network)),
                "--allow-read=" + str(args.config.resolve()),
                str(BROWSER_ROOT / "native_training.mjs"),
                str(args.config.resolve()),
                args.bind,
                str(args.port),
            ],
        )
    if (
        min(args.steps, args.eval_examples, args.eval_interval, args.timeout)
        < 1
        or args.verify_steps < 0
    ):
        parser.error("Positive run limits are required")
    run(args, config)
    return 0
