"""Run a small local DiLoCo experiment with the existing translation
Transformer.

This driver runs replicas sequentially to check the algorithm, not to measure
distributed throughput. No database, server, PyTorch or downloads are required.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ygaraty.training import (
    Batch,
    Coordinator,
    DiLoCoConfig,
    Trainer,
    TransformerRecipe,
)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lan":
        from .lan import main as lan

        return lan(argv[1:])
    if argv and argv[0] == "serve":
        from .launch import main as serve

        return serve(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--local-steps",
        type=int,
        default=2,
        help="Updates per round (typically 100–500 in larger experiments)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON model config; defaults to a tiny Transformer",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="NPZ with padded source, target_input and targets arrays; "
        "sharded by worker",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="NPZ with existing Transformer parameters, using PyTorch names",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--outer-learning-rate", type=float, default=0.7)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Zero selects outer SGD; otherwise use Nesterov",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Use browser-compatible primitive operations instead of fused "
        "layers",
    )
    parser.add_argument(
        "--export-round",
        type=Path,
        help="Save worker 0’s first portable round for browser/Node execution",
    )
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.workers < 1:
        parser.error("rounds and workers must be positive")
    config = (
        json.loads(args.config.read_text())
        if args.config
        else dict(
            vocab=16,
            embedding=8,
            hidden=16,
            layers=1,
            heads=2,
            max_length=8,
            dropout=0.0,
        )
    )
    recipe = TransformerRecipe(
        config,
        learning_rate=args.learning_rate,
        tile_size=args.tile_size,
        portable=args.portable or bool(args.export_round),
    )
    if args.weights:
        with np.load(args.weights, allow_pickle=False) as file:
            weights = {
                name: file[name].reshape(spec["shape"])
                for name, spec in recipe.specs.items()
            }
    else:
        weights = recipe.initial_weights()
    if args.batch:
        with np.load(args.batch, allow_pickle=False) as file:
            source, inputs, targets = (
                file[name] for name in ("source", "target_input", "targets")
            )
        if len(source) < args.workers:
            parser.error("batch needs at least one example per worker")
    else:
        # A deterministic token fixture: independent examples with EOS=3,
        # BOS=2.
        if config["vocab"] < 5:
            parser.error(
                "the built-in fixture needs vocab >= 5; supply --batch"
            )
        ids = np.arange(args.workers) % (config["vocab"] - 4) + 4
        source = np.column_stack([ids, np.full(args.workers, 3)])
        inputs = np.column_stack([np.full(args.workers, 2), ids])
        targets = source.copy()
    shards = [
        Batch(
            source[i :: args.workers],
            inputs[i :: args.workers],
            targets[i :: args.workers],
        )
        for i in range(args.workers)
    ]
    coordinator = Coordinator(
        weights,
        DiLoCoConfig(
            args.local_steps,
            args.outer_learning_rate,
            args.momentum,
            args.momentum > 0,
        ),
    )
    trainers = [Trainer(f"worker-{i}", recipe) for i in range(args.workers)]
    for index in range(args.rounds):
        coordinator.begin_round(t.worker for t in trainers)
        for trainer, shard in zip(trainers, shards):
            assignment = coordinator.assignment(trainer.worker)
            if args.export_round and index == 0 and trainer is trainers[0]:
                bundle = recipe.round_bundle(
                    assignment, [shard] * args.local_steps
                )
                with args.export_round.open("x") as file:
                    json.dump(
                        bundle, file, allow_nan=False, separators=(",", ":")
                    )
            coordinator.submit(
                trainer.run_round(
                    assignment, (shard for _ in range(args.local_steps))
                )
            )
        print(json.dumps(coordinator.commit_round(), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
