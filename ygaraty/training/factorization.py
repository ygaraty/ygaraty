"""Dense factorization SGD reference; no host services required."""

import numpy as np


def reference_losses(args, steps):
    if args.rows * args.columns * 8 > 2 << 30:
        raise SystemExit(
            "--check needs the dense matrix in memory; it is limited to 2 "
            "GiB of X"
        )
    """Dense NumPy SGD on the same seeded problem: the loss before each update."""  # noqa: E501 -- preserve literal/report layout
    from ygaraty.models.factorization_sharded import (
        initializers,
        join_shards,
        shard_values,
    )

    specs = initializers(
        args.rows,
        args.columns,
        args.rank,
        args.block,
        args.block,
        seed=args.seed,
    )

    def blocks(prefix):
        return sorted(
            (n for n in specs if n.startswith(prefix)),
            key=lambda n: int(n.split("/")[1]),
        )

    row_names, column_names = blocks("U/"), blocks("V/")
    u = join_shards({n: shard_values(specs[n]) for n in row_names}, row_names)
    v = join_shards(
        {n: shard_values(specs[n]) for n in column_names}, column_names
    )
    x = np.block(
        [
            [
                shard_values(specs[f"X/{i}/{j}"])
                for j in range(len(column_names))
            ]
            for i in range(len(row_names))
        ]
    )
    n, losses = x.size, []
    for _ in range(steps):
        residual = u @ v.T - x
        losses.append(
            float(
                np.sum(residual**2) / (2 * n)
                + args.regularization / 2 * (np.sum(u**2) + np.sum(v**2))
            )
        )
        gu = residual @ v / n + args.regularization * u
        gv = residual.T @ u / n + args.regularization * v
        u, v = u - args.learning_rate * gu, v - args.learning_rate * gv
    return losses
