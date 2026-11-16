"""Reusable float64 references for browser factorization experiments.

Hosted application orchestration is archived; these helpers need only NumPy
and public model initializers.
"""

import numpy as np


def factor_space_reference(a, args):
    """Exact float64 gradient descent for X = a a^T / sqrt(d) without forming
    X: the loss before
    each update and ||X||^2 / (2 N^2) (the loss of U V^T = 0)."""
    from ygaraty.models.factorization_sharded import (
        initializers,
        join_shards,
        shard_values,
    )

    specs = initializers(
        args.rows, args.rows, args.rank, args.block, args.block, seed=args.seed
    )

    def blocks(prefix):
        return sorted(
            (n for n in specs if n.startswith(prefix)),
            key=lambda n: int(n.split("/")[1]),
        )

    u = join_shards(
        {n: shard_values(specs[n]) for n in blocks("U/")}, blocks("U/")
    )
    v = join_shards(
        {n: shard_values(specs[n]) for n in blocks("V/")}, blocks("V/")
    )
    n, root, gram = args.rows * args.rows, np.sqrt(a.shape[1]), a.T @ a
    x_norm = float(np.sum(gram * gram)) / a.shape[1]
    losses = []
    for _ in range(args.updates):
        ua, va = u.T @ a, v.T @ a  # rank x d
        fit = (
            float(np.sum((u.T @ u) * (v.T @ v)))
            - 2 * float(np.sum(ua * va)) / root
            + x_norm
        )
        losses.append(
            fit / (2 * n)
            + args.regularization
            / 2
            * (float(np.sum(u * u)) + float(np.sum(v * v)))
        )
        gu = (u @ (v.T @ v) - a @ va.T / root) / n + args.regularization * u
        gv = (v @ (u.T @ u) - a @ ua.T / root) / n + args.regularization * v
        u, v = u - args.learning_rate * gu, v - args.learning_rate * gv
    return losses, x_norm / (2 * n)


def dense_reference(x, args):
    """Dense float64 SGD on a given X with the job's seeded U and V: the loss
    before each update."""
    from ygaraty.models.factorization_sharded import (
        initializers,
        join_shards,
        shard_values,
    )

    specs = initializers(
        args.rows, args.rows, args.rank, args.block, args.block, seed=args.seed
    )

    def blocks(prefix):
        return sorted(
            (n for n in specs if n.startswith(prefix)),
            key=lambda n: int(n.split("/")[1]),
        )

    u = join_shards(
        {n: shard_values(specs[n]) for n in blocks("U/")}, blocks("U/")
    )
    v = join_shards(
        {n: shard_values(specs[n]) for n in blocks("V/")}, blocks("V/")
    )
    n, losses = x.size, []
    for _ in range(args.updates):
        residual = u @ v.T - x
        losses.append(
            float(
                np.sum(residual**2) / (2 * n)
                + args.regularization / 2 * (np.sum(u**2) + np.sum(v**2))
            )
        )
        gu, gv = (
            residual @ v / n + args.regularization * u,
            residual.T @ u / n + args.regularization * v,
        )
        u, v = u - args.learning_rate * gu, v - args.learning_rate * gv
    return losses
