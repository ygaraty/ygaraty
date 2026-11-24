"""Muon momentum and Newton–Schulz updates expressed as portable Tape ops.

Uses the five-step quintic iteration from Keller Jordan's MIT Muon reference.
Convolution matrices use [out_channels, in_channels * kernel_area].
No optimizer
backpropagation is requested; these nodes are appended after model backward.
"""

import math


def muon_options(momentum=0.95, steps=5):
    if (
        type(momentum) not in (int, float)
        or not math.isfinite(momentum)
        or not 0 <= momentum < 1
    ):
        raise ValueError("Muon momentum must be finite in [0, 1)")
    if type(steps) is not int or not 1 <= steps <= 10:
        raise ValueError("Muon Newton-Schulz steps must be in 1..10")
    return dict(momentum=momentum, steps=steps)


def append_muon(tape, parameters, learning_rate, *, momentum=0.95, steps=5):
    """Return parameter/state updates; preserve whole-matrix orthogonalization.

    Momentum uses the reference EMA convention: m = beta*m + (1-beta)*g,
    Nesterov direction = beta*m + (1-beta)*g. Unlike Adam, no bias correction.
    Normalize in float64 CPU / compensated GPU arithmetic, not BF16.
    """
    muon_options(momentum, steps)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Muon learning rate must be positive and finite")
    start = len(tape.nodes)
    updates = {}
    for p in parameters:
        rows, cols = tape.shapes[p]
        state = tape.declare(f"muon_m_{p}", (rows, cols), dict(kind="zeros"))
        g = tape.gradients[p]
        m = tape.add(tape.scale(state, momentum), tape.scale(g, 1 - momentum))
        x = tape.add(tape.scale(m, momentum), tape.scale(g, 1 - momentum))
        squares = tape.sum(tape.mul(x, x))
        inverse = tape._op(
            "inverse_norm", [squares], (1, 1), dict(epsilon=1e-7)
        )
        x = tape.mul(x, tape._broadcast(inverse, (rows, cols)))
        if rows > cols:
            x = tape.transpose(x)
        for _ in range(steps):
            a = tape.matmul(x, tape.transpose(x))
            b = tape.add(
                tape.scale(a, -4.7750), tape.scale(tape.matmul(a, a), 2.0315)
            )
            x = tape.add(tape.scale(x, 3.4445), tape.matmul(b, x))
        if rows > cols:
            x = tape.transpose(x)
        updates[p] = tape.add(
            tape.scale(x, -learning_rate * math.sqrt(max(1, rows / cols))), p
        )
        updates[state] = m
    for node in tape.nodes[start:]:
        node.attrs["optimizer"] = True
    return updates
