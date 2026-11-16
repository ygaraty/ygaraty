"""Kernels for batched sequences: the fused LSTM step, its gradients, and
batched attention.

Sequences are time-major: row t*batch+b is step t of example b. An `lstm` value
holds, per row, [h_state | c_state | h_out | i | f | g | o], each `hidden` wide.
Whole-layer kernels (lstm, lstm_backward) run every step in turn and define the
result; the planner splits them into one task per step that calls the same step
functions, so both give identical numbers.
"""  # noqa: E501 -- preserve documentation layout

import numpy as np

from ygaraty.compiler.layouts import C_STATE, GATES, H_OUT, H_STATE


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def block(a, index, hidden, count=1):
    return a[:, index * hidden : (index + count) * hidden]


def recurrent(partials, blocks, rows):
    """Sum matmul partial products laid out as [[col_start, col_stop, count],
    ...]."""
    width = blocks[-1][1] if blocks else 0
    out, k = np.zeros((rows, width)), 0
    for start, stop, count in blocks:
        for _ in range(count):
            out[:, start:stop] += partials[k]
            k += 1
    return out


def cell(xp, hw, bias_ih, bias_hh, h_prev, c_prev, mask):
    """One step: returns [h_state | c_state | h_out | i | f | g | o]."""
    hidden = h_prev.shape[1]
    # Same order as PyTorch: (x @ W_ih.T + b_ih) + (h @ W_hh.T + b_hh).
    a = (xp + bias_ih) + (hw + bias_hh)
    i, f, o = (sigmoid(a[:, k * hidden : (k + 1) * hidden]) for k in (0, 1, 3))
    g = np.tanh(a[:, 2 * hidden : 3 * hidden])
    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    live = mask > 0
    return np.concatenate(
        [
            np.where(live, h_new, h_prev),
            np.where(live, c_new, c_prev),
            np.where(live, h_new, 0.0),
            i,
            f,
            g,
            o,
        ],
        axis=1,
    )


def cell_backward(state, c_prev, mask, dh_state, dc_state, dh_out):
    """Gate gradients (rows x 4h) and the carry [dh_prev passed through | dc_prev]."""  # noqa: E501 -- preserve documentation layout
    hidden = c_prev.shape[1]
    i, f, g, o = (block(state, GATES + k, hidden) for k in range(4))
    live = mask > 0
    tc = np.tanh(block(state, C_STATE, hidden))  # c_new on live rows
    dh_new = np.where(live, dh_state + dh_out, 0.0)
    dc_new = np.where(live, dc_state + dh_new * o * (1 - tc * tc), 0.0)
    gates = np.concatenate(
        [
            dc_new * g * i * (1 - i),
            dc_new * c_prev * f * (1 - f),
            dc_new * i * (1 - g * g),
            dh_new * tc * o * (1 - o),
        ],
        axis=1,
    )
    carry = np.concatenate(
        [
            np.where(live, 0.0, dh_state),
            dc_new * f + np.where(live, 0.0, dc_state),
        ],
        axis=1,
    )
    return gates, carry


def order(steps, reverse):
    return range(steps - 1, -1, -1) if reverse else range(steps)


def lstm(xp, whh_t, bias_ih, bias_hh, mask, h0, c0, batch, hidden, reverse):
    steps = len(xp) // batch
    out = np.zeros((len(xp), 7 * hidden))
    h, c = h0, c0
    for t in order(steps, reverse):
        rows = slice(t * batch, (t + 1) * batch)
        state = cell(xp[rows], h @ whh_t, bias_ih, bias_hh, h, c, mask[rows])
        out[rows] = state
        h, c = block(state, H_STATE, hidden), block(state, C_STATE, hidden)
    return out


def lstm_backward(inputs, attrs):
    batch, hidden, reverse = attrs["batch"], attrs["hidden"], attrs["reverse"]
    values = list(inputs)
    out_value = values.pop(0)
    full = values.pop(0) if attrs.get("full") else None
    g_out = values.pop(0) if attrs.get("has_out") else None
    g_state = values.pop(0) if attrs.get("has_state") else None
    whh, mask, h0, c0 = values
    rows = len(out_value)
    steps = rows // batch
    if full is not None:
        g_out = block(full, H_OUT, hidden)
    result = np.zeros((rows + batch, 6 * hidden))
    walk = list(order(steps, reverse))
    _final = walk[-1]
    dh = np.zeros((batch, hidden))
    dc = np.zeros((batch, hidden))
    if g_state is not None:
        dh, dc = g_state[:, :hidden].copy(), g_state[:, hidden:].copy()
    for k, t in enumerate(reversed(walk)):
        rows_t = slice(t * batch, (t + 1) * batch)
        if full is not None:
            dh = dh + block(full[rows_t], H_STATE, hidden)
            dc = dc + block(full[rows_t], C_STATE, hidden)
        previous = walk[len(walk) - 2 - k] if k < len(walk) - 1 else None
        c_prev = (
            c0
            if previous is None
            else block(
                out_value[previous * batch : (previous + 1) * batch],
                C_STATE,
                hidden,
            )
        )
        gates, carry = cell_backward(
            out_value[rows_t],
            c_prev,
            mask[rows_t],
            dh,
            dc,
            np.zeros((batch, hidden)) if g_out is None else g_out[rows_t],
        )
        result[rows_t, : 4 * hidden], result[rows_t, 4 * hidden :] = (
            gates,
            carry,
        )
        dh, dc = carry[:, :hidden] + gates @ whh, carry[:, hidden:]
    result[rows:, :hidden], result[rows:, hidden : 2 * hidden] = dh, dc
    return result


def lstm_chain_grad(inputs, attrs):
    """Backward steps of one chunk of a chain (experiment 27): the per-step
    arithmetic of
    lstm_cell_grad, run for `steps` consecutive steps in one task. Returns rows
    in step order,
    each [gate gradients (4h) | carry (2h)], exactly as the per-step tasks would."""  # noqa: E501 -- preserve documentation layout
    hidden, batch, reverse, k, offset = (
        attrs["hidden"],
        attrs["batch"],
        attrs["reverse"],
        attrs["steps"],
        attrs["offset"],
    )
    values = list(inputs)
    out = values.pop(0)
    c0 = values.pop(0) if attrs.get("has_c0") else None
    mask = values.pop(0)
    g_out = values.pop(0) if attrs.get("has_out") else None
    carry = values.pop(0) if attrs.get("has_carry") else None
    gates_prev = values.pop(0) if attrs.get("has_gates") else None
    whh = values.pop(0)
    dh = np.zeros((batch, hidden)) if carry is None else carry[:, :hidden]
    if gates_prev is not None:
        dh = dh + gates_prev @ whh
    dc = (
        np.zeros((batch, hidden))
        if carry is None
        else carry[:, hidden : 2 * hidden]
    )
    result = np.zeros((k * batch, 6 * hidden))
    proc = list(
        reversed(list(order(k, reverse)))
    )  # local step indexes, in processing order
    for n, j in enumerate(proc):
        rows = slice(j * batch, (j + 1) * batch)
        state = out[(offset + j) * batch : (offset + j + 1) * batch]
        prev_j = (
            j + 1 if reverse else j - 1
        )  # the step before j in the forward walk
        if c0 is not None and n == k - 1:
            c_prev = c0
        else:
            c_prev = block(
                out[(offset + prev_j) * batch : (offset + prev_j + 1) * batch],
                C_STATE,
                hidden,
            )
        gates, carried = cell_backward(
            state,
            c_prev,
            mask[rows],
            dh,
            dc,
            np.zeros((batch, hidden)) if g_out is None else g_out[rows],
        )
        result[rows, : 4 * hidden], result[rows, 4 * hidden :] = gates, carried
        dh, dc = carried[:, :hidden] + gates @ whh, carried[:, hidden:]
    return result


def by_example(a, batch):
    """(steps*batch x n) time-major rows -> (batch, steps, n)."""
    return a.reshape(len(a) // batch, batch, -1).transpose(1, 0, 2)


def from_examples(a):
    return a.transpose(1, 0, 2).reshape(-1, a.shape[2])


def bqk(q, k, batch):
    return from_examples(
        by_example(q, batch) @ by_example(k, batch).transpose(0, 2, 1)
    )


def bpv(p, v, batch):
    return from_examples(by_example(p, batch) @ by_example(v, batch))


def bptq(p, q, batch):
    return from_examples(
        by_example(p, batch).transpose(0, 2, 1) @ by_example(q, batch)
    )


def place(shape, values, offsets):
    out = np.zeros(shape)
    for value, (r, c) in zip(values, offsets):
        out[r : r + value.shape[0], c : c + value.shape[1]] += value
    return out


def execute(op, inputs, attrs):
    """Sequence kernels by name; None if `op` isn't one of them."""
    if op == "view":
        r0, r1, c0, c1 = attrs["region"]
        return inputs[0][r0:r1, c0:c1]
    if op == "concat_rows":
        return np.concatenate(inputs, axis=0)
    if op in ("place", "place_block"):
        return place(attrs["shape"], inputs, attrs["offsets"])
    if op == "lstm":
        return lstm(*inputs, attrs["batch"], attrs["hidden"], attrs["reverse"])
    if op == "lstm_backward":
        return lstm_backward(inputs, attrs)
    if op == "lstm_chain":
        # A chunk of consecutive forward steps in one task: the whole-layer
        # kernel over its rows.
        xp, bias_ih, bias_hh, h0, c0, mask, whh_t = inputs
        return lstm(
            xp,
            whh_t,
            bias_ih,
            bias_hh,
            mask,
            h0,
            c0,
            attrs["batch"],
            attrs["hidden"],
            attrs["reverse"],
        )
    if op == "lstm_chain_grad":
        return lstm_chain_grad(inputs, attrs)
    if op in ("bqk", "bpv", "bptq"):
        return dict(bqk=bqk, bpv=bpv, bptq=bptq)[op](*inputs, attrs["batch"])
    if op == "lstm_cell":
        xp, bias_ih, bias_hh, h_prev, c_prev, mask, *partials = inputs
        # The recurrent product inline (the whole W_hh.T as the last input), or
        # as summed partials.
        hw = (
            h_prev @ partials[0]
            if attrs.get("inline")
            else recurrent(partials, attrs["blocks"], len(xp))
        )
        return cell(xp, hw, bias_ih, bias_hh, h_prev, c_prev, mask)
    if op in (
        "lstm_cell_backward",
        "lstm_cell_carry",
        "lstm_cell_grad",
        "lstm_state_grad",
    ):
        # Inputs: [state, c_prev, mask] (not for lstm_state_grad), then g_out
        # if has_out,
        # the incoming [dh | dc] if has_carry, then the recurrent partials
        # gates_next @ W_hh.
        hidden, values = attrs["hidden"], list(inputs)
        if op != "lstm_state_grad":
            state, c_prev, mask = values[:3]
            values = values[3:]
        rows = len(values[0]) if values else len(state)
        g_out = (
            values.pop(0) if attrs.get("has_out") else np.zeros((rows, hidden))
        )
        carry = (
            values.pop(0)
            if attrs.get("has_carry")
            else np.zeros((rows, 2 * hidden))
        )
        if attrs.get("inline"):
            gates_prev, whh = values
            dh = carry[:, :hidden] + gates_prev @ whh
        else:
            dh = carry[:, :hidden] + (
                recurrent(values, attrs["blocks"], rows) if values else 0.0
            )
        dc = carry[:, hidden : 2 * hidden]
        if op == "lstm_state_grad":
            return np.concatenate(
                [
                    dh,
                    dc,
                    np.zeros(
                        (rows, attrs.get("width", 4 * hidden) - 2 * hidden)
                    ),
                ],
                axis=1,
            )
        gates, carried = cell_backward(state, c_prev, mask, dh, dc, g_out)
        if (
            op == "lstm_cell_grad"
        ):  # one task per step: [gate gradients (4h) | carry (2h)]
            return np.concatenate([gates, carried], axis=1)
        return gates if op == "lstm_cell_backward" else carried
    return None
