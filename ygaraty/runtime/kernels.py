"""Allowlisted numerical operations; no executable code arrives over the
wire."""

import numpy as np

from ygaraty.runtime import convolution_kernels as convolution
from ygaraty.runtime import factorization_kernels as factorization
from ygaraty.runtime import sequence_kernels as sequence
from ygaraty.runtime import transformer_kernels as transformer

SEQUENCE_OPS = {
    "view",
    "concat_rows",
    "place",
    "place_block",
    "lstm",
    "lstm_backward",
    "bqk",
    "bpv",
    "bptq",
    "lstm_cell",
    "lstm_cell_backward",
    "lstm_cell_carry",
    "lstm_cell_grad",
    "lstm_state_grad",
    "lstm_chain",
    "lstm_chain_grad",
}


def execute(op, inputs, attrs):
    a = inputs[0] if inputs else None
    if op in convolution.OPS:
        result = convolution.execute(op, inputs, attrs)
    elif op in factorization.OPS:
        result = factorization.execute(op, inputs, attrs)
    elif op in transformer.OPS:
        result = transformer.execute(op, inputs, attrs)
    elif op in SEQUENCE_OPS:
        result = sequence.execute(op, inputs, attrs)
    elif op == "matmul":
        result = a @ inputs[1]
    elif op in ("add", "add_bias"):
        result = a + inputs[1]
    elif op == "mul":
        result = a * inputs[1]
    elif op == "embedding":
        result = a[attrs["ids"]]
    elif op == "embedding_backward":
        a = (
            np.concatenate(inputs, axis=0) if len(inputs) > 1 else a
        )  # row parts, in order
        result = np.zeros(attrs["shape"])
        ids = np.asarray(attrs["ids"])
        valid = ids != attrs.get("padding_idx")
        rows = (
            a[valid] * attrs["factor"]
            if attrs.get("factor", 1.0) != 1.0
            else a[valid]
        )
        np.add.at(result, ids[valid], rows)
    elif op == "embed":
        # A token embedding, scaled, plus its position embedding (tape.embed).
        result = (
            a[attrs["ids"]] * attrs["factor"] + inputs[1][attrs["positions"]]
        )
    elif op == "embed_rows":
        # One tile of `embed`: n token rows then n position rows, each (1,
        # columns).
        n = attrs["rows"]
        result = np.concatenate(inputs[:n], axis=0) * attrs[
            "factor"
        ] + np.concatenate(inputs[n:], axis=0)
    elif op == "gather_rows":
        result = np.concatenate(inputs, axis=0)
    elif op == "scatter_rows":
        a = (
            np.concatenate(inputs, axis=0) if len(inputs) > 1 else a
        )  # row parts, in order
        result = np.zeros(attrs["shape"])
        ids = np.asarray(attrs["ids"])
        selected = (ids >= 0) & (ids < result.shape[0])
        rows = (
            a[selected] * attrs["factor"]
            if attrs.get("factor", 1.0) != 1.0
            else a[selected]
        )
        np.add.at(result, ids[selected], rows)
    elif op == "inverse_norm":
        result = 1.0 / (np.sqrt(a) + attrs["epsilon"])
    elif op == "sum_squares":
        result = np.array([[np.sum(a * a)]])
    elif op == "clip_gradient":
        norm = np.sqrt(inputs[1][0, 0])
        result = a * (min(1.0, attrs["clip_norm"] / norm) if norm else 1.0)
    elif op == "sgd":
        result = a - attrs["learning_rate"] * inputs[1]
    elif op == "adam_m":
        result = attrs["beta1"] * a + (1 - attrs["beta1"]) * inputs[1]
    elif op == "adam_v":
        result = attrs["beta2"] * a + (1 - attrs["beta2"]) * inputs[1] ** 2
    elif op == "adam_step":
        # Adam's moments and update in one task: [m'; v'; w'], the arithmetic
        # of adam_m, adam_v
        # and adam_update. Uploaded as three row bands (upload_rows), each
        # assembled by region.
        b1, b2, step = attrs["beta1"], attrs["beta2"], attrs["step"]
        m = b1 * inputs[1] + (1 - b1) * inputs[3]
        v = b2 * inputs[2] + (1 - b2) * inputs[3] ** 2
        w = a - attrs["learning_rate"] * (m / (1 - b1**step)) / (
            np.sqrt(v / (1 - b2**step)) + attrs["epsilon"]
        )
        result = np.concatenate([m, v, w], axis=0)
    elif op == "adam_update":
        m = inputs[1] / (1 - attrs["beta1"] ** attrs["step"])
        v = inputs[2] / (1 - attrs["beta2"] ** attrs["step"])
        result = a - attrs["learning_rate"] * m / (
            np.sqrt(v) + attrs["epsilon"]
        )
    elif op == "sigmoid":
        result = np.exp(-np.logaddexp(0, -a))
    elif op == "tanh":
        result = np.tanh(a)
    elif op == "sigmoid_backward":
        result = inputs[1] * a * (1 - a)
    elif op == "tanh_backward":
        result = inputs[1] * (1 - a * a)
    elif op == "scale":
        result = a * attrs["factor"]
    elif op == "slice_columns":
        result = a[:, attrs["start"] : attrs["start"] + attrs["width"]]
    elif op == "slice_backward":
        result = np.zeros(attrs["shape"])
        result[:, attrs["start"] : attrs["start"] + attrs["width"]] = a
    elif op == "slice_block":
        result = np.zeros(attrs["shape"])
        if inputs:
            result[:, attrs["offset"] : attrs["offset"] + a.shape[1]] = a
    elif op == "concat_columns":
        result = np.concatenate(inputs, axis=1)
    elif op == "softmax":
        scores = np.where(
            attrs.get("mask", np.zeros_like(a, dtype=bool)), -np.inf, a
        )
        ex = np.exp(scores - scores.max(axis=1, keepdims=True))
        result = ex / ex.sum(axis=1, keepdims=True)
    elif op == "softmax_backward":
        result = a * (inputs[1] - (a * inputs[1]).sum(axis=1, keepdims=True))
    elif op in ("layer_norm", "layer_norm_backward"):
        centered = a - a.mean(axis=1, keepdims=True)
        inv = 1 / np.sqrt(
            np.mean(centered**2, axis=1, keepdims=True) + attrs["epsilon"]
        )
        normalized = centered * inv
        if op == "layer_norm":
            result = normalized * inputs[1] + inputs[2]
        elif attrs["index"] == 0:
            g = inputs[2] * inputs[1]
            result = inv * (
                g
                - g.mean(axis=1, keepdims=True)
                - normalized * (g * normalized).mean(axis=1, keepdims=True)
            )
        elif attrs["index"] == 1:
            result = (inputs[2] * normalized).sum(axis=0, keepdims=True)
        else:
            result = inputs[2].sum(axis=0, keepdims=True)
    elif op == "relu":
        result = np.maximum(a, 0)
    elif op == "relu_backward":
        result = inputs[1] * (a > 0)
    elif op == "sum":
        result = np.array([[a.sum()]])
    elif op == "transpose":
        result = a.T
    elif op == "reduce_rows":
        result = a.sum(axis=0, keepdims=True)
    elif op == "broadcast":
        result = np.broadcast_to(a, attrs["shape"])
    elif op in ("cross_entropy", "ce_backward"):
        shifted = a - a.max(axis=1, keepdims=True)
        ex = np.exp(shifted)
        total = ex.sum(axis=1, keepdims=True)
        target = np.asarray(attrs["targets"], dtype=np.int64)
        valid = target != attrs.get("ignore_index")
        # A row chunk of a longer cross-entropy divides by the whole batch's
        # count.
        count = attrs.get("count", np.count_nonzero(valid))
        smoothing = attrs.get("label_smoothing", 0.0)
        if op == "cross_entropy":
            nll = np.log(total[:, 0]) - shifted[np.arange(len(a)), target]
            smooth = np.log(total[:, 0]) - shifted.mean(axis=1)
            result = np.array(
                [
                    [
                        (
                            (1 - smoothing) * nll[valid]
                            + smoothing * smooth[valid]
                        ).sum()
                        / count
                    ]
                ]
            )
        else:
            result = ex / total
            result -= smoothing / a.shape[1]
            result[np.arange(len(a)), target] -= 1 - smoothing
            result *= valid[:, None] * inputs[1][0, 0] / count
    elif op == "ce_fused":
        # The output layer for a band of rows: logits = x @ W + b, then the
        # loss and its gradient,
        # with the arithmetic of matmul, add_bias, cross_entropy and
        # ce_backward. Returns
        # [logit gradient | row loss]; the band's rows of the loss sum to its
        # share of the mean.
        logits = (a @ inputs[1]) + inputs[2]
        shifted = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(shifted)
        total = ex.sum(axis=1, keepdims=True)
        target = np.asarray(attrs["targets"], dtype=np.int64)
        valid = target != attrs.get("ignore_index")
        count, smoothing = attrs["count"], attrs.get("label_smoothing", 0.0)
        nll = np.log(total[:, 0]) - shifted[np.arange(len(logits)), target]
        smooth = np.log(total[:, 0]) - shifted.mean(axis=1)
        rows = (
            np.where(valid, (1 - smoothing) * nll + smoothing * smooth, 0.0)
            / count
        )
        grad = ex / total
        grad -= smoothing / logits.shape[1]
        grad[np.arange(len(logits)), target] -= 1 - smoothing
        grad *= valid[:, None] * inputs[3][0, 0] / count
        result = np.concatenate([grad, rows[:, None]], axis=1)
    elif op == "add_many":
        result = np.zeros_like(a)
        # Fixed input order, regardless of worker completion order.
        for value in inputs:
            result += value
    elif op == "assemble":
        result = np.zeros(attrs["shape"], dtype=np.float64)
        for value, (i, end_i, j, end_j) in zip(inputs, attrs["blocks"]):
            result[i:end_i, j:end_j] = value
    else:
        raise ValueError(f"Unsupported operation: {op}")
    result = np.asarray(result, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Kernel produced invalid/nonfinite output")
    return result


def evaluate(tape, values=None):
    """Local untiled reference execution, without networking."""
    store = {
        k: v.copy()
        for k, v in (tape.values if values is None else values).items()
    }
    for node in tape.nodes:
        store[node.id] = execute(
            node.op, [store[k] for k in node.inputs], node.attrs
        )
    return store


def execute_task(op, inputs, attrs):
    """Return all original results, including grouped intermediates for
    auditing."""
    if op != "group":
        return [execute(op, inputs, attrs)]
    store, outputs = {}, []
    for node in attrs["nodes"]:
        args = []
        for ref in node["inputs"]:
            if "external" in ref:
                a = inputs[ref["external"]]
            else:
                a = store[ref["value"]]
                if ref["region"]:
                    i, ei, j, ej = ref["region"]
                    a = a[i:ei, j:ej]
            args.append(a)
        value = execute(node["op"], args, node["attrs"])
        if list(value.shape) != node["shape"]:
            raise ValueError("Invalid grouped output shape")
        store[node["id"]] = value
        outputs.append(value)
    return outputs
