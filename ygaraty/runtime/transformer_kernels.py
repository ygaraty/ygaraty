"""Fused Transformer layers: one task per band of examples (experiment 30).

Rows are example-major: row b*steps + t is position t of example b. A
`tf_layer` task runs one
pre-norm layer (self-attention, cross-attention for the decoder, the
feed-forward block, and
their residuals) over a band of examples. `tf_layer_grad` recomputes that
forward from the
layer's input and returns, packed into one matrix of width d: the input's
gradient, the
memory's gradient (decoder), then every parameter's gradient (packed_layout).
The attention
probabilities and the feed-forward activations never leave the worker.
"""

import os

import numpy as np

from ygaraty.compiler.layouts import (
    DECODER_PARAMS,
    ENCODER_PARAMS,
    packed_layout,
)

OPS = ("tf_layer", "tf_layer_grad", "vocab_loss")


def pack(grads, layout, d):
    out = []
    for g, (offset, rows, mode) in zip(grads, layout):
        if mode == "rows":
            out.append(g)
        elif mode == "transposed":
            out.append(g.T)
        else:
            flat = np.zeros(rows * d)
            flat[: g.size] = g.ravel()
            out.append(flat.reshape(rows, d))
    return np.concatenate(out, axis=0) if out else np.zeros((0, d))


def masks(key_mask, tq, tk, causal):
    """(examples, tq, tk) boolean: True where a query may not attend to a
    key."""
    m = np.broadcast_to(
        np.asarray(key_mask, dtype=bool)[:, None, :], (len(key_mask), tq, tk)
    )
    if causal:
        m = m | np.triu(np.ones((tq, tk), dtype=bool), 1)[None]
    return m


def layer_norm(x, w, b, eps):
    xc = x - x.mean(axis=1, keepdims=True)
    inv = 1 / np.sqrt((xc * xc).mean(axis=1, keepdims=True) + eps)
    n = xc * inv
    return n * w + b, (inv, n)


def layer_norm_backward(dy, w, cache):
    inv, n = cache
    g = dy * w
    dx = inv * (
        g
        - g.mean(axis=1, keepdims=True)
        - n * (g * n).mean(axis=1, keepdims=True)
    )
    return (
        dx,
        (dy * n).sum(axis=0, keepdims=True),
        dy.sum(axis=0, keepdims=True),
    )


def attention(q_in, kv_in, in_w, in_b, out_w, out_b, nb, tq, tk, heads, mask):
    """Multi-head attention as nn.MultiheadAttention computes it (q, k, v rows
    of in_proj)."""
    d = q_in.shape[1]
    dh = d // heads
    q = q_in @ in_w[:d].T + in_b[:, :d]
    k = kv_in @ in_w[d : 2 * d].T + in_b[:, d : 2 * d]
    v = kv_in @ in_w[2 * d :].T + in_b[:, 2 * d :]
    Q = q.reshape(nb, tq, heads, dh).transpose(0, 2, 1, 3)
    K = k.reshape(nb, tk, heads, dh).transpose(0, 2, 1, 3)
    V = v.reshape(nb, tk, heads, dh).transpose(0, 2, 1, 3)
    s = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    s = np.where(mask[:, None], -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=-1, keepdims=True)
    o = (p @ V).transpose(0, 2, 1, 3).reshape(nb * tq, d)
    return o @ out_w.T + out_b, (q_in, kv_in, Q, K, V, p, o)


def weight_products(pairs):
    """A deferred weight gradient (DO-9): the products a.T @ b of its pairs,
    stacked by rows, with the arithmetic the combined backward pass uses."""
    if len(pairs) == 1:
        return pairs[0][0].T @ pairs[0][1]
    return np.concatenate([a.T @ b for a, b in pairs], axis=0)


def resolve_grads(grads):
    """Gradients with deferred weight products (tuples of pairs) computed."""
    return [weight_products(g) if isinstance(g, tuple) else g for g in grads]


def attention_backward(dy, in_w, out_w, cache, nb, tq, tk, heads, defer=False):
    q_in, kv_in, Q, K, V, p, o = cache
    d = q_in.shape[1]
    dh = d // heads
    d_out_w, d_out_b = ((dy, o),), dy.sum(axis=0, keepdims=True)
    dO = (dy @ out_w).reshape(nb, tq, heads, dh).transpose(0, 2, 1, 3)
    dp = dO @ V.transpose(0, 1, 3, 2)
    dV = p.transpose(0, 1, 3, 2) @ dO
    ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True)) / np.sqrt(dh)
    dQ, dK = ds @ K, ds.transpose(0, 1, 3, 2) @ Q
    dq = dQ.transpose(0, 2, 1, 3).reshape(nb * tq, d)
    dk = dK.transpose(0, 2, 1, 3).reshape(nb * tk, d)
    dv = dV.transpose(0, 2, 1, 3).reshape(nb * tk, d)
    d_in_w = ((dq, q_in), (dk, kv_in), (dv, kv_in))
    d_in_b = np.concatenate(
        [
            dq.sum(axis=0, keepdims=True),
            dk.sum(axis=0, keepdims=True),
            dv.sum(axis=0, keepdims=True),
        ],
        axis=1,
    )
    dq_in = dq @ in_w[:d]
    dkv_in = dk @ in_w[d : 2 * d] + dv @ in_w[2 * d :]
    grads = [d_in_w, d_in_b, d_out_w, d_out_b]
    return dq_in, dkv_in, grads if defer else resolve_grads(grads)


def layer_inputs(inputs, attrs):
    """A layer task's input, memory and remaining inputs. Experiment 78's parallel
    layers: with attrs["parts"] = k, the k values after the input (and the
    memory) are added to it, so the layer after a parallel group reads
    x + r_1 + ... directly from the group's tasks."""
    decoder = attrs.get("decoder", False)
    start = 1 + decoder
    x = inputs[0]
    for part in inputs[start : start + attrs.get("parts", 0)]:
        x = x + part
    mem = inputs[1] if decoder else None
    return x, mem, inputs[start + attrs.get("parts", 0) :]


def layer_forward(x, mem, p, attrs):
    """With attrs["residual"] (experiment 78's parallel layers after the first of a
    group), the layer returns only its residual update, y - x."""
    nb, T, heads, eps = (
        attrs["batch"],
        attrs["steps"],
        attrs["heads"],
        attrs.get("epsilon", 1e-5),
    )
    decoder, final = (
        attrs.get("decoder", False),
        attrs.get("final_norm", False),
    )
    if (
        len(p)
        != (len(DECODER_PARAMS) if decoder else len(ENCODER_PARAMS))
        + 2 * final
    ):
        raise ValueError("Wrong number of layer parameters")
    if final:
        # The stack's final norm, applied by its last layer (its owner holds
        # the weights).
        p, (fw, fb) = p[:-2], p[-2:]
    n1w, n1b, sa_in_w, sa_in_b, sa_out_w, sa_out_b, n2w, n2b = p[:8]
    n1, c_n1 = layer_norm(x, n1w, n1b, eps)
    a, c_sa = attention(
        n1,
        n1,
        sa_in_w,
        sa_in_b,
        sa_out_w,
        sa_out_b,
        nb,
        T,
        T,
        heads,
        masks(attrs["key_mask"], T, T, attrs.get("causal", False)),
    )
    x1 = x + a
    cache = dict(c_n1=c_n1, c_sa=c_sa, x1=x1)
    if decoder:
        ca_in_w, ca_in_b, ca_out_w, ca_out_b, nfw, nfb = p[8:14]
        l1w, l1b, l2w, l2b = p[14:18]
        S = attrs["mem_steps"]
        n2, c_n2 = layer_norm(x1, n2w, n2b, eps)
        c, c_ca = attention(
            n2,
            mem,
            ca_in_w,
            ca_in_b,
            ca_out_w,
            ca_out_b,
            nb,
            T,
            S,
            heads,
            masks(attrs["mem_mask"], T, S, False),
        )
        x2 = x1 + c
        cache.update(c_n2=c_n2, c_ca=c_ca, x2=x2)
    else:
        nfw, nfb = n2w, n2b
        l1w, l1b, l2w, l2b = p[8:12]
        x2 = x1
    nf, c_nf = layer_norm(x2, nfw, nfb, eps)
    h_pre = nf @ l1w.T + l1b
    h = np.maximum(h_pre, 0.0)
    y = x2 + h @ l2w.T + l2b
    cache.update(c_nf=c_nf, nf=nf, h_pre=h_pre, h=h)
    if final:
        y, cache["c_final"] = layer_norm(y, fw, fb, eps)
    elif attrs.get("residual"):
        y = y - x
    return y, cache


def layer_backward(x, mem, dy, p, attrs, defer=False):
    """Gradients of a layer's input, memory and parameters, recomputing the
    forward pass. With `defer` (DO-9's split backward), the weight matrices'
    gradients come back as their pairs of factors, for resolve_grads to
    multiply after dx and dmem have gone on: the same products, so the same
    numbers."""
    nb, T, heads, _eps = (
        attrs["batch"],
        attrs["steps"],
        attrs["heads"],
        attrs.get("epsilon", 1e-5),
    )
    decoder, final = (
        attrs.get("decoder", False),
        attrs.get("final_norm", False),
    )
    _, c = layer_forward(x, mem, p, attrs)
    if final:
        p, (fw, _fb) = p[:-2], p[-2:]
        dy, d_fw, d_fb = layer_norm_backward(dy, fw, c["c_final"])
    n1w, n1b, sa_in_w, sa_in_b, sa_out_w, sa_out_b, n2w, n2b = p[:8]
    if decoder:
        ca_in_w, ca_in_b, ca_out_w, ca_out_b, nfw, nfb = p[8:14]
        l1w, l1b, l2w, l2b = p[14:18]
    else:
        nfw, _nfb = n2w, n2b
        l1w, l1b, l2w, l2b = p[8:12]
    # Feed-forward block and its norm.
    d_l2w, d_l2b = ((dy, c["h"]),), dy.sum(axis=0, keepdims=True)
    dh_pre = (dy @ l2w) * (c["h_pre"] > 0)
    d_l1w, d_l1b = ((dh_pre, c["nf"]),), dh_pre.sum(axis=0, keepdims=True)
    dxf, d_nfw, d_nfb = layer_norm_backward(dh_pre @ l1w, nfw, c["c_nf"])
    dx2 = dy + dxf
    if decoder:
        dn2, dmem, ca_grads = attention_backward(
            dx2,
            ca_in_w,
            ca_out_w,
            c["c_ca"],
            nb,
            T,
            attrs["mem_steps"],
            heads,
            defer=True,
        )
        dx1n, d_n2w, d_n2b = layer_norm_backward(dn2, n2w, c["c_n2"])
        dx1 = dx2 + dx1n
    else:
        dmem, dx1 = None, dx2
    dq_in, dkv_in, sa_grads = attention_backward(
        dx1, sa_in_w, sa_out_w, c["c_sa"], nb, T, T, heads, defer=True
    )
    dxn, d_n1w, d_n1b = layer_norm_backward(dq_in + dkv_in, n1w, c["c_n1"])
    dx = dx1 + dxn
    if decoder:
        grads = (
            [d_n1w, d_n1b]
            + sa_grads
            + [d_n2w, d_n2b]
            + ca_grads
            + [d_nfw, d_nfb, d_l1w, d_l1b, d_l2w, d_l2b]
        )
    else:
        grads = (
            [d_n1w, d_n1b]
            + sa_grads
            + [d_nfw, d_nfb, d_l1w, d_l1b, d_l2w, d_l2b]
        )
    if final:
        grads += [d_fw, d_fb]
    if attrs.get("residual") and not final:
        dx = dx - dy  # the residual update's gradient: the layer's, less dy
    return dx, dmem, grads if defer else resolve_grads(grads)


def execute(op, inputs, attrs):
    decoder = attrs.get("decoder", False)
    if op == "vocab_loss":
        return np.array([[vocab_row_loss(inputs, attrs).sum()]])
    if op == "tf_layer":
        x, mem, params = layer_inputs(inputs, attrs)
        return layer_forward(x, mem, params, attrs)[0]
    if op == "tf_layer_grad":
        x, mem, rest = layer_inputs(inputs, attrs)
        g, params = rest[0], rest[1:]
        dx, dmem, grads = layer_backward(x, mem, g, params, attrs)
        d = x.shape[1]
        layout, _ = packed_layout([q.shape for q in params], d)
        parts = [dx] + ([dmem] if decoder else []) + [pack(grads, layout, d)]
        return np.concatenate(parts, axis=0)
    return None


# Owned layers (experiment 31)
# ----------------------------------------------------------------
# A worker that owns a layer keeps its parameters and Adam moments between
# updates. Its band
# tasks take no parameters and its backward tasks keep the parameter gradients;
# one step task
# per layer sums them over the bands, in band order, and applies Adam with the
# arithmetic of the
# unowned plan (add_many, then adam_step), so both plans give the same numbers.
# Owned vocabulary (experiment 32): the tied embedding and output layer split
# by vocabulary rows.
# Owner s holds rows [v0, v1) of the table and of the output bias, with their
# Adam moments. It
# serves the embedding rows a batch uses (tf_vocab_gather); per band of decoder
# rows it returns
# logit statistics (tf_vocab_stats), then, from every shard's statistics, its
# part of the
# gradient of the decoder output (tf_vocab_grad). The logits, their gradient
# and the weight
# gradient never leave it; its step adds the embedding gradient rows and
# applies Adam.
VOCAB_OPS = (
    "tf_vocab_load",
    "tf_vocab_gather",
    "tf_vocab_stats",
    "tf_vocab_grad",
    "tf_vocab_embed_grad",
    "tf_vocab_step",
    "tf_vocab_publish",
)
OWNED_OPS = (
    "tf_owned_load",
    "tf_owned_layer",
    "tf_owned_grad",
    "tf_owned_step",
    "tf_owned_publish",
) + VOCAB_OPS


def vocab_stats(z, targets, v0):
    """Per row of a shard's logits: [max, sum of exp(z - max), the target's
    logit (0 if another
    shard holds the target), sum of the logits]."""
    top = z.max(axis=1)
    t = np.asarray(targets, dtype=np.int64) - v0
    inside = (t >= 0) & (t < z.shape[1])
    target = np.where(
        inside, z[np.arange(len(z)), np.clip(t, 0, z.shape[1] - 1)], 0.0
    )
    return np.stack(
        [top, np.exp(z - top[:, None]).sum(axis=1), target, z.sum(axis=1)],
        axis=1,
    )


def combine_stats(stats):
    """Every shard's statistics for a band, in shard order: per row the
    log-sum-exp over the whole
    vocabulary, the target's logit and the sum of the logits."""
    top = np.max([s[:, 0] for s in stats], axis=0)
    total = np.zeros_like(top)
    for s in stats:
        total += s[:, 1] * np.exp(s[:, 0] - top)
    target, logits = np.zeros_like(top), np.zeros_like(top)
    for s in stats:
        target += s[:, 2]
        logits += s[:, 3]
    return top + np.log(total), target, logits


def vocab_row_loss(stats, attrs):
    """A band's rows of the label-smoothed mean loss, as cross_entropy computes
    it."""
    lse, target, logits = combine_stats(stats)
    valid = np.asarray(attrs["targets"]) != attrs.get("ignore_index")
    smoothing = attrs.get("label_smoothing", 0.0)
    rows = (1 - smoothing) * (lse - target) + smoothing * (
        lse - logits / attrs["vocab"]
    )
    return np.where(valid, rows, 0.0) / attrs["count"]


NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)  # ygaraty/reference/muon.py


def orthogonalize(matrix, steps=5):
    """Muon's quintic Newton-Schulz iteration (reference/muon.py), in NumPy:
    the nearest semi-orthogonal matrix, approximately, on the smaller side."""
    a, b, c = NEWTON_SCHULZ
    x = matrix / (np.linalg.norm(matrix) + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        square = x @ x.T
        x = a * x + (b * square + c * square @ square) @ x
    return x.T if tall else x


def unpack(packed, layout, shapes):
    out = []
    for (offset, rows, mode), (a, b) in zip(layout, shapes):
        part = packed[offset : offset + rows]
        if mode == "rows":
            out.append(part.copy())
        elif mode == "transposed":
            out.append(part.T.copy())
        else:
            out.append(part.ravel()[:b].reshape(1, b).copy())
    return out


class OwnedLayers:
    """The layers this worker owns: parameters, moments and the current
    update's band gradients."""

    def __init__(self, split_backward=None):
        self.layers = {}
        # DO-9: tf_owned_grad returns dx and dmem and leaves the weight
        # products for flush(), which the worker calls once that result is on
        # its way, and the step calls in any case.
        self.split_backward = (
            os.environ.get("YGARATY_SPLIT_BACKWARD", "0") == "1"
            if split_backward is None
            else split_backward
        )

    def pending(self):
        return any(s.get("pending") for s in self.layers.values())

    def flush(self):
        """Compute the weight gradients tf_owned_grad deferred. Returns the
        number of bands resolved."""
        done = 0
        for layer in self.layers.values():
            for band, (grads, rows, d) in sorted(
                layer.get("pending", {}).items()
            ):
                layer["grads"][band] = pack(resolve_grads(grads), rows, d)
                done += 1
            layer["pending"] = {}
        return done

    def nbytes(self):
        return sum(
            a.nbytes
            for s in self.layers.values()
            for k in ("w", "m", "v")
            for a in s[k]
        ) + sum(
            g.nbytes for s in self.layers.values() for g in s["grads"].values()
        )

    def state(self, attrs):
        layer = self.layers.get(attrs["layer"])
        if layer is None:
            raise ValueError(
                f"Not holding owned layer {attrs['layer']}: its load "
                f"task ran elsewhere"
            )
        return layer

    def execute(self, op, inputs, attrs):
        if op in VOCAB_OPS:
            return self.vocab(op, inputs, attrs)
        decoder = attrs.get("decoder", False)
        if op == "tf_owned_load":
            n = len(inputs) // 3
            self.layers[attrs["layer"]] = dict(
                w=[np.array(a, dtype=np.float64) for a in inputs[:n]],
                m=[np.array(a, dtype=np.float64) for a in inputs[n : 2 * n]],
                v=[np.array(a, dtype=np.float64) for a in inputs[2 * n :]],
                grads={},
            )
            return np.zeros((1, 1))
        layer = self.state(attrs)
        if op in ("tf_owned_layer", "tf_owned_grad"):
            x, mem, rest = layer_inputs(inputs, attrs)
        if op == "tf_owned_layer":
            return layer_forward(x, mem, layer["w"], attrs)[0]
        if op == "tf_owned_grad":
            g = rest[0]
            dx, dmem, grads = layer_backward(
                x, mem, g, layer["w"], attrs, defer=self.split_backward
            )
            d = x.shape[1]
            packed_rows, _ = packed_layout([w.shape for w in layer["w"]], d)
            # A retry overwrites its band.
            if self.split_backward:
                layer["grads"].pop(attrs["band"], None)
                layer.setdefault("pending", {})[attrs["band"]] = (
                    grads,
                    packed_rows,
                    d,
                )
            else:
                layer["grads"][attrs["band"]] = pack(grads, packed_rows, d)
            return np.concatenate(
                [dx] + ([dmem] if decoder else []) + [np.zeros((1, d))], axis=0
            )
        if op == "tf_owned_step":
            self.flush()
            bands = attrs["bands"]
            if sorted(layer["grads"]) != list(range(bands)):
                raise ValueError(
                    f"Layer {attrs['layer']} has gradients for bands "
                    f"{sorted(layer['grads'])}, expected {bands}"
                )
            first = layer["grads"][0]
            total = np.zeros_like(first)
            for k in range(bands):  # add_many's order: zeros, then each band
                total += layer["grads"][k]
            shapes = [w.shape for w in layer["w"]]
            grads = unpack(
                total, packed_layout(shapes, first.shape[1])[0], shapes
            )
            b1, b2, step, lr, eps = (
                attrs["beta1"],
                attrs["beta2"],
                attrs["step"],
                attrs["learning_rate"],
                attrs["epsilon"],
            )
            muon = attrs.get("optimizer") == "muon"
            for i, g in enumerate(grads):  # adam_step's arithmetic
                if muon and min(g.shape) > 1:
                    # M-2: Muon on the layer's weight matrices, as
                    # ygaraty/reference/muon.py; its buffer lives in m.
                    momentum = attrs.get("momentum", 0.95)
                    buffer = momentum * layer["m"][i] + g
                    scale = max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                    layer["w"][i] = layer["w"][i] - attrs[
                        "muon_learning_rate"
                    ] * scale * orthogonalize(
                        g + momentum * buffer, attrs.get("ns_steps", 5)
                    )
                    layer["m"][i] = buffer
                    continue
                m = b1 * layer["m"][i] + (1 - b1) * g
                v = b2 * layer["v"][i] + (1 - b2) * g**2
                layer["w"][i] = layer["w"][i] - lr * (m / (1 - b1**step)) / (
                    np.sqrt(v / (1 - b2**step)) + eps
                )
                layer["m"][i], layer["v"][i] = m, v
            layer["grads"] = {}
            return np.zeros((1, 1))
        if op == "tf_owned_publish":
            return layer[attrs["kind"]][attrs["index"]].copy()
        return None

    def vocab(self, op, inputs, attrs):
        """A vocabulary shard's operations; its state is w = [table rows, bias
        columns] and their moments."""
        if op == "tf_vocab_load":
            copies = [np.array(a, dtype=np.float64) for a in inputs]
            self.layers[attrs["layer"]] = dict(
                w=copies[0:2],
                m=copies[2:4],
                v=copies[4:6],
                grads={},
                embed=None,
            )
            return np.zeros((1, 1))
        shard = self.state(attrs)
        table, bias = shard["w"]
        if op == "tf_vocab_gather":
            return table[attrs["ids"]]
        if op == "tf_vocab_stats":
            return vocab_stats(
                inputs[0] @ table.T + bias, attrs["targets"], attrs["rows"][0]
            )
        if op == "tf_vocab_grad":
            h, stats, seed = (
                inputs[0],
                inputs[1 : 1 + attrs["shards"]],
                inputs[1 + attrs["shards"]],
            )
            lse, _, _ = combine_stats(stats)
            target = np.asarray(attrs["targets"], dtype=np.int64)
            valid = target != attrs.get("ignore_index")
            local = target - attrs["rows"][0]
            inside = np.flatnonzero((local >= 0) & (local < len(table)))
            # ce_backward's arithmetic on this shard's columns of the logits.
            grad = np.exp(h @ table.T + bias - lse[:, None])
            grad -= attrs.get("label_smoothing", 0.0) / attrs["vocab"]
            grad[inside, local[inside]] -= 1 - attrs.get(
                "label_smoothing", 0.0
            )
            grad *= valid[:, None] * seed[0, 0] / attrs["count"]
            shard["grads"][attrs["band"]] = (
                grad.T @ h,
                grad.sum(axis=0, keepdims=True),
            )  # a retry overwrites its band
            return np.concatenate(
                [grad @ table, np.zeros((1, h.shape[1]))], axis=0
            )
        if op == "tf_vocab_embed_grad":
            if "scatter" in attrs:
                # The gathered rows' gradient from the embeddings' gradient
                # bands: each side's rows
                # scattered in order (embedding_backward's arithmetic), then
                # the sides summed.
                total, start = None, 0
                for count, factor in zip(attrs["sides"], attrs["factors"]):
                    side = np.zeros((len(attrs["ids"]), inputs[0].shape[1]))
                    for k in range(start, start + count):
                        local = np.asarray(attrs["scatter"][k], dtype=np.int64)
                        keep = local >= 0
                        np.add.at(
                            side,
                            local[keep],
                            inputs[k][keep] * factor
                            if factor != 1.0
                            else inputs[k][keep],
                        )
                    total = side if total is None else total + side
                    start += count
                shard["embed"] = (
                    np.asarray(attrs["ids"], dtype=np.int64),
                    total,
                )
            else:
                shard["embed"] = (
                    np.asarray(attrs["ids"], dtype=np.int64),
                    np.array(inputs[0]),
                )
            return np.zeros((1, 1))
        if op == "tf_vocab_step":
            bands = attrs["bands"]
            if sorted(shard["grads"]) != list(range(bands)):
                raise ValueError(
                    f"Shard {attrs['layer']} has gradients for bands "
                    f"{sorted(shard['grads'])}, expected {bands}"
                )
            if attrs.get("embed") and shard["embed"] is None:
                raise ValueError(
                    f"Shard {attrs['layer']} has no embedding gradient"
                )
            dtable, dbias = np.zeros_like(table), np.zeros_like(bias)
            for k in range(bands):
                dtable += shard["grads"][k][0]
                dbias += shard["grads"][k][1]
            if attrs.get("embed"):
                ids, rows = shard["embed"]
                np.add.at(dtable, ids, rows)
            b1, b2, step, lr, eps = (
                attrs["beta1"],
                attrs["beta2"],
                attrs["step"],
                attrs["learning_rate"],
                attrs["epsilon"],
            )
            for i, g in enumerate((dtable, dbias)):  # adam_step's arithmetic
                m = b1 * shard["m"][i] + (1 - b1) * g
                v = b2 * shard["v"][i] + (1 - b2) * g**2
                shard["w"][i] = shard["w"][i] - lr * (m / (1 - b1**step)) / (
                    np.sqrt(v / (1 - b2**step)) + eps
                )
                shard["m"][i], shard["v"][i] = m, v
            shard["grads"], shard["embed"] = {}, None
            return np.zeros((1, 1))
        if op == "tf_vocab_publish":
            return shard[attrs["kind"]][attrs["index"]].copy()
        raise ValueError(f"Unsupported vocabulary operation: {op}")
