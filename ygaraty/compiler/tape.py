"""Small rank-two tensor tape, with an explicit reverse-mode graph builder."""

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str]
    shape: tuple[int, int]
    attrs: dict = field(default_factory=dict)


class Tape:
    def __init__(self):
        self.values = {}
        self.shapes = {}
        self.nodes = []
        self.parameters = []
        self.gradients = {}
        self.loss = None
        self._counter = 0
        # Owned Transformer layers: tag -> dict(params, decoder, forward ids,
        # backward ids, ...).
        # Their parameters get no gradient on the tape; the owning worker
        # updates them.
        self.owned_layers = {}
        # Owned vocabulary (experiment 32): dict(params=[table, bias],
        # shards=[[v0, v1], ...],
        # grads={shard: [vocab_grad ids]}, embeds={shard: vocab_embed_grad id},
        # table=the
        # gathered rows' concatenation that `embed` reads), or None.
        self.owned_vocab = None
        # Values whose gradients are parameter gradients (parameters, and
        # vocabulary gathers
        # that stand for an owner's rows), and the nodes that compute those
        # gradients: the
        # planner keeps them at full width on a narrow wire
        # (plan.narrow_activations).
        self.parameter_like, self.parameter_side = set(), set()

    def forward(self, *, outputs):
        """Compile named outputs without a loss or differentiation.

        Returns an immutable inference plan. Compilation validates graph
        metadata only; call the plan's run(inputs=...) to evaluate it.
        """
        from ..inference.forward import ForwardPlan

        return ForwardPlan.from_tape(self, outputs=outputs)

    def _name(self, name=None):
        if name is None:
            while f"v{self._counter}" in self.shapes:
                self._counter += 1
            name = f"v{self._counter}"
            self._counter += 1
        if (
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or name in self.shapes
        ):
            raise ValueError(f"Invalid or duplicate tensor ID: {name!r}")
        return name

    def input(self, name, data, *, parameter=False):
        data = np.asarray(data, dtype=np.float64)
        if (
            data.ndim != 2
            or min(data.shape) < 1
            or not np.isfinite(data).all()
        ):
            raise ValueError("Inputs must be nonempty, finite rank-two arrays")
        name = self._name(name)
        self.values[name] = data.copy()
        self.shapes[name] = data.shape
        if parameter:
            self.parameters.append(name)
        return name

    def parameter(self, name, data):
        name = self.input(name, data, parameter=True)
        self.parameter_like.add(name)
        return name

    def _op(self, op, inputs, shape, attrs=None, name=None):
        name = self._name(name)
        self.shapes[name] = shape
        self.nodes.append(Node(name, op, list(inputs), shape, attrs or {}))
        return name

    def matmul(self, a, b, name=None):
        m, k = self.shapes[a]
        kb, n = self.shapes[b]
        if k != kb:
            raise ValueError("matmul inner dimensions differ")
        return self._op("matmul", [a, b], (m, n), name=name)

    def _elementwise(self, op, a, b, name):
        if self.shapes[a] != self.shapes[b]:
            raise ValueError(f"{op} requires equal shapes")
        return self._op(op, [a, b], self.shapes[a], name=name)

    def add(self, a, b, name=None):
        return self._elementwise("add", a, b, name)

    def mul(self, a, b, name=None):
        return self._elementwise("mul", a, b, name)

    def add_bias(self, a, b, name=None):
        if self.shapes[b] != (1, self.shapes[a][1]):
            raise ValueError("bias must be 1 x columns")
        return self._op("add_bias", [a, b], self.shapes[a], name=name)

    def relu(self, a, name=None):
        return self._op("relu", [a], self.shapes[a], name=name)

    def conv2d(self, x, weight, bias, *, channels, height, width, name=None):
        """Stride-one, same-padded 3x3 convolution on flattened NCHW rows."""
        batch, size = self.shapes[x]
        out, kernel_size = self.shapes[weight]
        if (
            any(type(n) is not int or n < 1 for n in (channels, height, width))
            or size != channels * height * width
            or kernel_size != channels * 9
            or self.shapes[bias] != (1, out)
        ):
            raise ValueError("Invalid flattened convolution shapes")
        return self._op(
            "conv2d",
            [x, weight, bias],
            (batch, out * height * width),
            dict(
                batch=batch,
                in_channels=channels,
                height=height,
                width=width,
                out_channels=out,
                kernel=3,
            ),
            name,
        )

    def pool2d(self, x, *, channels, height, width, average=False, name=None):
        """2x2 max pooling, or global average pooling, on NCHW rows."""
        batch, size = self.shapes[x]
        if (
            any(type(n) is not int or n < 1 for n in (channels, height, width))
            or size != channels * height * width
            or (not average and (height % 2 or width % 2))
        ):
            raise ValueError("Invalid flattened pooling shapes")
        shape = (
            batch,
            channels if average else channels * (height // 2) * (width // 2),
        )
        return self._op(
            "avgpool2d" if average else "maxpool2d",
            [x],
            shape,
            dict(
                batch=batch, in_channels=channels, height=height, width=width
            ),
            name,
        )

    def embedding(self, table, ids, name=None, padding_idx=None):
        ids = np.asarray(ids)
        if (
            ids.ndim != 1
            or len(ids) < 1
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= self.shapes[table][0])
        ):
            raise ValueError(
                "Embedding IDs must be a nonempty vector of valid integer rows"
            )
        return self._op(
            "embedding",
            [table],
            (len(ids), self.shapes[table][1]),
            {"ids": ids.tolist(), "padding_idx": padding_idx},
            name,
        )

    def embed(
        self,
        table,
        ids,
        positions,
        position_ids,
        factor,
        padding_idx=None,
        name=None,
    ):
        """table[ids] * factor + positions[position_ids]: a token
        embedding, its scaling and its position embedding as one
        operation (one task per tile; its backward scales inside the
        scatter). The same arithmetic as scale(embedding(...)) +
        embedding(...).
        """
        ids, position_ids = np.asarray(ids), np.asarray(position_ids)
        for rows, table_ in ((ids, table), (position_ids, positions)):
            if (
                rows.ndim != 1
                or len(rows) < 1
                or rows.dtype.kind not in "iu"
                or np.any(rows < 0)
                or np.any(rows >= self.shapes[table_][0])
            ):
                raise ValueError(
                    "Embedding IDs must be a nonempty vector "
                    "of valid integer rows"
                )
        if (
            len(ids) != len(position_ids)
            or self.shapes[table][1] != self.shapes[positions][1]
        ):
            raise ValueError(
                "Token and position embeddings must agree on rows and width"
            )
        return self._op(
            "embed",
            [table, positions],
            (len(ids), self.shapes[table][1]),
            {
                "ids": ids.tolist(),
                "positions": position_ids.tolist(),
                "factor": float(factor),
                "padding_idx": padding_idx,
            },
            name,
        )

    def softmax(self, a, mask=None):
        attrs = (
            {}
            if mask is None
            else {"mask": np.asarray(mask, dtype=bool).tolist()}
        )
        if mask is not None and (
            np.shape(mask) != self.shapes[a] or np.all(mask, axis=1).any()
        ):
            raise ValueError(
                "Attention mask must match scores and leave a key per row"
            )
        return self._op("softmax", [a], self.shapes[a], attrs)

    def layer_norm(self, a, weight, bias, epsilon=1e-5):
        if (
            self.shapes[weight] != (1, self.shapes[a][1])
            or self.shapes[bias] != self.shapes[weight]
        ):
            raise ValueError("LayerNorm affine shape mismatch")
        return self._op(
            "layer_norm",
            [a, weight, bias],
            self.shapes[a],
            {"epsilon": epsilon},
        )

    def concat_columns(self, values):
        if not values or len({self.shapes[v][0] for v in values}) != 1:
            raise ValueError("Concatenation requires equal row counts")
        return self._op(
            "concat_columns",
            values,
            (
                self.shapes[values[0]][0],
                sum(self.shapes[v][1] for v in values),
            ),
        )

    def sigmoid(self, a, name=None):
        return self._op("sigmoid", [a], self.shapes[a], name=name)

    def tanh(self, a, name=None):
        return self._op("tanh", [a], self.shapes[a], name=name)

    def scale(self, a, factor, name=None):
        return self._op(
            "scale", [a], self.shapes[a], {"factor": float(factor)}, name
        )

    def slice_columns(self, a, start, width, name=None):
        if start < 0 or width < 1 or start + width > self.shapes[a][1]:
            raise ValueError("Invalid column slice")
        return self._op(
            "slice_columns",
            [a],
            (self.shapes[a][0], width),
            {"start": start, "width": width},
            name,
        )

    def sum(self, a, name=None):
        return self._op("sum", [a], (1, 1), name=name)

    def add_many(self, values, name=None):
        """Elementwise sum of several equal-shape values (the planner tiles
        it).
        """
        if not values or len({self.shapes[v] for v in values}) != 1:
            raise ValueError("add_many requires equal shapes")
        return self._op(
            "add_many", list(values), self.shapes[values[0]], name=name
        )

    def transformer_layer(
        self,
        x,
        params,
        attrs,
        memory=None,
        norm=None,
        name=None,
        parts=(),
        residual=False,
    ):
        """One fused pre-norm Transformer layer over a band of examples
        (example-major rows; ygaraty.runtime.transformer_kernels).
        `params` in ENCODER_PARAMS or DECODER_PARAMS order. `norm`:
        (weight, bias) of a final layer norm the task applies to its
        output, for the stack's last layer, whose owner then holds those
        weights too.
        `parts` (experiment 78): values added to `x` inside the task,
        inputs after the memory. `residual`: the task returns only
        the layer's residual update, y - x.
        """
        from ygaraty.compiler.layouts import (
            DECODER_PARAMS,
            ENCODER_PARAMS,
        )

        parts = list(parts)
        if any(self.shapes[v] != self.shapes[x] for v in parts):
            raise ValueError("Layer input parts must be shaped as the input")
        if residual and norm is not None:
            raise ValueError(
                "A residual-only layer can't apply the final norm"
            )

        expected = DECODER_PARAMS if memory is not None else ENCODER_PARAMS
        if len(params) != len(expected):
            raise ValueError(
                f"A {'decoder' if memory is not None else 'encoder'} "
                f"layer takes {len(expected)} parameters"
            )
        if memory is not None and self.shapes[memory][1] != self.shapes[x][1]:
            raise ValueError("Memory width must match the layer width")
        if norm is not None and any(
            self.shapes[p] != (1, self.shapes[x][1]) for p in norm
        ):
            raise ValueError("Final norm weight and bias must be 1 x width")
        inputs = (
            [x]
            + ([memory] if memory is not None else [])
            + parts
            + list(params)
            + (list(norm) if norm is not None else [])
        )
        extra = dict(parts=len(parts)) if parts else {}
        if residual:
            extra["residual"] = True
        return self._op(
            "tf_layer",
            inputs,
            self.shapes[x],
            dict(
                attrs,
                decoder=memory is not None,
                final_norm=norm is not None,
                upload_rows=[self.shapes[x][0]],
                **extra,
            ),
            name,
        )

    # Owned vocabulary (experiment 32)
    # ----------------------------------------------
    # The tied embedding table and the output bias split by vocabulary rows
    # over owning workers.
    # The table and bias are inputs of these nodes so the tape keeps their
    # meaning; the planner
    # sends none of them to a worker, and they get no tape gradient.

    def vocab_gather(self, table, shard, rows, ids, name=None):
        """Rows `ids` (local to the shard's rows [v0, v1)) of the table,
        served by the shard's owner.
        """
        v0, v1 = rows
        if not ids or min(ids) < 0 or max(ids) >= v1 - v0:
            raise ValueError("Gathered rows must lie in the shard")
        node = self._op(
            "vocab_gather",
            [table],
            (len(ids), self.shapes[table][1]),
            dict(
                owner=f"vocab.{shard}",
                layer=f"vocab.{shard}",
                shard=shard,
                rows=[v0, v1],
                ids=list(ids),
            ),
            name,
        )
        self.parameter_like.add(node)
        return node

    def vocab_output(self, bands, table, bias, attrs, name=None):
        """The mean label-smoothed cross-entropy of logits = x @ table.T +
        bias, with x given as row bands (views of the decoder output, in
        order); attrs: shards, targets (per row), ignore_index,
        label_smoothing.
        """
        v, d = self.shapes[table]
        if self.shapes[bias] != (1, v) or any(
            self.shapes[b][1] != d for b in bands
        ):
            raise ValueError("Vocabulary output shapes disagree")
        targets = list(attrs["targets"])
        if len(targets) != sum(self.shapes[b][0] for b in bands):
            raise ValueError("One target per row is required")
        count = sum(1 for t in targets if t != attrs.get("ignore_index"))
        return self._op(
            "vocab_output",
            list(bands) + [table, bias],
            (1, 1),
            dict(
                attrs,
                vocab=v,
                count=count,
                rows=[self.shapes[b][0] for b in bands],
            ),
            name,
        )

    # Batched sequence operations
    # -------------------------------------------------
    # Sequences are time-major: row t*batch+b holds step t of example b.

    def view(self, a, region, name=None):
        """A rectangle [row_start, row_stop, column_start, column_stop] of
        `a`, without a copy. The planner turns it into a region of `a`
        for every task that reads it.
        """
        r0, r1, c0, c1 = region
        rows, cols = self.shapes[a]
        if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
            raise ValueError("Invalid view region")
        return self._op(
            "view", [a], (r1 - r0, c1 - c0), {"region": [r0, r1, c0, c1]}, name
        )

    def concat_rows(self, values, name=None):
        if not values or len({self.shapes[v][1] for v in values}) != 1:
            raise ValueError("Row concatenation requires equal column counts")
        if len(values) == 1:
            return values[0]
        return self._op(
            "concat_rows",
            values,
            (
                sum(self.shapes[v][0] for v in values),
                self.shapes[values[0]][1],
            ),
            name=name,
        )

    def place(self, shape, pieces, name=None):
        """Zeros of `shape` plus each (value, [row, column]) added at its
        offset.
        """
        for value, (r, c) in pieces:
            if (
                r < 0
                or c < 0
                or r + self.shapes[value][0] > shape[0]
                or c + self.shapes[value][1] > shape[1]
            ):
                raise ValueError("Piece outside the placed shape")
        return self._op(
            "place",
            [v for v, _ in pieces],
            tuple(shape),
            {"shape": list(shape), "offsets": [list(o) for _, o in pieces]},
            name,
        )

    def lstm(
        self,
        xp,
        whh_t,
        bias_ih,
        bias_hh,
        mask,
        h0,
        c0,
        batch,
        reverse=False,
        name=None,
    ):
        """One LSTM layer direction over a time-major batch, as one fused
        step per time step.

        xp: input projections X @ W_ih.T (steps*batch x 4h); whh_t:
        W_hh.T (h x 4h); mask (steps*batch x 1): 1 for real tokens, 0
        for padding, where the state is carried unchanged and the output
        is zero (packed-sequence semantics). Output per row: [h_state |
        c_state | h_out | i | f | g | o], each h wide (see
        LSTM_COLUMNS).
        """
        rows, four = self.shapes[xp]
        hidden = four // 4
        if (
            four != 4 * hidden
            or rows % batch
            or self.shapes[whh_t] != (hidden, four)
            or self.shapes[bias_ih] != (1, four)
            or self.shapes[bias_hh] != (1, four)
            or self.shapes[mask] != (rows, 1)
            or self.shapes[h0] != (batch, hidden)
            or self.shapes[c0] != (batch, hidden)
        ):
            raise ValueError("Inconsistent LSTM operand shapes")
        return self._op(
            "lstm",
            [xp, whh_t, bias_ih, bias_hh, mask, h0, c0],
            (rows, 7 * hidden),
            {"batch": batch, "hidden": hidden, "reverse": bool(reverse)},
            name,
        )

    def bqk(self, q, k, batch, name=None):
        """Per example b: Q_b @ K_b.T. Rows t*batch+b of q and s*batch+b of
        k; output (steps*batch x keys).
        """
        if (
            self.shapes[q][1] != self.shapes[k][1]
            or self.shapes[q][0] % batch
            or self.shapes[k][0] % batch
        ):
            raise ValueError(
                "bqk operands must share width and hold whole batches"
            )
        return self._op(
            "bqk",
            [q, k],
            (self.shapes[q][0], self.shapes[k][0] // batch),
            {"batch": batch},
            name,
        )

    def bpv(self, p, v, batch, name=None):
        """Per example b: P_b @ V_b, with P (steps*batch x keys) and V
        (keys*batch x width).
        """
        if (
            self.shapes[p][1] * batch != self.shapes[v][0]
            or self.shapes[p][0] % batch
        ):
            raise ValueError("bpv operands disagree on keys or batch")
        return self._op(
            "bpv",
            [p, v],
            (self.shapes[p][0], self.shapes[v][1]),
            {"batch": batch},
            name,
        )

    def bptq(self, p, q, batch, name=None):
        """Per example b: P_b.T @ Q_b, the gradient-side partner of bqk and
        bpv.
        """
        if self.shapes[p][0] != self.shapes[q][0] or self.shapes[p][0] % batch:
            raise ValueError(
                "bptq operands must share rows that hold whole batches"
            )
        return self._op(
            "bptq",
            [p, q],
            (self.shapes[p][1] * batch, self.shapes[q][1]),
            {"batch": batch},
            name,
        )

    def transpose(self, a, name=None):
        return self._op("transpose", [a], self.shapes[a][::-1], name=name)

    def _broadcast(self, a, shape):
        return self._op("broadcast", [a], shape, {"shape": list(shape)})

    def _rows(self, a):
        return self._op("reduce_rows", [a], (1, self.shapes[a][1]))

    def cross_entropy(
        self,
        scores,
        targets,
        name=None,
        ignore_index=None,
        label_smoothing=0.0,
    ):
        rows, cols = self.shapes[scores]
        labels = np.asarray(targets)
        if (
            labels.shape != (rows,)
            or labels.dtype.kind not in "iu"
            or np.any(labels < 0)
            or np.any(labels >= cols)
        ):
            raise ValueError(
                "One integer target per row, in [0, classes), is required"
            )
        if not 0 <= label_smoothing <= 1 or not np.any(labels != ignore_index):
            raise ValueError("Invalid smoothing or no unmasked targets")
        return self._op(
            "cross_entropy",
            [scores],
            (1, 1),
            {
                "targets": labels.tolist(),
                "ignore_index": ignore_index,
                "label_smoothing": label_smoothing,
            },
            name,
        )

    def backward(self, loss, input_gradients=True):
        """Append derivative operations once. Shared paths become explicit
        adds.

        input_gradients=False skips the gradient of every leaf input
        that is not a parameter (the data matrix of a factorization, an
        MLP's examples): nothing trains them, so their gradient tasks,
        often a full matrix product per tile, would be computed and
        dropped. The default keeps the complete graph, which pinned plan
        hashes depend on.
        """
        if self.loss is not None:
            raise ValueError("Call backward only once per tape")
        if self.shapes[loss] != (1, 1):
            raise ValueError("backward requires a scalar loss")
        forward = list(self.nodes)
        nodes = {n.id: n for n in forward}
        seed = self.input(None, [[1.0]])
        gradients = {loss: seed}
        # Gradients reaching a value through views, as (gradient, [row,
        # column]) pieces.
        # They are combined when the reverse pass reaches the value itself,
        # since every
        # view of a value comes after it.
        pieces = {}

        def acc(value, contribution):
            gradients[value] = (
                self.add(gradients[value], contribution)
                if value in gradients
                else contribution
            )
            if value in self.parameter_like:
                self.parameter_side.update((contribution, gradients[value]))

        # Owned vocabulary: the embeddings' gradients as their row bands (per
        # embed, in the
        # reverse pass's order), for the owners to scatter (vocab_embed_grad)
        # and for one
        # scatter of the position table, instead of a gradient of the gathered
        # table.
        embed_grads = []

        def node_of(ident):
            # Nodes made by this pass are recent: search from the end.
            return nodes.get(ident) or next(
                (n for n in reversed(self.nodes) if n.id == ident), None
            )

        def combine(value):
            parts = pieces.pop(value, [])
            if not parts:
                return
            shape = self.shapes[value]
            if (
                value not in gradients
                and len(parts) == 1
                and parts[0][1] == [0, 0]
                and self.shapes[parts[0][0]] == shape
            ):
                gradients[value] = parts[0][0]
                return
            if value not in gradients:
                # Full-width row bands covering the value in order: a
                # concatenation, which the
                # planner resolves without a copy, instead of placing them into
                # a fresh value.
                ordered, start, bands = (
                    sorted(parts, key=lambda p: p[1][0]),
                    0,
                    True,
                )
                for g, (r, c) in ordered:
                    bands = (
                        bands
                        and r == start
                        and c == 0
                        and self.shapes[g][1] == shape[1]
                    )
                    start += self.shapes[g][0]
                if bands and start == shape[0]:
                    gradients[value] = self.concat_rows(
                        [g for g, _ in ordered]
                    )
                    return
                # Several readers, each contributing the same full-width row
                # bands (the decoder
                # layers' gradients of the encoder output): one sum per band,
                # then their
                # concatenation, so a band's gradient waits only for its own
                # band's pieces. Tiles
                # of a placement would cross band boundaries. Each band's
                # pieces keep their order,
                # so the sums are place()'s, element by element.
                bands, start = {}, 0
                if all(
                    c == 0 and self.shapes[g][1] == shape[1]
                    for g, (r, c) in parts
                ):
                    for g, (r, c) in parts:
                        bands.setdefault((r, self.shapes[g][0]), []).append(g)
                    ranges = sorted(bands)
                    for r, rows in ranges:
                        if r != start:
                            break
                        start += rows
                if bands and start == shape[0] and len(bands) > 1:
                    sums = [
                        group[0] if len(group) == 1 else self.add_many(group)
                        for _, group in sorted(bands.items())
                    ]
                    gradients[value] = self.concat_rows(sums)
                    return
            whole = (
                [(gradients[value], [0, 0])] if value in gradients else []
            ) + parts
            gradients[value] = self.place(shape, whole)

        handled = set()
        for node in reversed(forward):
            if node.op == "vocab_output" and node.id in gradients:
                self._vocab_output_backward(node, gradients[node.id], acc)
                continue
            if node.op == "vocab_gather":
                combine(node.id)
                s = node.attrs["shard"]
                if embed_grads:
                    self.owned_vocab["embeds"][s] = self._vocab_embed_scatter(
                        node, embed_grads, nodes
                    )
                elif node.id in gradients:
                    self.owned_vocab["embeds"][s] = self._op(
                        "vocab_embed_grad",
                        [gradients[node.id]],
                        (1, 1),
                        dict(node.attrs),
                    )
                continue
            if node.op == "tf_layer":
                if node.id in handled:
                    continue
                # Every band of this layer at once, so each parameter's
                # gradient is one sum.
                group = [
                    n
                    for n in forward
                    if n.op == "tf_layer"
                    and n.attrs.get("layer") == node.attrs.get("layer")
                ]
                handled.update(n.id for n in group)
                self._tf_layer_backward(
                    [n for n in group if n.id in gradients], gradients, acc
                )
                continue
            if (
                node.op == "lstm"
                and node.id in pieces
                and node.id not in gradients
            ):
                self._lstm_backward(node, pieces.pop(node.id), acc)
                continue
            combine(node.id)
            if node.id not in gradients:
                continue
            g = gradients[node.id]
            a = node.inputs[0]
            b = node.inputs[1] if len(node.inputs) > 1 else None
            if node.op == "matmul":
                acc(a, self.matmul(g, self.transpose(b)))
                acc(b, self.matmul(self.transpose(a), g))
            elif node.op == "add":
                acc(a, g)
                acc(b, g)
            elif node.op == "mul":
                acc(a, self.mul(g, b))
                acc(b, self.mul(g, a))
            elif node.op == "add_bias":
                acc(a, g)
                acc(b, self._rows(g))
            elif node.op == "softmax":
                acc(
                    a,
                    self._op("softmax_backward", [node.id, g], self.shapes[a]),
                )
            elif node.op == "layer_norm":
                for index, value in enumerate(node.inputs):
                    acc(
                        value,
                        self._op(
                            "layer_norm_backward",
                            [a, node.inputs[1], g],
                            self.shapes[value],
                            dict(node.attrs, index=index),
                        ),
                    )
            elif node.op == "concat_columns":
                start = 0
                for value in node.inputs:
                    width = self.shapes[value][1]
                    acc(value, self.slice_columns(g, start, width))
                    start += width
            elif node.op == "relu":
                acc(a, self._op("relu_backward", [a, g], self.shapes[a]))
            elif node.op == "conv2d":
                for value, op, refs in (
                    (a, "conv2d_input_grad", [g, b]),
                    (b, "conv2d_weight_grad", [a, g]),
                    (node.inputs[2], "conv2d_bias_grad", [g]),
                ):
                    acc(
                        value,
                        self._op(op, refs, self.shapes[value], node.attrs),
                    )
            elif node.op in ("maxpool2d", "avgpool2d"):
                refs = [a, g] if node.op == "maxpool2d" else [g]
                acc(
                    a,
                    self._op(
                        node.op + "_grad", refs, self.shapes[a], node.attrs
                    ),
                )
            elif node.op == "embedding":
                acc(
                    a,
                    self._op(
                        "embedding_backward",
                        [g],
                        self.shapes[a],
                        dict(node.attrs, shape=list(self.shapes[a])),
                    ),
                )
            elif (
                node.op == "embed"
                and self.owned_vocab is not None
                and a == self.owned_vocab.get("table")
            ):
                # The gradient's row bands as the first layer's tasks produced
                # them (a row
                # concatenation of views costs no copy): read where they are,
                # by the owners.
                gnode = node_of(g)
                parts = (
                    list(gnode.inputs)
                    if gnode is not None and gnode.op == "concat_rows"
                    else [g]
                )
                # A band read by several layers (experiment 78) has
                # its gradient as a sum: the owners scatter each term
                # themselves, so the sum costs no task of its own.
                ids, positions = node.attrs["ids"], node.attrs["positions"]
                terms, term_ids, term_positions, row = [], [], [], 0
                for part in parts:
                    rows = self.shapes[part][0]
                    pnode = node_of(part)
                    addends = (
                        list(pnode.inputs)
                        if pnode is not None
                        and pnode.op == "add_many"
                        and all(
                            self.shapes[v] == self.shapes[part]
                            for v in pnode.inputs
                        )
                        else [part]
                    )
                    for addend in addends:
                        terms.append(addend)
                        term_ids += list(ids[row : row + rows])
                        term_positions += list(positions[row : row + rows])
                    row += rows
                embed_grads.append(
                    dict(
                        parts=terms,
                        ids=term_ids if len(terms) > len(parts) else ids,
                        factor=node.attrs["factor"],
                        padding_idx=node.attrs["padding_idx"],
                        positions=(
                            term_positions
                            if len(terms) > len(parts)
                            else positions
                        ),
                        table=b,
                    )
                )
            elif node.op == "embed":
                acc(
                    a,
                    self._op(
                        "embedding_backward",
                        [g],
                        self.shapes[a],
                        dict(
                            ids=node.attrs["ids"],
                            padding_idx=node.attrs["padding_idx"],
                            factor=node.attrs["factor"],
                            shape=list(self.shapes[a]),
                        ),
                    ),
                )
                acc(
                    b,
                    self._op(
                        "embedding_backward",
                        [g],
                        self.shapes[b],
                        dict(
                            ids=node.attrs["positions"],
                            padding_idx=None,
                            shape=list(self.shapes[b]),
                        ),
                    ),
                )
            elif node.op in ("sigmoid", "tanh"):
                acc(
                    a,
                    self._op(
                        node.op + "_backward", [node.id, g], self.shapes[a]
                    ),
                )
            elif node.op == "scale":
                acc(a, self.scale(g, node.attrs["factor"]))
            elif node.op == "slice_columns":
                acc(
                    a,
                    self._op(
                        "slice_backward",
                        [g],
                        self.shapes[a],
                        dict(node.attrs, shape=list(self.shapes[a])),
                    ),
                )
            elif node.op == "sum":
                acc(a, self._broadcast(g, self.shapes[a]))
            elif node.op == "transpose":
                acc(a, self.transpose(g))
            elif node.op == "cross_entropy":
                acc(
                    a,
                    self._op(
                        "ce_backward", [a, g], self.shapes[a], node.attrs
                    ),
                )
            elif node.op == "view":
                r0, _, c0, _ = node.attrs["region"]
                pieces.setdefault(a, []).append((g, [r0, c0]))
            elif node.op == "concat_rows":
                start = 0
                for value in node.inputs:
                    rows, cols = self.shapes[value]
                    acc(value, self.view(g, [start, start + rows, 0, cols]))
                    start += rows
            elif node.op == "place":
                for value, (r, c) in zip(node.inputs, node.attrs["offsets"]):
                    rows, cols = self.shapes[value]
                    acc(value, self.view(g, [r, r + rows, c, c + cols]))
            elif node.op == "lstm":
                self._lstm_backward(node, [(g, [0, 0])], acc)
            elif node.op == "bqk":
                acc(a, self.bpv(g, b, node.attrs["batch"]))
                acc(b, self.bptq(g, a, node.attrs["batch"]))
            elif node.op == "bpv":
                acc(a, self.bqk(g, b, node.attrs["batch"]))
                acc(b, self.bptq(a, g, node.attrs["batch"]))
            else:
                raise ValueError(f"No derivative rule for {node.op}")
        # The position table's gradient: one scatter over every embedding's row
        # bands that read it
        # (hint rows, experiment 55, read a zero table of their own).
        for table in dict.fromkeys(e["table"] for e in embed_grads):
            group = [e for e in embed_grads if e["table"] == table]
            acc(
                table,
                self._op(
                    "embedding_backward",
                    [p for e in group for p in e["parts"]],
                    self.shapes[table],
                    dict(
                        ids=[i for e in group for i in e["positions"]],
                        padding_idx=None,
                        shape=list(self.shapes[table]),
                    ),
                ),
            )
        for value in list(pieces):
            combine(value)
        self.loss = loss
        self.gradients = {}
        owned = {
            p for layer in self.owned_layers.values() for p in layer["params"]
        }
        owned.update(self.owned_vocab["params"] if self.owned_vocab else ())
        for p in self.parameters:
            if p in owned:
                continue
            g = (
                gradients[p]
                if p in gradients
                else self.input(None, np.zeros(self.shapes[p]))
            )
            # The optimizer reads gradients as stored values, never as views.
            self.gradients[p] = (
                self.scale(g, 1.0)
                if nodes.get(g) and nodes[g].op == "view"
                else g
            )
            self.parameter_side.add(self.gradients[p])
        # A rearrangement (a sum, a view, a transpose: the unpacking of a
        # layer's packed
        # gradients) read only by parameter-side nodes is parameter-side too. A
        # kernel's output
        # is not: a fused backward result holds activation gradients beside
        # parameter ones.
        rearrangements = {
            "transpose",
            "view",
            "concat_columns",
            "concat_rows",
            "add",
            "add_many",
            "scale",
            "place",
        }
        consumers = {}
        for n in self.nodes:
            for v in n.inputs:
                consumers.setdefault(v, set()).add(n.id)
        changed = True
        while changed:
            changed = False
            for n in self.nodes:
                if (
                    n.op in rearrangements
                    and n.id not in self.parameter_side
                    and consumers.get(n.id)
                    and consumers[n.id] <= self.parameter_side
                ):
                    self.parameter_side.add(n.id)
                    changed = True
        if (
            not input_gradients
            and not self.owned_layers
            and not self.owned_vocab
        ):
            # Drop the reverse-pass nodes that reach neither the loss nor a
            # parameter gradient:
            # the gradients of untrained inputs and the sums that fed them.
            # Forward nodes stay.
            by_id = {n.id: n for n in self.nodes}
            live, todo = (
                {loss, *self.gradients.values()},
                [loss, *self.gradients.values()],
            )
            while todo:
                node = by_id.get(todo.pop())
                for value in node.inputs if node else ():
                    if value not in live:
                        live.add(value)
                        todo.append(value)
            self.nodes = [
                n for n in self.nodes if n.id in live or n.id in nodes
            ]
            self.parameter_side &= {n.id for n in self.nodes} | set(
                self.values
            )
        return dict(self.gradients)

    def _vocab_embed_scatter(self, gather, embed_grads, nodes):
        """The shard's vocab_embed_grad reading the embeddings' gradient
        bands directly: per part, each row's place among the shard's
        gathered rows (-1: another shard's, or padding), so the owner
        scatters its rows itself. Sides (one per embed) are scattered in
        turn, then summed in the order the tape would have added them.
        """
        concat = nodes[self.owned_vocab["table"]]
        offset = 0
        for part in concat.inputs:
            if part == gather.id:
                break
            offset += self.shapes[part][0]
        count = self.shapes[gather.id][0]
        inputs, scatter, sides, factors = [], [], [], []
        for e in embed_grads:
            row = 0
            for part in e["parts"]:
                rows = self.shapes[part][0]
                ids = e["ids"][row : row + rows]
                scatter.append(
                    [
                        x - offset
                        if offset <= x < offset + count
                        and x != e["padding_idx"]
                        else -1
                        for x in ids
                    ]
                )
                inputs.append(part)
                row += rows
            sides.append(len(e["parts"]))
            factors.append(e["factor"])
        return self._op(
            "vocab_embed_grad",
            inputs,
            (1, 1),
            dict(gather.attrs, scatter=scatter, sides=sides, factors=factors),
        )

    def _vocab_output_backward(self, node, seed, acc):
        """Per band and shard, one vocab_grad: the shard's part of the
        band's gradient (its rows first, then a marker row the shard's
        step depends on). The band's gradient is their sum.
        """
        bands, (table, bias) = node.inputs[:-2], node.inputs[-2:]
        shards = self.owned_vocab["shards"]
        start = 0
        for k, band in enumerate(bands):
            rows, d = self.shapes[band]
            parts = []
            for s, (v0, v1) in enumerate(shards):
                attrs = dict(
                    owner=f"vocab.{s}",
                    layer=f"vocab.{s}",
                    shard=s,
                    shards=len(shards),
                    rows=[v0, v1],
                    band=k,
                    loss=node.id,
                    targets=node.attrs["targets"][start : start + rows],
                    vocab=node.attrs["vocab"],
                    count=node.attrs["count"],
                    ignore_index=node.attrs.get("ignore_index"),
                    label_smoothing=node.attrs.get("label_smoothing", 0.0),
                    upload_rows=[rows, 1],
                )
                full = self._op(
                    "vocab_grad",
                    [band, table, bias, seed],
                    (rows + 1, d),
                    attrs,
                )
                self.owned_vocab["grads"].setdefault(s, []).append(full)
                parts.append(self.view(full, [0, rows, 0, d]))
            acc(band, parts[0] if len(parts) == 1 else self.add_many(parts))
            start += rows

    def _tf_layer_backward(self, nodes, gradients, acc):
        """Gradients of fused layer bands: one tf_layer_grad per band,
        whose packed output holds the input's gradient, the memory's
        (decoder) and every parameter's (packed_layout).
        """
        from ygaraty.compiler.layouts import packed_layout

        contributions = {}
        owned = bool(nodes) and nodes[0].attrs.get("owned", False)
        if owned:
            tag = nodes[0].attrs["layer"]
            first = nodes[0]
            self.owned_layers[tag] = dict(
                params=list(
                    first.inputs[
                        (2 if first.attrs["decoder"] else 1)
                        + first.attrs.get("parts", 0) :
                    ]
                ),
                decoder=first.attrs["decoder"],
                forward=[n.id for n in nodes],
                backward=[],
            )
        for node in nodes:
            decoder = node.attrs["decoder"]
            x = node.inputs[0]
            mem = node.inputs[1] if decoder else None
            start = 2 if decoder else 1
            extra = node.inputs[start : start + node.attrs.get("parts", 0)]
            params = node.inputs[start + len(extra) :]
            rows, d = self.shapes[x]
            mrows = self.shapes[mem][0] if decoder else 0
            layout, packed = packed_layout([self.shapes[p] for p in params], d)
            # upload_rows: the bands a relay upload is cut into, one per reader
            # (the input's
            # gradient, the memory's, each parameter's), so no reader fetches
            # more than its own.
            bounds = (
                [rows]
                + ([mrows] if decoder else [])
                + [count for _, count, _ in layout]
            )
            full = self._op(
                "tf_layer_grad",
                [x]
                + ([mem] if decoder else [])
                + list(extra)
                + [gradients[node.id]]
                + list(params),
                (rows + mrows + packed, d),
                dict(node.attrs, upload_rows=bounds),
            )
            dx = self.view(full, [0, rows, 0, d])
            acc(x, dx)
            for part in extra:  # x + parts: the same gradient for each
                acc(part, dx)
            if decoder:
                acc(mem, self.view(full, [rows, rows + mrows, 0, d]))
            if owned:
                self.owned_layers[tag]["backward"].append(full)
                continue  # the owner keeps the parameter gradients
            base = rows + mrows
            for p, (offset, count, mode) in zip(params, layout):
                contributions.setdefault(p, []).append(
                    (
                        self.view(
                            full, [base + offset, base + offset + count, 0, d]
                        ),
                        count,
                        mode,
                    )
                )
        # Sum the packed rows over the bands first, then unpack once per layer:
        # one transpose
        # or concatenation per parameter instead of one per band.
        for p, parts in contributions.items():
            (first, count, mode), d = parts[0], self.shapes[parts[0][0]][1]
            summed = (
                first
                if len(parts) == 1
                else self.add_many([v for v, _, _ in parts])
            )
            if mode == "rows":
                part = summed
            elif mode == "transposed":
                part = self.transpose(summed)
            else:
                b = self.shapes[p][1]
                slices = [
                    self.view(summed, [r, r + 1, 0, min(d, b - r * d)])
                    for r in range(count)
                ]
                part = (
                    slices[0]
                    if len(slices) == 1
                    else self.concat_columns(slices)
                )
            acc(p, part)

    def _lstm_backward(self, node, parts, acc):
        """Gradient of an `lstm` node from its gradient pieces.

        Only h_out (all rows) and the final state (h_state, c_state of
        the last step processed) normally receive gradients; anything
        else is placed into a full-width gradient first. lstm_backward
        returns, per step row, the gate gradients (4h) and the carry it
        passed to the previous step (2h), then a last block of rows [dh0
        | dc0 | zeros].
        """
        xp, whh_t, bias_ih, bias_hh, mask, h0, c0 = node.inputs
        batch, hidden, reverse = (
            node.attrs["batch"],
            node.attrs["hidden"],
            node.attrs["reverse"],
        )
        rows = self.shapes[xp][0]
        final = 0 if reverse else rows - batch
        out, state, other = [], [], []
        for g, (r, c) in parts:
            gr, gc = self.shapes[g]
            if c >= 2 * hidden and c + gc <= 3 * hidden:
                out.append((g, [r, c - 2 * hidden]))
            elif (
                r >= final and r + gr <= final + batch and c + gc <= 2 * hidden
            ):
                state.append((g, [r - final, c]))
            else:
                other.append((g, [r, c]))
        inputs, flags = [node.id], {}
        if other:
            # Uncommon: gradient into gate columns or earlier states. Fold
            # everything into
            # one full-width gradient and let the kernel read every part of it.
            inputs.append(self.place(self.shapes[node.id], parts))
            flags["full"] = True
        else:
            if out:
                inputs.append(
                    out[0][0]
                    if len(out) == 1
                    and out[0][1] == [0, 0]
                    and self.shapes[out[0][0]] == (rows, hidden)
                    else self.place((rows, hidden), out)
                )
                flags["has_out"] = True
            if state:
                inputs.append(self.place((batch, 2 * hidden), state))
                flags["has_state"] = True
        whh = self.transpose(whh_t)
        inputs += [whh, mask, h0, c0]
        full = self._op(
            "lstm_backward",
            inputs,
            (rows + batch, 6 * hidden),
            dict(node.attrs, **flags),
        )
        gates = self.view(full, [0, rows, 0, 4 * hidden])
        acc(xp, gates)
        bias = self._rows(gates)
        acc(bias_ih, bias)
        acc(bias_hh, bias)
        # Each step's recurrent input is the previous step's h_state (h0 for
        # the first).
        steps = rows // batch
        if steps == 1:
            previous = h0
        elif reverse:
            previous = self.concat_rows(
                [self.view(node.id, [batch, rows, 0, hidden]), h0]
            )
        else:
            previous = self.concat_rows(
                [h0, self.view(node.id, [0, rows - batch, 0, hidden])]
            )
        acc(whh_t, self.matmul(self.transpose(previous), gates))
        acc(h0, self.view(full, [rows, rows + batch, 0, hidden]))
        acc(c0, self.view(full, [rows, rows + batch, hidden, 2 * hidden]))
