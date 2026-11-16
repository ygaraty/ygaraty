"""Recurrent task planning using the shared sequence lowering context."""

from .sequence_common import order, spans
from .tasks import Input


class RecurrentLowering:
    """Internal lowering mixin; state and emission belong to
    sequence.Lowering.
    """

    def recurrent(self, prefix, rows, left, left_columns, right, out_width):
        """Partial products left[:, left_columns] @ right, tiled; returns
        (inputs, blocks).
        """
        start, width = left_columns
        inputs, blocks = [], []
        for j0, j1 in spans(out_width, self.tile):
            ks = spans(width, self.tile)
            for q0, q1 in ks:
                inputs.append(
                    Input(
                        self.emit(
                            f"{prefix}/{j0}/{q0}",
                            "matmul",
                            [
                                Input(left, [0, rows, start + q0, start + q1]),
                                Input(right, [q0, q1, j0, j1]),
                            ],
                            (rows, j1 - j0),
                        )
                    )
                )
            blocks.append([j0, j1, len(ks)])
        return inputs, blocks

    def inline_recurrent(self, weight):
        """Whether a step task computes its recurrent product itself, reading
        the whole recurrent weight (cached at the worker, since it is a
        view of a leaf). One hop per step instead of two. Beyond four
        tiles' worth of weight the product stays split into separate tasks.
        """
        rows, cols = self.tape.shapes[weight]
        return rows * cols <= 4 * self.tile * self.tile

    def steps_per_task(self, per_step, fixed, width, batch):
        """Consecutive LSTM steps one chain task may run: at most chain_steps,
        with its output (`width` columns per step) under the output cap and
        its arrays (`fixed` elements plus `per_step` per step) within the
        task budget. 1 means one task per step, as before.
        """
        if self.chain_bytes:
            # admit_plan's estimate is four times the bytes a task moves
            # (plan.admit_plan).
            return max(
                1,
                min(
                    self.chain_steps,
                    (self.chain_bytes // 32 - fixed) // per_step,
                ),
            )
        k = min(
            self.chain_steps,
            max(1, self.output_cap // (batch * width * 8)),
            max(1, (self.budget // 8 - fixed) // per_step),
        )
        return max(1, k)

    def lower_lstm(self, node):
        xp, whh_t, bias_ih, bias_hh, mask, h0, c0 = node.inputs
        batch, hidden, reverse = (
            node.attrs["batch"],
            node.attrs["hidden"],
            node.attrs["reverse"],
        )
        rows = self.tape.shapes[xp][0]
        inline = self.inline_recurrent(whh_t)
        k = (
            self.steps_per_task(
                batch * (4 * hidden + 1 + 7 * hidden),
                hidden * 4 * hidden + 8 * hidden + 2 * batch * hidden,
                7 * hidden,
                batch,
            )
            if inline
            else 1
        )
        if k > 1:
            # Chunks of consecutive steps, in processing order, each one task
            # (the whole-layer
            # kernel over the chunk's rows). The next chunk reads its state
            # from the row of the
            # last step the previous chunk processed.
            walk, previous, parts = (
                list(order(rows // batch, reverse)),
                None,
                [],
            )
            for i in range(0, len(walk), k):
                chunk = walk[i : i + k]
                lo, hi = min(chunk), max(chunk)
                r = [lo * batch, (hi + 1) * batch]
                if previous is None:
                    state_h, state_c = Input(h0), Input(c0)
                else:
                    task, j = previous
                    state_h = Input(
                        task, [j * batch, (j + 1) * batch, 0, hidden]
                    )
                    state_c = Input(
                        task, [j * batch, (j + 1) * batch, hidden, 2 * hidden]
                    )
                ident = self.emit(
                    f"{node.id}/chain/{lo}",
                    "lstm_chain",
                    [
                        Input(xp, r + [0, 4 * hidden]),
                        Input(bias_ih),
                        Input(bias_hh),
                        state_h,
                        state_c,
                        Input(mask, r + [0, 1]),
                        Input(whh_t),
                    ],
                    (len(chunk) * batch, 7 * hidden),
                    {
                        "hidden": hidden,
                        "batch": batch,
                        "reverse": reverse,
                        "inline": True,
                    },
                )
                previous = (ident, chunk[-1] - lo)
                parts.append((ident, r + [0, 7 * hidden]))
            self.assemble(
                node.id, node.shape, sorted(parts, key=lambda p: p[1][0])
            )
            return
        previous, parts = None, []
        for t in order(rows // batch, reverse):
            r = [t * batch, (t + 1) * batch]
            h_source, h_columns = (
                (h0, 0) if previous is None else (previous, 0)
            )
            if inline:
                partials, attrs = (
                    [Input(whh_t)],
                    {"hidden": hidden, "inline": True},
                )
            else:
                partials, blocks = self.recurrent(
                    f"{node.id}/hw/{t}",
                    batch,
                    h_source,
                    (h_columns, hidden),
                    whh_t,
                    4 * hidden,
                )
                attrs = {"hidden": hidden, "blocks": blocks}
            state_h = (
                Input(h0)
                if previous is None
                else Input(previous, [0, batch, 0, hidden])
            )
            state_c = (
                Input(c0)
                if previous is None
                else Input(previous, [0, batch, hidden, 2 * hidden])
            )
            previous = self.emit(
                f"{node.id}/s/{t}",
                "lstm_cell",
                [
                    Input(xp, r + [0, 4 * hidden]),
                    Input(bias_ih),
                    Input(bias_hh),
                    state_h,
                    state_c,
                    Input(mask, r + [0, 1]),
                ]
                + partials,
                (batch, 7 * hidden),
                attrs,
            )
            parts.append((previous, r + [0, 7 * hidden]))
        self.assemble(
            node.id, node.shape, sorted(parts, key=lambda p: p[1][0])
        )

    def lower_lstm_backward(self, node):
        attrs = node.attrs
        if attrs.get("full"):
            raise NotImplementedError(
                "Tiled LSTM gradients support h_out and "
                "final-state gradients only"
            )
        values = list(node.inputs)
        out_value = values.pop(0)
        g_out = values.pop(0) if attrs.get("has_out") else None
        g_state = values.pop(0) if attrs.get("has_state") else None
        whh, mask, h0, c0 = values
        batch, hidden, reverse = (
            attrs["batch"],
            attrs["hidden"],
            attrs["reverse"],
        )
        rows = self.tape.shapes[out_value][0]
        walk = list(order(rows // batch, reverse))
        # One task per step produces [gate gradients (4h) | carry (2h)]; the
        # next step reads
        # the gates as the left operand of its recurrent products and the carry
        # as a region.
        inline = self.inline_recurrent(whh)
        k = (
            self.steps_per_task(
                batch * (7 * hidden + 1 + hidden + 6 * hidden),
                hidden * 4 * hidden + 8 * batch * hidden,
                6 * hidden,
                batch,
            )
            if inline
            else 1
        )
        if k > 1:
            # Chunks of consecutive steps in processing order (the reverse of
            # the forward walk).
            # A chunk reads the forward rows of its steps plus the row before
            # its last processed
            # step (for that step's c_prev), or c0 when there is none.
            proc, previous, parts = list(reversed(walk)), None, []
            for i in range(0, len(proc), k):
                chunk = proc[i : i + k]
                lo, hi = min(chunk), max(chunk)
                before = proc[i + k] if i + k < len(proc) else None
                out_lo, out_hi = (
                    (lo, hi)
                    if before is None
                    else (min(lo, before), max(hi, before))
                )
                inputs = [
                    Input(
                        out_value,
                        [out_lo * batch, (out_hi + 1) * batch, 0, 7 * hidden],
                    )
                ]
                flags = {
                    "hidden": hidden,
                    "batch": batch,
                    "reverse": reverse,
                    "steps": len(chunk),
                    "offset": lo - out_lo,
                    "inline": True,
                }
                if before is None:
                    inputs.append(Input(c0))
                    flags["has_c0"] = True
                inputs.append(
                    Input(mask, [lo * batch, (hi + 1) * batch, 0, 1])
                )
                if g_out is not None:
                    inputs.append(
                        Input(g_out, [lo * batch, (hi + 1) * batch, 0, hidden])
                    )
                    flags["has_out"] = True
                if previous is None:
                    if g_state is not None:
                        inputs.append(Input(g_state))
                        flags["has_carry"] = True
                else:
                    task, j = previous
                    inputs.append(
                        Input(
                            task,
                            [
                                j * batch,
                                (j + 1) * batch,
                                4 * hidden,
                                6 * hidden,
                            ],
                        )
                    )
                    flags["has_carry"] = True
                    inputs.append(
                        Input(
                            task, [j * batch, (j + 1) * batch, 0, 4 * hidden]
                        )
                    )
                    flags["has_gates"] = True
                inputs.append(Input(whh))
                ident = self.emit(
                    f"{node.id}/chain/{lo}",
                    "lstm_chain_grad",
                    inputs,
                    (len(chunk) * batch, 6 * hidden),
                    flags,
                )
                previous = (ident, chunk[-1] - lo)
                parts.append(
                    (ident, [lo * batch, (hi + 1) * batch, 0, 6 * hidden])
                )
            task, j = previous
            state = self.emit(
                f"{node.id}/dstate",
                "lstm_state_grad",
                [
                    Input(
                        task,
                        [j * batch, (j + 1) * batch, 4 * hidden, 6 * hidden],
                    ),
                    Input(task, [j * batch, (j + 1) * batch, 0, 4 * hidden]),
                    Input(whh),
                ],
                (batch, 6 * hidden),
                {
                    "hidden": hidden,
                    "has_carry": True,
                    "width": 6 * hidden,
                    "inline": True,
                },
            )
            parts = sorted(parts, key=lambda p: p[1][0]) + [
                (state, [rows, rows + batch, 0, 6 * hidden])
            ]
            self.assemble(node.id, node.shape, parts)
            return
        previous = None
        parts = []
        for k, t in enumerate(reversed(walk)):
            r = [t * batch, (t + 1) * batch]
            before = walk[len(walk) - 2 - k] if k < len(walk) - 1 else None
            c_prev = (
                Input(c0)
                if before is None
                else Input(
                    out_value,
                    [before * batch, (before + 1) * batch, hidden, 2 * hidden],
                )
            )
            inputs, flags = (
                [
                    Input(out_value, r + [0, 7 * hidden]),
                    c_prev,
                    Input(mask, r + [0, 1]),
                ],
                {"hidden": hidden},
            )
            if g_out is not None:
                inputs.append(Input(g_out, r + [0, hidden]))
                flags["has_out"] = True
            if previous is None and g_state is not None:
                inputs.append(Input(g_state))
                flags["has_carry"] = True
            elif previous is not None:
                inputs.append(
                    Input(previous, [0, batch, 4 * hidden, 6 * hidden])
                )
                flags["has_carry"] = True
            if previous is None:
                partials, blocks = [], []
            elif inline:
                partials, blocks = (
                    [Input(previous, [0, batch, 0, 4 * hidden]), Input(whh)],
                    None,
                )
                flags["inline"] = True
            else:
                partials, blocks = self.recurrent(
                    f"{node.id}/dh/{t}",
                    batch,
                    previous,
                    (0, 4 * hidden),
                    whh,
                    hidden,
                )
            if blocks is not None:
                flags["blocks"] = blocks
            previous = self.emit(
                f"{node.id}/dg/{t}",
                "lstm_cell_grad",
                inputs + partials,
                (batch, 6 * hidden),
                flags,
            )
            parts.append((previous, r + [0, 6 * hidden]))
        final = {"hidden": hidden, "has_carry": True, "width": 6 * hidden}
        if inline:
            partials, final["inline"] = (
                [Input(previous, [0, batch, 0, 4 * hidden]), Input(whh)],
                True,
            )
        else:
            partials, final["blocks"] = self.recurrent(
                f"{node.id}/dh0", batch, previous, (0, 4 * hidden), whh, hidden
            )
        state = self.emit(
            f"{node.id}/dstate",
            "lstm_state_grad",
            [Input(previous, [0, batch, 4 * hidden, 6 * hidden])] + partials,
            (batch, 6 * hidden),
            final,
        )
        parts = sorted(parts, key=lambda p: p[1][0]) + [
            (state, [rows, rows + batch, 0, 6 * hidden])
        ]
        self.assemble(node.id, node.shape, parts)
