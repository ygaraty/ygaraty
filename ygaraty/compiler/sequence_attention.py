"""Attention task planning using the shared sequence lowering context."""

from .sequence_common import spans
from .tasks import Input


class AttentionLowering:
    """Internal lowering mixin; state and emission belong to
    sequence.Lowering.
    """

    def chunk_steps(self, batch):
        return max(1, self.tile // batch) * batch

    def narrow(self, rows):
        """Column block width for an operand read at its full height: about
        one tile's elements.
        """
        return max(1, min(self.tile, self.tile * self.tile // rows))

    def lower_bqk(self, node):
        q, k = node.inputs
        batch, (rows, width), keys = (
            node.attrs["batch"],
            self.tape.shapes[q],
            self.tape.shapes[k][0],
        )
        parts = []
        for r0, r1 in spans(rows, self.chunk_steps(batch)):
            partials = [
                Input(
                    self.emit(
                        f"{node.id}/rows/{r0}/{d0}",
                        "bqk",
                        [
                            Input(q, [r0, r1, d0, d1]),
                            Input(k, [0, keys, d0, d1]),
                        ],
                        (r1 - r0, node.shape[1]),
                        node.attrs,
                    )
                )
                for d0, d1 in spans(width, self.narrow(keys))
            ]
            chunk = (
                partials[0].value
                if len(partials) == 1
                else self.emit(
                    f"{node.id}/rows/{r0}",
                    "add_many",
                    partials,
                    (r1 - r0, node.shape[1]),
                )
            )
            parts.append((chunk, [r0, r1, 0, node.shape[1]]))
        self.assemble(node.id, node.shape, parts)

    def lower_bpv(self, node):
        p, v = node.inputs
        batch, rows, keys = (
            node.attrs["batch"],
            node.shape[0],
            self.tape.shapes[p][1],
        )
        parts = []
        for r0, r1 in spans(rows, self.chunk_steps(batch)):
            for j0, j1 in spans(
                node.shape[1], self.narrow(self.tape.shapes[v][0])
            ):
                parts.append(
                    (
                        self.emit(
                            f"{node.id}/block/{r0}/{j0}",
                            "bpv",
                            [
                                Input(p, [r0, r1, 0, keys]),
                                Input(v, [0, self.tape.shapes[v][0], j0, j1]),
                            ],
                            (r1 - r0, j1 - j0),
                            node.attrs,
                        ),
                        [r0, r1, j0, j1],
                    )
                )
        self.assemble(node.id, node.shape, parts)

    def lower_bptq(self, node):
        p, q = node.inputs
        batch, rows = node.attrs["batch"], self.tape.shapes[p][0]
        parts = []
        for s0, s1 in spans(
            self.tape.shapes[p][1], self.chunk_steps(batch) // batch
        ):
            for j0, j1 in spans(node.shape[1], self.narrow(rows)):
                parts.append(
                    (
                        self.emit(
                            f"{node.id}/block/{s0}/{j0}",
                            "bptq",
                            [
                                Input(p, [0, rows, s0, s1]),
                                Input(q, [0, rows, j0, j1]),
                            ],
                            ((s1 - s0) * batch, j1 - j0),
                            node.attrs,
                        ),
                        [s0 * batch, s1 * batch, j0, j1],
                    )
                )
        self.assemble(node.id, node.shape, parts)

    def lower_rows(self, node):
        """Row-wise operations over long rows (the vocabulary): chunks of
        whole rows.
        """
        rows, cols = self.tape.shapes[node.inputs[0]]
        chunk = max(1, self.tile * self.tile // cols)
        if rows <= chunk:
            return False
        attrs, parts = dict(node.attrs), []
        if node.op in ("cross_entropy", "ce_backward"):
            targets = attrs["targets"]
            attrs["count"] = sum(
                1 for x in targets if x != attrs.get("ignore_index")
            )
        for r0, r1 in spans(rows, chunk):
            piece = dict(attrs)
            if "targets" in piece:
                piece["targets"] = attrs["targets"][r0:r1]
            if "mask" in piece:
                piece["mask"] = attrs["mask"][r0:r1]
            inputs = [Input(node.inputs[0], [r0, r1, 0, cols])]
            if node.op == "ce_backward":
                inputs.append(Input(node.inputs[1]))
            elif node.op == "softmax_backward":
                inputs.append(Input(node.inputs[1], [r0, r1, 0, cols]))
            shape = (1, 1) if node.op == "cross_entropy" else (r1 - r0, cols)
            parts.append(
                (
                    self.emit(
                        f"{node.id}/rows/{r0}", node.op, inputs, shape, piece
                    ),
                    [r0, r1, 0, cols],
                )
            )
        if node.op == "cross_entropy":
            self.emit(
                node.id, "add_many", [Input(v) for v, _ in parts], (1, 1)
            )
        else:
            self.assemble(node.id, node.shape, parts)
        return True
