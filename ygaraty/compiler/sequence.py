"""Sequence layout, dispatch and metadata passes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .sequence_attention import AttentionLowering
from .sequence_common import PUBLISH_WIRE, order, spans
from .sequence_recurrent import RecurrentLowering
from .sequence_transformer import TransformerLowering
from .sequence_views import prune, resolve_views
from .tasks import Input, Task

if TYPE_CHECKING:
    from .tape import Node, Tape

__all__ = [
    "Lowering",
    "PUBLISH_WIRE",
    "ROW_OPS",
    "SEQUENCE_OPS",
    "order",
    "prune",
    "resolve_views",
    "spans",
]

SEQUENCE_OPS = {
    "view",
    "concat_rows",
    "concat_columns",
    "place",
    "lstm",
    "lstm_backward",
    "bqk",
    "bpv",
    "bptq",
    "tf_layer",
    "tf_layer_grad",
    "vocab_gather",
    "vocab_output",
    "vocab_grad",
    "vocab_embed_grad",
}
ROW_OPS = {"cross_entropy", "ce_backward", "softmax", "softmax_backward"}


class Lowering(TransformerLowering, RecurrentLowering, AttentionLowering):
    """Shared per-plan state and emission for stateless operation-family
    mixins.
    """

    def __init__(
        self,
        tape: Tape,
        tile: int,
        tasks: list[Task],
        budget: int | None = None,
        output_cap: int | None = None,
        chain_steps: int = 1,
        chain_bytes: int | None = None,
        owned: dict[str, Any] | None = None,
    ) -> None:
        self.tape, self.tile, self.tasks = tape, tile, tasks
        self.budget = budget if budget is not None else 3 * tile * tile * 8
        self.output_cap = (
            output_cap if output_cap is not None else tile * tile * 8
        )
        self.chain_steps, self.chain_bytes = chain_steps, chain_bytes
        self.owned = owned
        self.views = {n.id for n in tape.nodes if n.op == "view"}

    def emit(
        self,
        ident: str,
        op: str,
        inputs: list[Input],
        shape: tuple[int, int],
        attrs: dict[str, Any] | None = None,
    ) -> str:
        self.tasks.append(Task(ident, op, inputs, tuple(shape), attrs or {}))
        return ident

    def assemble(self, ident, shape, parts):
        """parts: (whole value, [r0, r1, c0, c1]) blocks covering `shape`. One
        part is renamed, not assembled.

        Many full-width row parts (the steps of an LSTM chain) are first
        assembled per band of `tile` rows, so that a consumer of one band
        depends on that band's steps only.
        """
        if len(parts) == 1 and parts[0][1] == [0, shape[0], 0, shape[1]]:
            task = next(t for t in reversed(self.tasks) if t.id == parts[0][0])
            task.id = ident
            return ident
        rows, cols = shape
        if len(parts) > 2 and all(
            c0 == 0 and c1 == cols and r1 - r0 < self.tile
            for _, (r0, r1, c0, c1) in parts
        ):
            bands = []
            for b0, b1 in spans(rows, self.tile):
                inside = [
                    (v, r) for v, r in parts if b0 <= r[0] and r[1] <= b1
                ]
                if (
                    len(inside)
                    == len([1 for _, r in parts if r[0] < b1 and b0 < r[1]])
                    and inside
                ):
                    if len(inside) == 1:
                        bands.append((inside[0][0], inside[0][1]))
                        continue
                    band = self.emit(
                        f"{ident}/band/{b0}",
                        "assemble",
                        [Input(v) for v, _ in inside],
                        (b1 - b0, cols),
                        {
                            "shape": [b1 - b0, cols],
                            "blocks": [
                                [r0 - b0, r1 - b0, 0, cols]
                                for _, (r0, r1, _, _) in inside
                            ],
                        },
                    )
                    bands.append((band, [b0, b1, 0, cols]))
                else:
                    bands = None
                    break
            if bands:
                parts = bands
        return self.emit(
            ident,
            "assemble",
            [Input(v) for v, _ in parts],
            shape,
            {"shape": list(shape), "blocks": [r for _, r in parts]},
        )

    def place(self, ident, shape, pieces):
        """Zeros of `shape` plus each (value, [row, column]) piece, tiled into
        place_block tasks.
        """
        parts = []
        # Column span of a placement block: as wide as the budget allows (a
        # block reads at most its
        # own size in pieces and writes itself), so row consumers depend on one
        # block.
        width = self.tile
        while (
            width < shape[1]
            and 2 * self.tile * min(shape[1], 2 * width) * 8 <= self.budget
            and self.tile * min(shape[1], 2 * width) * 8 <= self.output_cap
        ):
            width *= 2
        for r0, r1 in spans(shape[0], self.tile):
            for c0, c1 in spans(shape[1], width):
                inputs, offsets = [], []
                for value, (r, c) in pieces:
                    rows, cols = self.tape.shapes[value]
                    i0, i1, j0, j1 = (
                        max(r0, r),
                        min(r1, r + rows),
                        max(c0, c),
                        min(c1, c + cols),
                    )
                    if i0 < i1 and j0 < j1:
                        inputs.append(
                            Input(value, [i0 - r, i1 - r, j0 - c, j1 - c])
                        )
                        offsets.append([i0 - r0, j0 - c0])
                parts.append(
                    (
                        self.emit(
                            f"{ident}/block/{r0}/{c0}",
                            "place_block",
                            inputs,
                            (r1 - r0, c1 - c0),
                            {"shape": [r1 - r0, c1 - c0], "offsets": offsets},
                        ),
                        [r0, r1, c0, c1],
                    )
                )
        return self.assemble(ident, shape, parts)

    def whole(self, value: str) -> str:
        """A stored copy of a view, for operations that read whole values
        (transpose, assemble).
        """
        if value not in self.views:
            return value
        # One copy per view, however many values read it whole (experiment 78:
        # a parallel layer group's input gradient feeds every part's).
        copies = self.__dict__.setdefault("_copies", {})
        if value not in copies:
            copies[value] = self.place(
                value + "/copy", self.tape.shapes[value], [(value, [0, 0])]
            )
        return copies[value]

    def lower(self, node: Node) -> bool:
        handler = getattr(self, "lower_" + node.op, None)
        if node.op == "transpose" and node.inputs[0] in self.views:
            self.emit(
                node.id,
                "transpose",
                [Input(self.whole(node.inputs[0]))],
                node.shape,
                node.attrs,
            )
            return True
        if node.op in ROW_OPS:
            return self.lower_rows(node)
        if node.op not in SEQUENCE_OPS:
            return False
        handler(node)
        return True

    def lower_view(self, node: Node) -> None:
        """Views are metadata; resolve_views rewrites their consumers
        later.
        """

    def lower_concat_rows(self, node):
        cols, start, parts = node.shape[1], 0, []
        for value in node.inputs:
            rows = self.tape.shapes[value][0]
            parts.append((self.whole(value), [start, start + rows, 0, cols]))
            start += rows
        self.assemble(node.id, node.shape, parts)

    def lower_concat_columns(self, node):
        offsets, start = [], 0
        for value in node.inputs:
            offsets.append((value, [0, start]))
            start += self.tape.shapes[value][1]
        self.place(node.id, node.shape, offsets)

    def lower_place(self, node):
        self.place(
            node.id, node.shape, list(zip(node.inputs, node.attrs["offsets"]))
        )
