"""Metadata-only view resolution and dead-task removal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .tasks import Input, Task

if TYPE_CHECKING:
    from .tape import Tape


class ViewResolver:
    """Index aliases and concatenations once, then rewrite consumer
    regions.
    """

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.alias, self.concats = {}, {}
        alias, concats = self.alias, self.concats
        for node in tape.nodes:
            if node.op == "view":
                base = node.inputs[0]
                r0, _, c0, _ = node.attrs["region"]
                if base in alias:
                    base, (br, bc) = alias[base][0], alias[base][1]
                    r0, c0 = r0 + br, c0 + bc
                alias[node.id] = (base, (r0, c0))
            elif node.op == "concat_rows":
                start, parts = 0, []
                for part in node.inputs:
                    parts.append((part, start))
                    start += tape.shapes[part][0]
                concats[node.id] = parts

    def resolve(self, value: str, region: list[int] | None):
        alias, concats, tape = self.alias, self.concats, self.tape
        while True:
            if value in alias:
                base, (r0, c0) = alias[value]
                rows, cols = tape.shapes[value]
                i, ei, j, ej = region or [0, rows, 0, cols]
                value, region = base, [i + r0, ei + r0, j + c0, ej + c0]
                continue
            if value in concats and region is not None:
                for part, start in concats[value]:
                    if (
                        start <= region[0]
                        and region[1] <= start + tape.shapes[part][0]
                    ):
                        value, region = (
                            part,
                            [
                                region[0] - start,
                                region[1] - start,
                                region[2],
                                region[3],
                            ],
                        )
                        break
                else:
                    return value, region
                continue
            return value, region

    def rewrite(self, tasks: list[Task]) -> list[Task]:
        """Mutate inputs in place, retaining whole-value assembly reads."""
        if not self.alias and not self.concats:
            return tasks
        for task in tasks:
            if task.op in ("assemble", "transpose"):
                continue
            inputs = []
            for x in task.inputs:
                value, region = self.resolve(x.value, x.region)
                inputs.append(
                    x
                    if value == x.value and region == x.region
                    else Input(value, region)
                )
            task.inputs = inputs
        return tasks


def resolve_views(tape: Tape, tasks: list[Task]) -> list[Task]:
    """Rewrite views and contained concatenation regions in task inputs."""
    return ViewResolver(tape).rewrite(tasks)


def prune(tasks: list[Task], keep: set[str]) -> list[Task]:
    """Drop tasks whose outputs nobody reads and that are not kept (gradients,
    the loss): the copies behind a concatenation every reader resolved
    past, for instance.
    """
    while True:
        used = {d for t in tasks for d in t.dependencies}
        alive = [
            t for t in tasks if t.id in used or t.id in keep or "/" not in t.id
        ]
        if len(alive) == len(tasks):
            return tasks
        tasks = alive
