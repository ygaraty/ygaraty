"""Task and input-region data contracts shared by compiler passes."""

from dataclasses import asdict, dataclass, field


@dataclass
class Input:
    value: str
    # [row_start, row_stop, column_start, column_stop], or the full tensor.
    region: list[int] | None = None


@dataclass
class Task:
    id: str
    op: str
    inputs: list[Input]
    shape: tuple[int, int]
    attrs: dict = field(default_factory=dict)

    @property
    def dependencies(self):
        return set(x.value for x in self.inputs)

    def arrays(self, store):
        result = []
        for spec in self.inputs:
            a = store[spec.value]
            if spec.region:
                i, ei, j, ej = spec.region
                a = a[i:ei, j:ej]
            result.append(a)
        return result

    def describe(self):
        return asdict(self)
