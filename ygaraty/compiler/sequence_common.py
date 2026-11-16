"""Shared sequence iteration contracts; no graph or execution state."""


def spans(size: int, tile: int) -> list[tuple[int, int]]:
    return [(start, min(start + tile, size)) for start in range(0, size, tile)]


def order(steps: int, reverse: bool) -> range:
    return range(steps - 1, -1, -1) if reverse else range(steps)


# Preserve owner checkpoints exactly across reloads and re-placement.
# Preserve owner checkpoints exactly across reloads and re-placement.
PUBLISH_WIRE = "<f8"
