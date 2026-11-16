"""Read one bounded next-byte window per batch lane; never load the corpus."""

from pathlib import Path


def token_window(config):
    batch, width, vocab = config["batch"], config["window"], config["classes"]
    index = config.get("window_index", 0)
    if type(index) is not int or index < 0:
        raise ValueError("window_index must be a nonnegative integer")
    if config.get("corpus"):
        if vocab != 256:
            raise ValueError("Byte corpora require classes=256")
        path = Path(config["corpus"])
        lane = path.stat().st_size // batch
        windows = (lane - 1) // width
        if windows < 1:
            raise ValueError("Corpus needs at least batch*(window+1) bytes")
        offset = (index % windows) * width
        with path.open("rb") as source:
            rows = []
            for row in range(batch):
                source.seek(row * lane + offset)
                data = source.read(width + 1)
                if len(data) != width + 1:
                    raise ValueError("Corpus changed while reading")
                rows.append(list(data))
        reset = index % windows == 0
    else:
        # A deterministic next-token task for tests and capacity measurements.
        rows = [
            [(index * width + t + row) % vocab for t in range(width + 1)]
            for row in range(batch)
        ]
        reset = index == 0
    tokens = [[row[t] for row in rows] for t in range(width)]
    labels = [[row[t + 1] for row in rows] for t in range(width)]
    return tokens, labels, reset
