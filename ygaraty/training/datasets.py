"""Verified IDX datasets and replayable, disjoint per-worker sample streams."""

import gzip
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .convnet_webgpu import ConvBatch

# Published upstream checksums (torchvision MNIST resources; Fashion-MNIST
# README). Downloads are HTTPS; run identity additionally records SHA-256.
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
SOURCES = {
    "mnist": (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        (
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            "d53e105ee54ea40749a09fcbcd1e9432",
            "9fb629c4189551a2d022fa330f9573f3",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ),
    "fashion-mnist": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        (
            "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "25c81989df183df01b3e8a0aad5dffbe",
            "bef4ecab320f06d8554ea6380940ec79",
            "bb300cfdad3c16e7a12a480ee83cd310",
        ),
    ),
}


def read_idx(raw, *, images):
    """Reject truncation, trailing bytes and unexpected geometry before reshape."""
    header = 16 if images else 8
    if len(raw) < header:
        raise ValueError("Truncated IDX header")
    magic, rows = struct.unpack_from(">II", raw)
    if magic != (2051 if images else 2049) or not 1 <= rows <= 60000:
        raise ValueError("Invalid IDX header")
    if images and struct.unpack_from(">II", raw, 8) != (28, 28):
        raise ValueError("Expected 28x28 images")
    shape = (rows, 1, 28, 28) if images else (rows,)
    if len(raw) != header + int(np.prod(shape)):
        raise ValueError("Invalid IDX payload length")
    return (
        np.frombuffer(raw, dtype=np.uint8, offset=header).reshape(shape).copy()
    )


@dataclass
class ImageDataset:
    name: str
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray

    def __post_init__(self):
        digest = hashlib.sha256(self.name.encode())
        for split in ("train", "test"):
            images, labels = (
                getattr(self, split + "_images"),
                getattr(self, split + "_labels"),
            )
            if (
                images.dtype != np.uint8
                or images.shape != (len(images), 1, 28, 28)
                or not len(images)
                or labels.dtype != np.uint8
                or labels.shape != (len(images),)
                or np.any(labels > 9)
            ):
                raise ValueError("Expected uint8 28x28 images and labels 0..9")
            for value in (images, labels):
                digest.update(str(value.shape).encode())
                digest.update(value.tobytes())
                value.setflags(write=False)
        self.sha256 = digest.hexdigest()

    def indices(self, worker, workers, cursor, size, seed):
        """Fixed disjoint shards, independently shuffled each epoch.

        The cursor counts samples consumed by this logical worker. Seed, shard
        and epoch reconstruct RNG state, including batches crossing epochs.
        """
        if (
            not 0 <= worker < workers <= len(self.train_images)
            or cursor < 0
            or size < 1
        ):
            raise ValueError("Invalid dataset cursor or membership")
        shard = np.arange(worker, len(self.train_images), workers)
        result = []
        while len(result) < size:
            epoch, offset = divmod(cursor, len(shard))
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, worker, epoch])
            )
            order = rng.permutation(shard)
            take = min(size - len(result), len(shard) - offset)
            result.extend(order[offset : offset + take])
            cursor += take
        return np.asarray(result, dtype=np.int64)

    def batches(self, worker, workers, step, steps, batch_size, seed):
        for offset in range(steps):
            ids = self.indices(
                worker, workers, (step + offset) * batch_size, batch_size, seed
            )
            yield ConvBatch(self.train_images[ids], self.train_labels[ids])

    def evaluate(self, recipe, weights, limit):
        count = min(limit, len(self.test_images))
        correct, loss = 0, 0.0
        for start in range(0, count, recipe.max_batch):
            end = min(start + recipe.max_batch, count)
            logits = recipe.predict(weights, self.test_images[start:end])
            labels = self.test_labels[start:end]
            stable = logits - logits.max(axis=1, keepdims=True)
            loss += float(
                (
                    np.log(np.exp(stable).sum(axis=1))
                    - stable[np.arange(end - start), labels]
                ).sum()
            )
            correct += int((logits.argmax(axis=1) == labels).sum())
        return dict(
            split="test",
            examples=count,
            loss=loss / count,
            accuracy=correct / count,
        )


def load_dataset(name, cache):
    base, checksums = SOURCES[name]
    directory = Path(cache) / name
    directory.mkdir(parents=True, exist_ok=True)
    arrays = []
    for i, (filename, expected) in enumerate(zip(FILES, checksums)):
        path = directory / filename
        if path.exists():
            data = path.read_bytes()
        else:
            with urllib.request.urlopen(
                base + filename, timeout=60
            ) as response:
                data = response.read(32 * 1024 * 1024 + 1)
        if (
            len(data) > 32 * 1024 * 1024
            or hashlib.md5(data, usedforsecurity=False).hexdigest() != expected
        ):
            raise ValueError(
                f"Dataset checksum mismatch: {path}; remove it and retry"
            )
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        arrays.append(read_idx(gzip.decompress(data), images=i % 2 == 0))
    return ImageDataset(name, *arrays)
