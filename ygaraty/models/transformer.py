"""Primitive pre-norm encoder/decoder Transformer graph construction.

Batch items and attention heads are explicit branches; embedding/output weights
are shared. This builder records operations and never executes model kernels.
"""

import math

import numpy as np

from ygaraty.compiler.parameters import Parameters
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.compiler.tape import Tape


class TransformerGraph:
    """One model bound to one tape, composed from named Transformer blocks.

    Construct a fresh instance for each training graph. Parameter transposes
    and
    Q/K/V slices are reused within that graph. Distribution and task budgets
    are
    concerns of the compiler's lowering stage, not this model definition.
    """

    def __init__(self, tape, config, state):
        self.tape = tape
        self.parameters = Parameters(tape, state)
        self.width = config["embedding"]
        self.heads = config["heads"]
        self.layers = config["layers"]
        self.max_length = config["max_length"]
        if self.width % self.heads:
            raise ValueError("Embedding width must divide into heads")
        self._projections = {}

    def embedding(self, ids):
        t, p = self.tape, self.parameters
        return t.add(
            t.scale(
                t.embedding(p["embedding.weight"], ids, padding_idx=0),
                math.sqrt(self.width),
            ),
            t.embedding(p["position.weight"], np.arange(len(ids))),
        )

    def _project(self, value, prefix, index):
        key = (prefix, index)
        if key not in self._projections:
            t, p = self.tape, self.parameters
            self._projections[key] = (
                t.slice_columns(
                    p.transpose(prefix + ".in_proj_weight"),
                    index * self.width,
                    self.width,
                ),
                t.slice_columns(
                    p[prefix + ".in_proj_bias"], index * self.width, self.width
                ),
            )
        weight, bias = self._projections[key]
        return self.tape.add_bias(self.tape.matmul(value, weight), bias)

    def attention(self, query, key, value, prefix, mask):
        t = self.tape
        projections = [
            self._project(x, prefix, i)
            for i, x in enumerate((query, key, value))
        ]
        width = self.width // self.heads
        output = []
        for head in range(self.heads):
            q, k, v = [
                t.slice_columns(x, head * width, width) for x in projections
            ]
            scores = t.scale(t.matmul(q, t.transpose(k)), 1 / math.sqrt(width))
            output.append(t.matmul(t.softmax(scores, mask), v))
        return self.parameters.linear(
            t.concat_columns(output), prefix + ".out_proj"
        )

    def feed_forward(self, value, prefix):
        p = self.parameters
        return p.linear(
            self.tape.relu(p.linear(value, prefix + ".linear1")),
            prefix + ".linear2",
        )

    def encode(self, source):
        t, p = self.tape, self.parameters
        value = self.embedding(source)
        mask = np.broadcast_to(source == 0, (len(source), len(source)))
        for layer in range(self.layers):
            prefix = f"encoder.layers.{layer}"
            normalized = p.layer_norm(value, prefix + ".norm1")
            value = t.add(
                value,
                self.attention(
                    normalized,
                    normalized,
                    normalized,
                    prefix + ".self_attn",
                    mask,
                ),
            )
            value = t.add(
                value,
                self.feed_forward(
                    p.layer_norm(value, prefix + ".norm2"), prefix
                ),
            )
        return p.layer_norm(value, "encoder.norm")

    def decode(self, inputs, memory, source):
        t, p = self.tape, self.parameters
        value = self.embedding(inputs)
        mask = (
            np.triu(np.ones((len(inputs), len(inputs)), dtype=bool), 1)
            | (inputs == 0)[None, :]
        )
        cross_mask = np.broadcast_to(source == 0, (len(inputs), len(source)))
        for layer in range(self.layers):
            prefix = f"decoder.layers.{layer}"
            normalized = p.layer_norm(value, prefix + ".norm1")
            value = t.add(
                value,
                self.attention(
                    normalized,
                    normalized,
                    normalized,
                    prefix + ".self_attn",
                    mask,
                ),
            )
            value = t.add(
                value,
                self.attention(
                    p.layer_norm(value, prefix + ".norm2"),
                    memory,
                    memory,
                    prefix + ".multihead_attn",
                    cross_mask,
                ),
            )
            value = t.add(
                value,
                self.feed_forward(
                    p.layer_norm(value, prefix + ".norm3"), prefix
                ),
            )
        return p.layer_norm(value, "decoder.norm")

    def logits(self, source, inputs):
        memory = self.encode(source)
        decoded = self.decode(inputs, memory, source)
        return self.tape.add_bias(
            self.tape.matmul(
                decoded, self.parameters.transpose("embedding.weight")
            ),
            self.parameters["output_bias"],
        )

    def training_loss(self, source, target_input, targets, smoothing=0.1):
        """Record token-weighted loss, leaving differentiation to the
        caller."""
        if (
            source.ndim != 2
            or target_input.shape != targets.shape
            or len(source) != len(target_input)
        ):
            raise ValueError(
                "Expected padded source, target input and target labels"
            )
        if max(source.shape[1], target_input.shape[1]) > self.max_length:
            raise ValueError("Sequence exceeds positional capacity")
        total_tokens = int(np.count_nonzero(targets))
        if not total_tokens:
            raise ValueError("Minibatch has no target tokens")
        losses, logits = [], []
        for src, inputs, labels in zip(source, target_input, targets):
            scores = self.logits(src, inputs)
            logits.append(scores)
            loss = self.tape.cross_entropy(
                scores, labels, ignore_index=0, label_smoothing=smoothing
            )
            losses.append(
                self.tape.scale(loss, np.count_nonzero(labels) / total_tokens)
            )
        loss = losses[0]
        for other in losses[1:]:
            loss = self.tape.add(loss, other)
        return loss, logits


def build_transformer(
    config,
    state,
    source,
    target_input,
    targets,
    smoothing=0.1,
    *,
    shape_only=False,
):
    """Compatibility recipe: construct, differentiate, return (tape, logits,
    params).

    Shape-only state contains parameter specs; numeric state contains arrays.
    Use TransformerGraph directly to construct a forward graph before
    backward().
    """
    tape = ShapeTape() if shape_only else Tape()
    model = TransformerGraph(tape, config, state)
    loss, logits = model.training_loss(
        source, target_input, targets, smoothing
    )
    tape.backward(loss)
    return tape, logits, dict(model.parameters)
