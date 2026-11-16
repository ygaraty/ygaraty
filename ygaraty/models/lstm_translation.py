"""Attention encoder-decoder LSTM, lowered as one batched, time-major graph.

Matches the pilot's stacked bidirectional encoder, learned h/c bridges, stacked
unidirectional decoder and tied embedding/output projection. Float64, dropout
disabled. The whole minibatch runs in lockstep: row t*batch+b is step t of
example b. Each layer direction is one `lstm` node (one fused task per step
after
planning); padding is masked so sources keep packed-sequence semantics: the
state
carries over padded steps, and padded encoder outputs are zero. The decoder
runs
over every target position, as PyTorch does over the padded teacher-forced
input.
"""

import math

import numpy as np

from ygaraty.compiler.layouts import C_STATE, H_OUT, H_STATE
from ygaraty.compiler.parameters import Parameters
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.compiler.tape import Tape


class Batched:
    """A time-major tensor (optionally a column range of a stored value), per
    example."""

    def __init__(self, value, batch, columns=None):
        self.value, self.batch, self.columns = value, batch, columns

    def examples(self, values):
        a = np.asarray(values[self.value])
        if self.columns:
            a = a[:, self.columns[0] : self.columns[1]]
        return a.reshape(len(a) // self.batch, self.batch, -1).transpose(
            1, 0, 2
        )

    def time_major(self, per_example):
        """(batch, steps, n) -> the (steps*batch, n) layout of `value`."""
        per_example = np.asarray(per_example)
        return per_example.transpose(1, 0, 2).reshape(-1, per_example.shape[2])

    reorder = time_major


def _validate_batch(config, source, target_input, targets):
    source, target_input, targets = map(
        np.asarray, (source, target_input, targets)
    )
    vocab, layers = config["vocab"], config["layers"]
    if (
        source.ndim != 2
        or target_input.ndim != 2
        or targets.shape != target_input.shape
        or len(source) != len(target_input)
        or not source.size
        or not target_input.size
    ):
        raise ValueError(
            "Expected nonempty padded source/input/target matrices "
            "with matching batches"
        )
    for ids in (source, target_input, targets):
        if (
            ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= vocab)
        ):
            raise ValueError(
                "Token IDs must be integers within the vocabulary"
            )
    lengths = np.count_nonzero(source, axis=1)
    if np.any(lengths == 0) or any(
        np.any(row[:n] == 0) or np.any(row[n:] != 0)
        for row, n in zip(source, lengths)
    ):
        raise ValueError(
            "Sources require nonempty tokens followed only by right padding"
        )
    if np.any(np.count_nonzero(targets, axis=1) == 0):
        raise ValueError("Each example must have a supervised target token")
    if layers < 1:
        raise ValueError("At least one LSTM layer is required")
    return source, target_input, targets, lengths


class RecurrentStateLayout:
    """Views into packed time-major hidden, cell and output states."""

    def __init__(self, tape, batch, hidden):
        self.tape, self.batch, self.hidden = tape, batch, hidden

    def columns(self, value, block):
        return self.tape.view(
            value,
            [
                0,
                self.tape.shapes[value][0],
                block * self.hidden,
                (block + 1) * self.hidden,
            ],
        )

    def final(self, value, step, block):
        return self.tape.view(
            value,
            [
                step * self.batch,
                (step + 1) * self.batch,
                block * self.hidden,
                (block + 1) * self.hidden,
            ],
        )


class LSTMTranslationGraph:
    """Compose recurrent layers, bridges and attention; record a forward loss
    only."""

    def __init__(self, tape, config, state):
        self.tape, self.config = tape, dict(config)
        self.parameters = Parameters(tape, state)
        self.hidden, self.layers = config["hidden"], config["layers"]
        self.diagnostics = {}

    def _constant(self, array):
        array = np.asarray(array, dtype=np.float64)
        if isinstance(self.tape, ShapeTape):
            return self.tape.declare(
                None, array.shape, dict(kind="constant", data=array.tolist())
            )
        return self.tape.input(None, array)

    def _zeros(self, shape):
        if isinstance(self.tape, ShapeTape):
            return self.tape.declare(None, shape, dict(kind="zeros"))
        return self.tape.input(None, np.zeros(shape))

    def _layer(
        self, value, prefix, index, mask, initial, layout, reverse=False
    ):
        t, p = self.tape, self.parameters
        suffix = f"_l{index}" + ("_reverse" if reverse else "")
        projected = t.matmul(
            value, p.transpose(prefix + ".weight_ih" + suffix)
        )
        hidden, cell = initial
        return t.lstm(
            projected,
            p.transpose(prefix + ".weight_hh" + suffix),
            p[prefix + ".bias_ih" + suffix],
            p[prefix + ".bias_hh" + suffix],
            mask,
            hidden,
            cell,
            layout.batch,
            reverse,
        )

    def _bridge(self, forward, backward, steps, layout):
        # Forward finishes at the last step; reverse finishes at zero. Masked
        # states have already carried over from each example's last real token.
        def project(block, prefix):
            states = self.tape.concat_columns(
                [
                    layout.final(forward, steps - 1, block),
                    layout.final(backward, 0, block),
                ]
            )
            return self.tape.tanh(self.parameters.linear(states, prefix))

        return project(H_STATE, "bridge_h"), project(C_STATE, "bridge_c")

    def encode(self, source, lengths, layout):
        steps = source.shape[1]
        mask = self._constant(
            (np.arange(steps)[:, None] < lengths[None, :]).reshape(-1, 1)
        )
        value = self.tape.embedding(
            self.parameters["embedding.weight"],
            source.T.reshape(-1),
            padding_idx=0,
        )
        initial = []
        shape = (layout.batch, self.hidden)
        for index in range(self.layers):
            forward = self._layer(
                value,
                "encoder",
                index,
                mask,
                (self._zeros(shape), self._zeros(shape)),
                layout,
            )
            backward = self._layer(
                value,
                "encoder",
                index,
                mask,
                (self._zeros(shape), self._zeros(shape)),
                layout,
                reverse=True,
            )
            value = self.tape.concat_columns(
                [
                    layout.columns(forward, H_OUT),
                    layout.columns(backward, H_OUT),
                ]
            )
            initial.append(self._bridge(forward, backward, steps, layout))
        return value, initial

    def decode(self, target_input, initial, layout):
        mask = self._constant(
            np.ones((target_input.shape[1] * layout.batch, 1))
        )
        value = self.tape.embedding(
            self.parameters["embedding.weight"],
            target_input.T.reshape(-1),
            padding_idx=0,
        )
        for index in range(self.layers):
            top = self._layer(
                value, "decoder", index, mask, initial[index], layout
            )
            value = layout.columns(top, H_OUT)
        return value, top

    def attention(self, decoded, encoded, keys, source, target_steps):
        t = self.tape
        batch, source_steps = source.shape
        scores = t.scale(
            t.bqk(decoded, keys, batch), 1 / math.sqrt(self.hidden)
        )
        mask = np.broadcast_to(
            (source == 0)[None], (target_steps, batch, source_steps)
        ).reshape(-1, source_steps)
        attention = t.softmax(scores, mask)
        return t.bpv(attention, encoded, batch), attention

    def output(self, decoded, context):
        t, p = self.tape, self.parameters
        features = t.tanh(
            p.linear(t.concat_columns([decoded, context]), "combine")
        )
        return t.add_bias(
            t.matmul(features, p.transpose("embedding.weight")),
            p["output_bias"],
        )

    def training_loss(self, source, target_input, targets, smoothing=0.0):
        source, target_input, targets, lengths = _validate_batch(
            self.config, source, target_input, targets
        )
        batch = len(source)
        layout = RecurrentStateLayout(self.tape, batch, self.hidden)
        encoded, initial = self.encode(source, lengths, layout)
        keys = self.tape.matmul(
            encoded, self.parameters.transpose("keys.weight")
        )
        decoded, top = self.decode(target_input, initial, layout)
        context, attention = self.attention(
            decoded, encoded, keys, source, target_input.shape[1]
        )
        logits = self.output(decoded, context)
        loss = self.tape.cross_entropy(
            logits,
            targets.T.reshape(-1).tolist(),
            ignore_index=0,
            label_smoothing=smoothing,
        )
        self.diagnostics = dict(
            encoder=Batched(encoded, batch),
            attention=Batched(attention, batch),
            decoder=Batched(
                top, batch, [H_OUT * self.hidden, (H_OUT + 1) * self.hidden]
            ),
        )
        return loss, Batched(logits, batch)


def build_lstm_translation(
    config,
    state,
    source,
    target_input,
    targets,
    smoothing=0.0,
    *,
    shape_only=False,
):
    """Compatibility recipe: differentiate and attach time-major
    diagnostics."""
    source, target_input, targets, _ = _validate_batch(
        config, source, target_input, targets
    )
    tape = ShapeTape() if shape_only else Tape()
    model = LSTMTranslationGraph(tape, config, state)
    loss, logits = model.training_loss(
        source, target_input, targets, smoothing
    )
    tape.backward(loss)
    tape.translation_diagnostics = model.diagnostics
    return tape, logits, dict(model.parameters)
