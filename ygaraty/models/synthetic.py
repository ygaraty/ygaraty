"""Shape-only MLP definition and reproducible bounded block initialization."""

import numpy as np

from ygaraty.compiler.shapes import ShapeTape
from ygaraty.compiler.shapes import regions as regions

from .initialization import initial_block as initial_block


def validate_config(config):
    for key in ("batch", "input_dim", "hidden_dim", "classes", "tile_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")


def build_model(config):
    validate_config(config)
    if config.get("model") == "lstm":
        tape, labels = build_lstm(config)
        if config.get("optimizer") == "adam":
            from ygaraty.compiler.optimizer import adam_options

            adam_options(config)  # Validate before writing any blocks.
            for p in tape.parameters:
                for moment in ("m", "v"):
                    tape.declare(
                        f"adam_{moment}_{p}",
                        tape.shapes[p],
                        dict(kind="zeros"),
                    )
        return tape, labels
    tape = ShapeTape()
    loss, labels = SyntheticMLPGraph(tape, config).training_loss()
    tape.backward(loss)
    return tape, labels


class SyntheticMLPGraph:
    """Shape-only two-layer classifier; forward construction has no optimizer
    state."""

    def __init__(self, tape, config):
        self.tape, self.config = tape, dict(config)

    def training_loss(self):
        batch, d, h, c = (
            self.config[k]
            for k in ("batch", "input_dim", "hidden_dim", "classes")
        )
        t = self.tape
        x = t.declare("x", (batch, d), dict(kind="normal", scale=1.0))
        w1 = t.declare(
            "W1", (d, h), dict(kind="normal", scale=np.sqrt(2 / d)), True
        )
        b1 = t.declare("b1", (1, h), dict(kind="zeros"), True)
        w2 = t.declare(
            "W2", (h, c), dict(kind="normal", scale=np.sqrt(1 / h)), True
        )
        b2 = t.declare("b2", (1, c), dict(kind="zeros"), True)
        hidden = t.relu(
            t.add_bias(t.matmul(x, w1, name="hidden_product"), b1),
            name="hidden",
        )
        scores = t.add_bias(
            t.matmul(hidden, w2, name="output_product"), b2, name="scores"
        )
        labels = (
            np.random.default_rng(self.config["seed"] + 1)
            .integers(0, c, size=batch)
            .tolist()
        )
        return t.cross_entropy(scores, labels, name="loss"), labels


class SyntheticLSTMGraph:
    """Unrolled recurrent classifier with explicit inputs, state and output
    head."""

    def __init__(self, tape, config):
        self.tape, self.config = tape, dict(config)
        self.batch, self.width, self.hidden, self.classes = (
            config[k] for k in ("batch", "input_dim", "hidden_dim", "classes")
        )
        self.carry = {}

    def _parameters(self):
        t, d, h, c = self.tape, self.width, self.hidden, self.classes
        self.wx = t.declare(
            "Wx", (d, 4 * h), dict(kind="normal", scale=np.sqrt(1 / d)), True
        )
        self.wh = t.declare(
            "Wh", (h, 4 * h), dict(kind="normal", scale=np.sqrt(1 / h)), True
        )
        self.bias = t.declare(
            "bias", (1, 4 * h), dict(kind="forget_bias", hidden=h), True
        )
        self.wy = t.declare(
            "Wy", (h, c), dict(kind="normal", scale=np.sqrt(1 / h)), True
        )
        self.by = t.declare("by", (1, c), dict(kind="zeros"), True)

    def _step(self, x, hidden, cell, step):
        t, h = self.tape, self.hidden
        z = t.add_bias(
            t.add(t.matmul(x, self.wx), t.matmul(hidden, self.wh)), self.bias
        )
        f, i, o = [t.sigmoid(t.slice_columns(z, k * h, h)) for k in range(3)]
        g = t.tanh(t.slice_columns(z, 3 * h, h))
        cell = t.add(t.mul(f, cell), t.mul(i, g), name=f"cell{step}")
        hidden = t.mul(o, t.tanh(cell), name=f"hidden{step}")
        return hidden, cell

    def training_loss(self, window=None):
        """Optional (tokens, targets, reset) is supplied by the data
        adapter."""
        steps = self.config["window"]
        if type(steps) is not int or not 1 <= steps <= 32:
            raise ValueError("window must be an integer in [1, 32]")
        t, b, d, h, c = (
            self.tape,
            self.batch,
            self.width,
            self.hidden,
            self.classes,
        )
        self._parameters()
        hidden = t.declare("h0", (b, h), dict(kind="zeros"))
        cell = t.declare("c0", (b, h), dict(kind="zeros"))
        if window is not None:
            tokens, targets, reset = window
            table = t.declare(
                "embedding", (c, d), dict(kind="normal", scale=0.1), True
            )
            hidden = t.scale(hidden, 0.0 if reset else 1.0)
            cell = t.scale(cell, 0.0 if reset else 1.0)
        rng = np.random.default_rng(self.config["seed"] + 1)
        labels, losses = [], []
        for step in range(steps):
            x = (
                t.embedding(table, tokens[step], name=f"x{step}")
                if window is not None
                else t.declare(
                    f"x{step}", (b, d), dict(kind="normal", scale=1.0)
                )
            )
            hidden, cell = self._step(x, hidden, cell, step)
            scores = t.add_bias(
                t.matmul(hidden, self.wy), self.by, name=f"scores{step}"
            )
            labels.append(
                targets[step]
                if window is not None
                else rng.integers(0, c, size=b).tolist()
            )
            losses.append(t.cross_entropy(scores, labels[-1]))
        loss = losses[0]
        for part in losses[1:]:
            loss = t.add(loss, part)
        self.carry = {"h0": hidden, "c0": cell}
        return t.scale(loss, 1 / steps, name="loss"), labels


def build_lstm(config):
    steps = config["window"]
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError("window must be an integer in [1, 32]")
    tape = ShapeTape()
    model = SyntheticLSTMGraph(tape, config)
    window = None
    if config.get("language_model", False):
        from ygaraty.datasets.language import token_window

        window = token_window(config)
    loss, labels = model.training_loss(window)
    tape.backward(loss)
    tape.carry = model.carry
    return tape, labels
