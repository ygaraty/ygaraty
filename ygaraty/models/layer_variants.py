"""Reference-model variants for M-7 (parallel layers, layer dropping) and WS-6
(layer gradients on stale weights, with or without a delay correction).

VariantTranslator runs the encoder and decoder layer by layer, as
nn.TransformerEncoder/Decoder do, so with no parallel layers and nothing
dropped it computes what TransformerTranslator does. Parallel layers (a run of
consecutive layer indexes) read the same input and add their residual updates:
x + sum(layer(x) - x). A dropped layer is skipped (its input passes through);
the first layer per side is never dropped, and nothing is dropped in eval mode.
"""

from collections import deque

import numpy as np

LAYER_PREFIXES = ("encoder.layers.", "decoder.layers.")


def parse_parallel(text, layers):
    """Parse 1-based inclusive ranges, e.g. enc:2-3 becomes {1, 2}."""
    out = {"enc": set(), "dec": set()}
    if not text:
        return out
    for part in text.split(","):
        side, span = part.split(":")
        if side not in out:
            raise ValueError(f"Unknown side {side!r}: use enc or dec")
        first, last = (int(v) for v in span.split("-"))
        if not 1 <= first < last <= layers:
            raise ValueError(
                f"Parallel span {span} needs 1 <= first < last <= {layers}"
            )
        out[side] |= set(range(first - 1, last))
    return out


def groups(n, parallel):
    """The layer indexes of one side as hops: runs of parallel layers are one
    group, every other layer its own."""
    out, i = [], 0
    while i < n:
        if i in parallel:
            run = [i]
            while i + 1 < n and i + 1 in parallel:
                i += 1
                run.append(i)
            out.append(run)
        else:
            out.append([i])
        i += 1
    return out


def layer_hops(layers, parallel, drop=None):
    """Layer hops on an update's chain: forward and backward through every
    live group of both sides."""
    drop = drop or {"enc": set(), "dec": set()}
    total = 0
    for side in ("enc", "dec"):
        for group in groups(layers, parallel[side]):
            if any(i not in drop[side] for i in group):
                total += 2
    return total


def drop_rate(schedule, p, seen, pairs):
    """Layer-drop probability after `seen` of `pairs` pairs: constant p, or
    progressive, rising linearly from 0 to p."""
    return p if schedule == "constant" else p * min(1.0, seen / max(1, pairs))


def draw_drop(rng, layers, rate):
    """Layers to skip this update, per side; never the first."""
    return {
        side: {i for i in range(1, layers) if rng.random() < rate}
        for side in ("enc", "dec")
    }


def variant_translator(parallel, **config):
    import torch
    from torch import nn
    from torch.nn.modules.transformer import _detect_is_causal_mask

    from ygaraty.reference.transformer import PAD, TransformerTranslator

    class VariantTranslator(TransformerTranslator):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.parallel = parallel
            self.drop = {"enc": set(), "dec": set()}

        def stack(self, side, layers, x, **kwargs):
            drop = self.drop[side] if self.training else set()
            for group in groups(len(layers), self.parallel[side]):
                live = [i for i in group if i not in drop]
                if not live:
                    continue
                if len(group) == 1:
                    x = layers[live[0]](x, **kwargs)
                else:
                    x = x + sum(layers[i](x, **kwargs) - x for i in live)
            return x

        def encode(self, source):
            torch.backends.mha.set_fastpath_enabled(False)
            pad = source.eq(PAD)
            x = self.stack(
                "enc",
                self.encoder.layers,
                self.embed(source),
                src_key_padding_mask=pad,
            )
            return self.encoder.norm(x)

        def decode(self, memory, source, target_input, ids=None, buckets=None):
            causal = torch.triu(
                torch.ones(
                    target_input.size(1),
                    target_input.size(1),
                    dtype=torch.bool,
                ),
                1,
            )
            x = self.stack(
                "dec",
                self.decoder.layers,
                self.embed(target_input),
                memory=memory,
                tgt_mask=causal,
                tgt_key_padding_mask=target_input.eq(PAD),
                memory_key_padding_mask=source.eq(PAD),
                tgt_is_causal=_detect_is_causal_mask(
                    causal, None, target_input.size(1)
                ),
            )
            return nn.functional.linear(
                self.decoder.norm(x), self.embedding.weight, self.output_bias
            )

        def forward(self, source, *args):
            return self.decode(self.encode(source), source, args[-1])

    return VariantTranslator(**config)


class StaleLayers:
    """WS-6's learning screen: the encoder and decoder layers' gradients are
    computed on their weights from k updates earlier, as a k-deep asynchronous
    pipeline would; the optimizer applies them to the current weights.

    correction="nesterov" (our reading of a Nesterov-style delay correction):
    the stale weights are extrapolated along their own last update, w(t-k) +
    k (w(t-k) - w(t-k-1)), a look-ahead toward the weights the gradient will be
    applied to."""

    def __init__(self, model, k, correction="none"):
        import torch

        self.torch, self.k, self.correction = torch, k, correction
        self.params = [
            p
            for n, p in model.named_parameters()
            if n.startswith(LAYER_PREFIXES)
        ]
        self.history = deque([self.snapshot()], maxlen=k + 2)
        self.current = None

    def snapshot(self):
        return [p.detach().clone() for p in self.params]

    def before_forward(self):
        """Load the weights this update's gradient is computed on."""
        if self.k == 0:
            return
        self.current = self.snapshot()
        h = list(self.history)
        # history ends with the current weights: index -1 is w(t), -1-k is
        # w(t-k).
        stale = h[max(0, len(h) - 1 - self.k)]
        older = h[max(0, len(h) - 2 - self.k)]
        with self.torch.no_grad():
            for p, s, o in zip(self.params, stale, older):
                p.copy_(
                    s + self.k * (s - o)
                    if self.correction == "nesterov"
                    else s
                )

    def after_backward(self):
        """Put the current weights back; the gradients stay."""
        if self.k == 0:
            return
        with self.torch.no_grad():
            for p, c in zip(self.params, self.current):
                p.copy_(c)

    def after_step(self):
        self.history.append(self.snapshot())


def check_seed(seed, update):
    """The per-update layer-drop generator, shared by every run of a seed."""
    return np.random.default_rng([seed, 707, update])
