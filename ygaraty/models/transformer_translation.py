"""Batched pre-norm encoder-decoder Transformer for SQL-fed training: fused
layer tasks.

Rows are example-major (row b*steps + t). Each layer runs as one `tf_layer`
task per band of
examples (ygaraty.runtime.transformer_kernels); bands are independent, so
the
critical path
of an update is a few hops per layer whatever the sequence length. The output
layer and the
loss are the same tape operations as the LSTM recipe's, so the fused output
layer applies.
"""

import math

import numpy as np

from ygaraty.compiler.layouts import DECODER_PARAMS, ENCODER_PARAMS
from ygaraty.compiler.parameters import Parameters
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.compiler.tape import Tape


class ExampleMajor:
    """A (batch*steps, n) value whose rows are example-major, per example."""

    def __init__(self, value, batch):
        self.value, self.batch = value, batch

    def examples(self, values):
        a = np.asarray(values[self.value])
        return a.reshape(self.batch, len(a) // self.batch, -1)

    def reorder(self, per_example):
        """(batch, steps, n) -> the layout of `value`: PyTorch's own row
        order."""
        per_example = np.asarray(per_example)
        return per_example.reshape(-1, per_example.shape[2])


def parameter_specs(config):
    """Shape-only parameter layout and initialization (no PyTorch),
    nn.Transformer names."""
    v, d, ffn, layers, maxlen = (
        config[k]
        for k in ("vocab", "embedding", "hidden", "layers", "max_length")
    )
    specs = {}

    def add(name, shape, **init):
        specs[name] = dict(
            shape=(1, shape[0]) if len(shape) == 1 else shape,
            original_shape=shape,
            **init,
        )

    add("embedding.weight", (v, d), kind="normal", scale=d**-0.5, zero_row=0)
    add("position.weight", (maxlen, d), kind="normal", scale=d**-0.5)
    add("output_bias", (v,), kind="zeros")

    def attention(prefix):
        add(
            prefix + ".in_proj_weight",
            (3 * d, d),
            kind="uniform",
            bound=math.sqrt(6 / (4 * d)),
        )  # xavier, as PyTorch
        add(prefix + ".in_proj_bias", (3 * d,), kind="zeros")
        add(
            prefix + ".out_proj.weight",
            (d, d),
            kind="uniform",
            bound=1 / math.sqrt(d),
        )
        add(prefix + ".out_proj.bias", (d,), kind="zeros")

    def norm(prefix):
        add(prefix + ".weight", (d,), kind="ones")
        add(prefix + ".bias", (d,), kind="zeros")

    for stack in ("encoder", "decoder"):
        for layer in range(layers):
            prefix = f"{stack}.layers.{layer}"
            norm(prefix + ".norm1")
            attention(prefix + ".self_attn")
            norm(prefix + ".norm2")
            if stack == "decoder":
                attention(prefix + ".multihead_attn")
                norm(prefix + ".norm3")
            add(
                prefix + ".linear1.weight",
                (ffn, d),
                kind="uniform",
                bound=1 / math.sqrt(d),
            )
            add(
                prefix + ".linear1.bias",
                (ffn,),
                kind="uniform",
                bound=1 / math.sqrt(d),
            )
            add(
                prefix + ".linear2.weight",
                (d, ffn),
                kind="uniform",
                bound=1 / math.sqrt(ffn),
            )
            add(
                prefix + ".linear2.bias",
                (d,),
                kind="uniform",
                bound=1 / math.sqrt(ffn),
            )
        norm(stack + ".norm")
    if config.get("hint_buckets"):
        # N-gram hints (experiment 55, batch_learning.hinted_translator):
        # per-bucket trust scales and
        # embeddings, zero at first, and one gate per injection point, one at
        # first.
        add("hint.trust_scale", (config["hint_buckets"], 1), kind="zeros")
        add("hint.trust_embedding", (config["hint_buckets"], d), kind="zeros")
        add("hint.gate", (3,), kind="ones")
    return specs


def _validate_batch(config, source, target_input, targets):
    source, target_input, targets = map(
        np.asarray, (source, target_input, targets)
    )
    if config["embedding"] % config["heads"]:
        raise ValueError("Embedding width must divide into heads")
    if (
        source.ndim != 2
        or target_input.shape != targets.shape
        or len(source) != len(target_input)
        or not source.size
    ):
        raise ValueError(
            "Expected padded source, target input and target label matrices"
        )
    if max(source.shape[1], target_input.shape[1]) > config["max_length"]:
        raise ValueError("Sequence exceeds positional capacity")
    if np.any(np.count_nonzero(source, axis=1) == 0) or np.any(
        np.count_nonzero(targets, axis=1) == 0
    ):
        raise ValueError(
            "Every example needs a source token and a target token"
        )
    return source, target_input, targets


def constant(tape, name, shape, kind):
    """A ones or zeros leaf named by its shape, once per tape (an update's
    leaves are kept by name)."""
    if name in tape.shapes:
        return name
    if isinstance(tape, ShapeTape):
        return tape.declare(name, shape, dict(kind=kind))
    return tape.input(
        name, np.ones(shape) if kind == "ones" else np.zeros(shape)
    )


class HintVector:
    """Experiment 55's hint vector, as batch_learning.hinted_translator
    computes it: per decoder input row
    h = trust_scale[b] * sqrt(d) * E[hint] + trust_embedding[b] (b its bucket,
    E the tied table; the
    "none" bucket's scale is held at zero), added at three places, each times
    its own gate."""

    def __init__(self, tape, parameters, vocab, width, ids, buckets, none):
        t, p = tape, parameters
        buckets = np.asarray(buckets).reshape(-1)
        ids = np.where(buckets == none, 0, np.asarray(ids).reshape(-1))
        self.tape, self.parameters, self.rows = t, p, len(ids)
        self.ones_width = constant(t, f"ones_1x{width}", (1, width), "ones")
        self.ones_rows = constant(
            t, f"ones_{self.rows}x1", (self.rows, 1), "ones"
        )
        # The none bucket as padding: its scale gets no gradient, so it stays
        # at its initial zero.
        scale = t.embedding(p["hint.trust_scale"], buckets, padding_idx=none)
        token = t.mul(t.matmul(scale, self.ones_width), vocab.scaled_rows(ids))
        self.value = t.add(
            token, t.embedding(p["hint.trust_embedding"], buckets)
        )

    def gated(self, index):
        t = self.tape
        gate = t.matmul(
            t.slice_columns(self.parameters["hint.gate"], index, 1),
            self.ones_width,
        )
        return t.mul(t.matmul(self.ones_rows, gate), self.value)


class ExampleBands:
    """Example-major row boundaries shared by layers, memory and output
    heads."""

    def __init__(self, batch, band=None):
        self.batch = batch
        size = batch if band is None else max(1, min(int(band), batch))
        self.bounds = tuple(
            (start, min(start + size, batch))
            for start in range(0, batch, size)
        )

    def view(self, tape, value, index, steps, width):
        if len(self.bounds) == 1:
            return value
        start, stop = self.bounds[index]
        return tape.view(value, [start * steps, stop * steps, 0, width])


class VocabularyProjection:
    """Tied embedding/output weights, optionally gathered from vocabulary
    owners."""

    def __init__(
        self,
        tape,
        parameters,
        width,
        vocab,
        shards,
        source,
        target_input,
        extra=None,
    ):
        self.tape, self.parameters, self.width = tape, parameters, width
        self.table, self.padding, self.used = (
            parameters["embedding.weight"],
            0,
            None,
        )
        if shards:
            self._gather(vocab, shards, source, target_input, extra)

    def _gather(self, vocab, count, source, target_input, extra=None):
        t, p = self.tape, self.parameters
        shards = [
            [s * vocab // count, (s + 1) * vocab // count]
            for s in range(count)
        ]
        if any(stop <= start for start, stop in shards):
            raise ValueError("More vocabulary shards than vocabulary rows")
        # `extra`: further rows read from the table (hint tokens), gathered
        # with the batch's own.
        self.used = np.unique(
            np.concatenate(
                [source.ravel(), target_input.ravel()]
                + ([np.asarray(extra).ravel()] if extra is not None else [])
            )
        )
        t.owned_vocab = dict(
            params=[p["embedding.weight"], p["output_bias"]],
            shards=shards,
            grads={},
            embeds={},
        )
        gathered = []
        for shard, (start, stop) in enumerate(shards):
            rows = self.used[(self.used >= start) & (self.used < stop)]
            if len(rows):
                gathered.append(
                    t.vocab_gather(
                        self.table,
                        shard,
                        (start, stop),
                        (rows - start).tolist(),
                    )
                )
        self.table = t.concat_rows(gathered)
        t.parameter_like.add(self.table)
        t.owned_vocab["table"] = self.table
        self.padding = (
            int(np.searchsorted(self.used, 0)) if self.used[0] == 0 else None
        )

    def embed(self, ids, steps, batch):
        positions = np.tile(np.arange(steps), batch)
        if self.used is not None:
            ids = np.searchsorted(self.used, ids)
        return self.tape.embed(
            self.table,
            ids,
            self.parameters["position.weight"],
            positions,
            math.sqrt(self.width),
            padding_idx=self.padding,
        )

    def scaled_rows(self, ids):
        """table[ids] * sqrt(width), no position: an `embed` whose position
        table is one zero row, so
        under vocabulary ownership its gradient reaches the owners as the batch
        embeddings' does."""
        zero = constant(
            self.tape, f"zero_row_{self.width}", (1, self.width), "zeros"
        )
        if self.used is not None:
            ids = np.searchsorted(self.used, ids)
        return self.tape.embed(
            self.table,
            ids,
            zero,
            np.zeros(len(ids), dtype=np.int64),
            math.sqrt(self.width),
            padding_idx=self.padding,
        )

    def loss(self, decoded, targets, bands, smoothing):
        t, p = self.tape, self.parameters
        labels = targets.reshape(-1).tolist()
        if self.used is not None:
            views = [
                bands.view(t, decoded, index, targets.shape[1], self.width)
                for index in range(len(bands.bounds))
            ]
            loss = t.vocab_output(
                views,
                p["embedding.weight"],
                p["output_bias"],
                dict(
                    targets=labels, ignore_index=0, label_smoothing=smoothing
                ),
            )
            return loss, None
        logits = t.add_bias(
            t.matmul(decoded, t.transpose(p["embedding.weight"])),
            p["output_bias"],
        )
        loss = t.cross_entropy(
            logits, labels, ignore_index=0, label_smoothing=smoothing
        )
        return loss, ExampleMajor(logits, bands.batch)


class FusedLayerStack:
    """Build fused layer bands, preserving layer ownership and final
    normalization."""

    def __init__(self, tape, parameters, config, bands, owned):
        self.tape, self.parameters, self.bands = tape, parameters, bands
        self.width, self.heads, self.layers = (
            config["embedding"],
            config["heads"],
            config["layers"],
        )
        self.owned = owned
        # Experiment 78: runs of layers that read the same input; the first of
        # a run
        # returns its full output, the others only their residual updates,
        # and the layer after the run adds them to its input in its own task.
        from ygaraty.models.layer_variants import groups, parse_parallel

        parallel = config.get("parallel_layers")
        self.parallel = (
            parse_parallel(parallel, self.layers)
            if isinstance(parallel, str) or parallel is None
            else {k: set(v) for k, v in parallel.items()}
        )
        self.groups = groups

    def build(
        self,
        value,
        prefix,
        names,
        steps,
        key_mask,
        causal,
        memory=None,
        mem_steps=None,
        mem_mask=None,
        inject=None,
    ):
        """`inject`: {layer index: value added to that layer's output}. The
        final norm stays fused into
        the last layer unless something is added after it, then it runs on its
        own."""
        p = self.parameters
        norm = (p[prefix + ".norm.weight"], p[prefix + ".norm.bias"])
        inject, last = inject or {}, self.layers - 1
        fused = last not in inject
        parallel = self.parallel["enc" if prefix == "encoder" else "dec"]
        pending = []  # the residual updates the next layer adds to its input
        for group in self.groups(self.layers, parallel):
            if len(group) > 1 and (last in group or inject):
                raise ValueError(
                    "Parallel layers must end before the last layer of a "
                    "stack (its task applies the final norm), and don't "
                    "combine with hints"
                )
            outputs = []
            for k, layer in enumerate(group):
                tag = f"{prefix}.layers.{layer}"
                parameters = [p[f"{tag}.{name}"] for name in names]
                bands = [
                    self._band(
                        value,
                        index,
                        tag,
                        parameters,
                        steps,
                        key_mask,
                        causal,
                        memory,
                        mem_steps,
                        mem_mask,
                        norm if layer == last and fused else None,
                        parts=pending,
                        residual=k > 0,
                    )
                    for index in range(len(self.bands.bounds))
                ]
                outputs.append(
                    self.tape.concat_rows(bands)
                    if len(bands) > 1
                    else bands[0]
                )
            value, pending = outputs[0], outputs[1:]
            if group[-1] in inject:
                value = self.tape.add(value, inject[group[-1]])
        return value if fused else self.tape.layer_norm(value, *norm)

    def _band(
        self,
        value,
        index,
        tag,
        parameters,
        steps,
        key_mask,
        causal,
        memory,
        mem_steps,
        mem_mask,
        norm,
        parts=(),
        residual=False,
    ):
        t = self.tape
        start, stop = self.bands.bounds[index]
        value = self.bands.view(t, value, index, steps, self.width)
        parts = [
            self.bands.view(t, v, index, steps, self.width) for v in parts
        ]
        attrs = dict(
            batch=stop - start,
            steps=steps,
            heads=self.heads,
            causal=causal,
            key_mask=key_mask[start:stop].tolist(),
            epsilon=1e-5,
            layer=tag,
            band=index,
            owned=self.owned,
        )
        if memory is not None:
            memory = self.bands.view(t, memory, index, mem_steps, self.width)
            attrs.update(
                mem_steps=mem_steps, mem_mask=mem_mask[start:stop].tolist()
            )
        return t.transformer_layer(
            value,
            parameters,
            attrs,
            memory=memory,
            norm=norm,
            parts=parts,
            residual=residual,
        )


class FusedTransformerGraph:
    """Compose fused stacks and a tied vocabulary head on one tape, without
    execution."""

    def __init__(
        self, tape, config, state, *, band=None, owned=False, vocab_shards=0
    ):
        self.tape, self.config = tape, dict(config)
        self.parameters = Parameters(tape, state)
        self.band, self.owned, self.vocab_shards = band, owned, vocab_shards

    def training_loss(
        self, source, target_input, targets, smoothing=0.1, hints=None
    ):
        """`hints`: (ids, buckets), each shaped as target_input, for a model
        with config['hint_buckets']."""
        source, target_input, targets = _validate_batch(
            self.config, source, target_input, targets
        )
        batch, source_steps = source.shape
        target_steps = target_input.shape[1]
        if hints is not None:
            if not self.config.get("hint_buckets"):
                raise ValueError("Hints need a model with hint_buckets")
            hints = tuple(np.asarray(h) for h in hints)
            if any(h.shape != target_input.shape for h in hints):
                raise ValueError(
                    "Hint ids and buckets must be shaped as the decoder input"
                )
        bands = ExampleBands(batch, self.band)
        vocab = VocabularyProjection(
            self.tape,
            self.parameters,
            self.config["embedding"],
            self.config["vocab"],
            self.vocab_shards,
            source,
            target_input,
            extra=hints[0] if hints is not None else None,
        )
        stack = FusedLayerStack(
            self.tape, self.parameters, self.config, bands, self.owned
        )
        source_mask, target_mask = source == 0, target_input == 0
        memory = stack.build(
            vocab.embed(source.reshape(-1), source_steps, batch),
            "encoder",
            ENCODER_PARAMS,
            source_steps,
            source_mask,
            False,
        )
        decoder_input = vocab.embed(
            target_input.reshape(-1), target_steps, batch
        )
        hint = inject = None
        if hints is not None:
            hint = HintVector(
                self.tape,
                self.parameters,
                vocab,
                self.config["embedding"],
                *hints,
                none=self.config["hint_buckets"] - 1,
            )
            decoder_input = self.tape.add(decoder_input, hint.gated(0))
            inject = {self.config["layers"] // 2: hint.gated(1)}
        decoded = stack.build(
            decoder_input,
            "decoder",
            DECODER_PARAMS,
            target_steps,
            target_mask,
            True,
            memory=memory,
            mem_steps=source_steps,
            mem_mask=source_mask,
            inject=inject,
        )
        if hint is not None:
            decoded = self.tape.add(decoded, hint.gated(2))
        return vocab.loss(decoded, targets, bands, smoothing)


def build_transformer_translation(
    config,
    state,
    source,
    target_input,
    targets,
    smoothing=0.1,
    *,
    shape_only=False,
    band=None,
    owned=False,
    vocab_shards=0,
    hints=None,
):
    """Compatibility recipe returning (differentiated tape, logits view,
    parameter IDs).

    `band` bounds examples per fused layer; `owned` keeps layer parameters on
    workers. Vocabulary ownership returns None instead of a materialized logits
    view.
    `hints`: experiment 55's (ids, buckets) for the decoder input, with
    config['hint_buckets'].
    """
    source, target_input, targets = _validate_batch(
        config, source, target_input, targets
    )
    tape = ShapeTape() if shape_only else Tape()
    model = FusedTransformerGraph(
        tape, config, state, band=band, owned=owned, vocab_shards=vocab_shards
    )
    loss, logits = model.training_loss(
        source, target_input, targets, smoothing, hints
    )
    tape.backward(loss)
    return tape, logits, dict(model.parameters)
