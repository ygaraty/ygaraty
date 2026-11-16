"""Transformer task planning using the shared sequence lowering context."""

from .sequence_common import PUBLISH_WIRE
from .tasks import Input, Task


class TransformerLowering:
    """Internal lowering mixin; state and emission belong to
    sequence.Lowering.
    """

    def lower_tf_layer(self, node):
        """A fused Transformer layer over a band of examples is one task
        (experiment 30). An owned layer's task takes no parameters: its
        owner holds them (experiment 31).
        """
        if not node.attrs.get("owned"):
            self.emit(
                node.id,
                node.op,
                [Input(v) for v in node.inputs],
                node.shape,
                node.attrs,
            )
            return
        decoder, tag = node.attrs["decoder"], node.attrs["layer"]
        # The input, the memory and experiment 78's parallel parts.
        data = node.inputs[
            : (2 if decoder else 1) + node.attrs.get("parts", 0)
        ]
        if node.op == "tf_layer":
            deps = (
                [Input(f"owner/{tag}/load")]
                if self.owned and self.owned.get("load")
                else []
            )
            self.emit(
                node.id,
                "tf_owned_layer",
                [Input(v) for v in data] + deps,
                node.shape,
                dict(node.attrs, owner=tag),
            )
        else:
            g = node.inputs[len(data)]
            rows, d = self.tape.shapes[node.inputs[0]]
            mrows = self.tape.shapes[node.inputs[1]][0] if decoder else 0
            self.emit(
                node.id,
                "tf_owned_grad",
                [Input(v) for v in data] + [Input(g)],
                (rows + mrows + 1, d),
                dict(
                    node.attrs,
                    owner=tag,
                    upload_rows=[rows] + ([mrows] if decoder else []) + [1],
                ),
            )

    def vocab_load(self, attrs):
        return (
            [Input(f"owner/{attrs['owner']}/load")]
            if self.owned and self.owned.get("load")
            else []
        )

    def lower_vocab_gather(self, node):
        self.emit(
            node.id,
            "tf_vocab_gather",
            self.vocab_load(node.attrs),
            node.shape,
            node.attrs,
        )

    def lower_vocab_output(self, node):
        """Per band and shard, the shard's logit statistics; per band, its
        rows of the loss from every shard's statistics; the loss is their
        sum.
        """
        bands, shards = node.inputs[:-2], self.tape.owned_vocab["shards"]
        losses, start = [], 0
        for k, band in enumerate(bands):
            rows = self.tape.shapes[band][0]
            targets = node.attrs["targets"][start : start + rows]
            stats = []
            for s, (v0, v1) in enumerate(shards):
                owner = dict(
                    owner=f"vocab.{s}",
                    layer=f"vocab.{s}",
                    shard=s,
                    rows=[v0, v1],
                )
                stats.append(
                    self.emit(
                        f"{node.id}/stats/{k}/{s}",
                        "tf_vocab_stats",
                        [Input(band)] + self.vocab_load(owner),
                        (rows, 4),
                        dict(owner, band=k, targets=targets),
                    )
                )
            losses.append(
                self.emit(
                    f"{node.id}/loss/{k}",
                    "vocab_loss",
                    [Input(v) for v in stats],
                    (1, 1),
                    dict(
                        targets=targets,
                        ignore_index=node.attrs.get("ignore_index"),
                        vocab=node.attrs["vocab"],
                        count=node.attrs["count"],
                        label_smoothing=node.attrs.get("label_smoothing", 0.0),
                    ),
                )
            )
            start += rows
        if len(losses) > 1:
            self.emit(node.id, "add_many", [Input(v) for v in losses], (1, 1))
        else:
            self.emit(
                node.id, "scale", [Input(losses[0])], (1, 1), {"factor": 1.0}
            )

    def lower_vocab_grad(self, node):
        band, seed, attrs = node.inputs[0], node.inputs[3], node.attrs
        stats = [
            Input(f"{attrs['loss']}/stats/{attrs['band']}/{s}")
            for s in range(attrs["shards"])
        ]
        self.emit(
            node.id,
            "tf_vocab_grad",
            [Input(band)] + stats + [Input(seed)],
            node.shape,
            attrs,
        )

    def lower_vocab_embed_grad(self, node):
        # Either the shard's gathered rows' gradient, or (attrs['scatter']) the
        # embeddings'
        # gradient bands, read as regions of the layers' packed results.
        self.emit(
            node.id,
            "tf_vocab_embed_grad",
            [Input(v) for v in node.inputs],
            node.shape,
            node.attrs,
        )

    def vocab_tasks(self, updates, keep):
        """Per vocabulary shard: the load of its rows (first update), its Adam
        step after every band's gradient and the embedding gradient, and on
        publish updates its rows of the table, the bias and their moments,
        assembled into whole values.
        """
        vocab = self.tape.owned_vocab
        table, bias = vocab["params"]
        v, d = self.tape.shapes[table]
        published = {}
        for s, (v0, v1) in enumerate(vocab["shards"]):
            tag = f"vocab.{s}"
            owner = dict(owner=tag, layer=tag, shard=s, rows=[v0, v1])
            if self.owned.get("load"):
                regions = [
                    Input(name, region)
                    for prefix in ("", "adam_m_", "adam_v_")
                    for name, region in (
                        (prefix + table, [v0, v1, 0, d]),
                        (prefix + bias, [0, 1, v0, v1]),
                    )
                ]
                self.tasks.insert(
                    0,
                    Task(
                        f"owner/{tag}/load",
                        "tf_vocab_load",
                        regions,
                        (1, 1),
                        owner,
                    ),
                )
            markers = []
            for ident in vocab["grads"][s]:
                rows = self.tape.shapes[ident][0]
                markers.append(Input(ident, [rows - 1, rows, 0, 1]))
            embed = vocab["embeds"].get(s)
            if embed is not None:
                markers.append(Input(embed))
            step = f"owner/{tag}/step"
            self.emit(
                step,
                "tf_vocab_step",
                markers,
                (1, 1),
                dict(
                    owner,
                    bands=len(vocab["grads"][s]),
                    embed=embed is not None,
                    step=self.owned["step"],
                    learning_rate=self.owned["learning_rate"],
                    beta1=self.owned["beta1"],
                    beta2=self.owned["beta2"],
                    epsilon=self.owned["epsilon"],
                ),
            )
            keep.add(step)
            if self.owned.get("publish"):
                for index, (name, region) in enumerate(
                    ((table, [v0, v1, 0, d]), (bias, [0, 1, v0, v1]))
                ):
                    for kind, prefix in (
                        ("w", ""),
                        ("m", "adam_m_"),
                        ("v", "adam_v_"),
                    ):
                        ident = f"owner/{tag}/publish/{kind}/{index}"
                        self.emit(
                            ident,
                            "tf_vocab_publish",
                            [Input(step)],
                            (region[1] - region[0], region[3] - region[2]),
                            dict(
                                owner,
                                kind=kind,
                                index=index,
                                wire=PUBLISH_WIRE,
                            ),
                        )
                        published.setdefault((prefix + name, name), []).append(
                            (ident, region)
                        )
        for (name, base), parts in published.items():
            # A moment has its parameter's shape (the job's moments are leaves,
            # not tape values).
            ident, shape = (
                f"owner/vocab/publish/{name}",
                self.tape.shapes[base],
            )
            self.emit(
                ident,
                "assemble",
                [Input(i) for i, _ in parts],
                shape,
                {"shape": list(shape), "blocks": [r for _, r in parts]},
            )
            updates[name] = ident
            keep.add(ident)

    def owned_tasks(self):
        """Per owned layer: the load (first update), the Adam step after every
        band's gradient, and the publish tasks (checkpoint updates) whose
        results replace the stored parameters.
        """
        updates, keep = {}, set()
        if self.tape.owned_vocab:
            self.vocab_tasks(updates, keep)
        for tag, layer in self.tape.owned_layers.items():
            params, owner = (
                layer["params"],
                dict(owner=tag, layer=tag, decoder=layer["decoder"]),
            )
            if self.owned.get("load"):
                # First in the plan: the layer's band tasks read it.
                self.tasks.insert(
                    0,
                    Task(
                        f"owner/{tag}/load",
                        "tf_owned_load",
                        [Input(p) for p in params]
                        + [Input("adam_m_" + p) for p in params]
                        + [Input("adam_v_" + p) for p in params],
                        (1, 1),
                        owner,
                    ),
                )
            markers = []
            for ident in layer["backward"]:
                _rows = self.tape.shapes[ident][0]
                task = next(t for t in reversed(self.tasks) if t.id == ident)
                markers.append(
                    Input(ident, [task.shape[0] - 1, task.shape[0], 0, 1])
                )
            step = f"owner/{tag}/step"
            self.emit(
                step,
                "tf_owned_step",
                markers,
                (1, 1),
                dict(
                    owner,
                    bands=len(markers),
                    step=self.owned["step"],
                    learning_rate=self.owned["learning_rate"],
                    beta1=self.owned["beta1"],
                    beta2=self.owned["beta2"],
                    epsilon=self.owned["epsilon"],
                    # M-2: Muon on the layer's matrices when the job asks.
                    **{
                        k: self.owned[k]
                        for k in (
                            "optimizer",
                            "muon_learning_rate",
                            "momentum",
                            "ns_steps",
                        )
                        if k in self.owned
                    },
                ),
            )
            keep.add(step)
            if self.owned.get("publish"):
                for index, p in enumerate(params):
                    for kind, name in (
                        ("w", p),
                        ("m", "adam_m_" + p),
                        ("v", "adam_v_" + p),
                    ):
                        ident = f"owner/{tag}/publish/{kind}/{index}"
                        self.emit(
                            ident,
                            "tf_owned_publish",
                            [Input(step)],
                            self.tape.shapes[p],
                            dict(
                                owner,
                                kind=kind,
                                index=index,
                                wire=PUBLISH_WIRE,
                            ),
                        )
                        updates[name] = ident
                        keep.add(ident)
        return updates, keep

    lower_tf_layer_grad = lower_tf_layer
