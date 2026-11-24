"""Turn a tensor graph into a dependency DAG of worker-sized operations."""

import math
from dataclasses import dataclass

from .tasks import Input, Task

# A matmul runs as one task per output block, over its whole inner dimension,
# when that task's
# two input blocks fit in this many bytes; otherwise one partial product per
# inner tile plus
# an add_many that reads them all back. A job with a larger task budget passes
# a larger value.
FUSED_K_BYTES = 12 << 20
# The largest result one task may produce (admit_plan enforces the same bound).
MAX_OUTPUT_BYTES = 8 << 20


CHAIN_OPS = ("lstm_chain", "lstm_chain_grad")
# Tasks admitted under the large-task budget rather than the per-tile one
# (results may exceed
# MAX_OUTPUT_BYTES, so they need the relay): LSTM chains and fused Transformer
# layers.
LARGE_OPS = CHAIN_OPS + (
    "tf_layer",
    "tf_layer_grad",
    "tf_owned_load",
    "tf_vocab_load",
    "tf_vocab_publish",
)
FUSED_OUTPUT_OPS = (
    "ce_fused",
    "factorization_tiles",
)  # results above MAX_OUTPUT_BYTES, bounded by the caller's output_budget


def split(
    tape,
    tile_size=64,
    fuse_bytes=None,
    chain_steps=1,
    chain_bytes=None,
    output_bytes=None,
    owned=None,
):
    """Lower a tape into tasks of about `tile_size` square tiles.

    `fuse_bytes` is the byte budget for the two inputs of a matmul fused
    over its inner dimension (default FUSED_K_BYTES). A caller that
    passes one (the production job, from its task budget) also allows
    output blocks wider than a tile, up to MAX_OUTPUT_BYTES; by default
    every task's output stays within one tile. `chain_steps` is the most
    consecutive LSTM steps one task may run (experiment 27); 1 keeps one
    task per step. `chain_bytes`, when set, is the admission budget for
    those chain tasks instead of the output cap (a whole layer per task
    needs results larger than MAX_OUTPUT_BYTES, which only relay uploads
    carry). `output_bytes`, when set, fuses the output layer (see
    fuse_output) under that task budget. `owned`, for a tape with owned
    layers: dict(load, publish, step, learning_rate, beta1, beta2,
    epsilon); the owned layers' extra tasks' results land in
    tape.owned_updates.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    from ygaraty.compiler.sequence import Lowering, prune, resolve_views

    tasks = []
    output_cap = (
        MAX_OUTPUT_BYTES
        if fuse_bytes is not None
        else tile_size * tile_size * 8
    )
    if fuse_bytes is None:
        fuse_bytes = FUSED_K_BYTES
    # Bytes one task may move: its inputs and its output (admit_plan budgets
    # four times that).
    budget = fuse_bytes + tile_size * tile_size * 8
    if (
        getattr(tape, "owned_layers", None)
        or getattr(tape, "owned_vocab", None)
    ) and owned is None:
        raise ValueError(
            "A tape with owned layers needs owned=dict(load, publish, step, "
            "learning_rate, beta1, beta2, epsilon)"
        )
    lowering = Lowering(
        tape,
        tile_size,
        tasks,
        budget,
        output_cap,
        chain_steps=chain_steps,
        chain_bytes=chain_bytes,
        owned=owned,
    )
    fused = find_output_layer(tape) if output_bytes else None
    aliases = {}
    for node in tape.nodes:
        if fused and node.id in fused["skip"]:
            continue
        if fused and node.id == fused["loss"]:
            aliases[fused["gradient"]] = fuse_output(
                tape, fused, tasks, output_cap, output_bytes
            )
            continue
        if fused and node.id == fused["gradient"]:
            continue  # an alias of the fused bands' first V columns
        if lowering.lower(node):
            continue
        if node.op in ("embedding", "embedding_backward", "embed"):
            blocks, positions = [], []
            m, n = node.shape
            for i in range(0, m, tile_size):
                for j in range(0, n, tile_size):
                    ei, ej = min(m, i + tile_size), min(n, j + tile_size)
                    shape, region = (ei - i, ej - j), [i, ei, j, ej]
                    ident = f"{node.id}/block/{i}/{j}"
                    if node.op == "embedding":
                        refs = [
                            Input(node.inputs[0], [row, row + 1, j, ej])
                            for row in node.attrs["ids"][i:ei]
                        ]
                        tasks.append(Task(ident, "gather_rows", refs, shape))
                    elif node.op == "embed":
                        refs = [
                            Input(node.inputs[0], [row, row + 1, j, ej])
                            for row in node.attrs["ids"][i:ei]
                        ] + [
                            Input(node.inputs[1], [row, row + 1, j, ej])
                            for row in node.attrs["positions"][i:ei]
                        ]
                        tasks.append(
                            Task(
                                ident,
                                "embed_rows",
                                refs,
                                shape,
                                {
                                    "rows": ei - i,
                                    "factor": node.attrs["factor"],
                                },
                            )
                        )
                    elif len(node.inputs) > 1:
                        # The gradient as row parts (the bands its producers
                        # made, read where they
                        # are): one scatter per output tile over all of them,
                        # nothing to sum after.
                        tasks.append(
                            Task(
                                ident,
                                "scatter_rows",
                                [
                                    Input(v, [0, tape.shapes[v][0], j, ej])
                                    for v in node.inputs
                                ],
                                shape,
                                {
                                    "shape": list(shape),
                                    "ids": [
                                        x - i
                                        if x != node.attrs.get("padding_idx")
                                        else -1
                                        for x in node.attrs["ids"]
                                    ],
                                    **(
                                        {"factor": node.attrs["factor"]}
                                        if node.attrs.get("factor", 1.0) != 1.0
                                        else {}
                                    ),
                                },
                            )
                        )
                    else:
                        parts = []
                        # Rows of the gradient per scatter task: as many as the
                        # budget allows.
                        chunk = max(
                            tile_size,
                            min(
                                len(node.attrs["ids"]),
                                budget // (16 * (ej - j)),
                            ),
                        )
                        for row in range(0, len(node.attrs["ids"]), chunk):
                            end = min(row + chunk, len(node.attrs["ids"]))
                            part = ident + f"/rows/{row}"
                            tasks.append(
                                Task(
                                    part,
                                    "scatter_rows",
                                    [Input(node.inputs[0], [row, end, j, ej])],
                                    shape,
                                    {
                                        "shape": list(shape),
                                        "ids": [
                                            x - i
                                            if x
                                            != node.attrs.get("padding_idx")
                                            else -1
                                            for x in node.attrs["ids"][row:end]
                                        ],
                                        **(
                                            {"factor": node.attrs["factor"]}
                                            if node.attrs.get("factor", 1.0)
                                            != 1.0
                                            else {}
                                        ),
                                    },
                                )
                            )
                            parts.append(Input(part))
                        tasks.append(Task(ident, "add_many", parts, shape))
                    blocks.append(Input(ident))
                    positions.append(region)
            tasks.append(
                Task(
                    node.id,
                    "assemble",
                    blocks,
                    node.shape,
                    {"shape": list(node.shape), "blocks": positions},
                )
            )
            continue
        # Bound elementwise operations, shared-weight gradient accumulation,
        # gate slicing/scattering, and bias reductions by the tile size too.
        tiled = {
            "add",
            "add_many",
            "mul",
            "add_bias",
            "relu",
            "relu_backward",
            "sigmoid",
            "tanh",
            "sigmoid_backward",
            "tanh_backward",
            "scale",
            "slice_columns",
            "slice_backward",
            "reduce_rows",
        }
        if node.op in tiled:
            blocks, positions = [], []
            m, n = node.shape
            for i in range(0, m, tile_size):
                for j in range(0, n, tile_size):
                    ei, ej = min(m, i + tile_size), min(n, j + tile_size)
                    region = [i, ei, j, ej]
                    shape = (ei - i, ej - j)
                    ident = f"{node.id}/block/{i}/{j}"
                    op, attrs = node.op, dict(node.attrs)
                    refs = [Input(x, region) for x in node.inputs]
                    if op == "add_bias":
                        refs[1] = Input(node.inputs[1], [0, 1, j, ej])
                    elif op == "slice_columns":
                        start = attrs["start"]
                        refs = [
                            Input(
                                node.inputs[0], [i, ei, j + start, ej + start]
                            )
                        ]
                        op, attrs = "scale", {"factor": 1.0}
                    elif op == "slice_backward":
                        start, end = (
                            attrs["start"],
                            attrs["start"] + attrs["width"],
                        )
                        lo, hi = max(j, start), min(ej, end)
                        refs = (
                            [
                                Input(
                                    node.inputs[0],
                                    [i, ei, lo - start, hi - start],
                                )
                            ]
                            if hi > lo
                            else []
                        )
                        op, attrs = (
                            "slice_block",
                            {"shape": list(shape), "offset": max(0, lo - j)},
                        )
                    elif op == "reduce_rows":
                        parts = []
                        total = tape.shapes[node.inputs[0]][0]
                        chunk = max(
                            tile_size, min(total, budget // (16 * (ej - j)))
                        )
                        for row in range(0, total, chunk):
                            part = ident + f"/rows/{row}"
                            tasks.append(
                                Task(
                                    part,
                                    op,
                                    [
                                        Input(
                                            node.inputs[0],
                                            [
                                                row,
                                                min(row + chunk, total),
                                                j,
                                                ej,
                                            ],
                                        )
                                    ],
                                    shape,
                                )
                            )
                            parts.append(Input(part))
                        op, refs, attrs = "add_many", parts, {}
                    tasks.append(Task(ident, op, refs, shape, attrs))
                    blocks.append(Input(ident))
                    positions.append(region)
            tasks.append(
                Task(
                    node.id,
                    "assemble",
                    blocks,
                    node.shape,
                    {"shape": list(node.shape), "blocks": positions},
                )
            )
            continue
        if node.op in ("layer_norm", "layer_norm_backward"):
            # Rows are independent: bands of rows, each within the output cap
            # and the budget. The
            # affine gradients (index 1 and 2) are row sums: one partial per
            # band, then a sum.
            rows, cols = tape.shapes[node.inputs[0]]
            reduces = (
                node.op == "layer_norm_backward"
                and node.attrs.get("index", 0) > 0
            )
            band = max(
                1,
                min(rows, output_cap // (cols * 8), budget // (8 * 3 * cols)),
            )
            blocks, positions, parts = [], [], []
            for i in range(0, rows, band):
                ei = min(rows, i + band)
                refs = [Input(node.inputs[0], [i, ei, 0, cols])] + [
                    Input(v)
                    if tape.shapes[v][0] == 1
                    else Input(v, [i, ei, 0, cols])
                    for v in node.inputs[1:]
                ]
                ident = f"{node.id}/rows/{i}"
                if reduces:
                    tasks.append(
                        Task(ident, node.op, refs, (1, cols), node.attrs)
                    )
                    parts.append(Input(ident))
                else:
                    tasks.append(
                        Task(ident, node.op, refs, (ei - i, cols), node.attrs)
                    )
                    blocks.append(Input(ident))
                    positions.append([i, ei, 0, cols])
            if reduces:
                tasks.append(Task(node.id, "add_many", parts, (1, cols)))
            else:
                tasks.append(
                    Task(
                        node.id,
                        "assemble",
                        blocks,
                        node.shape,
                        {"shape": list(node.shape), "blocks": positions},
                    )
                )
            continue
        if node.op != "matmul":
            tasks.append(
                Task(
                    node.id,
                    node.op,
                    [Input(x) for x in node.inputs],
                    node.shape,
                    node.attrs,
                )
            )
            continue
        a, b = node.inputs
        m, n = node.shape
        k = tape.shapes[a][1]
        blocks, positions = [], []
        for i in range(0, m, tile_size):
            ei, j = min(m, i + tile_size), 0
            while j < n:
                # Widest output block (in tiles, doubling) whose whole-K task
                # fits the budget: the
                # left row block is then read once, and a consumer of these
                # rows (an LSTM step
                # reading all gate columns) depends on this block alone, not on
                # the whole product.
                span = tile_size
                while (
                    span < n - j
                    and (
                        (ei - i) * k
                        + k * min(n - j, 2 * span)
                        + (ei - i) * min(n - j, 2 * span)
                    )
                    * 8
                    <= budget
                    and (ei - i) * min(n - j, 2 * span) * 8 <= output_cap
                ):
                    span *= 2
                ej = min(n, j + span)
                block = f"{node.id}/block/{i}/{j}"
                if (
                    (ei - i) * k + k * (ej - j) + (ei - i) * (ej - j)
                ) * 8 <= budget:
                    # The whole inner dimension in one task: no partial
                    # products written and read back.
                    tasks.append(
                        Task(
                            block,
                            "matmul",
                            [Input(a, [i, ei, 0, k]), Input(b, [0, k, j, ej])],
                            (ei - i, ej - j),
                        )
                    )
                    blocks.append(Input(block))
                    positions.append([i, ei, j, ej])
                    j = ej
                    continue
                partials = []
                for q in range(0, k, tile_size):
                    eq = min(k, q + tile_size)
                    ident = f"{block}/partial/{q}"
                    tasks.append(
                        Task(
                            ident,
                            "matmul",
                            [
                                Input(a, [i, ei, q, eq]),
                                Input(b, [q, eq, j, ej]),
                            ],
                            (ei - i, ej - j),
                        )
                    )
                    partials.append(Input(ident))
                tasks.append(
                    Task(block, "add_many", partials, (ei - i, ej - j))
                )
                blocks.append(Input(block))
                positions.append([i, ei, j, ej])
                j = ej
        tasks.append(
            Task(
                node.id,
                "assemble",
                blocks,
                node.shape,
                {"shape": list(node.shape), "blocks": positions},
            )
        )
    owned_keep = set()
    if getattr(tape, "owned_layers", None) or getattr(
        tape, "owned_vocab", None
    ):
        tape.owned_updates, owned_keep = lowering.owned_tasks()
    tasks = resolve_views(tape, tasks)
    if aliases:
        tasks = resolve_aliases(tasks, aliases, tape)
    # Every tape value stays available (tests and diagnostics read
    # intermediates by name), except
    # row concatenations nobody reads whole: their readers resolved to the
    # parts, and the copies
    # behind them (planner-made ids, with a '/') go with them.
    keep = {n.id for n in tape.nodes if n.op != "concat_rows"} | {tape.loss}
    keep.update(getattr(tape, "gradients", {}).values())
    keep.update(getattr(tape, "carry", {}).values())
    keep.update(owned_keep)
    result = prune(resolve_assemblies(tasks), keep)
    # Lowering may create tiles/assemblies without copying node attributes.
    blocks = {n.id: n.attrs["muon_block"] for n in tape.nodes
              if "muon_block" in n.attrs}
    for task in result:
        block = blocks.get(task.id.split("/")[0])
        if block is not None:
            task.attrs["muon_block"] = block
    return result


# Results that keep the wire's dtype on a narrow activation wire: losses,
# statistics and
# scalars, optimizer state and updates, owners' state, and metadata.
FULL_WIDTH_OPS = {
    "assemble",
    "transpose",
    "ce_fused",
    "cross_entropy",
    "sum",
    "tf_vocab_stats",
    "vocab_loss",
    "sum_squares",
    "scatter_rows",
    "embedding_backward",
    "clip_gradient",
    "adam_step",
    "adam_m",
    "adam_v",
    "adam_update",
    "sgd",
    "tf_owned_step",
    "tf_vocab_step",
    "tf_owned_load",
    "tf_vocab_load",
    "tf_owned_publish",
    "tf_vocab_publish",
    "tf_vocab_embed_grad",
    "tf_owned_share",
}


# Reviewed functional operations: a task of one of these that nothing reads can
# go. Anything
# else (owned loads and publishes, statistics a job reads from the store,
# kernels with side
# effects) is kept, with everything it depends on.
PURE_OPS = frozenset(
    {
        # runtime/kernels.py
        "matmul",
        "add",
        "add_bias",
        "mul",
        "scale",
        "relu",
        "relu_backward",
        "sum",
        "sum_squares",
        "transpose",
        "reduce_rows",
        "broadcast",
        "add_many",
        "assemble",
        "sigmoid",
        "tanh",
        "sigmoid_backward",
        "tanh_backward",
        "concat_columns",
        "slice_columns",
        "slice_backward",
        "slice_block",
        "softmax",
        "softmax_backward",
        "layer_norm",
        "layer_norm_backward",
        "cross_entropy",
        "ce_backward",
        "ce_fused",
        "embedding",
        "embedding_backward",
        "embed",
        "embed_rows",
        "gather_rows",
        "scatter_rows",
        "clip_gradient",
        "sgd",
        "adam_m",
        "adam_v",
        "adam_step",
        "adam_update",
        "group",
        # sequence_kernels.py
        "view",
        "concat_rows",
        "place",
        "place_block",
        "lstm",
        "lstm_backward",
        "bqk",
        "bpv",
        "bptq",
        "lstm_cell",
        "lstm_cell_backward",
        "lstm_cell_carry",
        "lstm_cell_grad",
        "lstm_state_grad",
        "lstm_chain",
        "lstm_chain_grad",
        # transformer_kernels.py: the unowned layer kernels (owned and
        # vocabulary ops read and write the store)
        "tf_layer",
        "tf_layer_grad",
        "vocab_loss",
    }
)


def prune_unused(tasks, keep):
    """The tasks that reach a kept value, in the given order: dead work
    leaves the plan.

    `keep` names the values the job reads after the update (its
    parameter updates and loss; leaves are not tasks). A task is kept
    when something kept depends on it, when its operation is not in
    PURE_OPS, or when it is owned (attrs 'owner'): those are roots too.
    Typical removals are the whole-matrix `assemble` the planner emits
    for every named value and the gradient of an untrained input. Values
    are unchanged: the kept tasks read exactly what they read before.
    """
    by_id = {t.id: t for t in tasks}
    roots = set(keep) | {
        t.id
        for t in tasks
        if t.op not in PURE_OPS or t.attrs.get("owner") is not None
    }
    live, todo = set(), [r for r in roots if r in by_id]
    while todo:
        ident = todo.pop()
        if ident in live:
            continue
        live.add(ident)
        todo.extend(
            d
            for d in by_id[ident].dependencies
            if d in by_id and d not in live
        )
    return [t for t in tasks if t.id in live]


def narrow_activations(tasks, tape, dtype="bf16"):
    """Mark the tasks whose results may travel as `dtype` (attrs['wire']):
    activations and their gradients. Parameters, their gradients
    (tape.parameter_side), optimizer state, losses, statistics and
    owners' state keep the wire's dtype; a fused layer's backward result
    is narrowed band by band (the input gradients, not the packed
    parameter gradients).
    """
    protected = set(tape.parameter_side) | set(tape.parameter_like)
    for task in tasks:
        node = task.id.split("/")[0]
        if (
            task.op in FULL_WIDTH_OPS
            or node in protected
            or math.prod(task.shape) <= 4
        ):
            continue
        if task.op == "tf_layer_grad":
            activations = 2 if task.attrs.get("decoder") else 1
            task.attrs["wire"] = [dtype] * activations + [None] * (
                len(task.attrs["upload_rows"]) - activations
            )
        else:
            task.attrs["wire"] = dtype
    return tasks


def critical_path(tasks):
    """Worker hops on the longest dependency path of a plan: the tasks a
    worker must run one after another, whatever the number of workers.
    Assemblies and transposes are coordinator metadata and cost no hop.
    Each hop costs at least one network round trip.
    """
    by_id = {t.id: t for t in tasks}
    indegree = {t.id: 0 for t in tasks}
    dependents = {}
    for t in tasks:
        for d in t.dependencies:
            if d in by_id:
                indegree[t.id] += 1
                dependents.setdefault(d, []).append(t.id)
    depth = {}
    ready = [i for i, n in indegree.items() if n == 0]
    for i in ready:
        depth[i] = 0 if by_id[i].op in ("assemble", "transpose") else 1
    while ready:
        i = ready.pop()
        for j in dependents.get(i, ()):
            cost = 0 if by_id[j].op in ("assemble", "transpose") else 1
            depth[j] = max(depth.get(j, 0), depth[i] + cost)
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(j)
    return max(depth.values(), default=0)


def find_output_layer(tape):
    """The pattern loss = cross_entropy(add_bias(matmul(x, W), b)) with its
    ce_backward, where the matmul and the bias add feed nothing else:
    the output layer of a classifier or language model.
    """
    nodes = {n.id: n for n in tape.nodes}
    consumers = {}
    for n in tape.nodes:
        for v in n.inputs:
            consumers.setdefault(v, set()).add(n.id)
    for c in tape.nodes:
        if c.op != "cross_entropy":
            continue
        biased = nodes.get(c.inputs[0])
        if not biased or biased.op != "add_bias":
            continue
        product = nodes.get(biased.inputs[0])
        if not product or product.op != "matmul":
            continue
        grads = [
            nodes[i]
            for i in consumers.get(biased.id, ())
            if nodes[i].op == "ce_backward"
        ]
        if len(grads) != 1 or consumers.get(biased.id, set()) != {
            c.id,
            grads[0].id,
        }:
            continue
        if consumers.get(product.id, set()) != {biased.id}:
            continue
        x, w = product.inputs
        return dict(
            skip={product.id, biased.id},
            loss=c.id,
            gradient=grads[0].id,
            x=x,
            w=w,
            bias=biased.inputs[1],
            seed=grads[0].inputs[1],
            attrs=c.attrs,
            rows=tape.shapes[x][0],
            width=tape.shapes[x][1],
            classes=tape.shapes[w][1],
        )


def fuse_output(tape, f, tasks, output_cap, budget):
    """Emit the fused output layer: per band of rows, one ce_fused task
    computes the logits, the loss and its gradient, returning [logit
    gradient | row loss]; the logits never leave the worker. The loss is
    the sum of the bands' row-loss columns. Returns (assembled value,
    columns) for the ce_backward alias.
    """
    rows, d, V = f["rows"], f["width"], f["classes"]
    per_row = d + 2 * (V + 1)  # input row, output row, and logits scratch
    fixed = d * V + V + 1
    band = min(output_cap // ((V + 1) * 8), (budget // 32 - fixed) // per_row)
    if band < 1:
        raise MemoryError(
            "The output-layer task budget is too small for one row; "
            "raise YGARATY_LSTM_OUTPUT_TASK_MIB"
        )
    attrs, targets = f["attrs"], f["attrs"]["targets"]
    count = sum(1 for t in targets if t != attrs.get("ignore_index"))
    parts, sums = [], []
    for r0 in range(0, rows, band):
        r1 = min(rows, r0 + band)
        ident = f"{f['loss']}/fused/{r0}"
        tasks.append(
            Task(
                ident,
                "ce_fused",
                [
                    Input(f["x"], [r0, r1, 0, d]),
                    Input(f["w"]),
                    Input(f["bias"]),
                    Input(f["seed"]),
                ],
                (r1 - r0, V + 1),
                dict(
                    targets=targets[r0:r1],
                    ignore_index=attrs.get("ignore_index"),
                    label_smoothing=attrs.get("label_smoothing", 0.0),
                    count=count,
                ),
            )
        )
        parts.append((ident, [r0, r1, 0, V + 1]))
        tasks.append(
            Task(
                ident + "/loss",
                "sum",
                [Input(ident, [0, r1 - r0, V, V + 1])],
                (1, 1),
            )
        )
        sums.append(Input(ident + "/loss"))
    tasks.append(
        Task(f["loss"], "add_many", sums, (1, 1))
        if len(sums) > 1
        else Task(f["loss"], "scale", sums, (1, 1), {"factor": 1.0})
    )
    whole = f"{f['loss']}/fused"
    tasks.append(
        Task(
            whole,
            "assemble",
            [Input(p) for p, _ in parts],
            (rows, V + 1),
            {"shape": [rows, V + 1], "blocks": [r for _, r in parts]},
        )
    )
    return whole, V


def resolve_aliases(tasks, aliases, tape):
    """Inputs naming an aliased value read the first columns of the value
    it aliases instead.
    """
    for task in tasks:
        inputs = []
        for x in task.inputs:
            if x.value in aliases:
                if task.op in ("assemble", "transpose"):
                    raise ValueError(
                        f"{task.id} needs {x.value} as a whole value; "
                        "it is fused into the output layer"
                    )
                base, width = aliases[x.value]
                rows = tape.shapes[x.value][0]
                x = Input(base, x.region or [0, rows, 0, width])
            inputs.append(x)
        task.inputs = inputs
    return tasks


def resolve_assemblies(tasks):
    """A task that reads a region of an assembled value reads it from the
    part that holds it.

    Assemblies are metadata, so this costs nothing, and it removes the
    barrier an assembly puts between producers and consumers: a step of
    one LSTM layer can start as soon as the rows it needs from the
    previous layer exist, instead of after that layer's last step.
    Regions that span several parts, and whole-value inputs, are left
    alone.
    """
    parts = {
        t.id: [
            (x.value, x.region, block)
            for x, block in zip(t.inputs, t.attrs["blocks"])
        ]
        for t in tasks
        if t.op == "assemble"
    }
    if not parts:
        return tasks

    def resolve(value, region):
        while value in parts and region is not None:
            i, ei, j, ej = region
            for part, source, (bi, bei, bj, bej) in parts[value]:
                if bi <= i and ei <= bei and bj <= j and ej <= bej:
                    r0, c0 = (source[0], source[2]) if source else (0, 0)
                    value, region = (
                        part,
                        [i - bi + r0, ei - bi + r0, j - bj + c0, ej - bj + c0],
                    )
                    break
            else:
                return value, region
        return value, region

    for task in tasks:
        if task.op in ("assemble", "transpose"):
            continue
        inputs = []
        for x in task.inputs:
            value, region = resolve(x.value, x.region)
            inputs.append(x if value == x.value else Input(value, region))
        task.inputs = inputs
    return tasks


def group(tasks, max_ops=4):
    """Fuse single-consumer chains, never crossing a fan-out or a join.

    Original operations and outputs remain inspectable inside each task.
    Restrict successors to one task dependency to avoid introducing cycles.
    """
    if max_ops < 1:
        raise ValueError("max_ops must be positive")
    by_id = {t.id: t for t in tasks}
    consumers = {t.id: set() for t in tasks}
    for t in tasks:
        for dep in t.dependencies & by_id.keys():
            consumers[dep].add(t.id)
    used, result = set(), []
    for first in tasks:
        if first.id in used:
            continue
        chain = [first]
        used.add(first.id)
        while len(chain) < max_ops and len(consumers[chain[-1].id]) == 1:
            following = by_id[next(iter(consumers[chain[-1].id]))]
            if (
                following.id in used
                or following.dependencies & by_id.keys() != {chain[-1].id}
            ):
                break
            chain.append(following)
            used.add(following.id)
        if len(chain) == 1:
            result.append(first)
            continue
        internal = {t.id for t in chain}
        external, indices, nodes = [], {}, []
        for t in chain:
            refs = []
            for ref in t.inputs:
                if ref.value in internal:
                    refs.append({"value": ref.value, "region": ref.region})
                else:
                    key = (ref.value, tuple(ref.region or []))
                    if key not in indices:
                        indices[key] = len(external)
                        external.append(ref)
                    refs.append({"external": indices[key]})
            nodes.append(
                dict(
                    id=t.id,
                    op=t.op,
                    inputs=refs,
                    shape=list(t.shape),
                    attrs=t.attrs,
                )
            )
        result.append(
            Task(
                chain[-1].id,
                "group",
                external,
                chain[-1].shape,
                {"nodes": nodes},
            )
        )
    return result


def bounded_reductions(tasks, fan_in=8, budget=None):
    """Keep joins from gathering an entire matrix's partial products at
    once.
    """
    if fan_in < 2:
        raise ValueError("fan_in must be at least 2")
    result = []
    for task in tasks:
        inputs = task.inputs
        if task.op == "add_many":
            limit = (
                min(
                    fan_in,
                    max(2, budget // (4 * math.prod(task.shape) * 8) - 1),
                )
                if budget
                else fan_in
            )
            level = 0
            while len(inputs) > limit:
                next_inputs = []
                for start in range(0, len(inputs), limit):
                    ident = f"{task.id}/join/{level}/{start}"
                    result.append(
                        Task(
                            ident,
                            "add_many",
                            inputs[start : start + limit],
                            task.shape,
                        )
                    )
                    next_inputs.append(Input(ident))
                inputs = next_inputs
                level += 1
            task = Task(task.id, task.op, inputs, task.shape, task.attrs)
        result.append(task)
    return result


@dataclass(frozen=True)
class AdmissionPolicy:
    """Per-task array/transport estimates, not a process RSS guarantee."""

    budget: int
    chain_budget: int | None = None
    output_budget: int | None = None

    def _special_budget(self, task: Task) -> int | None:
        if task.op in LARGE_OPS:
            return self.chain_budget
        if task.op in FUSED_OUTPUT_OPS:
            return self.output_budget
        return None

    def check(self, task: Task, shapes: dict) -> int:
        if task.op in ("assemble", "transpose"):
            return 0
        input_bytes = 0
        for ref in task.inputs:
            if ref.region:
                r0, r1, c0, c1 = ref.region
                dimensions = (r1 - r0, c1 - c0)
            else:
                dimensions = shapes[ref.value]
            input_bytes += math.prod(dimensions) * 8
        output_bytes = math.prod(task.shape) * 8
        estimate = 4 * (input_bytes + output_bytes)
        special = self._special_budget(task)
        if special:
            if estimate > special:
                raise MemoryError(
                    f"Task {task.id} ({task.op}) needs an estimated "
                    f"{estimate} bytes; "
                    f"its task budget is {special}."
                )
        elif estimate > self.budget or output_bytes > MAX_OUTPUT_BYTES:
            raise MemoryError(
                f"Task {task.id} ({task.op}) needs an estimated "
                f"{estimate} bytes; "
                f"budget is {self.budget}. "
                "Reduce the tile, batch or sentence size."
            )
        return estimate

    def admit(self, tasks, leaf_shapes) -> int:
        shapes = dict(leaf_shapes)
        shapes.update((task.id, task.shape) for task in tasks)
        return max((self.check(task, shapes) for task in tasks), default=0)


def admit_plan(
    tasks, leaf_shapes, budget, chain_budget=None, output_budget=None
):
    """Compatibility entry point for task-array and transport admission."""
    return AdmissionPolicy(budget, chain_budget, output_budget).admit(
        tasks, leaf_shapes
    )
