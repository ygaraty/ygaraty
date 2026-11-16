"""Pure workload metrics; all inputs are supplied by callers, with no dataset
access.

The nanochat yardstick is an illustrative extrapolation, not measured training
time.
"""

NANOCHAT_FLOPS = 4e19
YEAR = 365.25 * 86400


def layer_parameters(width, hidden, decoder=False):
    attention = 4 * (width * width + width)
    feed_forward = width * hidden + hidden + hidden * width + width
    return (
        (2 if decoder else 1) * attention
        + feed_forward
        + (6 if decoder else 4) * width
    )


def useful_flops(model, tokens):
    """Training FLOPs one update spent on real tokens. `model`: the job's
    model_config (vocab,
    embedding, hidden, layers); `tokens`: the batch's real tokens, as
    DatabaseTranslationJob
    records them in dataset_source['tokens'] (source and target, each with its
    EOS or BOS)."""
    width, hidden, layers = (
        model["embedding"],
        model["hidden"],
        model["layers"],
    )
    encoder = layers * layer_parameters(width, hidden)
    decoder = (
        layers * layer_parameters(width, hidden, decoder=True)
        + model["vocab"] * width
    )
    return 6 * (encoder * tokens["source"] + decoder * tokens["target"])


def nanochat_seconds(flops_per_second):
    return NANOCHAT_FLOPS / flops_per_second


def duration(seconds):
    days = seconds / 86400
    return (
        f"{seconds / YEAR:,.0f} years"
        if days >= 3 * 365
        else f"{days:,.0f} days"
        if days >= 3
        else f"{seconds / 3600:,.1f} hours"
    )


def steady(updates, checkpoint_every, last_update=None):
    """The updates the KPI counts: not the first (owners load their state), not
    a checkpoint's."""
    return [
        u
        for u in updates[1:]
        if u["update"] != last_update
        and not (checkpoint_every and u["update"] % checkpoint_every == 0)
    ]


def summarize(updates, model):
    """Steady updates' seconds, real tokens per second, useful FLOP/s and the
    nanochat time."""
    seconds = sum(u["seconds"] for u in updates)
    flops = sum(useful_flops(model, u["tokens"]) for u in updates)
    real = sum(u["tokens"]["source"] + u["tokens"]["target"] for u in updates)
    padded = sum(u["tokens"]["padded"] for u in updates)
    return dict(
        updates=len(updates),
        seconds_per_update=seconds / len(updates),
        real_tokens_per_second=real / seconds,
        real_share=real / padded,
        useful_gflops=flops / seconds / 1e9,
        nanochat_seconds=nanochat_seconds(flops / seconds),
    )
