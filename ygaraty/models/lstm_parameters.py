"""Shape-only LSTM parameter layout and bounded initialization, without
PyTorch."""

import math

from ygaraty.compiler.plan import bounded_reductions as bounded_reductions
from ygaraty.models.synthetic import initial_block, regions


def parameter_specs(config):
    v, e, h, layers = (
        config[k] for k in ("vocab", "embedding", "hidden", "layers")
    )
    specs = {}

    def add(name, shape, **initializer):
        specs[name] = dict(
            shape=(1, shape[0]) if len(shape) == 1 else shape,
            original_shape=shape,
            **initializer,
        )

    add("output_bias", (v,), kind="zeros")
    add("embedding.weight", (v, e), kind="normal", scale=1.0, zero_row=0)
    for prefix, directions in [
        ("encoder", ("", "_reverse")),
        ("decoder", ("",)),
    ]:
        for layer in range(layers):
            width = e if layer == 0 else (2 * h if prefix == "encoder" else h)
            for direction in directions:
                suffix = f"_l{layer}{direction}"
                for kind, shape in [
                    ("weight_ih", (4 * h, width)),
                    ("weight_hh", (4 * h, h)),
                    ("bias_ih", (4 * h,)),
                    ("bias_hh", (4 * h,)),
                ]:
                    add(
                        prefix + "." + kind + suffix,
                        shape,
                        kind="uniform",
                        bound=1 / math.sqrt(h),
                    )
    for prefix, width, output, bias in [
        ("bridge_h", 2 * h, h, True),
        ("bridge_c", 2 * h, h, True),
        ("keys", 2 * h, h, False),
        ("combine", 3 * h, e, True),
    ]:
        add(
            prefix + ".weight",
            (output, width),
            kind="uniform",
            bound=1 / math.sqrt(width),
        )
        if bias:
            add(
                prefix + ".bias",
                (output,),
                kind="uniform",
                bound=1 / math.sqrt(width),
            )
    return specs


def seed_tensor(store, version, name, spec, seed=2026):
    blocks = []
    for region in regions(spec["shape"], store.block_size):
        array = initial_block(
            dict(seed=seed, tile_size=store.block_size), name, spec, region
        )
        blocks.append(store.put_block(array, region))
        del array
    return store.publish(version, spec["shape"], blocks)
