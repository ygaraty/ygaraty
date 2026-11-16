"""Shared operation layouts used by graph builders, lowering, and numerical
kernels.

These definitions describe tensor representation only; importing them
never loads execution code. Parameter order and packed offsets are part
of the task contract.
"""

# Serialized block element sizes, shared by storage and offline traffic models.
BLOCK_DTYPES = {"<f8": 8, "<f4": 4, "bf16": 2}

H_STATE, C_STATE, H_OUT, GATES = 0, 1, 2, 3  # column blocks of an LSTM value

ENCODER_PARAMS = [
    "norm1.weight",
    "norm1.bias",
    "self_attn.in_proj_weight",
    "self_attn.in_proj_bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "norm2.weight",
    "norm2.bias",
    "linear1.weight",
    "linear1.bias",
    "linear2.weight",
    "linear2.bias",
]
DECODER_PARAMS = [
    "norm1.weight",
    "norm1.bias",
    "self_attn.in_proj_weight",
    "self_attn.in_proj_bias",
    "self_attn.out_proj.weight",
    "self_attn.out_proj.bias",
    "norm2.weight",
    "norm2.bias",
    "multihead_attn.in_proj_weight",
    "multihead_attn.in_proj_bias",
    "multihead_attn.out_proj.weight",
    "multihead_attn.out_proj.bias",
    "norm3.weight",
    "norm3.bias",
    "linear1.weight",
    "linear1.bias",
    "linear2.weight",
    "linear2.bias",
]


def packed_layout(shapes, d):
    """Where each parameter's gradient goes in the packed backward output
    (rows of width d): [(row offset, rows, mode)], total rows. Mode
    'rows': shape (a, d), stored as is. 'transposed': shape (d, b),
    stored as its transpose. 'vector': shape (1, k), k values over
    ceil(k/d) rows, zero padded.
    """
    layout, offset = [], 0
    for a, b in shapes:
        if b == d:
            mode, rows = "rows", a
        elif a == d:
            mode, rows = "transposed", b
        elif a == 1:
            mode, rows = "vector", -(-b // d)
        else:
            raise ValueError(
                f"Cannot pack a gradient of shape {(a, b)} "
                f"into rows of width {d}"
            )
        layout.append((offset, rows, mode))
        offset += rows
    return layout, offset
