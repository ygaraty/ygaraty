"""Q4_K and Q6_K quantization in NumPy (the `gguf` package only decodes K-quants).

A port of llama.cpp's reference quantizers (ggml-quants.c, `quantize_row_q4_K_ref` with `make_qkx2_quants`, and
`quantize_row_q6_K_ref` with `make_qx_quants`), without importance weights, vectorized over blocks of 256
values. The block layouts are GGML's, so `gguf.quants.dequantize` and the WebGPU kernels decode the output
unchanged. Bit-exactness with llama.cpp is not a goal (its float order differs); the format and the search are.

    quantize(x, "Q4_K") -> uint8 [rows, row_bytes]     x: float32 [rows, columns], columns a multiple of 256
"""

import numpy as np

QK = 256
BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}


def row_bytes(kind, columns):
    if columns % QK:
        raise ValueError(
            f"{kind} rows need a multiple of {QK} columns, not {columns}"
        )
    return columns // QK * BLOCK_BYTES[kind]


def _rint(x):
    return np.rint(x).astype(
        np.int32
    )  # round half to even, as llama.cpp's nearest_int


def _q4k_subblocks(x, steps=20):
    """make_qkx2_quants(32, 15, x, weights, rmin=-1, rdelta=0.1, nstep) over x[..., 32]: (scale, min, L)."""
    nmax = 15
    av = np.sqrt(np.mean(x * x, axis=-1, keepdims=True))
    w = av + np.abs(x)
    lo = np.minimum(x.min(axis=-1, keepdims=True), 0.0)
    hi = x.max(axis=-1, keepdims=True)
    span = hi - lo
    flat = span[..., 0] == 0
    span = np.where(flat[..., None], 1.0, span)
    sum_w = w.sum(axis=-1, keepdims=True)
    sum_x = (w * x).sum(axis=-1, keepdims=True)

    iscale = nmax / span
    scale = 1.0 / iscale
    L = np.clip(_rint(iscale * (x - lo)), 0, nmax)
    best = (w * (scale * L + lo - x) ** 2).sum(axis=-1, keepdims=True)
    best_min = lo.copy()
    for step in range(steps + 1):
        iscale = (-1.0 + 0.1 * step + nmax) / span
        La = np.clip(_rint(iscale * (x - lo)), 0, nmax).astype(np.float32)
        sum_l = (w * La).sum(axis=-1, keepdims=True)
        sum_l2 = (w * La * La).sum(axis=-1, keepdims=True)
        sum_xl = (w * La * x).sum(axis=-1, keepdims=True)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        Ds = np.where(ok, D, 1.0)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / Ds
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Ds
        positive = this_min > 0
        this_min = np.where(positive, 0.0, this_min)
        this_scale = np.where(
            positive, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), this_scale
        )
        error = (w * (this_scale * La + this_min - x) ** 2).sum(
            axis=-1, keepdims=True
        )
        better = ok & (error < best)
        L = np.where(better, La.astype(np.int32), L)
        best = np.where(better, error, best)
        scale = np.where(better, this_scale, scale)
        best_min = np.where(better, this_min, best_min)
    scale = np.where(flat[..., None], 0.0, scale)
    return scale[..., 0].astype(np.float32), (-best_min[..., 0]).astype(
        np.float32
    )


def quantize_q4_k(x, steps=20):
    """x float32 [blocks, 256] -> uint8 [blocks, 144]: d, dmin (float16), 12 bytes of 6-bit scales and mins,
    128 bytes of 4-bit quants (units j and j+1 of each 64 share bytes: low and high nibbles)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 8, 32)
    n = x.shape[0]
    scales, mins = _q4k_subblocks(x, steps)  # [n, 8]
    max_scale, max_min = scales.max(axis=1), mins.max(axis=1)
    inv_scale = np.where(
        max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0
    )
    inv_min = np.where(
        max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1.0), 0.0
    )
    ls = np.minimum(63, _rint(inv_scale[:, None] * scales)).astype(np.uint8)
    lm = np.minimum(63, _rint(inv_min[:, None] * mins)).astype(np.uint8)
    packed = np.zeros((n, 12), dtype=np.uint8)
    packed[:, 0:4] = ls[:, 0:4]
    packed[:, 4:8] = lm[:, 0:4]
    packed[:, 8:12] = (ls[:, 4:8] & 0xF) | ((lm[:, 4:8] & 0xF) << 4)
    packed[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    packed[:, 4:8] |= (lm[:, 4:8] >> 4) << 6
    d = (max_scale / 63.0).astype(np.float16)
    dmin = (max_min / 63.0).astype(np.float16)
    sd = d.astype(np.float32)[:, None] * ls.astype(np.float32)
    sm = dmin.astype(np.float32)[:, None] * lm.astype(np.float32)
    safe = np.where(sd != 0, sd, 1.0)[..., None]
    L = np.clip(_rint((x + sm[..., None]) / safe), 0, 15)
    L = np.where((sd != 0)[..., None], L, 0).astype(
        np.uint8
    )  # llama.cpp leaves a zero-scale unit's quants 0
    L = L.reshape(
        n, 4, 2, 32
    )  # 64-value groups: low nibble from the first 32, high from the second
    quants = (L[:, :, 0, :] | (L[:, :, 1, :] << 4)).reshape(n, 128)
    out = np.empty((n, 144), dtype=np.uint8)
    out[:, 0:2] = d.view(np.uint8).reshape(n, 2)
    out[:, 2:4] = dmin.view(np.uint8).reshape(n, 2)
    out[:, 4:16] = packed
    out[:, 16:] = quants
    return out


def _q6k_units(x):
    """make_qx_quants(16, 32, x, rmse_type=1) over x[..., 16]: the scale and quants L in 0..63."""
    nmax = 32
    a = np.abs(x)
    index = a.argmax(axis=-1)
    mx = np.take_along_axis(
        x, index[..., None], axis=-1
    )  # the signed value of largest magnitude
    amax = np.take_along_axis(a, index[..., None], axis=-1)
    tiny = amax < 1e-15
    mx_safe = np.where(tiny, 1.0, mx)
    w = x * x

    def fit(iscale):
        q = np.clip(_rint(iscale * x), -nmax, nmax - 1)
        qf = q.astype(np.float32)
        return (
            q,
            (w * x * qf).sum(axis=-1, keepdims=True),
            (w * qf * qf).sum(axis=-1, keepdims=True),
        )

    L, sumlx, suml2 = fit(-nmax / mx_safe)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1.0), 0.0)
    best = scale * sumlx
    for step in range(-9, 10):
        if step == 0:
            continue
        q, sumlx, suml2 = fit(-(nmax + 0.1 * step) / mx_safe)
        better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        L = np.where(better, q, L)
        new = sumlx / np.where(suml2 > 0, suml2, 1.0)
        scale = np.where(better, new, scale)
        best = np.where(better, new * sumlx, best)
    scale = np.where(tiny, 0.0, scale)
    return scale[..., 0].astype(np.float32)


def quantize_q6_k(x):
    """x float32 [blocks, 256] -> uint8 [blocks, 210]: 128 bytes of low 4 bits, 64 of high 2 bits, 16 int8
    scales, d (float16)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, 16, 16)
    n = x.shape[0]
    scales = _q6k_units(x)  # [n, 16]
    index = np.abs(scales).argmax(axis=1)
    max_scale = scales[np.arange(n), index]
    zero = np.abs(max_scale) < 1e-15
    iscale = np.where(zero, 0.0, -128.0 / np.where(zero, 1.0, max_scale))
    d = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, iscale)).astype(
        np.float16
    )
    q_scales = np.minimum(127, _rint(iscale[:, None] * scales)).astype(np.int8)
    unit = d.astype(np.float32)[:, None] * q_scales.astype(np.float32)
    safe = np.where(unit != 0, unit, 1.0)[..., None]
    L = np.clip(_rint(x / safe), -32, 31) + 32
    L = (
        np.where((unit != 0)[..., None], L, 32)
        .reshape(n, 2, 4, 32)
        .astype(np.uint8)
    )  # [n, half, quarter, 32]
    ql = np.empty((n, 2, 64), dtype=np.uint8)
    ql[:, :, :32] = (L[:, :, 0] & 0xF) | ((L[:, :, 2] & 0xF) << 4)
    ql[:, :, 32:] = (L[:, :, 1] & 0xF) | ((L[:, :, 3] & 0xF) << 4)
    qh = (
        (L[:, :, 0] >> 4)
        | ((L[:, :, 1] >> 4) << 2)
        | ((L[:, :, 2] >> 4) << 4)
        | ((L[:, :, 3] >> 4) << 6)
    )
    out = np.empty((n, 210), dtype=np.uint8)
    out[:, 0:128] = ql.reshape(n, 128)
    out[:, 128:192] = qh.reshape(n, 64)
    out[:, 192:208] = q_scales.view(np.uint8)
    out[:, 208:210] = d.view(np.uint8).reshape(n, 2)
    out[zero] = 0
    return out


def quantize(x, kind, *, steps=20, rows_per_call=4096):
    """Rows of x (float32 [rows, columns]) as GGML `kind` rows: uint8 [rows, row_bytes(kind, columns)]."""
    x = np.asarray(x, dtype=np.float32)
    rows, columns = x.reshape(-1, x.shape[-1]).shape
    width = row_bytes(kind, columns)
    out = np.empty((rows, width), dtype=np.uint8)
    flat = x.reshape(rows, columns)
    for first in range(0, rows, rows_per_call):
        part = flat[first : first + rows_per_call].reshape(-1, QK)
        packed = (
            quantize_q4_k(part, steps)
            if kind == "Q4_K"
            else quantize_q6_k(part)
            if kind == "Q6_K"
            else None
        )
        if packed is None:
            raise ValueError(f"No quantizer for {kind}")
        out[first : first + rows_per_call] = packed.reshape(-1, width)
    return out
