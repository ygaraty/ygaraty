"""Float64 portable convolution kernels; validation and operation dispatch are
separate."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ygaraty.compiler.convolution import OPS as OPS
from ygaraty.compiler.convolution import contract


def _convolve(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    p = k // 2
    x = np.pad(arrays[0].reshape(n, c, h, w), ((0, 0), (0, 0), (p, p), (p, p)))
    weight = arrays[1].reshape(o, c, k, k)
    result = np.broadcast_to(
        arrays[2].reshape(1, o, 1, 1), (n, o, h, w)
    ).copy()
    for ky in range(k):
        for kx in range(k):
            result += np.einsum(
                "nchw,oc->nohw",
                x[:, :, ky : ky + h, kx : kx + w],
                weight[:, :, ky, kx],
            )
    return result


def _input_gradient(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    p = k // 2
    dy = arrays[0].reshape(n, o, h, w)
    weight = arrays[1].reshape(o, c, k, k)
    dx = np.zeros((n, c, h + 2 * p, w + 2 * p))
    for ky in range(k):
        for kx in range(k):
            dx[:, :, ky : ky + h, kx : kx + w] += np.einsum(
                "nohw,oc->nchw",
                dy,
                weight[:, :, ky, kx],
            )
    result = dx[:, :, p : p + h, p : p + w]
    return result


def _weight_gradient(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    p = k // 2
    x = np.pad(arrays[0].reshape(n, c, h, w), ((0, 0), (0, 0), (p, p), (p, p)))
    dy = arrays[1].reshape(n, o, h, w)
    result = np.empty((o, c, k, k))
    for ky in range(k):
        for kx in range(k):
            result[:, :, ky, kx] = np.einsum(
                "nchw,nohw->oc",
                x[:, :, ky : ky + h, kx : kx + w],
                dy,
            )
    return result


def _bias_gradient(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    result = arrays[0].reshape(n, o, h, w).sum(axis=(0, 2, 3))
    return result


def _max_pool(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    x = arrays[0].reshape(n, c, h, w)
    blocks = (
        x.reshape(n, c, h // 2, 2, w // 2, 2)
        .transpose(0, 1, 2, 4, 3, 5)
        .reshape(n, c, h // 2, w // 2, 4)
    )
    if op == "maxpool2d":
        result = blocks.max(axis=-1)
    else:
        winners = blocks.argmax(axis=-1)
        dy = arrays[1].reshape(n, c, h // 2, w // 2)
        result = np.zeros_like(x)
        for index in range(4):
            result[:, :, index // 2 :: 2, index % 2 :: 2] = dy * (
                winners == index
            )
    return result


def _average_pool(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    result = arrays[0].reshape(n, c, h, w).mean(axis=(2, 3))
    return result


def _average_pool_gradient(arrays, dimensions, op):
    n, c, h, w, o, k = dimensions
    result = np.broadcast_to(
        arrays[0].reshape(n, c, 1, 1) / (h * w), (n, c, h, w)
    )
    return result


_OPERATIONS = {
    "conv2d": _convolve,
    "conv2d_input_grad": _input_gradient,
    "conv2d_weight_grad": _weight_gradient,
    "conv2d_bias_grad": _bias_gradient,
    "maxpool2d": _max_pool,
    "avgpool2d": _average_pool,
    "avgpool2d_grad": _average_pool_gradient,
    "maxpool2d_grad": _max_pool,
}


def execute(
    op: str, inputs: Sequence[NDArray], attrs: Mapping[str, object]
) -> NDArray[np.float64]:
    """Validate tensors, dispatch one operation, and return an owned finite
    result."""
    shapes, out, dimensions = contract(op, attrs)
    if len(inputs) != len(shapes):
        raise ValueError("Wrong convolution input count")
    arrays = []
    for value, shape in zip(inputs, shapes):
        a = np.asarray(value)
        if (
            a.shape != shape
            or a.dtype.kind not in "fiu"
            or (not np.isfinite(a).all())
        ):
            raise ValueError("Invalid convolution input")
        arrays.append(a.astype(np.float64, copy=False))
    result = _OPERATIONS[op](arrays, dimensions, op)
    result = np.asarray(result).reshape(out).copy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite convolution output")
    return result
