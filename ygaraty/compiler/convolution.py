"""Portable NCHW operation contracts, represented as rank-two tensors."""

from collections.abc import Mapping
from dataclasses import dataclass

Shape = tuple[int, int]
MAX_TENSOR_BYTES = 64 << 20
OPS = (
    "conv2d",
    "conv2d_input_grad",
    "conv2d_weight_grad",
    "conv2d_bias_grad",
    "maxpool2d",
    "maxpool2d_grad",
    "avgpool2d",
    "avgpool2d_grad",
)


@dataclass(frozen=True)
class ConvolutionSpec:
    """Validated dimensions shared by shape inference and numerical
    execution.
    """

    batch: int
    channels: int
    height: int
    width: int
    outputs: int
    kernel: int

    @classmethod
    def from_attributes(cls, attrs: Mapping[str, object]) -> "ConvolutionSpec":
        values = []
        for name, maximum in (
            ("batch", 64),
            ("in_channels", 256),
            ("height", 128),
            ("width", 128),
        ):
            value = attrs.get(name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid " + name)
            values.append(value)
        outputs = attrs.get("out_channels", values[1])
        kernel = attrs.get("kernel", 3)
        if (
            type(outputs) is not int
            or not 1 <= outputs <= 256
            or type(kernel) is not int
            or kernel not in (1, 3)
        ):
            raise ValueError("Invalid output channels or kernel")
        return cls(*values, outputs, kernel)

    def dimensions(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.batch,
            self.channels,
            self.height,
            self.width,
            self.outputs,
            self.kernel,
        )

    def shapes(self, operation: str) -> tuple[list[Shape], Shape]:
        n, c, h, w, o, k = self.dimensions()
        image, output, weight = (n, c * h * w), (n, o * h * w), (o, c * k * k)
        convolution = {
            "conv2d": ([image, weight, (1, o)], output),
            "conv2d_input_grad": ([output, weight], image),
            "conv2d_weight_grad": ([image, output], weight),
            "conv2d_bias_grad": ([output], (1, o)),
        }
        if operation in convolution:
            return convolution[operation]
        if operation in ("maxpool2d", "maxpool2d_grad"):
            if h % 2 or w % 2:
                raise ValueError(
                    "Max pooling requires even spatial dimensions"
                )
            pooled = (n, c * (h // 2) * (w // 2))
            return (
                ([image], pooled)
                if operation == "maxpool2d"
                else ([image, pooled], image)
            )
        if operation == "avgpool2d":
            return [image], (n, c)
        if operation == "avgpool2d_grad":
            return [(n, c)], image
        raise ValueError("Unknown convolution operation")


def contract(op: str, attrs: Mapping[str, object]):
    """Return legacy input/output shapes and dimensions with the same
    limits.
    """
    if op not in OPS:
        raise ValueError("Unknown convolution operation")
    spec = ConvolutionSpec.from_attributes(attrs)
    inputs, output = spec.shapes(op)
    if any(
        rows * columns * 8 > MAX_TENSOR_BYTES
        for rows, columns in [*inputs, output]
    ):
        raise ValueError("Convolution tensor exceeds 64 MiB")
    return inputs, output, spec.dimensions()
