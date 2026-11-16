"""Canonical inference operation ABI, shared with the generated browser schema.

Effects describe resources conservatively; they do not prove allocation
or scheduling safety.
No numerical implementation lives here.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

SCHEMA_VERSION = 1
STATE_CONTRACT = {
    "identity": ["stage_instance", "layer", "slot", "name"],
    "position_commit": "after_success",
    "failure": "buffers_may_be_modified",
    "reset": "adapter_owned",
    "scheduling": "ordered_effects",
}
VARIABLES = {
    **dict.fromkeys(
        (
            "tokens",
            "count",
            "position",
            "capacity",
            "head_tokens",
            "head_offset",
            "vocab",
            "chunk_first",
            "row",
        ),
        "integer",
    ),
    **dict.fromkeys(("head_all", "head_enabled", "batch"), "boolean"),
    "kv": "precision",
}
LOOPS = ("for_each_sequence", "for_each_chunk", "for_each_head_row")


@dataclass(frozen=True)
class Argument:
    name: str
    kind: str
    access: str = "none"
    precision: str = "none"
    optional: bool = False


def arg(name, kind="integer", access="none", precision="none", optional=False):
    return Argument(name, kind, access, precision, optional)


def buf(name, access="read", precision="f32"):
    return arg(name, "buffer", access, precision)


def state(name, access="read", precision="kv"):
    return arg(name, "state", access, precision)


N = arg("tokens")
POS = arg("position")
CAP = arg("capacity")
KV = arg("cache_precision", "precision")
HEADS = arg("heads")
DIM = arg("head_dim")
WEIGHT = arg("weight", "weight", "read", "weight")
OPTIONS = arg("options", "options", optional=True)
ATTENTION = (
    buf("query"),
    state("keys"),
    state("values"),
    buf("output", "write"),
    N,
    POS,
    CAP,
    KV,
    HEADS,
    arg("kv_heads"),
    DIM,
    arg("scale", "number"),
    buf("partials", "scratch"),
    buf("statistics", "scratch"),
)
OPERATIONS = {
    "embedding": (
        arg("weights", "weights", "read", "weight"),
        buf("ids", precision="u32"),
        buf("output", "write"),
        N,
    ),
    "rms_norm": (
        buf("input"),
        arg("weight", "vector", "read", "f32"),
        buf("output", "write"),
        N,
        arg("options", "options"),
    ),
    "matmul": (WEIGHT, buf("input"), buf("output", "write"), N, OPTIONS),
    "silu_mul": (buf("gate"), buf("output", "read_write"), arg("elements")),
    "argmax": (buf("input"), buf("output", "write", "u32"), arg("elements")),
    "causal_conv1d": (
        buf("input"),
        arg("weight", "layer_buffer", "read", "f32"),
        state("history", "read_write", "f32"),
        buf("output", "write"),
        N,
    ),
    "gated_delta": (
        buf("input"),
        buf("gates"),
        arg("weight", "layer_buffer", "read", "f32"),
        state("recurrence", "read_write", "f32"),
        buf("output", "write"),
        N,
    ),
    "rotary_partial": (
        buf("input", "read_write"),
        buf("frequencies"),
        N,
        HEADS,
        arg("stride"),
        DIM,
        POS,
    ),
    "rotary_adjacent": (
        buf("input", "read_write"),
        buf("frequencies"),
        N,
        HEADS,
        DIM,
        POS,
    ),
    "rotary_halves": (
        buf("input", "read_write"),
        buf("frequencies"),
        N,
        HEADS,
        DIM,
        POS,
    ),
    "cache_store": (buf("input"), state("cache", "write"), N, POS, CAP, KV),
    "cache_store_grouped": (
        buf("input"),
        state("cache", "write"),
        N,
        POS,
        CAP,
        KV,
        arg("kv_heads"),
        DIM,
    ),
    "grouped_attention": ATTENTION,
    "windowed_attention": (*ATTENTION, arg("window")),
    "gated_attention": (
        buf("query"),
        state("keys"),
        state("values"),
        buf("gate"),
        buf("output", "write"),
        N,
        POS,
        CAP,
        KV,
        buf("partials", "scratch"),
        buf("statistics", "scratch"),
    ),
    "add": (buf("input"), buf("output", "read_write"), arg("elements")),
    "moe_route": (
        buf("logits"),
        arg("bias", "vector", "read", "f32"),
        buf("ids", "write", "u32"),
        buf("weights", "write"),
        N,
        arg("experts"),
        arg("top_k"),
    ),
    # DO-80 (Qwen3-MoE): the top k of a softmax over all experts' logits, the
    # chosen probabilities renormalized to sum 1 when `normalize`; ties to
    # the lower expert index.
    "moe_route_softmax": (
        buf("logits"),
        buf("ids", "write", "u32"),
        buf("weights", "write"),
        N,
        arg("experts"),
        arg("top_k"),
        arg("normalize", "boolean"),
    ),
    "moe_matmul": (
        arg("weights", "layer_buffer", "read", "weight"),
        buf("input"),
        buf("output", "write"),
        buf("ids", precision="u32"),
        N,
        arg("top_k"),
        arg("per_slot", "boolean"),
    ),
    "silu_mul_halves": (
        buf("input"),
        buf("output", "write"),
        N,
        arg("width"),
    ),
    "moe_combine": (
        buf("input"),
        buf("weights"),
        buf("output", "read_write"),
        N,
        arg("top_k"),
        arg("width"),
    ),
}
# Options are named, typed, and closed to prevent silently ignored
# misspellings.
OPTION_FIELDS = {
    "matmul": {
        **dict.fromkeys(
            ("xStride", "xOffset", "yStride", "yOffset", "rows", "rowBase"),
            "integer",
        ),
        "accumulate": "boolean",
    },
    "rms_norm": {
        **dict.fromkeys(
            (
                "size",
                "groups",
                "inStride",
                "inGroup",
                "inOffset",
                "outStride",
                "outGroup",
                "outOffset",
                "gateStride",
                "gateGroup",
                "gateOffset",
            ),
            "integer",
        ),
        "eps": "number",
        "gate": "buffer",
    },
}
BINDINGS = {
    "buffer": {"buffer"},
    "state": {"state"},
    "weight": {"matrix", "resource"},
    "weights": {"resource"},
    "vector": {"vector", "buffer"},
    "layer_buffer": {"layer_buffer"},
}


def schema():
    return dict(
        version=SCHEMA_VERSION,
        loops=list(LOOPS),
        state=STATE_CONTRACT,
        variables=VARIABLES,
        operations={
            op: [asdict(a) for a in args] for op, args in OPERATIONS.items()
        },
        options=OPTION_FIELDS,
        bindings={kind: sorted(values) for kind, values in BINDINGS.items()},
    )


def _expression(value, kind, op, depth=0):
    if depth > 20:
        raise ValueError("Plan expression nesting limit")
    if isinstance(value, dict) and "$" in value:
        tag = value["$"]
        if tag == "choose":
            if set(value) != {"$", "condition", "yes", "no"} or value[
                "condition"
            ] not in ("batch", "head_all"):
                raise ValueError("Invalid choice expression")
            for branch in ("yes", "no"):
                _expression(value[branch], kind, op, depth + 1)
            return
        if tag == "var" and kind in (
            "integer",
            "number",
            "boolean",
            "precision",
        ):
            name = value.get("name")
            expected = VARIABLES.get(name) if isinstance(name, str) else None
            if set(value) != {"$", "name"} or not (
                expected == kind or kind == "number" and expected == "integer"
            ):
                raise ValueError("Invalid runtime variable")
            return
        if tag == "product" and kind in ("integer", "number"):
            if (
                set(value) != {"$", "values"}
                or not isinstance(value["values"], list)
                or not value["values"]
            ):
                raise ValueError("Invalid product expression")
            for item in value["values"]:
                _expression(item, kind, op, depth + 1)
            return
        if tag not in BINDINGS.get(kind, set()):
            raise ValueError(f"Invalid {kind} binding: {tag}")
        allowed = {"$", "name"} | (
            {"stride", "view"} if tag == "buffer" else set()
        )
        if (
            set(value) - allowed
            or not isinstance(value.get("name"), str)
            or not value["name"]
        ):
            raise ValueError("Invalid resource binding")
        if "stride" in value:
            if (
                type(value["stride"]) is not int
                or not 0 < value["stride"] <= 2**53 - 1
                or value.get("view") not in ("batch", "always", "head")
            ):
                raise ValueError("Invalid buffer view")
        elif "view" in value:
            raise ValueError("Buffer view needs a stride")
        return
    if kind == "options" and isinstance(value, dict):
        fields = OPTION_FIELDS[op]
        if set(value) - fields.keys() or (
            op == "rms_norm" and "size" not in value
        ):
            raise ValueError("Invalid operation options")
        for name, item in value.items():
            _expression(item, fields[name], op, depth + 1)
        return
    valid = (
        kind == "boolean"
        and type(value) is bool
        or kind == "integer"
        and type(value) is int
        and 0 <= value <= 2**53 - 1
        or kind == "number"
        and type(value) in (int, float)
        and isfinite(value)
        or kind == "precision"
        and value in ("f16", "f32")
    )
    if not valid:
        raise ValueError(f"Invalid {kind} argument")


def _resources(value, path):
    if not isinstance(value, dict):
        return
    if value.get("$") == "choose":
        for branch in ("yes", "no"):
            yield from _resources(value[branch], f"{path}.{branch}")
    elif value.get("$") in {
        "buffer",
        "matrix",
        "vector",
        "state",
        "layer_buffer",
        "resource",
    }:
        yield path, value
    elif "$" not in value:
        for key, item in value.items():
            yield from _resources(item, f"{path}.{key}")


def operation_contract(op: str, call: list[Any], layer: int | None):
    """Validate symbolic arguments and describe resource effects."""
    if layer is not None and (type(layer) is not int or layer < 0):
        raise ValueError("Invalid layer resource")
    if op in LOOPS:
        if call:
            raise ValueError("Loop has primitive arguments")
        return dict(schema=SCHEMA_VERSION, arguments=[], effects=[])
    if op not in OPERATIONS or not isinstance(call, list):
        raise ValueError(f"Unsupported inference operation: {op}")
    args = OPERATIONS[op]
    minimum = sum(not a.optional for a in args)
    if not minimum <= len(call) <= len(args):
        raise ValueError(f"Invalid {op} arity")
    effects = []
    for a, value in zip(args, call):
        _expression(value, a.kind, op)
        for path, binding in _resources(value, a.name):
            tag = binding["$"]
            if (
                tag in ("matrix", "vector", "layer_buffer", "state")
                and layer is None
            ):
                raise ValueError("Layer resource without a layer")
            access = "read" if a.kind == "options" else a.access
            # Conservatively include the read when accumulation is dynamic.
            if op == "matmul" and a.name == "output" and len(call) == 5:
                options = call[4]
                if options.get("$") == "choose" or options.get(
                    "accumulate", False
                ):
                    access = "read_write"
            persistent = tag == "state"
            lifetime = (
                "session"
                if persistent
                else "checkpoint"
                if tag != "buffer" or a.name in ("frequencies", "weight")
                else "invocation"
            )
            effects.append(
                dict(
                    argument=path,
                    binding=deepcopy(binding),
                    access=access,
                    precision=a.precision if a.kind != "options" else "f32",
                    lifetime="operation" if access == "scratch" else lifetime,
                    scope="layer_sequence" if persistent else "stage",
                    transition=(
                        "append"
                        if op.startswith("cache_store")
                        else "update"
                        if access == "read_write"
                        else "read"
                    )
                    if persistent
                    else "none",
                )
            )
    return dict(
        schema=SCHEMA_VERSION,
        arguments=[a.name for a in args[: len(call)]],
        effects=effects,
    )


def legacy_stage(plan):
    """Project a canonical v3 stage to its exact prior v2 representation."""
    result = deepcopy(plan)
    result["version"] = 2
    result.pop("contract_version", None)
    result.pop("state_contract", None)

    def strip(tasks):
        for task in tasks:
            task["attrs"].pop("contract", None)
            strip(task["attrs"].get("body", []))

    strip(result["tasks"])
    return result
