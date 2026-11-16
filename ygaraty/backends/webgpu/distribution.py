"""Lower shape-only inference Tape operations to browser stage tasks.

Version 3 adds schema-checked arguments and explicit resource effects.
Rows bind to the current prefill chunk or decode batch at runtime.
"""

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from typing import Any

from .graph import Input, Task
from .graph import ShapeTape as Tape
from .operations import (
    OPERATIONS,
    SCHEMA_VERSION,
    STATE_CONTRACT,
    legacy_stage,
    operation_contract,
)

# Manifests and task attributes are JSON objects at the compiler boundary.
JsonObject = dict[str, Any]


# Wire values shared with the browser executor. Keep these values stable.
class Operation(StrEnum):
    EMBEDDING = "embedding"
    RMS_NORM = "rms_norm"
    MATMUL = "matmul"
    SILU_MUL = "silu_mul"
    CAUSAL_CONV1D = "causal_conv1d"
    GATED_DELTA = "gated_delta"
    ROTARY_ADJACENT = "rotary_adjacent"
    ROTARY_HALVES = "rotary_halves"
    ROTARY_PARTIAL = "rotary_partial"
    CACHE_STORE_GROUPED = "cache_store_grouped"
    CACHE_STORE = "cache_store"
    GROUPED_ATTENTION = "grouped_attention"
    GATED_ATTENTION = "gated_attention"
    FOR_EACH_SEQUENCE = "for_each_sequence"
    FOR_EACH_CHUNK = "for_each_chunk"
    FOR_EACH_HEAD_ROW = "for_each_head_row"
    ARGMAX = "argmax"
    # Kolibri-1 (`kolibri1`): residual add, sliding-window attention and the
    # mixture of experts.
    ADD = "add"
    WINDOWED_ATTENTION = "windowed_attention"
    MOE_ROUTE = "moe_route"
    MOE_ROUTE_SOFTMAX = "moe_route_softmax"
    MOE_MATMUL = "moe_matmul"
    SILU_MUL_HALVES = "silu_mul_halves"
    MOE_COMBINE = "moe_combine"


class Binding(StrEnum):
    VAR = "var"
    BUFFER = "buffer"
    VECTOR = "vector"
    MATRIX = "matrix"
    LAYER_BUFFER = "layer_buffer"
    STATE = "state"
    RESOURCE = "resource"
    PRODUCT = "product"
    CHOOSE = "choose"


class RuntimeVar(StrEnum):
    BATCH = "batch"
    TOKENS = "tokens"
    COUNT = "count"
    POSITION = "position"
    CAPACITY = "capacity"
    KV = "kv"
    HEAD_TOKENS = "head_tokens"
    HEAD_ALL = "head_all"
    HEAD_ENABLED = "head_enabled"
    HEAD_OFFSET = "head_offset"
    VOCAB = "vocab"
    CHUNK_FIRST = "chunk_first"


class Buffer(StrEnum):
    RESIDUAL = "x"
    NORMALIZED = "h"
    QKV = "big"
    CONVOLVED = "big2"
    ATTENTION_HIDDEN = "mid"
    GATE = "z"
    RECURRENCE_GATES = "ab"
    QUERY = "q"
    KEY = "k"
    NORMALIZED_KEY = "kn"
    VALUE = "v"
    ATTENTION_OUTPUT = "att"
    MLP_GATE = "f1"
    MLP_UP = "f2"
    ROTARY_FREQUENCIES = "invFreq"
    ATTENTION_PARTIALS = "part"
    ATTENTION_STATS = "ml"
    OUTPUT_NORM = "outputNorm"
    LOGITS = "logits"
    TOKEN = "token"
    TOKEN_IDS = "ids"
    NORMALIZED_QUERY = "qn"
    ROUTER_LOGITS = "router"
    EXPERT_IDS = "experts"
    EXPERT_WEIGHTS = "expertWeights"
    EXPERT_GATE_UP = "eg"
    EXPERT_ACTIVATION = "ea"
    EXPERT_OUTPUT = "ed"


class Weight(StrEnum):
    ATTENTION_NORM = "attn_norm.weight"
    QKV = "attn_qkv.weight"
    ATTENTION_GATE = "attn_gate.weight"
    RECURRENCE_ALPHA = "ssm_alpha.weight"
    RECURRENCE_BETA = "ssm_beta.weight"
    RECURRENCE_NORM = "ssm_norm.weight"
    RECURRENCE_OUTPUT = "ssm_out.weight"
    QUERY = "attn_q.weight"
    KEY = "attn_k.weight"
    VALUE = "attn_v.weight"
    QUERY_NORM = "attn_q_norm.weight"
    KEY_NORM = "attn_k_norm.weight"
    ATTENTION_OUTPUT = "attn_output.weight"
    MLP_NORM = "ffn_norm.weight"
    POST_ATTENTION_NORM = "post_attention_norm.weight"
    MLP_GATE = "ffn_gate.weight"
    MLP_UP = "ffn_up.weight"
    MLP_DOWN = "ffn_down.weight"
    ATTENTION_POST_NORM = "attn_post_norm.weight"
    MLP_POST_NORM = "ffn_post_norm.weight"
    ROUTER = "ffn_gate_inp.weight"
    ROUTER_BIAS = "exp_probs_b.bias"
    SHARED_GATE = "ffn_gate_shexp.weight"
    SHARED_UP = "ffn_up_shexp.weight"
    SHARED_DOWN = "ffn_down_shexp.weight"


class Family(StrEnum):
    QWEN = "qwen35"
    DENSE = "dense"
    KOLIBRI = "kolibri1"
    QWEN3MOE = "qwen3moe"


class RopePairs(StrEnum):
    ADJACENT = "adjacent"
    HALVES = "halves"


class LayerKind(StrEnum):
    LINEAR = "linear"
    FULL = "full"
    DENSE = "dense"
    SLIDING = "sliding"


class View(StrEnum):
    BATCH = "batch"
    ALWAYS = "always"
    HEAD = "head"


class State(StrEnum):
    CONVOLUTION = "convStates"
    RECURRENCE = "Ss"
    KEY = "Ks"
    VALUE = "Vs"


class Resource(StrEnum):
    EMBEDDING = "embed"
    OUTPUT_CHUNK = "chunk"
    CONVOLUTION = "conv"
    RECURRENCE_HEAD = "head"
    # A layer's expert chunks: [{first, count, matrix}], gate/up and down.
    EXPERTS_GATE_UP = "gateUp"
    EXPERTS_DOWN = "down"
    # Qwen3-MoE stores gate and up as separate expert tensors (DO-80).
    EXPERTS_GATE = "gate"
    EXPERTS_UP = "up"


class Value(StrEnum):
    TOKENS = "tokens"
    EMBEDDING = "embedding"
    NEXT_TOKEN = "next_token"


PLAN_VERSION = 3
SUBMIT_TOKEN_LAYERS = 64
QWEN_HEAD = 256  # Qwen3.5 attention head width
QWEN_LINEAR_HEAD = 128  # Qwen3.5 DeltaNet key/value head width


def gate_stride(linear_v: int, aligned: bool = True) -> int:
    """Floats per token row of the DeltaNet gates buffer (a then b).

    Multi-slot decode binds one row per sequence, and WebGPU storage binding
    offsets must be multiples of 256 bytes (64 floats). 2 x 32 heads (9B) is
    aligned; 2 x 16 (0.8B) and 2 x 48 (Qwen3.8-27B) are padded. Bundles
    prepared before the padding stored unaligned plans (``aligned=False``).
    """
    width = 2 * linear_v
    return -(-width // 64) * 64 if aligned else width


def _stage_layout(
    manifest: Mapping[str, Any], owner: int
) -> tuple[str, int, str]:
    config = manifest.get("config") or {}
    family = config.get("family", Family.QWEN)
    if family not in (
        Family.QWEN,
        Family.DENSE,
        Family.KOLIBRI,
        Family.QWEN3MOE,
    ):
        raise ValueError("Unsupported inference plan family")

    hidden = config.get("hidden", 4096)
    if type(hidden) is not int or hidden <= 0:
        raise ValueError("Invalid hidden width")

    layers = manifest["layers"]
    if not layers:
        raise ValueError("An inference stage needs layers")

    first, last = layers[0]["index"], layers[-1]["index"] + 1
    if [layer["index"] for layer in layers] != list(range(first, last)):
        raise ValueError("Stage layers must be contiguous")

    if type(owner) is not int or owner < 0:
        raise ValueError("Invalid stage owner")

    kinds = {
        Family.DENSE: (LayerKind.DENSE,),
        Family.KOLIBRI: (LayerKind.SLIDING, LayerKind.FULL),
        Family.QWEN3MOE: (LayerKind.FULL,),
    }.get(family, (LayerKind.LINEAR, LayerKind.FULL))
    for layer in layers:
        if layer["type"] not in kinds:
            raise ValueError("Layer kind differs from plan family")

    incoming = Value.TOKENS if manifest.get("embed") else f"hidden_{first - 1}"

    return family, hidden, incoming


def ref(kind: Binding, name: str | None = None, **attrs: Any) -> JsonObject:
    expression = {"$": kind, **attrs}
    if name is not None:
        expression["name"] = name

    return expression


def variable(name: RuntimeVar) -> JsonObject:
    return ref(Binding.VAR, name)


def buffer(
    name: str, stride: int | None = None, view: View = View.BATCH
) -> JsonObject:
    return ref(
        Binding.BUFFER,
        name,
        **({"stride": stride, "view": view} if stride is not None else {}),
    )


class Program:
    """An ordered Tape over explicit persistent and scratch buffer bindings."""

    def __init__(
        self, incoming: str, hidden: int, owner: int, layer: int | None = None
    ) -> None:
        self.tape = Tape()
        self.tape.shapes[incoming] = (1, hidden)

        self.value = incoming
        self.hidden = hidden
        self.owner = owner
        self.layer = layer

    def emit(
        self,
        op: Operation,
        *args: Any,
        width: int | None = None,
        name: str | None = None,
        **attrs: Any,
    ) -> str:
        contract = operation_contract(op, list(args), self.layer)
        self.value = self.tape._op(
            op,
            [self.value],
            (1, self.hidden if width is None else width),
            dict(
                owner=self.owner,
                layer=self.layer,
                call=list(args),
                contract=contract,
                **attrs,
            ),
            name,
        )

        return self.value

    def invoke(
        self, op: Operation, arguments: Mapping[str, Any], **attrs: Any
    ) -> str:
        """Emit a primitive using schema names instead of positions."""
        specs = OPERATIONS[op]
        if set(arguments) - {a.name for a in specs}:
            raise ValueError("Unknown operation argument")
        call = []
        for a in specs:
            if a.name not in arguments:
                if not a.optional:
                    raise ValueError(f"Missing operation argument: {a.name}")
                continue
            call.append(arguments[a.name])
        return self.emit(op, *call, **attrs)

    def body(self, *, hidden: int | None = None) -> "Program":
        """Start a nested scope at the current dependency frontier."""
        return Program(
            self.value,
            self.hidden if hidden is None else hidden,
            self.owner,
            self.layer,
        )

    def loop(
        self, operation: Operation, body: "Program", *, name: str | None = None
    ) -> None:
        self.emit(
            operation,
            body=body.tasks(),
            width=body.tape.shapes[body.value][1],
            name=name,
        )

    def norm(self, src: str, weight: str, dst: str, **options: Any) -> None:
        self.emit(
            Operation.RMS_NORM,
            buffer(src),
            ref(Binding.VECTOR, weight),
            buffer(dst),
            variable(RuntimeVar.TOKENS),
            options,
            width=options.get(
                "outStride", options["size"] * options.get("groups", 1)
            ),
        )

    def project(
        self,
        weight: str,
        src: str,
        dst: str,
        *,
        width: int,
        name: str | None = None,
        layer_end: bool = False,
        **options: Any,
    ) -> None:
        args = [
            ref(Binding.MATRIX, weight),
            buffer(src),
            buffer(dst),
            variable(RuntimeVar.TOKENS),
        ]
        if options:
            args.append(options)

        boundary = {"layer_end": True} if layer_end else {}

        self.emit(Operation.MATMUL, *args, width=width, name=name, **boundary)

    def tasks(self) -> list[JsonObject]:
        tasks = []
        for node in self.tape.nodes:
            task = Task(
                node.id,
                node.op,
                [Input(value) for value in node.inputs],
                node.shape,
                node.attrs,
            ).describe()
            task["shape"] = list(node.shape)
            tasks.append(task)

        return tasks


class LayerComposition:
    """Compose one layer using common operations and explicit geometry.

    Architecture choices belong here; Program only builds ordered Tape nodes.
    No callbacks capture a partially constructed layer.
    """

    def __init__(self, program: Program, config: Mapping[str, Any]) -> None:
        self.program = program
        self.config = config
        self.hidden = config["hidden"]
        self.intermediate = config["intermediate"]

        self.dense = config.get("family") == Family.DENSE
        self.norm_options = {"eps": config["eps"]} if self.dense else {}
        if not self.dense:
            # Qwen3.5 geometry; the defaults are the 9B's, which manifests
            # prepared before these fields existed assume.
            # Distinct keys: dense configs use "heads"/"kv_heads".
            self.heads = config.get("attention_heads", 16)
            self.kv_heads = config.get("attention_kv_heads", 4)
            linear_k = config.get("linear_k_heads", 16)
            self.linear_v = config.get("linear_v_heads", 32)
            self.value_dim = self.linear_v * QWEN_LINEAR_HEAD
            self.conv_dim = 2 * linear_k * QWEN_LINEAR_HEAD + self.value_dim
            self.gates = gate_stride(
                self.linear_v, not config.get("_unaligned_gates")
            )

    def compose(self, layer: Mapping[str, Any]) -> None:
        self.program.layer = layer["index"]
        self.program.norm(
            Buffer.RESIDUAL,
            Weight.ATTENTION_NORM,
            Buffer.NORMALIZED,
            size=self.hidden,
            **self.norm_options,
        )

        if layer["type"] == LayerKind.LINEAR:
            self.linear_attention()
        else:
            self.full_attention()

        self.mlp()

    def linear_attention(self) -> None:
        """Convolution and recurrent state update, followed by gated norm."""
        p = self.program

        conv, value, heads = self.conv_dim, self.value_dim, self.linear_v
        p.project(Weight.QKV, Buffer.NORMALIZED, Buffer.QKV, width=conv)
        p.project(
            Weight.ATTENTION_GATE, Buffer.NORMALIZED, Buffer.GATE, width=value
        )
        p.project(
            Weight.RECURRENCE_ALPHA,
            Buffer.NORMALIZED,
            Buffer.RECURRENCE_GATES,
            yStride=self.gates,
            yOffset=0,
            width=heads,
        )
        p.project(
            Weight.RECURRENCE_BETA,
            Buffer.NORMALIZED,
            Buffer.RECURRENCE_GATES,
            yStride=self.gates,
            yOffset=heads,
            width=heads,
        )

        body = p.body()
        body.emit(
            Operation.CAUSAL_CONV1D,
            buffer(Buffer.QKV, conv),
            ref(Binding.LAYER_BUFFER, Resource.CONVOLUTION),
            ref(Binding.STATE, State.CONVOLUTION),
            buffer(Buffer.CONVOLVED, conv),
            variable(RuntimeVar.COUNT),
            width=conv,
        )
        body.emit(
            Operation.GATED_DELTA,
            buffer(Buffer.CONVOLVED, conv),
            buffer(Buffer.RECURRENCE_GATES, self.gates),
            ref(Binding.LAYER_BUFFER, Resource.RECURRENCE_HEAD),
            ref(Binding.STATE, State.RECURRENCE),
            buffer(Buffer.ATTENTION_HIDDEN, value),
            variable(RuntimeVar.COUNT),
            width=value,
        )

        p.loop(Operation.FOR_EACH_SEQUENCE, body)

        p.norm(
            Buffer.ATTENTION_HIDDEN,
            Weight.RECURRENCE_NORM,
            Buffer.NORMALIZED,
            size=QWEN_LINEAR_HEAD,
            groups=heads,
            inStride=value,
            outStride=value,
            gate=buffer(Buffer.GATE),
            gateStride=value,
            gateGroup=QWEN_LINEAR_HEAD,
        )
        p.project(
            Weight.RECURRENCE_OUTPUT,
            Buffer.NORMALIZED,
            Buffer.RESIDUAL,
            accumulate=True,
            width=self.hidden,
        )

    def full_attention(self) -> None:
        """Project Q/K/V, update each sequence cache, then attend."""
        p = self.program
        dense = self.dense
        heads, kv, dim = (
            (
                self.config["heads"],
                self.config["kv_heads"],
                self.config["head_dim"],
            )
            if dense
            else (self.heads, self.kv_heads, QWEN_HEAD)
        )
        q, k, out = (
            (Buffer.QUERY, Buffer.KEY, Buffer.ATTENTION_OUTPUT)
            if dense
            else (
                Buffer.ATTENTION_HIDDEN,
                Buffer.NORMALIZED_KEY,
                Buffer.NORMALIZED,
            )
        )

        p.project(
            Weight.QUERY,
            Buffer.NORMALIZED,
            Buffer.QUERY if dense else Buffer.QKV,
            width=heads * dim * (1 if dense else 2),
        )
        p.project(Weight.KEY, Buffer.NORMALIZED, Buffer.KEY, width=kv * dim)
        p.project(
            Weight.VALUE, Buffer.NORMALIZED, Buffer.VALUE, width=kv * dim
        )

        if not dense:
            p.norm(
                Buffer.QKV,
                Weight.QUERY_NORM,
                Buffer.ATTENTION_HIDDEN,
                size=dim,
                groups=heads,
                inStride=2 * heads * dim,
                inGroup=2 * dim,
                outStride=heads * dim,
            )
            p.norm(
                Buffer.KEY,
                Weight.KEY_NORM,
                Buffer.NORMALIZED_KEY,
                size=dim,
                groups=kv,
            )

        body = p.body()
        view = View.ALWAYS if dense else View.BATCH
        rotary = Operation.ROTARY_PARTIAL
        if dense:
            pairs = self.config.get("rope_pairs", RopePairs.ADJACENT)
            if pairs not in (RopePairs.ADJACENT, RopePairs.HALVES):
                raise ValueError(f"Unsupported RoPE pairing: {pairs}")
            rotary = (
                Operation.ROTARY_HALVES
                if pairs == RopePairs.HALVES
                else Operation.ROTARY_ADJACENT
            )

        def row_buffer(name: str, stride: int) -> JsonObject:
            return buffer(name, stride, view)

        count, pos = variable(RuntimeVar.COUNT), variable(RuntimeVar.POSITION)

        for source, head_count in ((q, heads), (k, kv)):
            args = [
                row_buffer(source, head_count * dim),
                buffer(Buffer.ROTARY_FREQUENCIES),
                count,
                head_count,
            ]
            if not dense:
                args.append(head_count * dim)

            body.emit(
                rotary,
                *args,
                dim,
                pos,
            )

        for src, state in ((k, State.KEY), (Buffer.VALUE, State.VALUE)):
            args = [
                row_buffer(src, kv * dim),
                ref(Binding.STATE, state),
                count,
                pos,
                variable(RuntimeVar.CAPACITY),
                variable(RuntimeVar.KV),
            ]
            body.emit(
                Operation.CACHE_STORE_GROUPED
                if dense
                else Operation.CACHE_STORE,
                *(args + ([kv, dim] if dense else [])),
            )

        if dense:
            body.emit(
                Operation.GROUPED_ATTENTION,
                row_buffer(q, heads * dim),
                ref(Binding.STATE, State.KEY),
                ref(Binding.STATE, State.VALUE),
                row_buffer(out, heads * dim),
                count,
                pos,
                variable(RuntimeVar.CAPACITY),
                variable(RuntimeVar.KV),
                heads,
                kv,
                dim,
                self.config["attention_scale"],
                buffer(Buffer.ATTENTION_PARTIALS),
                buffer(Buffer.ATTENTION_STATS),
            )
        else:
            body.emit(
                Operation.GATED_ATTENTION,
                row_buffer(q, heads * dim),
                ref(Binding.STATE, State.KEY),
                ref(Binding.STATE, State.VALUE),
                row_buffer(Buffer.QKV, 2 * heads * dim),
                row_buffer(out, heads * dim),
                count,
                pos,
                variable(RuntimeVar.CAPACITY),
                variable(RuntimeVar.KV),
                buffer(Buffer.ATTENTION_PARTIALS),
                buffer(Buffer.ATTENTION_STATS),
            )

        p.loop(Operation.FOR_EACH_SEQUENCE, body)

        p.project(
            Weight.ATTENTION_OUTPUT,
            out,
            Buffer.RESIDUAL,
            accumulate=True,
            width=self.hidden,
        )

    def mlp(self) -> None:
        """Pre-normalized gated MLP shared by both attention architectures."""
        p = self.program
        hidden, intermediate = self.hidden, self.intermediate
        dense, eps = self.dense, self.norm_options

        tokens = variable(RuntimeVar.TOKENS)

        p.norm(
            Buffer.RESIDUAL,
            Weight.MLP_NORM if dense else Weight.POST_ATTENTION_NORM,
            Buffer.NORMALIZED,
            size=hidden,
            **eps,
        )

        p.project(
            Weight.MLP_GATE,
            Buffer.NORMALIZED,
            Buffer.MLP_GATE,
            width=self.intermediate,
        )
        p.project(
            Weight.MLP_UP,
            Buffer.NORMALIZED,
            Buffer.MLP_UP,
            width=self.intermediate,
        )

        p.emit(
            Operation.SILU_MUL,
            buffer(Buffer.MLP_GATE),
            buffer(Buffer.MLP_UP),
            ref(Binding.PRODUCT, values=[tokens, intermediate]),
            width=intermediate,
        )

        p.project(
            Weight.MLP_DOWN,
            Buffer.MLP_UP,
            Buffer.RESIDUAL,
            width=hidden,
            accumulate=True,
            name=f"hidden_{p.layer}",
            layer_end=True,
        )


class KolibriComposition:
    """Kolibri-1: per-head Q/K norms, RoPE only on sliding-window layers
    (the global layers use no position encoding), norms after attention and
    after the MLP, and a mixture of experts beside one shared expert.

    The router selects experts on the GPU, so the expert products read their
    expert ids from a buffer (`moe_matmul`); every other step is an ordinary
    projection, norm or loop over sequences.
    """

    def __init__(self, program: Program, config: Mapping[str, Any]) -> None:
        self.program = program
        self.config = config
        self.hidden = config["hidden"]
        self.eps = {"eps": config["eps"]}

    def compose(self, layer: Mapping[str, Any]) -> None:
        self.program.layer = layer["index"]
        self.attention(layer["type"] == LayerKind.SLIDING)
        self.mixture()

    def attention(self, sliding: bool) -> None:
        self.attention_core(
            rope=sliding, window=self.config["window"] if sliding else 0
        )

        self.program.project(
            Weight.ATTENTION_OUTPUT,
            Buffer.ATTENTION_OUTPUT,
            Buffer.ATTENTION_HIDDEN,
            width=self.hidden,
        )
        self.residual(Weight.ATTENTION_POST_NORM)

    def attention_core(self, *, rope: bool, window: int) -> None:
        """Norm, Q/K/V projections, per-head Q/K norms, RoPE over halves
        (when `rope`), cache stores and grouped attention over the last
        `window` keys (0: all), leaving the heads in ATTENTION_OUTPUT."""
        p, c, hidden = self.program, self.config, self.hidden
        heads, kv, dim = c["heads"], c["kv_heads"], c["head_dim"]

        p.norm(
            Buffer.RESIDUAL,
            Weight.ATTENTION_NORM,
            Buffer.NORMALIZED,
            size=hidden,
            **self.eps,
        )
        p.project(
            Weight.QUERY, Buffer.NORMALIZED, Buffer.QUERY, width=heads * dim
        )
        p.project(Weight.KEY, Buffer.NORMALIZED, Buffer.KEY, width=kv * dim)
        p.project(
            Weight.VALUE, Buffer.NORMALIZED, Buffer.VALUE, width=kv * dim
        )
        p.norm(
            Buffer.QUERY,
            Weight.QUERY_NORM,
            Buffer.NORMALIZED_QUERY,
            size=dim,
            groups=heads,
            **self.eps,
        )
        p.norm(
            Buffer.KEY,
            Weight.KEY_NORM,
            Buffer.NORMALIZED_KEY,
            size=dim,
            groups=kv,
            **self.eps,
        )

        body = p.body()
        count, pos = variable(RuntimeVar.COUNT), variable(RuntimeVar.POSITION)
        q = buffer(Buffer.NORMALIZED_QUERY, heads * dim, View.ALWAYS)
        k = buffer(Buffer.NORMALIZED_KEY, kv * dim, View.ALWAYS)
        if rope:
            for source, head_count in ((q, heads), (k, kv)):
                body.emit(
                    Operation.ROTARY_HALVES,
                    source,
                    buffer(Buffer.ROTARY_FREQUENCIES),
                    count,
                    head_count,
                    dim,
                    pos,
                )
        for source, state in (
            (k, State.KEY),
            (buffer(Buffer.VALUE, kv * dim, View.ALWAYS), State.VALUE),
        ):
            body.emit(
                Operation.CACHE_STORE_GROUPED,
                source,
                ref(Binding.STATE, state),
                count,
                pos,
                variable(RuntimeVar.CAPACITY),
                variable(RuntimeVar.KV),
                kv,
                dim,
            )
        body.emit(
            Operation.WINDOWED_ATTENTION,
            q,
            ref(Binding.STATE, State.KEY),
            ref(Binding.STATE, State.VALUE),
            buffer(Buffer.ATTENTION_OUTPUT, heads * dim, View.ALWAYS),
            count,
            pos,
            variable(RuntimeVar.CAPACITY),
            variable(RuntimeVar.KV),
            heads,
            kv,
            dim,
            c["attention_scale"],
            buffer(Buffer.ATTENTION_PARTIALS),
            buffer(Buffer.ATTENTION_STATS),
            window,
        )
        p.loop(Operation.FOR_EACH_SEQUENCE, body)

    def mixture(self) -> None:
        p, c, hidden = self.program, self.config, self.hidden
        experts, top_k = c["experts"], c["top_k"]
        expert_ffn, shared_ffn = c["expert_ffn"], c["shared_ffn"]
        tokens = variable(RuntimeVar.TOKENS)

        p.norm(
            Buffer.RESIDUAL,
            Weight.MLP_NORM,
            Buffer.NORMALIZED,
            size=hidden,
            **self.eps,
        )

        p.project(
            Weight.ROUTER,
            Buffer.NORMALIZED,
            Buffer.ROUTER_LOGITS,
            width=experts,
        )
        p.emit(
            Operation.MOE_ROUTE,
            buffer(Buffer.ROUTER_LOGITS),
            ref(Binding.VECTOR, Weight.ROUTER_BIAS),
            buffer(Buffer.EXPERT_IDS),
            buffer(Buffer.EXPERT_WEIGHTS),
            tokens,
            experts,
            top_k,
            width=top_k,
        )

        # The shared expert into the MLP output, then the routed experts added.
        p.project(
            Weight.SHARED_GATE,
            Buffer.NORMALIZED,
            Buffer.MLP_GATE,
            width=shared_ffn,
        )
        p.project(
            Weight.SHARED_UP,
            Buffer.NORMALIZED,
            Buffer.MLP_UP,
            width=shared_ffn,
        )
        p.emit(
            Operation.SILU_MUL,
            buffer(Buffer.MLP_GATE),
            buffer(Buffer.MLP_UP),
            ref(Binding.PRODUCT, values=[tokens, shared_ffn]),
            width=shared_ffn,
        )
        p.project(
            Weight.SHARED_DOWN,
            Buffer.MLP_UP,
            Buffer.ATTENTION_HIDDEN,
            width=hidden,
        )

        p.emit(
            Operation.MOE_MATMUL,
            ref(Binding.LAYER_BUFFER, Resource.EXPERTS_GATE_UP),
            buffer(Buffer.NORMALIZED),
            buffer(Buffer.EXPERT_GATE_UP),
            buffer(Buffer.EXPERT_IDS),
            tokens,
            top_k,
            False,  # one input row per token, shared by its top_k slots
            width=2 * expert_ffn,
        )
        p.emit(
            Operation.SILU_MUL_HALVES,
            buffer(Buffer.EXPERT_GATE_UP),
            buffer(Buffer.EXPERT_ACTIVATION),
            ref(Binding.PRODUCT, values=[tokens, top_k]),
            expert_ffn,
            width=expert_ffn,
        )
        p.emit(
            Operation.MOE_MATMUL,
            ref(Binding.LAYER_BUFFER, Resource.EXPERTS_DOWN),
            buffer(Buffer.EXPERT_ACTIVATION),
            buffer(Buffer.EXPERT_OUTPUT),
            buffer(Buffer.EXPERT_IDS),
            tokens,
            top_k,
            True,  # one input row per slot
            width=hidden,
        )
        p.emit(
            Operation.MOE_COMBINE,
            buffer(Buffer.EXPERT_OUTPUT),
            buffer(Buffer.EXPERT_WEIGHTS),
            buffer(Buffer.ATTENTION_HIDDEN),
            tokens,
            top_k,
            hidden,
        )
        self.residual(Weight.MLP_POST_NORM, last=True)

    def residual(self, norm: str, last: bool = False) -> None:
        """x += RMSNorm(the block's output, held in ATTENTION_HIDDEN)."""
        p = self.program
        p.norm(
            Buffer.ATTENTION_HIDDEN,
            norm,
            Buffer.NORMALIZED,
            size=self.hidden,
            **self.eps,
        )
        p.emit(
            Operation.ADD,
            buffer(Buffer.NORMALIZED),
            buffer(Buffer.RESIDUAL),
            ref(
                Binding.PRODUCT,
                values=[variable(RuntimeVar.TOKENS), self.hidden],
            ),
            name=f"hidden_{p.layer}" if last else None,
            **({"layer_end": True} if last else {}),
        )


class Qwen3MoeComposition(KolibriComposition):
    """Qwen3-MoE (DO-80): Kolibri-1's attention core with RoPE over halves on
    every layer and no window, the attention output added straight to the
    residual (no post-branch norms), and a mixture of experts without a shared
    expert: softmax routing over all experts, the top k renormalized when
    `norm_topk`, and separate gate and up expert tensors.
    """

    def compose(self, layer: Mapping[str, Any]) -> None:
        self.program.layer = layer["index"]
        self.attention_core(rope=True, window=0)
        self.program.project(
            Weight.ATTENTION_OUTPUT,
            Buffer.ATTENTION_OUTPUT,
            Buffer.RESIDUAL,
            width=self.hidden,
            accumulate=True,
        )
        self.mixture()

    def mixture(self) -> None:
        p, c, hidden = self.program, self.config, self.hidden
        experts, top_k, expert_ffn = c["experts"], c["top_k"], c["expert_ffn"]
        tokens = variable(RuntimeVar.TOKENS)

        p.norm(
            Buffer.RESIDUAL,
            Weight.MLP_NORM,
            Buffer.NORMALIZED,
            size=hidden,
            **self.eps,
        )

        p.project(
            Weight.ROUTER,
            Buffer.NORMALIZED,
            Buffer.ROUTER_LOGITS,
            width=experts,
        )
        p.emit(
            Operation.MOE_ROUTE_SOFTMAX,
            buffer(Buffer.ROUTER_LOGITS),
            buffer(Buffer.EXPERT_IDS),
            buffer(Buffer.EXPERT_WEIGHTS),
            tokens,
            experts,
            top_k,
            bool(c["norm_topk"]),
            width=top_k,
        )

        # Gate and up rows of each token's experts, SwiGLU, then down.
        for resource, output in (
            (Resource.EXPERTS_GATE, Buffer.EXPERT_GATE_UP),
            (Resource.EXPERTS_UP, Buffer.EXPERT_ACTIVATION),
        ):
            p.emit(
                Operation.MOE_MATMUL,
                ref(Binding.LAYER_BUFFER, resource),
                buffer(Buffer.NORMALIZED),
                buffer(output),
                buffer(Buffer.EXPERT_IDS),
                tokens,
                top_k,
                False,  # one input row per token, shared by its top_k slots
                width=expert_ffn,
            )
        p.emit(
            Operation.SILU_MUL,
            buffer(Buffer.EXPERT_GATE_UP),
            buffer(Buffer.EXPERT_ACTIVATION),
            ref(Binding.PRODUCT, values=[tokens, top_k, expert_ffn]),
            width=expert_ffn,
        )
        p.emit(
            Operation.MOE_MATMUL,
            ref(Binding.LAYER_BUFFER, Resource.EXPERTS_DOWN),
            buffer(Buffer.EXPERT_ACTIVATION),
            buffer(Buffer.EXPERT_OUTPUT),
            buffer(Buffer.EXPERT_IDS),
            tokens,
            top_k,
            True,  # one input row per slot
            width=hidden,
        )

        # The weighted experts added straight into the residual.
        p.emit(
            Operation.MOE_COMBINE,
            buffer(Buffer.EXPERT_OUTPUT),
            buffer(Buffer.EXPERT_WEIGHTS),
            buffer(Buffer.RESIDUAL),
            tokens,
            top_k,
            hidden,
            name=f"hidden_{p.layer}",
            layer_end=True,
        )


def _compose_head(p: Program, config: Mapping[str, Any], vocab: int) -> None:
    """Normalize and project output chunks, then select each output token."""
    first_node = len(p.tape.nodes)
    options = {"size": p.hidden}
    if config.get("family") in (Family.DENSE, Family.KOLIBRI, Family.QWEN3MOE):
        options["eps"] = config["eps"]

    p.emit(
        Operation.RMS_NORM,
        buffer(Buffer.RESIDUAL),
        buffer(Buffer.OUTPUT_NORM),
        buffer(Buffer.NORMALIZED),
        variable(RuntimeVar.HEAD_TOKENS),
        ref(
            Binding.CHOOSE,
            condition=RuntimeVar.HEAD_ALL,
            yes=options,
            no=dict(options, inOffset=variable(RuntimeVar.HEAD_OFFSET)),
        ),
    )

    body = p.body()
    body.emit(
        Operation.MATMUL,
        ref(Binding.RESOURCE, Resource.OUTPUT_CHUNK),
        buffer(Buffer.NORMALIZED),
        buffer(Buffer.LOGITS),
        variable(RuntimeVar.HEAD_TOKENS),
        {
            "yStride": variable(RuntimeVar.VOCAB),
            "rowBase": variable(RuntimeVar.CHUNK_FIRST),
        },
        width=vocab,
    )
    p.loop(Operation.FOR_EACH_CHUNK, body)

    body = p.body(hidden=1)
    body.emit(
        Operation.ARGMAX,
        buffer(Buffer.LOGITS, vocab, View.HEAD),
        buffer(Buffer.TOKEN, 64, View.HEAD),
        variable(RuntimeVar.VOCAB),
        width=1,
    )
    p.loop(Operation.FOR_EACH_HEAD_ROW, body, name=Value.NEXT_TOKEN)

    for node in p.tape.nodes[first_node:]:
        node.attrs["when"] = RuntimeVar.HEAD_ENABLED


def _operations(tasks: list[JsonObject]) -> set[str]:
    operations = set()
    for task in tasks:
        operations.add(task["op"])
        operations.update(_operations(task["attrs"].get("body", [])))

    return operations


def compile_stage(
    manifest: Mapping[str, Any], owner: int = 0, *, aligned_gates: bool = True
) -> JsonObject:
    """Compile a composition; there are no model-layer execution opcodes."""
    family, hidden, incoming = _stage_layout(manifest, owner)
    config = dict(manifest.get("config") or {})
    if not aligned_gates:
        config["_unaligned_gates"] = True
    config.setdefault("hidden", hidden)
    config.setdefault("intermediate", 12288)

    p = Program(incoming, hidden, owner)
    if manifest.get("embed"):
        p.emit(
            Operation.EMBEDDING,
            ref(Binding.RESOURCE, Resource.EMBEDDING),
            buffer(Buffer.TOKEN_IDS),
            buffer(Buffer.RESIDUAL),
            variable(RuntimeVar.TOKENS),
            name=Value.EMBEDDING,
        )

    composition = {
        Family.KOLIBRI: KolibriComposition,
        Family.QWEN3MOE: Qwen3MoeComposition,
    }.get(family, LayerComposition)(p, config)
    for layer in manifest["layers"]:
        composition.compose(layer)

    p.layer = None
    if manifest.get("head"):
        _compose_head(
            p, config, manifest["head"].get("vocab", config.get("vocab", 1))
        )

    tasks = p.tasks()

    return dict(
        version=PLAN_VERSION,
        contract_version=SCHEMA_VERSION,
        state_contract=deepcopy(STATE_CONTRACT),
        family=family,
        owner=owner,
        rows=RuntimeVar.TOKENS,
        hidden=hidden,
        input=incoming,
        output=p.value,
        tasks=tasks,
        required_ops=sorted(_operations(tasks)),
        submit_token_layers=SUBMIT_TOKEN_LAYERS,
    )


def _compile_assigned_stage(
    stage: JsonObject, owner: int, last_owner: int, aligned_gates: bool = True
) -> JsonObject:
    """Check one assignment against its manifest and any stored plan."""
    manifest = stage["manifest"]
    indices = [layer["index"] for layer in manifest["layers"]]
    if not indices or stage["layers"] != [indices[0], indices[-1] + 1]:
        raise ValueError("Layer assignment differs from manifest")
    if bool(manifest.get("embed")) != (owner == 0) or bool(
        manifest.get("head")
    ) != (owner == last_owner):
        raise ValueError("Pipeline endpoints differ from stage assignments")

    expected = compile_stage(manifest, owner)
    existing = manifest.get("plan")
    if existing is not None and existing not in (
        expected,
        legacy_stage(expected),
    ):
        # Stored before the gates rows were padded: accept, run the fresh plan.
        unaligned = compile_stage(manifest, owner, aligned_gates=False)
        if existing not in (unaligned, legacy_stage(unaligned)):
            raise ValueError("Compiled stage plan differs from its assignment")
    if not aligned_gates:
        expected = compile_stage(manifest, owner, aligned_gates=False)

    return expected


def compile_distribution(
    stages: list[JsonObject], *, aligned_gates: bool = True
) -> JsonObject:
    """Validate the full distribution before attaching plans and routes."""
    if not stages:
        raise ValueError("A distribution needs stages")

    plans = [
        _compile_assigned_stage(stage, owner, len(stages) - 1, aligned_gates)
        for owner, stage in enumerate(stages)
    ]

    transfers = []
    for left, right in zip(plans, plans[1:]):
        if (
            left["output"] != right["input"]
            or left["hidden"] != right["hidden"]
        ):
            raise ValueError("Disconnected inference stage assignments")
        transfers.append(
            dict(
                value=left["output"],
                source=left["owner"],
                destination=right["owner"],
                precision=stages[left["owner"]]["fixture"]["wire"],
            )
        )

    by_source = {transfer["source"]: transfer for transfer in transfers}
    feedback = dict(
        value=plans[-1]["output"],
        source=plans[-1]["owner"],
        destination=0,
        precision="token-id",
    )

    routes = []

    for owner, stage in enumerate(stages):
        transfer = by_source.get(owner)
        route = dict(
            next=transfer["destination"]
            if transfer
            else feedback["destination"],
            output="hidden" if transfer else "token",
        )
        if stage["fixture"].get("plan_route", route) != route:
            raise ValueError("Compiled transfer route differs from stage")
        routes.append(route)

    for stage, plan, route in zip(stages, plans, routes, strict=True):
        stage["manifest"]["plan"] = plan
        stage["fixture"]["plan_route"] = route

    return dict(
        version=PLAN_VERSION,
        stages=plans,
        transfers=transfers,
        feedback=feedback,
    )
