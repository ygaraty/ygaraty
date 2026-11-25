"""Inkling 2 Mini (`inkling2_loop`) from ggml-org's GGUF: the loader and a NumPy reference (DO-95 phase 0).

The reference follows llama.cpp's `src/models/inkling2.cpp` and `inkling2-loop.cpp` (afe77e1) operation by
operation, so its logits can be compared with llama.cpp's on the same file. The formulas are written out in
`research/DO-95.md` ("Reference formulas"). In short:

- 24 stored layers, 48 visits per token: layers 0-5 once, 6-17 three times (shared weights; passes 2 and 3 restart
  from the output of layer 5 and mix in messages from the previous pass through rank-64 controllers), 18-23 once.
- Block attention residuals: in blocks of 6 layers, inputs are softmax mixes over the earlier block streams and
  the current one (`mix`); the first layer of a block starts a new stream from its attention output.
- Attention without RoPE: learned relative-position bias by distance (extent 1024) minus a per-token sink, added
  after the 1/sqrt(head_dim) scale (llama.cpp passes them as the softmax mask); Q/K/R L2 clips, V RMS-normalized, sigmoid output gate;
  sliding window 1024 on 20 layers. Causal 4-tap convolutions on K, V and both branch outputs (input-dependent
  taps on K and V of the 4 full-attention layers).
- MoE in every layer: sigmoid router, top 8 chosen with a bias, weights renormalized and scaled by 8; experts
  with the SiTU activation over fused gate/up rows.

This module computes a whole prompt from an empty state (prefill); per-visit decode state belongs to the stage.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ygaraty.backends.webgpu.checkpoints import gguf_remote, moe_gguf

ARCHITECTURE = "inkling2_loop"
EXPERT_CHUNK_BYTES = int(
    os.environ.get("YGARATY_INKLING2_EXPERT_CHUNK_BYTES", 128 * 2**20)
)
NORM_EPS = 1e-6  # the weightless RMS norms of mix, control and V (llama.cpp's literal 1e-6)
LOOP_CTRLS = ("q", "k", "v", "ffn")


@dataclass(frozen=True)
class Inkling2Config:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int  # padded rows of the embedding and output
    vocab_unpadded: int
    experts: int
    top_k: int
    expert_ffn: int
    expert_scale: float
    eps: float
    window: int
    sliding: tuple
    conv: int
    res_block: int
    rel_dim: int
    rel_extent: int
    qk_clip: float
    rel_clip: float
    log_floor: int
    log_alpha: float
    situ_beta: float
    situ_linear_beta: float
    logit_cap: float
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int

    def visits(self):
        """[(physical layer, pass)] in execution order: 48 for Inkling 2 Mini."""
        out = [(i, 0) for i in range(self.loop_start)]
        for p in range(self.passes):
            out += [(i, p) for i in range(self.loop_start, self.loop_end)]
        return out + [(i, 0) for i in range(self.loop_end, self.layers)]


def config_from_header(header):
    def key(name, default=None):
        value = header.get(name, default)  # under the architecture's prefix
        if value is None:
            raise ValueError(f"GGUF lacks {ARCHITECTURE}.{name}")
        return value

    heads, kv = key("attention.head_count"), key("attention.head_count_kv")
    if isinstance(heads, list):
        if len(set(heads)) != 1 or len(set(kv)) != 1:
            raise ValueError("Per-layer head counts must be equal")
        heads, kv = heads[0], kv[0]
    if key("loop.varying", False):
        raise ValueError("Variable-pass loops are not supported")
    if key("rope.dimension_count", 0):
        raise ValueError("Inkling 2 uses no RoPE")
    return Inkling2Config(
        layers=int(key("block_count")),
        hidden=int(key("embedding_length")),
        heads=int(heads),
        kv_heads=int(kv),
        head_dim=int(key("attention.key_length")),
        vocab=int(key("vocab_size")),
        vocab_unpadded=int(key("vocab_size_unpadded")),
        experts=int(key("expert_count")),
        top_k=int(key("expert_used_count")),
        expert_ffn=int(key("expert_feed_forward_length")),
        expert_scale=float(key("expert_weights_scale", 1.0)),
        eps=float(key("attention.layer_norm_rms_epsilon")),
        window=int(key("attention.sliding_window")),
        sliding=tuple(bool(v) for v in key("attention.sliding_window_pattern")),
        conv=int(key("shortconv.l_cache")),
        res_block=int(key("attn_res.block_size")),
        rel_dim=int(key("attention.relative_length")),
        rel_extent=int(key("attention.relative_extent")),
        qk_clip=float(key("attention.qk_clip")),
        rel_clip=float(key("attention.relative_clip")),
        log_floor=int(key("attention.log_scale_floor")),
        log_alpha=float(key("attention.log_scale_alpha")),
        situ_beta=float(key("activation.situ_beta")),
        situ_linear_beta=float(key("activation.situ_linear_beta")),
        logit_cap=float(key("final_logit_softcapping", 0.0)),
        loop_start=int(key("loop.start")),
        loop_end=int(key("loop.end")),
        passes=int(key("loop.count")),
        gate_rank=int(key("loop.gate_rank")),
    )


class Model:
    """An Inkling 2 GGUF: its tensors (memory-mapped, packed), its configuration, and dequantized dense weights."""

    family = "inkling2"

    def __init__(self, path):
        import gguf

        self.path = Path(path)
        self.header = gguf_remote.read_header(self.path)
        if self.header.architecture() != ARCHITECTURE:
            raise ValueError(f"Not an {ARCHITECTURE} GGUF: {self.header.architecture()}")
        self.config = config_from_header(self.header)
        self.reader = gguf.GGUFReader(self.path)
        self.tensors, self.experts = moe_gguf.read_tensors(self.reader, EXPERT_CHUNK_BYTES)
        self._dense = {}

    def has(self, name):
        return name in self.tensors

    def dense(self, name):
        """Float32 values of a non-expert tensor, NumPy order (rows are output features); cached."""
        if name not in self._dense:
            self._dense[name] = self.tensors[name].dequantize()
        return self._dense[name]

    def rows(self, name, rows):
        return self.tensors[name].dequantize(rows=np.asarray(rows))

    def expert(self, layer, e):
        """(gate_up [2F, H], down [H, F]) of one expert, float32."""
        c = self.config
        gu = self.experts[(layer, "ffn_gate_up_exps")]
        dn = self.experts[(layer, "ffn_down_exps")]
        two_f, h = 2 * c.expert_ffn, c.hidden
        return (
            gu.dequantize(rows=np.arange(e * two_f, (e + 1) * two_f)),
            dn.dequantize(rows=np.arange(e * h, (e + 1) * h)),
        )


# --- NumPy reference ---------------------------------------------------------------------------------------


def rms(x, eps=NORM_EPS):
    x = x.astype(np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def clip(x, threshold, scale=None):
    """Rows scaled down to an L2 norm of at most `threshold`, after the optional per-channel scale."""
    if scale is not None:
        x = x * scale
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + 1e-6
    return x * np.clip(threshold / norm, 0.0, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mix(blocks, cur, query):
    """Attention over depth: [T, Q, H] mixes of the RMS-normalized states [blocks..., cur] with queries [Q, H]."""
    states = rms(np.stack(list(blocks) + [cur], axis=1))  # [T, S, H]
    q = np.atleast_2d(query)  # [Q, H]
    scores = np.einsum("tsh,qh->tqs", states, q) / np.sqrt(states.shape[-1])
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    scores /= scores.sum(axis=-1, keepdims=True)
    return np.einsum("tqs,tsh->tqh", scores, states)


def convolve(c, x, weight, dynamic=None):
    """Causal depthwise convolution from an empty history, with the residual and llama.cpp's scale.

    x [T, C]; weight [C, K] (tap K-1 is the current token); dynamic [T, heads*K] (index head*K + lag, lag 0 is
    the current token), its taps shared by each head's C/heads channels."""
    T, C = x.shape
    K = c.conv
    xx = np.concatenate([np.zeros((K - 1, C), np.float32), x], axis=0)
    out = np.zeros_like(x)
    for k in range(K):
        out += xx[k : k + T] * weight[:, k]
    if dynamic is not None:
        heads = dynamic.shape[1] // K
        d = dynamic.reshape(T, heads, K)
        for lag in range(K):
            window = xx[K - 1 - lag : K - 1 - lag + T].reshape(T, heads, C // heads)
            out += (window * d[:, :, lag : lag + 1]).reshape(T, C)
    return (x + out) / np.sqrt((2 if dynamic is not None else 1) * K + 1)


def log_scale(c, T):
    pos = np.arange(T, dtype=np.float32)
    return 1.0 + c.log_alpha * np.log(np.maximum(1.0, (pos + 1.0) / c.log_floor))


def attention(c, q, k, v, table, sliding):
    """q [T, Hq, D], k/v [T, Hkv, D], table [T, Hq, extent+1] (bias by distance, sink already subtracted)."""
    T = q.shape[0]
    group = c.heads // c.kv_heads
    extent = table.shape[-1] - 1
    t = np.arange(T)
    dist = t[:, None] - t[None, :]
    allowed = dist >= 0
    if sliding:
        allowed &= dist < c.window
    index = np.where((dist >= 0) & (dist < extent), dist, extent)
    out = np.empty_like(q)
    scale = 1.0 / np.sqrt(c.head_dim)
    for h in range(c.heads):
        s = (q[:, h] @ k[:, h // group].T) * scale  # [T, T]
        s = s + np.take_along_axis(table[:, h], index, axis=1)  # the bias enters as the softmax mask: unscaled
        s = np.where(allowed, s, -np.inf)
        m = np.maximum(s.max(axis=1, keepdims=True), 0.0)  # the zero sink logit joins the softmax
        p = np.exp(s - m)
        p /= p.sum(axis=1, keepdims=True) + np.exp(-m)
        out[:, h] = p @ v[:, h // group]
    return out


def control(model, layer, which, x, message):
    """(1 - forget) x + gain m, with rank-`gate_rank` gates on [rms(x); rms(m)]."""
    n = f"blk.{layer}.loop_{which}_ctrl_"
    g = np.concatenate([rms(x), rms(message)], axis=-1) @ model.dense(n + "down.weight").T
    gain = np.tanh(g @ model.dense(n + "gain.weight").T)
    forget = np.tanh(g @ model.dense(n + "forget.weight").T)
    return x - forget * x + gain * message


def moe(model, layer, x):
    c = model.config
    w = lambda name: model.dense(f"blk.{layer}.{name}")
    probs = sigmoid(x @ w("ffn_gate_inp.weight").T)  # [T, E]
    selection = probs + w("exp_probs_b.bias") if model.has(f"blk.{layer}.exp_probs_b.bias") else probs
    top = np.argsort(-selection, axis=1, kind="stable")[:, : c.top_k]  # [T, k]
    weights = np.take_along_axis(probs, top, axis=1)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-20) * c.expert_scale
    out = np.zeros_like(x)
    F, b, lb = c.expert_ffn, c.situ_beta, c.situ_linear_beta
    for e in np.unique(top):
        rows, slots = np.nonzero(top == e)
        gate_up, down = model.expert(layer, int(e))
        gu = x[rows] @ gate_up.T
        g, u = gu[:, :F], gu[:, F:]
        y = b * np.tanh(g / b) * sigmoid(g) * (lb * np.tanh(u / lb))
        out[rows] += weights[rows, slots][:, None] * (y @ down.T)
    return out


def layer_forward(model, il, ip, cur, blocks, tau, messages=None):
    """One visit of physical layer `il` at block position `ip`; appends to `blocks` when a block starts."""
    c = model.config
    w = lambda name: model.dense(f"blk.{il}.{name}")
    sliding = c.sliding[il]
    T = cur.shape[0]
    inp = cur

    inputs = [cur, cur, cur]
    if model.has(f"blk.{il}.attn_res_query.weight"):
        mixed = mix(blocks, cur, w("attn_res_query.weight"))
        inputs = [mixed[:, 0 if sliding else j] for j in range(3)]
    if messages is not None:
        inputs = [control(model, il, LOOP_CTRLS[j], inputs[j], messages[:, j]) for j in range(3)]
    if sliding:
        inputs = [rms(x, c.eps) * w("attn_norm.weight") for x in inputs]
    else:
        inputs = [rms(x, c.eps) * w(f"attn_{n}_inp_norm.weight") for x, n in zip(inputs, "qkv")]
    xq, xk, xv = inputs

    q = xq @ w("attn_q.weight").T
    dyn_k = None if sliding else xk @ w("attn_k_conv_proj.weight").T
    dyn_v = None if sliding else xv @ w("attn_v_conv_proj.weight").T
    k = convolve(c, xk @ w("attn_k.weight").T, w("attn_k_conv.weight"), dyn_k)
    v = convolve(c, xv @ w("attn_v.weight").T, w("attn_v_conv.weight"), dyn_v)
    r = xq @ w("attn_rel.weight").T

    q = clip(q.reshape(T, c.heads, c.head_dim), c.qk_clip, w("attn_q_clip.weight"))
    k = clip(k.reshape(T, c.kv_heads, c.head_dim), c.qk_clip)
    v = rms(v.reshape(T, c.kv_heads, c.head_dim), c.eps)
    r = clip(r.reshape(T, c.heads, c.rel_dim), c.rel_clip, w("attn_rel_clip.weight"))
    if not sliding:
        q = q * tau[:, None, None]
        r = r * tau[:, None, None]

    bias = w("attn_rel_bias.weight")  # [extent, rel_dim]
    bias = np.concatenate([bias, np.zeros((1, bias.shape[1]), np.float32)])
    sinks = xq @ w("attn_sink_proj.weight").T  # [T, Hq]
    table = np.einsum("thr,dr->thd", r, bias) - sinks[:, :, None]

    o = attention(c, q, k, v, table, sliding).reshape(T, -1)
    o = o * sigmoid(xq @ w("attn_gate.weight").T)
    a = convolve(c, o @ w("attn_output.weight").T, w("attn_post_conv.weight"))

    if ip % c.res_block == 0:
        blocks.append(inp)
        cur = a
    else:
        cur = inp + a
    ffn_inp = cur

    f = mix(blocks, cur, w("ffn_res_query.weight"))[:, 0]
    if messages is not None:
        f = control(model, il, "ffn", f, messages[:, 3])
    f = rms(f, c.eps) * w("ffn_norm.weight")
    y = convolve(c, moe(model, il, f), w("ffn_post_conv.weight"))
    return ffn_inp + y


def embed(model, tokens):
    c = model.config
    x = model.rows("token_embd.weight", tokens)
    return rms(x, c.eps) * model.dense("token_embd_norm.weight")


def run(model, tokens, trace=None):
    """Final normalized states [T, H] of a prompt from an empty state. `trace(visit, il, pass, cur)` sees every
    visit's output (visit 0..47, llama.cpp's cache slot order is different: see `slot`)."""
    c = model.config
    tokens = np.asarray(tokens, dtype=np.int64)
    tau = log_scale(c, len(tokens))
    cur = embed(model, tokens)
    blocks = []
    visit = 0

    def step(il, p, cur, messages=None):
        nonlocal visit
        out = layer_forward(model, il, il, cur, blocks, tau, messages)
        if trace is not None:
            trace(visit, il, p, out)
        visit += 1
        return out

    for il in range(c.loop_start):
        cur = step(il, 0, cur)
    prelude, prelude_blocks = cur, list(blocks)
    for p in range(c.passes):
        bank = bank_out = None
        if p > 0:
            bank, bank_out = blocks[len(prelude_blocks) + 1 :], cur
            cur, blocks[:] = prelude, prelude_blocks
        for il in range(c.loop_start, c.loop_end):
            messages = None
            if p > 0:
                messages = mix(bank, bank_out, model.dense(f"blk.{il}.loop_msg_query.weight"))
            cur = step(il, p, cur, messages)
    for il in range(c.loop_end, c.layers):
        cur = step(il, 0, cur)

    out = mix(blocks, cur, model.dense("output_res_query.weight"))[:, 0]
    return rms(out, c.eps) * model.dense("output_norm.weight")


def slot(c, il, p):
    """llama.cpp's cache slot (the `il` in its tensor names) of physical layer `il` in pass `p`."""
    if p == 0:
        return il
    return c.layers + (p - 1) * (c.loop_end - c.loop_start) + (il - c.loop_start)


def logits(model, hidden, chunk=16384):
    """Softcapped logits [T, vocab_unpadded] of final normalized states."""
    c = model.config
    out = np.empty((hidden.shape[0], c.vocab_unpadded), np.float32)
    for first in range(0, c.vocab_unpadded, chunk):
        last = min(first + chunk, c.vocab_unpadded)
        w = model.rows("output.weight", np.arange(first, last))
        out[:, first:last] = hidden @ w.T
    if c.logit_cap:
        out = c.logit_cap * np.tanh(out / c.logit_cap)
    return out


# --- distribution (arm B) ----------------------------------------------------------------------------------


def boundary_vectors(c, layer):
    """Vectors per token crossing a stage boundary before `layer`: the attention-residual block streams started
    before it (layers 0, res_block, ...), then the current stream (`inkling2_gpu.mts` boundaryVectors)."""
    return 1 + sum(1 for il in range(layer) if il % c.res_block == 0)


def stage_manifest(path, first, last, header=None):
    """The `Inkling2Stage` manifest of layers [first, last) and its files {key: (path, offset, length)}: a file per
    tensor, so the stage's offsets are 0 within each file. The first stage takes the embedding, the last the
    output head; a stage holds the whole loop core or none of it."""
    import dataclasses

    header = header or gguf_remote.read_header(path)
    c = config_from_header(header)
    if not 0 <= first < last <= c.layers:
        raise ValueError("Invalid stage layer range")
    if first < c.loop_end and last > c.loop_start and (first > c.loop_start or last < c.loop_end):
        raise ValueError(f"Stage {first}-{last - 1} splits the loop core {c.loop_start}-{c.loop_end - 1}")
    names = [n for n in header.tensors if n.startswith("blk.") and first <= int(n.split(".")[1]) < last]
    if first == 0:
        names += ["token_embd.weight", "token_embd_norm.weight"]
    if last == c.layers:
        names += ["output.weight" if "output.weight" in header.tensors else "token_embd.weight",
                  "output_norm.weight", "output_res_query.weight"]
    tensors, files = {}, {}
    for name in dict.fromkeys(names):
        t = header.tensors[name]
        tensors[name] = dict(kind=t.kind, shape=list(t.shape), offset=0, bytes=t.nbytes, file=name)
        files[name] = (str(path), t.offset, t.nbytes)
    config = dataclasses.asdict(c) | {"family": "inkling2", "sliding": list(c.sliding)}
    manifest = dict(config=config, first=first, last=last, tensors=tensors,
                    layers=[dict(index=i, type="sliding" if c.sliding[i] else "full") for i in range(first, last)],
                    embed=first == 0, head=last == c.layers)
    return manifest, files
