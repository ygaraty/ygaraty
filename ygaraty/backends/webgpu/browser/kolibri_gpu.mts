import {validatePlan, executePlan, bindKernel, assertStateReady, resetPlanState} from "./inference_plan.mjs";
import type {StagePlan, PlanContext, SequencePart, KernelRegistry} from "./inference_plan.mjs";
// Generated deployment module: edit kolibri_gpu.mts, then npm run build:webgpu.
import { Gpu as KernelGpu, objectRecord, positiveInteger, isPackedKind, BYTES, PAIR, MAX_DISPATCH, fastKind, REPACK, wgslMemo } from "./webgpu_kernels.mjs";
import type { BufferBinding, PackedKind, PackedMatrix, EmbeddingChunk, FetchTensor, StageInput, StageRunner } from "./webgpu_kernels.mjs";
import { denseRope, denseStore, denseWindowedAttention } from "./dense_gpu.mjs";
import type { CachePrecision } from "./dense_gpu.mjs";

// Kolibri-1 layers (the `kolibri1` model family) on WebGPU over our GGUF's packed weights. See
// ygaraty/inference/kolibri_gguf.py for the layer and the layout.
//
// A layer: RMSNorm, Q/K/V projections, per-head RMSNorm of Q and K, RoPE over halves on sliding-window layers
// only (global layers use no position encoding), grouped-query attention over a float16 (default) or float32
// KV cache (sliding layers see the last `window` positions), the output projection normalized and added to
// the residual; then RMSNorm, the router (float32 logits) and its top-k selection on the GPU, the shared
// SwiGLU expert plus the selected experts weighted by sigmoid(logit), normalized and added to the residual.
//
// The routed experts are chosen on the GPU, so their products read the expert ids from a buffer: a
// workgroup per (row group, token slot) decodes rows of its slot's expert only, and exits when the expert is
// in another chunk (each chunk of whole experts is its own buffer, at most WebGPU's default binding size).
// Every layer keeps its whole KV cache (4 KV heads make it small); attention reads only the window.

// DO-80: the same adapter binds Qwen3-MoE (`qwen3moe`): full attention with RoPE on every layer, softmax routing
// (top k renormalized when `norm_topk`), separate gate and up expert tensors, no shared expert or post-branch
// norms. Its plan (distribution.Qwen3MoeComposition) decides the math; this module only loads and binds.
export type MoeFamily = "kolibri1" | "qwen3moe";
export interface KolibriConfig {
  family: MoeFamily;
  layers: number; hidden: number; heads: number; kv_heads: number; head_dim: number; vocab: number;
  experts: number; top_k: number; expert_ffn: number; shared_ffn?: number; window?: number;
  rope_theta: number; eps: number; attention_scale: number; sliding?: boolean[]; intermediate: number;
  norm_topk?: boolean;
  tied_embeddings?: boolean;
}
export interface TensorManifest { file: string; kind: PackedKind; shape: number[]; row_bytes: number }
export interface ChunkManifest extends TensorManifest { first: number }
export interface KolibriManifest {
  plan?: unknown;
  config: KolibriConfig;
  layers: { index: number; type: "sliding" | "full"; tensors: Record<string, TensorManifest> }[];
  embed?: { chunks: ChunkManifest[] } | null;
  head?: { chunks: ChunkManifest[]; norm: TensorManifest; vocab: number; shared_embedding?: boolean } | null;
}
export interface ExpertChunk extends PackedMatrix { first: number; count: number }

// Per family: the layer's dense matrices and vectors, and its expert tensors as plan resources (`gateUp`, `down`,
// or `gate`, `up`, `down`) with each expert's rows ("2F": gate then up rows; "F"; "H": the hidden width).
const PROFILES: Record<MoeFamily, { matrices: string[]; vectors: string[]; experts: Record<string, [string, "2F" | "F" | "H"]> }> = {
  kolibri1: {
    matrices: ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate_shexp.weight",
               "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp.weight"],
    vectors: ["attn_norm.weight", "attn_post_norm.weight", "ffn_norm.weight", "ffn_post_norm.weight",
              "attn_q_norm.weight", "attn_k_norm.weight", "exp_probs_b.bias"],
    experts: { gateUp: ["ffn_gate_up_exps", "2F"], down: ["ffn_down_exps", "H"] },
  },
  qwen3moe: {
    matrices: ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate_inp.weight"],
    vectors: ["attn_norm.weight", "ffn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"],
    experts: { gate: ["ffn_gate_exps", "F"], up: ["ffn_up_exps", "F"], down: ["ffn_down_exps", "H"] },
  },
};
const expertShape = (c: KolibriConfig, rows: "2F" | "F" | "H") =>
  rows === "H" ? [c.hidden, c.expert_ffn] : [(rows === "2F" ? 2 : 1) * c.expert_ffn, c.hidden];
const FAST_KINDS = new Set(["Q4_K", "Q5_K", "Q6_K"]);

export function isKolibriManifest(value: unknown): boolean {
  const family = typeof value === "object" && value !== null ? (value as { config?: { family?: unknown } }).config?.family : undefined;
  return family === "kolibri1" || family === "qwen3moe";
}

// The expert chunk names of one tensor, in order: `ffn_gate_up_exps.0.weight`, `.1.`, ...
export function expertChunkNames(tensors: Record<string, unknown>, base: string): string[] {
  const pattern = new RegExp(`^${base}\\.(\\d+)\\.weight$`);
  const found = Object.keys(tensors).map((name) => [name, pattern.exec(name)] as const).filter(([, m]) => m)
    .map(([name, m]) => [name, Number(m![1])] as const).sort((a, b) => a[1] - b[1]);
  if (!found.length || found.some(([, k], i) => k !== i)) throw new Error(`Missing or unordered ${base} chunks`);
  return found.map(([name]) => name);
}

export function parseManifest(value: unknown): KolibriManifest {
  const m = objectRecord(value, "kolibri manifest");
  const c = objectRecord(m.config, "kolibri config");
  if (c.family !== "kolibri1" && c.family !== "qwen3moe") throw new Error("Not a kolibri1- or qwen3moe-family manifest");
  const kolibri = c.family === "kolibri1";
  for (const key of ["layers", "hidden", "heads", "kv_heads", "head_dim", "vocab", "experts", "top_k", "expert_ffn",
                     ...(kolibri ? ["shared_ffn", "window"] : []), "intermediate"]) positiveInteger(c[key], key);
  for (const key of ["rope_theta", "eps", "attention_scale"])
    if (typeof c[key] !== "number" || !Number.isFinite(c[key] as number) || (c[key] as number) <= 0) throw new Error(`Invalid ${key}`);
  const cfg = c as unknown as KolibriConfig;
  if (kolibri && (!Array.isArray(cfg.sliding) || cfg.sliding.length !== cfg.layers || !cfg.sliding.every((v) => typeof v === "boolean")))
    throw new Error("Invalid sliding-layer list");
  if (!kolibri && (typeof cfg.norm_topk !== "boolean" || cfg.sliding !== undefined || cfg.shared_ffn !== undefined))
    throw new Error("Invalid qwen3moe routing or layer options");
  const profile = PROFILES[cfg.family];
  if (![64, 128].includes(cfg.head_dim)) throw new Error("Unsupported head_dim");
  if (cfg.heads % cfg.kv_heads || cfg.heads / cfg.kv_heads > 16) throw new Error("Unsupported query group");
  if (cfg.hidden % 256 || cfg.expert_ffn % 256 || (cfg.shared_ffn ?? 0) % 256 || (cfg.heads * cfg.head_dim) % 256) throw new Error("Unsupported widths");
  if (cfg.top_k > 16 || cfg.top_k > cfg.experts || cfg.experts > 4096) throw new Error("Unsupported routing");
  if (cfg.intermediate !== cfg.top_k * 2 * cfg.expert_ffn) throw new Error("Inconsistent expert scratch width");
  const tensor = (value: unknown, chunk = false) => {
    const t = objectRecord(value, "tensor");
    if (typeof t.file !== "string" || !t.file || !isPackedKind(t.kind) || !Array.isArray(t.shape) ||
        t.shape.length < 1 || t.shape.length > 2) throw new Error("Invalid tensor descriptor");
    for (const width of t.shape) positiveInteger(width, "tensor shape");
    positiveInteger(t.row_bytes, "row_bytes");
    if (chunk && (typeof t.first !== "number" || !Number.isSafeInteger(t.first) || t.first < 0)) throw new Error("Invalid chunk offset");
  };
  if (!Array.isArray(m.layers) || !m.layers.length) throw new Error("Missing layer list");
  let previous: number | undefined;
  for (const value of m.layers) {
    const layer = objectRecord(value, "layer");
    if (typeof layer.index !== "number" || !Number.isInteger(layer.index) || layer.index < 0 || layer.index >= cfg.layers ||
        (previous !== undefined && layer.index !== previous + 1) || layer.type !== (cfg.sliding?.[layer.index] ? "sliding" : "full"))
      throw new Error("Invalid layer order/type");
    previous = layer.index;
    const tensors = objectRecord(layer.tensors, "layer tensors");
    for (const name of [...profile.matrices, ...profile.vectors]) if (!(name in tensors)) throw new Error(`Missing ${name}`);
    for (const value of Object.values(tensors)) tensor(value);
    for (const [base, kind] of Object.values(profile.experts)) {
      const [rows, cols] = expertShape(cfg, kind);
      let experts = 0;
      for (const name of expertChunkNames(tensors, base)) {
        const t = tensors[name] as TensorManifest;
        if (!FAST_KINDS.has(t.kind) || t.shape.length !== 2 || t.shape[1] !== cols || t.shape[0] % rows) throw new Error(`Invalid ${name}`);
        experts += t.shape[0] / rows;
      }
      if (experts !== cfg.experts) throw new Error(`${base}: ${experts} experts, expected ${cfg.experts}`);
    }
  }
  for (const key of ["embed", "head"]) {
    if (m[key] === undefined || m[key] === null) continue;
    const projection = objectRecord(m[key], key);
    if (!Array.isArray(projection.chunks) || !projection.chunks.length) throw new Error("Missing projection chunks");
    let offset = 0;
    for (const value of projection.chunks) {
      tensor(value, true);
      const chunk = objectRecord(value, "chunk");
      if ((chunk.shape as number[]).length !== 2 || chunk.first !== offset) throw new Error("Invalid row coverage");
      offset += (chunk.shape as number[])[0];
    }
    if (key === "head") {
      tensor(projection.norm);
      positiveInteger(projection.vocab, "vocab");
      if (offset !== projection.vocab) throw new Error("Output row count differs from vocab");
    }
  }
  return value as KolibriManifest;
}

const NEG = "bitcast<f32>(0xff7fffffu)";  // the most negative float32: below any score

// Top-k routing of one token per workgroup: scores = logits + bias; k rounds of argmax (the first index
// among equal scores, as a stable descending sort), each writing the expert id and sigmoid(unbiased logit).
const routeShaderSource = (E: number) => /* wgsl */ `
struct P { tokens: u32, top_k: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;  // [T, E]
@group(0) @binding(1) var<storage, read> bias: array<f32>;    // [E]
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;      // [T, k]
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;  // [T, k]
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> score: array<f32, ${E}>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<u32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x;
  for (var e = li.x; e < ${E}u; e += 64u) { score[e] = logits[t * ${E}u + e] + bias[e]; }
  workgroupBarrier();
  for (var j = 0u; j < p.top_k; j++) {
    var v = ${NEG}; var i = 0xffffffffu;
    for (var e = li.x; e < ${E}u; e += 64u) { if (score[e] > v) { v = score[e]; i = e; } }
    bv[li.x] = v; bi[li.x] = i;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
      if (li.x < s) {
        let ov = bv[li.x + s]; let oi = bi[li.x + s];
        if (ov > bv[li.x] || (ov == bv[li.x] && oi < bi[li.x])) { bv[li.x] = ov; bi[li.x] = oi; }
      }
      workgroupBarrier();
    }
    if (li.x == 0u) {
      let e = bi[0];
      ids[t * p.top_k + j] = e;
      weights[t * p.top_k + j] = 1.0 / (1.0 + exp(-logits[t * ${E}u + e]));
      score[e] = ${NEG};
    }
    workgroupBarrier();
  }
}`;
const routeShader = wgslMemo(routeShaderSource);  // WS-33: one source string object per arguments

// y[slot, row] = sum_e W_expert(slot)[row, e] * x[slot or token, e] over one chunk of whole experts
// (`first` .. `first + count - 1`): the matrix-vector kernel's word-wise decoding with MR rows per
// workgroup of 64 threads and MT threads per row, one token slot per workgroup row of the dispatch.
const MR = 4, MT = 64 / MR;
const expertShaderSource = (kind: keyof typeof PAIR) => /* wgsl */ `
struct P { rows: u32, cols: u32, row_bytes: u32, first: u32, count: u32, top_k: u32, per_slot: u32, x_stride: u32,
           y_stride: u32, slots: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> part: array<f32, 64>;
var<workgroup> chosen: u32;
${BYTES}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let slot = wg.y;
  if (li.x == 0u) { chosen = ids[slot]; }
  let e = workgroupUniformLoad(&chosen);
  if (e < p.first || e >= p.first + p.count) { return; }
  let r = li.x / ${MT}u; let k = li.x % ${MT}u;
  let row = (wg.x + wg.z * ${MAX_DISPATCH}u) * ${MR}u + r;
  let xb = (select(slot / p.top_k, slot, p.per_slot == 1u) * p.x_stride) / 4u;
  var acc = 0.0;
  if (row < p.rows) {
    let items = p.cols / 64u;
    for (var it = k; it < items; it += ${MT}u) {
      let sb = it / 4u; let g = it % 4u;
      let b = ((e - p.first) * p.rows + row) * p.row_bytes + sb * ${PAIR[kind].block}u;
      ${PAIR[kind].setup}
      let xl = sb * 64u + el / 4u; let xh = sb * 64u + eh / 4u;
      for (var i = 0u; i < 8u; i++) {
        ${PAIR[kind].word}
        acc += dot(lo, x[xb + xl + i]) + dot(hi, x[xb + xh + i]);
      }
    }
  }
  part[li.x] = acc;
  workgroupBarrier();
  for (var s = ${MT / 2}u; s > 0u; s >>= 1u) {
    if (k < s) { part[li.x] += part[li.x + s]; }
    workgroupBarrier();
  }
  if (row < p.rows && k == 0u) { y[slot * p.y_stride + row] = part[li.x]; }
}`;
const expertShader = wgslMemo(expertShaderSource);  // WS-33: one source string object per arguments

// y[s, c] = silu(a[s, c]) * a[s, half + c]: SwiGLU over a fused [gate | up] row.
const SILU_HALVES = /* wgsl */ `
struct P { n: u32, half: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.n) { return; }
  let s = i / p.half; let c = i % p.half;
  let g = a[s * 2u * p.half + c];
  y[i] = g / (1.0 + exp(-g)) * a[s * 2u * p.half + p.half + c];
}`;

// out[t, c] += sum_j weight[t, j] * ed[t * k + j, c], slots in routing order.
const COMBINE = /* wgsl */ `
struct P { n: u32, hidden: u32, top_k: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> ed: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.n) { return; }
  let t = i / p.hidden; let c = i % p.hidden;
  var v = out[i];
  for (var j = 0u; j < p.top_k; j++) { v += weights[t * p.top_k + j] * ed[(t * p.top_k + j) * p.hidden + c]; }
  out[i] = v;
}`;

const ADD = /* wgsl */ `
struct P { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.n) { return; }
  y[i] = y[i] + a[i];
}`;

const elementwise = (n: number): [number, number, number] => {
  const groups = Math.ceil(n / 256);
  return [Math.min(groups, MAX_DISPATCH), Math.ceil(groups / MAX_DISPATCH), 1];
};

// The family's own operations, as functions over any shared-kernel GPU.
export function moeRoute(gpu: KernelGpu, logits: BufferBinding, bias: BufferBinding, ids: BufferBinding, weights: BufferBinding,
                         tokens: number, experts: number, topK: number) {
  const u = gpu.uniform([["u", tokens], ["u", topK], ["u", 0], ["u", 0]]);
  gpu.dispatch(`moe_route_${experts}`, routeShader(experts), [logits, bias, ids, weights, u], [tokens, 1, 1]);
}

// DO-80: softmax routing of one token per workgroup: the largest logit and the sum of exp(logit - max) over all
// experts, then k rounds of argmax over the logits (the softmax keeps their order; the first index among equal
// logits, as a stable descending sort), each writing the expert id and its probability; with `normalize`, the
// chosen probabilities divided by their sum (Qwen3-MoE's norm_topk_prob).
const routeSoftmaxShaderSource = (E: number) => /* wgsl */ `
struct P { tokens: u32, top_k: u32, normalize: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;  // [T, E]
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;      // [T, k]
@group(0) @binding(2) var<storage, read_write> weights: array<f32>;  // [T, k]
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> score: array<f32, ${E}>;
var<workgroup> bv: array<f32, 64>;
var<workgroup> bi: array<u32, 64>;
var<workgroup> picked: f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x;
  var m = ${NEG};
  for (var e = li.x; e < ${E}u; e += 64u) { let v = logits[t * ${E}u + e]; score[e] = v; m = max(m, v); }
  bv[li.x] = m;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li.x < s) { bv[li.x] = max(bv[li.x], bv[li.x + s]); } workgroupBarrier(); }
  let top = bv[0];
  workgroupBarrier();
  var z = 0.0;
  for (var e = li.x; e < ${E}u; e += 64u) { z += exp(score[e] - top); }
  bv[li.x] = z;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) { if (li.x < s) { bv[li.x] += bv[li.x + s]; } workgroupBarrier(); }
  let total = bv[0];
  if (li.x == 0u) { picked = 0.0; }
  workgroupBarrier();
  for (var j = 0u; j < p.top_k; j++) {
    var v = ${NEG}; var i = 0xffffffffu;
    for (var e = li.x; e < ${E}u; e += 64u) { if (score[e] > v) { v = score[e]; i = e; } }
    bv[li.x] = v; bi[li.x] = i;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
      if (li.x < s) {
        let ov = bv[li.x + s]; let oi = bi[li.x + s];
        if (ov > bv[li.x] || (ov == bv[li.x] && oi < bi[li.x])) { bv[li.x] = ov; bi[li.x] = oi; }
      }
      workgroupBarrier();
    }
    if (li.x == 0u) {
      let e = bi[0];
      let w = exp(score[e] - top) / total;
      ids[t * p.top_k + j] = e;
      weights[t * p.top_k + j] = w;
      picked += w;
      score[e] = ${NEG};
    }
    workgroupBarrier();
  }
  if (p.normalize == 1u && li.x < p.top_k) { weights[t * p.top_k + li.x] = weights[t * p.top_k + li.x] / picked; }
}`;
const routeSoftmaxShader = wgslMemo(routeSoftmaxShaderSource);  // WS-33: one source string object per arguments

export function moeRouteSoftmax(gpu: KernelGpu, logits: BufferBinding, ids: BufferBinding, weights: BufferBinding,
                                tokens: number, experts: number, topK: number, normalize: boolean) {
  const u = gpu.uniform([["u", tokens], ["u", topK], ["u", normalize ? 1 : 0], ["u", 0]]);
  gpu.dispatch(`moe_route_softmax_${experts}`, routeSoftmaxShader(experts), [logits, ids, weights, u], [tokens, 1, 1]);
  captureRoutes(gpu, ids, tokens * topK);
}

// DO-80 diagnostic: with `routeCapture` set on the GPU object, each routing's expert ids are appended to that buffer
// (a copy dispatch inside the pass), so a reference can follow the GPU's own expert choices layer by layer.
const CAPTURE = /* wgsl */ `
struct P { n: u32, at: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { if (gid.x < p.n) { dst[p.at + gid.x] = src[gid.x]; } }`;
export type RouteCapture = { buffer: GPUBuffer; at: number };
function captureRoutes(gpu: KernelGpu, ids: BufferBinding, n: number) {
  const c = (gpu as unknown as { routeCapture?: RouteCapture }).routeCapture;
  if (!c) return;
  const u = gpu.uniform([["u", n], ["u", c.at], ["u", 0], ["u", 0]]);
  gpu.dispatch("route_capture", CAPTURE, [ids, c.buffer, u], [Math.ceil(n / 64), 1, 1]);
  c.at += n;
}

// Expert products for `tokens` x `topK` slots over every chunk; `perSlot`: the input has one row per slot
// (the down projection), else one per token (gate/up).
export function moeMatmul(gpu: KernelGpu, chunks: readonly ExpertChunk[], x: BufferBinding, y: BufferBinding, ids: BufferBinding,
                          tokens: number, topK: number, perSlot: boolean) {
  const slots = tokens * topK;
  if (slots > MAX_DISPATCH) throw new Error(`${slots} expert slots exceed one dispatch`);
  for (const c of chunks) {
    const rows = c.rows / c.count;
    const groups = Math.ceil(rows / MR);
    const u = gpu.uniform([["u", rows], ["u", c.cols], ["u", c.rowBytes], ["u", c.first], ["u", c.count], ["u", topK],
                           ["u", perSlot ? 1 : 0], ["u", c.cols], ["u", rows], ["u", slots], ["u", 0], ["u", 0]]);
    gpu.dispatch(`moe_matvec_${c.kind}`, expertShader(fastKind(c.kind)), [c.buffer, x, y, ids, u],
      [Math.min(groups, MAX_DISPATCH), slots, Math.ceil(groups / MAX_DISPATCH)]);
  }
}

export function siluMulHalves(gpu: KernelGpu, a: BufferBinding, y: BufferBinding, slots: number, half: number) {
  const u = gpu.uniform([["u", slots * half], ["u", half], ["u", 0], ["u", 0]]);
  gpu.dispatch("silu_mul_halves", SILU_HALVES, [a, y, u], elementwise(slots * half));
}

export function moeCombine(gpu: KernelGpu, ed: BufferBinding, weights: BufferBinding, out: BufferBinding, tokens: number, topK: number,
                           hidden: number) {
  const u = gpu.uniform([["u", tokens * hidden], ["u", hidden], ["u", topK], ["u", 0]]);
  gpu.dispatch("moe_combine", COMBINE, [ed, weights, out, u], elementwise(tokens * hidden));
}

export function addInto(gpu: KernelGpu, a: BufferBinding, y: BufferBinding, n: number) {
  const u = gpu.uniform([["u", n], ["u", 0], ["u", 0], ["u", 0]]);
  gpu.dispatch("add", ADD, [a, y, u], elementwise(n));
}

export class Gpu extends KernelGpu {}

interface Layer {
  index: number;
  type: "sliding" | "full";
  matrices: Record<string, PackedMatrix>;
  vectors: Record<string, GPUBuffer>;
  gateUp?: ExpertChunk[]; gate?: ExpertChunk[]; up?: ExpertChunk[]; down: ExpertChunk[];  // per family profile
  K: GPUBuffer; V: GPUBuffer; Ks: GPUBuffer[]; Vs: GPUBuffer[];
}

// One pipeline stage: optionally the embedding, then layers first..last-1, optionally the output head.
export class Stage implements StageRunner<void> {
  readonly config: KolibriConfig;
  readonly hidden: number;
  readonly intermediate: number;
  readonly manifest: KolibriManifest;
  readonly kv: CachePrecision;
  readonly slots: number;
  readonly maxBatch: number;
  position = 0;
  positions: number[];
  slot = 0;
  private loaded = false;
  readonly plan: StagePlan | null;
  layers: Layer[] = [];
  embed?: EmbeddingChunk[];
  output?: EmbeddingChunk[];
  vocab = 0;
  submitMs: number[] = [];
  submitLayers: number[] = [];
  x!: GPUBuffer; h!: GPUBuffer; mid!: GPUBuffer; q!: GPUBuffer; k!: GPUBuffer; v!: GPUBuffer; qn!: GPUBuffer; kn!: GPUBuffer;
  att!: GPUBuffer; f1!: GPUBuffer; f2!: GPUBuffer; router!: GPUBuffer; experts!: GPUBuffer; expertWeights!: GPUBuffer;
  eg!: GPUBuffer; ea!: GPUBuffer; ed!: GPUBuffer; part!: GPUBuffer; ml!: GPUBuffer; ids!: GPUBuffer; invFreq!: GPUBuffer;
  outputNorm!: GPUBuffer; logits!: GPUBuffer; token!: GPUBuffer;

  readonly mergeExperts: boolean;

  constructor(readonly gpu: KernelGpu, manifest: unknown, readonly maxTokens: number, readonly maxPositions: number,
              { kv = "f16", slots = 1, mergeExperts = true }: { kv?: CachePrecision; slots?: number; mergeExperts?: boolean } = {}) {
    this.mergeExperts = mergeExperts;
    if (!["f16", "f32"].includes(kv)) throw new Error(`kv ${kv}`);
    this.manifest = parseManifest(manifest);
    this.config = this.manifest.config;
    this.hidden = this.config.hidden;
    this.intermediate = this.config.intermediate;
    this.plan = this.manifest.plan === undefined ? null : validatePlan(this.manifest.plan, this.manifest, this.config.family, this.hidden);
    if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || !Number.isSafeInteger(maxPositions) || maxPositions < maxTokens)
      throw new Error("Invalid token/context capacity");
    if (maxTokens * this.config.top_k > MAX_DISPATCH) throw new Error("Too many tokens per call for the expert kernel");
    if (!Number.isSafeInteger(slots) || slots < 1) throw Error("Invalid slot count");
    this.kv = kv;
    this.slots = slots;
    this.maxBatch = Math.min(slots, maxTokens);
    this.positions = new Array<number>(slots).fill(0);
  }

  use(slot: number) {
    if (!Number.isInteger(slot) || !(slot >= 0 && slot < this.slots)) throw new Error(`slot ${slot} of ${this.slots}`);
    for (const L of this.layers) { L.K = L.Ks[slot]; L.V = L.Vs[slot]; }
    this.slot = slot;
    this.position = this.positions[slot];
  }

  async load(fetchTensor: FetchTensor, log: (message: string) => void = () => {}) {
    const gpu = this.gpu, m = this.manifest, c = this.config;
    // Experiments 141/142: K-quant matrices listed in gpu.repack are repacked exactly at load (scales unpacked;
    // same products, bit for bit). `packed` returns the bytes to upload and the matrix description.
    const packed = (t: TensorManifest, data: ArrayBuffer | ArrayBufferView) => {
      const rows = t.shape[0], cols = t.shape[t.shape.length - 1], r = gpu.repack.includes(t.kind) ? REPACK[t.kind] : undefined;
      return r ? { kind: r.kind, rows, cols, rowBytes: (cols / 256) * r.block, bytes: r.fn(data, rows, cols) as ArrayBufferView<ArrayBuffer> }
               : { kind: t.kind, rows, cols, rowBytes: t.row_bytes, bytes: data as ArrayBufferView<ArrayBuffer> };
    };
    const tensor = async (t: TensorManifest, repack = true): Promise<PackedMatrix> => {
      const data = await fetchTensor(t.file);
      if (!repack) return { kind: t.kind, rows: t.shape[0], cols: t.shape[t.shape.length - 1], rowBytes: t.row_bytes, buffer: gpu.upload(data) };
      const { bytes, ...w } = packed(t, data);
      return { ...w, buffer: gpu.upload(bytes) };
    };
    const floats = async (t: TensorManifest) => gpu.upload(await fetchTensor(t.file));
    if (m.embed) {
      this.embed = [];
      for (const ch of m.embed.chunks) this.embed.push({ ...(await tensor(ch, false)), first: ch.first });
      log(`embedding: ${this.embed.length} chunks`);
    }
    const cacheBytes = this.maxPositions * c.kv_heads * c.head_dim * (this.kv === "f16" ? 2 : 4);
    this.layers = [];
    for (const L of m.layers) {
      const matrices: Record<string, PackedMatrix> = {};
      const vectors: Record<string, GPUBuffer> = {};
      const profile = PROFILES[c.family];
      for (const name of profile.matrices) matrices[name] = await tensor(L.tensors[name]);
      for (const name of profile.vectors) vectors[name] = await floats(L.tensors[name]);
      const experts = async (base: string, rows: number) => {
        const out: ExpertChunk[] = [];
        let first = 0;
        const names = expertChunkNames(L.tensors, base);
        const metas = names.map((name) => L.tensors[name]);
        const rep = metas.length ? (gpu.repack.includes(metas[0].kind) ? REPACK[metas[0].kind] : undefined) : undefined;
        const rowBytesOf = (t: TensorManifest) => rep ? (t.shape[t.shape.length - 1] / 256) * rep.block : t.row_bytes;
        const sizes = metas.map((t) => t.shape[0] * rowBytesOf(t));
        const total = sizes.reduce((a, b) => a + b, 0);
        const limit = Math.min(gpu.device.limits.maxBufferSize, gpu.device.limits.maxStorageBufferBindingSize);
        // Fusion step 1: a layer's expert chunks (each at most WebGPU's default binding size) in one buffer
        // when this device can bind it whole, so each expert product is one dispatch instead of one per chunk.
        if (this.mergeExperts && names.length > 1 && total <= limit && sizes.every((n) => n % 4 === 0) &&
            metas.every((t) => t.kind === metas[0].kind && t.row_bytes === metas[0].row_bytes &&
                               t.shape[t.shape.length - 1] === metas[0].shape[metas[0].shape.length - 1])) {
          const buffer = gpu.buffer(total);
          let offset = 0;
          for (const [i, t] of metas.entries()) {
            const raw = await fetchTensor(t.file);
            if (raw.byteLength !== t.shape[0] * t.row_bytes) throw new Error(`${names[i]}: ${raw.byteLength} bytes, expected ${t.shape[0] * t.row_bytes}`);
            const data = packed(t, raw).bytes;
            gpu.device.queue.writeBuffer(buffer, offset, data);
            offset += sizes[i];
          }
          const allRows = metas.reduce((n, t) => n + t.shape[0], 0);
          return [{ kind: rep ? rep.kind : metas[0].kind, rows: allRows, cols: metas[0].shape[metas[0].shape.length - 1],
                    rowBytes: rowBytesOf(metas[0]), buffer, first: 0, count: allRows / rows }];
        }
        for (const name of names) {
          const t = await tensor(L.tensors[name]);
          const count = t.rows / rows;
          out.push({ ...t, first, count });
          first += count;
        }
        return out;
      };
      const resources: Record<string, ExpertChunk[]> = {};
      for (const [key, [base, kind]] of Object.entries(profile.experts)) resources[key] = await experts(base, expertShape(c, kind)[0]);
      const Ks = Array.from({ length: this.slots }, () => gpu.buffer(cacheBytes));
      const Vs = Array.from({ length: this.slots }, () => gpu.buffer(cacheBytes));
      this.layers.push({ index: L.index, type: L.type, matrices, vectors, ...resources, down: resources.down, K: Ks[0], V: Vs[0], Ks, Vs });
      log(`layer ${L.index} loaded; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
    }
    if (m.head) {
      this.outputNorm = await floats(m.head.norm);
      this.output = [];
      for (const ch of m.head.chunks) this.output.push({ ...(await tensor(ch)), first: ch.first });
      this.vocab = m.head.vocab;
      this.logits = gpu.buffer(this.maxBatch * this.vocab * 4);
      this.token = gpu.buffer(this.maxBatch * 256);
      log(`output: ${this.output.length} chunks; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
    }
    const D = c.head_dim, invFreq = new Float32Array(D / 2);
    for (let i = 0; i < D / 2; i++) invFreq[i] = Math.pow(c.rope_theta, -(2 * i) / D);
    this.invFreq = gpu.upload(invFreq);
    const T = this.maxTokens, K = c.top_k;
    this.ids = gpu.buffer(T * 4);
    for (const key of ["x", "h", "mid"] as const) this[key] = gpu.buffer(T * c.hidden * 4);
    for (const key of ["q", "qn", "att"] as const) this[key] = gpu.buffer(T * c.heads * D * 4);
    for (const key of ["k", "kn", "v"] as const) this[key] = gpu.buffer(T * c.kv_heads * D * 4);
    this.f1 = gpu.buffer(T * (c.shared_ffn ?? 1) * 4);  // the shared expert's (Kolibri-1); unused by qwen3moe
    this.f2 = gpu.buffer(T * (c.shared_ffn ?? 1) * 4);
    this.router = gpu.buffer(T * c.experts * 4);
    this.experts = gpu.buffer(T * K * 4);
    this.expertWeights = gpu.buffer(T * K * 4);
    this.eg = gpu.buffer(T * K * 2 * c.expert_ffn * 4);
    this.ea = gpu.buffer(T * K * c.expert_ffn * 4);
    this.ed = gpu.buffer(T * K * c.hidden * 4);
    const splits = Math.ceil(this.maxPositions / 512);
    this.part = gpu.buffer(T * splits * c.heads * D * 4);
    this.ml = gpu.buffer(T * splits * c.heads * 2 * 4);
    this.loaded = true;
  }

  private checkInput(ids: StageInput["ids"], hidden: StageInput["hidden"], tokens: number) {
    if (this.embed) {
      const vocab = this.embed.reduce((n, chunk) => n + chunk.rows, 0);
      if (hidden != null || !ids || ids.length !== tokens || !Array.from(ids).every((id) => Number.isSafeInteger(id) && id >= 0 && id < vocab))
        throw new Error("Invalid embedding input");
    } else if (ids != null || !(hidden instanceof Float32Array) || hidden.length !== tokens * this.hidden || !hidden.every(Number.isFinite)) {
      throw new Error("Invalid hidden input");
    }
  }

  // Run `tokens` positions of one slot: from token ids (embedding stage) or hidden input [tokens, hidden].
  // WS-32: `gpuIds`: one decode token whose id is already in this.ids (copied there on the GPU from this.token).
  readonly residentDecode = true;  // KV caches only: a step run past the stop rule is undone by rewinding
  async run({ ids = null, hidden = null, tokens, slot = 0, gpuIds = false }: StageInput & { slot?: number; gpuIds?: boolean }): Promise<void> {
    if (!this.loaded) throw new Error("Stage is not loaded");
    const gpu = this.gpu;
    assertStateReady(this, [slot]);
    this.use(slot);
    if (!Number.isSafeInteger(tokens) || tokens < 1) throw new Error("Invalid token count");
    if (gpuIds) { if (!this.embed || tokens !== 1 || ids != null || hidden != null) throw new Error("Invalid resident input"); }
    else this.checkInput(ids, hidden, tokens);
    if (tokens > this.maxTokens) throw new Error(`chunk ${tokens} > ${this.maxTokens}`);
    if (this.position + tokens > this.maxPositions) throw new Error(`position ${this.position + tokens} > ${this.maxPositions}`);
    const upload = () => {
      if (ids) gpu.device.queue.writeBuffer(this.ids, 0, new Uint32Array(ids));
      else if (!gpuIds) gpu.device.queue.writeBuffer(this.x, 0, hidden!);
    };
    this.submitMs = []; this.submitLayers = [];
    await executePlan(this.plan, tokens, gpu, this.kernels(), this.submitMs, this.submitLayers,
      this.planContext(tokens, false, [{slot, row:0, count:tokens, position:this.position}]), upload);
    this.position += tokens;
    this.positions[slot] = this.position;
  }

  // One decode token for each of several slots (rows 0..T-1).
  async runBatch({ slots, ids = null, hidden = null }: { slots: number[] } & Omit<StageInput, "tokens">) {
    assertStateReady(this, slots);
    const gpu = this.gpu, T = slots.length;
    if (!this.loaded || !T) throw Error("Stage unavailable or empty batch");
    this.checkInput(ids, hidden, T);
    if (T > this.maxBatch) throw new Error(`batch ${T} > ${this.maxBatch}`);
    if (new Set(slots).size !== T) throw new Error("a batch holds one token per slot");
    for (const slot of slots) {
      if (!Number.isInteger(slot) || !(slot >= 0 && slot < this.slots)) throw new Error(`slot ${slot} of ${this.slots}`);
      if (this.positions[slot] + 1 > this.maxPositions) throw new Error(`slot ${slot}: position ${this.positions[slot] + 1} > ${this.maxPositions}`);
    }
    const upload = () => {
      if (ids) gpu.device.queue.writeBuffer(this.ids, 0, new Uint32Array(ids));
      else gpu.device.queue.writeBuffer(this.x, 0, hidden!);
    };
    const saved = gpu.prefill, savedGemm = gpu.gemm;
    // WS-45 and experiment 127: a many-token kernel on request, or the GEMM for big batches (summation order differs).
    const big = gpu.batchGemm && T >= gpu.batchGemm.from;
    gpu.prefill = gpu.batchKernel ?? (big ? "gemm" : "matvec");
    if (big) gpu.gemm = gpu.batchGemm!.tile;
    this.submitMs = []; this.submitLayers = [];
    try {
      await executePlan(this.plan, T, gpu, this.kernels(), this.submitMs, this.submitLayers,
        this.planContext(T, true, slots.map((slot,row)=>({slot,row,count:1,position:this.positions[slot]}))), upload);
    } finally {
      gpu.prefill = saved; gpu.gemm = savedGemm;
    }
    for (const slot of slots) this.positions[slot] += 1;
    this.position = this.positions[this.slot];
  }

  // Adapters bind resources and primitives; mathematical composition lives on Tape.
  kernels(): KernelRegistry {
    const gpu = this.gpu;
    return {
      embedding: bindKernel(gpu.gather.bind(gpu)), rms_norm: bindKernel(gpu.norm.bind(gpu)),
      matmul: bindKernel(gpu.matmul.bind(gpu)), silu_mul: bindKernel(gpu.siluMul.bind(gpu)),
      argmax: bindKernel(gpu.argmax.bind(gpu)),
      rotary_halves: bindKernel((x: BufferBinding, inv: BufferBinding, tokens: number, heads: number, dim: number, position: number) =>
        denseRope(gpu, x, inv, tokens, heads, dim, position, "halves")),
      cache_store_grouped: bindKernel(denseStore.bind(null, gpu)),
      windowed_attention: bindKernel(denseWindowedAttention.bind(null, gpu)),
      moe_route: bindKernel(moeRoute.bind(null, gpu)), moe_route_softmax: bindKernel(moeRouteSoftmax.bind(null, gpu)),
      moe_matmul: bindKernel(moeMatmul.bind(null, gpu)),
      silu_mul_halves: bindKernel(siluMulHalves.bind(null, gpu)), moe_combine: bindKernel(moeCombine.bind(null, gpu)),
      add: bindKernel(addInto.bind(null, gpu)),
    };
  }

  planContext(tokens: number, batch: boolean, sequences: SequencePart[]): PlanContext {
    return {
      stateOwner:this,
      buffers: {x:this.x,h:this.h,mid:this.mid,q:this.q,k:this.k,v:this.v,qn:this.qn,kn:this.kn,att:this.att,f1:this.f1,f2:this.f2,
        router:this.router,experts:this.experts,expertWeights:this.expertWeights,eg:this.eg,ea:this.ea,ed:this.ed,part:this.part,ml:this.ml,
        ids:this.ids,invFreq:this.invFreq,outputNorm:this.outputNorm,logits:this.logits,token:this.token},
      layer: index => this.layers[index - this.manifest.layers[0].index],
      resources: {embed:this.embed, output:this.output},
      variables: {tokens,batch,capacity:this.maxPositions,kv:this.kv,vocab:this.vocab,
        head_enabled:true,head_all:batch,head_tokens:batch?tokens:1,head_offset:(tokens-1)*this.hidden},
      sequences,
    };
  }

  // Start a new sequence (the cache needs no clearing: attention reads positions below `position` only).
  async reset(slot: number | null = null) {
    if (slot !== null && (!Number.isInteger(slot) || slot < 0 || slot >= this.slots)) throw Error("Invalid reset slot");
    const resetSlots = slot === null ? [...this.positions.keys()] : [slot];
    await resetPlanState(this, resetSlots, async () => {
      for (const k of slot === null ? [...this.positions.keys()] : [slot]) this.positions[k] = 0;
      this.position = this.positions[this.slot];
    });
  }

  // The layer as direct kernel calls: the regression oracle for the compiled plan (tests compare traces); Kolibri-1 only.
  layer(L: Layer, T: number, parts: { slot: number; position: number; row: number; count: number }[]) {
    const gpu = this.gpu, c = this.config, D = c.head_dim, H = c.heads, KV = c.kv_heads, K = c.top_k, eps = c.eps;
    const row = (buffer: GPUBuffer, r: number, floats: number) => ({ buffer, offset: r * floats * 4 });
    gpu.norm(this.x, L.vectors["attn_norm.weight"], this.h, T, { size: c.hidden, eps });
    gpu.matmul(L.matrices["attn_q.weight"], this.h, this.q, T);
    gpu.matmul(L.matrices["attn_k.weight"], this.h, this.k, T);
    gpu.matmul(L.matrices["attn_v.weight"], this.h, this.v, T);
    gpu.norm(this.q, L.vectors["attn_q_norm.weight"], this.qn, T, { size: D, groups: H, eps });
    gpu.norm(this.k, L.vectors["attn_k_norm.weight"], this.kn, T, { size: D, groups: KV, eps });
    const sliding = L.type === "sliding";
    for (const p of parts) {
      const q = row(this.qn, p.row, H * D), k = row(this.kn, p.row, KV * D), v = row(this.v, p.row, KV * D);
      if (sliding) {
        denseRope(gpu, q, this.invFreq, p.count, H, D, p.position, "halves");
        denseRope(gpu, k, this.invFreq, p.count, KV, D, p.position, "halves");
      }
      denseStore(gpu, k, L.Ks[p.slot], p.count, p.position, this.maxPositions, this.kv, KV, D);
      denseStore(gpu, v, L.Vs[p.slot], p.count, p.position, this.maxPositions, this.kv, KV, D);
      denseWindowedAttention(gpu, q, L.Ks[p.slot], L.Vs[p.slot], row(this.att, p.row, H * D), p.count, p.position, this.maxPositions,
                             this.kv, H, KV, D, c.attention_scale, this.part, this.ml, sliding ? c.window! : 0);
    }
    gpu.matmul(L.matrices["attn_output.weight"], this.att, this.mid, T);
    gpu.norm(this.mid, L.vectors["attn_post_norm.weight"], this.h, T, { size: c.hidden, eps });
    addInto(gpu, this.h, this.x, T * c.hidden);
    gpu.norm(this.x, L.vectors["ffn_norm.weight"], this.h, T, { size: c.hidden, eps });
    gpu.matmul(L.matrices["ffn_gate_inp.weight"], this.h, this.router, T);
    moeRoute(gpu, this.router, L.vectors["exp_probs_b.bias"], this.experts, this.expertWeights, T, c.experts, K);
    gpu.matmul(L.matrices["ffn_gate_shexp.weight"], this.h, this.f1, T);
    gpu.matmul(L.matrices["ffn_up_shexp.weight"], this.h, this.f2, T);
    gpu.siluMul(this.f1, this.f2, T * c.shared_ffn!);
    gpu.matmul(L.matrices["ffn_down_shexp.weight"], this.f2, this.mid, T);
    moeMatmul(gpu, L.gateUp!, this.h, this.eg, this.experts, T, K, false);
    siluMulHalves(gpu, this.eg, this.ea, T * K, c.expert_ffn);
    moeMatmul(gpu, L.down, this.ea, this.ed, this.experts, T, K, true);
    moeCombine(gpu, this.ed, this.expertWeights, this.mid, T, K, c.hidden);
    gpu.norm(this.mid, L.vectors["ffn_post_norm.weight"], this.h, T, { size: c.hidden, eps });
    addInto(gpu, this.h, this.x, T * c.hidden);
  }

  // Logits of the call's last position and its greedy token.
  head(T: number) {
    const gpu = this.gpu;
    gpu.norm(this.x, this.outputNorm, this.h, 1, { size: this.hidden, inOffset: (T - 1) * this.hidden, eps: this.config.eps });
    for (const ch of this.output ?? []) gpu.matmul(ch, this.h, this.logits, 1, { yStride: this.vocab, rowBase: ch.first });
    gpu.argmax(this.logits, this.token, this.vocab);
  }

  async next() {
    const v = await this.gpu.read(this.token, 4);
    return new Uint32Array(v.buffer)[0];
  }

  headBatch(T: number) {
    const gpu = this.gpu;
    gpu.norm(this.x, this.outputNorm, this.h, T, { size: this.hidden, eps: this.config.eps });
    for (const ch of this.output ?? []) gpu.matmul(ch, this.h, this.logits, T, { yStride: this.vocab, rowBase: ch.first });
    for (let t = 0; t < T; t++)
      gpu.argmax({ buffer: this.logits, offset: t * this.vocab * 4 }, { buffer: this.token, offset: t * 256 }, this.vocab);
  }

  async nextBatch(T: number) {
    const v = new Uint32Array((await this.gpu.read(this.token, T * 256)).buffer);
    return Array.from({ length: T }, (_, t) => v[t * 64]);
  }
}
