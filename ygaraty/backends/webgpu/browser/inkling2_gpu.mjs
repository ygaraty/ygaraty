// Generated deployment module: edit inkling2_gpu.mts, then npm run build:webgpu.
import { BYTES, MAX_DISPATCH, deqSource, wgslMemo, objectRecord, positiveInteger, isPackedKind } from "./webgpu_kernels.mjs";
const LOOP_CTRLS = ["q", "k", "v", "ffn"];
const SUBMIT_EVERY_VISIT_FROM = 16;
export let GROUPED_FROM = 8; // runs of at least this many tokens take the tiled per-expert products
export function setGroupedFrom(tokens) { GROUPED_FROM = tokens; } // diagnostics: compare both paths
const NORM_EPS = 1e-6; // the weightless norms of mix and control (llama.cpp's literal)
const MATRICES = new Set(["attn_q", "attn_k", "attn_v", "attn_gate", "attn_output", "attn_rel", "attn_sink_proj", "attn_k_conv_proj",
    "attn_v_conv_proj", "ffn_gate_inp", ...LOOP_CTRLS.flatMap((w) => ["down", "gain", "forget"].map((g) => `loop_${w}_ctrl_${g}`))]);
export function isInkling2Manifest(value) {
    return typeof value === "object" && value !== null && value.config?.family === "inkling2";
}
export function parseInkling2Manifest(value) {
    const m = objectRecord(value, "inkling2 manifest");
    const c = objectRecord(m.config, "inkling2 config");
    for (const key of ["layers", "hidden", "heads", "kv_heads", "head_dim", "vocab", "vocab_unpadded", "experts", "top_k", "expert_ffn",
        "window", "conv", "res_block", "rel_dim", "rel_extent", "log_floor", "loop_end", "passes", "gate_rank"])
        positiveInteger(c[key], key);
    for (const key of ["expert_scale", "eps", "qk_clip", "rel_clip", "log_alpha", "situ_beta", "situ_linear_beta", "logit_cap"])
        if (typeof c[key] !== "number" || !Number.isFinite(c[key]) || c[key] < 0)
            throw new Error(`Invalid ${key}`);
    if (!Number.isSafeInteger(c.loop_start) || c.loop_start < 0 || c.loop_start >= c.loop_end || c.loop_end > c.layers)
        throw new Error("Invalid loop");
    if (!Array.isArray(c.sliding) || c.sliding.length !== c.layers || !c.sliding.every((v) => typeof v === "boolean"))
        throw new Error("Invalid sliding list");
    if (c.head_dim > 128 || c.rel_dim > 32 || c.top_k > 8 || c.experts > 256 || c.heads % c.kv_heads || c.conv > 8)
        throw new Error("Unsupported geometry");
    if (c.loop_start % c.res_block || c.loop_end % c.res_block)
        throw new Error("Loop must align with residual blocks");
    const tensors = objectRecord(m.tensors, "tensors");
    for (const [name, value] of Object.entries(tensors)) {
        const t = objectRecord(value, name);
        if (!isPackedKind(t.kind) || !Array.isArray(t.shape) || !t.shape.length || t.shape.length > 3)
            throw new Error(`Invalid tensor ${name}`);
        for (const n of t.shape)
            positiveInteger(n, `${name} shape`);
        if (!Number.isSafeInteger(t.offset) || t.offset < 0)
            throw new Error(`Invalid offset of ${name}`);
        positiveInteger(t.bytes, `${name} bytes`);
    }
    return m;
}
/** Vectors per token crossing a stage boundary before layer c: every attention-residual block stream started
 * before c (layers 0, res_block, ...), then the current stream. 1 at c = 0 (the embedding's place). */
export function boundaryVectors(c, layer) {
    let n = 1;
    for (let il = 0; il < layer; il++)
        if (il % c.res_block === 0)
            n++;
    return n;
}
/** [(layer, pass)] in execution order. */
export function visits(c) {
    const out = [];
    for (let i = 0; i < c.loop_start; i++)
        out.push([i, 0]);
    for (let p = 0; p < c.passes; p++)
        for (let i = c.loop_start; i < c.loop_end; i++)
            out.push([i, p]);
    for (let i = c.loop_end; i < c.layers; i++)
        out.push([i, 0]);
    return out;
}
// --- kernels ---------------------------------------------------------------------------------------------
const TANH = /* wgsl */ `
fn tnh(x: f32) -> f32 { let e = exp(-2.0 * abs(x)); return sign(x) * (1.0 - e) / (1.0 + e); }
fn sgm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }`;
// Attention over depth: out[q * block + t] = sum_s softmax_s(rms(state_s[t]) . query_q / sqrt(H)) rms(state_s[t]), with
// the states pool slots idx[0..S) (the current stream last). One workgroup per token.
const MIX = /* wgsl */ `
struct P { tokens: u32, hidden: u32, queries: u32, states: u32, slot: u32, eps: f32, block: u32, p1: u32, idx: array<vec4<u32>, 2> };
@group(0) @binding(0) var<storage, read> pool: array<f32>;
@group(0) @binding(1) var<storage, read> query: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 1280>;
var<workgroup> inv: array<f32, 8>;
var<workgroup> wgt: array<f32, 32>;
fn slot_of(s: u32) -> u32 { return p.idx[s / 4u][s % 4u]; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x;
  let H = p.hidden;
  for (var s = 0u; s < p.states; s++) {
    let base = slot_of(s) * p.slot + t * H;
    var acc = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
    for (var e = li.x; e < H; e += 256u) {
      let v = pool[base + e];
      acc[0] += v * v;
      for (var q = 0u; q < p.queries; q++) { acc[q + 1u] += v * query[q * H + e]; }
    }
    for (var k = 0u; k < 5u; k++) { red[k * 256u + li.x] = acc[k]; }
    workgroupBarrier();
    for (var n = 128u; n > 0u; n >>= 1u) {
      if (li.x < n) { for (var k = 0u; k < 5u; k++) { red[k * 256u + li.x] += red[k * 256u + li.x + n]; } }
      workgroupBarrier();
    }
    if (li.x == 0u) {
      let r = inverseSqrt(red[0] / f32(H) + p.eps);
      inv[s] = r;
      for (var q = 0u; q < p.queries; q++) { wgt[q * 8u + s] = red[(q + 1u) * 256u] * r / sqrt(f32(H)); }
    }
    workgroupBarrier();
  }
  if (li.x == 0u) {
    for (var q = 0u; q < p.queries; q++) {
      var m = -3.0e38;
      for (var s = 0u; s < p.states; s++) { m = max(m, wgt[q * 8u + s]); }
      var z = 0.0;
      for (var s = 0u; s < p.states; s++) { let v = exp(wgt[q * 8u + s] - m); wgt[q * 8u + s] = v; z += v; }
      for (var s = 0u; s < p.states; s++) { wgt[q * 8u + s] = wgt[q * 8u + s] / z * inv[s]; }
    }
  }
  workgroupBarrier();
  for (var e = li.x; e < H; e += 256u) {
    for (var q = 0u; q < p.queries; q++) {
      var v = 0.0;
      for (var s = 0u; s < p.states; s++) { v += wgt[q * 8u + s] * pool[slot_of(s) * p.slot + t * H + e]; }
      y[q * p.block + t * H + e] = v;
    }
  }
}`;
// Causal depthwise convolution with K-1 rows of history: y[t,c] = (x[t,c] + sum_k w[c,k] x[t-K+1+k, c]
// [+ sum_lag dyn[t, head*K + lag] x[t-lag, c]]) * scale. History rows are oldest first.
const CONV = /* wgsl */ `
struct P { tokens: u32, channels: u32, taps: u32, heads: u32, dynamic: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> dyn: array<f32>;
@group(0) @binding(3) var<storage, read> hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
fn at(t: i32, c: u32) -> f32 {
  if (t >= 0) { return x[u32(t) * p.channels + c]; }
  return hist[u32(t + i32(p.taps) - 1) * p.channels + c];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.tokens * p.channels) { return; }
  let t = i / p.channels; let c = i % p.channels; let K = p.taps;
  var out = 0.0;
  for (var k = 0u; k < K; k++) { out += w[c * K + k] * at(i32(t) - i32(K) + 1 + i32(k), c); }
  if (p.dynamic == 1u) {
    let head = c / (p.channels / p.heads);
    for (var lag = 0u; lag < K; lag++) { out += dyn[t * p.heads * K + head * K + lag] * at(i32(t) - i32(lag), c); }
  }
  y[i] = (x[i] + out) * p.scale;
}`;
// The history after a call: the last K-1 rows of [hist; x]. One thread per channel walks the rows in order, so
// it reads every old row before overwriting it.
const SHIFT = /* wgsl */ `
struct P { tokens: u32, channels: u32, taps: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.channels) { return; }
  let keep = p.taps - 1u;
  for (var r = 0u; r < keep; r++) {
    let t = i32(p.tokens) - i32(keep) + i32(r);  // row r of the new history is token t of [hist; x]
    if (t >= 0) { hist[r * p.channels + c] = x[u32(t) * p.channels + c]; }
    else { hist[r * p.channels + c] = hist[u32(t + i32(keep)) * p.channels + c]; }
  }
}`;
// DO-96: one token, convolution and history shift in one dispatch: a thread per channel reads its K-1 history rows,
// writes y, then the new history (rows moved up one, x last). Same arithmetic, in the same order, as CONV + SHIFT.
const CONV1 = /* wgsl */ `
struct P { tokens: u32, channels: u32, taps: u32, heads: u32, dynamic: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> dyn: array<f32>;
@group(0) @binding(3) var<storage, read_write> hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.channels) { return; }
  let K = p.taps; let keep = K - 1u;
  var h: array<f32, 8>;
  for (var r = 0u; r < keep; r++) { h[r] = hist[r * p.channels + c]; }
  let xc = x[c];
  var out = 0.0;
  for (var k = 0u; k < K; k++) { out += w[c * K + k] * select(h[k], xc, k == keep); }
  if (p.dynamic == 1u) {
    let head = c / (p.channels / p.heads);
    for (var lag = 0u; lag < K; lag++) { out += dyn[head * K + lag] * select(h[keep - min(lag, keep)], xc, lag == 0u); }
  }
  y[c] = (xc + out) * p.scale;
  for (var r = 0u; r + 1u < keep; r++) { hist[r * p.channels + c] = h[r + 1u]; }
  if (keep > 0u) { hist[(keep - 1u) * p.channels + c] = xc; }
}`;
// DO-98: a loop control for one token in two dispatches instead of six. CTRL_DOWN: gate[r] = W[r, :H] . x / rms(x) +
// W[r, H:] . m / rms(m) (the weightless norms of x and m folded into the down product; their sums of squares reduce
// as the norm kernel's do), a workgroup per row. CTRL_MIX: per channel e, g = gain[e] . gate, f = forget[e] . gate,
// out = x - tanh(f) x + tanh(g) m (elementwise op 4). Same products as the generic kernels in another order.
const CTRL_DOWN = /* wgsl */ `
struct P { hidden: u32, rank: u32, eps: f32, p0: u32 };
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<vec4<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = wg.x; let H = p.hidden; let base = r * 2u * H;
  var a = vec4<f32>(0.0);
  for (var e = li.x; e < H; e += 256u) {
    let xv = x[e]; let mv = m[e];
    a += vec4<f32>(xv * xv, mv * mv, w[base + e] * xv, w[base + H + e] * mv);
  }
  red[li.x] = a;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li.x < k) { red[li.x] += red[li.x + k]; }
    workgroupBarrier();
  }
  if (li.x == 0u) {
    let s = red[0];
    gate[r] = s.z * inverseSqrt(s.x / f32(H) + p.eps) + s.w * inverseSqrt(s.y / f32(H) + p.eps);
  }
}`;
const CTRL_MIX = /* wgsl */ `
struct P { hidden: u32, rank: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> gain: array<f32>;
@group(0) @binding(1) var<storage, read> forget: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> m: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
${TANH}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= p.hidden) { return; }
  var g = 0.0; var f = 0.0;
  for (var k = 0u; k < p.rank; k++) { let z = gate[k]; g += gain[e * p.rank + k] * z; f += forget[e * p.rank + k] * z; }
  y[e] = x[e] - tnh(f) * x[e] + tnh(g) * m[e];
}`;
// Per (token, group) row of `size`: x * scale (optional, per group and channel), then either an L2 clip to
// `threshold` (mode 0) or an RMS norm (mode 1); then times tau(position) when `logscale`. In place.
const ROWS = /* wgsl */ `
struct P { tokens: u32, groups: u32, size: u32, mode: u32, threshold: f32, eps: f32, use_scale: u32, logscale: u32,
           position: u32, floor: f32, alpha: f32, p0: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.y; let g = wg.x;
  let base = (t * p.groups + g) * p.size;
  var v = 0.0;
  if (li.x < p.size) {
    v = x[base + li.x];
    if (p.use_scale == 1u) { v = v * scale[g * p.size + li.x]; }
  }
  red[li.x] = v * v;
  workgroupBarrier();
  for (var n = 64u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] += red[li.x + n]; } workgroupBarrier(); }
  var f = 1.0;
  if (p.mode == 0u) { f = clamp(p.threshold / (sqrt(red[0]) + 1e-6), 0.0, 1.0); }
  else { f = inverseSqrt(red[0] / f32(p.size) + p.eps); }
  if (p.logscale == 1u) { f = f * (1.0 + p.alpha * log(max(1.0, (f32(p.position + t) + 1.0) / p.floor))); }
  if (li.x < p.size) { x[base + li.x] = v * f; }
}`;
// DO-98: ROWS for Q, K, V and R of one visit in one dispatch: segment i's rows are workgroups [sum of earlier groups,
// + groups_i); each row does ROWS's arithmetic in ROWS's order. Segment 1/2 (K, V) take no scale (sc2/sc3 unused).
const ROWS4 = /* wgsl */ `
struct S { groups: u32, size: u32, mode: u32, use_scale: u32, threshold: f32, eps: f32, logscale: u32, p0: u32 };
struct P { tokens: u32, position: u32, floor: f32, alpha: f32, s0: S, s1: S, s2: S, s3: S };
@group(0) @binding(0) var<storage, read_write> x0: array<f32>;
@group(0) @binding(1) var<storage, read_write> x1: array<f32>;
@group(0) @binding(2) var<storage, read_write> x2: array<f32>;
@group(0) @binding(3) var<storage, read_write> x3: array<f32>;
@group(0) @binding(4) var<storage, read> sc0: array<f32>;
@group(0) @binding(5) var<storage, read> sc3: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> red: array<f32, 128>;
var<private> sc1: array<f32, 1>;
var<private> sc2: array<f32, 1>;
var<private> t: u32;
fn rowfactor(v: f32, i: u32, s: S) -> f32 {
  red[i] = v * v;
  workgroupBarrier();
  for (var n = 64u; n > 0u; n >>= 1u) { if (i < n) { red[i] += red[i + n]; } workgroupBarrier(); }
  var f = 1.0;
  if (s.mode == 0u) { f = clamp(s.threshold / (sqrt(red[0]) + 1e-6), 0.0, 1.0); }
  else { f = inverseSqrt(red[0] / f32(s.size) + s.eps); }
  if (s.logscale == 1u) { f = f * (1.0 + p.alpha * log(max(1.0, (f32(p.position + t) + 1.0) / p.floor))); }
  return f;
}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  t = wg.y; var g = wg.x;
    if (g < p.s0.groups) {
      let base = (t * p.s0.groups + g) * p.s0.size;
      var v = 0.0;
      if (li.x < p.s0.size) { v = x0[base + li.x]; if (p.s0.use_scale == 1u) { v = v * sc0[g * p.s0.size + li.x]; } }
      let f = rowfactor(v, li.x, p.s0);
      if (li.x < p.s0.size) { x0[base + li.x] = v * f; }
      return;
    }
    g -= p.s0.groups;
    if (g < p.s1.groups) {
      let base = (t * p.s1.groups + g) * p.s1.size;
      var v = 0.0;
      if (li.x < p.s1.size) { v = x1[base + li.x]; if (p.s1.use_scale == 1u) { v = v * sc1[g * p.s1.size + li.x]; } }
      let f = rowfactor(v, li.x, p.s1);
      if (li.x < p.s1.size) { x1[base + li.x] = v * f; }
      return;
    }
    g -= p.s1.groups;
    if (g < p.s2.groups) {
      let base = (t * p.s2.groups + g) * p.s2.size;
      var v = 0.0;
      if (li.x < p.s2.size) { v = x2[base + li.x]; if (p.s2.use_scale == 1u) { v = v * sc2[g * p.s2.size + li.x]; } }
      let f = rowfactor(v, li.x, p.s2);
      if (li.x < p.s2.size) { x2[base + li.x] = v * f; }
      return;
    }
    g -= p.s2.groups;
    if (g < p.s3.groups) {
      let base = (t * p.s3.groups + g) * p.s3.size;
      var v = 0.0;
      if (li.x < p.s3.size) { v = x3[base + li.x]; if (p.s3.use_scale == 1u) { v = v * sc3[g * p.s3.size + li.x]; } }
      let f = rowfactor(v, li.x, p.s3);
      if (li.x < p.s3.size) { x3[base + li.x] = v * f; }
      return;
    }
    g -= p.s3.groups;
}`;
// K and V rows into a visit's cache, at row (position + t) % rows.
const STORE = /* wgsl */ `
struct P { tokens: u32, width: u32, position: u32, rows: u32 };
@group(0) @binding(0) var<storage, read> k: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<f32>;
@group(0) @binding(3) var<storage, read_write> vc: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.tokens * p.width) { return; }
  let t = i / p.width; let e = i % p.width;
  let row = (p.position + t) % p.rows;
  kc[row * p.width + e] = k[i];
  vc[row * p.width + e] = v[i];
}`;
// Attention without RoPE: per query token and head, softmax over the visible keys of
// q.k * scale + r . bias[distance] - sink (bias zero at distance >= extent), with an extra zero logit (the
// sink), weighting V. Online softmax over blocks of 128 keys; one workgroup per (head, token), a thread per
// output channel. window 0: every earlier key (global visits).
const ATTENTION = /* wgsl */ `
struct P { tokens: u32, position: u32, heads: u32, kv_heads: u32, dim: u32, rows: u32, window: u32, extent: u32,
           rel: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> sink: array<f32>;
@group(0) @binding(6) var<storage, read_write> o: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
var<workgroup> qs: array<f32, 128>;
var<workgroup> rs: array<f32, 32>;
var<workgroup> sc: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let h = wg.x; let t = wg.y; let D = p.dim;
  let kvh = h / (p.heads / p.kv_heads);
  let pos = p.position + t;
  if (li.x < D) { qs[li.x] = q[(t * p.heads + h) * D + li.x]; }
  if (li.x < p.rel) { rs[li.x] = r[(t * p.heads + h) * p.rel + li.x]; }
  let s0 = sink[t * p.heads + h];
  workgroupBarrier();
  var first = 0u;
  if (p.window > 0u && pos + 1u > p.window) { first = pos + 1u - p.window; }
  var m = 0.0;  // the zero sink logit: running max starts there, with its exp(0) = 1 in the sum
  var l = 1.0;
  var acc = 0.0;
  for (var kb = first; kb <= pos; kb += 128u) {
    let j = kb + li.x;
    var s = -3.0e38;
    if (j <= pos) {
      let row = j % p.rows;
      var dot = 0.0;
      for (var d = 0u; d < D; d++) { dot += qs[d] * kc[(row * p.kv_heads + kvh) * D + d]; }
      let dist = pos - j;
      var b = 0.0;
      if (dist < p.extent) { for (var e = 0u; e < p.rel; e++) { b += rs[e] * bias[dist * p.rel + e]; } }
      s = dot * p.scale + b - s0;
    }
    red[li.x] = s;
    workgroupBarrier();
    for (var n = 64u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] = max(red[li.x], red[li.x + n]); } workgroupBarrier(); }
    let mn = max(m, red[0]);
    workgroupBarrier();
    var pj = 0.0;
    if (j <= pos) { pj = exp(s - mn); }
    sc[li.x] = pj;
    red[li.x] = pj;
    workgroupBarrier();
    for (var n = 64u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] += red[li.x + n]; } workgroupBarrier(); }
    let corr = exp(m - mn);
    l = l * corr + red[0];
    acc = acc * corr;
    if (li.x < D) {
      let count = min(128u, pos + 1u - kb);
      for (var jj = 0u; jj < count; jj++) { acc += sc[jj] * vc[(((kb + jj) % p.rows) * p.kv_heads + kvh) * D + li.x]; }
    }
    m = mn;
    workgroupBarrier();
  }
  if (li.x < D) { o[(t * p.heads + h) * D + li.x] = acc / l; }
}`;
// DO-101: single-token attention split over positions. ATTN_SPLIT: one workgroup per (head, 128-position block);
// each writes its block's max logit m, sum l = sum exp(s - m) and acc = sum exp(s - m) v ([m, l, acc[D]] at
// part[(h * p0 + block) * (D + 2)]). ATTN_MERGE: one workgroup per head adds the blocks and the zero sink logit.
// Same terms as ATTENTION, summed in another order.
const ATTN_SPLIT = /* wgsl */ `
struct P { tokens: u32, position: u32, heads: u32, kv_heads: u32, dim: u32, rows: u32, window: u32, extent: u32,
           rel: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> sink: array<f32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
var<workgroup> qs: array<f32, 128>;
var<workgroup> rs: array<f32, 32>;
var<workgroup> sc: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let h = wg.x; let blk = wg.y; let D = p.dim;
  let kvh = h / (p.heads / p.kv_heads);
  let pos = p.position;
  if (li.x < D) { qs[li.x] = q[h * D + li.x]; }
  if (li.x < p.rel) { rs[li.x] = r[h * p.rel + li.x]; }
  let s0 = sink[h];
  workgroupBarrier();
  var first = 0u;
  if (p.window > 0u && pos + 1u > p.window) { first = pos + 1u - p.window; }
  let kb = first + blk * 128u;
  let j = kb + li.x;
  var s = -3.0e38;
  if (j <= pos) {
    let row = j % p.rows;
    var dot = 0.0;
    for (var d = 0u; d < D; d++) { dot += qs[d] * kc[(row * p.kv_heads + kvh) * D + d]; }
    let dist = pos - j;
    var b = 0.0;
    if (dist < p.extent) { for (var e = 0u; e < p.rel; e++) { b += rs[e] * bias[dist * p.rel + e]; } }
    s = dot * p.scale + b - s0;
  }
  red[li.x] = s;
  workgroupBarrier();
  for (var n = 64u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] = max(red[li.x], red[li.x + n]); } workgroupBarrier(); }
  let mx = red[0];
  workgroupBarrier();
  var pj = 0.0;
  if (j <= pos) { pj = exp(s - mx); }
  sc[li.x] = pj;
  red[li.x] = pj;
  workgroupBarrier();
  for (var n = 64u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] += red[li.x + n]; } workgroupBarrier(); }
  let base = (h * p.p0 + blk) * (D + 2u);
  if (li.x < D) {
    var acc = 0.0;
    let count = select(0u, min(128u, pos + 1u - kb), kb <= pos);
    for (var jj = 0u; jj < count; jj++) { acc += sc[jj] * vc[(((kb + jj) % p.rows) * p.kv_heads + kvh) * D + li.x]; }
    part[base + 2u + li.x] = acc;
  }
  if (li.x == 0u) { part[base] = mx; part[base + 1u] = red[0]; }
}`;
const ATTN_MERGE = /* wgsl */ `
struct P { tokens: u32, position: u32, heads: u32, kv_heads: u32, dim: u32, rows: u32, window: u32, extent: u32,
           rel: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let h = wg.x; let D = p.dim;
  if (li.x >= D) { return; }
  var M = 0.0;  // the sink's zero logit
  for (var b = 0u; b < p.p1; b++) { M = max(M, part[(h * p.p0 + b) * (D + 2u)]); }
  var L = exp(-M);
  var acc = 0.0;
  for (var b = 0u; b < p.p1; b++) {
    let base = (h * p.p0 + b) * (D + 2u);
    let w = exp(part[base] - M);
    L += w * part[base + 1u];
    acc += w * part[base + 2u + li.x];
  }
  o[h * D + li.x] = acc / L;
}`;
// Elementwise operations, chosen by `op`:
// 0: y = a * sigmoid(b)         1: y = a + b          2: y = a
// 3: y = cap * tanh(a[at + i] / cap)    5: y = a[at + i]    4: y = a - tanh(c) a + tanh(d) b  (loop control: a = x, b = message, c = forget, d = gain)
const ELEMENTWISE = /* wgsl */ `
struct P { n: u32, op: u32, cap: f32, at: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
${TANH}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.n) { return; }
  switch p.op {
    case 0u: { y[i] = a[i] * sgm(b[i]); }
    case 1u: { y[i] = a[i] + b[i]; }
    case 2u: { y[i] = a[i]; }
    case 3u: { y[i] = p.cap * tnh(a[p.at + i] / p.cap); }
    case 5u: { y[i] = a[p.at + i]; }
    default: { y[i] = a[i] - tnh(c[i]) * a[i] + tnh(d[i]) * b[i]; }
  }
}`;
// DO-98: residual stream updates inside the pool through one read_write binding (WebGPU forbids binding the pool twice,
// which is why the general path goes through tmp): pool[out + i] = pool[a + i] + b[i] (add = 1) or pool[a + i].
const POOL_OP = /* wgsl */ `
struct P { n: u32, a: u32, out: u32, add: u32 };
@group(0) @binding(0) var<storage, read_write> pool: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  var v = pool[p.a + i];
  if (p.add == 1u) { v = v + b[i]; }
  pool[p.out + i] = v;
}`;
// Router: per token, sigmoid scores; top k on score + bias (ties to the lower id, as the reference's stable
// sort); weights scale * p / (sum p + 1e-20). forced = 1: the ids are given, only the weights are computed.
// DO-96: one workgroup per token, a thread per expert; each of the k picks is a (value, lower id) max reduction,
// which chooses what the serial scan chose. Same scores, same order of the weight sum: bit for bit as before.
const ROUTE = /* wgsl */ `
struct P { tokens: u32, experts: u32, k: u32, forced: u32, scale: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> score: array<f32, 256>;
var<workgroup> taken: array<u32, 256>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> chosen: array<u32, 8>;
fn sgm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x; let E = p.experts; let K = p.k; let e = li.x;
  if (e < E) {
    let s = sgm(logits[t * E + e]);
    logits[t * E + e] = s;  // scores, read back by the routing callback
    score[e] = s;
  }
  taken[e] = 0u;
  workgroupBarrier();
  if (p.forced == 1u) {
    if (e < K) { chosen[e] = ids[t * K + e]; }
  } else {
    for (var k = 0u; k < K; k++) {
      var v = -3.0e38; var id = 0xffffffffu;
      if (e < E && taken[e] == 0u) { v = score[e] + bias[e]; id = e; }
      rv[e] = v; ri[e] = id;
      workgroupBarrier();
      for (var n = 128u; n > 0u; n >>= 1u) {
        if (e < n) {
          let ov = rv[e + n]; let oi = ri[e + n];
          if (ov > rv[e] || (ov == rv[e] && oi < ri[e])) { rv[e] = ov; ri[e] = oi; }
        }
        workgroupBarrier();
      }
      if (e == 0u) { chosen[k] = ri[0]; taken[ri[0]] = 1u; }
      workgroupBarrier();
    }
  }
  workgroupBarrier();
  if (e == 0u) {
    var z = 0.0;
    for (var k = 0u; k < K; k++) { z += score[chosen[k]]; }
    for (var k = 0u; k < K; k++) {
      ids[t * K + k] = chosen[k];
      wts[t * K + k] = p.scale * score[chosen[k]] / (z + 1e-20);
    }
  }
}`;
// Report 156's router: one thread per token (kept for DO-96's same-machine baseline, FAST_DECODE.router = false).
const ROUTE_SERIAL = /* wgsl */ `
struct P { tokens: u32, experts: u32, k: u32, forced: u32, scale: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> wts: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
fn sgm(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= p.tokens) { return; }
  let E = p.experts; let K = p.k;
  for (var e = 0u; e < E; e++) { logits[t * E + e] = sgm(logits[t * E + e]); }
  var chosen: array<u32, 8>;
  if (p.forced == 1u) {
    for (var k = 0u; k < K; k++) { chosen[k] = ids[t * K + k]; }
  } else {
    for (var k = 0u; k < K; k++) {
      var best = 0u; var bv = -3.0e38;
      for (var e = 0u; e < E; e++) {
        var taken = false;
        for (var j = 0u; j < k; j++) { if (chosen[j] == e) { taken = true; } }
        let v = logits[t * E + e] + bias[e];
        if (!taken && v > bv) { bv = v; best = e; }
      }
      chosen[k] = best;
    }
  }
  var z = 0.0;
  for (var k = 0u; k < K; k++) { z += logits[t * E + chosen[k]]; }
  for (var k = 0u; k < K; k++) {
    ids[t * K + k] = chosen[k];
    wts[t * K + k] = p.scale * logits[t * E + chosen[k]] / (z + 1e-20);
  }
}`;
// Routed expert rows: y[slot, row] = W_e[row] . x[slot / div], e = ids[slot], for the experts of this chunk.
// One workgroup per (row, slot).
const expertShaderSource = (kind) => /* wgsl */ `
struct P { rows: u32, cols: u32, row_bytes: u32, first: u32, count: u32, slots: u32, div: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
${BYTES}
${deqSource(kind)}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.x;
  let slot = wg.y + wg.z * ${MAX_DISPATCH}u;
  if (slot >= p.slots) { return; }
  let e = ids[slot];
  if (e < p.first || e >= p.first + p.count) { return; }
  let base = ((e - p.first) * p.rows + row) * p.row_bytes;
  let xb = (slot / p.div) * p.cols;
  var acc = 0.0;
  for (var c = li.x; c < p.cols; c += 64u) { acc += deq(base, c) * x[xb + c]; }
  red[li.x] = acc;
  workgroupBarrier();
  for (var n = 32u; n > 0u; n >>= 1u) { if (li.x < n) { red[li.x] += red[li.x + n]; } workgroupBarrier(); }
  if (li.x == 0u) { y[slot * p.rows + row] = red[0]; }
}`;
const expertShader = wgslMemo(expertShaderSource);
// DO-96: one-token expert products. 8 rows per workgroup of 64, 8 threads per row; thread k takes blocks k, k+8,
// ... of its row and decodes each block word by word (4 IQ4_NL nibble words or 8 Q8_0 byte words, vec4 products
// with x). Mode 0 (gate_up + SiTU): each row j yields gate row j and up row j + ffn, and writes
// sy[slot, j] = SiTU(gate, up): no gu buffer, no SiTU dispatch. Mode 1 (down): y[slot, row] = W_e[row] . x[slot].
// x of slot s is row s / div. Same products as the generic kernels in another summation order.
const EXPERT_BLOCK = {
    IQ4_NL: { bytes: 18, dot: /* wgsl */ `
  let d = half_at(bb);
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) {
    let q = word_at(bb + 2u + 4u * i);
    let lo = vec4<f32>(tbl[q & 15u], tbl[(q >> 8u) & 15u], tbl[(q >> 16u) & 15u], tbl[(q >> 24u) & 15u]);
    let hi = vec4<f32>(tbl[(q >> 4u) & 15u], tbl[(q >> 12u) & 15u], tbl[(q >> 20u) & 15u], tbl[q >> 28u]);
    s += dot(lo, x[xo + i]) + dot(hi, x[xo + 4u + i]);
  }
  return d * s;` },
    Q8_0: { bytes: 34, dot: /* wgsl */ `
  let d = half_at(bb);
  var s = 0.0;
  for (var i = 0u; i < 8u; i++) {
    let v = bitcast<vec4<i32>>(bytes4(word_at(bb + 2u + 4u * i)) << vec4<u32>(24u)) >> vec4<u32>(24u);
    s += dot(vec4<f32>(v), x[xo + i]);
  }
  return d * s;` },
};
const expertVecSource = (kind, mode) => /* wgsl */ `
struct P { rows: u32, cols: u32, row_bytes: u32, first: u32, count: u32, slots: u32, div: u32, ffn: u32, beta: f32, lbeta: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 128>;
var<workgroup> tbl: array<f32, 16>;
${BYTES}
${TANH}
fn blockdot(bb: u32, xo: u32) -> f32 {${EXPERT_BLOCK[kind].dot}
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  if (li.x < 16u) {
    var t = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
    tbl[li.x] = t[li.x];
  }
  workgroupBarrier();
  let slot = wg.y;
  let e = ids[slot];
  if (slot >= p.slots || e < p.first || e >= p.first + p.count) { return; }
  let r = li.x / 8u; let k = li.x % 8u;
  let row = wg.x * 8u + r;
  let out_rows = select(p.rows, p.ffn, ${mode}u == 0u);
  let live = row < out_rows;
  let blocks = p.cols / 32u;
  let xb = (slot / p.div) * (p.cols / 4u);
  let base = (e - p.first) * p.rows;
  var a = 0.0; var b = 0.0;
  if (live) {
    for (var j = k; j < blocks; j += 8u) {
      a += blockdot((base + row) * p.row_bytes + j * ${EXPERT_BLOCK[kind].bytes}u, xb + j * 8u);
      ${mode === 0 ? `b += blockdot((base + row + p.ffn) * p.row_bytes + j * ${EXPERT_BLOCK[kind].bytes}u, xb + j * 8u);` : ""}
    }
  }
  red[li.x] = a; red[64u + li.x] = b;
  workgroupBarrier();
  for (var n = 4u; n > 0u; n >>= 1u) {
    if (k < n) { red[li.x] += red[li.x + n]; red[64u + li.x] += red[64u + li.x + n]; }
    workgroupBarrier();
  }
  if (k == 0u && live) {
    let g = red[li.x];
    ${mode === 0
    ? `let u = red[64u + li.x]; y[slot * p.ffn + row] = p.beta * tnh(g / p.beta) * sgm(g) * p.lbeta * tnh(u / p.lbeta);`
    : `y[slot * p.rows + row] = g;`}
  }
}`;
const expertVec = wgslMemo(expertVecSource);
// DO-96: one-token kernels, each switchable for A/B checks (false: the kernels of report 156).
export const FAST_DECODE = { experts: true, conv: true, router: true, dense: true, chunks: true,
    ctrl: true, rows: true, group: true, pool: true, pipeline: true, // DO-98
    attn: true }; // DO-101: single-token attention split over 128-position blocks
const PIPELINE_VISITS = 8;
export function setFastDecode(flags) { Object.assign(FAST_DECODE, flags); }
// Prefill: the token slots grouped by expert. One workgroup counts each expert's slots, writes their offsets in
// expert order, then each slot's index into `list` (order within an expert is arbitrary: every output depends
// on its own slot only).
const BUCKET = /* wgsl */ `
struct P { slots: u32, experts: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> list: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> cnt: array<atomic<u32>, 256>;
var<workgroup> base: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
  atomicStore(&cnt[li.x], 0u);
  workgroupBarrier();
  for (var s = li.x; s < p.slots; s += 256u) { atomicAdd(&cnt[ids[s]], 1u); }
  workgroupBarrier();
  if (li.x == 0u) {
    var run = 0u;
    for (var e = 0u; e < p.experts; e++) {
      let n = atomicLoad(&cnt[e]);
      base[e] = run; offsets[e] = run; counts[e] = n; run += n;
    }
  }
  workgroupBarrier();
  atomicStore(&cnt[li.x], 0u);
  workgroupBarrier();
  for (var s = li.x; s < p.slots; s += 256u) {
    let e = ids[s];
    list[base[e] + atomicAdd(&cnt[e], 1u)] = s;
  }
}`;
// Prefill expert products, tiled: a workgroup takes 32 rows of one expert and 32 of that expert's slots, so
// each weight row is decoded once per 32 slots (not once per slot). y[slot, row] = W_e[row] . x[slot / div].
const groupedShaderSource = (kind) => /* wgsl */ `
struct P { rows: u32, cols: u32, row_bytes: u32, first: u32, div: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(3) var<storage, read> counts: array<u32>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> ws: array<f32, 1056>;
var<workgroup> xs: array<f32, 1056>;
${BYTES}
${deqSource(kind)}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let e = p.first + wg.y;
  let n = counts[e]; let off = offsets[e];
  let s0 = wg.z * 32u;
  if (s0 >= n) { return; }
  let r0 = wg.x * 32u;
  let tx = li.x % 8u; let ty = li.x / 8u;
  var acc: array<array<f32, 4>, 4>;
  for (var c0 = 0u; c0 < p.cols; c0 += 32u) {
    for (var k = 0u; k < 16u; k++) {
      let i = li.x + k * 64u; let r = i / 32u; let cc = i % 32u; let col = c0 + cc;
      let row = r0 + r;
      var wv = 0.0;
      if (row < p.rows && col < p.cols) { wv = deq((wg.y * p.rows + row) * p.row_bytes, col); }
      ws[r * 33u + cc] = wv;
      var xv = 0.0;
      if (s0 + r < n && col < p.cols) { xv = x[(list[off + s0 + r] / p.div) * p.cols + col]; }
      xs[r * 33u + cc] = xv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      for (var i = 0u; i < 4u; i++) {
        let a = ws[(ty * 4u + i) * 33u + kk];
        for (var j = 0u; j < 4u; j++) { acc[i][j] += a * xs[(tx * 4u + j) * 33u + kk]; }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let row = r0 + ty * 4u + i; let sp = s0 + tx * 4u + j;
      if (row < p.rows && sp < n) { y[list[off + sp] * p.rows + row] = acc[i][j]; }
    }
  }
}`;
const groupedShader = wgslMemo(groupedShaderSource);
// SiTU over fused gate/up rows: y[s, i] = beta tanh(g / beta) sigmoid(g) * lbeta tanh(u / lbeta).
const SITU = /* wgsl */ `
struct P { slots: u32, ffn: u32, beta: f32, lbeta: f32 };
@group(0) @binding(0) var<storage, read> gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
${TANH}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.slots * p.ffn) { return; }
  let s = i / p.ffn; let j = i % p.ffn;
  let g = gu[s * 2u * p.ffn + j]; let u = gu[s * 2u * p.ffn + p.ffn + j];
  y[i] = p.beta * tnh(g / p.beta) * sgm(g) * p.lbeta * tnh(u / p.lbeta);
}`;
// y[t, e] = sum_k wts[t, k] * rows[t * K + k, e]
const COMBINE = /* wgsl */ `
struct P { tokens: u32, hidden: u32, k: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> rows: array<f32>;
@group(0) @binding(1) var<storage, read> wts: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.tokens * p.hidden) { return; }
  let t = i / p.hidden; let e = i % p.hidden;
  var v = 0.0;
  for (var k = 0u; k < p.k; k++) { v += wts[t * p.k + k] * rows[(t * p.k + k) * p.hidden + e]; }
  y[i] = v;
}`;
// Stage hand-off: x[t][v][e] <-> pool slot idx[v] row t (dir 0: pool to x; dir 1: x to pool).
const PACK = /* wgsl */ `
struct P { tokens: u32, hidden: u32, vectors: u32, slot: u32, dir: u32, p0: u32, p1: u32, p2: u32, idx: array<vec4<u32>, 2> };
@group(0) @binding(0) var<storage, read_write> pool: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.tokens * p.vectors * p.hidden) { return; }
  let t = i / (p.vectors * p.hidden); let v = (i / p.hidden) % p.vectors; let e = i % p.hidden;
  let pi = p.idx[v / 4u][v % 4u] * p.slot + t * p.hidden + e;
  if (p.dir == 0u) { x[i] = pool[pi]; } else { pool[pi] = x[i]; }
}`;
const grid = (n) => { const g = Math.ceil(n / 256); return [Math.min(g, MAX_DISPATCH), Math.ceil(g / MAX_DISPATCH), 1]; };
const f32 = (b, floats, at = 0) => ({ buffer: b, offset: at * 4, size: Math.max(4, floats * 4) });
const aligned = (floats) => Math.ceil(floats / 64) * 64; // block strides: bindings start on 256 bytes
export class Inkling2Stage {
    gpu;
    config;
    chunk;
    capacity;
    attnBlocks;
    logitRows;
    first;
    last;
    inputVectors; // 0 on the first stage (token ids)
    outputVectors; // 0 on the last stage (logits and a token)
    order; // this stage's visits
    globalVisit; // their indices among the model's visits
    position = 0;
    manifest;
    read;
    options;
    layers = [];
    states = [];
    embed = [];
    head = null;
    embedNorm;
    b = {};
    ones;
    slots;
    blockCount;
    token;
    constructor(gpu, manifest, read, options) {
        this.gpu = gpu;
        this.manifest = manifest;
        this.config = manifest.config;
        this.read = read;
        this.options = options;
        this.chunk = options.chunk ?? 512;
        this.capacity = options.capacity ?? 4096;
        this.attnBlocks = Math.ceil(this.capacity / 128) + 1; // DO-101: 128-position blocks a decode step's attention can span
        this.logitRows = options.logitRows ?? 64;
        const c = this.config;
        this.first = options.first ?? 0;
        this.last = options.last ?? c.layers;
        if (!(0 <= this.first && this.first < this.last && this.last <= c.layers))
            throw new Error("Invalid stage layers");
        if (this.first < c.loop_end && this.last > c.loop_start && (this.first > c.loop_start || this.last < c.loop_end))
            throw new Error("A stage must hold the whole loop core or none of it");
        this.inputVectors = this.first ? boundaryVectors(c, this.first) : 0;
        this.outputVectors = this.last < c.layers ? boundaryVectors(c, this.last) : 0;
        if (Math.max(this.inputVectors, this.outputVectors) > 8)
            throw new Error("At most 8 vectors per hand-off");
        const all = visits(c);
        this.globalVisit = all.map((_, i) => i).filter((i) => all[i][0] >= this.first && all[i][0] < this.last);
        this.order = this.globalVisit.map((i) => all[i]);
        this.blockCount = Math.ceil(this.config.layers / this.config.res_block);
        this.slots = 2 * this.blockCount + 4; // blocks, bank, bank output, two working streams, FFN input
    }
    static async load(gpu, manifest, read, options = {}) {
        const s = new Inkling2Stage(gpu, manifest, read, options);
        // DO-96: one-token dense products through the coalesced fast decoders (Q8_0 included): 76 to 38 ms per token
        // on the 680M. A deployment's own choice (fixture matvec_c / q8_fast_up_to) is kept.
        if (FAST_DECODE.dense) {
            if (!gpu.matvecC && !gpu.generic)
                gpu.matvecC = { tpr: 16, lanes: 4 };
            gpu.q8FastUpTo = Math.max(gpu.q8FastUpTo, 1);
            if (FAST_DECODE.group)
                gpu.groupMatmul = true; // DO-98: used by this stage's one-token products only when shapes allow
        }
        await s.allocate();
        return s;
    }
    tensor(name) {
        const t = this.manifest.tensors[name];
        if (!t)
            throw new Error(`Missing tensor ${name}`);
        return t;
    }
    has(name) { return name in this.manifest.tensors; }
    async bytes(name) {
        const t = this.tensor(name), data = await this.read(t.offset, t.bytes, name);
        const view = data instanceof Uint8Array ? data : new Uint8Array(data);
        if (view.byteLength !== t.bytes)
            throw new Error(`Short read of ${name}`);
        return view;
    }
    async matrix(name) {
        const t = this.tensor(name);
        if (t.shape.length !== 2)
            throw new Error(`${name} is not a matrix`);
        return { kind: t.kind, rows: t.shape[0], cols: t.shape[1], rowBytes: t.bytes / t.shape[0], buffer: this.gpu.upload(await this.bytes(name)) };
    }
    async vector(name) {
        if (this.tensor(name).kind !== "F32")
            throw new Error(`${name} must be F32`);
        return this.gpu.upload(await this.bytes(name));
    }
    // Row chunks of whole groups (experts or vocabulary rows), each at most `limit` bytes.
    async chunks(name, groupRows, limit, rows) {
        const t = this.tensor(name);
        const cols = t.shape.at(-1), total = rows ?? t.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const rowBytes = t.bytes / t.shape.slice(0, -1).reduce((a, b) => a * b, 1);
        const per = Math.max(1, Math.floor(limit / (groupRows * rowBytes)));
        const out = [];
        for (let first = 0; first * groupRows < total; first += per) {
            const count = Math.min(per, Math.ceil(total / groupRows) - first);
            const n = Math.min(count * groupRows, total - first * groupRows);
            const data = await this.read(t.offset + first * groupRows * rowBytes, n * rowBytes, name);
            out.push({ kind: t.kind, rows: groupRows === 1 ? n : groupRows, cols, rowBytes, first, count, buffer: this.gpu.upload(data instanceof Uint8Array ? data : new Uint8Array(data)) });
        }
        return out;
    }
    async loadExperts(il) {
        // DO-96: chunks as large as the device binds (at most 1 GiB): a layer's gate_up and down in one buffer each,
        // so a visit dispatches 2 expert products instead of 5. Chunking does not change the arithmetic.
        const L = this.gpu.device.limits;
        const c = this.config, limit = this.options.expertChunkBytes
            ?? (FAST_DECODE.chunks ? Math.min(2 ** 30, L.maxStorageBufferBindingSize, L.maxBufferSize) : 128 * 2 ** 20);
        return {
            gateUp: await this.chunks(`blk.${il}.ffn_gate_up_exps.weight`, 2 * c.expert_ffn, limit),
            down: await this.chunks(`blk.${il}.ffn_down_exps.weight`, c.hidden, limit),
        };
    }
    async allocate() {
        const c = this.config, g = this.gpu, T = this.chunk, H = c.hidden, K = c.top_k;
        const vec = (n) => g.buffer(n * 4);
        for (let il = this.first; il < this.last; il++) {
            const prefix = `blk.${il}.`, layer = { matrices: {}, vectors: {}, experts: null };
            for (const name of Object.keys(this.manifest.tensors)) {
                if (!name.startsWith(prefix) || name.includes("_exps."))
                    continue;
                const short = name.slice(prefix.length).replace(/\.(weight|bias)$/, "");
                if (MATRICES.has(short))
                    layer.matrices[short] = await this.matrix(name);
                else
                    layer.vectors[short] = await this.vector(name);
            }
            if (this.options.residentExperts?.(il) ?? true)
                layer.experts = await this.loadExperts(il);
            this.layers[il] = layer;
        }
        const vocabLimit = 128 * 2 ** 20;
        if (this.first === 0) {
            this.embed = await this.chunks("token_embd.weight", 1, vocabLimit);
            this.embedNorm = await this.vector("token_embd_norm.weight");
        }
        if (this.last === c.layers && (this.options.head ?? true)) {
            const head = this.has("output.weight") ? "output.weight" : "token_embd.weight";
            this.head = { chunks: await this.chunks(head, 1, vocabLimit, c.vocab_unpadded), norm: await this.vector("output_norm.weight"),
                query: await this.vector("output_res_query.weight") };
        }
        for (const [il] of this.order) {
            const rows = c.sliding[il] ? c.window + T : this.capacity, kv = c.kv_heads * c.head_dim, keep = c.conv - 1;
            this.states.push({ k: vec(rows * kv), v: vec(rows * kv), rows,
                conv: { k: vec(keep * kv), v: vec(keep * kv), attn: vec(keep * H), ffn: vec(keep * H) } });
        }
        const q = c.heads * c.head_dim, kv = c.kv_heads * c.head_dim, dynW = c.kv_heads * c.conv;
        const sizes = {
            pool: this.slots * aligned(T * H), mix: 4 * aligned(T * H), msg: 4 * aligned(T * H), ctl: 4 * aligned(T * H), xin: 3 * aligned(T * H), cat: 2 * T * H,
            gate: T * c.gate_rank, gain: T * H, forget: T * H, q: T * q, kpre: T * kv, vpre: T * kv, k: T * kv, v: T * kv, vn: T * kv,
            dynk: T * dynW, dynv: T * dynW, r: T * c.heads * c.rel_dim, sink: T * c.heads, g: T * q, o: T * q, og: T * q, a: T * H, a2: T * H,
            fmix: T * H, fctl: T * H, fn: T * H, rlog: T * c.experts, wts: T * K, gu: T * K * 2 * c.expert_ffn, sy: T * K * c.expert_ffn,
            yd: T * K * H, moe: T * H, y2: T * H, tmp: T * H, hn: T * H, hx: T * H,
            raw: this.logitRows * c.vocab_unpadded, logits: this.logitRows * c.vocab_unpadded, last: c.vocab_unpadded,
            streams: T * Math.max(1, this.inputVectors, this.outputVectors) * H,
            part: c.heads * this.attnBlocks * (c.head_dim + 2), // DO-101
        };
        for (const [name, n] of Object.entries(sizes))
            this.b[name] = vec(n);
        this.b.ids = g.buffer(T * K * 4);
        this.b.list = g.buffer(T * K * 4);
        this.b.counts = g.buffer(c.experts * 4);
        this.b.offsets = g.buffer(c.experts * 4);
        this.b.tokens = g.buffer(T * 4);
        this.token = g.buffer(4);
        this.ones = g.upload(new Float32Array(Math.max(H, 2 * H, 128)).fill(1));
        await this.reset();
    }
    /** Empty every visit's caches and histories; the next run starts at position 0. */
    async reset() {
        const enc = this.gpu.device.createCommandEncoder();
        for (const s of this.states)
            for (const b of [s.k, s.v, ...Object.values(s.conv)])
                enc.clearBuffer(b);
        this.gpu.device.queue.submit([enc.finish()]);
        await this.gpu.device.queue.onSubmittedWorkDone();
        this.position = 0;
    }
    // --- operations ---
    // DO-98: pool[slot out] = pool[slot a] (+ b), one token, one dispatch.
    poolOp(a, out, b) {
        const H = this.config.hidden, stride = aligned(this.chunk * H);
        const u = this.gpu.uniform([["u", H], ["u", a * stride], ["u", out * stride], ["u", b ? 1 : 0]]);
        this.gpu.dispatch("ink_pool_op", POOL_OP, [this.b.pool, b ?? this.ones, u], [Math.ceil(H / 256), 1, 1]);
    }
    slot(i, T) { return f32(this.b.pool, T * this.config.hidden, i * aligned(this.chunk * this.config.hidden)); }
    block(name, j, T) { return f32(this.b[name], T * this.config.hidden, j * aligned(T * this.config.hidden)); }
    mix(states, query, queries, out, T) {
        if (states.length > 8 || queries > 4)
            throw new Error("mix: at most 8 states and 4 queries");
        const idx = [...states, ...Array(8 - states.length).fill(0)];
        const u = this.gpu.uniform([["u", T], ["u", this.config.hidden], ["u", queries], ["u", states.length], ["u", aligned(this.chunk * this.config.hidden)],
            ["f", NORM_EPS], ["u", aligned(T * this.config.hidden)], ["u", 0], ...idx.map((v) => ["u", v])]);
        this.gpu.dispatch("ink_mix", MIX, [this.b.pool, query, out, u], [T, 1, 1]);
    }
    elementwise(op, n, y, a, b, c, d, cap = 0, at = 0) {
        const u = this.gpu.uniform([["u", n], ["u", op], ["f", cap], ["u", at]]);
        this.gpu.dispatch("ink_elementwise", ELEMENTWISE, [a, b ?? a, c ?? a, d ?? a, y, u], grid(n));
    }
    conv(x, w, dyn, hist, y, T, channels) {
        const c = this.config, heads = dyn ? c.kv_heads : 1;
        const u = this.gpu.uniform([["u", T], ["u", channels], ["u", c.conv], ["u", heads], ["u", dyn ? 1 : 0],
            ["f", 1 / Math.sqrt((dyn ? 2 : 1) * c.conv + 1)], ["u", 0], ["u", 0]]);
        if (T === 1 && FAST_DECODE.conv) { // DO-96: decode, one dispatch
            this.gpu.dispatch("ink_conv1", CONV1, [x, w, dyn ?? w, hist, y, u], [Math.ceil(channels / 256), 1, 1]);
            return;
        }
        this.gpu.dispatch("ink_conv", CONV, [x, w, dyn ?? w, hist, y, u], grid(T * channels));
        const v = this.gpu.uniform([["u", T], ["u", channels], ["u", c.conv], ["u", 0]]);
        this.gpu.dispatch("ink_shift", SHIFT, [x, hist, v], [Math.ceil(channels / 256), 1, 1]);
    }
    rows(x, T, groups, size, mode, threshold, scale, logscale, eps = 0) {
        const c = this.config;
        const u = this.gpu.uniform([["u", T], ["u", groups], ["u", size], ["u", mode], ["f", threshold], ["f", eps], ["u", scale ? 1 : 0],
            ["u", logscale ? 1 : 0], ["u", this.position], ["f", c.log_floor], ["f", c.log_alpha], ["u", 0]]);
        this.gpu.dispatch("ink_rows", ROWS, [x, scale ?? this.ones, u], [groups, T, 1]);
    }
    norm(x, w, y, T, eps, outStride, outOffset = 0) {
        this.gpu.norm(x, w, y, T, { size: this.config.hidden, eps, outStride, outOffset });
    }
    control(il, which, x, m, out, T) {
        const L = this.layers[il], H = this.config.hidden, n = `loop_${which}_ctrl_`;
        const M = L.matrices, R = this.config.gate_rank;
        if (T === 1 && FAST_DECODE.ctrl && [M[n + "down"], M[n + "gain"], M[n + "forget"]].every((W) => W.kind === "F32")) {
            const du = this.gpu.uniform([["u", H], ["u", R], ["f", NORM_EPS], ["u", 0]]);
            this.gpu.dispatch("ink_ctrl_down", CTRL_DOWN, [M[n + "down"].buffer, x, m, this.b.gate, du], [R, 1, 1]);
            const mu = this.gpu.uniform([["u", H], ["u", R], ["u", 0], ["u", 0]]);
            this.gpu.dispatch("ink_ctrl_mix", CTRL_MIX, [M[n + "gain"].buffer, M[n + "forget"].buffer, this.b.gate, x, m, out, mu], [Math.ceil(H / 256), 1, 1]);
            return;
        }
        this.norm(x, this.ones, this.b.cat, T, NORM_EPS, 2 * H, 0);
        this.norm(m, this.ones, this.b.cat, T, NORM_EPS, 2 * H, H);
        this.gpu.matmul(L.matrices[n + "down"], this.b.cat, this.b.gate, T);
        this.gpu.matmul(L.matrices[n + "gain"], this.b.gate, this.b.gain, T);
        this.gpu.matmul(L.matrices[n + "forget"], this.b.gate, this.b.forget, T);
        this.elementwise(4, T * H, out, x, m, this.b.forget, this.b.gain);
    }
    experts(chunks, x, y, slots, div) {
        const T = slots / this.config.top_k;
        if (T >= GROUPED_FROM) { // prefill: tiled per expert over the slots bucketed by `bucket`
            for (const ch of chunks) {
                const u = this.gpu.uniform([["u", ch.rows], ["u", ch.cols], ["u", ch.rowBytes], ["u", ch.first], ["u", div], ["u", 0], ["u", 0], ["u", 0]]);
                this.gpu.dispatch(`ink_grouped_${ch.kind}`, groupedShader(ch.kind), [ch.buffer, x, this.b.list, this.b.counts, this.b.offsets, y, u], [Math.ceil(ch.rows / 32), ch.count, Math.ceil(T / 32)]); // an expert has at most one slot per token
            }
            return;
        }
        for (const ch of chunks) {
            const u = this.gpu.uniform([["u", ch.rows], ["u", ch.cols], ["u", ch.rowBytes], ["u", ch.first], ["u", ch.count], ["u", slots], ["u", div], ["u", 0]]);
            this.gpu.dispatch(`ink_expert_${ch.kind}`, expertShader(ch.kind), [ch.buffer, x, this.b.ids, y, u], [ch.rows, Math.min(slots, MAX_DISPATCH), Math.ceil(slots / MAX_DISPATCH)]);
        }
    }
    expertsVec(chunks, x, y, slots, div, mode) {
        const c = this.config;
        for (const ch of chunks) {
            if (ch.cols % 32)
                throw new Error("expertsVec: columns in whole blocks");
            const u = this.gpu.uniform([["u", ch.rows], ["u", ch.cols], ["u", ch.rowBytes], ["u", ch.first], ["u", ch.count], ["u", slots], ["u", div],
                ["u", c.expert_ffn], ["f", c.situ_beta], ["f", c.situ_linear_beta], ["u", 0], ["u", 0]]);
            const out = mode === 0 ? c.expert_ffn : ch.rows;
            this.gpu.dispatch(`ink_expvec${mode ? "_down" : "_gu"}_${ch.kind}`, expertVec(ch.kind, mode), [ch.buffer, x, this.b.ids, y, u], [Math.ceil(out / 8), slots, 1]);
        }
    }
    // One visit of layer il in pass p, from the stream in slot `cur` into slot `out`; `blocks` are the block
    // stream slots (a block-start layer's input slot is pushed, as the reference appends `inp`).
    visit(v, il, p, cur, out, blocks, T, forced) {
        const c = this.config, g = this.gpu, L = this.layers[il], st = this.states[v], H = c.hidden;
        const sliding = c.sliding[il], M = L.matrices, V = L.vectors, B = this.b;
        const kv = c.kv_heads * c.head_dim, q = c.heads * c.head_dim;
        const inp = cur;
        let srcs = [this.slot(cur, T), this.slot(cur, T), this.slot(cur, T)];
        if (V.attn_res_query) {
            const Q = sliding ? 1 : 3;
            this.mix([...blocks, cur], V.attn_res_query, Q, B.mix, T);
            srcs = [0, 1, 2].map((j) => this.block("mix", sliding ? 0 : j, T));
        }
        const looped = p > 0;
        if (looped)
            srcs = [0, 1, 2].map((j) => { const o = this.block("ctl", j, T); this.control(il, LOOP_CTRLS[j], srcs[j], this.block("msg", j, T), o, T); return o; });
        const distinct = sliding && !looped ? 1 : 3;
        for (let j = 0; j < distinct; j++)
            this.norm(srcs[j], sliding ? V.attn_norm : V[`attn_${"qkv"[j]}_inp_norm`], this.block("xin", j, T), T, c.eps);
        const xq = this.block("xin", 0, T), xk = this.block("xin", distinct === 1 ? 0 : 1, T), xv = this.block("xin", distinct === 1 ? 0 : 2, T);
        // DO-98: one-token products of the same input in one dispatch (WS-36's grouped kernel, matvecC's sums).
        const grouped = T === 1 && FAST_DECODE.group && g.groupMatmul;
        const product = (items, x) => {
            if (grouped && g.groupShape(items.map(([W]) => W)))
                g.matmulGroup(items.map(([W, y]) => ({ W, y })), x);
            else
                for (const [W, y] of items)
                    g.matmul(W, x, y, T);
        };
        product([[M.attn_q, B.q], [M.attn_rel, B.r], [M.attn_sink_proj, B.sink], [M.attn_gate, B.g]], xq);
        if (!sliding) {
            product([[M.attn_k, B.kpre], [M.attn_k_conv_proj, B.dynk]], xk);
            product([[M.attn_v, B.vpre], [M.attn_v_conv_proj, B.dynv]], xv);
        }
        else if (distinct === 1)
            product([[M.attn_k, B.kpre], [M.attn_v, B.vpre]], xk);
        else {
            product([[M.attn_k, B.kpre]], xk);
            product([[M.attn_v, B.vpre]], xv);
        }
        this.conv(f32(B.kpre, T * kv), V.attn_k_conv, sliding ? null : B.dynk, st.conv.k, f32(B.k, T * kv), T, kv);
        this.conv(f32(B.vpre, T * kv), V.attn_v_conv, sliding ? null : B.dynv, st.conv.v, f32(B.v, T * kv), T, kv);
        if (T === 1 && FAST_DECODE.rows) {
            const S = (groups, size, mode, scale, threshold, eps, logscale) => [["u", groups], ["u", size], ["u", mode], ["u", scale ? 1 : 0], ["f", threshold], ["f", eps], ["u", logscale ? 1 : 0], ["u", 0]];
            const u = g.uniform([["u", T], ["u", this.position], ["f", c.log_floor], ["f", c.log_alpha],
                ...S(c.heads, c.head_dim, 0, !!V.attn_q_clip, c.qk_clip, 0, !sliding), ...S(c.kv_heads, c.head_dim, 0, false, c.qk_clip, 0, false),
                ...S(c.kv_heads, c.head_dim, 1, false, 0, c.eps, false), ...S(c.heads, c.rel_dim, 0, !!V.attn_rel_clip, c.rel_clip, 0, !sliding)]);
            g.dispatch("ink_rows4", ROWS4, [f32(B.q, T * q), f32(B.k, T * kv), f32(B.v, T * kv), f32(B.r, T * c.heads * c.rel_dim),
                V.attn_q_clip ?? this.ones, V.attn_rel_clip ?? this.ones, u], [2 * c.heads + 2 * c.kv_heads, T, 1]);
        }
        else {
            this.rows(f32(B.q, T * q), T, c.heads, c.head_dim, 0, c.qk_clip, V.attn_q_clip, !sliding);
            this.rows(f32(B.k, T * kv), T, c.kv_heads, c.head_dim, 0, c.qk_clip, null, false);
            this.rows(f32(B.v, T * kv), T, c.kv_heads, c.head_dim, 1, 0, null, false, c.eps);
            this.rows(f32(B.r, T * c.heads * c.rel_dim), T, c.heads, c.rel_dim, 0, c.rel_clip, V.attn_rel_clip, !sliding);
        }
        const su = g.uniform([["u", T], ["u", kv], ["u", this.position], ["u", st.rows]]);
        g.dispatch("ink_store", STORE, [B.k, B.v, st.k, st.v, su], grid(T * kv));
        const au = g.uniform([["u", T], ["u", this.position], ["u", c.heads], ["u", c.kv_heads], ["u", c.head_dim], ["u", st.rows],
            ["u", sliding ? c.window : 0], ["u", c.rel_extent], ["u", c.rel_dim], ["f", 1 / Math.sqrt(c.head_dim)], ["u", 0], ["u", 0]]);
        if (T === 1 && FAST_DECODE.attn) {
            const first = sliding && this.position + 1 > c.window ? this.position + 1 - c.window : 0;
            const blocks = Math.ceil((this.position + 1 - first) / 128), stride = this.attnBlocks;
            const pu = g.uniform([["u", T], ["u", this.position], ["u", c.heads], ["u", c.kv_heads], ["u", c.head_dim], ["u", st.rows],
                ["u", sliding ? c.window : 0], ["u", c.rel_extent], ["u", c.rel_dim], ["f", 1 / Math.sqrt(c.head_dim)], ["u", stride], ["u", blocks]]);
            g.dispatch("ink_attn_split", ATTN_SPLIT, [B.q, st.k, st.v, B.r, V.attn_rel_bias, B.sink, B.part, pu], [c.heads, blocks, 1]);
            g.dispatch("ink_attn_merge", ATTN_MERGE, [B.part, B.o, pu], [c.heads, 1, 1]);
        }
        else
            g.dispatch("ink_attention", ATTENTION, [B.q, st.k, st.v, B.r, V.attn_rel_bias, B.sink, B.o, au], [c.heads, T, 1]);
        this.elementwise(0, T * q, f32(B.og, T * q), f32(B.o, T * q), f32(B.g, T * q));
        g.matmul(M.attn_output, B.og, B.a, T);
        this.conv(f32(B.a, T * H), V.attn_post_conv, null, st.conv.attn, f32(B.a2, T * H), T, H);
        const F = this.slots - 1; // the FFN input slot
        const pooled = T === 1 && FAST_DECODE.pool;
        if (il % c.res_block === 0) {
            blocks.push(inp);
            this.elementwise(2, T * H, this.slot(F, T), f32(B.a2, T * H));
        }
        else if (pooled)
            this.poolOp(inp, F, f32(B.a2, T * H));
        else {
            this.elementwise(1, T * H, f32(B.tmp, T * H), this.slot(inp, T), f32(B.a2, T * H));
            this.elementwise(2, T * H, this.slot(F, T), f32(B.tmp, T * H));
        }
        this.mix([...blocks, F], V.ffn_res_query, 1, B.fmix, T);
        let f = f32(B.fmix, T * H);
        if (looped) {
            this.control(il, "ffn", f, this.block("msg", 3, T), f32(B.fctl, T * H), T);
            f = f32(B.fctl, T * H);
        }
        this.norm(f, V.ffn_norm, B.fn, T, c.eps);
        if (!V.exp_probs_b)
            throw new Error("inkling2 needs exp_probs_b");
        g.matmul(M.ffn_gate_inp, B.fn, B.rlog, T);
        const ru = g.uniform([["u", T], ["u", c.experts], ["u", c.top_k], ["u", forced ? 1 : 0], ["f", c.expert_scale], ["u", 0], ["u", 0], ["u", 0]]);
        if (FAST_DECODE.router)
            g.dispatch("ink_route", ROUTE, [B.rlog, V.exp_probs_b, B.ids, B.wts, ru], [T, 1, 1]);
        else
            g.dispatch("ink_route_serial", ROUTE_SERIAL, [B.rlog, V.exp_probs_b, B.ids, B.wts, ru], [Math.ceil(T / 64), 1, 1]);
        const E = L.experts, slots = T * c.top_k;
        if (T >= GROUPED_FROM) {
            const bu = g.uniform([["u", slots], ["u", c.experts], ["u", 0], ["u", 0]]);
            g.dispatch("ink_bucket", BUCKET, [B.ids, B.counts, B.offsets, B.list, bu], [1, 1, 1]);
        }
        if (T === 1 && FAST_DECODE.experts && E.gateUp.every((ch) => ch.kind in EXPERT_BLOCK) && E.down.every((ch) => ch.kind in EXPERT_BLOCK)) {
            this.expertsVec(E.gateUp, B.fn, B.sy, slots, c.top_k, 0);
            this.expertsVec(E.down, B.sy, B.yd, slots, 1, 1);
        }
        else {
            this.experts(E.gateUp, B.fn, B.gu, slots, c.top_k);
            const tu = g.uniform([["u", slots], ["u", c.expert_ffn], ["f", c.situ_beta], ["f", c.situ_linear_beta]]);
            g.dispatch("ink_situ", SITU, [B.gu, B.sy, tu], grid(slots * c.expert_ffn));
            this.experts(E.down, B.sy, B.yd, slots, 1);
        }
        const cu = g.uniform([["u", T], ["u", H], ["u", c.top_k], ["u", 0]]);
        g.dispatch("ink_combine", COMBINE, [B.yd, B.wts, B.moe, cu], grid(T * H));
        this.conv(f32(B.moe, T * H), V.ffn_post_conv, null, st.conv.ffn, f32(B.y2, T * H), T, H);
        // A dispatch may not read and write the same buffer (the pool), even in disjoint ranges: through tmp.
        if (pooled) {
            this.poolOp(F, out, f32(B.y2, T * H));
            return;
        }
        this.elementwise(1, T * H, f32(B.tmp, T * H), this.slot(F, T), f32(B.y2, T * H));
        this.elementwise(2, T * H, this.slot(out, T), f32(B.tmp, T * H));
    }
    /** Runs `ids` (at most `chunk` tokens) from the current position; greedy next token in `token`. Returns
     * softcapped logits (last row, every row, or none). */
    async run(opts) {
        const c = this.config, g = this.gpu, H = c.hidden, B = this.b;
        const T = this.first === 0 ? (opts.ids?.length ?? 0) : (opts.hidden?.length ?? 0) / (this.inputVectors * H);
        if (!Number.isInteger(T) || T < 1 || T > this.chunk)
            throw new Error(`run: 1..${this.chunk} tokens of ${this.first ? "streams" : "ids"}`);
        if (this.position + T > this.capacity)
            throw new Error("run: past the KV capacity");
        if (this.first === 0)
            g.device.queue.writeBuffer(B.tokens, 0, Uint32Array.from(opts.ids));
        else
            g.device.queue.writeBuffer(B.streams, 0, opts.hidden);
        const order = this.order, nb = this.blockCount;
        const bankSlot = (i) => nb + i, bankOut = 2 * nb, work = [2 * nb + 1, 2 * nb + 2];
        const copy = (from, to) => {
            if (from === to)
                return;
            if (T === 1 && FAST_DECODE.pool) {
                this.poolOp(from, to, null);
                return;
            }
            this.elementwise(2, T * H, f32(B.tmp, T * H), this.slot(from, T));
            this.elementwise(2, T * H, this.slot(to, T), f32(B.tmp, T * H));
        };
        g.begin();
        let cur = 0, blocks = [], preludeBlocks = [], bank = [];
        if (this.first === 0) {
            g.gather(this.embed, B.tokens, B.hx, T);
            this.norm(B.hx, this.embedNorm, this.slot(0, T), T, c.eps);
        }
        else { // the incoming blocks into their block slots, the current stream into a working slot
            blocks = Array.from({ length: this.inputVectors - 1 }, (_, i) => i);
            cur = work[0];
            this.pack(1, [...blocks, cur], T);
        }
        for (let v = 0; v < order.length; v++) {
            const [il, p] = order[v], gv = this.globalVisit[v];
            if (il === c.loop_start) {
                if (p === 0)
                    preludeBlocks = [...blocks];
                else {
                    const tail = blocks.slice(preludeBlocks.length + 1);
                    bank = tail.map((s, i) => { copy(s, bankSlot(i)); return bankSlot(i); });
                    copy(cur, bankOut);
                    blocks = [...preludeBlocks];
                    cur = preludeBlocks.length; // the prelude output, kept in the block slot layer loop_start pushed
                }
            }
            if (il >= c.loop_start && il < c.loop_end && p > 0)
                this.mix([...bank, bankOut], this.layers[il].vectors.loop_msg_query, 4, B.msg, T);
            if (il % c.res_block === 0 && il > 0) {
                copy(cur, blocks.length);
                cur = blocks.length;
            }
            let L = this.layers[il], streamed = false;
            if (!L.experts) {
                await g.end();
                L.experts = await this.loadExperts(il);
                streamed = true;
                g.begin();
            }
            const out = cur === work[0] ? work[1] : work[0];
            const forced = opts.forced?.(gv) ?? null;
            if (forced) { // earlier visits of this submission write the ids buffer too: flush, then upload
                if (forced.length !== T * c.top_k)
                    throw new Error("forced routing: tokens * top_k ids");
                await g.end();
                g.device.queue.writeBuffer(B.ids, 0, forced);
                g.begin();
            }
            this.visit(v, il, p, cur, out, blocks, T, !!forced);
            cur = out;
            // DO-98: one token, submitted every PIPELINE_VISITS visits without waiting, so the GPU runs the first visits while
            // the host encodes the rest (WS-32's deferred end(); queue order keeps every later write after this work).
            if (T === 1 && FAST_DECODE.pipeline && (v + 1) % PIPELINE_VISITS === 0 && v + 1 < order.length
                && !(opts.trace || opts.routing || streamed)) {
                const defer = g.defer;
                g.defer = true;
                try {
                    await g.end();
                }
                finally {
                    g.defer = defer;
                }
                g.begin();
            }
            // Long runs submit per visit: one submission must stay under GPU watchdogs (amdgpu stops a job after ~10 s).
            if (opts.trace || opts.routing || streamed || T >= SUBMIT_EVERY_VISIT_FROM) {
                await g.end();
                if (opts.trace) {
                    const at = cur * aligned(this.chunk * H);
                    opts.trace(gv, il, p, (await g.read(B.pool, (at + T * H) * 4)).subarray(at));
                }
                if (opts.routing) {
                    const raw = await this.readBytes(B.ids, T * c.top_k * 4);
                    opts.routing(gv, new Uint32Array(raw), (await g.read(B.rlog, T * c.experts * 4)));
                }
                if (streamed) {
                    for (const ch of [...L.experts.gateUp, ...L.experts.down])
                        g.release(ch.buffer);
                    L.experts = null;
                }
                g.begin();
            }
        }
        let result = null;
        const mode = opts.logits ?? "last";
        if (this.outputVectors) { // hand-off: [tokens][vectors][hidden] in `x`
            if (blocks.length + 1 !== this.outputVectors)
                throw new Error("Stage output differs from its boundary's streams");
            this.pack(0, [...blocks, cur], T);
            await g.end();
            this.outputTokens = T;
        }
        else if (this.head && mode !== "none") {
            this.mix([...blocks, cur], this.head.query, 1, B.hx, T);
            this.norm(B.hx, this.head.norm, B.hn, T, c.eps);
            const rows = mode === "all" ? T : 1, at = mode === "all" ? 0 : T - 1, Vu = c.vocab_unpadded;
            if (rows > this.logitRows)
                throw new Error(`run: at most ${this.logitRows} logit rows`);
            for (const ch of this.head.chunks)
                g.matmul(ch, B.hn, B.raw, rows, { xOffset: at * H, yStride: Vu, yOffset: ch.first });
            this.elementwise(c.logit_cap ? 3 : 2, rows * Vu, f32(B.logits, rows * Vu), f32(B.raw, rows * Vu), undefined, undefined, undefined, c.logit_cap);
            this.elementwise(c.logit_cap ? 3 : 5, Vu, f32(B.last, Vu), f32(B.raw, rows * Vu), undefined, undefined, undefined, c.logit_cap, (rows - 1) * Vu);
            g.argmax(B.last, this.token, Vu);
            await g.end();
            result = await g.read(B.logits, rows * Vu * 4);
        }
        else
            await g.end();
        this.position += T;
        return result;
    }
    outputTokens = 0;
    /** The hand-off of the last run, [tokens][outputVectors][hidden], blocks then the current stream. */
    get x() { return this.b.streams; }
    async readStreams() {
        return this.gpu.read(this.b.streams, this.outputTokens * this.outputVectors * this.config.hidden * 4);
    }
    pack(dir, slots, T) {
        const H = this.config.hidden, V = slots.length, idx = [...slots, ...Array(8 - V).fill(0)];
        const u = this.gpu.uniform([["u", T], ["u", H], ["u", V], ["u", aligned(this.chunk * H)], ["u", dir], ["u", 0], ["u", 0], ["u", 0],
            ...idx.map((v) => ["u", v])]);
        this.gpu.dispatch("ink_pack", PACK, [this.b.pool, this.b.streams, u], grid(T * V * H));
    }
    async readBytes(buffer, bytes) {
        const staging = this.gpu.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = this.gpu.device.createCommandEncoder();
        enc.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
        this.gpu.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const out = staging.getMappedRange().slice(0);
        staging.unmap();
        staging.destroy();
        return out;
    }
    get logitsBuffer() { return this.b.logits; }
    async readToken() { return new Uint32Array(await this.readBytes(this.token, 4))[0]; }
}
/** The stage interface qwen35_contributor.mjs drives (Qwen3.5, dense and Kolibri stages have the same shape): an
 * Inkling 2 stage of layers [first, last). Its hand-off is `vectors` streams per token, so `hidden` (the floats
 * per token the transport sizes and reads from `x`) is vectors * hidden. One conversation; no resident decode. */
export class Inkling2ContributorStage {
    gpu;
    manifest;
    plan;
    output;
    vocab;
    hidden;
    stage = null;
    maxTokens;
    maxPositions;
    constructor(gpu, manifest, maxTokens, maxPositions, options = {}) {
        const m = parseInkling2Manifest(manifest);
        if ((options.slots ?? 1) !== 1)
            throw new Error("Inkling 2 stages serve one conversation");
        if (!m.plan || m.plan.layers?.[0] !== m.first || m.plan.layers?.[1] !== m.last)
            throw new Error("Missing or mismatched inkling2 stage plan");
        const c = m.config;
        const vin = m.first ? boundaryVectors(c, m.first) : 0, vout = m.last < c.layers ? boundaryVectors(c, m.last) : 0;
        if (m.plan.input_vectors !== vin || m.plan.output_vectors !== vout)
            throw new Error("Stage plan vectors differ from its layers");
        this.gpu = gpu;
        this.manifest = m;
        this.plan = m.plan;
        this.maxTokens = maxTokens;
        this.maxPositions = maxPositions;
        this.output = m.last === c.layers;
        this.vocab = c.vocab_unpadded;
        this.hidden = (vout || vin) * c.hidden; // boundary vectors never decrease along the model: output >= input
    }
    get s() { if (!this.stage)
        throw new Error("Stage not loaded"); return this.stage; }
    get positions() { return [this.stage?.position ?? 0]; }
    get x() { return this.s.x; }
    get token() { return this.s.token; }
    get logits() { return this.s.logitsBuffer; }
    /** Loads from per-tensor files: fetch(file) returns the tensor's bytes; expert tensors are read in chunks. */
    async load(fetch) {
        let cached = null;
        const read = async (offset, bytes, tensor) => {
            const file = this.manifest.tensors[tensor]?.file;
            if (!file)
                throw new Error(`No file for ${tensor}`);
            if (cached?.file !== file)
                cached = { file, data: new Uint8Array(await fetch(file)) };
            if (offset + bytes > cached.data.byteLength)
                throw new Error(`Short file ${file}`);
            return cached.data.subarray(offset, offset + bytes);
        };
        this.stage = await Inkling2Stage.load(this.gpu, this.manifest, read, { first: this.manifest.first, last: this.manifest.last, chunk: this.maxTokens, capacity: this.maxPositions, logitRows: 1 });
    }
    async reset() { await this.s.reset(); }
    async run(input) {
        if ((input.slot ?? 0) !== 0)
            throw new Error("Inkling 2 stages serve one conversation");
        await this.s.run({ ids: input.ids ?? undefined, hidden: input.hidden ?? undefined, logits: this.output ? "last" : "none" });
    }
    async next() { return this.s.readToken(); }
}
