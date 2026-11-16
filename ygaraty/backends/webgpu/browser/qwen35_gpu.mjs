import { validatePlan, executePlan, bindKernel, assertStateReady, resetPlanState } from "./inference_plan.mjs";
// Generated: edit qwen35_gpu.mts and run npm run build:webgpu.
import { Gpu as KernelGpu, F16, objectRecord, positiveInteger, isPackedKind, REPACK, wgslMemo } from "./webgpu_kernels.mjs";
export { F16, GEMM } from "./webgpu_kernels.mjs";
export function qwenGeometry(config) {
    const c = config ?? {};
    const layers = c.layers ?? 32, heads = c.attention_heads ?? 16, kvHeads = c.attention_kv_heads ?? 4;
    const linearK = c.linear_k_heads ?? 16, linearV = c.linear_v_heads ?? 32;
    for (const [name, value] of [["layers", layers], ["attention_heads", heads], ["attention_kv_heads", kvHeads],
        ["linear_k_heads", linearK], ["linear_v_heads", linearV]]) {
        if (!Number.isSafeInteger(value) || value < 1 || value > 256)
            throw new Error(`Unsupported Qwen ${name}`);
    }
    if (layers % 4)
        throw new Error("Unsupported Qwen layer count");
    if (heads % kvHeads)
        throw new Error("Unsupported Qwen attention_heads");
    if (linearV % linearK)
        throw new Error("Unsupported Qwen linear_v_heads");
    const value = linearV * 128;
    return { layers, heads, kvHeads, linearK, linearV, conv: 2 * linearK * 128 + value, value,
        attention: heads * 256, kv: kvHeads * 256 };
}
export const DEFAULT_GEOMETRY = qwenGeometry();
// Floats per token row of the DeltaNet gates (a then b), padded to 64 so a
// one-row binding per sequence meets WebGPU's 256-byte offset alignment
// (distribution.gate_stride). 9B: 64, unchanged; 0.8B: 32 -> 64; 27B: 96 -> 128.
export const gateStride = (linearV) => Math.ceil(2 * linearV / 64) * 64;
export function parseManifest(value) {
    const m = objectRecord(value, "Qwen manifest");
    if (m.config !== undefined) {
        const c = objectRecord(m.config, "Qwen config");
        for (const key of ["hidden", "intermediate", "layers"])
            positiveInteger(c[key], key);
        for (const key of ["attention_heads", "attention_kv_heads", "linear_k_heads", "linear_v_heads"])
            if (c[key] !== undefined)
                positiveInteger(c[key], key);
        if (c.tied_embeddings !== undefined && typeof c.tied_embeddings !== "boolean")
            throw new Error("Invalid tied_embeddings");
    }
    const layerCount = qwenGeometry(m.config).layers;
    const tensor = (value, chunk = false) => {
        const t = objectRecord(value, "tensor");
        if (typeof t.file !== "string" || !t.file || !isPackedKind(t.kind) ||
            !Array.isArray(t.shape) || t.shape.length < 1 || t.shape.length > 2)
            throw new Error("Invalid tensor descriptor");
        for (const width of t.shape)
            positiveInteger(width, "tensor shape");
        positiveInteger(t.row_bytes, "row_bytes");
        if (chunk && (typeof t.first !== "number" || !Number.isSafeInteger(t.first) || t.first < 0))
            throw new Error("Invalid chunk offset");
    };
    if (!Array.isArray(m.layers))
        throw new Error("Missing layer list");
    let previous;
    for (const value of m.layers) {
        const layer = objectRecord(value, "layer");
        if (typeof layer.index !== "number" || !Number.isInteger(layer.index) ||
            layer.index < 0 || layer.index >= layerCount ||
            (previous !== undefined && layer.index !== previous + 1) ||
            layer.type !== (layer.index % 4 === 3 ? "full" : "linear"))
            throw new Error("Invalid layer order/type");
        previous = layer.index;
        for (const value of Object.values(objectRecord(layer.tensors, "layer tensors")))
            tensor(value);
    }
    for (const key of ["embed", "head"]) {
        if (m[key] === undefined || m[key] === null)
            continue;
        const projection = objectRecord(m[key], key);
        if (!Array.isArray(projection.chunks) || !projection.chunks.length)
            throw new Error("Missing projection chunks");
        let offset = 0;
        for (const value of projection.chunks) {
            tensor(value, true);
            const chunk = objectRecord(value, "chunk");
            const shape = chunk.shape;
            if (shape.length !== 2 || chunk.first !== offset)
                throw new Error("Invalid row coverage");
            offset += shape[0];
        }
        if (key === "head") {
            tensor(projection.norm);
            positiveInteger(projection.vocab, "vocab");
            if (offset !== projection.vocab)
                throw new Error("Output row count differs from vocab");
            if (projection.shared_embedding !== undefined && typeof projection.shared_embedding !== "boolean")
                throw new Error("Invalid shared_embedding");
        }
    }
    // All fields consumed by the adapter have been checked above. Shape/model
    // semantics also pass the Python loader's architecture-specific preflight.
    const manifest = value;
    stageDimensions(manifest);
    return manifest;
}
export const HIDDEN = 4096, FFN = 12288;
export const K_HEADS = 16, V_HEADS = 32, HEAD_K = 128, HEAD_V = 128;
export const CONV_DIM = 2 * K_HEADS * HEAD_K + V_HEADS * HEAD_V; // 8192
export const Q_HEADS = 16, KV_HEADS = 4, HEAD = 256, ROPE_DIMS = 64;
export const ROPE_THETA = 10000000, EPS = 1e-6;
const SPLIT = 512; // keys per attention workgroup (split-K, combined after)
// A submission holds at most this many token-layers: decode steps submit a
// whole stage at once, prefill chunks one layer at a time.
const CONV = /* wgsl */ `
struct P { tokens: u32, channels: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
fn input(t: i32, c: u32) -> f32 {
  if (t >= 0) { return x[u32(t) * p.channels + c]; }
  return state[u32(3 + t) * p.channels + c];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x; let t = gid.y;
  if (c >= p.channels || t >= p.tokens) { return; }
  var s = 0.0;
  for (var k = 0u; k < 4u; k++) { s += wt[c * 4u + k] * input(i32(t) + i32(k) - 3, c); }
  y[t * p.channels + c] = s / (1.0 + exp(-s));
}`;
const CONV_STATE = /* wgsl */ `
struct P { tokens: u32, channels: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.channels) { return; }
  var v: array<f32, 3>;
  for (var i = 0; i < 3; i++) {
    let t = i32(p.tokens) - 3 + i;  // the new state's i-th row is input t
    if (t >= 0) { v[i] = x[u32(t) * p.channels + c]; }
    else { v[i] = state[u32(3 + t) * p.channels + c]; }
  }
  for (var i = 0u; i < 3u; i++) { state[i * p.channels + c] = v[i]; }
}`;
// Gated delta rule, one workgroup per value head, one thread per value dim j;
// thread j keeps column j of the head's 128x128 state.
const deltaShaderSource = (linearK, linearV) => /* wgsl */ `
struct P { tokens: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> qkv: array<f32>;     // [T, ${2 * linearK * 128 + linearV * 128}] after conv
@group(0) @binding(1) var<storage, read> ab: array<f32>;      // [T, ${gateStride(linearV)}]: a (${linearV}) then b (${linearV})
@group(0) @binding(2) var<storage, read> head: array<f32>;    // [${2 * linearV}]: ssm_a (${linearV}) then dt_bias (${linearV})
@group(0) @binding(3) var<storage, read_write> S: array<f32>; // [${linearV}][128 k][128 j]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [T, ${linearV * 128}]
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<f32, 256>;
fn softplus(x: f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let h = wg.x; let j = li.x; let kh = h % ${linearK}u;
  var s: array<f32, 128>;
  for (var k = 0u; k < 128u; k++) { s[k] = S[(h * 128u + k) * 128u + j]; }
  for (var t = 0u; t < p.tokens; t++) {
    let row = t * ${2 * linearK * 128 + linearV * 128}u;
    let q = qkv[row + kh * 128u + j];
    let kk = qkv[row + ${linearK * HEAD_K}u + kh * 128u + j];
    let v = qkv[row + ${2 * linearK * HEAD_K}u + h * 128u + j];
    red[j] = q * q; red[128u + j] = kk * kk;
    workgroupBarrier();
    for (var m = 64u; m > 0u; m >>= 1u) {
      if (j < m) { red[j] += red[j + m]; red[128u + j] += red[128u + j + m]; }
      workgroupBarrier();
    }
    qs[j] = q * inverseSqrt(red[0] + 1e-6) / sqrt(128.0);
    ks[j] = kk * inverseSqrt(red[128] + 1e-6);
    workgroupBarrier();
    let a = ab[t * ${gateStride(linearV)}u + h]; let b = ab[t * ${gateStride(linearV)}u + ${linearV}u + h];
    let beta = 1.0 / (1.0 + exp(-b));
    let decay = exp(head[h] * softplus(a + head[${linearV}u + h]));
    var kv = 0.0;
    for (var k = 0u; k < 128u; k++) { s[k] = s[k] * decay; kv += s[k] * ks[k]; }
    let delta = (v - kv) * beta;
    var o = 0.0;
    for (var k = 0u; k < 128u; k++) { s[k] = s[k] + ks[k] * delta; o += s[k] * qs[k]; }
    out[t * ${linearV * HEAD_V}u + h * 128u + j] = o;
    workgroupBarrier();
  }
  for (var k = 0u; k < 128u; k++) { S[(h * 128u + k) * 128u + j] = s[k]; }
}`;
const deltaShader = wgslMemo(deltaShaderSource);
// Partial RoPE (first 64 of 256 dims, halves rotated as rotate_half) in place.
const ROPE = /* wgsl */ `
struct P { tokens: u32, heads: u32, stride: u32, head_stride: u32, position: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x / p.heads; let h = wg.x % p.heads; let i = li.x;
  let angle = f32(p.position + t) * inv_freq[i];
  let c = cos(angle); let s = sin(angle);
  let b = t * p.stride + h * p.head_stride;
  let x1 = x[b + i]; let x2 = x[b + i + 32u];
  x[b + i] = x1 * c - x2 * s;
  x[b + i + 32u] = x2 * c + x1 * s;
}`;
// float32 <-> float16 in integer arithmetic: round to nearest even (as NumPy
// and JavaScript's Float16Array), subnormals kept, overflow to infinity.
// WGSL leaves pack2x16float's rounding to the implementation (RADV on the
// 680M rounds toward zero) and unpack2x16float may flush subnormals, so the
// cache does not use them: every contributor stores the same bits.
// Write K or V rows [T, 4 heads x 256] into the cache, laid out
// [4 heads][capacity][256], at row `position`; float16 packs two per word.
const storeShaderSource = (kv, kvHeads = 4) => /* wgsl */ `
struct P { tokens: u32, position: u32, capacity: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<${kv === "f16" ? "u32" : "f32"}>;
@group(0) @binding(2) var<uniform> p: P;
${F16}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.y;
  if (t >= p.tokens) { return; }
  ${kv === "f16" ? `
  let i = gid.x; let g = i / 128u; let c = i % 128u;  // 512 words per token
  let src = t * ${kvHeads * HEAD}u + g * 256u + 2u * c;
  cache[(g * p.capacity + p.position + t) * 128u + c] = f16_bits(x[src]) | (f16_bits(x[src + 1u]) << 16u);` : `
  let i = gid.x; let g = i / 256u; let c = i % 256u;
  cache[(g * p.capacity + p.position + t) * 256u + c] = x[t * ${kvHeads * HEAD}u + i];`}
}`;
const storeShader = wgslMemo(storeShaderSource); // WS-33: one source string object per arguments
// Causal attention, split over keys: a workgroup takes one token, one KV
// head (its 4 query heads share each key and value read) and up to SPLIT
// keys, in blocks of 64 with an online softmax, and writes its unnormalized
// output and (max, sum) for the combine pass. With another number of query
// heads per KV head (Qwen3.8-27B: 6), a workgroup takes G = 4, 2 or 1 of
// them (the largest that divides it); unused lanes repeat its heads and are
// not written. With 4 the source is unchanged.
const queryGroup = (heads, kvHeads) => {
    const r = heads / kvHeads;
    return r % 4 === 0 ? 4 : r % 2 === 0 ? 2 : 1;
};
const attentionShaderSource = (kv, heads = 16, kvHeads = 4) => {
    const r = heads / kvHeads, G = queryGroup(heads, kvHeads), same = r === 4;
    return /* wgsl */ `
struct P { tokens: u32, position: u32, splits: u32, capacity: u32 };
@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;     // [T, ${heads} heads, 256]
@group(0) @binding(1) var<storage, read> K: array<${kv === "f16" ? "vec2<u32>" : "vec4<f32>"}>;
@group(0) @binding(2) var<storage, read> V: array<${kv === "f16" ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;  // [T][splits][${heads}][256]
@group(0) @binding(4) var<storage, read_write> ml: array<vec2<f32>>; // [T][splits][${heads}]
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<vec4<f32>, 256>;  // the 4 query heads
var<workgroup> pr: array<f32, 1024>;       // partial scores [4 heads][64 keys][4]
var<workgroup> sc: array<f32, 256>;        // scores [4 heads][64 keys]
var<workgroup> mx: array<f32, 256>;
${F16}
fn k4(i: u32) -> vec4<f32> {
  ${kv === "f16" ? "let u = K[i]; return vec4<f32>(f16x2(u.x), f16x2(u.y));" : "return K[i];"}
}
fn v1(i: u32) -> f32 {
  ${kv === "f16" ? "return f16_value((V[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu);" : "return V[i];"}
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let s = wg.x % p.splits; let tg = wg.x / p.splits;
  let g = tg % ${heads / G}u; let t = tg / ${heads / G}u;
  let id = li.x;
  qs[id] = q[t * ${heads * HEAD / 4}u + ${same ? "g * 256u + id" : `(g * ${G}u + (id / 64u) % ${G}u) * 64u + id % 64u`}];
  workgroupBarrier();
  let begin = s * ${SPLIT}u;
  let end = min(begin + ${SPLIT}u, p.position + t + 1u);  // keys 0..position+t
  let kk = id / 4u; let pp = id % 4u;  // scores: key kk, dims pp, pp+4, ...
  let hh = id / 64u; let k = id % 64u;  // softmax: head hh, key k
  let base = ${same ? "g" : `(g * ${G}u / ${r}u)`} * p.capacity;
  var m = vec4<f32>(-3.0e38); var l = vec4<f32>(0.0); var acc = vec4<f32>(0.0);
  for (var k0 = begin; k0 < end; k0 += 64u) {
    var s4 = vec4<f32>(0.0);
    if (k0 + kk < end) {
      let kb = (base + k0 + kk) * 64u;
      for (var i = 0u; i < 16u; i++) {
        let c = pp + 4u * i;
        let kv = k4(kb + c);
        s4 += vec4<f32>(dot(qs[c], kv), dot(qs[64u + c], kv), dot(qs[128u + c], kv), dot(qs[192u + c], kv));
      }
    }
    for (var h = 0u; h < 4u; h++) { pr[(h * 64u + kk) * 4u + pp] = s4[h]; }
    workgroupBarrier();
    var score = -3.0e38;
    if (k0 + k < end) { score = (pr[id * 4u] + pr[id * 4u + 1u] + pr[id * 4u + 2u] + pr[id * 4u + 3u]) * ${1 / Math.sqrt(HEAD)}; }
    sc[id] = score; mx[id] = score;
    workgroupBarrier();
    for (var r = 32u; r > 0u; r >>= 1u) {
      if (k < r) { mx[id] = max(mx[id], mx[id + r]); }
      workgroupBarrier();
    }
    let mn = max(m, vec4<f32>(mx[0], mx[64], mx[128], mx[192]));
    let corr = exp(m - mn);
    acc *= corr; l *= corr;
    let n = min(64u, end - k0);
    let vb = (base + k0) * 256u + id;
    for (var e = 0u; e < n; e++) {
      let pe = exp(vec4<f32>(sc[e], sc[64u + e], sc[128u + e], sc[192u + e]) - mn);
      l += pe;
      acc += pe * v1(vb + e * 256u);
    }
    m = mn;
    workgroupBarrier();
  }
  let o = ((t * p.splits + s) * ${heads}u + g * ${G}u) * 256u + id;
  ${["part[o] = acc.x;", "part[o + 256u] = acc.y;", "part[o + 512u] = acc.z;", "part[o + 768u] = acc.w;"].slice(0, G).join(" ")}
  if (id < ${G}u) { ml[(t * p.splits + s) * ${heads}u + g * ${G}u + id] = vec2<f32>(m[id], l[id]); }
}`;
};
const attentionShader = wgslMemo(attentionShaderSource); // WS-33: one source string object per arguments
// Combine the splits of one (token, query head) and multiply by sigmoid(gate).
const combineShaderSource = (heads = 16) => /* wgsl */ `
struct P { tokens: u32, splits: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> ml: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> qg: array<f32>;     // [T, ${heads}*512]: gate at +256
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [T, ${heads}*256]
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x / ${heads}u; let h = wg.x % ${heads}u; let d = li.x;
  var mm = -3.0e38;
  for (var s = 0u; s < p.splits; s++) { mm = max(mm, ml[(t * p.splits + s) * ${heads}u + h].x); }
  var l = 0.0; var o = 0.0;
  for (var s = 0u; s < p.splits; s++) {
    let i = (t * p.splits + s) * ${heads}u + h;
    let f = exp(ml[i].x - mm);
    l += ml[i].y * f;
    o += part[i * 256u + d] * f;
  }
  let gate = qg[t * ${heads * 2 * HEAD}u + h * 512u + 256u + d];
  out[t * ${heads * HEAD}u + h * 256u + d] = o / l / (1.0 + exp(-gate));
}`;
const combineShader = wgslMemo(combineShaderSource);
// b[t, i] = silu(a[t, i]) * b[t, i], in place in b
export class Gpu extends KernelGpu {
    // `g`: the stage's geometry (the 9B's by default).
    conv(x, weight, state, y, tokens, g = DEFAULT_GEOMETRY) {
        const u = this.uniform([["u", tokens], ["u", g.conv], ["u", 0], ["u", 0]]);
        this.dispatch("conv", CONV, [x, weight, state, y, u], [g.conv / 256, tokens, 1]);
        const v = this.uniform([["u", tokens], ["u", g.conv], ["u", 0], ["u", 0]]);
        this.dispatch("conv_state", CONV_STATE, [x, state, v], [g.conv / 256, 1, 1]);
    }
    delta(qkv, ab, head, S, out, tokens, g = DEFAULT_GEOMETRY) {
        const u = this.uniform([["u", tokens], ["u", 0], ["u", 0], ["u", 0]]);
        this.dispatch("delta", deltaShader(g.linearK, g.linearV), [qkv, ab, head, S, out, u], [g.linearV, 1, 1]);
    }
    rope(x, invFreq, tokens, heads, stride, headStride, position) {
        const u = this.uniform([["u", tokens], ["u", heads], ["u", stride], ["u", headStride], ["u", position], ["u", 0], ["u", 0], ["u", 0]]);
        this.dispatch("rope", ROPE, [x, invFreq, u], [tokens * heads, 1, 1]);
    }
    store(x, cache, tokens, position, capacity, kv, g = DEFAULT_GEOMETRY) {
        const u = this.uniform([["u", tokens], ["u", position], ["u", capacity], ["u", 0]]);
        const words = kv === "f16" ? g.kv / 2 : g.kv;
        this.dispatch(`store_${kv}`, storeShader(kv, g.kvHeads), [x, cache, u], [words / 256, tokens, 1]);
    }
    // part, ml: scratch of at least tokens x splits x heads x 256 and x heads x 2 floats.
    attention(q, K, V, qg, out, tokens, position, capacity, kv, part, ml, g = DEFAULT_GEOMETRY) {
        const splits = Math.ceil((position + tokens) / SPLIT);
        const u = this.uniform([["u", tokens], ["u", position], ["u", splits], ["u", capacity]]);
        this.dispatch(`attention_${kv}`, attentionShader(kv, g.heads, g.kvHeads), [q, K, V, part, ml, u], [tokens * (g.heads / queryGroup(g.heads, g.kvHeads)) * splits, 1, 1]);
        const v = this.uniform([["u", tokens], ["u", splits], ["u", 0], ["u", 0]]);
        this.dispatch("combine", combineShader(g.heads), [part, ml, qg, out, v], [tokens * g.heads, 1, 1]);
    }
}
// Widths and head counts from the manifest's config (9B defaults).
export function stageDimensions(manifest) {
    const c = manifest.config ?? { hidden: HIDDEN, intermediate: FFN, layers: 32 };
    for (const name of ["hidden", "intermediate"]) {
        if (!Number.isSafeInteger(c[name]) || c[name] <= 0 || c[name] % 256)
            throw new Error(`Unsupported Qwen ${name}`);
    }
    const geometry = qwenGeometry(c);
    return {
        hidden: c.hidden, intermediate: c.intermediate, geometry,
        scratch: Math.max(c.hidden, geometry.attention, geometry.value),
    };
}
// One pipeline stage: optionally the embedding, then layers first..last-1.
export class Stage {
    gpu;
    maxTokens;
    maxPositions;
    hidden;
    intermediate;
    scratch;
    geometry;
    position = 0;
    slots;
    maxBatch;
    positions;
    slot = 0;
    loaded = false;
    plan;
    layers = [];
    embed;
    output;
    vocab = 0;
    submitMs = [];
    submitLayers = [];
    x;
    h;
    big;
    big2;
    mid;
    z;
    ab;
    k;
    kn;
    v;
    f1;
    f2;
    part;
    ml;
    ids;
    invFreq;
    outputNorm;
    logits;
    token;
    kv;
    manifest;
    // DO-25 speculative decoding: each slot's DeltaNet states and position at the start of the last checking
    // pass, restored when some of its drafts were rejected (allocated on first use).
    snaps = new Map();
    constructor(gpu, manifest, maxTokens, maxPositions, { kv = "f16", slots = 1, rows = 1 } = {}) {
        this.gpu = gpu;
        this.maxTokens = maxTokens;
        this.maxPositions = maxPositions;
        if (!["f16", "f32"].includes(kv))
            throw new Error(`kv ${kv}`);
        this.manifest = parseManifest(manifest);
        const dimensions = stageDimensions(this.manifest);
        this.hidden = dimensions.hidden;
        this.intermediate = dimensions.intermediate;
        this.plan = this.manifest.plan === undefined ? null : validatePlan(this.manifest.plan, this.manifest, "qwen35", this.hidden);
        this.scratch = dimensions.scratch;
        this.geometry = dimensions.geometry;
        this.gpu = gpu;
        if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 ||
            !Number.isSafeInteger(maxPositions) || maxPositions < maxTokens)
            throw new Error("Invalid token/context capacity");
        this.maxTokens = maxTokens;
        this.maxPositions = maxPositions;
        this.kv = kv;
        if (!Number.isSafeInteger(slots) || slots < 1)
            throw Error("Invalid slot count");
        this.slots = slots;
        // `rows`: positions whose logits one call returns (a speculative checking pass: 1 + drafts).
        if (!Number.isSafeInteger(rows) || rows < 1 || rows > maxTokens)
            throw Error("Invalid head row count");
        this.maxBatch = Math.max(Math.min(slots, maxTokens), rows);
        this.positions = new Array(slots).fill(0);
        this.position = 0;
    }
    use(slot) {
        if (!Number.isInteger(slot) || !(slot >= 0 && slot < this.slots))
            throw new Error(`slot ${slot} of ${this.slots}`);
        for (const L of this.layers) {
            if (L.type === "linear") {
                L.S = L.Ss[slot];
                L.convState = L.convStates[slot];
            }
            else {
                L.K = L.Ks[slot];
                L.V = L.Vs[slot];
            }
        }
        this.slot = slot;
        this.position = this.positions[slot];
    }
    async load(fetchTensor, log = () => { }) {
        const gpu = this.gpu, m = this.manifest;
        const tensor = async (t) => {
            const data = await fetchTensor(t.file);
            return { kind: t.kind, rows: t.shape[0], cols: t.shape[t.shape.length - 1], rowBytes: t.row_bytes, buffer: gpu.upload(data) };
        };
        // WS-40 (opt-in): layer Q4_K matrices repacked at load into Q4_KR (scales unpacked; same products, 4/3 the bytes).
        // DO-80: also Q5_K and Q6_K (layer matrices and the head) when listed in gpu.repack.
        const matrix = async (t) => {
            const on = (gpu.repackQ4K && t.kind === "Q4_K") || gpu.repack.includes(t.kind);
            const r = on ? REPACK[t.kind] : undefined;
            if (!r)
                return tensor(t);
            const rows = t.shape[0], cols = t.shape[t.shape.length - 1];
            return { kind: r.kind, rows, cols, rowBytes: (cols / 256) * r.block, buffer: gpu.upload(r.fn(await fetchTensor(t.file), rows, cols)) };
        };
        const floats = async (t) => gpu.upload(await fetchTensor(t.file));
        if (m.embed) {
            this.embed = [];
            for (const c of m.embed.chunks) {
                const w = await tensor(c);
                this.embed.push({ ...w, first: c.first });
            }
            log(`embedding: ${this.embed.length} chunks`);
        }
        this.layers = [];
        for (const L of m.layers) {
            const matrices = {};
            const vectors = {};
            for (const [name, t] of Object.entries(L.tensors)) {
                if (t.kind === "F32" && t.shape.length === 1)
                    vectors[name] = await floats(t);
                else
                    matrices[name] = await matrix(t);
            }
            const common = { index: L.index, matrices, vectors };
            if (L.type === "linear") {
                const a = new Float32Array(await fetchTensor(L.tensors["ssm_a"].file));
                const dt = new Float32Array(await fetchTensor(L.tensors["ssm_dt.bias"].file));
                const g = this.geometry;
                const head = new Float32Array(2 * g.linearV);
                head.set(a, 0);
                head.set(dt, g.linearV);
                const convStates = Array.from({ length: this.slots }, () => gpu.buffer(3 * g.conv * 4));
                const Ss = Array.from({ length: this.slots }, () => gpu.buffer(g.linearV * HEAD_K * HEAD_V * 4));
                this.layers.push({ ...common, type: "linear",
                    head: gpu.upload(head),
                    conv: gpu.upload(await fetchTensor(L.tensors["ssm_conv1d.weight"].file)),
                    convState: convStates[0], S: Ss[0], convStates, Ss,
                });
            }
            else {
                const bytes = this.maxPositions * this.geometry.kv * (this.kv === "f16" ? 2 : 4);
                const Ks = Array.from({ length: this.slots }, () => gpu.buffer(bytes));
                const Vs = Array.from({ length: this.slots }, () => gpu.buffer(bytes));
                this.layers.push({ ...common, type: "full", K: Ks[0], V: Vs[0], Ks, Vs });
            }
            log(`layer ${L.index} (${L.type}) loaded; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
        }
        if (m.head) {
            this.outputNorm = await floats(m.head.norm);
            this.output = m.head.shared_embedding && this.embed ? this.embed : [];
            for (const c of (this.output.length ? [] : m.head.chunks)) {
                const w = await matrix(c);
                this.output.push({ ...w, first: c.first });
            }
            this.vocab = m.head.vocab;
            this.logits = gpu.buffer(this.maxBatch * this.vocab * 4);
            this.token = gpu.buffer(this.maxBatch * 256);
            log(`output: ${this.output.length} chunks; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
        }
        const invFreq = new Float32Array(ROPE_DIMS / 2);
        for (let i = 0; i < ROPE_DIMS / 2; i++)
            invFreq[i] = 1 / Math.pow(ROPE_THETA, (2 * i) / ROPE_DIMS);
        this.invFreq = gpu.upload(invFreq);
        const T = this.maxTokens, g = this.geometry;
        this.ids = gpu.buffer(T * 4);
        this.x = gpu.buffer(T * this.hidden * 4);
        this.h = gpu.buffer(T * this.scratch * 4);
        this.big = gpu.buffer(T * Math.max(g.conv, 2 * g.attention) * 4);
        this.big2 = gpu.buffer(T * g.conv * 4);
        this.mid = gpu.buffer(T * this.scratch * 4);
        this.z = gpu.buffer(T * g.value * 4);
        this.ab = gpu.buffer(T * gateStride(g.linearV) * 4);
        this.k = gpu.buffer(T * g.kv * 4);
        this.kn = gpu.buffer(T * g.kv * 4);
        this.v = gpu.buffer(T * g.kv * 4);
        this.f1 = gpu.buffer(T * this.intermediate * 4);
        this.f2 = gpu.buffer(T * this.intermediate * 4);
        if (this.layers.some((L) => L.type === "full")) {
            const splits = Math.ceil(this.maxPositions / SPLIT);
            this.part = gpu.buffer(T * splits * g.attention * 4);
            this.ml = gpu.buffer(T * splits * g.heads * 2 * 4);
        }
        this.loaded = true;
    }
    // Run `tokens` positions: from token ids (embedding stage) or a hidden
    // input [tokens, hidden]. Leaves the result in this.x. The last stage's
    // head: the last position's greedy token (`heads` "last"), every
    // position's ("all", read with nextBatch), or none. `exact`: matrix-vector
    // kernels at any token count, so each position's sums are a single
    // decode step's (DO-25: a checking pass gives plain decoding's tokens);
    // `gemm`: the prefill GEMM from 5 tokens (prefill's rounding).
    // WS-43: `gpuIds`: one decode token whose id is already in this.ids (copied there on the GPU from this.token);
    // the resident loop undoes a step run past the stop rule with snapshot/restore (the DeltaNet states).
    async run({ ids = null, hidden = null, tokens, slot = 0, heads = "last", exact = false, gemm = false, gpuIds = false }) {
        if (!this.loaded)
            throw new Error("Stage is not loaded");
        const gpu = this.gpu;
        assertStateReady(this, [slot]);
        this.use(slot);
        if (!Number.isSafeInteger(tokens) || tokens < 1)
            throw new Error("Invalid token count");
        if (gpuIds) {
            if (!this.embed || tokens !== 1 || ids != null || hidden != null)
                throw new Error("Invalid resident input");
        }
        else if (this.embed) {
            if (hidden !== null || !ids || ids.length !== tokens ||
                !Array.from(ids).every(id => Number.isSafeInteger(id) && id >= 0 &&
                    id < this.embed.reduce((n, chunk) => n + chunk.rows, 0)))
                throw new Error("Invalid embedding input");
        }
        else if (ids !== null || !(hidden instanceof Float32Array) ||
            hidden.length !== tokens * this.hidden || !hidden.every(Number.isFinite)) {
            throw new Error("Invalid hidden input");
        }
        if (tokens > this.maxTokens)
            throw new Error(`chunk ${tokens} > ${this.maxTokens}`);
        if (heads === "all" && tokens > this.maxBatch)
            throw new Error(`head rows ${tokens} > ${this.maxBatch}`);
        if (this.position + tokens > this.maxPositions)
            throw new Error(`position ${this.position + tokens} > ${this.maxPositions}`);
        // Queue writes run before the command buffer submitted after them.
        const upload = () => {
            if (ids)
                gpu.device.queue.writeBuffer(this.ids, 0, new Uint32Array(ids));
            else if (!gpuIds)
                gpu.device.queue.writeBuffer(this.x, 0, hidden);
        };
        // Short submissions: a long one trips the driver's GPU timeout (amdgpu
        // reset its ring at 8 layers x 128 tokens; Windows' TDR is about 2 s).
        // At most SUBMIT_TOKEN_LAYERS token-layers each: one layer per prefill
        // chunk, the whole stage per decode step. `submitMs`: each submission's
        // encode-to-done time, with its layer count in `submitLayers`.
        this.submitMs = [];
        this.submitLayers = [];
        const saved = gpu.prefill;
        if (exact)
            gpu.prefill = "matvec";
        else if (gemm)
            gpu.prefill = "gemm";
        try {
            await executePlan(this.plan, tokens, gpu, this.kernels(), this.submitMs, this.submitLayers, this.planContext(tokens, false, [{ slot, row: 0, count: tokens, position: this.position }], heads), upload);
        }
        finally {
            gpu.prefill = saved;
        }
        this.position += tokens;
        this.positions[slot] = this.position;
    }
    async runBatch({ slots, ids = null, hidden = null }) {
        assertStateReady(this, slots);
        const gpu = this.gpu, T = slots.length;
        if (!this.loaded || !T)
            throw Error("Stage unavailable or empty batch");
        if (this.embed) {
            const vocab = this.embed.reduce((n, c) => n + c.rows, 0);
            if (hidden !== null || !ids || ids.length !== T || !Array.from(ids).every(id => Number.isSafeInteger(id) && id >= 0 && id < vocab))
                throw Error("Invalid batch IDs");
        }
        else if (ids !== null || !(hidden instanceof Float32Array) || hidden.length !== T * this.hidden || !hidden.every(Number.isFinite))
            throw Error("Invalid batch hidden input");
        if (T > this.maxBatch)
            throw new Error(`batch ${T} > ${this.maxBatch}`);
        if (new Set(slots).size !== T)
            throw new Error("a batch holds one token per slot");
        for (const slot of slots) {
            if (!Number.isInteger(slot) || !(slot >= 0 && slot < this.slots))
                throw new Error(`slot ${slot} of ${this.slots}`);
            if (this.positions[slot] + 1 > this.maxPositions)
                throw new Error(`slot ${slot}: position ${this.positions[slot] + 1} > ${this.maxPositions}`);
        }
        const upload = () => {
            if (ids)
                gpu.device.queue.writeBuffer(this.ids, 0, new Uint32Array(ids));
            else
                gpu.device.queue.writeBuffer(this.x, 0, hidden);
        };
        // The matrix-vector kernel for any batch size (groups of 4 tokens): its
        // per-token sums are a single token's, so batching keeps every token.
        const saved = gpu.prefill, savedGemm = gpu.gemm;
        // WS-45 and experiment 127: a many-token kernel on request, or the GEMM for big batches (summation order differs).
        const big = gpu.batchGemm && T >= gpu.batchGemm.from;
        gpu.prefill = gpu.batchKernel ?? (big ? "gemm" : "matvec");
        if (big)
            gpu.gemm = gpu.batchGemm.tile;
        this.submitMs = [];
        this.submitLayers = [];
        try {
            await executePlan(this.plan, T, gpu, this.kernels(), this.submitMs, this.submitLayers, this.planContext(T, true, slots.map((slot, row) => ({ slot, row, count: 1, position: this.positions[slot] }))), upload);
        }
        finally {
            gpu.prefill = saved;
            gpu.gemm = savedGemm;
        }
        for (const slot of slots)
            this.positions[slot] += 1;
        this.position = this.positions[this.slot ?? 0];
    }
    // Adapters bind resources and primitives; mathematical composition lives on Tape.
    kernels() {
        const gpu = this.gpu, g = this.geometry;
        return {
            embedding: bindKernel(gpu.gather.bind(gpu)), rms_norm: bindKernel(gpu.norm.bind(gpu)),
            matmul: bindKernel(gpu.matmul.bind(gpu)), silu_mul: bindKernel(gpu.siluMul.bind(gpu)),
            argmax: bindKernel(gpu.argmax.bind(gpu)),
            // The plan's operation signatures carry no head counts: append the stage's geometry.
            causal_conv1d: bindKernel((...a) => gpu.conv(a[0], a[1], a[2], a[3], a[4], g)),
            gated_delta: bindKernel((...a) => gpu.delta(a[0], a[1], a[2], a[3], a[4], a[5], g)),
            rotary_partial: bindKernel(gpu.rope.bind(gpu)),
            cache_store: bindKernel((...a) => gpu.store(a[0], a[1], a[2], a[3], a[4], a[5], g)),
            gated_attention: bindKernel((...a) => gpu.attention(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], g)),
        };
    }
    planContext(tokens, batch, sequences, heads = "last") {
        return {
            stateOwner: this,
            buffers: { x: this.x, h: this.h, f1: this.f1, f2: this.f2, part: this.part, ml: this.ml, ids: this.ids, invFreq: this.invFreq, outputNorm: this.outputNorm, logits: this.logits, token: this.token, big: this.big, big2: this.big2, mid: this.mid, z: this.z, ab: this.ab, k: this.k, kn: this.kn, v: this.v },
            layer: index => this.layers[index - this.manifest.layers[0].index],
            resources: { embed: this.embed, output: this.output },
            variables: { tokens, batch, capacity: this.maxPositions, kv: this.kv, vocab: this.vocab,
                head_enabled: heads !== "none", head_all: batch || heads === "all",
                head_tokens: (batch || heads === "all") ? tokens : 1, head_offset: (tokens - 1) * this.hidden },
            sequences,
            geometry: this.geometry,
        };
    }
    // DO-25: keep a slot's DeltaNet states and position before a checking pass. Attention's KV cache needs no
    // copy: rolling the position back makes the rejected positions unread, and the next pass overwrites them.
    snapshot(slot) {
        assertStateReady(this, [slot]);
        let snap = this.snaps.get(slot);
        const linear = this.layers.filter((L) => L.type === "linear");
        if (!snap) {
            snap = { position: 0, S: linear.map((L) => this.gpu.buffer(L.Ss[slot].size)), conv: linear.map((L) => this.gpu.buffer(L.convStates[slot].size)) };
            this.snaps.set(slot, snap);
        }
        const encoder = this.gpu.device.createCommandEncoder();
        linear.forEach((L, i) => {
            encoder.copyBufferToBuffer(L.Ss[slot], 0, snap.S[i], 0, L.Ss[slot].size);
            encoder.copyBufferToBuffer(L.convStates[slot], 0, snap.conv[i], 0, L.convStates[slot].size);
        });
        this.gpu.device.queue.submit([encoder.finish()]); // queue order: before the pass that follows
        snap.position = this.positions[slot];
    }
    // DO-25: back to the last snapshot of a slot (its checking pass had a rejected draft).
    restore(slot) {
        assertStateReady(this, [slot]);
        const snap = this.snaps.get(slot);
        if (!snap)
            throw new Error(`slot ${slot} has no snapshot`);
        const linear = this.layers.filter((L) => L.type === "linear");
        const encoder = this.gpu.device.createCommandEncoder();
        linear.forEach((L, i) => {
            encoder.copyBufferToBuffer(snap.S[i], 0, L.Ss[slot], 0, L.Ss[slot].size);
            encoder.copyBufferToBuffer(snap.conv[i], 0, L.convStates[slot], 0, L.convStates[slot].size);
        });
        this.gpu.device.queue.submit([encoder.finish()]);
        this.positions[slot] = snap.position;
        if (this.slot === slot)
            this.position = snap.position;
    }
    // Start a new sequence: position 0 and zero DeltaNet states (the KV cache
    // needs no clearing: attention reads positions below `position` only).
    async reset(slot = null) {
        if (slot !== null && (!Number.isInteger(slot) || slot < 0 || slot >= this.slots))
            throw Error("Invalid reset slot");
        const resetSlots = slot === null ? [...this.positions.keys()] : [slot];
        await resetPlanState(this, resetSlots, async () => {
            const encoder = this.gpu.device.createCommandEncoder();
            const slots = slot === null ? [...this.positions.keys()] : [slot];
            for (const L of this.layers) {
                if (L.type !== "linear")
                    continue;
                for (const k of slots) {
                    encoder.clearBuffer(L.Ss[k]);
                    encoder.clearBuffer(L.convStates[k]);
                }
            }
            this.gpu.device.queue.submit([encoder.finish()]);
            await this.gpu.device.queue.onSubmittedWorkDone();
            if (this.gpu.lost || this.gpu.errors?.length)
                throw Error('GPU reset failed; recreate the stage');
            for (const k of slots)
                this.positions[k] = 0;
            this.position = this.positions[this.slot ?? 0];
        });
    }
    // Logits of the call's last position and its greedy token.
    head(T) {
        const gpu = this.gpu;
        gpu.norm(this.x, this.outputNorm, this.h, 1, { size: this.hidden, inOffset: (T - 1) * this.hidden });
        for (const c of this.output ?? []) {
            gpu.matmul(c, this.h, this.logits, 1, { yStride: this.vocab, rowBase: c.first });
        }
        gpu.argmax(this.logits, this.token, this.vocab);
    }
    async next() {
        const v = await this.gpu.read(this.token, 4);
        return new Uint32Array(v.buffer)[0];
    }
    // Logits and greedy token of every batched token (rows 0..T-1 of this.x).
    headBatch(T) {
        const gpu = this.gpu;
        gpu.norm(this.x, this.outputNorm, this.h, T, { size: this.hidden });
        for (const c of this.output ?? []) {
            gpu.matmul(c, this.h, this.logits, T, { yStride: this.vocab, rowBase: c.first });
        }
        for (let t = 0; t < T; t++) {
            gpu.argmax({ buffer: this.logits, offset: t * this.vocab * 4 }, { buffer: this.token, offset: t * 256 }, this.vocab);
        }
    }
    async nextBatch(T) {
        const v = new Uint32Array((await this.gpu.read(this.token, T * 256)).buffer);
        return Array.from({ length: T }, (_, t) => v[t * 64]);
    }
    layerBatch(L, slots) {
        const gpu = this.gpu, T = slots.length, g = this.geometry;
        const row = (buffer, t, floats) => ({ buffer, offset: t * floats * 4 });
        gpu.norm(this.x, L.vectors["attn_norm.weight"], this.h, T, { size: this.hidden });
        if (L.type === "linear") {
            gpu.matmul(L.matrices["attn_qkv.weight"], this.h, this.big, T);
            gpu.matmul(L.matrices["attn_gate.weight"], this.h, this.z, T);
            gpu.matmul(L.matrices["ssm_alpha.weight"], this.h, this.ab, T, { yStride: gateStride(g.linearV), yOffset: 0 });
            gpu.matmul(L.matrices["ssm_beta.weight"], this.h, this.ab, T, { yStride: gateStride(g.linearV), yOffset: g.linearV });
            slots.forEach((slot, t) => {
                gpu.conv(row(this.big, t, g.conv), L.conv, L.convStates[slot], row(this.big2, t, g.conv), 1, g);
                gpu.delta(row(this.big2, t, g.conv), row(this.ab, t, gateStride(g.linearV)), L.head, L.Ss[slot], row(this.mid, t, g.value), 1, g);
            });
            gpu.norm(this.mid, L.vectors["ssm_norm.weight"], this.h, T, {
                size: HEAD_V, groups: g.linearV, inStride: g.value, outStride: g.value,
                gate: this.z, gateStride: g.value, gateGroup: HEAD_V,
            });
            gpu.matmul(L.matrices["ssm_out.weight"], this.h, this.x, T, { accumulate: true });
        }
        else {
            gpu.matmul(L.matrices["attn_q.weight"], this.h, this.big, T);
            gpu.matmul(L.matrices["attn_k.weight"], this.h, this.k, T);
            gpu.matmul(L.matrices["attn_v.weight"], this.h, this.v, T);
            gpu.norm(this.big, L.vectors["attn_q_norm.weight"], this.mid, T, {
                size: HEAD, groups: g.heads, inStride: 2 * g.attention, inGroup: 2 * HEAD, outStride: g.attention,
            });
            gpu.norm(this.k, L.vectors["attn_k_norm.weight"], this.kn, T, { size: HEAD, groups: g.kvHeads });
            slots.forEach((slot, t) => {
                const position = this.positions[slot];
                const q = row(this.mid, t, g.attention), k = row(this.kn, t, g.kv);
                gpu.rope(q, this.invFreq, 1, g.heads, g.attention, HEAD, position);
                gpu.rope(k, this.invFreq, 1, g.kvHeads, g.kv, HEAD, position);
                gpu.store(k, L.Ks[slot], 1, position, this.maxPositions, this.kv, g);
                gpu.store(row(this.v, t, g.kv), L.Vs[slot], 1, position, this.maxPositions, this.kv, g);
                gpu.attention(q, L.Ks[slot], L.Vs[slot], row(this.big, t, 2 * g.attention), row(this.h, t, g.attention), 1, position, this.maxPositions, this.kv, this.part, this.ml, g);
            });
            gpu.matmul(L.matrices["attn_output.weight"], this.h, this.x, T, { accumulate: true });
        }
        gpu.norm(this.x, L.vectors["post_attention_norm.weight"], this.h, T, { size: this.hidden });
        gpu.matmul(L.matrices["ffn_gate.weight"], this.h, this.f1, T);
        gpu.matmul(L.matrices["ffn_up.weight"], this.h, this.f2, T);
        gpu.siluMul(this.f1, this.f2, T * this.intermediate);
        gpu.matmul(L.matrices["ffn_down.weight"], this.f2, this.x, T, { accumulate: true });
    }
    layer(L, T) {
        const gpu = this.gpu, g = this.geometry;
        gpu.norm(this.x, L.vectors["attn_norm.weight"], this.h, T, { size: this.hidden });
        if (L.type === "linear") {
            gpu.matmul(L.matrices["attn_qkv.weight"], this.h, this.big, T);
            gpu.matmul(L.matrices["attn_gate.weight"], this.h, this.z, T);
            gpu.matmul(L.matrices["ssm_alpha.weight"], this.h, this.ab, T, { yStride: gateStride(g.linearV), yOffset: 0 });
            gpu.matmul(L.matrices["ssm_beta.weight"], this.h, this.ab, T, { yStride: gateStride(g.linearV), yOffset: g.linearV });
            gpu.conv(this.big, L.conv, L.convState, this.big2, T, g);
            gpu.delta(this.big2, this.ab, L.head, L.S, this.mid, T, g);
            gpu.norm(this.mid, L.vectors["ssm_norm.weight"], this.h, T, {
                size: HEAD_V, groups: g.linearV, inStride: g.value, outStride: g.value,
                gate: this.z, gateStride: g.value, gateGroup: HEAD_V,
            });
            gpu.matmul(L.matrices["ssm_out.weight"], this.h, this.x, T, { accumulate: true });
        }
        else {
            gpu.matmul(L.matrices["attn_q.weight"], this.h, this.big, T); // [T, heads x (256 q | 256 gate)]
            gpu.matmul(L.matrices["attn_k.weight"], this.h, this.k, T);
            gpu.matmul(L.matrices["attn_v.weight"], this.h, this.v, T);
            gpu.norm(this.big, L.vectors["attn_q_norm.weight"], this.mid, T, {
                size: HEAD, groups: g.heads, inStride: 2 * g.attention, inGroup: 2 * HEAD, outStride: g.attention,
            });
            gpu.norm(this.k, L.vectors["attn_k_norm.weight"], this.kn, T, { size: HEAD, groups: g.kvHeads });
            gpu.rope(this.mid, this.invFreq, T, g.heads, g.attention, HEAD, this.position);
            gpu.rope(this.kn, this.invFreq, T, g.kvHeads, g.kv, HEAD, this.position);
            gpu.store(this.kn, L.K, T, this.position, this.maxPositions, this.kv, g);
            gpu.store(this.v, L.V, T, this.position, this.maxPositions, this.kv, g);
            gpu.attention(this.mid, L.K, L.V, this.big, this.h, T, this.position, this.maxPositions, this.kv, this.part, this.ml, g);
            gpu.matmul(L.matrices["attn_output.weight"], this.h, this.x, T, { accumulate: true });
        }
        gpu.norm(this.x, L.vectors["post_attention_norm.weight"], this.h, T, { size: this.hidden });
        gpu.matmul(L.matrices["ffn_gate.weight"], this.h, this.f1, T);
        gpu.matmul(L.matrices["ffn_up.weight"], this.h, this.f2, T);
        gpu.siluMul(this.f1, this.f2, T * this.intermediate);
        gpu.matmul(L.matrices["ffn_down.weight"], this.f2, this.x, T, { accumulate: true });
    }
}
