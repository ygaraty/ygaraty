import { validatePlan, executePlan, bindKernel, assertStateReady, resetPlanState } from "./inference_plan.mjs";
// Generated deployment module: edit dense_gpu.mts, then npm run build:webgpu.
import { Gpu as KernelGpu, F16, objectRecord, positiveInteger, isPackedKind, wgslMemo } from "./webgpu_kernels.mjs";
const LAYER_MATRICES = ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];
const LAYER_VECTORS = ["attn_norm.weight", "ffn_norm.weight"];
export function isDenseManifest(value) {
    return typeof value === "object" && value !== null &&
        value.config?.family === "dense";
}
export function parseManifest(value) {
    const m = objectRecord(value, "dense manifest");
    const c = objectRecord(m.config, "dense config");
    if (c.family !== "dense")
        throw new Error("Not a dense-family manifest");
    for (const key of ["layers", "hidden", "intermediate", "heads", "kv_heads", "head_dim", "vocab"])
        positiveInteger(c[key], key);
    for (const key of ["rope_theta", "eps", "attention_scale"])
        if (typeof c[key] !== "number" || !Number.isFinite(c[key]) || c[key] <= 0)
            throw new Error(`Invalid ${key}`);
    if (c.rope_pairs !== undefined && c.rope_pairs !== "adjacent" && c.rope_pairs !== "halves")
        throw new Error("Invalid rope_pairs");
    const cfg = c;
    if (![64, 128].includes(cfg.head_dim))
        throw new Error("Unsupported head_dim");
    if (cfg.heads % cfg.kv_heads || cfg.heads / cfg.kv_heads > 8)
        throw new Error("Unsupported query group");
    if (cfg.heads * cfg.head_dim !== cfg.hidden || cfg.hidden % 256 || cfg.intermediate % 256)
        throw new Error("Unsupported widths");
    const tensor = (value, chunk = false) => {
        const t = objectRecord(value, "tensor");
        if (typeof t.file !== "string" || !t.file || !isPackedKind(t.kind) || !Array.isArray(t.shape) ||
            t.shape.length < 1 || t.shape.length > 2)
            throw new Error("Invalid tensor descriptor");
        for (const width of t.shape)
            positiveInteger(width, "tensor shape");
        positiveInteger(t.row_bytes, "row_bytes");
        if (chunk && (typeof t.first !== "number" || !Number.isSafeInteger(t.first) || t.first < 0))
            throw new Error("Invalid chunk offset");
    };
    if (!Array.isArray(m.layers) || !m.layers.length)
        throw new Error("Missing layer list");
    let previous;
    for (const value of m.layers) {
        const layer = objectRecord(value, "layer");
        if (typeof layer.index !== "number" || !Number.isInteger(layer.index) || layer.index < 0 || layer.index >= cfg.layers ||
            (previous !== undefined && layer.index !== previous + 1) || layer.type !== "dense")
            throw new Error("Invalid layer order/type");
        previous = layer.index;
        const tensors = objectRecord(layer.tensors, "layer tensors");
        for (const name of [...LAYER_MATRICES, ...LAYER_VECTORS])
            if (!(name in tensors))
                throw new Error(`Missing ${name}`);
        for (const value of Object.values(tensors))
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
            if (chunk.shape.length !== 2 || chunk.first !== offset)
                throw new Error("Invalid row coverage");
            offset += chunk.shape[0];
        }
        if (key === "head") {
            tensor(projection.norm);
            positiveInteger(projection.vocab, "vocab");
            if (offset !== projection.vocab)
                throw new Error("Output row count differs from vocab");
        }
    }
    return value;
}
const SPLIT = 512; // keys per attention workgroup (split-K, combined after)
// WS-35: a decode step's keys in splits of DECODE_SPLIT, so many workgroups share a long context (Kolibri's
// 1,024-key window was 2-3 splits x 4 KV heads: about a dozen workgroups on the whole GPU). gpu.attentionSplit
// (e.g. 64) turns it on for steps whose partial results fit the buffers sized for SPLIT; unset: SPLIT.
function splitFor(gpu, part, tokens, position, H, D) {
    const want = gpu.attentionSplit ?? SPLIT; // opt-in until WS-35's gate
    const rows = ("buffer" in part ? (part.size ?? part.buffer.size - (part.offset ?? 0)) : part.size) / (H * D * 4);
    return want < SPLIT && tokens * Math.ceil((position + tokens) / want) <= rows ? want : SPLIT;
}
const BLOCK = 64; // keys per online-softmax block
// RoPE in place: x[t, head, a] and x[t, head, b] rotated by (position + t) * inv_freq[i], with (a, b) =
// (2i, 2i+1) for adjacent pairs and (i, i + D/2) for halves.
const ropeSource = (D, pairs) => /* wgsl */ `
struct P { tokens: u32, heads: u32, stride: u32, position: u32 };
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> inv_freq: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${D / 2})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x / p.heads; let h = wg.x % p.heads; let i = li.x;
  let angle = f32(p.position + t) * inv_freq[i];
  let c = cos(angle); let s = sin(angle);
  let b = t * p.stride + h * ${D}u + ${pairs === "halves" ? "i" : "2u * i"};
  let e = b + ${pairs === "halves" ? `${D / 2}u` : "1u"};
  let x1 = x[b]; let x2 = x[e];
  x[b] = x1 * c - x2 * s;
  x[e] = x2 * c + x1 * s;
}`;
const rope = wgslMemo(ropeSource); // WS-33: one source string object per arguments
// K or V rows [T, kvHeads x D] into the cache [kvHeads][capacity][D] at row `position`; float16 packs pairs.
const storeSource = (kv, KV, D) => /* wgsl */ `
struct P { tokens: u32, position: u32, capacity: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> cache: array<${kv === "f16" ? "u32" : "f32"}>;
@group(0) @binding(2) var<uniform> p: P;
${F16}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.y; let i = gid.x;
  if (t >= p.tokens) { return; }
  ${kv === "f16" ? `
  if (i >= ${KV * D / 2}u) { return; }
  let g = i / ${D / 2}u; let c = i % ${D / 2}u;
  let src = t * ${KV * D}u + g * ${D}u + 2u * c;
  cache[(g * p.capacity + p.position + t) * ${D / 2}u + c] = f16_bits(x[src]) | (f16_bits(x[src + 1u]) << 16u);` : `
  if (i >= ${KV * D}u) { return; }
  let g = i / ${D}u; let c = i % ${D}u;
  cache[(g * p.capacity + p.position + t) * ${D}u + c] = x[t * ${KV * D}u + i];`}
}`;
const store = wgslMemo(storeSource); // WS-33: one source string object per arguments
// Causal attention, split over keys: a workgroup takes one token, one KV head (its G query heads share each
// key and value read) and up to SPLIT keys, in blocks of 64 with an online softmax; it writes its
// unnormalized outputs and (max, sum) per query head for the combine pass. Thread d owns output dimension d.
// `windowed` (Kolibri-1's sliding layers): token t at position p sees only keys above p - window (the
// uniform's `span`; 0: all), so a split wholly before the window writes an empty partial.
const attentionSource = (kv, H, KV, D, windowed = false, SPLIT = 512) => {
    const G = H / KV;
    const key = kv === "f16"
        ? `fn kd(i: u32) -> f32 { return f16_value((K[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu); }
fn vd(i: u32) -> f32 { return f16_value((V[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu); }`
        : `fn kd(i: u32) -> f32 { return K[i]; }
fn vd(i: u32) -> f32 { return V[i]; }`;
    return /* wgsl */ `
struct P { tokens: u32, position: u32, splits: u32, capacity: u32, scale: f32, ${windowed ? "span" : "p0"}: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> q: array<f32>;     // [T, H, D] after RoPE
@group(0) @binding(1) var<storage, read> K: array<${kv === "f16" ? "u32" : "f32"}>;
@group(0) @binding(2) var<storage, read> V: array<${kv === "f16" ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;      // [T][splits][H][D]
@group(0) @binding(4) var<storage, read_write> ml: array<vec2<f32>>;  // [T][splits][H]
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<f32, ${G * D}>;
var<workgroup> sc: array<f32, ${G * BLOCK}>;
var<workgroup> mh: array<f32, ${G}>;
var<workgroup> lh: array<f32, ${G}>;
var<workgroup> ch: array<f32, ${G}>;
${F16}
${key}
@compute @workgroup_size(${D})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let s = wg.x % p.splits; let tg = wg.x / p.splits;
  let g = tg % ${KV}u; let t = tg / ${KV}u; let d = li.x;
  for (var h = 0u; h < ${G}u; h++) { qs[h * ${D}u + d] = q[(t * ${H}u + g * ${G}u + h) * ${D}u + d]; }
  if (d < ${G}u) { mh[d] = -3.0e38; lh[d] = 0.0; }
  workgroupBarrier();
  let begin = ${windowed ? `max(s * ${SPLIT}u, select(0u, p.position + t + 1u - p.span, p.span > 0u && p.position + t + 1u > p.span))` : `s * ${SPLIT}u`};
  let end = min(${windowed ? `s * ${SPLIT}u` : "begin"} + ${SPLIT}u, p.position + t + 1u);  // keys 0..position+t
  let base = g * p.capacity;
  var acc: array<f32, ${G}>;
  for (var h = 0u; h < ${G}u; h++) { acc[h] = 0.0; }
  for (var k0 = begin; k0 < end; k0 += ${BLOCK}u) {
    let n = min(${BLOCK}u, end - k0);
    for (var w = d; w < ${G * BLOCK}u; w += ${D}u) {  // scores: head w / 64, key w % 64
      let h = w / ${BLOCK}u; let k = w % ${BLOCK}u;
      var score = -3.0e38;
      if (k < n) {
        let kb = (base + k0 + k) * ${D}u;
        var dot = 0.0;
        for (var e = 0u; e < ${D}u; e++) { dot += qs[h * ${D}u + e] * kd(kb + e); }
        score = dot * p.scale;
      }
      sc[w] = score;
    }
    workgroupBarrier();
    if (d < ${G}u) {  // thread h: the block's max, the rescale of what came before, the new sum
      var m = mh[d];
      for (var k = 0u; k < n; k++) { m = max(m, sc[d * ${BLOCK}u + k]); }
      let corr = exp(mh[d] - m);
      var l = lh[d] * corr;
      for (var k = 0u; k < n; k++) { let e = exp(sc[d * ${BLOCK}u + k] - m); sc[d * ${BLOCK}u + k] = e; l += e; }
      mh[d] = m; lh[d] = l; ch[d] = corr;
    }
    workgroupBarrier();
    let vb = (base + k0) * ${D}u + d;
    for (var h = 0u; h < ${G}u; h++) {
      var a = acc[h] * ch[h];
      for (var k = 0u; k < n; k++) { a += sc[h * ${BLOCK}u + k] * vd(vb + k * ${D}u); }
      acc[h] = a;
    }
    workgroupBarrier();
  }
  for (var h = 0u; h < ${G}u; h++) {
    let qh = g * ${G}u + h;
    part[((t * p.splits + s) * ${H}u + qh) * ${D}u + d] = acc[h];
    if (d == 0u) { ml[(t * p.splits + s) * ${H}u + qh] = vec2<f32>(mh[h], lh[h]); }
  }
}`;
};
const attention = wgslMemo(attentionSource); // WS-33: one source string object per arguments
// Combine the splits of one (token, query head).
const combineSource = (H, D) => /* wgsl */ `
struct P { tokens: u32, splits: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> ml: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>; // [T, H*D]
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(${D})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x / ${H}u; let h = wg.x % ${H}u; let d = li.x;
  var mm = -3.0e38;
  for (var s = 0u; s < p.splits; s++) { mm = max(mm, ml[(t * p.splits + s) * ${H}u + h].x); }
  var l = 0.0; var o = 0.0;
  for (var s = 0u; s < p.splits; s++) {
    let i = (t * p.splits + s) * ${H}u + h;
    let f = exp(ml[i].x - mm);
    l += ml[i].y * f;
    o += part[i * ${D}u + d] * f;
  }
  out[(t * ${H}u + h) * ${D}u + d] = o / l;
}`;
const combine = wgslMemo(combineSource); // WS-33: one source string object per arguments
// The stage's own operations, as functions over any shared-kernel GPU: the contributor page makes its GPU
// before it learns which family its stage belongs to (the self-test runs first).
export function denseRope(gpu, x, invFreq, tokens, heads, D, position, pairs = "adjacent") {
    const u = gpu.uniform([["u", tokens], ["u", heads], ["u", heads * D], ["u", position]]);
    gpu.dispatch(`dense_rope_${pairs}${D}`, rope(D, pairs), [x, invFreq, u], [tokens * heads, 1, 1]);
}
export function denseStore(gpu, x, cache, tokens, position, capacity, kv, KV, D) {
    const u = gpu.uniform([["u", tokens], ["u", position], ["u", capacity], ["u", 0]]);
    const words = kv === "f16" ? KV * D / 2 : KV * D;
    gpu.dispatch(`dense_store_${kv}_${KV}x${D}`, store(kv, KV, D), [x, cache, u], [Math.ceil(words / 256), tokens, 1]);
}
export function denseAttention(gpu, q, K, V, out, tokens, position, capacity, kv, H, KV, D, scale, part, ml) {
    const S = splitFor(gpu, part, tokens, position, H, D), splits = Math.ceil((position + tokens) / S);
    const u = gpu.uniform([["u", tokens], ["u", position], ["u", splits], ["u", capacity], ["f", scale], ["u", 0], ["u", 0], ["u", 0]]);
    gpu.dispatch(`dense_attention_${kv}_${H}_${KV}x${D}${S === SPLIT ? "" : `_s${S}`}`, attention(kv, H, KV, D, false, S), [q, K, V, part, ml, u], [tokens * KV * splits, 1, 1]);
    const v = gpu.uniform([["u", tokens], ["u", splits], ["u", 0], ["u", 0]]);
    gpu.dispatch(`dense_combine_${H}x${D}`, combine(H, D), [part, ml, out, v], [tokens * H, 1, 1]);
}
// As denseAttention, over the keys of the last `window` positions (0: all of them), for sliding-window layers.
export function denseWindowedAttention(gpu, q, K, V, out, tokens, position, capacity, kv, H, KV, D, scale, part, ml, window) {
    const S = splitFor(gpu, part, tokens, position, H, D), splits = Math.ceil((position + tokens) / S);
    const u = gpu.uniform([["u", tokens], ["u", position], ["u", splits], ["u", capacity], ["f", scale], ["u", window], ["u", 0], ["u", 0]]);
    gpu.dispatch(`dense_attention_window_${kv}_${H}_${KV}x${D}${S === SPLIT ? "" : `_s${S}`}`, attention(kv, H, KV, D, true, S), [q, K, V, part, ml, u], [tokens * KV * splits, 1, 1]);
    const v = gpu.uniform([["u", tokens], ["u", splits], ["u", 0], ["u", 0]]);
    gpu.dispatch(`dense_combine_${H}x${D}`, combine(H, D), [part, ml, out, v], [tokens * H, 1, 1]);
}
export class Gpu extends KernelGpu {
}
// One pipeline stage: optionally the embedding, then layers first..last-1, optionally the output head.
export class Stage {
    gpu;
    maxTokens;
    maxPositions;
    config;
    hidden;
    intermediate;
    manifest;
    kv;
    slots;
    maxBatch;
    position = 0;
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
    q;
    k;
    v;
    att;
    f1;
    f2;
    part;
    ml;
    ids;
    invFreq;
    outputNorm;
    logits;
    token;
    constructor(gpu, manifest, maxTokens, maxPositions, { kv = "f16", slots = 1 } = {}) {
        this.gpu = gpu;
        this.maxTokens = maxTokens;
        this.maxPositions = maxPositions;
        if (!["f16", "f32"].includes(kv))
            throw new Error(`kv ${kv}`);
        this.manifest = parseManifest(manifest);
        this.config = this.manifest.config;
        this.hidden = this.config.hidden;
        this.intermediate = this.config.intermediate;
        this.plan = this.manifest.plan === undefined ? null : validatePlan(this.manifest.plan, this.manifest, "dense", this.hidden);
        if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || !Number.isSafeInteger(maxPositions) || maxPositions < maxTokens)
            throw new Error("Invalid token/context capacity");
        if (!Number.isSafeInteger(slots) || slots < 1)
            throw Error("Invalid slot count");
        this.kv = kv;
        this.slots = slots;
        this.maxBatch = Math.min(slots, maxTokens);
        this.positions = new Array(slots).fill(0);
    }
    use(slot) {
        if (!Number.isInteger(slot) || !(slot >= 0 && slot < this.slots))
            throw new Error(`slot ${slot} of ${this.slots}`);
        for (const L of this.layers) {
            L.K = L.Ks[slot];
            L.V = L.Vs[slot];
        }
        this.slot = slot;
        this.position = this.positions[slot];
    }
    async load(fetchTensor, log = () => { }) {
        const gpu = this.gpu, m = this.manifest, c = this.config;
        const tensor = async (t) => {
            const data = await fetchTensor(t.file);
            return { kind: t.kind, rows: t.shape[0], cols: t.shape[t.shape.length - 1], rowBytes: t.row_bytes, buffer: gpu.upload(data) };
        };
        const floats = async (t) => gpu.upload(await fetchTensor(t.file));
        if (m.embed) {
            this.embed = [];
            for (const ch of m.embed.chunks)
                this.embed.push({ ...(await tensor(ch)), first: ch.first });
            log(`embedding: ${this.embed.length} chunks`);
        }
        const cacheBytes = this.maxPositions * c.kv_heads * c.head_dim * (this.kv === "f16" ? 2 : 4);
        this.layers = [];
        for (const L of m.layers) {
            const matrices = {};
            const vectors = {};
            for (const name of LAYER_MATRICES)
                matrices[name] = await tensor(L.tensors[name]);
            for (const name of LAYER_VECTORS)
                vectors[name] = await floats(L.tensors[name]);
            const Ks = Array.from({ length: this.slots }, () => gpu.buffer(cacheBytes));
            const Vs = Array.from({ length: this.slots }, () => gpu.buffer(cacheBytes));
            this.layers.push({ index: L.index, matrices, vectors, K: Ks[0], V: Vs[0], Ks, Vs });
            log(`layer ${L.index} loaded; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
        }
        if (m.head) {
            this.outputNorm = await floats(m.head.norm);
            this.output = m.head.shared_embedding && this.embed ? this.embed : [];
            for (const ch of (this.output.length ? [] : m.head.chunks))
                this.output.push({ ...(await tensor(ch)), first: ch.first });
            this.vocab = m.head.vocab;
            this.logits = gpu.buffer(this.maxBatch * this.vocab * 4);
            this.token = gpu.buffer(this.maxBatch * 256);
            log(`output: ${this.output.length} chunks; allocated ${(gpu.allocated / 2 ** 20).toFixed(1)} MiB`);
        }
        const D = c.head_dim, invFreq = new Float32Array(D / 2);
        for (let i = 0; i < D / 2; i++)
            invFreq[i] = Math.pow(c.rope_theta, -(2 * i) / D);
        this.invFreq = gpu.upload(invFreq);
        const T = this.maxTokens;
        this.ids = gpu.buffer(T * 4);
        this.x = gpu.buffer(T * c.hidden * 4);
        this.h = gpu.buffer(T * c.hidden * 4);
        this.q = gpu.buffer(T * c.heads * D * 4);
        this.k = gpu.buffer(T * c.kv_heads * D * 4);
        this.v = gpu.buffer(T * c.kv_heads * D * 4);
        this.att = gpu.buffer(T * c.heads * D * 4);
        this.f1 = gpu.buffer(T * c.intermediate * 4);
        this.f2 = gpu.buffer(T * c.intermediate * 4);
        const splits = Math.ceil(this.maxPositions / SPLIT);
        this.part = gpu.buffer(T * splits * c.heads * D * 4);
        this.ml = gpu.buffer(T * splits * c.heads * 2 * 4);
        this.loaded = true;
    }
    checkInput(ids, hidden, tokens) {
        if (this.embed) {
            const vocab = this.embed.reduce((n, chunk) => n + chunk.rows, 0);
            if (hidden != null || !ids || ids.length !== tokens || !Array.from(ids).every((id) => Number.isSafeInteger(id) && id >= 0 && id < vocab))
                throw new Error("Invalid embedding input");
        }
        else if (ids != null || !(hidden instanceof Float32Array) || hidden.length !== tokens * this.hidden || !hidden.every(Number.isFinite)) {
            throw new Error("Invalid hidden input");
        }
    }
    // Run `tokens` positions of one slot: from token ids (embedding stage) or hidden input [tokens, hidden].
    // WS-32: `gpuIds`: one decode token whose id is already in this.ids (copied there on the GPU from this.token).
    residentDecode = true; // KV caches only: a step run past the stop rule is undone by rewinding
    async run({ ids = null, hidden = null, tokens, slot = 0, gpuIds = false }) {
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
        else
            this.checkInput(ids, hidden, tokens);
        if (tokens > this.maxTokens)
            throw new Error(`chunk ${tokens} > ${this.maxTokens}`);
        if (this.position + tokens > this.maxPositions)
            throw new Error(`position ${this.position + tokens} > ${this.maxPositions}`);
        const upload = () => {
            if (ids)
                gpu.device.queue.writeBuffer(this.ids, 0, new Uint32Array(ids));
            else if (!gpuIds)
                gpu.device.queue.writeBuffer(this.x, 0, hidden);
        };
        this.submitMs = [];
        this.submitLayers = [];
        await executePlan(this.plan, tokens, gpu, this.kernels(), this.submitMs, this.submitLayers, this.planContext(tokens, false, [{ slot, row: 0, count: tokens, position: this.position }]), upload);
        this.position += tokens;
        this.positions[slot] = this.position;
    }
    // One decode token for each of several slots (rows 0..T-1).
    async runBatch({ slots, ids = null, hidden = null }) {
        assertStateReady(this, slots);
        const gpu = this.gpu, T = slots.length;
        if (!this.loaded || !T)
            throw Error("Stage unavailable or empty batch");
        this.checkInput(ids, hidden, T);
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
        this.position = this.positions[this.slot];
    }
    // Adapters bind resources and primitives; mathematical composition lives on Tape.
    kernels() {
        const gpu = this.gpu;
        return {
            embedding: bindKernel(gpu.gather.bind(gpu)), rms_norm: bindKernel(gpu.norm.bind(gpu)),
            matmul: bindKernel(gpu.matmul.bind(gpu)), silu_mul: bindKernel(gpu.siluMul.bind(gpu)),
            argmax: bindKernel(gpu.argmax.bind(gpu)),
            rotary_adjacent: bindKernel(denseRope.bind(null, gpu)),
            rotary_halves: bindKernel((x, inv, tokens, heads, dim, position) => denseRope(gpu, x, inv, tokens, heads, dim, position, "halves")), cache_store_grouped: bindKernel(denseStore.bind(null, gpu)),
            grouped_attention: bindKernel(denseAttention.bind(null, gpu)),
        };
    }
    planContext(tokens, batch, sequences) {
        return {
            stateOwner: this,
            buffers: { x: this.x, h: this.h, f1: this.f1, f2: this.f2, part: this.part, ml: this.ml, ids: this.ids, invFreq: this.invFreq, outputNorm: this.outputNorm, logits: this.logits, token: this.token, q: this.q, k: this.k, v: this.v, att: this.att },
            layer: index => this.layers[index - this.manifest.layers[0].index],
            resources: { embed: this.embed, output: this.output },
            variables: { tokens, batch, capacity: this.maxPositions, kv: this.kv, vocab: this.vocab,
                head_enabled: true, head_all: batch, head_tokens: batch ? tokens : 1, head_offset: (tokens - 1) * this.hidden },
            sequences,
        };
    }
    // Start a new sequence (the cache needs no clearing: attention reads positions below `position` only).
    async reset(slot = null) {
        if (slot !== null && (!Number.isInteger(slot) || slot < 0 || slot >= this.slots))
            throw Error("Invalid reset slot");
        const resetSlots = slot === null ? [...this.positions.keys()] : [slot];
        await resetPlanState(this, resetSlots, async () => {
            for (const k of slot === null ? [...this.positions.keys()] : [slot])
                this.positions[k] = 0;
            this.position = this.positions[this.slot];
        });
    }
    // T rows through one layer; `parts`: which rows belong to which slot at which position (one part for a
    // prefill chunk, one per slot for a batch).
    layer(L, T, parts) {
        const gpu = this.gpu, c = this.config, D = c.head_dim, H = c.heads, KV = c.kv_heads;
        const row = (buffer, r, floats) => ({ buffer, offset: r * floats * 4 });
        gpu.norm(this.x, L.vectors["attn_norm.weight"], this.h, T, { size: c.hidden, eps: c.eps });
        gpu.matmul(L.matrices["attn_q.weight"], this.h, this.q, T);
        gpu.matmul(L.matrices["attn_k.weight"], this.h, this.k, T);
        gpu.matmul(L.matrices["attn_v.weight"], this.h, this.v, T);
        for (const p of parts) {
            const q = row(this.q, p.row, H * D), k = row(this.k, p.row, KV * D), v = row(this.v, p.row, KV * D);
            const pairs = c.rope_pairs ?? "adjacent";
            denseRope(gpu, q, this.invFreq, p.count, H, D, p.position, pairs);
            denseRope(gpu, k, this.invFreq, p.count, KV, D, p.position, pairs);
            denseStore(gpu, k, L.Ks[p.slot], p.count, p.position, this.maxPositions, this.kv, KV, D);
            denseStore(gpu, v, L.Vs[p.slot], p.count, p.position, this.maxPositions, this.kv, KV, D);
            denseAttention(gpu, q, L.Ks[p.slot], L.Vs[p.slot], row(this.att, p.row, H * D), p.count, p.position, this.maxPositions, this.kv, H, KV, D, c.attention_scale, this.part, this.ml);
        }
        gpu.matmul(L.matrices["attn_output.weight"], this.att, this.x, T, { accumulate: true });
        gpu.norm(this.x, L.vectors["ffn_norm.weight"], this.h, T, { size: c.hidden, eps: c.eps });
        gpu.matmul(L.matrices["ffn_gate.weight"], this.h, this.f1, T);
        gpu.matmul(L.matrices["ffn_up.weight"], this.h, this.f2, T);
        gpu.siluMul(this.f1, this.f2, T * c.intermediate);
        gpu.matmul(L.matrices["ffn_down.weight"], this.f2, this.x, T, { accumulate: true });
    }
    // Logits of the call's last position and its greedy token.
    head(T) {
        const gpu = this.gpu;
        gpu.norm(this.x, this.outputNorm, this.h, 1, { size: this.hidden, inOffset: (T - 1) * this.hidden, eps: this.config.eps });
        for (const ch of this.output ?? [])
            gpu.matmul(ch, this.h, this.logits, 1, { yStride: this.vocab, rowBase: ch.first });
        gpu.argmax(this.logits, this.token, this.vocab);
    }
    async next() {
        const v = await this.gpu.read(this.token, 4);
        return new Uint32Array(v.buffer)[0];
    }
    headBatch(T) {
        const gpu = this.gpu;
        gpu.norm(this.x, this.outputNorm, this.h, T, { size: this.hidden, eps: this.config.eps });
        for (const ch of this.output ?? [])
            gpu.matmul(ch, this.h, this.logits, T, { yStride: this.vocab, rowBase: ch.first });
        for (let t = 0; t < T; t++)
            gpu.argmax({ buffer: this.logits, offset: t * this.vocab * 4 }, { buffer: this.token, offset: t * 256 }, this.vocab);
    }
    async nextBatch(T) {
        const v = new Uint32Array((await this.gpu.read(this.token, T * 256)).buffer);
        return Array.from({ length: T }, (_, t) => v[t * 64]);
    }
}
