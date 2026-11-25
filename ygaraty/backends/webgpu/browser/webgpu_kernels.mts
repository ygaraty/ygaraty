// Generated deployment module: edit webgpu_kernels.mts, then npm run build:webgpu.
// Shared inference WebGPU runtime and numerical primitives.
// No model topology, layer count, hidden width or persistent model state.
// Matrix weights are row-major packed GGUF rows; activations/arithmetic are F32.
// Consumers own buffers and sequence operations inside begin()/end().

export type BufferBinding = GPUBuffer | GPUBufferBinding;
export interface ProfileRecord { encodeMs: number; gpuMs: number | null; wallMs: number; dispatches: number; uniforms: number;
                                 byName: Record<string, number>; gpuByName?: Record<string, number>; bytesByName?: Record<string, number>;
                                 expandMs?: number }
export type PackedKind = "F32" | "F16" | "Q4_0" | "IQ4_NL" | "Q4_K" | "Q5_K" | "Q6_K" | "Q8_0" | "Q4_KR" | "Q5_KR" | "Q6_KR";  // *_KR: exact repacks (experiment 141, DO-80)
export interface PackedMatrix {
  kind: PackedKind;
  rows: number;
  cols: number;
  rowBytes: number;
  buffer: GPUBuffer;
}
export interface EmbeddingChunk extends PackedMatrix { first: number }
export interface GemmTile { GR: number; GT: number; RR: number; RT: number; V?: boolean }  // V: WS-50 vec4 staging
export interface MatmulOptions {
  xStride?: number; xOffset?: number; yStride?: number; yOffset?: number;
  accumulate?: boolean; rows?: number; rowBase?: number;
}
export interface NormOptions {
  size: number; groups?: number; inStride?: number; inGroup?: number;
  inOffset?: number; outStride?: number; outGroup?: number; outOffset?: number;
  gate?: BufferBinding | null; gateStride?: number; gateGroup?: number;
  gateOffset?: number; eps?: number;
}
export type UniformValue = readonly ["u" | "f", number];
export type Workgroups = [number, number?, number?];

/** Network bytes are validated by the model adapter before GPU upload. */
export type FetchTensor = (name: string) => Promise<ArrayBuffer>;
export type StageInput = { tokens: number } & (
  | { ids: readonly number[] | Uint32Array; hidden?: null }
  | { ids?: null; hidden: Float32Array<ArrayBuffer> }
);
export interface StageRunner<Result> {
  readonly position: number;
  run(input: StageInput): Promise<Result>;
  reset(): void | Promise<void>;
}

/** Structural boundary helpers: JSON is unknown until checked. */
export function objectRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`Expected object: ${label}`);
  return value as Record<string, unknown>;
}
export function positiveInteger(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1 || value > 0xffffffff)
    throw new Error(`Invalid positive integer: ${label}`);
}
/** A packed kind's WGSL element decoder `deq(base_byte, element)` over `w: array<u32>` (needs BYTES). */
export function deqSource(kind: PackedKind): string {
  if (!Object.hasOwn(DEQ, kind)) throw new Error(`Unsupported packed matrix type: ${kind}`);
  return DEQ[kind as keyof typeof DEQ];
}
export function isPackedKind(value: unknown): value is PackedKind {
  return typeof value === "string" && Object.hasOwn(DEQ, value);
}

const TT = 8;
export const MAX_DISPATCH = 65535;

// WS-33: each shader generator returns the same string object for the same arguments, so a dispatch neither rebuilds
// its WGSL source nor rehashes a fresh multi-kilobyte string for the pipeline cache (V8 caches a string's hash on the
// object). The text is unchanged.
export const wgslCache = { enabled: true };  // WS-33 arms: false regenerates every source as before
export function wgslMemo<A extends unknown[]>(generate: (...args: A) => string): (...args: A) => string {
  const cache = new Map<string, string>();
  return (...args: A) => {
    if (!wgslCache.enabled) return generate(...args);
    const key = JSON.stringify(args);
    let code = cache.get(key);
    if (code === undefined) { code = generate(...args); cache.set(key, code); }
    return code;
  };
}

export const BYTES = /* wgsl */ `
fn byte_at(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
fn word_at(o: u32) -> u32 {
  let s = (o & 3u) * 8u; let i = o >> 2u;
  if (s == 0u) { return w[i]; }
  return (w[i] >> s) | (w[i + 1u] << (32u - s));
}
fn bytes4(v: u32) -> vec4<u32> { return (vec4<u32>(v) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u); }
fn half_at(o: u32) -> f32 { return unpack2x16float(byte_at(o) | (byte_at(o + 1u) << 8u)).x; }
fn sbyte_at(o: u32) -> f32 { return f32(bitcast<i32>(byte_at(o) << 24u) >> 24u); }
fn scale_min(b: u32, j: u32) -> vec2<f32> {
  // Q4_K / Q5_K 6-bit scale and min of sub-block j, from 12 bytes at b
  if (j < 4u) {
    return vec2<f32>(f32(byte_at(b + j) & 63u), f32(byte_at(b + j + 4u) & 63u));
  }
  let s = (byte_at(b + j + 4u) & 15u) | ((byte_at(b + j - 4u) >> 6u) << 4u);
  let m = (byte_at(b + j + 4u) >> 4u) | ((byte_at(b + j) >> 6u) << 4u);
  return vec2<f32>(f32(s), f32(m));
}`;

const DEQ = {
  Q4_0: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 32u) * 18u; let j = e % 32u;
    let q = (byte_at(b + 2u + j % 16u) >> ((j / 16u) * 4u)) & 15u;
    return half_at(b) * (f32(q) - 8.0);
  }`,
  // DO-95: IQ4_NL is Q4_0's 18-byte block (f16 scale, 16 nibble bytes) with ggml's 16-level non-linear table.
  IQ4_NL: /* wgsl */ `var<private> IQ4NL: array<f32, 16> = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
  fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 32u) * 18u; let j = e % 32u;
    let q = (byte_at(b + 2u + j % 16u) >> ((j / 16u) * 4u)) & 15u;
    return half_at(b) * IQ4NL[q];
  }`,
  F16: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let v = unpack2x16float(w[(base >> 2u) + (e >> 1u)]);
    return v[e & 1u];
  }`,
  F32: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 { return bitcast<f32>(w[(base >> 2u) + e]); }`,
  Q8_0: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 32u) * 34u;
    return half_at(b) * sbyte_at(b + 2u + e % 32u);
  }`,
  Q4_K: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 144u; let j = e % 256u; let sub = j / 32u; let l = j % 32u;
    let sm = scale_min(b + 4u, sub);
    let q = (byte_at(b + 16u + (sub / 2u) * 32u + l) >> ((sub % 2u) * 4u)) & 15u;
    return half_at(b) * sm.x * f32(q) - half_at(b + 2u) * sm.y;
  }`,
  // WS-40: Q4_K repacked at load (repackQ4K): per 256-element block, 8 (scale, min) f32 pairs (each d * sc and
  // dm * m exactly as the decoders below compute them), then the 128 quant bytes unchanged: 192 bytes.
  Q4_KR: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 192u; let j = e % 256u; let sub = j / 32u; let l = j % 32u;
    let q = (byte_at(b + 64u + (sub / 2u) * 32u + l) >> ((sub % 2u) * 4u)) & 15u;
    return bitcast<f32>(w[(b >> 2u) + 2u * sub]) * f32(q) - bitcast<f32>(w[(b >> 2u) + 2u * sub + 1u]);
  }`,
  // DO-80: Q5_K repacked (repackQ5K): 8 (scale, min) f32 pairs, the 32 high-bit bytes, the 128 low-nibble bytes: 224.
  Q5_KR: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 224u; let j = e % 256u; let sub = j / 32u; let l = j % 32u;
    let lo = (byte_at(b + 96u + (sub / 2u) * 32u + l) >> ((sub % 2u) * 4u)) & 15u;
    let hi = (byte_at(b + 64u + l) >> sub) & 1u;
    return bitcast<f32>(w[(b >> 2u) + 2u * sub]) * f32(lo | (hi << 4u)) - bitcast<f32>(w[(b >> 2u) + 2u * sub + 1u]);
  }`,
  // DO-80: Q6_K repacked (repackQ6K): 16 scales d * sc as f32, ql (128), qh (64): 256 bytes, every block aligned.
  Q6_KR: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 256u; let j = e % 256u;
    let h = j / 128u; let r = j % 128u;
    let lo = (byte_at(b + 64u + h * 64u + r % 64u) >> ((r / 64u) * 4u)) & 15u;
    let hi = (byte_at(b + 192u + h * 32u + r % 32u) >> ((r / 32u) * 2u)) & 3u;
    let q = f32(i32(lo | (hi << 4u)) - 32);
    return bitcast<f32>(w[(b >> 2u) + j / 16u]) * q;
  }`,
  Q5_K: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 176u; let j = e % 256u; let sub = j / 32u; let l = j % 32u;
    let sm = scale_min(b + 4u, sub);
    let lo = (byte_at(b + 48u + (sub / 2u) * 32u + l) >> ((sub % 2u) * 4u)) & 15u;
    let hi = (byte_at(b + 16u + l) >> sub) & 1u;
    return half_at(b) * sm.x * f32(lo | (hi << 4u)) - half_at(b + 2u) * sm.y;
  }`,
  Q6_K: /* wgsl */ `fn deq(base: u32, e: u32) -> f32 {
    let b = base + (e / 256u) * 210u; let j = e % 256u;
    let h = j / 128u; let r = j % 128u;
    let lo = (byte_at(b + h * 64u + r % 64u) >> ((r / 64u) * 4u)) & 15u;
    let hi = (byte_at(b + 128u + h * 32u + r % 32u) >> ((r / 32u) * 2u)) & 3u;
    let q = f32(i32(lo | (hi << 4u)) - 32);
    return half_at(b + 208u) * sbyte_at(b + 192u + j / 16u) * q;
  }`,
};

// y[t, y_offset + row] (+)= sum_e W[row, e] * x[t, x_offset + e]
const matmulShaderSource = (kind: PackedKind) => /* wgsl */ `
struct P { rows: u32, cols: u32, tokens: u32, row_bytes: u32,
           x_stride: u32, x_offset: u32, y_stride: u32, y_offset: u32,
           accumulate: u32, row_base: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> part: array<f32, ${64 * TT}>;
${BYTES}
${DEQ[kind]}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.x + wg.z * ${MAX_DISPATCH}u;
  if (row >= p.rows) { return; }
  let t0 = wg.y * ${TT}u;
  var acc: array<f32, ${TT}>;
  let base = row * p.row_bytes;
  for (var e = li.x; e < p.cols; e += 64u) {
    let v = deq(base, e);
    for (var t = 0u; t < ${TT}u; t++) {
      if (t0 + t < p.tokens) { acc[t] += v * x[(t0 + t) * p.x_stride + p.x_offset + e]; }
    }
  }
  for (var t = 0u; t < ${TT}u; t++) { part[t * 64u + li.x] = acc[t]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (li.x < s) {
      for (var t = 0u; t < ${TT}u; t++) { part[t * 64u + li.x] += part[t * 64u + li.x + s]; }
    }
    workgroupBarrier();
  }
  if (li.x < ${TT}u && t0 + li.x < p.tokens) {
    let i = (t0 + li.x) * p.y_stride + p.y_offset + p.row_base + row;
    let v = part[li.x * 64u];
    if (p.accumulate == 1u) { y[i] = y[i] + v; } else { y[i] = v; }
  }
}`;
const matmulShader = wgslMemo(matmulShaderSource);  // WS-33: one source string object per arguments

// Word-wise decoding of the K-quants. A work item is one pair of 32-element
// units that share their 32 quant bytes (low and high nibbles): pair g of a
// 256-element block. `setup` (given block byte offset b and pair g) defines
// the scales and the units' first elements el, eh; `word` (given i in 0..7)
// decodes elements el+4i..+3 into `lo` and eh+4i..+3 into `hi`.
export const PAIR = {
  // WS-46: Q8_0 in the same shape: a 256-element "block" is 8 Q8_0 blocks of 34 bytes (f16 scale, 32 int8);
  // pair g is Q8_0 blocks 2g (elements el..el+31) and 2g+1 (eh..eh+31).
  Q8_0: {
    block: 272,
    setup: `
      let qa = b + 68u * g;
      let sl = half_at(qa); let sh = half_at(qa + 34u);
      let qo = qa + 2u; let ho = qa + 36u;
      let el = 64u * g; let eh = el + 32u;`,
    word: `
      let lv = bitcast<vec4<i32>>(bytes4(word_at(qo + 4u * i)) << vec4<u32>(24u)) >> vec4<u32>(24u);
      let hv = bitcast<vec4<i32>>(bytes4(word_at(ho + 4u * i)) << vec4<u32>(24u)) >> vec4<u32>(24u);
      let lo = sl * vec4<f32>(lv);
      let hi = sh * vec4<f32>(hv);`,
  },
  Q5_KR: {
    block: 224,
    setup: `
      let qa = (b >> 2u) + 4u * g;
      let sl = bitcast<f32>(w[qa]); let ml = bitcast<f32>(w[qa + 1u]); let sh = bitcast<f32>(w[qa + 2u]); let mh = bitcast<f32>(w[qa + 3u]);
      let qo = b + 96u + 32u * g; let ho = b + 64u;
      let el = 64u * g; let eh = el + 32u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i)); let hv = bytes4(word_at(ho + 4u * i));
      let lo = sl * vec4<f32>((qv & vec4<u32>(15u)) | (((hv >> vec4<u32>(2u * g)) & vec4<u32>(1u)) << vec4<u32>(4u))) - ml;
      let hi = sh * vec4<f32>((qv >> vec4<u32>(4u)) | (((hv >> vec4<u32>(2u * g + 1u)) & vec4<u32>(1u)) << vec4<u32>(4u))) - mh;`,
  },
  Q6_KR: {
    block: 256,
    setup: `
      let h = g / 2u; let c = g % 2u;
      let qa = (b >> 2u) + 8u * h + 2u * c;
      let la = bitcast<f32>(w[qa]); let lb = bitcast<f32>(w[qa + 1u]);
      let ha = bitcast<f32>(w[qa + 4u]); let hb = bitcast<f32>(w[qa + 5u]);
      let qo = b + 64u + 64u * h + 32u * c; let ho = b + 192u + 32u * h;
      let el = 128u * h + 32u * c; let eh = el + 64u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i)); let hv = bytes4(word_at(ho + 4u * i));
      let sl = select(la, lb, i >= 4u); let sh = select(ha, hb, i >= 4u);
      let lo = sl * (vec4<f32>((qv & vec4<u32>(15u)) | (((hv >> vec4<u32>(2u * c)) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);
      let hi = sh * (vec4<f32>((qv >> vec4<u32>(4u)) | (((hv >> vec4<u32>(4u + 2u * c)) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);`,
  },
  Q4_KR: {
    block: 192,
    setup: `
      let qa = (b >> 2u) + 4u * g;
      let sl = bitcast<f32>(w[qa]); let ml = bitcast<f32>(w[qa + 1u]); let sh = bitcast<f32>(w[qa + 2u]); let mh = bitcast<f32>(w[qa + 3u]);
      let qo = b + 64u + 32u * g;
      let el = 64u * g; let eh = el + 32u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i));
      let lo = sl * vec4<f32>(qv & vec4<u32>(15u)) - ml;
      let hi = sh * vec4<f32>(qv >> vec4<u32>(4u)) - mh;`,
  },
  Q4_K: {
    block: 144,
    setup: `
      let d = half_at(b); let dm = half_at(b + 2u);
      let s0 = scale_min(b + 4u, 2u * g); let s1 = scale_min(b + 4u, 2u * g + 1u);
      let sl = d * s0.x; let ml = dm * s0.y; let sh = d * s1.x; let mh = dm * s1.y;
      let qo = b + 16u + 32u * g;
      let el = 64u * g; let eh = el + 32u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i));
      let lo = sl * vec4<f32>(qv & vec4<u32>(15u)) - ml;
      let hi = sh * vec4<f32>(qv >> vec4<u32>(4u)) - mh;`,
  },
  Q5_K: {
    block: 176,
    setup: `
      let d = half_at(b); let dm = half_at(b + 2u);
      let s0 = scale_min(b + 4u, 2u * g); let s1 = scale_min(b + 4u, 2u * g + 1u);
      let sl = d * s0.x; let ml = dm * s0.y; let sh = d * s1.x; let mh = dm * s1.y;
      let qo = b + 48u + 32u * g; let ho = b + 16u;
      let el = 64u * g; let eh = el + 32u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i)); let hv = bytes4(word_at(ho + 4u * i));
      let lo = sl * vec4<f32>((qv & vec4<u32>(15u)) | (((hv >> vec4<u32>(2u * g)) & vec4<u32>(1u)) << vec4<u32>(4u))) - ml;
      let hi = sh * vec4<f32>((qv >> vec4<u32>(4u)) | (((hv >> vec4<u32>(2u * g + 1u)) & vec4<u32>(1u)) << vec4<u32>(4u))) - mh;`,
  },
  Q6_K: {
    block: 210,
    setup: `
      let h = g / 2u; let c = g % 2u;
      let d = half_at(b + 208u); let si = b + 192u + 8u * h + 2u * c;
      let la = d * sbyte_at(si); let lb = d * sbyte_at(si + 1u);
      let ha = d * sbyte_at(si + 4u); let hb = d * sbyte_at(si + 5u);
      let qo = b + 64u * h + 32u * c; let ho = b + 128u + 32u * h;
      let el = 128u * h + 32u * c; let eh = el + 64u;`,
    word: `
      let qv = bytes4(word_at(qo + 4u * i)); let hv = bytes4(word_at(ho + 4u * i));
      let sl = select(la, lb, i >= 4u); let sh = select(ha, hb, i >= 4u);
      let lo = sl * (vec4<f32>((qv & vec4<u32>(15u)) | (((hv >> vec4<u32>(2u * c)) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);
      let hi = sh * (vec4<f32>((qv >> vec4<u32>(4u)) | (((hv >> vec4<u32>(4u + 2u * c)) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);`,
  },
};
export type FastKind = keyof typeof PAIR;
const FAST = new Set(Object.keys(PAIR));
export function fastKind(kind: PackedKind): FastKind {
  if (kind === "Q4_K" || kind === "Q5_K" || kind === "Q6_K" || kind === "Q8_0" || kind === "Q4_KR" || kind === "Q5_KR" || kind === "Q6_KR") return kind;
  throw new Error(`No vectorized decoder for ${kind}`);
}

const MATMUL_PARAMS = /* wgsl */ `
struct P { rows: u32, cols: u32, tokens: u32, row_bytes: u32,
           x_stride: u32, x_offset: u32, y_stride: u32, y_offset: u32,
           accumulate: u32, row_base: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${BYTES}`;

const store = (rowExpr: string, tokenExpr: string, value: string) => `{
    let i = (${tokenExpr}) * p.y_stride + p.y_offset + p.row_base + ${rowExpr};
    if (p.accumulate == 1u) { y[i] = y[i] + ${value}; } else { y[i] = ${value}; }
  }`;

// Few tokens (decode): MR rows per workgroup of 64 threads, MT threads per
// row, up to VT tokens; thread k of a row takes work items k, k+MT, ...
const VT = 4, MR = 4, MT = 64 / MR;
const matvecShaderSource = (kind: FastKind, VT: number) => /* wgsl */ `
${MATMUL_PARAMS}
var<workgroup> part: array<vec4<f32>, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${MT}u; let k = li.x % ${MT}u;
  let row = (wg.x + wg.z * ${MAX_DISPATCH}u) * ${MR}u + r;
  let t0 = wg.y * ${VT}u;
  let n = min(${VT}u, p.tokens - t0);
  var xb: array<u32, ${VT}>;
  for (var t = 0u; t < ${VT}u; t++) { xb[t] = ((t0 + min(t, n - 1u)) * p.x_stride + p.x_offset) / 4u; }
  var acc = vec4<f32>(0.0);
  if (row < p.rows) {
    let items = p.cols / 64u;
    for (var it = k; it < items; it += ${MT}u) {
      let sb = it / 4u; let g = it % 4u;
      let b = row * p.row_bytes + sb * ${PAIR[kind].block}u;
      ${PAIR[kind].setup}
      let xl = sb * 64u + el / 4u; let xh = sb * 64u + eh / 4u;
      for (var i = 0u; i < 8u; i++) {
        ${PAIR[kind].word}
        ${[...Array(VT).keys()].map((t) => `acc[${t}] += dot(lo, x[xb[${t}] + xl + i]) + dot(hi, x[xb[${t}] + xh + i]);`).join("\n        ")}
      }
    }
  }
  part[li.x] = acc;
  workgroupBarrier();
  for (var s = ${MT / 2}u; s > 0u; s >>= 1u) {
    if (k < s) { part[li.x] += part[li.x + s]; }
    workgroupBarrier();
  }
  if (row < p.rows && k < n) ${store("row", "t0 + k", `part[r * ${MT}u][k]`)}
}`;
const matvecShader = wgslMemo(matvecShaderSource);  // WS-33: one source string object per arguments

// WS-15: the matrix-vector kernel with R rows per thread. Thread k of lane r
// (MR lanes of MT threads) takes the same work items k, k+MT, ... as the
// kernel above, for rows r, r+MR, ..., r+(R-1)MR of its workgroup's MR*R
// rows, decoding each row's weights and loading the inputs of up to VT tokens
// once per word for all R rows. Each row's sums and reduction tree are the
// kernel above's, so the outputs are bit for bit the same; the inputs are
// read R times less.
const MV_IDS = ["b", "d", "dm", "s0", "s1", "sl", "ml", "sh", "mh", "qo", "ho", "el", "eh", "h", "c", "si", "la", "lb", "ha", "hb", "qv", "hv", "lo", "hi", "qa", "lv"];
const renameRow = (code: string, j: number) => code.replace(new RegExp(`\\b(${MV_IDS.join("|")})\\b`, "g"), `$1_${j}`);
const matvecRowsShaderSource = (kind: FastKind, VT: number, R: number) => {
  const Q = Math.ceil(VT / 4), J = [...Array(R).keys()], T = [...Array(VT).keys()], Qs = [...Array(Q).keys()];
  return /* wgsl */ `
${MATMUL_PARAMS}
var<workgroup> part: array<vec4<f32>, ${64 * R * Q}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${MT}u; let k = li.x % ${MT}u;
  let first = (wg.x + wg.z * ${MAX_DISPATCH}u) * ${MR * R}u + r;
  let t0 = wg.y * ${VT}u;
  let n = min(${VT}u, p.tokens - t0);
  var xb: array<u32, ${VT}>;
  for (var t = 0u; t < ${VT}u; t++) { xb[t] = ((t0 + min(t, n - 1u)) * p.x_stride + p.x_offset) / 4u; }
  ${J.map((j) => `let row_${j} = first + ${j * MR}u; let w_${j} = select(0u, row_${j}, row_${j} < p.rows);`).join("\n  ")}
  ${J.flatMap((j) => Qs.map((q) => `var acc_${j}_${q} = vec4<f32>(0.0);`)).join(" ")}
  let items = p.cols / 64u;
  for (var it = k; it < items; it += ${MT}u) {
    let sb = it / 4u; let g = it % 4u;
    ${J.map((j) => `let b_${j} = w_${j} * p.row_bytes + sb * ${PAIR[kind].block}u;${renameRow(PAIR[kind].setup, j)}`).join("\n    ")}
    let xl = sb * 64u + el_0 / 4u; let xh = sb * 64u + eh_0 / 4u;
    for (var i = 0u; i < 8u; i++) {
      ${T.map((t) => `let xa${t} = x[xb[${t}] + xl + i]; let xc${t} = x[xb[${t}] + xh + i];`).join("\n      ")}
      ${J.map((j) => `{${renameRow(PAIR[kind].word, j)}
        ${T.map((t) => `acc_${j}_${Math.floor(t / 4)}[${t % 4}] += dot(lo_${j}, xa${t}) + dot(hi_${j}, xc${t});`).join("\n        ")}
      }`).join("\n      ")}
    }
  }
  ${J.flatMap((j) => Qs.map((q) => `part[${(j * Q + q) * 64}u + li.x] = acc_${j}_${q};`)).join(" ")}
  workgroupBarrier();
  for (var s = ${MT / 2}u; s > 0u; s >>= 1u) {
    if (k < s) { ${J.flatMap((j) => Qs.map((q) => `part[${(j * Q + q) * 64}u + li.x] += part[${(j * Q + q) * 64}u + li.x + s];`)).join(" ")} }
    workgroupBarrier();
  }
  ${J.map((j) => `if (row_${j} < p.rows && k < n) ${store(`row_${j}`, "t0 + k", Q === 1 ? `part[${j * 64}u + r * ${MT}u][k]` : `select(part[${j * Q * 64}u + r * ${MT}u][k % 4u], part[${(j * Q + 1) * 64}u + r * ${MT}u][k % 4u], k >= 4u)`)}`).join("\n  ")}
}`;
};
const matvecRowsShader = wgslMemo(matvecRowsShaderSource);  // WS-33: one source string object per arguments

// WS-34: one decode token, coalesced: TPR threads per row (64/TPR rows per workgroup of 64), L threads per work
// item (a 64-element pair sharing 32 quant bytes), each taking 8/L consecutive words, so neighbouring threads
// read neighbouring bytes. The sums' order differs from the kernels above (not bit for bit; within tolerance).
// DO-74: W threads per workgroup (64, 128 or 256; W/TPR rows each). The reduction over a row's TPR threads is the
// same for every W, so W changes only the scheduling, never a sum.
const matvecCShaderSource = (kind: FastKind, TPR: number, L: number, W = 64) => /* wgsl */ `
${MATMUL_PARAMS}
var<workgroup> part: array<vec4<f32>, ${W}>;
@compute @workgroup_size(${W})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${TPR}u; let k = li.x % ${TPR}u;
  let row = (wg.x + wg.z * ${MAX_DISPATCH}u) * ${W / TPR}u + r;
  let xb = p.x_offset / 4u;
  var acc = vec4<f32>(0.0);
  if (row < p.rows) {
    let items = p.cols / 64u;
    let part_i = k % ${L}u;
    for (var it = k / ${L}u; it < items; it += ${TPR / L}u) {
      let sb = it / 4u; let g = it % 4u;
      let b = row * p.row_bytes + sb * ${PAIR[kind].block}u;
      ${PAIR[kind].setup}
      let xl = xb + sb * 64u + el / 4u; let xh = xb + sb * 64u + eh / 4u;
      for (var j = 0u; j < ${8 / L}u; j++) {
        let i = part_i * ${8 / L}u + j;
        ${PAIR[kind].word}
        acc += lo * x[xl + i] + hi * x[xh + i];
      }
    }
  }
  part[li.x] = acc;
  workgroupBarrier();
  for (var s = ${TPR / 2}u; s > 0u; s >>= 1u) {
    if (k < s) { part[li.x] += part[li.x + s]; }
    workgroupBarrier();
  }
  if (row < p.rows && k == 0u) ${store("row", "0u", "dot(part[li.x], vec4<f32>(1.0))")}
}`;
const matvecCShader = wgslMemo(matvecCShaderSource);  // WS-33: one source string object per arguments

// DO-69: several decode tokens (a batch of conversations), coalesced as matvecC: each workgroup decodes its rows'
// weights for T tokens at a time (passes of T over all tokens, the weights cached between passes). Every token's
// sums and reduction tree are matvecC's, so a batched token gets a single token's outputs.
const matvecBShaderSource = (kind: FastKind, TPR: number, L: number, T: number, W = 64) => {
  const Ts = [...Array(T).keys()];
  return /* wgsl */ `
${MATMUL_PARAMS}
var<workgroup> part: array<vec4<f32>, ${W * T}>;
@compute @workgroup_size(${W})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${TPR}u; let k = li.x % ${TPR}u;
  let row = (wg.x + wg.z * ${MAX_DISPATCH}u) * ${W / TPR}u + r;
  let wrow = select(0u, row, row < p.rows);
  let items = p.cols / 64u;
  let part_i = k % ${L}u;
  for (var t0 = 0u; t0 < p.tokens; t0 += ${T}u) {
    ${Ts.map((t) => `let xb${t} = ((t0 + min(${t}u, p.tokens - 1u - t0)) * p.x_stride + p.x_offset) / 4u; var acc${t} = vec4<f32>(0.0);`).join("\n    ")}
    for (var it = k / ${L}u; it < items; it += ${TPR / L}u) {
      let sb = it / 4u; let g = it % 4u;
      let b = wrow * p.row_bytes + sb * ${PAIR[kind].block}u;
      ${PAIR[kind].setup}
      let xo = sb * 64u;
      for (var j = 0u; j < ${8 / L}u; j++) {
        let i = part_i * ${8 / L}u + j;
        ${PAIR[kind].word}
        ${Ts.map((t) => `{ let xl = xb${t} + xo + el / 4u; let xh = xb${t} + xo + eh / 4u; acc${t} += lo * x[xl + i] + hi * x[xh + i]; }`).join("\n        ")}
      }
    }
    ${Ts.map((t) => `part[${t * W}u + li.x] = acc${t};`).join(" ")}
    workgroupBarrier();
    for (var s = ${TPR / 2}u; s > 0u; s >>= 1u) {
      if (k < s) { ${Ts.map((t) => `part[${t * W}u + li.x] += part[${t * W}u + li.x + s];`).join(" ")} }
      workgroupBarrier();
    }
    if (row < p.rows && k == 0u) {
      ${Ts.map((t) => `if (t0 + ${t}u < p.tokens) ${store("row", `t0 + ${t}u`, `dot(part[${t * W}u + li.x], vec4<f32>(1.0))`)}`).join("\n      ")}
    }
    workgroupBarrier();
  }
}`;
};
const matvecBShader = wgslMemo(matvecBShaderSource);  // WS-33: one source string object per arguments

// A few tokens (5-15): 16 rows x 16 tokens per workgroup; the inputs of
// one 256-column block for the 16 tokens are staged in workgroup memory
// (16 KiB), and thread (r, g) decodes pair g of row r's block once for all
// 16 tokens.
const TR = 16, TQ = 16;
const tiledShaderSource = (kind: FastKind) => /* wgsl */ `
${MATMUL_PARAMS}
var<workgroup> xs: array<vec4<f32>, ${TQ * 64}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / 4u; let g = li.x % 4u;
  let row = wg.x * ${TR}u + r;
  let t0 = wg.y * ${TQ}u;
  ${[...Array(TQ / 4).keys()].map((k) => `var a${k} = vec4<f32>(0.0);`).join(" ")}
  let blocks = p.cols / 256u;
  for (var sb = 0u; sb < blocks; sb++) {
    workgroupBarrier();
    for (var j = 0u; j < ${TQ}u; j++) {
      var v = vec4<f32>(0.0);
      if (t0 + j < p.tokens) { v = x[((t0 + j) * p.x_stride + p.x_offset) / 4u + sb * 64u + li.x]; }
      xs[j * 64u + li.x] = v;
    }
    workgroupBarrier();
    if (row < p.rows) {
      let b = row * p.row_bytes + sb * ${PAIR[kind].block}u;
      ${PAIR[kind].setup}
      for (var i = 0u; i < 8u; i++) {
        ${PAIR[kind].word}
        let cl = el / 4u + i; let ch = eh / 4u + i;
        ${[...Array(TQ).keys()].map((t) => `a${t >> 2}[${t & 3}] += dot(lo, xs[${t * 64}u + cl]) + dot(hi, xs[${t * 64}u + ch]);`).join("\n        ")}
      }
    }
  }
  workgroupBarrier();
  ${[...Array(TQ / 4).keys()].map((k) => `xs[li.x * 4u + ${k}u] = a${k};`).join(" ")}
  workgroupBarrier();
  let rr = li.x / 4u; let qq = li.x % 4u;  // row rr, tokens 4qq..4qq+3
  let orow = wg.x * ${TR}u + rr;
  let s = xs[(rr * 4u) * 4u + qq] + xs[(rr * 4u + 1u) * 4u + qq] + xs[(rr * 4u + 2u) * 4u + qq] + xs[(rr * 4u + 3u) * 4u + qq];
  if (orow < p.rows) {
    for (var k = 0u; k < 4u; k++) {
      let t = t0 + 4u * qq + k;
      if (t < p.tokens) ${store("orow", "t", "s[k]")}
    }
  }
}`;
const tiledShader = wgslMemo(tiledShaderSource);  // WS-33: one source string object per arguments

// Unit-wise decoding of the K-quants: unit j (elements 32j..32j+31) of a
// 256-element block at byte b; `setup` defines its scale(s) and byte offsets,
// `word` (given i in 0..7) decodes elements 32j+4i..+3 into `v`.
const UNIT = {
  Q5_KR: {
    block: 224,
    setup: `
      let s = bitcast<f32>(w[(b >> 2u) + 2u * j]); let m = bitcast<f32>(w[(b >> 2u) + 2u * j + 1u]);
      let qo = b + 96u + 32u * (j / 2u); let ho = b + 64u; let sh = vec4<u32>(4u * (j % 2u)); let hs = vec4<u32>(j);`,
    word: `let v = s * vec4<f32>(((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) | (((bytes4(word_at(ho + 4u * i)) >> hs) & vec4<u32>(1u)) << vec4<u32>(4u))) - m;`,
  },
  Q6_KR: {
    block: 256,
    setup: `
      let h = j / 4u;
      let s0 = bitcast<f32>(w[(b >> 2u) + 2u * j]); let s1 = bitcast<f32>(w[(b >> 2u) + 2u * j + 1u]);
      let qo = b + 64u + 64u * h + 32u * (j % 2u); let ho = b + 192u + 32u * h;
      let sh = vec4<u32>(((j % 4u) / 2u) * 4u); let hs = vec4<u32>((j % 4u) * 2u);`,
    word: `let v = select(s0, s1, i >= 4u) * (vec4<f32>(((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) | (((bytes4(word_at(ho + 4u * i)) >> hs) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);`,
  },
  Q4_KR: {
    block: 192,
    setup: `
      let s = bitcast<f32>(w[(b >> 2u) + 2u * j]); let m = bitcast<f32>(w[(b >> 2u) + 2u * j + 1u]);
      let qo = b + 64u + 32u * (j / 2u); let sh = vec4<u32>(4u * (j % 2u));`,
    word: `let v = s * vec4<f32>((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) - m;`,
  },
  Q4_K: {
    block: 144,
    setup: `
      let sm = scale_min(b + 4u, j); let s = half_at(b) * sm.x; let m = half_at(b + 2u) * sm.y;
      let qo = b + 16u + 32u * (j / 2u); let sh = vec4<u32>(4u * (j % 2u));`,
    word: `let v = s * vec4<f32>((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) - m;`,
  },
  Q5_K: {
    block: 176,
    setup: `
      let sm = scale_min(b + 4u, j); let s = half_at(b) * sm.x; let m = half_at(b + 2u) * sm.y;
      let qo = b + 48u + 32u * (j / 2u); let ho = b + 16u; let sh = vec4<u32>(4u * (j % 2u)); let hs = vec4<u32>(j);`,
    word: `let v = s * vec4<f32>(((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) | (((bytes4(word_at(ho + 4u * i)) >> hs) & vec4<u32>(1u)) << vec4<u32>(4u))) - m;`,
  },
  Q6_K: {
    block: 210,
    setup: `
      let h = j / 4u; let d = half_at(b + 208u);
      let s0 = d * sbyte_at(b + 192u + 2u * j); let s1 = d * sbyte_at(b + 193u + 2u * j);
      let qo = b + 64u * h + 32u * (j % 2u); let ho = b + 128u + 32u * h;
      let sh = vec4<u32>(((j % 4u) / 2u) * 4u); let hs = vec4<u32>((j % 4u) * 2u);`,
    word: `let v = select(s0, s1, i >= 4u) * (vec4<f32>(((bytes4(word_at(qo + 4u * i)) >> sh) & vec4<u32>(15u)) | (((bytes4(word_at(ho + 4u * i)) >> hs) & vec4<u32>(3u)) << vec4<u32>(4u))) - 32.0);`,
  },
};

// Many tokens (prefill), register-tiled: a workgroup computes GR rows x GT
// tokens, each thread RR rows x RT tokens. Per 32-column step the rows'
// decoded weights and the tokens' inputs are staged in workgroup memory
// (column-major, (GR + GT) x 32 floats: 16 KiB at 64 x 64, WebGPU's default
// limit); each thread then does RR x RT x 32 multiply-adds from
// (RR + RT) x 32 loads. Each 256-column block is summed on its own before
// it is added to the total, which keeps the error of 12,288-column rows at
// the level of the other kernels. On the 680M 64 x 64 with 4 x 8 per
// thread (128 threads) was fastest among 4x4, 4x8, 8x4 and 8x8 (and 32- or
// 64-wide tiles): 1.4-1.9 TFLOPS at 64 tokens, 2.6-4.5x the tiled kernel.
export const GEMM: GemmTile = { GR: 64, GT: 64, RR: 4, RT: 8 };
const GEMM_MIN = 16; // tokens from which the register-tiled kernel is used
// WS-50 `V`: the staged weights and inputs as vec4 arrays (four rows or four tokens per element), so the inner loop
// reads each operand vector in one load instead of four; every output's accumulation order and multiply-adds are
// the same, so outputs are bit for bit the scalar layout's.
const gemmShaderSource = (kind: keyof typeof UNIT, { GR, GT, RR, RT, V = false }: GemmTile) => {
  const N = (GR / RR) * (GT / RT), WPT = (GR * 8) / N, XPT = (GT * 8) / N;
  if (V && (RR % 4 || RT % 4 || GR % 4 || GT % 4)) throw new Error("vec4 GEMM staging needs multiples of 4");
  if (N < GR ? GR % N : 8 % WPT) throw new Error("GEMM tile: threads must divide the tile's rows or their decode words");
  if ((GT * 8) % N) throw new Error("GEMM tile: threads must divide the input staging");
  const wsDecl = V ? `var<workgroup> ws: array<vec4<f32>, ${8 * GR}>;` : `var<workgroup> ws: array<f32, ${32 * GR}>;`;
  const xsDecl = V ? `var<workgroup> xs: array<vec4<f32>, ${8 * GT}>;` : `var<workgroup> xs: array<f32, ${32 * GT}>;`;
  const wPut = (k: string, comp: string) => V ? `ws[(${k}) * ${GR / 4}u + dr / 4u][dr % 4u] = ${comp};` : `ws[(${k}) * ${GR}u + dr] = ${comp};`;
  const xPut = (k: string, comp: string) => V ? `xs[(${k}) * ${GT / 4}u + t / 4u][t % 4u] = ${comp};` : `xs[(${k}) * ${GT}u + t] = ${comp};`;
  const wGet = (g: number) => V ? `ws[k * ${GR / 4}u + tr * ${RR / 4}u + ${g}u]`
    : `vec4<f32>(ws[k * ${GR}u + tr * ${RR}u + ${4 * g}u], ws[k * ${GR}u + tr * ${RR}u + ${4 * g + 1}u], ws[k * ${GR}u + tr * ${RR}u + ${4 * g + 2}u], ws[k * ${GR}u + tr * ${RR}u + ${4 * g + 3}u])`;
  const xGet = (c: number) => V ? `xs[k * ${GT / 4}u + tc * ${RT / 4}u + ${c}u]`
    : `vec4<f32>(xs[k * ${GT}u + tc * ${RT}u + ${4 * c}u], xs[k * ${GT}u + tc * ${RT}u + ${4 * c + 1}u], xs[k * ${GT}u + tc * ${RT}u + ${4 * c + 2}u], xs[k * ${GT}u + tc * ${RT}u + ${4 * c + 3}u])`;
  const acc = [...Array(RR).keys()].flatMap((r) => [...Array(RT / 4).keys()].map((c) => `a${r}_${c}`));
  return /* wgsl */ `
${MATMUL_PARAMS}
${wsDecl}
${xsDecl}
@compute @workgroup_size(${N})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let tid = li.x;
  let r0 = wg.x * ${GR}u; let t0 = wg.y * ${GT}u;
  let tr = tid % ${GR / RR}u; let tc = tid / ${GR / RR}u;
  ${N >= GR ? `// decode: row dr of the tile, words di..di+WPT-1 of the step's unit
  let dr = tid / ${8 / WPT}u; let di = (tid % ${8 / WPT}u) * ${WPT}u;
  let drow = min(r0 + dr, p.rows - 1u);` : `// WS-50: fewer threads than tile rows: each thread decodes ${GR / N} whole rows (all 8 words)`}
  ${acc.map((a) => `var ${a} = vec4<f32>(0.0); var s${a} = vec4<f32>(0.0);`).join(" ")}
  let steps = p.cols / 32u;
  for (var st = 0u; st < steps; st++) {
    if (st % 8u == 0u) { ${acc.map((a) => `s${a} += ${a}; ${a} = vec4<f32>(0.0);`).join(" ")} }
    workgroupBarrier();
    ${N >= GR ? `{
      let b = drow * p.row_bytes + (st / 8u) * ${UNIT[kind].block}u; let j = st % 8u;
      ${UNIT[kind].setup}
      for (var w = 0u; w < ${WPT}u; w++) {
        let i = di + w;
        ${UNIT[kind].word}
        let k = 4u * i;
        ${wPut("k", "v.x")} ${wPut("k + 1u", "v.y")}
        ${wPut("k + 2u", "v.z")} ${wPut("k + 3u", "v.w")}
      }
    }` : `for (var m = 0u; m < ${GR / N}u; m++) {
      let dr = tid + m * ${N}u; let drow = min(r0 + dr, p.rows - 1u);
      let b = drow * p.row_bytes + (st / 8u) * ${UNIT[kind].block}u; let j = st % 8u;
      ${UNIT[kind].setup}
      for (var i = 0u; i < 8u; i++) {
        ${UNIT[kind].word}
        let k = 4u * i;
        ${wPut("k", "v.x")} ${wPut("k + 1u", "v.y")}
        ${wPut("k + 2u", "v.z")} ${wPut("k + 3u", "v.w")}
      }
    }`}
    for (var q = 0u; q < ${XPT}u; q++) {
      let e = tid + q * ${N}u; let t = e / 8u; let c = e % 8u;
      var v = vec4<f32>(0.0);
      if (t0 + t < p.tokens) { v = x[((t0 + t) * p.x_stride + p.x_offset) / 4u + st * 8u + c]; }
      ${xPut("4u * c", "v.x")} ${xPut("4u * c + 1u", "v.y")}
      ${xPut("4u * c + 2u", "v.z")} ${xPut("4u * c + 3u", "v.w")}
    }
    workgroupBarrier();
    for (var k = 0u; k < 32u; k++) {
      ${[...Array(RR / 4).keys()].map((g) => `let w${g} = ${wGet(g)};`).join("\n      ")}
      ${[...Array(RT / 4).keys()].map((c) => `let x${c} = ${xGet(c)};`).join("\n      ")}
      ${[...Array(RR).keys()].flatMap((r) => [...Array(RT / 4).keys()].map((c) => `a${r}_${c} += w${r >> 2}[${r & 3}] * x${c};`)).join("\n      ")}
    }
  }
  ${acc.map((a) => `s${a} += ${a};`).join(" ")}
  ${[...Array(RR).keys()].flatMap((r) => [...Array(RT / 4).keys()].map((c) => `{
    let row = r0 + tr * ${RR}u + ${r}u;
    if (row < p.rows) {
      for (var q = 0u; q < 4u; q++) {
        let t = t0 + tc * ${RT}u + ${4 * c}u + q;
        if (t < p.tokens) ${store("row", "t", `sa${r}_${c}[q]`)}
      }
    }
  }`)).join("\n  ")}
}`;
};
const gemmShader = wgslMemo(gemmShaderSource);  // WS-33: one source string object per arguments

// Gather embedding rows for the tokens whose id falls in this chunk of rows.
const gatherShaderSource = (kind: PackedKind) => /* wgsl */ `
struct P { first: u32, count: u32, cols: u32, row_bytes: u32, tokens: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${BYTES}
${DEQ[kind]}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x;
  let id = ids[t];
  if (id < p.first || id >= p.first + p.count) { return; }
  let base = (id - p.first) * p.row_bytes;
  for (var e = li.x; e < p.cols; e += 256u) { y[t * p.cols + e] = deq(base, e); }
}`;
const gatherShader = wgslMemo(gatherShaderSource);  // WS-33: one source string object per arguments

// RMSNorm over groups of `size` values: out = x * rsqrt(mean(x^2) + eps) * w
// [* silu(gate)]. Offsets are in floats.
const NORM = /* wgsl */ `
struct P { size: u32, groups: u32, in_stride: u32, in_group: u32,
           out_stride: u32, out_group: u32, gate_stride: u32, gate_group: u32,
           in_offset: u32, out_offset: u32, gate_offset: u32, use_gate: u32,
           eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let t = wg.x / p.groups; let g = wg.x % p.groups;
  let xi = t * p.in_stride + g * p.in_group + p.in_offset;
  var s = 0.0;
  for (var e = li.x; e < p.size; e += 256u) { let v = x[xi + e]; s += v * v; }
  part[li.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li.x < k) { part[li.x] += part[li.x + k]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(part[0] / f32(p.size) + p.eps);
  let yi = t * p.out_stride + g * p.out_group + p.out_offset;
  let gi = t * p.gate_stride + g * p.gate_group + p.gate_offset;
  for (var e = li.x; e < p.size; e += 256u) {
    var v = x[xi + e] * r * wt[e];
    if (p.use_gate == 1u) { let z = gate[gi + e]; v = v * (z / (1.0 + exp(-z))); }
    y[yi + e] = v;
  }
}`;

// DO-65 fusion: a sandwich norm and its residual add in one dispatch: y = RMSNorm(x) * w (still written, as the
// separate norm did) and res += y, for one group with contiguous rows (the shape the plan executor checks).
const NORM_ADD = NORM
  .replace("@group(0) @binding(4) var<uniform> p: P;",
           "@group(0) @binding(4) var<uniform> p: P;\n@group(0) @binding(5) var<storage, read_write> res: array<f32>;")
  .replace("    y[yi + e] = v;\n", "    y[yi + e] = v;\n    res[yi + e] = res[yi + e] + v;\n");
if (!NORM_ADD.includes("binding(5)") || !NORM_ADD.includes("res[yi + e] = res[yi + e] + v;")) throw new Error("NORM_ADD derivation failed");

// DO-65 fusion: two independent grouped RMS norms of the same group size in one dispatch (QK-norm: the query
// heads and the key heads); workgroups below tokens * groups_a do pair a, the rest pair b. Contiguous rows.
const NORM_PAIR = /* wgsl */ `
struct P { size: u32, ga: u32, gb: u32, tokens: u32, eps: f32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> xa: array<f32>;
@group(0) @binding(1) var<storage, read> wa: array<f32>;
@group(0) @binding(2) var<storage, read_write> ya: array<f32>;
@group(0) @binding(3) var<storage, read> xb: array<f32>;
@group(0) @binding(4) var<storage, read> wb: array<f32>;
@group(0) @binding(5) var<storage, read_write> yb: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let second = wg.x >= p.tokens * p.ga;
  let w = select(wg.x, wg.x - p.tokens * p.ga, second);
  let groups = select(p.ga, p.gb, second);
  let t = w / groups; let g = w % groups;
  let base = t * p.size * groups + g * p.size;
  var s = 0.0;
  for (var e = li.x; e < p.size; e += 256u) { let v = select(xa[base + e], xb[base + e], second); s += v * v; }
  part[li.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li.x < k) { part[li.x] += part[li.x + k]; }
    workgroupBarrier();
  }
  let r = inverseSqrt(part[0] / f32(p.size) + p.eps);
  for (var e = li.x; e < p.size; e += 256u) {
    if (second) { yb[base + e] = xb[base + e] * r * wb[e]; } else { ya[base + e] = xa[base + e] * r * wa[e]; }
  }
}`;

// Depthwise causal convolution (kernel 4) with SiLU; state holds the last 3
// inputs per channel ([3][channels], oldest first).
export const F16 = /* wgsl */ `
fn f16_bits(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  let sign = (b >> 16u) & 0x8000u;
  let a = b & 0x7fffffffu;
  if (a >= 0x7f800000u) { return sign | 0x7c00u | select(0u, 0x200u, a > 0x7f800000u); }
  if (a >= 0x477ff000u) { return sign | 0x7c00u; }
  if (a < 0x38800000u) {
    if (a < 0x33000000u) { return sign; }
    let shift = 126u - (a >> 23u);
    let m = (a & 0x7fffffu) | 0x800000u;
    let q = m >> shift; let rem = m & ((1u << shift) - 1u); let half = 1u << (shift - 1u);
    return sign | (q + select(0u, 1u, rem > half || (rem == half && (q & 1u) == 1u)));
  }
  return sign | ((a + 0xfffu + ((a >> 13u) & 1u) - 0x38000000u) >> 13u);
}
fn f16_value(h: u32) -> f32 {
  let s = (h & 0x8000u) << 16u; let e = (h >> 10u) & 0x1fu; let m = h & 0x3ffu;
  if (e == 0u) { return bitcast<f32>(s | bitcast<u32>(f32(m) * 5.9604644775390625e-8)); }
  if (e == 31u) { return bitcast<f32>(s | 0x7f800000u | (m << 13u)); }
  return bitcast<f32>(s | ((e + 112u) << 23u) | (m << 13u));
}
fn f16x2(u: u32) -> vec2<f32> { return vec2<f32>(f16_value(u & 0xffffu), f16_value(u >> 16u)); }`;

const SILU_MUL = /* wgsl */ `
struct P { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * ${MAX_DISPATCH * 256}u;
  if (i >= p.n) { return; }
  let v = a[i];
  y[i] = v / (1.0 + exp(-v)) * y[i];
}`;

// Greedy token: the first index of the largest logit (as NumPy's argmax).
const ARGMAX = /* wgsl */ `
struct P { n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
  var v = -3.0e38; var i = 0xffffffffu;
  for (var e = li.x; e < p.n; e += 256u) {
    if (x[e] > v) { v = x[e]; i = e; }
  }
  bv[li.x] = v; bi[li.x] = i;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (li.x < s) {
      let ov = bv[li.x + s]; let oi = bi[li.x + s];
      if (ov > bv[li.x] || (ov == bv[li.x] && oi < bi[li.x])) { bv[li.x] = ov; bi[li.x] = oi; }
    }
    workgroupBarrier();
  }
  if (li.x == 0u) { out[0] = bi[0]; }
}`;

// WS-36: one decode token through several matrices that read the same input, in one dispatch: segment s (its own
// weight binding w<s>, decoder and output) takes consecutive workgroups, each row exactly as matvecC computes it
// (same loop, decoder and reduction), so every output is matvecC's bit for bit. `glu`: two products (gate, up) of
// equal rows and format: each thread computes both rows' sums (each in matvecC's order), and the up output becomes
// silu(gate) * up as silu_mul would write it (the gate output is still written).
const SEG_FNS = /\b(byte_at|word_at|half_at|sbyte_at|scale_min|bytes4)\b/g;
const segment = (code: string, s: number) => code.replace(SEG_FNS, `$1_${s}`).replace(/\bw\[/g, `w${s}[`);
// WS-49 `norm`: the input is RMSNorm(src) * nw computed here: each workgroup repeats the norm kernel's sum of squares
// with its exact order (256 lanes striding the row, then the same tree), normalizes each input element as the norm
// kernel would ((x * r) * w), and workgroup 0 also writes the normalized row to `h` (the norm's own output).
// DO-76 `R`: each row slot computes R consecutive rows from the same input loads (each row's sum in matvecC's order,
// so still bit for bit); p.n bit 0: add into the output (a residual) instead of writing it.
const matvecGroupShaderSource = (kinds: FastKind[], outs: number[], TPR: number, L: number, W: number, glu: boolean, norm = false, R = 1) => {
  const n = kinds.length, nY = Math.max(...outs) + 1, RPW = W / TPR, S = [...Array(n).keys()], Q = [...Array(R).keys()];
  const X = (i: string) => norm ? `(x[${i}] * rr * nw4[${i}])` : `x[${i}]`;  // with norm, binding x is the norm's input
  const B = n + 1 + nY;  // first binding after the outputs
  const body = (s: number, acc: string, q: number) => `if (row0 + ${q}u < p.seg[${s}].x) {
        let b = (row0 + ${q}u) * p.seg[${s}].y + sb * ${PAIR[kinds[s]].block}u;
        ${segment(PAIR[kinds[s]].setup, s)}
        let xl = xb + sb * 64u + el / 4u; let xh = xb + sb * 64u + eh / 4u;
        for (var j = 0u; j < ${8 / L}u; j++) {
          let i = part_i * ${8 / L}u + j;
          ${segment(PAIR[kinds[s]].word, s)}
          ${acc}${q} += lo * ${X("xl + i")} + hi * ${X("xh + i")};
        }
      }`;
  const reduce = (arr: string) => `for (var t = ${TPR / 2}u; t > 0u; t >>= 1u) {
    if (k < t) { ${Q.map((q) => `${arr}[${q * W}u + li.x] += ${arr}[${q * W}u + li.x + t];`).join(" ")} }
    workgroupBarrier();
  }`;
  const put = (o: number, s: number, row: string, v: string) => `{ let yi = p.seg[${s}].z + ${row};
      if ((p.n & 1u) == 1u) { y${o}[yi] = y${o}[yi] + ${v}; } else { y${o}[yi] = ${v}; } }`;
  return /* wgsl */ `
struct P { cols: u32, x_offset: u32, n: u32, p0: u32, seg: array<vec4<u32>, 4> };  // seg: rows, row_bytes, y_offset, first workgroup
${S.map((s) => `@group(0) @binding(${s}) var<storage, read> w${s}: array<u32>;`).join("\n")}
@group(0) @binding(${n}) var<storage, read> x: array<vec4<f32>>;
${[...Array(nY).keys()].map((o) => `@group(0) @binding(${n + 1 + o}) var<storage, read_write> y${o}: array<f32>;`).join("\n")}
@group(0) @binding(${B}) var<uniform> p: P;
${norm ? `@group(0) @binding(${B + 1}) var<storage, read> nw4: array<vec4<f32>>;
@group(0) @binding(${B + 2}) var<storage, read_write> h: array<f32>;
var<workgroup> npart: array<f32, 256>;
var<private> rr: f32;` : ""}
${S.map((s) => segment(BYTES, s)).join("\n")}
var<workgroup> part: array<vec4<f32>, ${W * R}>;
${glu ? `var<workgroup> part1: array<vec4<f32>, ${W * R}>;` : ""}
@compute @workgroup_size(${W})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${TPR}u; let k = li.x % ${TPR}u;
  let gi = wg.x + wg.z * ${MAX_DISPATCH}u;
  ${glu || n === 1 ? `let s = 0u;
  let row0 = (gi * ${RPW}u + r) * ${R}u;` : `var s = 0u;
  ${S.slice(1).map((t) => `if (gi >= p.seg[${t}].w) { s = ${t}u; }`).join("\n  ")}
  let row0 = ((gi - p.seg[s].w) * ${RPW}u + r) * ${R}u;`}
  let xb = p.x_offset / 4u;
  ${norm ? `for (var q = li.x; q < 256u; q += ${W}u) {
    var ss = 0.0;
    for (var e = q; e < p.cols; e += 256u) { let v = x[e >> 2u][e & 3u]; ss += v * v; }
    npart[q] = ss;
  }
  workgroupBarrier();
  for (var kk = 128u; kk > 0u; kk >>= 1u) {
    for (var q = li.x; q < kk; q += ${W}u) { npart[q] += npart[q + kk]; }
    workgroupBarrier();
  }
  rr = inverseSqrt(npart[0] / f32(p.cols) + bitcast<f32>(p.p0));
  if (gi == 0u) {
    for (var e = li.x; e < p.cols; e += ${W}u) { h[e] = x[e >> 2u][e & 3u] * rr * nw4[e >> 2u][e & 3u]; }
  }` : ""}
  ${Q.map((q) => `var acc${q} = vec4<f32>(0.0);${glu ? ` var accu${q} = vec4<f32>(0.0);` : ""}`).join("\n  ")}
  if (row0 < p.seg[s].x) {
    let items = p.cols / 64u;
    let part_i = k % ${L}u;
    for (var it = k / ${L}u; it < items; it += ${TPR / L}u) {
      let sb = it / 4u; let g = it % 4u;
      ${Q.map((q) => glu ? `${body(0, "acc", q)}\n      ${body(1, "accu", q)}`
        : n === 1 ? body(0, "acc", q) : S.map((t) => `if (s == ${t}u) { ${body(t, "acc", q)} }`).join(" else ")).join("\n      ")}
    }
  }
  ${Q.map((q) => `part[${q * W}u + li.x] = acc${q};${glu ? ` part1[${q * W}u + li.x] = accu${q};` : ""}`).join("\n  ")}
  workgroupBarrier();
  ${reduce("part")}
  ${glu ? reduce("part1") : ""}
  if (k == 0u) {
    ${Q.map((q) => `if (row0 + ${q}u < p.seg[s].x) {
      let v = dot(part[${q * W}u + li.x], vec4<f32>(1.0));
      ${glu ? `y${outs[0]}[p.seg[0].z + row0 + ${q}u] = v;
      let u = dot(part1[${q * W}u + li.x], vec4<f32>(1.0));
      y${outs[1]}[p.seg[1].z + row0 + ${q}u] = v / (1.0 + exp(-v)) * u;`
      : S.map((t) => `if (s == ${t}u) ${put(outs[t], t, `row0 + ${q}u`, "v")}`).join(" else ")}
    }`).join("\n    ")}
  }
}`;
};
export const matvecGroupShader = wgslMemo(matvecGroupShaderSource);  // WS-33: one source string object per arguments

// WS-40: Q4_K -> Q4_KR, exactly: per block the 8 sub-block scales d * sc and mins dm * m as float32 (an f16 times a
// 6-bit integer is exact in a double, so Math.fround gives the same float32 as the GPU's multiply), then the 128
// quant bytes. Decoding then needs no scale unpacking; every product is the same, so outputs are bit for bit.
export function repackQ4K(data: ArrayBuffer | ArrayBufferView, rows: number, cols: number): Uint8Array<ArrayBuffer> {
  const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const blocks = rows * (cols / 256), out = new Uint8Array(new ArrayBuffer(blocks * 192)), f = new Float32Array(out.buffer);
  const half = (lo: number, hi: number) => {
    const h = lo | (hi << 8), e = (h >> 10) & 31, m = h & 1023, sign = h & 32768 ? -1 : 1;
    return sign * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1024 + m) * 2 ** (e - 25));
  };
  for (let k = 0; k < blocks; k++) {
    const b = k * 144, o = k * 192, d = half(src[b], src[b + 1]), dm = half(src[b + 2], src[b + 3]);
    for (let j = 0; j < 8; j++) {
      const q = b + 4;  // 12 scale bytes, as scale_min reads them
      const sc = j < 4 ? src[q + j] & 63 : (src[q + j + 4] & 15) | ((src[q + j - 4] >> 6) << 4);
      const mn = j < 4 ? src[q + j + 4] & 63 : (src[q + j + 4] >> 4) | ((src[q + j] >> 6) << 4);
      f[o / 4 + 2 * j] = Math.fround(d * sc); f[o / 4 + 2 * j + 1] = Math.fround(dm * mn);
    }
    out.set(src.subarray(b + 16, b + 144), o + 64);
  }
  return out;
}

const halfValue = (lo: number, hi: number) => {
  const h = lo | (hi << 8), e = (h >> 10) & 31, m = h & 1023, sign = h & 32768 ? -1 : 1;
  return sign * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1024 + m) * 2 ** (e - 25));
};
// DO-80: Q5_K -> Q5_KR exactly (as repackQ4K: scales and mins as float32, then qh and qs unchanged).
export function repackQ5K(data: ArrayBuffer | ArrayBufferView, rows: number, cols: number): Uint8Array<ArrayBuffer> {
  const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const blocks = rows * (cols / 256), out = new Uint8Array(new ArrayBuffer(blocks * 224)), f = new Float32Array(out.buffer);
  for (let k = 0; k < blocks; k++) {
    const b = k * 176, o = k * 224, d = halfValue(src[b], src[b + 1]), dm = halfValue(src[b + 2], src[b + 3]), q = b + 4;
    for (let j = 0; j < 8; j++) {
      const sc = j < 4 ? src[q + j] & 63 : (src[q + j + 4] & 15) | ((src[q + j - 4] >> 6) << 4);
      const mn = j < 4 ? src[q + j + 4] & 63 : (src[q + j + 4] >> 4) | ((src[q + j] >> 6) << 4);
      f[o / 4 + 2 * j] = Math.fround(d * sc); f[o / 4 + 2 * j + 1] = Math.fround(dm * mn);
    }
    out.set(src.subarray(b + 16, b + 176), o + 64);  // qh (32) then qs (128)
  }
  return out;
}
// DO-80: Q6_K -> Q6_KR exactly: the 16 scales d * sc as float32 (an f16 times an int8 is exact in float32), then ql
// and qh unchanged; 256 bytes, so every block is 4-byte aligned (Q6_K's 210-byte blocks are not).
export function repackQ6K(data: ArrayBuffer | ArrayBufferView, rows: number, cols: number): Uint8Array<ArrayBuffer> {
  const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const blocks = rows * (cols / 256), out = new Uint8Array(new ArrayBuffer(blocks * 256)), f = new Float32Array(out.buffer);
  for (let k = 0; k < blocks; k++) {
    const b = k * 210, o = k * 256, d = halfValue(src[b + 208], src[b + 209]);
    for (let j = 0; j < 16; j++) { const sc = src[b + 192 + j]; f[o / 4 + j] = Math.fround(d * (sc > 127 ? sc - 256 : sc)); }
    out.set(src.subarray(b, b + 192), o + 64);  // ql (128) then qh (64)
  }
  return out;
}
// Repacked kind, block size and repack function per K-quant (experiment 141, DO-80).
export const REPACK: Record<string, { kind: PackedKind; block: number; from: number; fn: (d: ArrayBuffer | ArrayBufferView, r: number, c: number) => Uint8Array<ArrayBuffer> }> = {
  Q4_K: { kind: "Q4_KR", block: 192, from: 144, fn: repackQ4K },
  Q5_K: { kind: "Q5_KR", block: 224, from: 176, fn: repackQ5K },
  Q6_K: { kind: "Q6_KR", block: 256, from: 210, fn: repackQ6K },
};

// DO-83: whether the exact repack (experiments 141/142) pays on this GPU. It is faster where the one-token products
// are decoding-bound (A100: -23%) and slower where they are memory-bound (Radeon 680M: +25%), so each contributor
// measures it: per K-quant format, a synthetic 8,192 x 4,096 matrix in enough copies to exceed `mb` MiB (the L2
// cannot serve them, experiment 134), `dispatches` one-token products per timing cycling over the copies, raw and
// repacked alternating for `rounds` rounds; a format is repacked when its median time is at most `ratio` of raw.
// Outputs are bit for bit the same either way, so the choice changes speed only. Run after the kernel options.
export async function calibrateRepack(gpu: Gpu, { mb = 64, rounds = 5, dispatches = 60, ratio = 0.9, kinds = ["Q4_K", "Q5_K", "Q6_K"] } = {}) {
  const rows = 8192, cols = 4096, out: { repack: string[]; ms: Record<string, { raw: number; packed: number }> } = { repack: [], ms: {} };
  const x = gpu.upload(new Float32Array(cols).fill(0.01)), y = gpu.buffer(rows * 4);
  for (const kind of kinds) {
    const r = REPACK[kind], rawBytes = rows * (cols / 256) * r.from;
    const raw = new Uint8Array(rawBytes);
    let seed = 2026;
    for (let i = 0; i < rawBytes; i++) { seed = (seed * 1103515245 + 12345) >>> 0; raw[i] = seed >>> 24; }
    const scale = kind === "Q6_K" ? [208] : [0, 2];  // f16 scale fields: 1.0, so the products stay finite
    for (let b = 0; b < rawBytes; b += r.from) for (const o of scale) { raw[b + o] = 0x00; raw[b + o + 1] = 0x3c; }
    const packedBytes = r.fn(raw, rows, cols);
    const copies = Math.max(2, Math.ceil((mb * 2 ** 20) / rawBytes));
    const make = (bytes: Uint8Array<ArrayBuffer>, k: PackedKind, block: number) =>
      Array.from({ length: copies }, () => ({ kind: k, rows, cols, rowBytes: (cols / 256) * block, buffer: gpu.upload(bytes) }));
    const arms = { raw: make(raw, kind as PackedKind, r.from), packed: make(packedBytes, r.kind, r.block) };
    const times: Record<"raw" | "packed", number[]> = { raw: [], packed: [] };
    for (let k = 0; k < rounds; k++) {
      for (const arm of (k % 2 ? ["packed", "raw"] : ["raw", "packed"]) as ("raw" | "packed")[]) {
        gpu.begin();
        for (let d = 0; d < dispatches; d++) gpu.matmul(arms[arm][d % copies], x, y, 1);
        const t0 = performance.now();
        await gpu.end();
        times[arm].push((performance.now() - t0) / dispatches);
      }
    }
    for (const W of [...arms.raw, ...arms.packed]) gpu.release(W.buffer);
    const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)];
    out.ms[kind] = { raw: median(times.raw), packed: median(times.packed) };
    if (out.ms[kind].packed <= ratio * out.ms[kind].raw) out.repack.push(kind);
  }
  gpu.release(x); gpu.release(y);
  return out;
}

export type MatvecC = { tpr: number; lanes: number; wg?: number; rows?: number };  // DO-76: rows per thread (default 1)

export class Gpu {
  readonly device: GPUDevice;
  readonly pipelines = new Map<string, GPUComputePipeline>();
  readonly live = new Set<GPUBuffer>();
  allocated = 0;
  peak = 0;
  adapter?: GPUAdapter;
  lost?: string;
  errors: string[] = [];
  transient: GPUBuffer[] = [];
  encoder?: GPUCommandEncoder;
  pass?: GPUComputePassEncoder;
  generic = false;
  fastQ8 = false;  // WS-46 (experiment 130, opt-in): Q8_0 through the vectorized decoders instead of the generic kernel
  q8FastUpTo = 0;  // WS-47: Q8_0 through the vectorized decoders for calls of at most this many tokens (decode)
  prefill?: "tiled" | "gemm" | "matvec";
  gemm?: GemmTile;
  // WS-15: the matrix-vector kernel with `rows` rows per thread and up to `tokens` (4 or 8) tokens per
  // dispatch row; `single`: rows per thread for one token (WS-16: 2, with 4 for more tokens). Unset: the
  // kernel above. Every choice gives the same outputs bit for bit (experiment 104).
  matvec?: { rows: number; tokens: number; single?: number };
  // DO-74: wg = threads per workgroup (default 64); kinds = per-format overrides, e.g. { Q4_K: { tpr: 8, lanes: 2, wg: 256 } }
  matvecC?: MatvecC & { kinds?: Partial<Record<string, MatvecC>> };  // WS-34 diagnostic: the coalesced one-token kernel
  batchKernel?: "tiled" | "gemm";
  batchGemm?: { from: number; tile: GemmTile };  // experiment 127: batches of `from` or more through the GEMM with this tile  // WS-45: decode batches through a many-token kernel instead of the matrix-vector ones
  repackQ4K = false;  // experiment 141 (opt-in): layer Q4_K matrices repacked at load into Q4_KR (exact)
  repack: string[] = [];  // DO-80 (opt-in): K-quant formats repacked at load (layer matrices and the head), e.g. ["Q4_K", "Q5_K", "Q6_K"]
  normFold = false;  // WS-49 (opt-in): a plain one-token norm folded into the grouped product that reads it
  groupMatmul = false;  // WS-36 (opt-in): consecutive one-token products of the same input in one dispatch
  matvecB?: number;  // DO-69: decode batches through the coalesced kernel, this many tokens per pass (needs matvecC)
  // Profiling (off by default): per submission, the JavaScript time from begin() to submit, the GPU's own
  // time for the pass (timestamp queries, when the device has them), the wall time to completion and the
  // dispatches by name; takeProfile() returns and clears the records.
  // "detail": every dispatch in its own timestamped pass, GPU time summed by name (gpuByName); slower, for
  // finding which operations cost what.
  profiling: boolean | "detail" = false;
  pendingExpandMs?: number;  // executePlan: the last plan expansion's JavaScript time, put on the next record
  // Host-side caches (DO-65): checked plan tasks (inference_plan.mts) and bind groups by pipeline and entries
  // (with the uniform pool a dispatch binds the same buffers at the same offsets every token). false: neither.
  hostCache = true;
  // DO-99: cheaper host path per dispatch (bind groups reused per dispatch slot, uniforms written in place,
  // repeated setPipeline skipped). false: the previous path.
  fastHost = true;
  private slot = 0;
  private slots: ({ pipeline: GPUComputePipeline; refs: unknown[]; bind: GPUBindGroup } | undefined)[] = [];
  private lastPipeline?: GPUComputePipeline;
  private uniformU32?: Uint32Array;
  private uniformF32?: Float32Array;
  // WS-32: end() submits and returns without waiting for the GPU (no profiling), so the host encodes the next
  // step while this one runs; queue order keeps every write and copy after the work submitted before it.
  defer = false;
  // DO-73: host time spent in queue.submit, onSubmittedWorkDone and mapAsync (ms, summed until reset).
  host = { submit: 0, wait: 0, map: 0, submits: 0, maps: 0 };
  private hostAt() { return performance.now(); }
  fuse = true;  // DO-65: the plan executor may fuse a norm and its residual add (normAdd); false: never
  private bindGroups = new Map<string, GPUBindGroup>();
  private ids = new WeakMap<object, number>();
  private nextId = 1;
  private id(o: object) { let v = this.ids.get(o); if (v === undefined) { v = this.nextId++; this.ids.set(o, v); } return v; }
  private detailNames: string[] = [];
  private detailBytes: number[] = [];  // WS-31: each detail dispatch's bound bytes (sum of its bindings' sizes)
  private detailSet?: GPUQuerySet;
  private detailBuffer?: GPUBuffer;
  static readonly DETAIL_MAX = 2048;  // 2 timestamps each; devices allow 4096 queries per set
  lastSequence: [string, number][] = [];  // detail: the last submission's dispatches in order, with GPU ms
  profileLog: ProfileRecord[] = [];
  private profileStart = 0;
  private profileNames = new Map<string, number>();
  private querySet?: GPUQuerySet;
  private queryBuffer?: GPUBuffer;
  // Uniform pool: between begin() and end(), uniforms are 256-byte slots of one buffer, written once at
  // submit, instead of a new buffer and a write per dispatch.
  private uniformPool?: GPUBuffer;
  private uniformStaging = new ArrayBuffer(0);
  private uniformUsed = 0;
  private uniformLimit = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.allocated = 0;
    this.peak = 0;
  }

  static async create<T extends Gpu>(
    this: new (device: GPUDevice) => T,
  ): Promise<T> {
    if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available");
    const limits: Record<string, number> = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize", "maxStorageBuffersPerShaderStage"] as const) {
      limits[key] = adapter.limits[key];
    }
    const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    const device = await adapter.requestDevice({ requiredLimits: limits, requiredFeatures });
    const gpu = new this(device);
    gpu.adapter = adapter;
    device.lost.then((info) => { gpu.lost = info.message || info.reason; });
    gpu.errors = [];
    device.addEventListener("uncapturederror", (event) => gpu.errors.push(event.error.message));
    return gpu;
  }

  buffer(bytes: number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const size = Math.max(16, Math.ceil(bytes / 16) * 16);
    const buffer = this.device.createBuffer({ size, usage });
    this.allocated += size;
    this.peak = Math.max(this.peak, this.allocated);
    this.live.add(buffer);
    return buffer;
  }

  release(buffer: GPUBuffer) {
    if (this.live.delete(buffer)) this.allocated -= buffer.size;
    buffer.destroy();
  }

  upload(data: ArrayBuffer | ArrayBufferView<ArrayBuffer>) {
    const bytes = data.byteLength;
    const buffer = this.buffer(bytes);
    const padded = Math.ceil(bytes / 4) * 4;
    if (padded === bytes) {
      this.device.queue.writeBuffer(buffer, 0, data);
    } else {
      const copy = new Uint8Array(padded);
      copy.set(ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, bytes)
        : new Uint8Array(data));
      this.device.queue.writeBuffer(buffer, 0, copy);
    }
    return buffer;
  }

  pipeline(name: string, code: string): GPUComputePipeline {
    // Names are diagnostic labels, not shader identities. Two model adapters
    // may use the same operation name with different dimensions or layouts.
    if (!this.pipelines.has(code)) {
      const module = this.device.createShaderModule({ label: name, code });
      this.pipelines.set(code, this.device.createComputePipeline({
        label: name,
        layout: "auto", compute: { module, entryPoint: "main" },
      }));
    }
    return this.pipelines.get(code)!;
  }

  uniform(values: readonly UniformValue[]) {
    // values: array of [type, value]; all 4-byte, padded to 16 bytes
    const n = Math.ceil(values.length / 4) * 4;
    if (this.fastHost && this.pass && this.uniformPool && this.uniformUsed + 256 <= this.uniformLimit && n * 4 <= 256) {
      // DO-99: straight into the staging copy, without a per-call ArrayBuffer
      const offset = this.uniformUsed, at = offset >> 2, u = this.uniformU32!, f = this.uniformF32!;
      for (let i = 0; i < n; i++) {
        const v = values[i];
        if (v === undefined) u[at + i] = 0; else if (v[0] === "f") f[at + i] = v[1]; else u[at + i] = v[1];
      }
      this.uniformUsed += 256;
      return { buffer: this.uniformPool, offset, size: n * 4 } as GPUBufferBinding;
    }
    const data = new ArrayBuffer(n * 4);
    const u = new Uint32Array(data), f = new Float32Array(data);
    values.forEach(([type, value], i) => { if (type === "f") f[i] = value; else u[i] = value; });
    if (this.pass && this.uniformPool && this.uniformUsed + 256 <= this.uniformLimit) {
      const offset = this.uniformUsed;
      new Uint8Array(this.uniformStaging, offset, n * 4).set(new Uint8Array(data));
      this.uniformUsed += Math.max(256, Math.ceil((n * 4) / 256) * 256);
      return { buffer: this.uniformPool, offset, size: n * 4 } as GPUBufferBinding;
    }
    const buffer = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data);
    this.transient.push(buffer);
    return buffer;
  }

  begin() {
    if (this.profiling) { this.profileStart = performance.now(); this.profileNames.clear(); }
    if (!this.uniformPool) {
      // 4096 uniforms of up to 256 bytes per submission; more fall back to their own buffers
      this.uniformLimit = 4096 * 256;
      this.uniformPool = this.device.createBuffer({ size: this.uniformLimit, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.uniformStaging = new ArrayBuffer(this.uniformLimit);
      this.uniformU32 = new Uint32Array(this.uniformStaging); this.uniformF32 = new Float32Array(this.uniformStaging);
    }
    this.slot = 0; this.lastPipeline = undefined;
    this.uniformUsed = 0;
    this.encoder = this.device.createCommandEncoder();
    const timed = this.profiling && this.device.features.has("timestamp-query");
    this.detailNames = [];
    this.detailBytes = [];
    if (timed && this.profiling === "detail" && !this.detailSet) {
      this.detailSet = this.device.createQuerySet({ type: "timestamp", count: 2 * Gpu.DETAIL_MAX });
      this.detailBuffer = this.device.createBuffer({ size: 16 * Gpu.DETAIL_MAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    if (timed && !this.querySet) {
      this.querySet = this.device.createQuerySet({ type: "timestamp", count: 2 });
      this.queryBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    }
    this.pass = this.encoder.beginComputePass(timed && this.profiling !== "detail"
      ? { timestampWrites: { querySet: this.querySet!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    this.transient = this.transient || [];
  }

  takeProfile(): ProfileRecord[] {
    const out = this.profileLog;
    this.profileLog = [];
    return out;
  }

  dispatch(name: string, code: string, buffers: readonly BufferBinding[], groups: Workgroups) {
    if (!this.pass) throw new Error("Call begin() before dispatch()");
    if (this.profiling) this.profileNames.set(name, (this.profileNames.get(name) || 0) + 1);
    if (this.profiling === "detail" && this.detailSet && this.detailNames.length < Gpu.DETAIL_MAX) {
      const k = this.detailNames.length;
      this.pass.end();
      this.pass = this.encoder!.beginComputePass({ timestampWrites: { querySet: this.detailSet,
        beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
      this.detailNames.push(name);
      this.detailBytes.push(buffers.reduce((n, b) => n + ("buffer" in b ? (b.size ?? b.buffer.size - (b.offset ?? 0)) : b.size), 0));
    }
    const pipeline = this.pipeline(name, code);
    if (this.fastHost && this.hostCache && this.profiling !== "detail") {
      // DO-99: a decode step repeats the same dispatches; slot k of a submission usually binds what it bound last
      // time, so compare the bindings directly instead of building the cache key string.
      const k = this.slot++, last = this.slots[k];
      if (last && last.pipeline === pipeline && last.refs.length === buffers.length * 3) {
        let same = true;
        for (let i = 0, r = last.refs; i < buffers.length; i++) {
          const b = buffers[i] as any, isBinding = "buffer" in b;
          if (r[3 * i] !== (isBinding ? b.buffer : b) || r[3 * i + 1] !== (isBinding ? b.offset ?? 0 : 0) || r[3 * i + 2] !== (isBinding ? b.size ?? -1 : -1)) { same = false; break; }
        }
        if (same) { this.encodeBound(pipeline, last.bind, groups); return; }
      }
      const bind = this.bindFor(pipeline, buffers);
      const refs: unknown[] = [];
      let keep = true;
      for (const b of buffers as any[]) {
        const isBinding = "buffer" in b, buf = isBinding ? b.buffer : b;
        if (this.transient.includes(buf)) { keep = false; break; }
        refs.push(buf, isBinding ? b.offset ?? 0 : 0, isBinding ? b.size ?? -1 : -1);
      }
      if (keep) this.slots[k] = { pipeline, refs, bind }; else this.slots[k] = undefined;
      this.encodeBound(pipeline, bind, groups);
      return;
    }
    const entries = buffers.map((buffer, binding) => ({ binding, resource: "buffer" in buffer ? buffer : { buffer } }));
    let bind: GPUBindGroup | undefined, key = "";
    // Only bindings of long-lived buffers are cached: a transient uniform buffer dies at end().
    if (this.hostCache && !entries.some((e) => this.transient.includes(e.resource.buffer as GPUBuffer))) {
      key = this.id(pipeline) + ":" + entries.map((e) => {
        const r = e.resource as GPUBufferBinding;
        return `${this.id(r.buffer)}@${r.offset ?? 0}+${r.size ?? -1}`;
      }).join(",");
      bind = this.bindGroups.get(key);
    }
    if (!bind) {
      bind = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      if (key) {
        if (this.bindGroups.size >= 65536) this.bindGroups.clear();
        this.bindGroups.set(key, bind);
      }
    }
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bind);
    this.pass.dispatchWorkgroups(...groups);
  }

  // DO-99: the bind group for these bindings, from the keyed cache when long-lived (as dispatch() without fastHost).
  private bindFor(pipeline: GPUComputePipeline, buffers: readonly BufferBinding[]): GPUBindGroup {
    const entries = buffers.map((buffer, binding) => ({ binding, resource: "buffer" in buffer ? buffer : { buffer } }));
    let bind: GPUBindGroup | undefined, key = "";
    if (!entries.some((e) => this.transient.includes(e.resource.buffer as GPUBuffer))) {
      key = this.id(pipeline) + ":" + entries.map((e) => {
        const r = e.resource as GPUBufferBinding;
        return `${this.id(r.buffer)}@${r.offset ?? 0}+${r.size ?? -1}`;
      }).join(",");
      bind = this.bindGroups.get(key);
    }
    if (!bind) {
      bind = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      if (key) {
        if (this.bindGroups.size >= 65536) this.bindGroups.clear();
        this.bindGroups.set(key, bind);
      }
    }
    return bind;
  }

  private encodeBound(pipeline: GPUComputePipeline, bind: GPUBindGroup, groups: Workgroups) {
    if (pipeline !== this.lastPipeline) { this.pass!.setPipeline(pipeline); this.lastPipeline = pipeline; }
    this.pass!.setBindGroup(0, bind);
    this.pass!.dispatchWorkgroups(...groups);
  }

  async end() {
    if (!this.pass || !this.encoder) throw new Error("No active submission");
    this.pass.end();
    const detail = this.profiling === "detail" && !!this.detailSet && this.detailNames.length > 0;
    const timed = this.profiling && !!this.querySet && !detail;
    let staging: GPUBuffer | undefined, detailStaging: GPUBuffer | undefined;
    const names = this.detailNames, namesBytes = this.detailBytes;
    if (timed) {
      this.encoder.resolveQuerySet(this.querySet!, 0, 2, this.queryBuffer!, 0);
      staging = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.encoder.copyBufferToBuffer(this.queryBuffer!, 0, staging, 0, 16);
    }
    if (detail) {
      const bytes = 16 * names.length;
      this.encoder.resolveQuerySet(this.detailSet!, 0, 2 * names.length, this.detailBuffer!, 0);
      detailStaging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      this.encoder.copyBufferToBuffer(this.detailBuffer!, 0, detailStaging, 0, bytes);
    }
    if (this.uniformUsed) this.device.queue.writeBuffer(this.uniformPool!, 0, this.uniformStaging, 0, this.uniformUsed);
    const encodeMs = this.profiling ? performance.now() - this.profileStart : 0;
    { const t = this.hostAt(); this.device.queue.submit([this.encoder.finish()]); this.host.submit += this.hostAt() - t; this.host.submits++; }
    this.pass = undefined;
    if (this.defer && !this.profiling) {
      for (const b of this.transient) b.destroy();  // destroyed once the submitted work using it completes
      this.transient = [];
      if (this.lost) throw new Error(`device lost: ${this.lost}`);
      if (this.errors.length) throw new Error(`WebGPU errors: ${this.errors.slice(0, 5).join(" | ")}`);
      return;
    }
    { const t = this.hostAt(); await this.device.queue.onSubmittedWorkDone(); this.host.wait += this.hostAt() - t; }
    if (this.profiling) {
      let gpuMs: number | null = null;
      if (staging) {
        { const h = this.hostAt(); await staging.mapAsync(GPUMapMode.READ); this.host.map += this.hostAt() - h; this.host.maps++; }
        const t = new BigUint64Array(staging.getMappedRange());
        gpuMs = Number(t[1] - t[0]) / 1e6;
        staging.unmap();
        staging.destroy();
      }
      let gpuByName: Record<string, number> | undefined, bytesByName: Record<string, number> | undefined;
      if (detailStaging) {
        { const h = this.hostAt(); await detailStaging.mapAsync(GPUMapMode.READ); this.host.map += this.hostAt() - h; this.host.maps++; }
        const t = new BigUint64Array(detailStaging.getMappedRange());
        gpuByName = {};
        bytesByName = {};
        let sum = 0;
        const bytes = namesBytes;
        names.forEach((name, k) => {
          const ms = Number(t[2 * k + 1] - t[2 * k]) / 1e6;
          gpuByName![name] = (gpuByName![name] || 0) + ms;
          bytesByName![name] = (bytesByName![name] || 0) + (bytes[k] || 0);
          sum += ms;
        });
        gpuMs = sum;
        this.lastSequence = names.map((name, k) => [name, Number(t[2 * k + 1] - t[2 * k]) / 1e6] as [string, number]);
        detailStaging.unmap();
        detailStaging.destroy();
      }
      const dispatches = [...this.profileNames.values()].reduce((a, b) => a + b, 0);
      const expandMs = this.pendingExpandMs;
      this.pendingExpandMs = undefined;
      this.profileLog.push({ encodeMs, gpuMs, wallMs: performance.now() - this.profileStart, dispatches,
                             ...(expandMs !== undefined ? { expandMs } : {}),
                             uniforms: this.uniformUsed / 256, byName: Object.fromEntries(this.profileNames),
                             ...(gpuByName ? { gpuByName, bytesByName } : {}) });
    }
    for (const b of this.transient) b.destroy();
    this.transient = [];
    if (this.lost) throw new Error(`device lost: ${this.lost}`);
    if (this.errors.length) throw new Error(`WebGPU errors: ${this.errors.slice(0, 5).join(" | ")}`);
  }

  async read(buffer: GPUBuffer, bytes: number) {
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    { const t = this.hostAt(); this.device.queue.submit([encoder.finish()]); this.host.submit += this.hostAt() - t; this.host.submits++; }
    { const t = this.hostAt(); await staging.mapAsync(GPUMapMode.READ); this.host.map += this.hostAt() - t; this.host.maps++; }
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // --- operations -------------------------------------------------------

  matmul(W: PackedMatrix, x: BufferBinding, y: BufferBinding, tokens: number, { xStride, xOffset = 0, yStride, yOffset = 0, accumulate = false, rows, rowBase = 0 }: MatmulOptions = {}) {
    if (!Object.hasOwn(DEQ, W.kind))
      throw new Error(`Unsupported packed matrix type: ${W.kind}`);
    rows = rows ?? W.rows;
    const groupsX = Math.min(rows, MAX_DISPATCH), groupsZ = Math.ceil(rows / MAX_DISPATCH);
    const u = this.uniform([
      ["u", rows], ["u", W.cols], ["u", tokens], ["u", W.rowBytes],
      ["u", xStride ?? W.cols], ["u", xOffset], ["u", yStride ?? rows], ["u", yOffset],
      ["u", accumulate ? 1 : 0], ["u", rowBase], ["u", 0], ["u", 0],
    ]);
    const kernel = !FAST.has(W.kind) || this.generic || (W.kind === "Q8_0" && !this.fastQ8 && tokens > this.q8FastUpTo) ? "generic"
      : tokens <= (this.matvec?.tokens ?? VT) ? "matvec" : this.prefill || (tokens >= GEMM_MIN ? "gemm" : "tiled");
    // WS-46: the GEMM has no Q8_0 decoder (its matrices are 32 rows in Qwen3.5): the tiled kernel instead.
    const many = kernel === "gemm" && !Object.hasOwn(UNIT, W.kind) ? "tiled" : kernel;
    if (many === "gemm") {
      const g = this.gemm || GEMM, key = `${g.GR}x${g.GT}_${g.RR}x${g.RT}${g.V ? "v" : ""}`;
      this.dispatch(`gemm${key}_${W.kind}`, gemmShader(fastKind(W.kind) as keyof typeof UNIT, g), [W.buffer, x, y, u],
        [Math.ceil(rows / g.GR), Math.ceil(tokens / g.GT), 1]);
    } else if (many === "matvec" && tokens === 1 && this.matvecC && ((this.matvecC.kinds?.[W.kind] ?? this.matvecC).rows ?? 1) > 1) {
      // DO-76: several rows per thread through the grouped kernel (one product; same sums as matvecC)
      this.groupDispatch(this.matvecC.kinds?.[W.kind] ?? this.matvecC, [{ W, y, yOffset: yOffset + rowBase, rows }], x, false, null, xOffset, accumulate);
    } else if (many === "matvec" && tokens === 1 && this.matvecC) {
      const { tpr, lanes, wg = 64 } = this.matvecC.kinds?.[W.kind] ?? this.matvecC, groups = Math.ceil(rows / (wg / tpr));
      this.dispatch(`matvecC${tpr}x${lanes}${wg === 64 ? "" : `w${wg}`}_${W.kind}`, matvecCShader(fastKind(W.kind), tpr, lanes, wg), [W.buffer, x, y, u],
        [Math.min(groups, MAX_DISPATCH), 1, Math.ceil(groups / MAX_DISPATCH)]);
    } else if (many === "matvec" && tokens > 1 && this.matvecC && this.matvecB) {
      if (![2, 4, 8].includes(this.matvecB)) throw new Error("matvecB tokens per pass 2, 4 or 8");
      // The smallest pass that holds the batch (unused token lanes would still cost their arithmetic).
      const { tpr, lanes, wg = 64 } = this.matvecC.kinds?.[W.kind] ?? this.matvecC, T = Math.min(this.matvecB, tokens <= 2 ? 2 : tokens <= 4 ? 4 : 8);
      const groups = Math.ceil(rows / (wg / tpr));
      this.dispatch(`matvecB${tpr}x${lanes}x${T}${wg === 64 ? "" : `w${wg}`}_${W.kind}`, matvecBShader(fastKind(W.kind), tpr, lanes, T, wg), [W.buffer, x, y, u],
        [Math.min(groups, MAX_DISPATCH), 1, Math.ceil(groups / MAX_DISPATCH)]);
    } else if (many === "matvec" && this.matvec) {
      const R = tokens === 1 ? this.matvec.single ?? this.matvec.rows : this.matvec.rows;
      const vt = tokens === 1 ? 1 : this.matvec.tokens, groups = Math.ceil(rows / (MR * R));
      if (![1, 2, 4].includes(R) || ![4, 8].includes(this.matvec.tokens)) throw new Error("matvec rows 1, 2 or 4; tokens 4 or 8");
      this.dispatch(`matvecR${R}x${vt}_${W.kind}`, matvecRowsShader(fastKind(W.kind), vt, R), [W.buffer, x, y, u],
        [Math.min(groups, MAX_DISPATCH), Math.ceil(tokens / vt), Math.ceil(groups / MAX_DISPATCH)]);
    } else if (many === "matvec") {
      const vt = tokens === 1 ? 1 : VT, groups = Math.ceil(rows / MR);
      this.dispatch(`matvec${vt}_${W.kind}`, matvecShader(fastKind(W.kind), vt), [W.buffer, x, y, u],
        [Math.min(groups, MAX_DISPATCH), Math.ceil(tokens / vt), Math.ceil(groups / MAX_DISPATCH)]);
    } else if (many === "tiled") {
      this.dispatch(`tiled_${W.kind}`, tiledShader(fastKind(W.kind)), [W.buffer, x, y, u],
        [Math.ceil(rows / TR), Math.ceil(tokens / TQ), 1]);
    } else {
      this.dispatch(`matmul_${W.kind}`, matmulShader(W.kind), [W.buffer, x, y, u],
        [groupsX, Math.ceil(tokens / TT), groupsZ]);
    }
  }

  gather(chunks: readonly EmbeddingChunk[], ids: GPUBuffer, y: GPUBuffer, tokens: number) {
    for (const c of chunks) {
      if (!Object.hasOwn(DEQ, c.kind))
        throw new Error(`Unsupported embedding type: ${c.kind}`);
      const u = this.uniform([["u", c.first], ["u", c.rows], ["u", c.cols], ["u", c.rowBytes], ["u", tokens], ["u", 0], ["u", 0], ["u", 0]]);
      this.dispatch(`gather_${c.kind}`, gatherShader(c.kind), [c.buffer, ids, y, u], [tokens, 1, 1]);
    }
  }

  norm(x: BufferBinding, weight: BufferBinding, y: BufferBinding, tokens: number, { size, groups = 1, inStride, inGroup = 0, inOffset = 0, outStride, outGroup = 0, outOffset = 0,
    gate = null, gateStride = 0, gateGroup = 0, gateOffset = 0, eps = 1e-6 }: NormOptions) {
    const u = this.uniform([
      ["u", size], ["u", groups], ["u", inStride ?? size * groups], ["u", inGroup || size],
      ["u", outStride ?? size * groups], ["u", outGroup || size], ["u", gateStride], ["u", gateGroup || size],
      ["u", inOffset], ["u", outOffset], ["u", gateOffset], ["u", gate ? 1 : 0],
      ["f", eps], ["u", 0], ["u", 0], ["u", 0],
    ]);
    this.dispatch("norm", NORM, [x, weight, gate ?? weight, y, u], [tokens * groups, 1, 1]);
  }

  // x -> y = RMSNorm(x) * weight, then residual += y: `norm` followed by `add` in one dispatch (one group).
  normAdd(x: BufferBinding, weight: BufferBinding, y: BufferBinding, residual: BufferBinding, tokens: number, size: number, eps: number) {
    const u = this.uniform([
      ["u", size], ["u", 1], ["u", size], ["u", size], ["u", size], ["u", size], ["u", 0], ["u", size],
      ["u", 0], ["u", 0], ["u", 0], ["u", 0], ["f", eps], ["u", 0], ["u", 0], ["u", 0],
    ]);
    this.dispatch("norm_add", NORM_ADD, [x, weight, weight, y, u, residual], [tokens, 1, 1]);
  }

  // Two grouped RMS norms with the same group size (QK-norm) in one dispatch: a then b, contiguous rows.
  normPair(xa: BufferBinding, wa: BufferBinding, ya: BufferBinding, ga: number, xb: BufferBinding, wb: BufferBinding,
           yb: BufferBinding, gb: number, tokens: number, size: number, eps: number) {
    const u = this.uniform([["u", size], ["u", ga], ["u", gb], ["u", tokens], ["f", eps], ["u", 0], ["u", 0], ["u", 0]]);
    this.dispatch("norm_pair", NORM_PAIR, [xa, wa, ya, xb, wb, yb, u], [tokens * (ga + gb), 1, 1]);
  }

  argmax(x: BufferBinding, out: BufferBinding, n: number) {
    const u = this.uniform([["u", n], ["u", 0], ["u", 0], ["u", 0]]);
    this.dispatch("argmax", ARGMAX, [x, out, u], [1, 1, 1]);
  }

  // WS-36: whether matmulGroup can take these one-token products (coalesced kernel, fast decoders, one shape).
  groupShape(Ws: readonly PackedMatrix[]): MatvecC | null {
    if (!this.groupMatmul || !this.matvecC || this.generic || Ws.length < 2 || Ws.length > 4) return null;
    if (!Ws.every((W) => FAST.has(W.kind) && W.cols === Ws[0].cols && (W.kind !== "Q8_0" || this.fastQ8 || this.q8FastUpTo >= 1))) return null;
    const shapes = Ws.map((W) => this.matvecC!.kinds?.[W.kind] ?? this.matvecC!);
    if (!shapes.every((c) => c.tpr === shapes[0].tpr && c.lanes === shapes[0].lanes)) return null;
    return shapes[0];
  }

  // WS-36: y_i[yOffset_i + row] = W_i x for one token, every product in one dispatch (see matvecGroupShader);
  // glu: [gate, up] and up's output becomes silu(gate) * up (silu_mul fused). The caller checks groupShape first.
  matmulGroup(items: readonly { W: PackedMatrix; y: BufferBinding; yOffset?: number }[], x: BufferBinding, glu = false,
              norm: { src: BufferBinding; weight: BufferBinding; eps: number } | null = null) {
    const shape = this.groupShape(items.map((i) => i.W));
    if (!shape) throw new Error("matmulGroup: products cannot share a dispatch");
    if (glu && (items.length !== 2 || items[0].W.rows !== items[1].W.rows || items[0].W.kind !== items[1].W.kind)) throw new Error("matmulGroup: glu needs two equal products");
    this.groupDispatch(shape, items, x, glu, norm, 0, false);
  }

  // The grouped kernel's dispatch (also DO-76's path for single products with several rows per thread).
  groupDispatch(shape: MatvecC, items: readonly { W: PackedMatrix; y: BufferBinding; yOffset?: number; rows?: number }[], x: BufferBinding,
                glu: boolean, norm: { src: BufferBinding; weight: BufferBinding; eps: number } | null, xOffset: number, accumulate: boolean) {
    const { tpr, lanes } = shape, wg = shape.wg ?? 64, R = shape.rows ?? 1, RPW = wg / tpr;
    const ys: BufferBinding[] = [], outs = items.map((i) => { let o = ys.indexOf(i.y); if (o < 0) { ys.push(i.y); o = ys.length - 1; } return o; });
    const seg: number[] = [];
    let first = 0;
    for (const i of items) {
      const rows = i.rows ?? i.W.rows;
      seg.push(rows, i.W.rowBytes, i.yOffset ?? 0, first);
      first += glu ? 0 : Math.ceil(rows / (RPW * R));
    }
    const groups = glu ? Math.ceil(items[0].W.rows / (RPW * R)) : first;
    while (seg.length < 16) seg.push(0, 0, 0, 0xffffffff);
    const u = this.uniform([["u", items[0].W.cols], ["u", xOffset], ["u", accumulate ? 1 : 0], ["f", norm ? norm.eps : 0], ...seg.slice(0, 16).map((v) => ["u", v] as UniformValue)]);
    const kinds = items.map((i) => fastKind(i.W.kind));
    this.dispatch(`matvecG${glu ? "lu" : ""}${norm ? "N" : ""}${tpr}x${lanes}${wg === 64 ? "" : `w${wg}`}${R === 1 ? "" : `r${R}`}_${kinds.join("_")}`,
      matvecGroupShader(kinds, outs, tpr, lanes, wg, glu, !!norm, R),
      [...items.map((i) => i.W.buffer), norm ? norm.src : x, ...ys, u, ...(norm ? [norm.weight, x] : [])],
      [Math.min(groups, MAX_DISPATCH), 1, Math.ceil(groups / MAX_DISPATCH)]);
  }

  siluMul(a: BufferBinding, b: BufferBinding, n: number) {
    const u = this.uniform([["u", n], ["u", 0], ["u", 0], ["u", 0]]);
    const groups = Math.ceil(n / 256);
    this.dispatch("silu_mul", SILU_MUL, [a, b, u],
      [Math.min(groups, MAX_DISPATCH), Math.ceil(groups / MAX_DISPATCH), 1]);
  }
}


/** Per-call resources; never destroys a caller-owned device or input buffer. */
export class BufferScope {
  readonly buffers: GPUBuffer[] = [];
  constructor(readonly device: GPUDevice) {}
  empty(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({size: Math.max(4, size), usage});
    this.buffers.push(buffer);
    return buffer;
  }
  upload(data: ArrayBufferView<ArrayBuffer>, usage: GPUBufferUsageFlags): GPUBuffer {
    const buffer = this.device.createBuffer({size: Math.max(4,data.byteLength), usage, mappedAtCreation:true});
    this.buffers.push(buffer);
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
    buffer.unmap();
    return buffer;
  }
  destroy(): void { for (const buffer of this.buffers) buffer.destroy(); this.buffers.length = 0; }
}
const programs = new WeakMap<GPUDevice, Map<string, Promise<GPUComputePipeline[]>>>();
/** Entry points and explicit binding kinds are part of pipeline identity. */
export function compileProgram(device: GPUDevice, code: string, entries: readonly string[], kinds?: readonly GPUBufferBindingType[]): Promise<GPUComputePipeline[]> {
  let cache = programs.get(device);
  if (!cache) { cache = new Map(); programs.set(device,cache); }
  const key = JSON.stringify([code,entries,kinds]);
  const old = cache.get(key);
  if (old) return old;
  const pending = (async () => {
    const module = device.createShaderModule({code});
    const diagnostics = await module.getCompilationInfo();
    const errors = diagnostics.messages.filter(m=>m.type==='error');
    if (errors.length) throw Error(errors.map(m=>m.message).join('\n'));
    const layout = kinds ? device.createPipelineLayout({bindGroupLayouts:[device.createBindGroupLayout({entries:kinds.map((type,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type}}))})]}) : 'auto';
    return Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({layout,compute:{module,entryPoint}})));
  })();
  cache.set(key,pending);
  void pending.catch(()=>cache!.delete(key));
  return pending;
}
/** One pass per entry, preserving ordering; raw F32 readback (no conversion). */
export async function runReadback(scope: BufferScope, pipelines: readonly GPUComputePipeline[], buffers: readonly GPUBuffer[], dispatches: readonly Workgroups[], output: GPUBuffer, bytes: number): Promise<Float32Array<ArrayBuffer>> {
  if (pipelines.length !== dispatches.length) throw Error('Dispatch count differs from entry points');
  const device=scope.device;
  const read=scope.empty(bytes,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
  device.pushErrorScope('validation');
  try {
    const encoder=device.createCommandEncoder();
    pipelines.forEach((pipeline,i)=>{
      const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
      const pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,bind);
      pass.dispatchWorkgroups(...dispatches[i]);pass.end();
    });
    encoder.copyBufferToBuffer(output,0,read,0,bytes);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    try { return new Float32Array(read.getMappedRange().slice(0)); }
    finally { read.unmap(); }
  } finally {
    const error=await device.popErrorScope();
    if(error)throw Error(error.message);
  }
}

/** Callers serialize operations on a device while these error scopes are open. */
export async function checkedGPU<T>(device: GPUDevice, operation: () => Promise<T>): Promise<T> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  try { return await operation(); }
  finally {
    const validation = await device.popErrorScope();
    const memory = await device.popErrorScope();
    if (validation || memory) throw Error([validation?.message, memory?.message].filter(Boolean).join('; '));
  }
}

export function executeResidentKernel(gpu: Gpu, request: import('./webgpu_contracts.mjs').ResidentKernelRequest): void {
  if(request.arithmetic!=='float32')throw Error('Resident kernels require float32 arithmetic');
  switch(request.op) {
    case 'packed_matmul': return gpu.matmul(request.weight,request.input,request.output,request.tokens,request.options);
    case 'rms_norm': return gpu.norm(request.input,request.weight,request.output,request.tokens,request.options);
    case 'embedding': return gpu.gather(request.chunks,request.ids,request.output,request.tokens);
    case 'silu_mul': return gpu.siluMul(request.input,request.output,request.elements);
    case 'argmax': return gpu.argmax(request.input,request.output,request.elements);
    default: throw Error('Unknown resident operation');
  }
}
