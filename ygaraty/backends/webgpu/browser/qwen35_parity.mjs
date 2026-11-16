// DO-36 parity page: runs kernels, one stage, all stages, or one contributor's
// stage on WebGPU and posts the results to the harness
// (ygaraty/experiments/qwen35_webgpu.py). The stage implementation follows the
// manifest's model family: Qwen3.5 (qwen35_gpu.mjs), dense (dense_gpu.mjs) or Kolibri-1 (kolibri_gpu.mjs).
import { Gpu as QwenGpu, Stage as QwenStage, F16, V_HEADS, HEAD_K, HEAD_V, CONV_DIM } from "./qwen35_gpu.mjs";
import { Stage as DenseStage } from "./dense_gpu.mjs";
import { Stage as KolibriStage } from "./kolibri_gpu.mjs";
import { residentDecode } from "./inference_plan.mjs";
import { BYTES, REPACK, calibrateRepack, wgslCache } from "./webgpu_kernels.mjs";
let Stage = QwenStage;

const log = (line) => fetch("/log", { method: "POST", body: line });
const get = async (name) => {
  const response = await fetch(`/f/${name}`);
  if (!response.ok) throw new Error(`fetch ${name}: ${response.status}`);
  return response.arrayBuffer();
};
const blob = (name, array) => fetch(`/blob/${name}`, { method: "POST", body: array });

async function kernels(gpu, manifest) {
  const timings = [], sweeps = {}, roofs = {};
  // DO-83: the contributor's load-time repack measurement on this GPU (with the session's kernel options).
  if (manifest.calibrate_repack) return { timings, sweeps, roofs, repackCalibration: await calibrateRepack(gpu) };
  for (const [i, c] of manifest.cases.entries()) {
    // Experiment 141 / DO-80: the repacked decoders, checked against NumPy like the originals.
    const rp = (manifest.repack_q4k && c.kind === "Q4_K") || (manifest.repack || []).includes(c.kind) ? REPACK[c.kind] : null;
    const rep = !!rp;
    const raw = await get(c.file);
    const W = { kind: rp ? rp.kind : c.kind, rows: c.shape[0], cols: c.shape[1], rowBytes: rp ? (c.shape[1] / 256) * rp.block : c.row_bytes,
                buffer: gpu.upload(rp ? rp.fn(raw, c.shape[0], c.shape[1]) : raw) };
    const x = gpu.upload(await get(c.x));
    const y = gpu.buffer(c.tokens * W.rows * 4);
    gpu.begin();
    gpu.matmul(W, x, y, c.tokens);
    const start = performance.now();
    await gpu.end();
    timings.push(performance.now() - start);
    await blob(`y${i}.f32`, await gpu.read(y, c.tokens * W.rows * 4));
    // Throughput: `repeat` dispatches in one submission, after the warm one.
    const repeat = manifest.repeat || 0;
    if (repeat) {
      gpu.begin();
      for (let r = 0; r < repeat; r++) gpu.matmul(W, x, y, c.tokens);
      const begin = performance.now();
      await gpu.end();
      timings[i] = (performance.now() - begin) / repeat;
    }
    // DO-74: every coalesced-kernel shape in `sweep` on this case (one-token cases), its output kept for the
    // harness's check and its time the median of `rounds` rounds, the shapes' order reversed every other round.
    if (manifest.sweep && c.tokens === 1) {
      const base = gpu.matvecC, n = manifest.sweep.length, times = manifest.sweep.map(() => []);
      for (const [s, cfg] of manifest.sweep.entries()) {
        gpu.matvecC = cfg;
        gpu.begin(); gpu.matmul(W, x, y, 1); await gpu.end();
        await blob(`y${i}_s${s}.f32`, await gpu.read(y, W.rows * 4));
      }
      for (let r = 0; r < (manifest.rounds || 3); r++) {
        for (let o = 0; o < n; o++) {
          const s = r % 2 ? n - 1 - o : o;
          gpu.matvecC = manifest.sweep[s];
          gpu.begin();
          for (let k = 0; k < (repeat || 20); k++) gpu.matmul(W, x, y, 1);
          const begin = performance.now();
          await gpu.end();
          times[s].push((performance.now() - begin) / (repeat || 20));
        }
      }
      gpu.matvecC = base;
      (sweeps[i] = times);
    }
    if (manifest.roof && c.tokens === 1 && !rep) roofs[i] = await roof(gpu, manifest, c, W, x, y, i);
    for (const b of [W.buffer, x, y]) gpu.release(b);
    await log(`case ${i} ${c.name} ${c.kind} done`);
  }
  return { timings, sweeps, roofs };
}

// DO-78: on `copies` distinct copies of a one-token case's matrix (together more than the GPU's L2), the median
// ms per dispatch over `rounds` rounds of A (a plain 16-byte streaming read), B (matvecC's thread-to-byte mapping
// and loop, raw words XORed and inputs summed, no decoding) and C (the production one-token kernel).
const ROOF_TPR = 16, ROOF_L = 4;
const ROOF_READ = /* wgsl */ `
struct P { n4: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  var a = vec4<u32>(0u);
  for (var i = wg.x * 256u + li.x; i < p.n4; i += nw.x * 256u) { a ^= w[i]; }
  part[li.x] = a.x ^ a.y ^ a.z ^ a.w;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li.x < s) { part[li.x] ^= part[li.x + s]; } workgroupBarrier(); }
  if (li.x == 0u) { y[wg.x] = part[0]; }
}`;
const ROOF_OFFSETS = {  // the byte offsets of matvecC's quant words (qo) and extra words (ho), and its input elements
  Q4_K: { block: 144, setup: "let qo = b + 16u + 32u * g; let ho = 0u; let el = 64u * g; let eh = el + 32u;", hi: false },
  Q5_K: { block: 176, setup: "let qo = b + 48u + 32u * g; let ho = b + 16u; let el = 64u * g; let eh = el + 32u;", hi: true },
  Q6_K: { block: 210, setup: "let h = g / 2u; let c = g % 2u; let qo = b + 64u * h + 32u * c; let ho = b + 128u + 32u * h; let el = 128u * h + 32u * c; let eh = el + 64u;", hi: true },
};
const roofPattern = (kind) => { const o = ROOF_OFFSETS[kind]; return /* wgsl */ `
struct P { rows: u32, cols: u32, row_bytes: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
${BYTES}
var<workgroup> part: array<u32, 64>;
var<workgroup> fpart: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let r = li.x / ${ROOF_TPR}u; let k = li.x % ${ROOF_TPR}u;
  let row = wg.x * ${64 / ROOF_TPR}u + r;
  var cs = 0u; var xs = 0.0;
  if (row < p.rows) {
    let items = p.cols / 64u; let part_i = k % ${ROOF_L}u;
    for (var it = k / ${ROOF_L}u; it < items; it += ${ROOF_TPR / ROOF_L}u) {
      let sb = it / 4u; let g = it % 4u;
      let b = row * p.row_bytes + sb * ${o.block}u;
      cs ^= word_at(b) ^ word_at(b + 4u);  // the block header (read for every pair, so it cancels in the XOR)
      ${o.setup}
      for (var j = 0u; j < ${8 / ROOF_L}u; j++) {
        let i = part_i * ${8 / ROOF_L}u + j;
        cs ^= word_at(qo + 4u * i)${o.hi ? " ^ word_at(ho + 4u * i)" : ""};
        xs += dot(x[el / 4u + i] + x[eh / 4u + i], vec4<f32>(1.0));
      }
    }
  }
  part[li.x] = cs; fpart[li.x] = xs;
  workgroupBarrier();
  for (var s = ${ROOF_TPR / 2}u; s > 0u; s >>= 1u) {
    if (k < s) { part[li.x] ^= part[li.x + s]; fpart[li.x] += fpart[li.x + s]; }
    workgroupBarrier();
  }
  if (row < p.rows && k == 0u) { y[row] = part[li.x]; y[p.rows + row] = bitcast<u32>(fpart[li.x]); }
}`; };

async function roof(gpu, manifest, c, W, x, y, i) {
  const copies = c.copies || 1, n = manifest.roof_dispatches || 30, rounds = manifest.rounds || 5;
  const data = await get(c.file), bytes = data.byteLength;
  const Ws = [W.buffer, ...Array.from({ length: copies - 1 }, () => gpu.upload(data))];
  const ya = gpu.buffer(4096 * 4), yb = gpu.buffer(2 * W.rows * 4);
  const n4 = Math.floor(bytes / 16);
  const ub = () => gpu.uniform([["u", W.rows], ["u", W.cols], ["u", W.rowBytes], ["u", 0]]);
  const run = {
    A: (k) => gpu.dispatch("roof_read", ROOF_READ, [Ws[k], ya, gpu.uniform([["u", n4], ["u", 0], ["u", 0], ["u", 0]])], [4096, 1, 1]),
    B: (k) => gpu.dispatch(`roof_pattern_${c.kind}`, roofPattern(c.kind), [Ws[k], x, yb, ub()], [Math.ceil(W.rows / (64 / ROOF_TPR)), 1, 1]),
    C: (k) => gpu.matmul({ ...W, buffer: Ws[k] }, x, y, 1),
  };
  // Checksums from copy 0 (the harness checks them against NumPy), then the timed rounds.
  gpu.begin(); run.A(0); run.B(0); await gpu.end();
  await blob(`roofA${i}.u32`, await gpu.read(ya, 4096 * 4));
  await blob(`roofB${i}.u32`, await gpu.read(yb, 2 * W.rows * 4));
  const times = { A: [], B: [], C: [] }, order = ["A", "B", "C"];
  for (let r = 0; r < rounds; r++) {
    for (const key of r % 2 ? [...order].reverse() : order) {
      gpu.begin();
      for (let d = 0; d < n; d++) run[key](d % copies);
      const t0 = performance.now();
      await gpu.end();
      times[key].push((performance.now() - t0) / n);
    }
  }
  for (const b of [...Ws.slice(1), ya, yb]) gpu.release(b);
  return { copies, bytes, dispatches: n, times };
}

async function stage(gpu, manifest) {
  const f = manifest.fixture;
  const s = new Stage(gpu, manifest, f.max_tokens, f.max_positions, { kv: f.kv });
  const t0 = performance.now();
  await s.load(get, log);
  const loadMs = performance.now() - t0;
  const afterLoad = gpu.allocated;
  const hidden = f.hidden ? new Float32Array(await get(f.hidden)) : null;
  const runs = [];
  let at = 0;
  // DO-80: each call's expert ids per layer (MoE stages), for a reference that follows the GPU's routing.
  const capture = manifest.capture_routes && s.config?.top_k
    ? { buffer: gpu.buffer(manifest.layers.length * f.max_tokens * s.config.top_k * 4), at: 0 } : null;
  for (const [i, n] of f.chunks.entries()) {
    const start = performance.now();
    if (capture) { capture.at = 0; gpu.routeCapture = capture; }
    if (manifest.embed) await s.run({ ids: f.tokens.slice(at, at + n), tokens: n });
    else await s.run({ hidden: hidden.subarray(at * s.hidden, (at + n) * s.hidden), tokens: n });
    if (capture) { gpu.routeCapture = undefined; await blob(`routes${i}.u32`, await gpu.read(capture.buffer, capture.at * 4)); }
    runs.push({ tokens: n, ms: performance.now() - start, submitMs: s.submitMs });
    await log(`call ${i}: ${n} tokens, submissions ${s.submitMs.map((v) => v.toFixed(0)).join(" ")} ms`);
    await blob(`out${i}.f32`, await gpu.read(s.x, n * s.hidden * 4));
    at += n;
  }
  const states = [];
  for (const L of s.layers) {
    if (L.type !== "linear") continue;
    await blob(`S${L.index}.f32`, await gpu.read(L.S, V_HEADS * HEAD_K * HEAD_V * 4));
    await blob(`conv${L.index}.f32`, await gpu.read(L.convState, 3 * CONV_DIM * 4));
    states.push(L.index);
  }
  return { loadMs, runs, states, allocated: afterLoad, peak: gpu.peak };
}

// The wire: hidden states between stages, float16 (rounded to nearest even,
// as NumPy's float16) or float32.
const toWire = (hidden, wire) => (wire === "f16" ? new Float16Array(hidden) : hidden);
const fromWire = (data) => (data instanceof Float32Array ? data : new Float32Array(data));
const maxAbs = (a) => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };

// All stages in one browser; between stages the hidden states go through the
// CPU as they would over the wire. Prefill in chunks, then greedy steps.
async function pipeline(gpu, manifest) {
  const f = manifest.fixture;
  const stages = [];
  const t0 = performance.now();
  for (const m of manifest.stages) {
    const s = new Stage(gpu, m, f.max_tokens, f.max_positions, { kv: f.kv, mergeExperts: f.merge_experts !== false });
    await s.load(get, log);
    stages.push(s);
  }
  const loadMs = performance.now() - t0, allocated = gpu.allocated;
  gpu.profiling = !!f.profile;
  gpu.hostCache = f.host_cache !== false;
  gpu.fuse = f.fuse !== false;
  const boundaryMax = stages.slice(1).map(() => 0);
  const forward = async (ids) => {
    const stageMs = [];
    let hidden = null;
    for (const [k, s] of stages.entries()) {
      const start = performance.now();
      if (k === 0) await s.run({ ids, tokens: ids.length });
      else await s.run({ hidden, tokens: ids.length });
      if (k < stages.length - 1) {
        const out = await gpu.read(s.x, ids.length * s.hidden * 4);
        boundaryMax[k] = Math.max(boundaryMax[k], maxAbs(out));
        hidden = fromWire(toWire(out, f.wire));
      }
      stageMs.push(performance.now() - start);
    }
    const last = stages[stages.length - 1];
    return { token: await last.next(), stageMs };
  };
  const prefill = [], steps = [], generated = [];
  let at = 0, out;
  for (const n of f.chunks) {
    const start = performance.now();
    out = await forward(f.tokens.slice(at, at + n));
    prefill.push({ tokens: n, ms: performance.now() - start, stageMs: out.stageMs });
    at += n;
  }
  const last = stages[stages.length - 1];
  for (let i = 0; i < f.generate; i++) {
    generated.push(out.token);
    await blob(`logits${i}.f32`, await gpu.read(last.logits, last.vocab * 4));
    await log(`step ${i}: token ${out.token}`);
    if (i === f.generate - 1 || (f.eos.includes(out.token) && f.stop_at_eos)) break;
    const start = performance.now();
    if (f.profile) gpu.takeProfile();
    out = await forward([out.token]);
    steps.push({ ms: performance.now() - start, stageMs: out.stageMs, ...(f.profile ? { profile: gpu.takeProfile() } : {}) });
  }
  return { loadMs, allocated, peak: gpu.peak, prefill, steps, generated, boundaryMax };
}

// One contributor: one stage alone in this browser, as a contributor runs
// it. Prefill `chunks`, then single-token steps; the inputs are token ids
// (first stage) or the previous contributor's wire output, and each call's
// output goes back as a blob (wire format), so the harness can chain the
// four contributors' runs. `phase` lines mark points for the memory sampler.
async function contributor(gpu, manifest) {
  const f = manifest.fixture;
  await log("phase device");
  const s = new Stage(gpu, manifest, f.max_tokens, f.max_positions, { kv: f.kv });
  const t0 = performance.now();
  await s.load(get, log);
  const loadMs = performance.now() - t0, allocated = gpu.allocated;
  await log("phase loaded");
  const calls = [], tokens = [];
  let at = 0;
  for (const [i, n] of f.chunks.entries()) {
    if (i === f.prefill_calls) await log("phase prefilled");
    let input;
    if (manifest.embed) input = { ids: f.tokens.slice(at, at + n), tokens: n };
    else input = { hidden: fromWire(f.wire === "f16" ? new Float16Array(await get(`in${i}`)) : new Float32Array(await get(`in${i}`))), tokens: n };
    const start = performance.now();
    await s.run(input);
    if (s.output) tokens.push(await s.next());
    else await blob(`out${i}`, toWire(await gpu.read(s.x, n * s.hidden * 4), f.wire));
    calls.push({ tokens: n, position: at, ms: performance.now() - start, submitMs: s.submitMs });
    if (i % 16 === 0 || i >= f.prefill_calls) await log(`call ${i}: ${n} tokens at ${at}, ${calls[i].ms.toFixed(0)} ms`);
    at += n;
  }
  await log("phase done");
  return { loadMs, allocated, peak: gpu.peak, calls, tokens };
}

const result = {};
try {
  const manifest = JSON.parse(new TextDecoder().decode(await get("manifest.json")));
  const family = (manifest.config || manifest.stages?.[0]?.config || {}).family;
  if (family === "dense") Stage = DenseStage;
  if (family === "kolibri1" || family === "qwen3moe") Stage = KolibriStage;  // DO-80: one MoE adapter, two profiles
  const gpu = await QwenGpu.create();  // either stage runs on it (the dense one adds functions, not methods)
  // Diagnostics: force the many-token kernel ("tiled" or "gemm") and its tiling.
  if (manifest.prefill_kernel) gpu.prefill = manifest.prefill_kernel;
  if (manifest.gemm) gpu.gemm = manifest.gemm;
  if (manifest.matvec) gpu.matvec = manifest.matvec;  // WS-15: the R-rows-per-thread matrix-vector kernel
  if (manifest.matvec_c) gpu.matvecC = manifest.matvec_c;  // WS-34: the coalesced one-token kernel
  if (manifest.matvec_b) gpu.matvecB = manifest.matvec_b;
  if (manifest.group_matmul) gpu.groupMatmul = true;  // WS-36
  if (manifest.norm_fold) gpu.normFold = true;  // WS-49
  if (manifest.repack_q4k) gpu.repackQ4K = true;  // experiment 141
  if (manifest.repack) gpu.repack = manifest.repack;  // DO-80
  if (manifest.q8_fast) gpu.fastQ8 = true;  // WS-46: Q8_0 through the vectorized decoders (opt-in)
  if (manifest.q8_fast_up_to) gpu.q8FastUpTo = manifest.q8_fast_up_to;  // WS-47  // DO-69: batches through the coalesced kernel
  if (manifest.attention_split) gpu.attentionSplit = manifest.attention_split;  // WS-35: decode attention splits
  const info = gpu.adapter.info || {};
  result.adapter = { vendor: info.vendor, architecture: info.architecture, description: info.description,
                     features: [...gpu.adapter.features].sort(),
                     limits: { maxStorageBuffersPerShaderStage: gpu.device.limits.maxStorageBuffersPerShaderStage } };  // WS-49  // DO-68: what each WebGPU implementation offers
  // float32 -> float16 by WGSL's pack2x16float and by the cache's own f16_bits,
// and back by f16_value; the harness compares them with NumPy.
async function pack(gpu, manifest) {
  const x = new Float32Array(await get("x"));
  const n = x.length;
  const code = `
    @group(0) @binding(0) var<storage, read> x: array<f32>;
    @group(0) @binding(1) var<storage, read_write> y: array<u32>;
    @group(0) @binding(2) var<storage, read_write> z: array<u32>;
    @group(0) @binding(3) var<storage, read_write> back: array<f32>;
    ${F16}
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i < ${n / 2}u) {
        y[i] = pack2x16float(vec2<f32>(x[2u * i], x[2u * i + 1u]));
        let u = f16_bits(x[2u * i]) | (f16_bits(x[2u * i + 1u]) << 16u);
        z[i] = u;
        let v = f16x2(u);
        back[2u * i] = v.x; back[2u * i + 1u] = v.y;
      }
    }`;
  const xb = gpu.upload(x), yb = gpu.buffer(n * 2), zb = gpu.buffer(n * 2), bb = gpu.buffer(n * 4);
  gpu.begin();
  gpu.dispatch("pack", code, [xb, yb, zb, bb], [Math.ceil(n / 2 / 256), 1, 1]);
  await gpu.end();
  await blob("pack.f16", await gpu.read(yb, n * 2));
  await blob("bits.f16", await gpu.read(zb, n * 2));
  await blob("back.f32", await gpu.read(bb, n * 4));
  return {};
}

// DO-63 bench: one stage loaded once, the prompt prefilled into slot 0, then decode variants in turn without
// a reference: `profile` true (per submission) or "detail" (GPU time per dispatch, with the ordered sequence of
// the variant's last token), `matvec` ({rows, tokens, single} or null), `batch` (T conversations per step).
async function bench(gpu, manifest) {
  const f = manifest.fixture;
  const s = new Stage(gpu, manifest.stages[0], f.max_tokens, f.max_positions,
                      { kv: f.kv, slots: f.slots || 1, mergeExperts: f.merge_experts !== false });
  gpu.repackQ4K = !!f.repack_q4k;  // experiment 141: applies at load
  gpu.repack = f.repack || [];  // DO-80
  wgslCache.enabled = !f.no_wgsl_memo;  // WS-33 arms
  const t0 = performance.now();
  await s.load(get, log);
  const loadMs = performance.now() - t0;
  gpu.hostCache = f.host_cache !== false;
  gpu.fuse = f.fuse !== false;
  // WS-31: the device's copy bandwidth (read + write of a `copy_mb` buffer, GPU time), the roof for GB/s.
  const copy = f.copy_mb ? await copyBandwidth(gpu, f.copy_mb) : null;
  if (copy) await log(`copy ${copy.gbs.map((x) => x.toFixed(1)).join(" ")} GB/s`);
  let at = 0;
  for (const n of f.chunks) { await s.run({ ids: f.tokens.slice(at, at + n), tokens: n, slot: 0 }); at += n; }
  let token = await s.next();
  const variants = [];
  for (const v of f.variants) {
    if (v.check) {  // DO-69: T conversations decoded together against one alone, all from the same fresh prompt
      variants.push(await batchCheck(gpu, s, f, v));
      continue;
    }
    // WS-31: `context` and `slot`: the variant starts in its own slot, prefilled to `context` positions (the
    // prompt's tokens repeated), so every variant decodes from a known position.
    const slot = v.slot ?? 0;
    if (v.context) {
      gpu.profiling = false;
      for (let at = 0; at < v.context; at += f.max_tokens) {
        const n = Math.min(f.max_tokens, v.context - at);
        const ids = Array.from({ length: n }, (_, i) => f.tokens[(at + i) % f.tokens.length]);
        await s.run({ ids, tokens: n, slot });
      }
      token = await s.next();
    }
    gpu.matvec = v.matvec || f.matvec || undefined;  // WS-34: the fixture's kernel settings for every variant
    gpu.matvecC = f.matvec_c || undefined;
    gpu.matvecB = f.matvec_b || undefined;
    gpu.groupMatmul = !!f.group_matmul;  // WS-36
    gpu.normFold = !!f.norm_fold;  // WS-49
    gpu.batchKernel = f.batch_kernel || undefined;  // WS-45
    gpu.fastQ8 = !!f.q8_fast;  // WS-46 (opt-in)
    gpu.q8FastUpTo = f.q8_fast_up_to || 0;  // WS-47
    if (f.gemm) gpu.gemm = f.gemm;
    gpu.attentionSplit = f.attention_split || undefined;
    gpu.profiling = v.profile ?? true;
    const T = v.batch || 1, steps = [], tokens = [];
    if (v.resident) {  // WS-32: the token stays on the GPU; a step's time is the gap between tokens read back
      let last = performance.now();
      const got = await residentDecode(s, gpu, token, v.tokens, slot, () => false, () => {
        const now = performance.now(); steps.push({ ms: now - last, profile: [] }); last = now;
      });
      tokens.push(...got);
      token = got[got.length - 1];
      gpu.takeProfile();
    }
    for (let i = 0; i < (v.resident ? 0 : v.tokens); i++) {
      gpu.takeProfile();
      gpu.host = { submit: 0, wait: 0, map: 0, submits: 0, maps: 0 };  // DO-73
      const start = performance.now();
      if (T === 1) { await s.run({ ids: [token], tokens: 1, slot }); token = await s.next(); tokens.push(token); }
      else { await s.runBatch({ slots: [...Array(T).keys()], ids: new Array(T).fill(token) }); await s.nextBatch(T); }
      steps.push({ ms: performance.now() - start, profile: gpu.takeProfile(), host: { ...gpu.host } });
    }
    variants.push({ name: v.name, batch: T, context: v.context ?? null, steps, tokens, ...(v.profile === "detail" ? { sequence: gpu.lastSequence } : {}) });
    await log(`variant ${v.name}: ${steps.map((x) => x.ms.toFixed(1)).join(" ")}`);
  }
  gpu.profiling = false;
  gpu.matvec = undefined;
  return { loadMs, allocated: gpu.allocated, peak: gpu.peak, copy, variants };
}

// DO-69: slots 0..T-1 decode together and slot T alone, each after the same prompt from position 0, `v.tokens`
// steps; the batch rows' tokens must equal the lone conversation's, and the last step's logits are compared.
async function batchCheck(gpu, s, f, v) {
  const T = v.batch, n = f.chunks[0], prompt = f.tokens.slice(0, n);
  gpu.matvec = f.matvec || undefined; gpu.matvecC = f.matvec_c || undefined; gpu.matvecB = f.matvec_b || undefined;
  gpu.groupMatmul = !!f.group_matmul; gpu.normFold = !!f.norm_fold;
  gpu.attentionSplit = f.attention_split || undefined; gpu.profiling = false;
  gpu.batchKernel = f.batch_kernel || undefined; if (f.gemm) gpu.gemm = f.gemm;
  const slots = [...Array(T).keys()];
  for (let slot = 0; slot <= T; slot++) { await s.reset(slot); await s.run({ ids: prompt, tokens: n, slot }); }
  const first = await s.next();
  const single = [];
  let token = first;
  for (let i = 0; i < v.tokens; i++) { await s.run({ ids: [token], tokens: 1, slot: T }); token = await s.next(); single.push(token); }
  const ref = await gpu.read(s.logits, s.vocab * 4);
  const rows = slots.map(() => []);
  let ids = new Array(T).fill(first);
  for (let i = 0; i < v.tokens; i++) {
    await s.runBatch({ slots, ids });
    ids = await s.nextBatch(T);
    ids.forEach((t, r) => rows[r].push(t));
  }
  const got = await gpu.read(s.logits, T * s.vocab * 4);
  let scale = 0;
  for (let j = 0; j < s.vocab; j++) scale = Math.max(scale, Math.abs(ref[j]));
  const maxAbs = slots.map((r) => { let m = 0; for (let j = 0; j < s.vocab; j++) m = Math.max(m, Math.abs(got[r * s.vocab + j] - ref[j])); return m; });
  const equal = rows.map((r) => r.every((t, i) => t === single[i]));
  await log(`check ${T}: rows equal ${equal.filter(Boolean).length}/${T}, last logits max |diff| ${Math.max(...maxAbs)} (largest ${scale})`);
  return { name: v.name, batch: T, check: true, single, rows, tokens_equal: equal, logits_max_abs: maxAbs, logits_scale: scale, steps: [] };
}

// WS-31: GPU time of copying `mb` MiB (vec4<u32> grid-stride), 4 repeats after a warm-up; GB/s counts the bytes
// read and written.
async function copyBandwidth(gpu, mb) {
  const size = mb * 1024 * 1024;
  const src = gpu.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
  const dst = gpu.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
  const code = `@group(0) @binding(0) var<storage, read> a: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<u32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) w: vec3<u32>) {
  let n = arrayLength(&a); let stride = w.x * 256u;
  for (var i = g.x; i < n; i += stride) { b[i] = a[i]; }
}`;
  const gbs = [];
  for (let r = 0; r < 5; r++) {
    gpu.profiling = "detail";
    gpu.begin();
    gpu.dispatch("copy", code, [src, dst], [4096, 1, 1]);
    await gpu.end();
    const ms = gpu.takeProfile()[0]?.gpuMs;
    if (r > 0 && ms) gbs.push(2 * size / ms / 1e6);
  }
  gpu.profiling = false;
  src.destroy(); dst.destroy();
  return { mb, gbs, median: gbs.slice().sort((x, y) => x - y)[gbs.length >> 1] };
}

const modes = { kernels, stage, pipeline, contributor, pack, bench };
  Object.assign(result, await modes[manifest.mode](gpu, manifest));
  result.lost = gpu.lost || null;
} catch (error) {
  result.error = String((error && error.stack) || error);
}
await fetch("/result", { method: "POST", body: JSON.stringify(result) });
