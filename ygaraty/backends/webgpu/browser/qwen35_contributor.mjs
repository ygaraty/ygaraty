import {ChatSlots} from "./chat_slots.mjs";
// DO-36 contributor page: one browser's part of a Qwen3.5-9B session served
// by ygaraty/experiments/qwen35_session.py.
//
// 1. Join: report the adapter and its limits.
// 2. Self-test: the matrix kernels on slices of the model's own weights
//    (every kernel path and block type) and the float16 conversions (the
//    cache's, and this page's wire conversion); the coordinator compares the
//    outputs with NumPy and admits the device only if they agree.
// 3. Load the assigned stage's weights: each file's SHA-256 is checked on
//    download and the verified bytes are kept in the Cache API, so a reload
//    does not download them again.
// 4. Run: hidden states go A -> B -> C -> D through the coordinator (FP16 on
//    the wire), D's greedy token back to A. Prefill is pipelined: a stage
//    sends each chunk on as soon as it is done and starts the next.
// 5. Several conversations (DO-38): every message names its conversation
//    (`conv`), each with its own caches on every stage; a stage takes all its
//    waiting messages at once and, with `batch`, runs the waiting decode
//    tokens of different conversations as one batch.
// 6. DO-40 (`route: "webrtc"`): hidden states go straight to the next stage
//    and the last stage's tokens straight to the first, over WebRTC data
//    channels (qwen35_peer.mjs); the coordinator still gets every token, to
//    record and stop the session. A link that does not open falls back to
//    the coordinator relay. In user sessions (user_webrtc.md) the TURN
//    credentials come from /api/ice/<id>, the first stage stops each chat
//    turn at its own `limit` and reports the tokens it got (`fed`), and the
//    neighbours of a replaced stage link to its new browser (`relink`).
import { downloadPiece } from "./spaces_download.mjs";
import { readSizedResponse, requiredWeightSizes, weightProgress, prefetchWeights } from "./contributor_loading.mjs";
import { Gpu, Stage as QwenStage, F16 } from "./qwen35_gpu.mjs";
import { Stage as DenseStage, isDenseManifest } from "./dense_gpu.mjs";
import { Stage as KolibriStage, isKolibriManifest } from "./kolibri_gpu.mjs";
import {contributorChat} from "./chat_direct.mjs";
import {monitorContributorNetwork} from "./contributor_network.mjs";
import { Peers } from "./qwen35_peer.mjs";
import { Drafter, acceptDrafts } from "./speculative.mjs";
import { residentDecode } from "./inference_plan.mjs";
import { REPACK, calibrateRepack } from "./webgpu_kernels.mjs";
// DO-83: this GPU's repack measurement (null when not asked), reported with `ready`.
let repackChoice = null;

// Stage names: A-Z, then S27, S28, ... (Kolibri-1's plans use up to about 10 stages).
const stageLabel = (stage) => stage < 26 ? String.fromCharCode(65 + stage) : `S${stage + 1}`;
const key = new URLSearchParams(location.hash.slice(1)).get("key") || "";
const BASE = location.pathname.replace(/\/(chat|coordinator)?\/?$/, '');  // '' alone, or /sessions/<job> in the general coordinator
const headers = { "X-Key": key };
const $status = document.getElementById("status");
const $detail = document.getElementById("detail");
const show = (status, detail = "") => {
  if ($status) $status.textContent = status;
  if ($detail) $detail.textContent = detail;
};

// Local participation is distinct from the model session's serving status.
let reconnectRequested=false, connectionGeneration=0, residentStage=null, stageLoading=false;
const connectionEvent = event => console.info('YGARATY_CONNECTION', JSON.stringify({at:Date.now(),event}));
let contributionPhase='idle', lastContributorContact=0;
function renderContribution(){
  const button=document.getElementById('accept'), badge=document.getElementById('contributor-presence');
  const active=['checking','loading','ready','working'].includes(contributionPhase);
  const online=active&&lastContributorContact>0&&Date.now()-lastContributorContact<35000;
  const labels={idle:'Not contributing',starting:'Starting',reconnecting:'Reconnecting · Restoring contribution',checking:'Checking GPU',loading:'Loading model',ready:'Ready for inference',working:'Serving inference',stopped:'Offline · Contribution stopped',finished:'Offline · Contribution finished'};
  if(button){button.disabled=contributionPhase!=='idle';button.textContent=contributionPhase==='idle'?'Accept and contribute':labels[contributionPhase];}
  if(badge){badge.dataset.state=online?'online':active?'reconnecting':contributionPhase;
    badge.textContent=(online?'Online · ':active?'Connecting · ':'')+labels[contributionPhase];}
}
function setContributionPhase(phase){contributionPhase=phase;renderContribution();}
function contributorContact(){lastContributorContact=Date.now();renderContribution();}
setInterval(renderContribution,5000);
window.addEventListener('offline',()=>{lastContributorContact=0;renderContribution();});
window.addEventListener('pagehide',()=>setContributionPhase('stopped'));

// A network failure or a proxy/gateway error is retried for up to two minutes (a laptop's Wi-Fi blinking, a
// tab woken from the background): the mailboxes take acknowledged receives and numbered sends, so a retry
// neither loses nor repeats a message. Errors the coordinator itself reports are not retried.
async function call(path, body, raw = false) {
  const generation=connectionGeneration;
  const options = { headers: { ...headers }, method: body === undefined ? "GET" : "POST" };
  if (body !== undefined) options.body = body instanceof ArrayBuffer || ArrayBuffer.isView(body) || typeof body === "string" ? body : JSON.stringify(body);
  const until = performance.now() + 120000;
  for (let wait = 500; ; wait = Math.min(wait * 2, 8000)) {
    if(generation!==connectionGeneration)throw Error("contributor_rejoin_required");
    let response;
    try {
      response = await fetch(`${BASE}/api/${path}`, {...options,signal:AbortSignal.timeout(45000)});
    } catch (error) {
      if (performance.now() > until) throw error;
      connectionEvent("transport_retry");
      show("Connection interrupted", "Retrying coordinator connection; retaining loaded weights.");
      await new Promise((resolve) => setTimeout(resolve, wait));
      continue;
    }
    if(generation!==connectionGeneration)throw Error('contributor_rejoin_required');
    if ([502, 503, 504].includes(response.status) && performance.now() < until) {
      await new Promise((resolve) => setTimeout(resolve, wait));
      continue;
    }
    if (!response.ok) {
      connectionEvent(`http_${response.status}`);
      const text=await response.text();
      const error=new Error(`${path}: ${response.status} ${text}`);
      beatFailed(error);
      throw error;
    }
    if(/^(heartbeat|ready|recv|poll)\//.test(path))contributorContact();
    return raw ? response : response.json();
  }
}
const get = async (name, expected, onProgress = () => {}) => {
  const response = await fetch(`${BASE}/f/${name}`, { headers });
  if (!response.ok) throw new Error(`fetch ${name}: ${response.status}`);
  if(expected === undefined) return response.arrayBuffer();
  return readSizedResponse(response, expected, onProgress);
};

// --- float16 wire --------------------------------------------------------
// Float16Array where the browser has it, otherwise the same rounding (to
// nearest even, overflow to infinity) in integer arithmetic.
function f16bits(u) {
  const sign = (u >>> 16) & 0x8000, a = u & 0x7fffffff;
  if (a >= 0x7f800000) return sign | (a > 0x7f800000 ? 0x7e00 : 0x7c00);
  if (a >= 0x477ff000) return sign | 0x7c00;
  if (a < 0x38800000) {
    if (a <= 0x33000000) return sign;
    const e = a >>> 23, m = (a & 0x7fffff) | 0x800000, shift = 126 - e;
    let q = m >>> shift;
    const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (q & 1))) q++;
    return sign | q;
  }
  const r = a - 0x38000000;
  let q = r >>> 13;
  const rem = r & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (q & 1))) q++;
  return sign | q;
}
function f16value(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const NATIVE_F16 = typeof Float16Array !== "undefined";
export function toWire(f32) {
  if (NATIVE_F16) return new Uint16Array(new Float16Array(f32).buffer);
  const u = new Uint32Array(f32.buffer, f32.byteOffset, f32.length), out = new Uint16Array(f32.length);
  for (let i = 0; i < u.length; i++) out[i] = f16bits(u[i]);
  return out;
}
export function fromWire(buffer) {
  if (NATIVE_F16) return new Float32Array(new Float16Array(buffer));
  const h = new Uint16Array(buffer), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = f16value(h[i]);
  return out;
}

// UI receives only server-owned assignments and byte counts.
function showAssignment(assignment){
 const element=document.getElementById('assignment');if(!element||!assignment)return;
 const layers=assignment.layers;
 element.textContent=`Stage ${assignment.stage}`+(layers?` · Layers ${layers[0]}–${layers[1]-1}`:'')+
   (assignment.pieces?.length?` · ${assignment.pieces.join(', ')}`:' · Full assigned layers');
}
function showRoster(roster){
 if(!roster)return;
 for(const name of ['active','preparing']){
  const element=document.getElementById('contributors-'+name);
  if(element){element.textContent=`${roster[name]} / ${roster.required}`; if(name==='preparing')element.parentElement.hidden=!(roster.preparing>0);}
 }
}
// Availability is independent of this browser's GPU work and loading progress.
async function refreshAvailability() {
  const availability = document.getElementById('serving-status');
  if (!availability || !key) return;
  try {
    const response = await fetch(`${BASE}/api/status`, { headers });
    if (!response.ok) throw Error('Status unavailable');
    const status = await response.json();
    if (id && status.rejoin_required) beatFailed(new Error('contributor_rejoin_required'));
    if (status.model?.name) {
      for (const id of ['session-model-name', 'model-name']) {
        const element = document.getElementById(id);
        if (element) element.textContent = status.model.name;
      }
    }
    const workers = status.workers ?? status.stages?.length;
    if (workers) for (const id of ['session-workers', 'worker-count']) {
      const element = document.getElementById(id);
      if (element) element.textContent = `${workers} browsers`;
    }
    showRoster(status.roster);
    availability.textContent = status.state === 'running'
      ? 'Model is serving · Ready for chat'
      : status.state === 'waiting'
        ? `Preparing model · ${status.roster?.active ?? 0} of ${workers} contributors ready`
        : `Model is not serving · ${status.state}`;
  } catch {
    availability.textContent = 'Model status unavailable · Reconnecting…';
  } finally {
    setTimeout(refreshAvailability, 5000);
  }
}
refreshAvailability();

// Heartbeats come from a small worker: a browser slows the timers of a background tab (to about once a
// minute after a while), which made the coordinator count a loading browser as gone; a worker's timers are
// not slowed that way. Loading progress travels with them, and is also sent as it changes (at most every 2 s).
let lastProgress = { phase: "joining" }, progressSent = 0, beats = null, ended = false;
const BEAT = `let state = {}, timer = null;
async function beat() {
  if (!state.url) return;
  try {
    const r = await fetch(state.url, { method: "POST", headers: state.headers || { "X-Key": state.key }, signal:AbortSignal.timeout(10000), body: JSON.stringify({ progress: state.progress }) });
    postMessage({ status: r.status, body: r.ok ? await r.json() : await r.text() });
  } catch (error) { postMessage({ error: String(error) }); }
}
onmessage = (e) => {
  Object.assign(state, e.data);
  if (e.data.stop) { clearInterval(timer); timer = null; return; }
  if (e.data.url && !timer) { beat(); timer = setInterval(beat, e.data.ms || 15000); }
  if (e.data.now) beat();
};`;
function startBeats(cid) {
  try {
    beats = new Worker(URL.createObjectURL(new Blob([BEAT], { type: "text/javascript" })));
  } catch {
    beats = null;  // no workers: the page's own timer
    heartbeat = setInterval(() => call(`heartbeat/${cid}`, { progress: lastProgress }).then((r) => showRoster(r.roster)).catch(beatFailed), 15000);
    return;
  }
  beats.onmessage = (e) => {
    const m = e.data;
    if (m.status === 200) {contributorContact();showRoster(m.body.roster);}
    else if (m.status >= 400) beatFailed(new Error(String(m.body)));
    else if (m.error) show("Connection interrupted", `retrying (${m.error})`);
  };
  beats.postMessage({ url: `${location.origin}${BASE}/api/heartbeat/${cid}`, key, headers: {...headers}, progress: lastProgress, ms: 15000 });
}
function stopBeats() { setContributionPhase('stopped');clearInterval(heartbeat); beats?.terminate(); beats=null; }
// Restart recovery never reclaims a replaced reservation or retries terminal sessions.
function beatFailed(error) {
  if (ended) return;
  const text=String(error);
  const restart=text.includes('contributor_rejoin_required');
  if (!restart && !/reservation (has )?ended|session_stopped/i.test(text)) return;
  lastContributorContact=0;
  if (!restart) {
    ended = true;
    stopBeats();
    connectionEvent('reservation_ended');
    show('Contribution ended', 'This session or reservation ended. Return to the invitation page to check availability.');
    return;
  }
  if(!reconnectRequested){connectionGeneration++;connectionEvent('coordinator_rejoin');}
  reconnectRequested=true;
  setContributionPhase('reconnecting');
  show('Coordinator restarted', 'Reconnecting in this page. Loaded weights are retained while the assignment is checked.');
}
function reportProgress(progress, force = false) {
  lastProgress = progress;
  beats?.postMessage({ progress });
  if (!id) return;
  const now = performance.now();
  if (!force && now - progressSent < 2000) return;
  progressSent = now;
  call(`heartbeat/${id}`, { progress }).then((r) => showRoster(r.roster)).catch(beatFailed);
}
function loadingProgress(loaded,total,stats,complete=false){
 reportProgress({ phase: complete ? "loaded" : "loading", loaded, total, cached: stats.cachedBytes }, complete);
 const bar=document.getElementById('weight-progress'),label=document.getElementById('weight-progress-label');
 if(bar){bar.max=Math.max(1,total);bar.value=total === 0 && complete ? 1 : loaded;}
 if(label)label.textContent=`${loaded.toLocaleString()} / ${total.toLocaleString()} bytes loaded · ${stats.cachedBytes.toLocaleString()} from cache`;
}
let heartbeat=null;
window.addEventListener('pagehide',()=>{
 stopBeats();
 if(id)fetch(`${BASE}/api/leave/${id}`,{method:'POST',headers,body:'{}',keepalive:true}).catch(()=>{});
});

// --- self-test -----------------------------------------------------------
async function selftest(gpu, test, id) {
  // The session's matrix-vector kernel choice (WS-16), so that admission checks the kernel the stage will run.
  if (test.matvec) gpu.matvec = test.matvec;
  if (test.matvec_c) gpu.matvecC = test.matvec_c;
  if (test.matvec_b) gpu.matvecB = test.matvec_b;
  if (test.group_matmul) gpu.groupMatmul = true;  // experiments 135/136
  if (test.batch_gemm) gpu.batchGemm = test.batch_gemm;
  if (test.q8_fast_up_to) gpu.q8FastUpTo = test.q8_fast_up_to;
  // DO-83: with the exact repack possible, each K-quant case also runs repacked; admission requires the same
  // bytes as the raw decoder's (the repack must not change a single output).
  const repackCheck = test.repack === "auto" || (Array.isArray(test.repack) && test.repack.length > 0);
  for (const [i, c] of test.cases.entries()) {
    show("Checking this GPU", `kernel ${i + 1} of ${test.cases.length}`);
    const W = { kind: c.kind, rows: c.shape[0], cols: c.shape[1], rowBytes: c.row_bytes, buffer: gpu.upload(await get(c.file)) };
    const x = gpu.upload(await get(c.x));
    const y = gpu.buffer(c.tokens * W.rows * 4);
    const saved = gpu.prefill;
    if (c.kernel) gpu.prefill = c.kernel;  // e.g. "matvec": the batched decode path
    try {
      gpu.begin();
      gpu.matmul(W, x, y, c.tokens);
      await gpu.end();
    } finally { gpu.prefill = saved; }
    const yRaw = await gpu.read(y, c.tokens * W.rows * 4);
    const r = REPACK[c.kind];
    if (repackCheck && r) {
      const P = { kind: r.kind, rows: W.rows, cols: W.cols, rowBytes: (W.cols / 256) * r.block, buffer: gpu.upload(r.fn(await get(c.file), W.rows, W.cols)) };
      const saved2 = gpu.prefill;
      if (c.kernel) gpu.prefill = c.kernel;
      try { gpu.begin(); gpu.matmul(P, x, y, c.tokens); await gpu.end(); } finally { gpu.prefill = saved2; }
      const yPacked = new Uint8Array(await gpu.read(y, c.tokens * W.rows * 4));
      gpu.release(P.buffer);
      const a = new Uint8Array(yRaw);
      if (a.length !== yPacked.length || a.some((v, k) => v !== yPacked[k])) throw new Error(`repacked ${c.kind} differs from the raw decoder (kernel ${i})`);
    }
    await call(`blob/${id}/y${i}`, yRaw);
    for (const b of [W.buffer, x, y]) gpu.release(b);
  }
  show("Checking this GPU", "float16 conversion");
  const x = new Float32Array(await get(test.f16));
  const n = x.length;
  const code = `
    @group(0) @binding(0) var<storage, read> x: array<f32>;
    @group(0) @binding(1) var<storage, read_write> z: array<u32>;
    @group(0) @binding(2) var<storage, read_write> back: array<f32>;
    ${F16}
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i < ${n / 2}u) {
        let u = f16_bits(x[2u * i]) | (f16_bits(x[2u * i + 1u]) << 16u);
        z[i] = u;
        let v = f16x2(u);
        back[2u * i] = v.x; back[2u * i + 1u] = v.y;
      }
    }`;
  const xb = gpu.upload(x), zb = gpu.buffer(n * 2), bb = gpu.buffer(n * 4);
  gpu.begin();
  gpu.dispatch("selftest_f16", code, [xb, zb, bb], [Math.ceil(n / 2 / 256), 1, 1]);
  await gpu.end();
  await call(`blob/${id}/bits`, await gpu.read(zb, n * 2));
  await call(`blob/${id}/back`, await gpu.read(bb, n * 4));
  await call(`blob/${id}/wire`, toWire(x));
  await call(`blob/${id}/unwire`, fromWire(toWire(x).buffer));
  for (const b of [xb, zb, bb]) gpu.release(b);
}

// --- weights -------------------------------------------------------------
async function sha256(buffer) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", buffer));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function weights(hashes, sizes, stats, progress, delivery = 'local') {
  let cache = null;
  const open = async () => {
    if (cache === null) {
      cache = Promise.resolve().then(() => caches.open("ygaraty-qwen35")).catch(() => false);
    }
    return cache;
  };
  return async (name, signal) => {
    signal?.throwIfAborted();
    const want = hashes[name];
    const c = await open();
    const request = `${location.origin}/w/${want}`;
    if (c) {
      const hit = await c.match(request);
      if (hit) {
        const data = await hit.arrayBuffer();
        if(data.byteLength===sizes[name] && await sha256(data)===want){
          signal?.throwIfAborted();
          stats.cachedBytes += data.byteLength;progress(name,data.byteLength,true);return data;
        }
        await c.delete(request);
      }
    }
    const data = delivery === 'spaces'
      ? await downloadPiece(() => call(`download/${encodeURIComponent(name)}`), sizes[name], want, bytes=>progress(name,bytes), fetch, {signal})
      : await get(name, sizes[name], bytes=>progress(name,bytes));
    signal?.throwIfAborted();
    const got = await sha256(data);
    if (got !== want) throw new Error(`${name}: SHA-256 ${got}, expected ${want}`);
    signal?.throwIfAborted();
    stats.downloadedBytes += data.byteLength;progress(name,data.byteLength,true);
    if (c) {
      try { await c.put(request, new Response(data)); } catch (error) { stats.cacheErrors.push(String(error)); }
    }
    return data;
  };
}

// --- the session ---------------------------------------------------------
const transportCall=call;
async function session(gpu, id, a) {
  let sessionRunning=true;
  const sessionGeneration=connectionGeneration;
  const call=(...args)=>{
    if(!sessionRunning || sessionGeneration!==connectionGeneration)
      return Promise.reject(Error('contributor_rejoin_required'));
    return transportCall(...args);
  };
  try { sessionStorage.setItem("ygaraty-stage", String(a.stage)); } catch {}  // asked for again on a rejoin
  const f = a.fixture;
  let peers = null, chatEdge = null, lastNetworkReport = -Infinity, reportingNetwork = false;
  const network = f.route === 'webrtc' ? monitorContributorNetwork({
    element: document.getElementById('contributor-network'), stage: a.stage, stages: f.stages,
    next: f.plan_route?.next, getWorkers: () => peers, getChat: () => chatEdge,
    onSample: report => {
      if (reportingNetwork || performance.now() - lastNetworkReport < 5000) return;
      lastNetworkReport = performance.now(); reportingNetwork = true;
      call(`route/${id}`, report).catch(() => {}).finally(() => { reportingNetwork = false; });
    },
  }) : null;
  const stopNetwork = () => network?.stop();
  window.addEventListener('pagehide', stopNetwork, {once:true});
  try {
  const stats = { downloadedBytes: 0, cachedBytes: 0, cacheErrors: [] };
  const conversations = f.conversations || 1;
  // The stage implementation of the checkpoint's model family (dense_gpu.mjs and kolibri_gpu.mjs run on the
  // same Gpu).
  const Stage = isDenseManifest(a.manifest) ? DenseStage : isKolibriManifest(a.manifest) ? KolibriStage : QwenStage;
  // DO-25: `draft` {n, k}: n-gram drafts checked k at a time (one conversation at a time, Qwen3.5 stages).
  const draft = f.draft || null;
  // WS-16: the matrix-vector kernel's rows per thread by token count ({single, rows, tokens}; experiment 104).
  if (f.matvec) gpu.matvec = f.matvec;
  // Experiments 121, 125, 127: the coalesced one-token kernel, its batched form, the GEMM for big batches.
  if (f.matvec_c) gpu.matvecC = f.matvec_c;
  if (f.matvec_b) gpu.matvecB = f.matvec_b;
  if (f.group_matmul) gpu.groupMatmul = true;  // experiments 135/136
  if (f.batch_gemm) gpu.batchGemm = f.batch_gemm;
  if (f.q8_fast_up_to) gpu.q8FastUpTo = f.q8_fast_up_to;  // experiment 132
  // Experiments 141/142, DO-83: the exact repack where it pays on this GPU ("auto": measured here), or as listed.
  if (f.repack === "auto") {
    show("Checking this GPU", "repacked weights: measuring");
    repackChoice = await calibrateRepack(gpu);
    gpu.repack = repackChoice.repack;
    console.log("repack calibration", JSON.stringify(repackChoice));
  } else if (Array.isArray(f.repack)) gpu.repack = f.repack;
  if (draft && (Stage !== QwenStage || f.batch)) throw new Error("speculative decoding needs Qwen3.5 stages and no --batch");
  const fingerprint=JSON.stringify([a.stage,a.manifest,a.hashes,a.file_bytes,f]);
  const reuse=residentStage && residentStage.fingerprint===fingerprint && !gpu.lost;
  if(residentStage && !reuse)throw Error('Assignment changed; reload is required before loading a different stage.');
  stageLoading=!reuse;
  const s = reuse ? residentStage.stage : new Stage(gpu, a.manifest, f.max_tokens, f.max_positions, { kv: f.kv, slots: conversations, ...(draft ? { rows: draft.k + 1 } : {}) });
  if (!s.plan || s.plan.owner !== a.stage) throw Error('Compiled plan owner differs from assignment');
  if (!f.plan_route || !Number.isInteger(f.plan_route.next) || f.plan_route.next < 0 || f.plan_route.next >= f.stages ||
      f.plan_route.output !== (a.manifest.head ? 'token' : 'hidden')) throw Error('Invalid compiled transfer route');
  const t0 = performance.now();
  let files = 0;
  showAssignment(a.assignment);
  const totalBytes=a.total_bytes;
  if(!Number.isSafeInteger(totalBytes) || totalBytes<0 || !a.file_bytes)throw Error('Server did not provide exact weight sizes');
  if(Object.keys(a.hashes).reduce((n,key)=>n+a.file_bytes[key],0)!==totalBytes)throw Error('Inconsistent weight manifest sizes');
  const requiredSizes = requiredWeightSizes(a.manifest, a.file_bytes);
  const total = Object.keys(requiredSizes).length;
  const loadingTracker = weightProgress(requiredSizes, state => loadingProgress(state.loaded,state.total,stats,state.complete));
  const fetchOne = weights(a.hashes, a.file_bytes, stats, (name,bytes,verified)=>loadingTracker.update(name,bytes,verified), a.delivery);
  // DO-62 arms: `?prefetch=off` loads one piece at a time (the earlier behavior); `?prefetch=N` runs N speculative pieces.
  const prefetchParam = new URLSearchParams(location.search).get("prefetch") ?? new URLSearchParams(location.hash.slice(1)).get("prefetch");
  const prefetchLanes = /^[1-8]$/.test(prefetchParam ?? "") ? Number(prefetchParam) : 3;
  const prefetch = !reuse && a.delivery === 'spaces' && prefetchParam !== "off" ? prefetchWeights(requiredSizes, fetchOne, {concurrency: prefetchLanes}) : null;
  connectionEvent(`weights_prefetch_${prefetch ? prefetchLanes : "off"}`);
  setContributionPhase('loading');
  try { if(reuse){await s.reset();
    for(const [name,bytes] of Object.entries(requiredSizes))loadingTracker.update(name,bytes,true);
    connectionEvent("weights_reused");}
  else await s.load(async (name) => {
    files++;
    show(`Loading stage ${stageLabel(a.stage)}`, `${files} of ${total} files, ${((stats.downloadedBytes + stats.cachedBytes) / 1e9).toFixed(2)} GB`);
    return prefetch ? prefetch.get(name) : fetchOne(name);
  }); } finally { prefetch?.close(); }
  stageLoading=false;
  residentStage={stage:s,fingerprint};
  loadingTracker.finish();
  const loadMs = performance.now() - t0;
  await call(`ready/${id}`, { loadMs, allocated: gpu.allocated, peak: gpu.peak, ...stats, ...(repackChoice ? { repack: repackChoice } : {}) });
  setContributionPhase('ready');
  reportProgress({ ...lastProgress, phase: "ready" }, true);
  show(`Stage ${stageLabel(a.stage)} ready`, "waiting for the other contributors");

  const calls = [];
  let sending = Promise.resolve(), sent = 0, seq = 0, epoch = 0;
  const send = (kind, meta, body) => {
    const n = sent++;  // numbered, so a retried send is taken once
    meta = { epoch, ...meta };  // the coordinator drops what was sent before a restart's reset; a resend keeps its own
    sending = sending.then(() => call(`send/${id}?kind=${kind}&n=${n}&meta=${encodeURIComponent(JSON.stringify(meta))}`, body));
    sending.catch(()=>{}); // final awaits still propagate errors; detached sends are observed
    return sending;
  };
  // All waiting messages at once: [{ kind, meta, body }].
  const httpRecv = async () => {
    for (;;) {
      const response = await call(`recv/${id}?all=1&ack=${seq}`, undefined, true);
      if (response.status === 204) continue;
      const kind = response.headers.get("X-Kind"), meta = JSON.parse(response.headers.get("X-Meta") || "{}");
      const body = await response.arrayBuffer();
      if (response.headers.get("X-Seq")) seq = Number(response.headers.get("X-Seq"));  // acknowledged next time
      if (kind !== "bundle") return [{ kind, meta, body }];
      let at = 0;
      return meta.map(([k, m, n]) => { const part = body.slice(at, at + n); at += n; return { kind: k, meta: m, body: part }; });
    }
  };
  // DO-40: with WebRTC the coordinator's mailbox is read all the time in the background and merged with the
  // messages from the neighbouring stages; `recv` takes whatever has arrived from either.
  const webrtc = f.route === "webrtc" && (f.stages || 1) > 1 && typeof RTCPeerConnection !== "undefined";
  // DO-51: direct chat needs no link between stages, so a one-stage session (first and last stage at once)
  // has the chat link alone: prompts in and tokens out over it, its own tokens back through the relay.
  const chatDirect = f.chat_direct && f.route === "webrtc" && (a.stage === 0 || a.stage === f.stages - 1)
    && typeof RTCPeerConnection !== "undefined";
  const next = f.plan_route.next, last = f.plan_route.output === "token";
  // DO-61: a one-stage session (first and last stage at once) feeds its own token to its next step and only
  // reports it to the coordinator (`direct`: recorded, not sent back), so no decode step waits for a round
  // trip. It stops each conversation by the coordinator's rule (`stop_ids`, the turn's `limit`), as the first
  // stage of a direct session does. Fixture `local_loop: false`, or `?loop=relay` on the page, turns it off.
  const loopParam = new URLSearchParams(location.search).get("loop") ?? new URLSearchParams(location.hash.slice(1)).get("loop");
  const selfFeed = a.stage === 0 && last && f.local_loop !== false && loopParam !== "relay"
    && Array.isArray(f.stop_ids) && Number.isSafeInteger(f.generate);
  // Experiments 120 and 129: a self-fed single conversation runs the rest of a turn with the token kept on the
  // GPU (read back while the next step runs). Fixture `resident: false`, or `?resident=off`, turns it off.
  // DO-87: with batched chat slots too, while one conversation is the only one decoding (nothing else waiting);
  // any message arriving ends the burst and the usual (batched) loop takes over.
  const residentParam = new URLSearchParams(location.search).get("resident") ?? new URLSearchParams(location.hash.slice(1)).get("resident");
  const resident = selfFeed && f.resident !== false && residentParam !== "off" && !draft && !f.digests
    && (conversations === 1 || f.batch) && !!a.manifest.embed && (s.residentDecode || (typeof s.snapshot === "function" && typeof s.restore === "function"));
  // DO-85: the self-fed tokens' reports wait behind one another on the serial send chain, so the coordinator (and
  // the user's stream) got one token per round trip. Reports that queue behind an in-flight one go out together
  // as one `tokens` report. Fixture `report_batch: false`, or `?reports=serial`, sends them one by one again.
  const reportsParam = new URLSearchParams(location.search).get("reports") ?? new URLSearchParams(location.hash.slice(1)).get("reports");
  const batchReports = f.report_batch !== false && reportsParam !== "serial";
  const reportQueue = [];
  let reportBusy = null;  // the pump's promise while reports are in flight
  const reportDirect = (items) => {
    reportQueue.push(...items);
    if (reportBusy) return;
    reportBusy = (async () => {
      while (reportQueue.length) {
        const e = reportQueue[0].epoch;
        let n = 0;  // one epoch per report, and a bounded URL (meta travels in the query)
        while (n < reportQueue.length && n < 24 && reportQueue[n].epoch === e) n++;
        const group = reportQueue.splice(0, n);
        const sent = group.length === 1 ? send("token", { ...group[0], direct: true }, "")
                                         : send("tokens", { epoch: e, direct: true, items: group }, "");
        await sent.catch(() => {});
      }
      reportBusy = null;
    })();
  };
  const reportsIdle = async () => { while (reportBusy) await reportBusy; };
  let recv = httpRecv, routeStats = null, chatStats = null, connecting = null, feedSelf = null;
  const arrived = [];
  let wake = null;
  const push = (items) => { arrived.push(...items); if (wake) { wake(); wake = null; } };
  // Messages from a peer are checked when their turn comes (`admit`), so the epoch and the first stage's
  // stop rule are the current ones: a peer's message of a later epoch than this stage has reached waits for
  // the reset (`held`), one of an earlier epoch is dropped.
  const held = [];
  const chatSlots = f.concurrent_chat ? new ChatSlots(conversations) : null;
  const counts = new Map(), turnLimits = new Map(), fed = new Map();  // the first stage, per conversation
  let turn = null;
  let chatMeta = null;
  const slotMeta = new Map(), generations = new Map();
  const ended = new Set();  // conversations whose last token the first stage has had
  const drafters = new Map();  // DO-25, first stage: per conversation
  const flushFed = (conv) => {
    const items = fed.get(conv);
    if (items?.length) { send("fed", chatSlots ? chatSlots.stamp({conv, turn:chatSlots.turns[conv], items:items.splice(0)}) : { conv, turn, ...slotMeta.get(conv), items: items.splice(0) }, ""); }
  };
  let lastPseq = null;  // [pseq, psub] of the last message taken from the previous stage, this epoch
  const admit = (m) => {
    const e = m.meta.epoch ?? epoch;
    if (e < epoch) return null;
    if (e > epoch && m.kind !== 'reset') { held.push(m); return null; }
    if (Number.isSafeInteger(m.meta.generation) && m.kind !== 'reset') {
      const generation = generations.get(m.meta.conv ?? 0) ?? 0;
      if (m.meta.generation < generation) return null;
      if (m.meta.generation > generation) {
        if (held.length >= 4096) throw new Error('Too many messages awaiting a conversation reset');
        held.push(m); return null;
      }
    }
    if (chatSlots && ['prompt','token','hidden'].includes(m.kind) && !chatSlots.admit(m, held)) return null;
    if (chatSlots && m.kind === 'reset' && m.meta.conv !== undefined && !chatSlots.admit(m, held)) return null;
    if (Number.isSafeInteger(m.meta.pseq)) {  // a relayed copy of a message the link did deliver: dropped
      const at = [m.meta.pseq, m.meta.psub ?? 0];
      if (lastPseq && (at[0] < lastPseq[0] || (at[0] === lastPseq[0] && at[1] <= lastPseq[1]))) return null;
      lastPseq = at;
    }
    // The first stage counts every token it gets, straight or relayed (the coordinator never relays a
    // conversation's last one), and stops a conversation by the coordinator's rule.
    if ((webrtc || selfFeed) && m.kind === "token") {
      const conv = m.meta.conv ?? 0, n = (counts.get(conv) || 0) + 1;
      if (ended.has(conv)) return null;  // DO-25: the rest of a checked group after the conversation's last token
      counts.set(conv, n);
      const end = n >= (turnLimits.get(conv) ?? f.generate) || f.stop_ids.includes(m.meta.token);
      if (webrtc) {  // a one-stage session's own tokens need no check against another stage's
        if (!fed.has(conv)) fed.set(conv, []);
        fed.get(conv).push(m.meta.token);  // reported to the coordinator, which checks them against the last stage's
        if (end || fed.get(conv).length >= 32) flushFed(conv);
      }
      if (end) { ended.add(conv); return null; }  // the conversation's last token: not run
    }
    return m;
  };
  if (webrtc || chatDirect || selfFeed) {
    const fromPeer = (m) => {
      if (m.kind === "hiddens" || m.kind === "tokens") {  // a batch: one message per conversation
        const items = Array.isArray(m.meta.items) ? m.meta.items : [];
        if (!items.length || (m.kind === "hiddens" && m.body.byteLength % items.length)) return [];
        const size = m.kind === "hiddens" ? m.body.byteLength / items.length : 0;
        return items.flatMap((item, i) => fromPeer({ kind: m.kind.slice(0, -1), from: m.from,
          meta: m.kind === "tokens" ? { ...item, epoch: m.meta.epoch, pseq: m.meta.pseq, psub: i }
                                    : { ...item, chat_lease:item.chat_lease ?? m.meta.chat_lease, chat_turn:item.chat_turn ?? m.meta.chat_turn, epoch: m.meta.epoch, pseq: m.meta.pseq, psub: i },
          body: m.body.slice(i * size, (i + 1) * size) }));
      }
      return [{ kind: m.kind, meta: m.meta, body: m.body, peer: true }];
    };
    // The coordinator's short-lived TURN credentials, fetched for every new link; an experiment's fixture
    // may carry its own list instead.
    const ice = Array.isArray(f.ice) ? f.ice : async () => (await call(`ice/${id}`)).ice;
    if (selfFeed) feedSelf = (kind, meta) => push(fromPeer({ kind, meta, body: new ArrayBuffer(0) }));
    if (chatDirect) {
      chatEdge = await contributorChat({stage:a.stage,stages:f.stages,capacity:f.max_positions,
        ice,policy:f.ice_policy,api:body=>call('chat-direct',body),onPrompt:m=>push([m])});
      window.addEventListener('pagehide',()=>chatEdge?.close(),{once:true});
    }
    if (webrtc) {
      const wireBytes = f.wire === "f16" ? 2 : 4;
      peers = new Peers({ id, stage: a.stage, stages: f.stages, ice, policy: f.ice_policy, call,
                          limits: { hidden: Math.max(f.max_tokens, conversations) * s.hidden * wireBytes },
                          onMessage: (m) => push(fromPeer(m)) });
      // Signaling starts now; the session waits for the links after go. A link that cannot be made (no ICE
      // servers, say) leaves that pair on the relay.
      connecting = peers.connect().catch((error) => show(`Stage ${stageLabel(a.stage)}`, `direct links unavailable: ${error}`));
    }
    (async () => {
      while (sessionRunning) {
        const items = await httpRecv();
        if(!sessionRunning)return;
        push(items);
        if (items.some((m) => m.kind === "stop")) return;
      }
    })().catch((error) => push([{ kind: "failed", meta: { error: String(error) }, body: null }]));
    recv = async () => {
      while (!arrived.length) await new Promise((resolve) => { wake = resolve; });
      const items = arrived.splice(0), failed = items.find((m) => m.kind === "failed");
      if (failed) throw new Error(failed.meta.error);
      return items;
    };
  }
  network?.ready();
  const peerBytes = { sent: 0 };
  // Hidden states to the next stage, tokens to the first: directly when the link is open.
  // A message the link could not deliver (closed before it was sent, or before the peer acknowledged it) goes
  // through the relay with its sequence number (`pseq`, so the receiver drops a copy it already has) and the
  // epoch it was made in.
  const handOn = (kind, meta, body) => {
    const kept = epoch;
    meta = meta.items
      ? {...meta, items:meta.items.map(item=>({...item,...slotMeta.get(item.conv ?? 0)}))}
      : {...meta,...(slotMeta.get(meta.conv ?? 0) ?? chatMeta)};
    if (chatSlots) {
      meta = meta.items ? {...meta, items:meta.items.map(item=>chatSlots.stamp(item))} : chatSlots.stamp(meta);
    }
    if (last && chatEdge) {
      const items = kind === 'tokens' ? meta.items : kind === 'token' ? [meta] : [];
      for (const item of items) chatEdge.token({...item,epoch:kept},item.token);
    }
    if (feedSelf && (kind === "token" || kind === "tokens")) {  // DO-61: run next here, reported in the background
      feedSelf(kind, { ...meta, epoch: kept });
      if (!batchReports) { send(kind, { ...meta, epoch: kept, direct: true }, ""); return; }
      const { items, ...top } = meta;
      reportDirect(kind === "tokens" ? items.map((item) => ({ ...top, ...item, epoch: kept })) : [{ ...meta, epoch: kept }]);
      return;
    }
    if (webrtc && !last && peers.has(next)) {
      peerBytes.sent += body.byteLength;
      peers.send(next, kind, { ...meta, epoch: kept }, body, (pseq) => send(kind, { ...meta, epoch: kept, pseq }, body));
      return;
    }
    if (webrtc && last && peers.has(0)) {
      // Recorded by the coordinator, not forwarded (`direct`). Not sent on the link: relayed as usual. Sent but
      // never acknowledged: already recorded, so only forwarded to the first stage (`resend`).
      peers.send(0, kind, { ...meta, epoch: kept }, "", (pseq, resent) => {
        if (resent) send(kind, { ...meta, epoch: kept, pseq, resend: true }, "");
        return false;
      }).then((direct) => send(kind, direct === true ? { ...meta, direct: true } : meta, ""));
      return;
    }
    send(kind, meta, body);
  };
  const stageName = stageLabel(a.stage);
  let index = 0;
  const progress = (emit) => {
    if (index % 16 === 0 || emit) {setContributionPhase("working");show(`Stage ${stageName} running`, `${index} calls, positions ${s.positions.join(", ")}`);}
  };
  // `emit`: the call's last position is the end of a prompt or a generated
  // token, so D sends its greedy token back; the flag travels with the
  // hidden states from A to D.
  const forward = async (input, n, waitMs, emit, conv) => {
    const start = performance.now();
    let digests;
    await s.run({ ...input, tokens: n, slot: conv });
    const computeMs = performance.now() - start;
    if (s.output) {
      if (emit) {
        const token = await s.next();
        handOn("token", { token, conv, call: index }, "");
        digests = await logitDigests(1);
      }
    } else {
      const out = await gpu.read(s.x, n * s.hidden * 4);
      handOn("hidden", { call: index, tokens: n, emit, conv }, toWire(out));
    }
    calls.push({ call: index, conv, tokens: n, waitMs, computeMs, ms: performance.now() - start, ...(digests ? { digests } : {}) });
    index++;
    progress(emit);
  };
  // Experiments 120 and 129: the rest of a self-fed turn from `first` (already admitted and counted, and in
  // s.token from the step that produced it), counted and reported token by token by admit's rule. Anything
  // arriving meanwhile (a stop, a reset, a new prompt) ends the burst: the newest token goes back through the
  // usual loop, which runs it after those messages.
  // DO-87: a burst's reports go through DO-85's batching too, or the stream is one token per round trip again.
  const reportResident = (meta) => batchReports ? reportDirect([meta]) : send("token", { ...meta, direct: true }, "");
  const forwardResident = async (first, conv, waitMs) => {
    const limit = turnLimits.get(conv) ?? f.generate;
    const budget = limit - (counts.get(conv) || 0);
    let started = false, handedBack = false, at = performance.now();
    const stop = (token) => {
      if (!started) { started = true; return false; }  // `first`, already admitted
      const now = performance.now();
      calls.push({ call: index, conv, tokens: 1, resident: true, waitMs, computeMs: now - at, ms: now - at });
      waitMs = 0; at = now; index++;
      if (arrived.length) {  // reported as usual, and queued ahead of what arrived (the order the usual loop has)
        handedBack = true;
        const meta = { token, conv, call: index, ...(slotMeta.get(conv) ?? chatMeta), epoch };
        if (last && chatEdge) chatEdge.token(meta, token);
        reportResident(meta);
        arrived.unshift({ kind: "token", meta, body: new ArrayBuffer(0), peer: true });
        return true;
      }
      const n = (counts.get(conv) || 0) + 1;
      counts.set(conv, n);
      const end = n >= limit || f.stop_ids.includes(token);
      const meta = { token, conv, call: index, ...(slotMeta.get(conv) ?? chatMeta), epoch };
      if (last && chatEdge) chatEdge.token(meta, token);
      reportResident(meta);
      if (end) ended.add(conv);
      progress(end);
      return end;
    };
    if (budget <= 0) return false;
    gpu.device.queue.writeBuffer(s.token, 0, new Uint32Array([first]));  // s.token may hold a step undone since
    await residentDecode(s, gpu, first, budget, conv, stop);
    return handedBack;
  };
  // DO-25: a checking pass. `spec`: {r, drafts}; the input's first r rows rerun positions of the previous pass
  // that were kept (its states restored first: some of its drafts were rejected), the others are the last
  // token and the drafts. The last stage sends the kept tokens.
  const forwardSpec = async (input, n, spec, waitMs, conv) => {
    const start = performance.now();
    const rows = (from, to) => input.ids ? { ids: input.ids.slice(from, to) }
                                         : { hidden: input.hidden.slice(from * s.hidden, to * s.hidden) };
    // After a rejection: back to the state saved before the rejected pass, whose kept rows lead this pass
    // (WS-20: one run, not a replay run and then the pass). Otherwise a pass with drafts saves the state first.
    if (spec.r > 0) s.restore(conv);
    else if (spec.drafts.length) s.snapshot(conv);
    await s.run({ ...rows(0, n), tokens: n, slot: conv, heads: "all", exact: draft.exact !== false, gemm: draft.exact === false });
    const computeMs = performance.now() - start;
    let kept = null, digests;
    if (s.output) {
      const out = (await s.nextBatch(n)).slice(spec.r);
      const tokens = acceptDrafts(out, spec.drafts, f.stop_ids);
      kept = tokens.length - 1;
      handOn("tokens", { items: tokens.map((token) => ({ conv, token })), call: index }, "");
      digests = await logitDigests(tokens.length, spec.r);
    } else {
      handOn("hidden", { call: index, tokens: n, emit: true, conv, spec }, toWire(await gpu.read(s.x, n * s.hidden * 4)));
    }
    calls.push({ call: index, conv, tokens: n, replay: spec.r, drafts: spec.drafts.length, kept, waitMs, computeMs, ms: performance.now() - start,
                 ...(digests ? { digests } : {}) });
    index++;
    progress(true);
  };
  // DO-25, first stage: a token back from the last stage; a checking pass once its group is complete.
  const draftToken = async (token, conv, waitMs) => {
    const d = drafters.get(conv);
    // A pass gives at most 1 + drafts tokens: none past the conversation's limit, no position past the cache.
    const generated = d.history.length - d.prompt + 1;  // with this token
    const room = Math.min((turnLimits.get(conv) ?? f.generate) - generated - 1, f.max_positions - d.history.length - 1);
    const next = d.take(token, room, performance.now());
    if (!next) return;
    const ids = [...next.replay, next.token, ...next.drafts];
    await forwardSpec({ ids }, ids.length, { r: next.replay.length, drafts: next.drafts }, waitMs, conv);
  };
  // DO-25 check (fixture `digests`): SHA-256 of the logits rows of the tokens a call emits, so that a plain and
  // a speculative run can be compared bit for bit.
  const logitDigests = async (rows, from = 0) => {  // `from`: rows before the emitted ones (a pass's replay)
    if (!f.digests || !rows) return undefined;
    const all = await gpu.read(s.logits, (from + rows) * s.vocab * 4);
    return Promise.all(Array.from({ length: rows }, (_, i) => sha256(all.slice((from + i) * s.vocab, (from + i + 1) * s.vocab).buffer)));
  };
  // A checking pass's description from the previous stage, checked before it shapes a run.
  const checkSpec = (spec, n) => {
    const r = spec?.r, drafts = spec?.drafts;
    if (!Number.isSafeInteger(r) || r < 0 || r > draft.k + 1 || !Array.isArray(drafts) || drafts.length > draft.k ||
        !drafts.every((t) => Number.isSafeInteger(t) && t >= 0) || n !== r + 1 + drafts.length || n > draft.k + 1) throw new Error("Invalid checking pass");
    return { r, drafts };
  };

  // One decode token for each of several conversations, as one batch.
  const forwardBatch = async (items, waitMs) => {
    const start = performance.now();
    const slots = items.map((m) => m.conv);
    if (a.manifest.embed) await s.runBatch({ slots, ids: items.map((m) => m.token) });
    else {
      const hidden = new Float32Array(items.length * s.hidden);
      items.forEach((m, t) => hidden.set(m.hidden, t * s.hidden));
      await s.runBatch({ slots, hidden });
    }
    const computeMs = performance.now() - start;
    if (s.output) {
      const tokens = await s.nextBatch(items.length);
      handOn("tokens", { items: items.map((m, t) => ({ conv: m.conv, token: tokens[t] })), call: index }, "");
    } else {
      const out = await gpu.read(s.x, items.length * s.hidden * 4);
      handOn("hiddens", { items: items.map((m) => ({ conv: m.conv, tokens: 1, emit: true })), call: index }, toWire(out));
    }
    calls.push({ call: index, convs: slots, tokens: 1, batch: items.length, waitMs, computeMs, ms: performance.now() - start });
    index++;
    progress(true);
  };
  // A: a prompt in chunks of at most max_tokens; the last chunk emits.
  const prompt = async (tokens, conv) => {
    if (draft) drafters.set(conv, Object.assign(new Drafter(tokens, draft), { prompt: tokens.length }));
    for (let at = 0; at < tokens.length; at += f.max_tokens) {
      const ids = tokens.slice(at, at + f.max_tokens);
      await forward({ ids }, ids.length, 0, at + f.max_tokens >= tokens.length, conv);
    }
  };
  // A decode step: a generated token at A, a single emitting position elsewhere.
  const decodeItem = (m) => {
    const conv = m.meta.conv ?? 0;
    if (m.kind === "token") return { conv, token: m.meta.token };
    if (m.kind === "hidden" && m.meta.tokens === 1 && m.meta.emit) return { conv, hidden: fromWire(m.body) };
    return null;
  };

  let messages = await recv();
  if (messages[0].kind !== "go") throw new Error(`expected go, got ${messages[0].kind}`);
  epoch = messages[0].meta.epoch ?? 0;  // a browser that replaces a lost stage starts in the current epoch
  if (webrtc) {  // every stage is ready, so every neighbour is signaling: wait for the links (or their timeout)
    // At most 30 s from go for everything: the links' set-up (ICE servers, offers), re-offers and opening.
    // A link not open by then is closed and its pair uses the relay (DO-44: a set-up that took 70 s, then a
    // late link, stalled a turn).
    show(`Stage ${stageName} connecting`, "direct links to the neighbouring stages (up to 30 s)");
    const until = performance.now() + 30000, left = () => Math.max(0, until - performance.now());
    const within = (p) => Promise.race([p, new Promise((resolve) => setTimeout(resolve, left()))]);
    peers.arm();
    await within(connecting);
    await within(peers.reoffer().catch(() => {}));  // an offer made before the other stage had a browser was dropped
    routeStats = await peers.settle(left());
    call(`route/${id}`, { links: routeStats }).catch(() => {});
  }
  if (a.manifest.embed) {
    const prompts = f.prompts || (f.tokens && f.tokens.length ? [f.tokens] : []);
    for (const [conv, tokens] of prompts.entries()) await prompt(tokens, conv);
  }
  messages = messages.slice(1);
  let stopped = false;
  try {
  while (!stopped && !reconnectRequested) {
    let waitMs = 0;
    if (!messages.length) {
      const waitStart = performance.now();
      messages = await recv();
      waitMs = performance.now() - waitStart;
    }
    const batch = [];
    for (let i = 0; i < messages.length; i++) {
      // Flush before admission changes slot versions or stop counters.
      const incoming = messages[i];
      if (batch.length && (batch.length >= s.maxBatch ||
          batch.some((b) => b.conv === (incoming.meta.conv ?? 0)) ||
          !decodeItem(incoming))) {
        await forwardBatch(batch.splice(0), waitMs); waitMs = 0;
      }
      const m = admit(incoming);
      if (!m) continue;
      if (m.kind === 'prompt') chatMeta = m.meta.chat_lease ? {chat_lease:m.meta.chat_lease,chat_turn:m.meta.turn} : null;
      if (m.kind === 'hidden') chatMeta = m.meta.chat_lease ? {chat_lease:m.meta.chat_lease,chat_turn:m.meta.chat_turn} : null;
      if (m.kind === 'reset') chatMeta = null;
      const conv = m.meta.conv ?? 0;
      if (m.kind === 'prompt' || m.kind === 'hidden') {
        slotMeta.set(conv, {...chatMeta, ...(Number.isSafeInteger(m.meta.generation) ? {generation:m.meta.generation, turn:m.meta.turn} : {})});
      }
      // DO-87: the only conversation decoding (the last message in hand, no batch waiting) runs a resident burst.
      const solo = resident && m.kind === "token" && m.peer && i === messages.length - 1 && !batch.length;
      const item = f.batch && !solo ? decodeItem(m) : null;
      if (item && !batch.some((b) => b.conv === conv)) { batch.push(item); continue; }
      if (batch.length) { await forwardBatch(batch.splice(0), waitMs); waitMs = 0; }
      if (m.kind === "stop") { if(m.meta?.restarting)beatFailed(new Error("contributor_rejoin_required")); stopped = true; break; }
      if (item) { batch.push(item); continue; }  // a second token of the same conversation
      if (m.kind === "token" && draft) await draftToken(m.meta.token, conv, waitMs);
      // A burst only for the last message in hand: anything after it is handled first, step by step.
      else if (solo) await forwardResident(m.meta.token, conv, waitMs);
      else if (m.kind === "token") await forward({ ids: [m.meta.token] }, 1, waitMs, true, conv);
      else if (m.kind === "prompt") {
        // A chat turn: its own answer length (`limit`) and number, for the first stage's stop rule and reports.
        turn = m.meta.turn ?? turn;
        counts.set(conv, 0); fed.set(conv, []); ended.delete(conv);  // a new turn: its tokens are run again
        turnLimits.set(conv, Number.isSafeInteger(m.meta.limit) ? m.meta.limit : f.generate);
        await prompt([...new Uint32Array(m.body)], conv);
      }
      else if (m.kind === "unlink") peers?.unlink(m.meta.stage);  // its browser left
      else if (m.kind === "relink") peers?.relink(m.meta.stage).catch(() => {});  // a new browser holds it
      else if (m.kind === "hidden" && draft && m.meta.spec) await forwardSpec({ hidden: fromWire(m.body) }, m.meta.tokens, checkSpec(m.meta.spec, m.meta.tokens), waitMs, conv);
      else if (m.kind === "hidden") await forward({ hidden: fromWire(m.body) }, m.meta.tokens, waitMs, m.meta.emit, conv);
      else if (m.kind === "reset") {
        await reportsIdle(); await sending;  // this stage's outputs before the reset go out under the old epoch
        if (Number.isSafeInteger(m.meta.generation)) {
          if (m.meta.generation < (generations.get(conv) ?? 0)) continue;
          generations.set(conv, m.meta.generation);
          counts.delete(conv); fed.delete(conv); ended.delete(conv); drafters.delete(conv);
          slotMeta.delete(conv); turnLimits.delete(conv);
          await s.reset(conv);
          messages.splice(i + 1, 0, ...held.splice(0));
          continue;
        }
        if (chatSlots && m.meta.conv !== undefined) {
          counts.delete(conv); fed.delete(conv); ended.delete(conv); drafters.delete(conv);
          await s.reset(conv);
          messages.splice(i + 1, 0, ...held.splice(0));
          continue;
        }
        chatSlots?.clear();
        epoch = m.meta.epoch ?? epoch;
        generations.clear(); slotMeta.clear();
        counts.clear(); fed.clear(); ended.clear(); drafters.clear(); lastPseq = null;
        messages.splice(i + 1, 0, ...held.splice(0));  // peers' messages that were waiting for this epoch
        await s.reset();
        show(`Stage ${stageName} ready`, m.meta.epoch ? "a contributor left: waiting for a replacement" : "conversation reset");
      }
      else throw new Error(`unexpected message ${m.kind}`);
      waitMs = 0;
    }
    if (!stopped && batch.length) await forwardBatch(batch, waitMs);
    messages = [];
  }
  } finally {
    // DO-51: the chat link's counters (messages, not only bytes: a token's message has no body), taken before
    // it closes; the only link of a one-stage session.
    chatStats = chatEdge ? await chatEdge.stats().catch(() => null) : null;
    chatEdge?.close();
  }
  if(reconnectRequested)throw Error('contributor_rejoin_required');
  await reportsIdle(); await sending;
  if (peers) {  // the round trips measured after the run, then the links closed
    routeStats = { atStart: routeStats, atEnd: await peers.stats(), hiddenBytesSent: peerBytes.sent };
    peers.close();
  }
  if (chatStats?.length) routeStats = { ...(routeStats || {}), chat: chatStats };
  await call(`done/${id}`, { calls, allocated: gpu.allocated, peak: gpu.peak, lost: gpu.lost || null, route: routeStats });
  show(`Stage ${stageName} finished`, `${index} calls; thank you`);
  } finally {
    sessionRunning=false;
    network?.stop();
    window.removeEventListener('pagehide', stopNetwork);
    chatEdge?.close();
    peers?.close();
  }
}

// Nothing is reserved, downloaded or run on the GPU until the visitor accepts. An experiment's own headless
// browser (qwen35_session browser) opens the link with autostart=1, which accepts on its behalf.
async function acceptance() {
  const button = document.getElementById("accept");
  if (new URLSearchParams(location.hash.slice(1)).get("autostart") === "1" || !button) {setContributionPhase("starting");return;}
  show("Waiting for you", "Nothing is downloaded and your GPU is not used until you accept.");
  button.disabled = false;
  await new Promise((resolve) => button.addEventListener("click", () => {setContributionPhase("starting");resolve();}, { once: true }));
  button.disabled = true;
  button.closest("section")?.classList.add("accepted");
  renderContribution();
}

let id = null;
export let contributorError = null;
export let contributorReloadRequired = false;
try {
  await acceptance();
  if (!navigator.gpu) throw new Error(window.isSecureContext ? "WebGPU is not available in this browser" : "WebGPU needs a secure (https) page");
  show("Connecting to the GPU");
  const gpu = await Gpu.create();
  const info = gpu.adapter.info || {};
  const limits = {};
  for (const k of ["maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup", "maxStorageBuffersPerShaderStage"]) limits[k] = gpu.adapter.limits[k];
  gpu.device?.lost.then(()=>connectionEvent('gpu_device_lost'));
  do {
  reconnectRequested=false;
  try {
  const joined = await call("join", {
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    limits, features: [...gpu.adapter.features], userAgent: navigator.userAgent, nativeF16: NATIVE_F16,
    prefer: (() => { try { const v = sessionStorage.getItem("ygaraty-stage"); return v === null ? undefined : Number(v); } catch { return undefined; } })(),
  });
  if(joined.accepted===false){setContributionPhase('stopped');showRoster(joined.roster);show('All contributor slots are filled',joined.reason);}
  else {
  if (joined.worker_key) headers["X-Worker-Key"] = joined.worker_key;
  else delete headers["X-Worker-Key"];
  id = joined.id;contributorContact();setContributionPhase('checking');
  if (!gpu.contributionBudget && Number.isSafeInteger(joined.memory_budget_bytes)) {
    gpu.contributionBudget=joined.memory_budget_bytes;
    const allocate = gpu.buffer.bind(gpu), budget = joined.memory_budget_bytes;
    gpu.buffer = (bytes, usage) => {
      if (gpu.allocated + Math.max(16, Math.ceil(bytes / 16) * 16) > budget)
        throw Error('This stage exceeds your approved memory contribution.');
      return allocate(bytes, usage);
    };
  }
  if(gpu.contributionBudget && joined.memory_budget_bytes!==undefined && joined.memory_budget_bytes!==gpu.contributionBudget)
    throw Error('Memory contribution changed; reload required.');
  showAssignment(joined.assignment);showRoster(joined.roster);
  startBeats(id);
  reportProgress({ phase: "self-test" }, true);
  await selftest(gpu, joined.selftest, id);
  const result = await call(`selftest/${id}`, {});
  if (!result.ok) {
    contributorError = new Error("GPU self-test failed: " + result.reason);
    show("This GPU did not pass the check", result.reason);
  } else if (!result.assignment) {
    show("All contributor slots are filled", "this browser is not needed for this session");
  } else {
    await session(gpu, id, result.assignment);
  }
  if(!ended)stopBeats();
  if(!ended&&result.ok&&result.assignment)setContributionPhase("finished");
  if(!result.ok || !result.assignment)await call(`leave/${id}`,{});
  }
  } catch(error) {
    beatFailed(error);
    if(!reconnectRequested || ended || gpu.lost)throw error;
    if(stageLoading){
      ended=true;stopBeats();setContributionPhase('reconnecting');
      show('Reconnecting', 'Loading was interrupted. Reloading safely with verified cached weights.');
      contributorReloadRequired=true;
      setTimeout(()=>location.reload(),2000);break;
    }
  }
  if(reconnectRequested && !ended && !gpu.lost){
    stopBeats();setContributionPhase('reconnecting');
    await new Promise(resolve=>setTimeout(resolve,2000));
  }
  } while(reconnectRequested && !ended && !gpu.lost);
} catch (error) {
  contributorError = error;
  beatFailed(error);
  if(!ended){
  stopBeats();
  const text = String((error && error.stack) || error);
  show("Stopped", "Reload this page to rejoin (a chat waits for its contributors). " + text);
  try { await call(`error/${id ?? "unjoined"}`, text); } catch {}
  }
}
if(!ended && contributionPhase!=='stopped')setContributionPhase('finished');
