// Kolibri-1 stage (kolibri_gpu.mjs) without a GPU: the compiled plan issues exactly the primitive calls (and
// shaders) of the stage's direct-call reference layer, for prefill chunks, single tokens and batches; the
// manifest checks; the expert kernels' dispatch over chunks of experts.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {validatePlan} from '../../../ygaraty/backends/webgpu/browser/inference_plan.mjs';
import {Stage, parseManifest, expertChunkNames, moeMatmul} from '../../../ygaraty/backends/webgpu/browser/kolibri_gpu.mjs';

const fixture = JSON.parse(readFileSync(new URL('./kolibri-plans.json', import.meta.url), 'utf8'));
const config = fixture.config;

function setup(raw) {
  const manifest = {layers: raw.tasks.filter(t => t.attrs.layer_end).map(t => ({index: t.attrs.layer,
    type: config.sliding[t.attrs.layer] ? 'sliding' : 'full'})),
    embed: raw.input === 'tokens' ? {} : null, head: raw.output === 'next_token' ? {} : null};
  const plan = validatePlan(raw, manifest, 'kolibri1', config.hidden);
  const trace = [], gpu = {prefill: 'tiled', device: {queue: {writeBuffer: () => trace.push('write')}},
    begin: () => trace.push('begin'), end: async () => { trace.push('end'); },
    uniform: values => ({uniform: values}),
    dispatch: (name, shader, ...args) => trace.push([name, createHash('sha256').update(shader).digest('hex'), ...args])};
  for (const name of ['gather', 'norm', 'matmul', 'siluMul', 'argmax']) gpu[name] = (...args) => trace.push([name, ...args]);
  const s = Object.create(Stage.prototype);
  const named = name => ({name,size:2**30});
  const packed = name => ({name,kind:'F32',rows:1,cols:1,rowBytes:4,buffer:named(name)});
  const weights = prefix => new Proxy({}, {get: (_, key) => ({...packed(`${prefix}.${String(key)}`),size:2**30})});
  const chunks = (index, base, rows, cols) => [0, 1].map(k => ({name: `${base}${index}.${k}`, kind: 'Q4_K', rows: rows * 192, cols,
    rowBytes: cols / 256 * 144, buffer: named(`${base}${index}.${k}`), first: 192 * k, count: 192}));
  const layers = manifest.layers.map(L => ({...L, matrices: weights(`L${L.index}`), vectors: weights(`L${L.index}`),
    gateUp: chunks(L.index, 'gu', 1024, 2560), down: chunks(L.index, 'dn', 2560, 512),
    ...Object.fromEntries(['Ks', 'Vs'].map(key => [key, [0, 1].map(slot => named(`${key}${L.index}.${slot}`))]))}));
  Object.assign(s, {gpu, plan, loaded: true, manifest, hidden: config.hidden, layers, config, intermediate: config.intermediate,
    kv: 'f16', vocab: config.vocab, embed: manifest.embed ? [{...packed('embed'),rows:10,first:0}] : undefined,
    output: manifest.head ? [{...packed('head0'),first:0},{...packed('head7'),first:7}] : undefined,
    slots: 2, maxBatch: 2, maxTokens: 128, maxPositions: 1024, positions: [3, 7], position: 3, slot: 0});
  for (const key of ['x', 'h', 'mid', 'q', 'k', 'v', 'qn', 'kn', 'att', 'f1', 'f2', 'router', 'experts', 'expertWeights', 'eg', 'ea', 'ed',
                     'part', 'ml', 'ids', 'invFreq', 'outputNorm', 'logits', 'token']) s[key] = named(key);
  return {s, gpu, trace, manifest};
}

async function legacy(s, tokens, batch = false) {
  const {gpu} = s, per = Math.max(1, Math.floor(64 / tokens));
  const parts = batch ? [{slot: 0, row: 0, count: 1, position: 3}, {slot: 1, row: 1, count: 1, position: 7}]
    : [{slot: 1, row: 0, count: tokens, position: 7}];
  if (!batch) s.use(1); else gpu.prefill = 'matvec';
  gpu.device.queue.writeBuffer();
  for (let k = 0; k < s.layers.length; k += per) {
    gpu.begin();
    if (s.embed && k === 0) gpu.gather(s.embed, s.ids, s.x, tokens);
    for (const L of s.layers.slice(k, k + per)) s.layer(L, tokens, parts);
    if (s.output && k + per >= s.layers.length) { if (batch) s.headBatch(tokens); else s.head(tokens); }
    await gpu.end();
  }
}

for (const raw of fixture.plan.stages) {
  for (const tokens of [1, 9, 64, 128]) test(`Kolibri plan trace equals the reference layer, stage ${raw.owner}, tokens=${tokens}`, async () => {
    const {s, trace, manifest} = setup(raw), old = setup(raw);
    const input = manifest.embed ? {ids: Array(tokens).fill(1)} : {hidden: new Float32Array(tokens * config.hidden)};
    await s.run({...input, tokens, slot: 1});
    await legacy(old.s, tokens);
    assert.deepEqual(trace, old.trace);
    assert.deepEqual(s.positions, [3, 7 + tokens]);
    assert.equal(s.submitLayers.reduce((a, b) => a + b, 0), manifest.layers.length);
  });
  test(`Kolibri batch trace and a failed batch keep state, stage ${raw.owner}`, async () => {
    const {s, gpu, trace, manifest} = setup(raw), old = setup(raw);
    const input = manifest.embed ? {ids: [1, 2]} : {hidden: new Float32Array(2 * config.hidden)};
    await s.runBatch({...input, slots: [0, 1]});
    await legacy(old.s, 2, true);
    assert.deepEqual(trace, old.trace);
    assert.deepEqual(s.positions, [4, 8]);
    assert.equal(gpu.prefill, 'tiled');
    gpu.end = async () => { throw Error('device failed'); };
    await assert.rejects(s.runBatch({...input, slots: [0, 1]}), /device failed/);
    assert.deepEqual(s.positions, [4, 8]);
  });
}

test('global layers skip RoPE and attend to every key; sliding layers rotate and attend within the window', async () => {
  const raw = fixture.plan.stages[0], {s, trace} = setup(raw);
  await s.run({ids: [1, 2, 3], tokens: 3, slot: 0});
  const ropes = trace.filter(t => Array.isArray(t) && t[0].startsWith('dense_rope'));
  assert.equal(ropes.length, 2 * 4);  // layers 0-3 (q and k); layer 4 is global
  const attention = trace.filter(t => Array.isArray(t) && t[0].startsWith('dense_attention_window'));
  assert.deepEqual(attention.map(t => t[2].at(-1).uniform[5][1]), [513, 513, 513, 513, 0]);  // [name, shader, buffers, groups]
});

test('expert products dispatch once per chunk with one workgroup row per token slot', () => {
  const calls = [], gpu = {uniform: values => values, dispatch: (name, code, buffers, groups) => calls.push({name, values: buffers[4], groups})};
  const chunk = first => ({kind: 'Q4_K', rows: 1024 * 77, cols: 2560, rowBytes: 1440, buffer: {}, first, count: 77});
  moeMatmul(gpu, [chunk(0), chunk(77)], {}, {}, {}, 3, 6, false);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0].groups, [256, 18, 1]);
  assert.deepEqual(calls[1].values.slice(0, 7).map(v => v[1]), [1024, 2560, 1440, 77, 77, 6, 0]);
  assert.throws(() => moeMatmul(gpu, [chunk(0)], {}, {}, {}, 11000, 6, true), /dispatch/);
});

test('manifest checks', () => {
  const tensor = (shape, kind = 'Q6_K') => ({file: 'f', kind, shape, row_bytes: 1});
  const small = {...config, layers: 1, experts: 4, top_k: 2, intermediate: 2048, sliding: [true]};
  const layer = {index: 0, type: 'sliding', tensors: {
    ...Object.fromEntries(['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight', 'ffn_gate_shexp.weight',
      'ffn_up_shexp.weight', 'ffn_down_shexp.weight', 'ffn_gate_inp.weight', 'attn_norm.weight', 'attn_post_norm.weight', 'ffn_norm.weight',
      'ffn_post_norm.weight', 'attn_q_norm.weight', 'attn_k_norm.weight', 'exp_probs_b.bias'].map(n => [n, tensor([8, 8])])),
    'ffn_gate_up_exps.0.weight': tensor([3 * 1024, 2560], 'Q4_K'), 'ffn_gate_up_exps.1.weight': tensor([1024, 2560], 'Q4_K'),
    'ffn_down_exps.0.weight': tensor([4 * 2560, 512], 'Q4_K')}};
  const manifest = {config: small, layers: [layer]};
  assert.doesNotThrow(() => parseManifest(manifest));
  assert.deepEqual(expertChunkNames(layer.tensors, 'ffn_gate_up_exps'), ['ffn_gate_up_exps.0.weight', 'ffn_gate_up_exps.1.weight']);
  const broken = structuredClone(manifest);
  delete broken.layers[0].tensors['ffn_gate_up_exps.1.weight'];
  assert.throws(() => parseManifest(broken), /3 experts, expected 4/);
  const wrongType = structuredClone(manifest);
  wrongType.layers[0].type = 'full';
  assert.throws(() => parseManifest(wrongType), /layer order\/type/);
  const generic = structuredClone(manifest);
  generic.layers[0].tensors['ffn_down_exps.0.weight'].kind = 'Q8_0';
  assert.throws(() => parseManifest(generic), /Invalid ffn_down_exps/);
  assert.throws(() => parseManifest({...manifest, config: {...small, intermediate: 1}}), /scratch/);
});

// Merge guard: optimized dispatch must remain available after plan admission and bounds checks.
for (const tokens of [1, 9]) test(`admitted Kolibri retains fusion and profiling, tokens=${tokens}`, async () => {
  const raw=fixture.plan.stages[0], fast=setup(raw), baseline=setup(raw), diagnostic=setup(raw);
  const install=({gpu,trace},fuse,hostCache)=>{
    Object.assign(gpu,{fuse,hostCache,profiling:true,
      normAdd:(...args)=>trace.push(['norm_add',...args]),
      normPair:(...args)=>trace.push(['norm_pair',...args])});
  };
  install(fast,true,true);install(baseline,false,true);install(diagnostic,true,false);
  const input=fast.manifest.embed?{ids:Array(tokens).fill(1)}:{hidden:new Float32Array(tokens*config.hidden)};
  for(const {s} of [fast,baseline,diagnostic])await s.run({...input,tokens});
  const count=(entry,name)=>entry.trace.filter(t=>Array.isArray(t)&&t[0]===name).length;
  const layers=fast.manifest.layers.length;
  assert.equal(count(fast,'norm_add'),2*layers);
  assert.equal(count(fast,'norm_pair'),layers);
  assert.equal(count(baseline,'norm_add')+count(baseline,'norm_pair'),0);
  assert.equal(count(baseline,'norm')-count(fast,'norm'),4*layers);
  assert.deepEqual(fast.trace,diagnostic.trace);
  assert.deepEqual(fast.s.submitLayers,baseline.s.submitLayers);
  assert.ok(Number.isFinite(fast.gpu.pendingExpandMs));
  assert.deepEqual(fast.s.positions,baseline.s.positions);
});

test('fusion cannot bypass buffer validation or failed-slot recovery',async()=>{
  const {s,gpu,trace}=setup(fixture.plan.stages[0]);
  let fused=0;
  gpu.normPair=()=>{fused++;throw Error('fused encode failed');};
  gpu.normAdd=()=>{fused++;};
  const saved=s.q;s.q={size:4};
  await assert.rejects(s.run({ids:[1],tokens:1}),/allocation/);
  assert.equal(fused,0);assert.ok(!trace.includes('write'));
  s.q=saved;
  await assert.rejects(s.run({ids:[1],tokens:1}),/fused encode failed/);
  await assert.rejects(s.run({ids:[1],tokens:1}),/requires reset/);
  await s.reset(0);gpu.fuse=false;
  await s.run({ids:[1],tokens:1});
});
