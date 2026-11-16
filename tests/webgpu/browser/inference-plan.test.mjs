import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {validatePlan, executePlan} from '../../../ygaraty/backends/webgpu/browser/inference_plan.mjs';
import {Stage as Qwen, qwenGeometry} from '../../../ygaraty/backends/webgpu/browser/qwen35_gpu.mjs';
import {Stage as Dense} from '../../../ygaraty/backends/webgpu/browser/dense_gpu.mjs';
const fixtures=JSON.parse(readFileSync(new URL('./inference-plans.json',import.meta.url),'utf8'));
function setup(fixture, raw) {
  const manifest={layers:raw.tasks.filter(t=>t.attrs.layer_end).map(t=>({index:t.attrs.layer,
    type:fixture.family==='dense'?'dense':t.attrs.layer%4===3?'full':'linear'})),
    embed:raw.input==='tokens'?{}:null,head:raw.output==='next_token'?{}:null};
  const plan=validatePlan(raw,manifest,fixture.family,fixture.hidden);
  const trace=[],gpu={prefill:'tiled',device:{queue:{writeBuffer:()=>trace.push('write')}},
    begin:()=>trace.push('begin'),end:async()=>{trace.push('end');},
    uniform:values=>({uniform:values}),
    dispatch:(name,shader,...args)=>trace.push([name,createHash('sha256').update(shader).digest('hex'),...args])};
  for(const name of ['gather','norm','matmul','siluMul','argmax','conv','delta','rope','store','attention'])
    gpu[name]=(...args)=>trace.push([name,...args]);
  const s=Object.create((fixture.family==='dense'?Dense:Qwen).prototype);
  const named=name=>({name,size:2**30});
  const packed = name => ({name,kind:'F32',rows:1,cols:1,rowBytes:4,buffer:named(name)});
  const weights = prefix => new Proxy({}, {get: (_, key) => ({...packed(`${prefix}.${String(key)}`),size:2**30})});
  const layers=manifest.layers.map(L=>({...L,matrices:weights(`L${L.index}`),vectors:weights(`L${L.index}`),
    conv:named(`conv${L.index}`),head:named(`head${L.index}`),
    ...Object.fromEntries(['Ks','Vs','Ss','convStates'].map(key=>[key,[0,1].map(slot=>named(`${key}${L.index}.${slot}`))]))}));
  // Object.create skips the constructor: set the geometry it derives from the manifest.
  if(fixture.family!=='dense')s.geometry=qwenGeometry(fixture.config);  // 9B unless the config has head counts
  Object.assign(s,{gpu,plan,loaded:true,manifest,hidden:fixture.hidden,layers,
    config:fixture.config,intermediate:fixture.config.intermediate,kv:'f16',vocab:fixture.config.vocab,
    embed:manifest.embed?[{...packed('embed'),rows:10,first:0}]:undefined,output:manifest.head?[{...packed('head0'),first:0},{...packed('head7'),first:7}]:undefined,
    slots:2,maxBatch:2,maxTokens:128,maxPositions:1024,positions:[3,7],position:3,slot:0});
  for(const key of ['x','h','f1','f2','part','ml','ids','invFreq','outputNorm','logits','token','big','big2','mid','z','ab','q','k','kn','v','att'])s[key]=named(key);
  return {s,gpu,trace,manifest};
}
async function legacy(s,tokens,batch=false,heads="last") {
  const {gpu}=s;const per=Math.max(1,Math.floor(64/tokens));
  const parts=batch?[{slot:0,row:0,count:1,position:3},{slot:1,row:1,count:1,position:7}]:[{slot:1,row:0,count:tokens,position:7}];
  if(!batch)s.use(1);else gpu.prefill='matvec';
  gpu.device.queue.writeBuffer();
  for(let k=0;k<s.layers.length;k+=per) {
    gpu.begin();if(s.embed&&k===0)gpu.gather(s.embed,s.ids,s.x,tokens);
    for(const L of s.layers.slice(k,k+per)) {
      if(s.config.family==='dense')s.layer(L,tokens,parts);
      else if(batch)s.layerBatch(L,[0,1]);else s.layer(L,tokens);
    }
    if(s.output&&k+per>=s.layers.length&&heads!=='none'){if(batch||heads==='all')s.headBatch(tokens);else s.head(tokens);}
    await gpu.end();
  }
}
for(const [f,fixture] of fixtures.entries()) for(const raw of fixture.plan.stages) {
  for(const tokens of [1,9,64,128]) test(`primitive trace matches prior arithmetic ${f}/${raw.owner}, tokens=${tokens}`,async()=>{
    const {s,trace,manifest}=setup(fixture,raw),old=setup(fixture,raw);
    const input=manifest.embed?{ids:Array(tokens).fill(1)}:{hidden:new Float32Array(tokens*fixture.hidden)};
    await s.run({...input,tokens,slot:1});await legacy(old.s,tokens);
    assert.deepEqual(trace,old.trace);assert.deepEqual(s.positions,[3,7+tokens]);
    assert.equal(s.submitLayers.reduce((a,b)=>a+b,0),manifest.layers.length);
  });
  test(`batch primitive trace and rejection preserve state ${f}/${raw.owner}`,async()=>{
    const {s,gpu,trace,manifest}=setup(fixture,raw),old=setup(fixture,raw);
    const input=manifest.embed?{ids:[1,2]}:{hidden:new Float32Array(2*fixture.hidden)};
    await s.runBatch({...input,slots:[0,1]});await legacy(old.s,2,true);
    assert.deepEqual(trace,old.trace);assert.deepEqual(s.positions,[4,8]);assert.equal(gpu.prefill,'tiled');
    gpu.end=async()=>{throw Error('device failed');};
    await assert.rejects(s.runBatch({...input,slots:[0,1]}),/device failed/);
    assert.deepEqual(s.positions,[4,8]);assert.equal(gpu.prefill,'tiled');
  });
}
test('unsupported or modified tasks fail before encoding',async()=>{
  const fixture=fixtures[0],raw=fixture.plan.stages[0],{manifest}=setup(fixture,raw);
  for(const mutate of [p=>p.tasks.reverse(),p=>p.tasks[0].attrs.owner++,p=>p.tasks[1].op='unknown',p=>p.output='wrong']) {
    const p=structuredClone(raw);mutate(p);assert.throws(()=>validatePlan(p,manifest,fixture.family,fixture.hidden));
  }
  let started=false;
  await assert.rejects(executePlan(raw,1,{begin:()=>{started=true;},end:async()=>{}},{},[],[]),/admitted/);
  assert.equal(started,false);
  await assert.rejects(executePlan(null,1,{}, {},[],[]),/required/);
});

test('fused arithmetic implementations are unchanged from the prior release',async()=>{
  const {createHash}=await import('node:crypto');
  const expected=JSON.parse(readFileSync(new URL('./inference-kernel-hashes.json',import.meta.url),'utf8'));
  for(const [family,Stage] of [['qwen35',Qwen],['dense',Dense]])
    for(const [name,hash] of Object.entries(expected[family]).filter(([name])=>name!=='reset'))
      assert.equal(createHash('sha256').update(Stage.prototype[name].toString()).digest('hex'),hash,`${family}.${name}`);
});

test('explicit Tape parameters control arithmetic and missing bindings fail before encoding',async()=>{
  const fixture=fixtures[0],raw=structuredClone(fixture.plan.stages[0]);
  const norm=raw.tasks.find(t=>t.op==='rms_norm');
  norm.attrs.call[4].eps=0.0125;
  const {s,trace}=setup(fixture,raw);
  // Opaque model methods must never participate in plan execution.
  s.layer=s.layerBatch=s.head=s.headBatch=()=>{throw Error('hidden model execution');};
  await s.run({ids:[1],tokens:1});
  assert.equal(trace.find(t=>Array.isArray(t)&&t[0]==='norm')[5].eps,0.0125);
  const missing=setup(fixture,raw);
  delete missing.s.x;
  await assert.rejects(missing.s.run({ids:[1],tokens:1}),/Missing plan binding/);
  assert.ok(!missing.trace.includes('begin'));
});

// Kernel dispatch follows the plan even if adapter metadata is changed.
test('Phi-4 rotation is selected by Tape, not hidden adapter configuration', async()=>{
  const fixture=fixtures.find(f=>f.config.rope_pairs==='halves');
  const raw=fixture.plan.stages[0];
  const planned=setup(fixture,raw), prior=setup(fixture,raw);
  planned.s.config={...planned.s.config,rope_pairs:'adjacent'};
  await planned.s.run({ids:[1],tokens:1,slot:1});
  await legacy(prior.s,1);
  assert.deepEqual(planned.trace,prior.trace);
});

for (const heads of ['last','all','none']) for (const mode of ['exact','gemm']) {
  test(`compiled speculative head=${heads}, mode=${mode}`,async()=>{
    const fixture=fixtures.find(f=>f.family==='qwen35'&&f.plan.stages.length===1);
    const raw=fixture.plan.stages[0];
    const planned=setup(fixture,raw),prior=setup(fixture,raw);
    planned.s.maxBatch=128;
    for(const instance of [planned,prior]){
      const matmul=instance.gpu.matmul;
      instance.gpu.matmul=(...args)=>{instance.trace.push(['prefill',instance.gpu.prefill]);matmul(...args);};
    }
    prior.gpu.prefill=mode==='exact'?'matvec':'gemm';
    await legacy(prior.s,9,false,heads);
    await planned.s.run({ids:Array(9).fill(1),tokens:9,slot:1,heads,[mode]:true});
    assert.deepEqual(planned.trace,prior.trace);
    assert.equal(planned.gpu.prefill,'tiled');
    assert.deepEqual(planned.s.positions,[3,16]);
    planned.gpu.end=async()=>{throw Error('failed submission');};
    await assert.rejects(planned.s.run({ids:[1],tokens:1,slot:1,heads,[mode]:true}),/failed submission/);
    assert.equal(planned.gpu.prefill,'tiled');
    assert.deepEqual(planned.s.positions,[3,16]);
  });
}

test('WS-36: one-token products of the same input share a dispatch, gate and up absorb silu_mul', async()=>{
  const fixture=fixtures.find(f=>f.family==='qwen35'),raw=fixture.plan.stages[0];
  for(const tokens of [1,9]){
    const {s,gpu,trace}=setup(fixture,raw);
    gpu.normAdd=(...a)=>trace.push(['normAdd',...a]);
    for(const l of s.layers)l.matrices=new Proxy({}, {get:(_,key)=>({name:`L${l.index}.${String(key)}`,kind:'Q4_K',cols:256,rowBytes:144,buffer:{size:2**30},
      rows:String(key).startsWith('ffn_')?fixture.config.intermediate:1})});
    gpu.groupShape=Ws=>Ws.length>=2&&Ws.length<=4?{tpr:16,lanes:4}:null;
    gpu.matmulGroup=(items,x,glu)=>trace.push(['group',items.map(i=>[i.W.name,i.y.name,i.yOffset]),x.name,glu]);
    await s.run({ids:Array(tokens).fill(1),tokens,slot:1});
    const groups=trace.filter(t=>t[0]==='group'),names=groups.map(g=>g[1].map(i=>i[0].split('.').slice(1).join('.')).join('+'));
    if(tokens>1){assert.equal(groups.length,0);continue;}
    const L=s.layers,linear=L.filter(l=>l.type==='linear').length,full=L.length-linear;
    assert.equal(names.filter(n=>n==='attn_qkv.weight+attn_gate.weight+ssm_alpha.weight+ssm_beta.weight').length,linear);
    assert.equal(names.filter(n=>n==='attn_q.weight+attn_k.weight+attn_v.weight').length,full);
    const ffn=groups.filter(g=>g[1][0][0].endsWith('ffn_gate.weight'));
    assert.equal(ffn.length,L.length);assert.ok(ffn.every(g=>g[3]===true&&g[1][1][0].endsWith('ffn_up.weight')));
    assert.ok(groups.filter(g=>!g[1][0][0].endsWith('ffn_gate.weight')).every(g=>g[3]===false));
    assert.equal(trace.filter(t=>t[0]==='siluMul').length,0);
    const ab=groups.find(g=>g[1].length===4);assert.deepEqual(ab[1].slice(2).map(i=>[i[1],i[2]]),[['ab',0],['ab',32]]);
  }
});

test('admitted plans are deeply frozen snapshots and cannot be forged',async()=>{
  const fixture=fixtures[0],raw=structuredClone(fixture.plan.stages[0]);
  const {s,gpu}=setup(fixture,raw);
  assert.ok(Object.isFrozen(s.plan.tasks[0].attrs.call));
  assert.throws(()=>s.plan.tasks.pop(),TypeError);
  raw.tasks.length=0;
  assert.ok(s.plan.tasks.length);
  await assert.rejects(executePlan(structuredClone(s.plan),1,gpu,s.kernels(),[],[],s.planContext(1,false,[{slot:0,row:0,count:1,position:0}])),/admitted/);
});

test('undersized buffers fail before encoding and do not dirty state',async()=>{
  const fixture=fixtures[0],{s,trace}=setup(fixture,fixture.plan.stages[0]);
  const original=s.x;s.x={size:4};
  await assert.rejects(s.run({ids:[1],tokens:1}),/allocation/);
  assert.ok(!trace.includes('begin'));
  assert.ok(!trace.includes('write'));
  s.x=original;await s.run({ids:[1],tokens:1});
});

test('failed submissions block affected slots until reset completes',async()=>{
  const fixture=fixtures.find(f=>f.family==='dense'),{s,gpu}=setup(fixture,fixture.plan.stages[0]);
  const input={ids:[1],tokens:1,slot:0};
  gpu.end=async()=>{throw Error('lost device');};
  await assert.rejects(s.run(input),/lost device/);
  gpu.end=async()=>{};
  await assert.rejects(s.run(input),/requires reset/);
  await s.run({...input,slot:1});
  await s.reset(0);await s.run(input);
  assert.equal(s.positions[0],1);
});

test('stage lock rejects concurrent runs and resets',async()=>{
  const fixture=fixtures[0],{s,gpu}=setup(fixture,fixture.plan.stages[0]);
  let release;const blocked=new Promise(resolve=>{release=resolve;});
  gpu.end=()=>blocked;
  const running=s.run({ids:[1],tokens:1});
  await assert.rejects(s.run({ids:[1],tokens:1,slot:1}),/busy/);
  await assert.rejects(s.reset(),/busy/);
  release();await running;
});

test('failed recurrent reset retains dirty slots and speculative restore cannot bypass it',async()=>{
  const fixture=fixtures.find(f=>f.family==='qwen35'),{s,gpu}=setup(fixture,fixture.plan.stages[0]);
  gpu.end=async()=>{throw Error('submission failed');};
  await assert.rejects(s.runBatch({ids:[1,2],slots:[0,1]}),/submission failed/);
  for(const slot of [0,1]){
    assert.throws(()=>s.snapshot(slot),/requires reset/);
    assert.throws(()=>s.restore(slot),/requires reset/);
  }
  gpu.device.createCommandEncoder=()=>({clearBuffer(){},finish(){return {};}});
  gpu.device.queue.submit=()=>{};
  gpu.device.queue.onSubmittedWorkDone=async()=>{throw Error('reset failed');};
  await assert.rejects(s.reset(0),/reset failed/);
  gpu.end=async()=>{};
  await assert.rejects(s.run({ids:[1],tokens:1,slot:0}),/requires reset/);
  gpu.device.queue.onSubmittedWorkDone=async()=>{};
  await s.reset(0);await s.run({ids:[1],tokens:1,slot:0});
  await assert.rejects(s.run({ids:[1],tokens:1,slot:1}),/requires reset/);
});
