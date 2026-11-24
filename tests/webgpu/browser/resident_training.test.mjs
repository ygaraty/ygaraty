import {test} from 'node:test';
import assert from 'node:assert/strict';
import {supportsResidentTraining,validateResidentTask} from '../../../ygaraty/backends/webgpu/browser/resident_training_plan.mjs';
import {residentTrainingGPU} from '../../../ygaraty/backends/webgpu/browser/resident_training.mjs';
import {DiLoCoTrainer} from '../../../ygaraty/backends/webgpu/browser/diloco.mjs';
import {ComputeSDK} from '../../../ygaraty/backends/webgpu/browser/sdk_compute.mjs';
import {tensor} from '../../../ygaraty/backends/webgpu/browser/sdk_tensor.mjs';
const t=x=>tensor([1,1],[x]);
const task=(id,op,inputs,attrs={})=>({id,op,inputs:inputs.map(value=>({value})),attrs,shape:[1,1]});
const plan=(step=1)=>({step,parameters:{w:'w'},state_parameters:{m:{w:'m'},v:{}},
 constants:{g:{shape:[1,1],data:[.1]}},updates:{w:'next',m:'moment'},loss:'loss',
 tasks:[task('moment','add',['m','g']),task('next','add',['w','moment']),task('loss','sum',['next'])]});
const request=()=>({weights:{w:t(1)},moments:{m:{w:t(0)},v:{}},steps:[{plan:plan(),constants:{g:t(.1)}}]});
const assignment=(version=0)=>({worker:'w',round_id:`r${version}`,version,steps:2,weights:{w:{shape:[1,1],data:[1]}}});
test('resident admission rejects unsupported graphs without touching a device',async()=>{
 const r=request();assert.equal(supportsResidentTraining(r),true);
 r.steps[0].plan.tasks[0].op='unknown';assert.equal(supportsResidentTraining(r),false);
 assert.equal(await residentTrainingGPU(null,r,'compensated'),null);
 assert.equal(await residentTrainingGPU(null,request(),'float32'),null);
});
test('resident admission validates every step before dispatch',()=>{
 for(const mutate of [
  r=>r.steps[0].plan.tasks[0].inputs[0].value='missing',
  r=>r.steps[0].plan.tasks[0].inputs[0].region=[0,2,0,1],
  r=>r.steps[0].plan.updates.w='missing',
  r=>r.steps[0].plan.tasks[0].id='w',
  r=>r.weights.w.data[0]=NaN,
  r=>r.steps[0].plan.state_parameters.m={other:'m'},
 ]){const r=request();mutate(r);assert.throws(()=>supportsResidentTraining(r));}
 const r=request();const invalid=structuredClone(r.steps[0].plan);invalid.step=2;invalid.updates.w='absent';
 r.steps.push({plan:invalid,constants:{g:t(.1)}});assert.throws(()=>supportsResidentTraining(r));
});
test('resident assembly forbids overlaps and holes; CE checks targets',()=>{
 const a={id:'assembled',op:'assemble',shape:[1,2],inputs:[],attrs:{blocks:[[0,1,0,1],[0,1,0,1]]}};
 assert.throws(()=>validateResidentTask(a,[[1,1],[1,1]]));
 a.attrs.blocks=[[0,1,0,1]];assert.throws(()=>validateResidentTask(a,[[1,1]]));
 assert.throws(()=>validateResidentTask({...task('loss','cross_entropy',['x']),attrs:{targets:[3]}},[[1,3]]));
});
test('resident failure and cancellation preserve the committed checkpoint',async()=>{
 for(const cancel of [false,true]){
  const controller=new AbortController(),sdk=new ComputeSDK();let calls=0;
  sdk.executeTrainingRound=async r=>{calls++;assert.equal(r.steps.length,2);if(cancel)controller.abort();throw Error('GPU round failed');};
  const trainer=new DiLoCoTrainer('w',{sdk,precision:'webgpu',signal:controller.signal});
  const saved={step:0,moments:{m:{w:{shape:[1,1],data:[.2]}},v:{}}};
  trainer.restore(saved,assignment().weights);
  await assert.rejects(trainer.runRound(assignment(),[plan(1),plan(2)]),/GPU round failed/);
  assert.deepEqual(trainer.checkpoint(),saved);assert.equal(trainer.weights,null);assert.equal(trainer.busy,false);assert.equal(calls,1);
 }
});
test('portable fallback carries moments across local steps and resume',async()=>{
 const sdk=new ComputeSDK();sdk.executeTrainingRound=async()=>null;
 const trainer=new DiLoCoTrainer('w',{sdk,precision:'webgpu'});
 const result=await trainer.runRound(assignment(),[plan(1),plan(2)]);
 assert.ok(Math.abs(result.pseudo_gradient.w.data[0]+.3)<1e-12);
 assert.equal(trainer.step,2);assert.ok(Math.abs(trainer.checkpoint().moments.m.w.data[0]-.2)<1e-12);
 const restored=new DiLoCoTrainer('w',{sdk,precision:'webgpu'});restored.restore(trainer.checkpoint(),assignment().weights);
 const a=await trainer.runRound(assignment(1),[plan(3),plan(4)]);
 const b=await restored.runRound(assignment(1),[plan(3),plan(4)]);
 assert.deepEqual(a,b);assert.deepEqual(trainer.checkpoint(),restored.checkpoint());
});
test('incomplete plan sequences fail before invoking the GPU',async()=>{
 const sdk=new ComputeSDK();sdk.executeTrainingRound=async()=>{throw Error('unexpected GPU invocation');};
 const trainer=new DiLoCoTrainer('w',{sdk,precision:'webgpu'});
 await assert.rejects(trainer.runRound(assignment(),[plan(1)]),/Not enough local plans/);
 await assert.rejects(trainer.runRound(assignment(),[plan(2),plan(3)]),/Wrong local optimizer step/);
});

test('oversized metadata returns to the portable executor before allocation',()=>{
 const r=request();
 r.steps[0].plan.tasks[0].inputs=Array.from({length:262145},()=>({value:'g'}));
 assert.equal(supportsResidentTraining(r),false);
});
