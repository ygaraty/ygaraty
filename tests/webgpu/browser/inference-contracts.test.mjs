import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {operationContract, encodeCall} from '../../../ygaraty/backends/webgpu/browser/inference_contracts.mjs';
import {validatePlan, executePlan} from '../../../ygaraty/backends/webgpu/browser/inference_plan.mjs';
const read=name=>JSON.parse(readFileSync(new URL(name,import.meta.url),'utf8'));
for(const c of read('./inference-contracts.json'))test(`shared schema: ${c.name}`,()=>{
  if(c.valid)assert.deepEqual(operationContract(c.op,c.call,c.layer),c.contract);
  else assert.throws(()=>operationContract(c.op,c.call,c.layer));
});
const fixture=read('./inference-plans.json')[0];
function setup(){
  const plan=structuredClone(fixture.plan.stages[0]);
  const manifest={layers:plan.tasks.filter(t=>t.attrs.layer_end).map(t=>({index:t.attrs.layer,type:'linear'})),embed:{}};
  if(plan.output==='next_token')manifest.head={};
  return {plan,manifest};
}
const validate=(plan,manifest)=>validatePlan(plan,manifest,fixture.family,fixture.hidden);
test('contracts reject tampering and state outside a sequence scope',()=>{
  for(const mutate of [
    p=>p.tasks[0].attrs.contract.effects[0].access='write',
    p=>delete p.tasks[0].attrs.contract,
    p=>p.state_contract.failure='rollback',
    p=>p.contract_version=99,
    p=>{const t=p.tasks.find(t=>t.op==='for_each_sequence');t.op='for_each_chunk';p.required_ops=[...new Set([...p.required_ops.filter(x=>x!=='for_each_sequence'),'for_each_chunk'])].sort();},
  ]){
    const {plan,manifest}=setup();mutate(plan);assert.throws(()=>validate(plan,manifest));
  }
});
test('legacy validation upgrades a copy without changing its calls',()=>{
  const {plan,manifest}=setup(),legacy=structuredClone(plan);
  legacy.version=2;delete legacy.contract_version;delete legacy.state_contract;
  function strip(tasks){for(const t of tasks){delete t.attrs.contract;if(t.attrs.body)strip(t.attrs.body);}}
  strip(legacy.tasks);const before=structuredClone(legacy);
  assert.deepEqual(validate(legacy,manifest),plan);assert.deepEqual(legacy,before);
  legacy.tasks[0].attrs.contract={};assert.throws(()=>validate(legacy,manifest));
});
test('named arguments retain the existing positional ABI',()=>{
  const args={weight:{$:'matrix',name:'w'},input:{$:'buffer',name:'x'},output:{$:'buffer',name:'y'},tokens:2};
  assert.deepEqual(encodeCall('matmul',args,0),[args.weight,args.input,args.output,2]);
  assert.throws(()=>encodeCall('matmul',{...args,typo:3},0));
});
test('dynamic scalar and state bounds reject before GPU begin',async()=>{
  const {plan,manifest}=setup();const p=validate(plan,manifest);
  let begun=0;const gpu={begin(){begun++;},async end(){}};
  const kernels=Object.fromEntries(p.required_ops.map(op=>[op,()=>{}]));
  const context={stateOwner:{},buffers:new Proxy({}, {get:()=>({})}),resources:{embed:[]},variables:{tokens:1,batch:false,capacity:4,kv:'f16',head_enabled:false},sequences:[{slot:0,row:0,count:1,position:4}],layer:()=>({})};
  await assert.rejects(executePlan(p,1,gpu,kernels,[],[],context),/capacity/);
  context.sequences[0].position=0;context.sequences[0].slot=-1;
  await assert.rejects(executePlan(p,1,gpu,kernels,[],[],context),/scope/);
  context.sequences[0].slot=0;context.variables.tokens='wrong';
  await assert.rejects(executePlan(p,1,gpu,kernels,[],[],context),/count mismatch/);
  context.variables.tokens=1;context.buffers=new Proxy({}, {get:()=>42});
  await assert.rejects(executePlan(p,1,gpu,kernels,[],[],context),/resource/);
  assert.equal(begun,0);
});

test('sequence scopes cannot overlap or leave input rows uncovered',async()=>{
  const {plan,manifest}=setup(),p=validate(plan,manifest);
  let begin=0;const gpu={begin(){begin++;},async end(){}};
  const kernels=Object.fromEntries(p.required_ops.map(op=>[op,()=>{}]));
  for(const sequences of [
    [{slot:0,row:0,count:1,position:0}],
    [{slot:0,row:0,count:1,position:0},{slot:1,row:0,count:1,position:0}],
  ]) {
    await assert.rejects(executePlan(p,2,gpu,kernels,[],[],{stateOwner:{},variables:{tokens:2},sequences}),/count mismatch|overlap/);
  }
  assert.equal(begin,0);
});
