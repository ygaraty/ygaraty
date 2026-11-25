import {test} from 'node:test';
import assert from 'node:assert/strict';
import {supportsMuonBlock,muonBlockGPU} from '../../../ygaraty/backends/webgpu/browser/muon_gpu.mjs';
import {executeLocalPlan} from '../../../ygaraty/backends/webgpu/browser/diloco.mjs';
import {ComputeSDK} from '../../../ygaraty/backends/webgpu/browser/sdk_compute.mjs';
import {tensor} from '../../../ygaraty/backends/webgpu/browser/sdk_tensor.mjs';
const input=()=>new Map([['x',tensor([2,2],[1,2,3,4])]]);
const task=(id,op,refs,attrs={})=>({id,op,inputs:refs.map(value=>({value})),shape:[2,2],attrs:{muon_block:'w',optimizer:true,...attrs}});
const tasks=()=>[task('a','scale',['x'],{factor:2}),task('b','add',['a','x'])];
test('resident admission rejects tiled, malformed and unsupported graphs before GPU allocation',async()=>{
  assert.equal(supportsMuonBlock(tasks(),input(),['b']),true);
  for(const mutate of [
    t=>t[0].inputs[0].region=[0,1,0,2],
    t=>t[0].attrs.factor=NaN,
    t=>t[0].shape=[-1,2],
    t=>t[0].op='inverse_norm',
    t=>t[1].inputs[0].value='missing',
    t=>t[0].op='assemble',
  ]){const t=tasks();mutate(t);assert.equal(supportsMuonBlock(t,input(),['b']),false);}
  assert.equal(supportsMuonBlock(tasks(),input(),['missing']),false);
  assert.equal(await muonBlockGPU(null,tasks(),input(),['b'],'float32'),null);
});
test('batch execution retains externally used intermediates and releases consumed inputs',async()=>{
  const t=tasks();t.push({...task('c','add',['a','b']),attrs:{}});
  const sdk=new ComputeSDK();let calls=0;
  sdk.executeMuonBlock=async(group,inputs,outputs)=>{
    calls++;assert.deepEqual(outputs,['a','b']);
    const result=await executeLocalPlan({tasks:group,loss:'b',updates:{a:'a',b:'b'}},Object.fromEntries(inputs),new ComputeSDK());
    return result.values;
  };
  const r=await executeLocalPlan({tasks:t,loss:'c',updates:{}},Object.fromEntries(input()),sdk,{precision:'webgpu'});
  assert.equal(calls,1);assert.deepEqual([...r.values.keys()],['c']);
  assert.deepEqual([...r.values.get('c').data],[5,10,15,20]);
});
test('unsupported blocks fall back; GPU errors propagate without silent CPU retry',async()=>{
  const sdk=new ComputeSDK();sdk.executeMuonBlock=async()=>null;
  const plan={tasks:tasks(),loss:'b',updates:{}};
  const result=await executeLocalPlan(plan,Object.fromEntries(input()),sdk,{precision:'webgpu'});
  assert.deepEqual([...result.values.get('b').data],[3,6,9,12]);
  sdk.executeMuonBlock=async()=>{throw Error('device lost');};
  await assert.rejects(()=>executeLocalPlan(plan,Object.fromEntries(input()),sdk,{precision:'webgpu'}),/device lost/);
});
test('CPU execution never invokes the resident path',async()=>{
  const sdk=new ComputeSDK();sdk.executeMuonBlock=async()=>{throw Error('unexpected');};
  await executeLocalPlan({tasks:tasks(),loss:'b',updates:{}},Object.fromEntries(input()),sdk);
});
