// Experiment 48 stage A in one headless page: each fixture case through factorizationTileGPU,
// and the noise hash for chosen entries; results go back to the harness
// (ygaraty/experiments/factorization_webgpu.py), which compares them with NumPy.
import {tensor} from './sdk.mjs';
import {factorizationTileGPU, hashProbeGPU} from './factorization_gpu.mjs';

const post=body=>fetch('/result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
try{
  const fixture=await (await fetch('/fixture.json')).json();
  const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)throw Error('No WebGPU adapter');
  const limits=adapter.limits;
  const device=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:limits.maxStorageBufferBindingSize,maxBufferSize:limits.maxBufferSize}});
  const info=adapter.info||{};
  const result={adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,
    isFallbackAdapter:adapter.isFallbackAdapter??info.isFallbackAdapter??null,maxBufferSize:limits.maxBufferSize,
    maxStorageBufferBindingSize:limits.maxStorageBufferBindingSize},userAgent:navigator.userAgent,cases:[],probes:[]};
  for(const probe of fixture.probes)
    result.probes.push({columns:probe.columns,seed:probe.seed,words:await hashProbeGPU(device,probe.pairs,probe.columns,probe.seed)});
  for(const c of fixture.cases){
    const inputs=['L','R','U','V'].map(k=>tensor(c[k].shape,Float64Array.from(c[k].data)));
    const runs=[];
    for(let i=0;i<fixture.repeats;i++)runs.push(await factorizationTileGPU(device,inputs,c.attrs,c.shape));
    result.cases.push({name:c.name,output:Array.from(runs.at(-1).value.data),timings:runs.map(r=>r.timings)});
  }
  await post(result);
}catch(error){await post({error:String(error?.stack||error)});}
