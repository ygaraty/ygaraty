// Experiment 48 stage C in one headless page: adversarial tasks through the worker's own admission
// (admitTask) and execution (ComputeSDK.execute), with every GPU buffer the device creates counted.
// The harness (ygaraty/experiments/factorization_memory.py) builds the cases and judges them.
import {ComputeSDK, admitTask, defaultBudget} from './sdk.mjs';

const post=body=>fetch('/result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
try{
  const fixture=await (await fetch('/fixture.json')).json();
  const sdk=new ComputeSDK();await sdk.initialize(true);
  if(!sdk.device)throw Error('No WebGPU device: '+sdk.reason);
  // Count every buffer: its size, while the case runs.
  let created=[];const createBuffer=sdk.device.createBuffer.bind(sdk.device);
  sdk.device.createBuffer=descriptor=>{created.push(descriptor.size);return createBuffer(descriptor);};
  const results=[];
  for(const c of fixture.cases){
    created=[];let refused=null,error=null,output=null;
    const work={task:c.task,inputs:c.inputs.map(x=>x.data?{...x,data:Float64Array.from(x.data)}:x)};
    try{
      refused=admitTask(work,c.budget);
      if(!refused){
        const {value}=await sdk.execute(work.task,work.inputs.map(x=>({shape:x.shape,data:x.data})));
        output=c.return_output?Array.from(value.data):null;
      }
    }catch(e){error=String(e.message||e);}
    results.push({name:c.name,refused,error,buffers:created.length,gpu_bytes:created.reduce((a,b)=>a+b,0),output});
  }
  await post({adapter:sdk.adapterInfo,default_budget:defaultBudget(sdk.adapterInfo),results});
}catch(error){await post({error:String(error?.stack||error)});}
