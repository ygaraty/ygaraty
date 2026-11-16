// Numerical integration fixture, not a network contributor or throughput benchmark.
import {ComputeSDK,tensor} from './sdk.mjs';
import {DiLoCoTrainer} from './diloco.mjs';
import {CONV_OPS} from './convolution.mjs';

export function compare(actual,expected,tolerance){
  if(actual.length!==expected.length)throw Error('Reference shape mismatch');
  let failed=0,maxAbs=0;
  for(let i=0;i<actual.length;i++){
    const error=Math.abs(actual[i]-expected[i]);maxAbs=Math.max(maxAbs,error);
    if(!Number.isFinite(actual[i])||!Number.isFinite(expected[i])||error>tolerance.atol+tolerance.rtol*Math.abs(expected[i]))failed++;
  }
  return {passed:failed===0,failed,max_abs_error:maxAbs};
}

export async function hardwareSDK(gpu=globalThis.navigator?.gpu){
  if(!gpu)throw Error('WebGPU is unavailable');
  const adapter=await gpu.requestAdapter({powerPreference:'high-performance',forceFallbackAdapter:false});
  if(!adapter)throw Error('No WebGPU adapter');
  const info=adapter.info??(adapter.requestAdapterInfo?await adapter.requestAdapterInfo():{});
  const evidence={vendor:info.vendor??'',architecture:info.architecture??'',device:info.device??'',description:info.description??'',
    isFallbackAdapter:info.isFallbackAdapter??adapter.isFallbackAdapter??null};
  if(evidence.isFallbackAdapter!==false||/swiftshader|llvmpipe|lavapipe|software|microsoft basic/i.test(JSON.stringify(evidence)))
    throw Error('Hardware GPU could not be established; refusing software/unknown adapter');
  const sdk=new ComputeSDK();sdk.device=await adapter.requestDevice();
  sdk.device.lost.then(()=>{sdk.device=null;});
  return {sdk,adapter:evidence};
}

export async function runConvnetFixture(fixture,{sdk=new ComputeSDK(),precision='float64',onProgress=()=>{}}={}){
  if(fixture.format!=='ygaraty-convnet-webgpu-v1'||!['float64','webgpu'].includes(precision))throw Error('Invalid fixture or precision');
  const tolerance=fixture.tolerances[precision],checks=[],replicas=[];
  const total=fixture.kernels.length+fixture.replicas.length+1;
  let completed=0;
  const progress=label=>onProgress({label,completed:++completed,total,checks,replicas});
  for(const [index,item] of fixture.kernels.entries()){
    const inputs=item.inputs.map(v=>tensor(v.shape,v.data));
    const result=await sdk.execute({...item,precision},inputs);
    if(precision==='webgpu'&&result.backend!=='webgpu')throw Error('Convolution silently fell back to CPU');
    checks.push({name:`kernel/${index}/${item.op}`,backend:result.backend,...compare(result.value.data,item.expected.data,tolerance)});
    progress(`Checked ${item.op}`);
  }
  for(const item of fixture.replicas){
    const bundle=item.bundle,trainer=new DiLoCoTrainer(bundle.assignment.worker,{sdk,precision,arithmetic:'float32'});
    const result=await trainer.runRound(bundle.assignment,bundle.plans);
    const expectedGPU=bundle.plans.flatMap(p=>p.tasks).filter(t=>CONV_OPS.includes(t.op)||t.op==='matmul').length;
    if(precision==='webgpu'&&result.stats.gpu_tasks!==expectedGPU)throw Error('Missing GPU work in local DiLoCo step');
    checks.push({name:`${result.worker}/loss`,...compare(result.losses,item.expected.losses,tolerance)});
    for(const [name,value] of Object.entries(result.pseudo_gradient))
      checks.push({name:`${result.worker}/pseudo/${name}`,...compare(value.data,item.expected.pseudo_gradient[name].data,tolerance)});
    replicas.push(result);
    progress(`Completed replica ${result.worker}`);
  }
  // First outer round, initially zero velocity, uniform synchronous mean.
  const base=fixture.replicas[0].bundle.assignment.weights,globalWeights={};
  for(const [name,value] of Object.entries(base)){
    const data=Float64Array.from(value.data,(x,i)=>{
      const delta=replicas.reduce((s,r)=>s+r.pseudo_gradient[name].data[i],0)/replicas.length;
      return x-fixture.outer.learning_rate*delta*(fixture.outer.nesterov?1+fixture.outer.momentum:1);
    });
    globalWeights[name]={shape:value.shape,data:Array.from(data)};
    checks.push({name:`global/${name}`,...compare(data,fixture.expected_global[name].data,tolerance)});
  }
  progress('Checked DiLoCo outer update');
  return {kind:'convnet_diloco_numerical_check',precision,passed:checks.every(c=>c.passed),tolerance,checks,
    replicas:replicas.map(r=>({worker:r.worker,losses:r.losses,stats:r.stats})),global_weights:globalWeights,
    limitations:['Two logical replicas in one browser; not multi-machine scalability.',
      'Convolution/pooling and matrix products use WebGPU in webgpu mode; other operations and Adam use JavaScript CPU.',
      'Buffers are uploaded/read back per operation; this prototype does not establish GPU training efficiency.',
      'No CIFAR learning-quality result; synthetic single-step fixture only.']};
}

if(globalThis.document?.getElementById('convnet-run')){
  const button=document.getElementById('convnet-run'),output=document.getElementById('convnet-output'),download=document.getElementById('convnet-download');
  button.addEventListener('click',async()=>{
    const set=(id,text)=>{document.getElementById(id).textContent=text;};
    const started=performance.now();
    document.body.classList.remove('completed','failed');document.body.classList.add('running');
    set('connection','Running');set('phase','CHECKING YOUR GPU');set('local-status','Loading the fixture and checking hardware WebGPU…');
    set('check-count','0');set('replica-count','0 / 2');set('check-progress','0%');set('elapsed','—');
    set('backend-counts','0 WebGPU · 0 CPU operations');document.getElementById('progress').value=0;
    button.disabled=true;output.textContent='Running numerical checks…';download.hidden=true;
    let report,device;
    try{
      const response=await fetch('./fixture.json');
      if(!response.ok)throw Error(`Fixture request failed (${response.status})`);
      const fixture=await response.json();
      const selected=await hardwareSDK();device=selected.sdk.device;
      set('engine','Hardware WebGPU');document.getElementById('engine').dataset.state='gpu';
      set('engine-detail',selected.adapter.description||selected.adapter.vendor||'Hardware adapter confirmed');
      set('phase','RUNNING NUMERICAL CHECKS');
      report={...await runConvnetFixture(fixture,{sdk:selected.sdk,precision:'webgpu',onProgress:state=>{
        const percent=Math.round(100*state.completed/state.total);
        set('operation',state.label);set('local-status',state.label);set('check-progress',`${percent}%`);
        set('check-count',state.checks.filter(c=>c.passed).length);set('replica-count',`${state.replicas.length} / ${fixture.replicas.length}`);
        document.getElementById('progress').value=percent;
        const gpu=state.checks.filter(c=>c.backend==='webgpu').length+state.replicas.reduce((n,r)=>n+r.stats.gpu_tasks,0);
        const cpu=state.replicas.reduce((n,r)=>n+r.stats.cpu_tasks,0);
        set('backend-counts',`${gpu} WebGPU · ${cpu} CPU operations`);
      }}),adapter:selected.adapter};
    }catch(error){report={kind:'convnet_diloco_numerical_check',passed:false,error:String(error)};}
    finally{device?.destroy();button.disabled=false;}
    document.body.classList.remove('running');document.body.classList.add(report.passed?'completed':'failed');
    set('connection',report.passed?'Passed':'Failed');set('phase',report.passed?'CHECKS COMPLETE':'CHECKS FAILED');
    set('operation',report.passed?'All numerical checks passed.':'The test did not pass.');
    set('operation-detail','Download the result for the full report.');
    set('local-status',report.error|| (report.passed?'Completed. Your result is ready to download.':'Numerical differences exceeded the test tolerance. See the detailed result.'));
    set('elapsed',((performance.now()-started)/1000).toFixed(2));
    output.textContent=JSON.stringify(report,null,2);
    if(download.dataset.objectUrl)URL.revokeObjectURL(download.dataset.objectUrl);
    const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));
    download.href=url;download.dataset.objectUrl=url;download.hidden=false;
  });
}
