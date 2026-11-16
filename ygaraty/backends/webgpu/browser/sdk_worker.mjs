import {LITTLE_ENDIAN, WIRE_BYTES, readFloats, frameWire} from './sdk_frames.mjs';
import {ComputeSDK} from './sdk_compute.mjs';
import {TileCache} from './sdk_cache.mjs';
import {RelayClient, assembleRegion} from './sdk_relay.mjs';
import {admitTask, defaultBudget} from './sdk_admission.mjs';
export class BrowserWorker {
  // maxTasks above 1 claims up to that many ready tasks per request and returns their results together.
  // cacheBytes: tile cache budget (0 disables it; the server then sends every input every time).
  // budgetBytes: the memory budget to advertise (runtime/budget.py); by default a quarter of the GPU's
  // buffer limit (defaultBudget), none without a GPU. Enforced only on jobs that enforce budgets.
  // relayURL: overrides the relay address claim replies name (a proxy, say); useRelay=false ignores the relay.
  // relayBatch: one relay request per result and per fetch (RelayClient batch). pipeline: each result
  // uploads while the next task computes, and the results go back with the next claim in one request.
  constructor({baseURL='',job,token,id=globalThis.crypto.randomUUID(),preferGPU=true,onEvent=()=>{},maxTasks=1,cacheBytes=512*1024*1024,budgetBytes=null,relayURL=null,useRelay=true,relayBatch=true,pipeline=true}){
    Object.assign(this,{baseURL,job,token,id,preferGPU,onEvent,maxTasks,budgetBytes,relayURL,useRelay,relayBatch,pipeline});this.stopped=false;this.sdk=new ComputeSDK();
    this.cache=cacheBytes>0?new TileCache(cacheBytes):null;
    this.wire='<f8';   // the job's frame wire, learned from its claim replies; results go back in it
    this.relay=null;   // a RelayClient once a claim reply names one (experiment 50); results then go up as blocks
    this.relayed=0;this.uploaded=0;this.inlineFallbacks=0;
  }
  stop(){this.stopped=true;}
  // Inputs as tensors: received tiles go into the cache; cached ones come from it, or the task fails
  // with a cache miss and the server resends the data on the next attempt.
  resolveInputs(inputs){
    return inputs.map(x=>{
      if(x.cached){const data=this.cache?.get(x.key);if(!data)throw Error('cache miss: '+x.key);return {shape:x.shape,data};}
      const data=x.data instanceof Float64Array?x.data:Float64Array.from(x.data);
      if(x.key&&this.cache)this.cache.put(x.key,data);
      return {shape:x.shape,data};
    });
  }
  async post(action,body){
    if(this.cache)body={...body,...this.cache.report()};
    if(this.budgetBytes)body={...body,budget_bytes:this.budgetBytes,...(this.sdk.adapterInfo?{device:this.sdk.adapterInfo}:{})};
    let payload,contentType='application/json';
    if(body.data instanceof Float64Array){
      const {data:wide,...meta}=body, narrow=this.wire==='<f4', data=narrow?Float32Array.from(wide):wide,
        header=new TextEncoder().encode(JSON.stringify({worker:this.id,...meta,...(narrow?{wire:'<f4'}:{})}));
      // Pad the header with spaces (still valid JSON) so the floats start on an 8-byte boundary.
      const size=header.length+(8-(4+header.length)%8)%8, start=4+size;
      payload=new Uint8Array(start+data.byteLength);new DataView(payload.buffer).setUint32(0,size,true);
      payload.fill(32,4,start);payload.set(header,4);
      if(LITTLE_ENDIAN)payload.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength),start);
      else{const view=new DataView(payload.buffer);for(let i=0;i<data.length;i++)narrow?view.setFloat32(start+4*i,data[i],true):view.setFloat64(start+8*i,data[i],true);}
      contentType='application/octet-stream';
    }else payload=JSON.stringify({worker:this.id,...body});
    const response=await fetch(`${this.baseURL}/api/jobs/${this.job}/${action}`,{method:'POST',headers:{'Content-Type':contentType,Authorization:`Bearer ${this.token}`},body:payload,signal:AbortSignal.timeout(60000)});
    let data;
    // A reply larger than every task it can carry could fit is refused before its body is read.
    const length=Number(response.headers.get('Content-Length'));
    if(this.budgetBytes&&length>this.maxTasks*this.budgetBytes){await response.body?.cancel();throw Error(`reply of ${length} bytes exceeds the memory budget`);}
    if(response.headers.get('Content-Type')==='application/octet-stream'){
      const buffer=await response.arrayBuffer(),view=new DataView(buffer),size=view.getUint32(0,true);
      data=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,size)));let offset=4+size;
      const wire=frameWire(data);this.wire=wire;
      const decode=list=>list.map(x=>{if(x.cached||x.relay)return x;const n=x.shape[0]*x.shape[1],values=readFloats(buffer,offset,n,wire);offset+=n*WIRE_BYTES[wire];return {...x,data:values};});   // relayed inputs carry no bytes here
      if(data.state==='tasks')for(const task of data.tasks)task.inputs=decode(task.inputs);else data.inputs=decode(data.inputs);
      // Relay blocks the coordinator pushed with the reply (Job.push_relayed) follow the inline arrays.
      const pushed=new Map();
      for(const block of data.pushed||[]){pushed.set(block.sha,new Uint8Array(buffer,offset,block.size));offset+=block.size;}
      if(offset!==buffer.byteLength)throw Error('Invalid binary task length');
      await this.fetchRelayed(data,pushed);
    }else data=await response.json();if(!response.ok)throw Error(data.error||`HTTP ${response.status}`);return data;
  }
  // Inputs the reply names as relay blocks: pushed bytes first, one relay request for the rest, each
  // region assembled as the coordinator's store would; they then travel on as inline inputs do.
  async fetchRelayed(data,pushed){
    const tasks=data.state==='tasks'?data.tasks:data.state==='task'?[data]:[];
    if(data.relay&&this.useRelay&&!this.relay)this.relay=new RelayClient({...data.relay,url:this.relayURL||data.relay.url,batch:this.relayBatch});
    const pending=tasks.flatMap(t=>t.inputs.filter(x=>x.relay&&!x.cached&&!x.data));
    if(!pending.length)return;
    if(!this.relay)throw Error('inputs are relay blocks but this worker has no relay');
    const shas=[...new Set(pending.flatMap(x=>x.relay.blocks.map(b=>b.sha)))],missing=shas.filter(s=>!pushed.has(s));
    const raws=new Map(pushed);
    if(missing.length)(await this.relay.getMany(missing)).forEach((raw,k)=>raws.set(missing[k],raw));
    const decoded=new Map();
    for(const x of pending){
      x.data=assembleRegion(x.relay.region,x.relay.blocks,x.relay.blocks.map(b=>raws.get(b.sha)),decoded);
      if(x.data.length!==x.shape[0]*x.shape[1])throw Error('relayed input has the wrong shape');
      this.relayed++;
    }
  }
  // A result for the coordinator: through the relay as blocks when the job has one (the bytes stay
  // off the coordinator), else inline. A relay failure falls back to inline, costing bytes, not the result.
  async packResult(work,value){
    if(this.relay&&work.binary){
      try{return {blocks:await this.relay.upload(value,work.task.attrs?.upload_rows||null),uploaded:true};}
      catch(e){this.inlineFallbacks++;this.onEvent({type:'relay_fallback',message:e.message});}
    }
    return {data:value.data};
  }
  async runMany(tasks){
    // Run claimed tasks in order; return every result in one request, floats concatenated (or, through
    // the relay, as block manifests in one JSON request). With pipeline, a result's relay upload runs
    // while the next task computes, and the final request also claims the next tasks: its reply is
    // returned for run() to use instead of a separate claim.
    const results=[],binary=tasks.every(t=>t.binary),uploads=[];
    this.cache?.pin(tasks.flatMap(t=>t.inputs.filter(x=>x.cached).map(x=>x.key)));
    for(const work of tasks){
      this.onEvent({type:'task',task:work.task.id,op:work.task.op,shape:work.task.shape});
      const start=performance.now();
      try{
        const refused=admitTask(work,this.budgetBytes);if(refused)throw Error(refused);
        const {value,backend}=await this.sdk.execute(work.task,this.resolveInputs(work.inputs));
        const meta={task_id:work.task.id,attempt:work.attempt,backend,compute_ms:performance.now()-start};
        results.push(meta);
        // packResult does not throw: a failed upload falls back to the inline bytes.
        const upload=this.packResult(work,value).then(packed=>{
          if(packed.blocks){meta.blocks=packed.blocks;this.uploaded++;}
          else if(binary)meta.inline=value.data;else meta.data=Array.from(value.data);});
        if(this.pipeline)uploads.push(upload);else await upload;
      }catch(e){results.push({task_id:work.task.id,attempt:work.attempt,error:e.message});}
    }
    await Promise.all(uploads);
    this.cache?.pin([]);
    const claim=this.pipeline?{claim:true,...(this.maxTasks>1?{max_tasks:this.maxTasks}:{})}:null;
    // A reply without a list (an older coordinator) accepts or refuses the request as a whole; one
    // without a state carried no claim, and run() then claims as before.
    const report=(group,ack)=>group.forEach((r,i)=>this.onEvent({type:'result',task:r.task_id,accepted:Array.isArray(ack.accepted)?ack.accepted[i]:ack.accepted,backend:r.backend,compute_ms:r.compute_ms,gpu_available:!!this.sdk.device}));
    if(!binary||results.every(r=>r.inline===undefined)){
      // Every result went to the relay (or failed), or the job takes JSON: one JSON request.
      const ack=await this.post('result',{results,...claim});report(results,ack);
      return claim&&ack.state?ack:null;
    }
    // Inline floats: requests under the server's 16 MiB body limit (12 MiB of floats each, in the
    // job's wire). A relayed result among them carries no floats of its own (size 0).
    for(const r of results)if(r.inline!==undefined)r.size=r.inline.length;else if(r.blocks)r.size=0;
    let ack=null;
    for(let start=0;start<results.length;){
      let stop=start,floats=0;const bytes=WIRE_BYTES[this.wire];
      while(stop<results.length&&(stop===start||bytes*(floats+(results[stop].size||0))<=12*1024*1024)){floats+=results[stop].size||0;stop++;}
      const group=results.slice(start,stop),data=new Float64Array(floats);let at=0;
      for(const r of group)if(r.inline!==undefined){data.set(r.inline,at);at+=r.inline.length;}
      const body={results:group.map(({inline,...r})=>r),data,...(stop===results.length?claim:null)};
      ack=await this.post('result',body);report(group,ack);
      start=stop;
    }
    return claim&&ack?.state?ack:null;
  }
  async run(){
    const backend=await this.sdk.initialize(this.preferGPU);
    this.budgetBytes??=defaultBudget(this.sdk.adapterInfo);
    this.onEvent({type:'ready',backend,gpu_available:!!this.sdk.device,worker:this.id,budget_bytes:this.budgetBytes});
    let failures=0,next=null;
    while(!this.stopped){
      try {
        // With pipeline, the previous results' reply already holds the next claim.
        const work=next??await this.post('claim',this.maxTasks>1?{max_tasks:this.maxTasks}:{});next=null;
        if(['completed','cancelled','failed'].includes(work.state)){this.onEvent({type:'done',state:work.state});break;}
        if(work.state==='tasks'||(work.state==='task'&&this.pipeline)){next=await this.runMany(work.state==='tasks'?work.tasks:[work]);failures=0;continue;}
        if(work.state!=='task'){this.onEvent({type:'waiting'});await new Promise(r=>setTimeout(r,work.retry_ms||300));continue;}
        this.onEvent({type:'task',task:work.task.id,op:work.task.op,shape:work.task.shape});
        const start=performance.now();let result;
        this.cache?.pin(work.inputs.filter(x=>x.cached).map(x=>x.key));
        try {
          const refused=admitTask(work,this.budgetBytes);if(refused)throw Error(refused);
          const {value,backend}=await this.sdk.execute(work.task,this.resolveInputs(work.inputs));
          const packed=await this.packResult(work,value);
          if(packed.blocks)this.uploaded++;
          result={...(packed.blocks?{blocks:packed.blocks}:{data:work.binary?value.data:Array.from(value.data)}),backend,compute_ms:performance.now()-start};
        }catch(e){result={error:e.message};}
        this.cache?.pin([]);
        const ack=await this.post('result',{task_id:work.task.id,attempt:work.attempt,...result});
        this.onEvent({type:'result',task:work.task.id,op:work.task.op,accepted:ack.accepted,backend:result.backend,compute_ms:result.compute_ms,gpu_available:!!this.sdk.device});
        failures=0;
      }catch(e){this.onEvent({type:'error',message:e.message});if(++failures>=5)throw e;await new Promise(r=>setTimeout(r,1000));}
    }
  }
}
