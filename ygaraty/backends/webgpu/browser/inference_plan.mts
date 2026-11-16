import {validateBufferLayout} from './inference_layout.mjs';
import type {LayoutGeometry} from './inference_layout.mjs';
import type {OperationName} from './inference_schema.mjs';
import {operationContract, checkContract, validateResolved, stateContract, loopOperations as loops} from './inference_contracts.mjs';
import type {OperationContract, Expression} from './inference_contracts.mjs';
// Model-independent execution of Tape compositions over registered GPU kernels.
export interface PlanTask {
  id: string; op: string; inputs: {value: string; region?: null}[]; shape: number[];
  attrs: {owner: number; layer: number | null; call: Expression[]; contract?: OperationContract; body?: PlanTask[]; layer_end?: boolean; when?: "head_enabled"};
}
interface PlanData {
  version: 3; contract_version: 1; state_contract: typeof stateContract; family: string; owner: number; hidden: number; rows: 'tokens';
  input: string; output: string; tasks: PlanTask[]; required_ops: string[]; submit_token_layers: number;
}
type ExecutableTask = Readonly<Omit<PlanTask, 'attrs'|'inputs'|'shape'>> & {
  readonly inputs:readonly Readonly<PlanTask['inputs'][number]>[];
  readonly shape:readonly number[];
  readonly attrs:Readonly<Omit<PlanTask['attrs'], 'body'|'call'>> & {
    readonly body?:readonly ExecutableTask[];
    readonly call:readonly Expression[];
  };
};
declare const admission:unique symbol;
export type StagePlan = Readonly<Omit<PlanData, 'tasks'|'required_ops'>> & {
  readonly tasks:readonly ExecutableTask[];
  readonly required_ops:readonly string[];
  readonly [admission]:true;
};
const admitted = new WeakSet<object>();
function freeze<T>(value:T):T {
  if(value && typeof value==='object') {
    for(const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
  return value;
}
interface SlotLifecycle {dirty:Set<number>; busy:boolean}
const lifecycles = new WeakMap<object, SlotLifecycle>();
function lifecycle(owner:object):SlotLifecycle {
  let state=lifecycles.get(owner);
  if(!state) {state={dirty:new Set(),busy:false};lifecycles.set(owner,state);}
  return state;
}
export function assertStateReady(owner:object, slots:readonly number[]):void {
  const state=lifecycle(owner);
  if(state.busy)throw Error('Inference stage is busy');
  for(const slot of slots)if(state.dirty.has(slot))throw Error(`Inference slot ${slot} requires reset after failure`);
}
/** Failed resets retain dirty state; reset and inference share the stage lock. */
export async function resetPlanState(owner:object, slots:readonly number[], reset:()=>Promise<void>):Promise<void> {
  const state=lifecycle(owner);
  assertStateIdle(owner);
  state.busy=true;
  for(const slot of slots)state.dirty.add(slot);
  try {
    await reset();
    for(const slot of slots)state.dirty.delete(slot);
  } finally {state.busy=false;}
}
export function assertStateIdle(owner:object):void {
  if(lifecycle(owner).busy)throw Error('Inference stage is busy');
}
interface Layout {layers: {index: number; type: string}[]; embed?: unknown; head?: unknown}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw Error('Invalid plan object');
  return value as Record<string, unknown>;
}
export function validatePlan(value: unknown, manifest: Layout, family: string, hidden: number): StagePlan {
  const p=structuredClone(value) as PlanData;
  const legacy=record(value).version===2;
  if (!p || (!legacy && (p.version!==3 || p.contract_version!==1)) || p.family!==family || p.hidden!==hidden || p.rows!=='tokens' ||
      !Number.isSafeInteger(p.owner) || p.owner<0 || p.submit_token_layers!==64 || !Array.isArray(p.tasks) ||
      !Array.isArray(p.required_ops) || !manifest.layers.length) throw Error('Missing or unsupported compiled inference plan');
  if(legacy){
    if(p.contract_version!==undefined||p.state_contract!==undefined)throw Error('Legacy plan has unexpected contracts');
    p.state_contract=structuredClone(stateContract);
  }else checkContract(p.state_contract,stateContract);
  const incoming=manifest.embed?'tokens':`hidden_${manifest.layers[0].index-1}`;
  if(p.input!==incoming)throw Error('Plan input differs from stage');
  const operations=new Set<string>(),ends:number[]=[];
  let budget=10000;
  function validate(tasks:PlanTask[],input:string,depth=0,sequence=false):string {
    if(depth>4 || !tasks.length)throw Error('Invalid plan composition');
    let previous=input;const ids=new Set([input]);
    for(const t of tasks){
      if(--budget<0 || !t || typeof t.id!=='string' || ids.has(t.id) || t.attrs?.owner!==p.owner ||
         !Array.isArray(t.inputs) || t.inputs.length!==1 || t.inputs[0].value!==previous ||
         (t.inputs[0].region!==null && t.inputs[0].region!==undefined) || !Array.isArray(t.attrs.call) ||
         !Array.isArray(t.shape) || t.shape.length!==2 || t.shape[0]!==1 || !Number.isSafeInteger(t.shape[1]) || t.shape[1]<1)
        throw Error('Invalid task dependency/shape/ownership');
      if(t.attrs.when!==undefined && (t.attrs.when!=="head_enabled" || depth || t.attrs.layer!==null))throw Error("Invalid execution guard");
      if(t.attrs.layer!==null && !manifest.layers.some(l=>l.index===t.attrs.layer)) throw Error('Unknown layer resource');
      const contract=operationContract(t.op,t.attrs.call,t.attrs.layer);
      if(legacy){
        if(t.attrs.contract!==undefined)throw Error('Legacy plan has unexpected contracts');
        t.attrs.contract=contract;
      }else checkContract(t.attrs.contract,contract);
      if(contract.effects.some(e=>e.scope==='layer_sequence')&&!sequence)throw Error('State access requires a sequence loop');
      if(loops.has(t.op)){
        if(t.attrs.call.length || !Array.isArray(t.attrs.body))throw Error('Invalid loop body');
        validate(t.attrs.body,previous,depth+1,sequence||t.op==='for_each_sequence');
      }else{
        if(t.attrs.body!==undefined)throw Error('Primitive has a loop body');
      }
      if(t.attrs.layer_end){if(depth || t.attrs.layer===null)throw Error('Invalid layer boundary');ends.push(t.attrs.layer);}
      operations.add(t.op);ids.add(t.id);previous=t.id;
    }
    return previous;
  }
  if(validate(p.tasks,incoming)!==p.output || p.output!==(manifest.head?'next_token':`hidden_${manifest.layers.at(-1)!.index}`) ||
     JSON.stringify(ends)!==JSON.stringify(manifest.layers.map(l=>l.index)) ||
     JSON.stringify([...operations].sort())!==JSON.stringify(p.required_ops))throw Error('Invalid plan output, boundaries or kernel requirements');
  p.version=3;p.contract_version=1;admitted.add(p);return freeze(p) as unknown as StagePlan;
}
export type Kernel = (...args: unknown[]) => void;
export type KernelRegistry = Partial<Record<OperationName, Kernel>>;
export function bindKernel<A extends unknown[]>(fn:(...args:A)=>void):Kernel {return (...args)=>fn(...args as A);}
export interface SequencePart {slot:number; row:number; count:number; position:number}
export interface PlanContext {
  stateOwner:object;
  buffers:Record<string, GPUBuffer>;
  layer(index:number):object;
  resources:Record<string,unknown>;
  variables:Record<string,unknown>;
  sequences:SequencePart[];
  /** Qwen3.5 head counts for buffer checks; the 9B's when absent. */
  geometry?:LayoutGeometry;
}
function resolve(value:unknown,ctx:PlanContext,task:ExecutableTask,part:SequencePart):unknown {
  if(Array.isArray(value))return value.map(v=>resolve(v,ctx,task,part));
  if(value===null || typeof value!=='object')return value;
  const v=record(value),name=String(v.name);
  const layer=()=>{if(task.attrs.layer===null)throw Error('No layer resource');return record(ctx.layer(task.attrs.layer));};
  const required=(x:unknown)=>{if(x===undefined||x===null)throw Error(`Missing plan binding: ${name}`);return x;};
  switch(v.$){
    case 'buffer':{
      const b=required(ctx.buffers[name]);
      if(v.stride!==undefined && (v.view==='always' || (v.view==='head' ? ctx.variables.head_all : ctx.variables.batch))){
        if(!Number.isSafeInteger(v.stride) || Number(v.stride)<1)throw Error('Invalid view stride');
        const offset=part.row*Number(v.stride)*4;
        if(!Number.isSafeInteger(offset)||offset<0 || (typeof (b as GPUBuffer).size==='number'&&offset>=(b as GPUBuffer).size))throw Error('Buffer view outside allocation');
        return {buffer:b,offset};
      }
      return b;
    }
    case 'matrix':return required(record(layer().matrices)[name]);
    case 'vector':return required(record(layer().vectors)[name]);
    case 'layer_buffer':return required(layer()[name]);
    case 'state':{const slots=layer()[name];if(!Array.isArray(slots))throw Error('Invalid state resource');return required(slots[part.slot]);}
    case 'var':return required(({row:part.row,count:part.count,position:part.position} as Record<string,unknown>)[name]??ctx.variables[name]);
    case 'resource':return required(ctx.resources[name]);
    case 'product':{
      if(!Array.isArray(v.values))throw Error('Invalid product expression');
      const values=v.values.map(x=>resolve(x,ctx,task,part));
      if(!values.every(x=>typeof x==='number' && Number.isFinite(x)))throw Error('Invalid numeric binding');
      return (values as number[]).reduce((a,b)=>a*b,1);
    }
    case 'choose':if(v.condition!=='batch' && v.condition!=='head_all')throw Error('Unknown predicate');return resolve(ctx.variables[String(v.condition)]?v.yes:v.no,ctx,task,part);
    case undefined:return Object.fromEntries(Object.entries(v).map(([k,x])=>[k,resolve(x,ctx,task,part)]));
    default:throw Error('Unknown binding');
  }
}
// Admission freezes and validates every task once. Keep hostCache=false as the
// diagnostic baseline that repeats contract checks without bypassing admission.
export async function executePlan(
  plan:StagePlan|null,tokens:number,gpu:{begin():void;end():Promise<unknown>;hostCache?:boolean;profiling?:unknown;pendingExpandMs?:number;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fuse?:boolean;normAdd?:(...args:any[])=>void;normPair?:(...args:any[])=>void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    groupShape?:(Ws:any[])=>unknown;matmulGroup?:(items:any[],x:any,glu?:boolean,norm?:any)=>void;normFold?:boolean;
    device?:{limits?:{maxStorageBuffersPerShaderStage:number}}},kernels:KernelRegistry,
  submitMs:number[],submitLayers:number[],context:PlanContext,upload?:()=>void,
):Promise<void>{
  if(!plan)throw Error('A compiled inference plan is required');
  if(!admitted.has(plan))throw Error('An admitted plan from validatePlan is required');
  for(const op of plan.required_ops)if(!loops.has(op)&&typeof kernels[op as OperationName]!=='function')throw Error(`Unavailable inference kernel: ${op}`);
  if(!context?.sequences?.length || !Number.isSafeInteger(tokens)||tokens<1)throw Error('Invalid execution context');
  if(!context.stateOwner)throw Error("Missing stage state owner");
  const slots=new Set<number>();
  for(const part of context.sequences){
    if(![part.slot,part.row,part.count,part.position].every(n=>Number.isSafeInteger(n)&&n>=0)||part.count<1||part.row+part.count>tokens||slots.has(part.slot))throw Error('Invalid sequence state scope');
    slots.add(part.slot);
    if(Number.isSafeInteger(context.variables.capacity)&&part.position+part.count>Number(context.variables.capacity))throw Error('State position exceeds capacity');
  }
  const ordered=[...context.sequences].sort((a,b)=>a.row-b.row);
  let nextRow=0;
  for(const part of ordered){
    if(part.row!==nextRow)throw Error('Sequence rows must cover tokens without overlap');
    nextRow+=part.count;
  }
  if(nextRow!==tokens||context.variables.tokens!==tokens)throw Error('Sequence token count mismatch');
  assertStateReady(context.stateOwner,[...slots]);
  const expandStart=performance.now();
  const commands:{kernel?:Kernel;args?:unknown[];layerEnd?:boolean;op?:string}[]=[];
  function expand(tasks:readonly ExecutableTask[],ctx:PlanContext,part:SequencePart,inSequence=false):void{
    for(const task of tasks){
      if(gpu.hostCache===false){
        const contract=operationContract(task.op,[...task.attrs.call],task.attrs.layer);
        checkContract(task.attrs.contract,contract);
        if(contract.effects.some(e=>e.scope==='layer_sequence')&&!inSequence)throw Error('State access requires a sequence loop');
      }
      if(task.attrs.when && !ctx.variables[task.attrs.when])continue;
      if(task.op==='for_each_sequence')for(const sequence of ctx.sequences)expand(task.attrs.body!,ctx,sequence,true);
      else if(task.op==='for_each_chunk'){
        const chunks=ctx.resources.output;if(!Array.isArray(chunks))throw Error('Missing output chunks');
        for(const chunk of chunks)expand(task.attrs.body!,{...ctx,resources:{...ctx.resources,chunk},variables:{...ctx.variables,chunk_first:record(chunk).first}},part,inSequence);
      }else if(task.op==='for_each_head_row'){
        for(let row=0;row<Number(ctx.variables.head_tokens);row++)expand(task.attrs.body!,ctx,{...part,row},inSequence);
      }else{
        const args=task.attrs.call.map(arg=>resolve(arg,ctx,task,part));
        validateResolved(task.op,args);
        validateBufferLayout(task.op,args,ctx.geometry);
        commands.push({kernel:kernels[task.op as OperationName],args,op:task.op});
      }
      if(task.attrs.layer_end)commands.push({layerEnd:true});
    }
  }
  expand(plan.tasks,context,context.sequences[0]); // Resolve everything before encoding GPU work.
  if(gpu.profiling)gpu.pendingExpandMs=performance.now()-expandStart;
  const per=Math.max(1,Math.floor(plan.submit_token_layers/tokens));
  const total=commands.filter(c=>c.layerEnd).length;
  const state=lifecycle(context.stateOwner);
  state.busy=true;
  try {
  upload?.();
  let count=0,encoded=0,start=performance.now();gpu.begin();
  const fuse=gpu.fuse!==false&&typeof gpu.normAdd==='function';
  for(let i=0;i<commands.length;i++){
    const command=commands[i];
    // A norm whose output the very next command adds into the residual, over exactly tokens x size (one group,
    // contiguous, no gate or offsets): one fused dispatch instead of two.
    const next=commands[i+1];
    if(fuse&&command.op==='rms_norm'&&next?.op==='add'){
      const [x,w,y,t,opts]=command.args as [unknown,unknown,unknown,number,Record<string,unknown>];
      const [a,res,n]=next.args as [unknown,unknown,number];
      const plain=Object.keys(opts||{}).every(k=>k==='size'||k==='eps'||(k==='groups'&&opts.groups===1));
      if(a===y&&plain&&typeof opts.size==='number'&&n===t*Number(opts.size)&&res!==y&&res!==x){
        gpu.normAdd!(x,w,y,res,t,Number(opts.size),Number(opts.eps??1e-6));
        i++;  // the add is done; its layer boundary (the next command) is still counted
        continue;
      }
    }
    // Two consecutive grouped norms of the same group size over different buffers (QK-norm): one dispatch.
    if(fuse&&typeof gpu.normPair==='function'&&command.op==='rms_norm'&&next?.op==='rms_norm'){
      const [xa,wa,ya,ta,oa]=command.args as [unknown,unknown,unknown,number,Record<string,unknown>];
      const [xb,wb,yb,tb,ob]=next.args as [unknown,unknown,unknown,number,Record<string,unknown>];
      const grouped=(o:Record<string,unknown>)=>Object.keys(o||{}).every(k=>k==='size'||k==='eps'||k==='groups')&&
        Number.isSafeInteger(o.groups)&&Number(o.groups)>=1&&typeof o.size==='number';
      const distinct=new Set([xa,ya,xb,yb]).size===4&&ya!==wb&&yb!==wa;
      if(grouped(oa)&&grouped(ob)&&ta===tb&&oa.size===ob.size&&(oa.eps??1e-6)===(ob.eps??1e-6)&&distinct){
        gpu.normPair(xa,wa,ya,Number(oa.groups),xb,wb,yb,Number(ob.groups),ta,Number(oa.size),Number(oa.eps??1e-6));
        i++;
        continue;
      }
    }
    // WS-36: consecutive one-token products of the same input (plain, or only an output offset) in one dispatch;
    // gate and up followed by silu_mul over the up output as one fused GLU dispatch. WS-49: a plain one-token
    // rms_norm whose output that group reads is folded into it (the norm output is still written).
    if(fuse&&typeof gpu.matmulGroup==='function'&&(command.op==='matmul'||(gpu.normFold&&command.op==='rms_norm'))){
      const plain=(c:typeof command)=>{
        if(c?.op!=='matmul')return null;
        const [W,x,y,t,opts]=c.args as [unknown,unknown,unknown,number,Record<string,unknown>|undefined];
        if(t!==1||Object.keys(opts||{}).some(k=>k!=='yOffset'&&k!=='yStride'))return null;
        if(x===y||(y&&typeof y==='object'&&'buffer' in (y as object)))return null;
        return {W,x,y,yOffset:Number(opts?.yOffset??0)};
      };
      let norm:{src:unknown;weight:unknown;eps:number}|null=null,at=i;
      if(command.op==='rms_norm'){
        const [nx,nw,ny,nt,no]=command.args as [unknown,unknown,unknown,number,Record<string,unknown>];
        const simple=Object.keys(no||{}).every(k=>k==='size'||k==='eps'||(k==='groups'&&no.groups===1));
        const objs=[nx,nw,ny].every(b=>!(b&&typeof b==='object'&&'buffer' in (b as object)));
        const W0=(plain(commands[i+1])?.W as {cols?:number}|undefined);
        if(!(nt===1&&simple&&objs&&nx!==ny&&W0&&no.size===W0.cols&&plain(commands[i+1])!.x===ny)){
          if(command.kernel)command.kernel(...command.args!);
          if(command.layerEnd){count++;encoded++;}
          continue;
        }
        norm={src:nx,weight:nw,eps:Number(no.eps??1e-6)};at=i+1;
      }
      const first=plain(commands[at]);
      const items=first?[first]:[];
      const limit=gpu.device?.limits?.maxStorageBuffersPerShaderStage??8;
      while(first&&items.length<4){
        const nxt=plain(commands[at+items.length]);
        if(!nxt||nxt.x!==first.x||items.some(it=>it.y===nxt.x))break;
        const cand=[...items,nxt],ys=new Set(cand.map(it=>it.y));
        if(norm&&[...ys].some(y=>y===norm!.src||y===norm!.weight))break;
        if(cand.length+1+ys.size+(norm?2:0)>limit||!gpu.groupShape!(cand.map(it=>it.W)))break;
        items.push(nxt);
      }
      if(items.length>=2){
        const after=commands[at+items.length];
        const [a,b,n]=(after?.op==='silu_mul'?after.args:[]) as [unknown,unknown,number];
        const W0=items[0].W as {rows:number};
        const glu=items.length===2&&after?.op==='silu_mul'&&a===items[0].y&&b===items[1].y&&n===W0.rows&&
          (items[1].W as {rows:number}).rows===W0.rows&&(items[1].W as {kind:string}).kind===(items[0].W as {kind:string}).kind&&items[0].yOffset===0&&items[1].yOffset===0;
        gpu.matmulGroup(items,first!.x,glu,norm);
        i=at+items.length-1+(glu?1:0);
        continue;
      }
      if(norm){if(command.kernel)command.kernel(...command.args!);continue;}
    }
    if(command.kernel)command.kernel(...command.args!);
    if(command.layerEnd){count++;encoded++;}
    if(count===per&&encoded<total){
      await gpu.end();submitMs.push(performance.now()-start);submitLayers.push(count);
      count=0;start=performance.now();gpu.begin();
    }
  }
  await gpu.end();submitMs.push(performance.now()-start);submitLayers.push(count);
  } catch(error) {
    for(const slot of slots)state.dirty.add(slot);
    throw error;
  } finally {state.busy=false;}
}

// WS-32: greedy decode with the token kept on the GPU. Each step's input id is copied on the GPU from the
// previous step's argmax (stage.token -> stage.ids), and the step is submitted before that token is read back,
// so the GPU never waits for the host. The host learns token i while step i runs; if token i ends the answer
// (`stop`), step i ran past the end: its position is rewound (stages with KV caches, `residentDecode`; WS-43:
// stages with recurrent state restore a snapshot taken before each step instead;
// the stale cache entry is overwritten before it is ever attended). `first`: the token the prefill produced
// (already in stage.token). Returns the tokens after `first`, at most `n`, the stopping token included.
export interface ResidentStage {
  readonly residentDecode?: boolean; token: GPUBuffer; ids: GPUBuffer; position: number; positions: number[];
  run(input: { tokens: number; slot: number; gpuIds: true }): Promise<void>;
  // WS-43: stages with recurrent state (Qwen3.5's DeltaNet) save it before each step and restore it to undo one.
  snapshot?(slot: number): void; restore?(slot: number): void;
}
export async function residentDecode(stage: ResidentStage, gpu: { device: GPUDevice; defer: boolean; profiling: unknown },
                                     first: number, n: number, slot: number, stop: (token: number) => boolean,
                                     onToken?: (token: number) => void): Promise<number[]> {
  const saves = !stage.residentDecode;  // WS-43: recurrent state, undone from a snapshot taken before each step
  if (saves && !(stage.snapshot && stage.restore)) throw new Error("This stage keeps recurrent state it cannot restore: no resident decode");
  const out: number[] = [];
  if (n < 1 || stop(first)) return out;
  const device = gpu.device, pool: GPUBuffer[] = [];
  const staging = () => pool.pop() ?? device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const saved = { defer: gpu.defer, profiling: gpu.profiling };
  gpu.defer = true; gpu.profiling = false;
  try {
    // Step 0 consumes `first` (already in stage.token); step i consumes token i, read back while step i runs.
    for (let i = 0; ; i++) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(stage.token, 0, stage.ids, 0, 4);
      const b = i > 0 ? staging() : null;
      if (b) enc.copyBufferToBuffer(stage.token, 0, b, 0, 4);
      device.queue.submit([enc.finish()]);
      // Mapped only after the copy into it is submitted (a buffer with a pending map cannot be used in a submit).
      const read = b ? b.mapAsync(GPUMapMode.READ).then(() => { const v = new Uint32Array(b.getMappedRange())[0]; b.unmap(); pool.push(b); return v; }) : null;
      const ran = i < n;  // step i produces token i+1; no step past the budget
      if (ran && saves && i > 0) stage.snapshot!(slot);  // queue order: before step i (step 0 is never undone)
      if (ran) await stage.run({ tokens: 1, slot, gpuIds: true });
      if (!read) continue;
      const token = await read;
      out.push(token);
      onToken?.(token);
      if (stop(token) || out.length >= n) {
        if (ran && saves) stage.restore!(slot);  // step i ran past the end: its state and position undone
        else if (ran) { stage.position -= 1; stage.positions[slot] = stage.position; }
        break;
      }
    }
    await device.queue.onSubmittedWorkDone();
  } finally {
    gpu.defer = saved.defer; gpu.profiling = saved.profiling;
    for (const b of pool) b.destroy();
  }
  return out;
}
