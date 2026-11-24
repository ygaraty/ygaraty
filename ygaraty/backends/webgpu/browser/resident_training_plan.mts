// Generated output: resident_training_plan.mjs. Edit this TypeScript source.
// Validate a portable Tape plan before any device work; no model-specific graph.
import {CONV_OPS, convolutionShapeContract} from './convolution.mjs';
import {RESIDENT_OPS} from './resident_training_kernels.mjs';
import type {ConvolutionOp, ConvolutionAttrs} from './webgpu_contracts.mjs';
import type {ResidentTrainingRequest, StepPlan, Task, Moments} from './training_contracts.mjs';
export const sameShape=(a: number[],b: number[])=>a.length===b.length&&a.every((n,i)=>n===b[i]);
export function stateReferences(plan: StepPlan): Moments<string> {
 return plan.state_parameters || {m:Object.fromEntries(Object.entries(plan.parameters).map(([n,p])=>[n,`adam_m_${p}`])),v:Object.fromEntries(Object.entries(plan.parameters).map(([n,p])=>[n,`adam_v_${p}`]))};
}
export function regionShape(shape: number[], region?: number[] | null): number[] {
 if(!region)return shape;
 const [r,end,c,right]=region;
 if(region.length!==4 || !region.every(Number.isSafeInteger) || r<0 || end>shape[0] || end<=r || c<0 || right>shape[1] || right<=c)throw Error('Invalid resident input region');
 return [end-r,right-c];
}
const shapeOK=(s:number[])=>s.length===2&&s.every(x=>Number.isSafeInteger(x)&&x>0)&&s[0]*s[1]*8<=64*1024*1024;
const names=(x:object)=>Object.keys(x).sort().join('\0');
const finite=(attrs:Record<string,unknown>,key:string,fallback?:number):number=>{
 const x=attrs[key]??fallback;if(typeof x!=='number'||!Number.isFinite(x))throw Error(`Invalid resident ${key}`);return x;
};
export function residentOp(op:string):string {return ({add_bias:'add',broadcast:'copy'} as Record<string,string>)[op]??op;}
export function validateResidentTask(t:Task,args:number[][]):void {
 const fail=()=>{throw Error(`Invalid resident ${t.op} contract`);};
 const exact=(count:number)=>{if(args.length!==count||args.some(s=>!sameShape(s,t.shape)))fail();};
 const broadcast=(count:number)=>{if(args.length!==count||args.some(s=>s.some((n,i)=>n!==1&&n!==t.shape[i])))fail();};
 if((CONV_OPS as readonly string[]).includes(t.op)) {
  convolutionShapeContract(t.op as ConvolutionOp,t.attrs as unknown as ConvolutionAttrs,args,t.shape);return;
 }
 switch(residentOp(t.op)){
  case 'matmul':if(args.length!==2||args[0][1]!==args[1][0]||!sameShape(t.shape,[args[0][0],args[1][1]]))fail();break;
  case 'transpose':if(args.length!==1||!sameShape(t.shape,[args[0][1],args[0][0]]))fail();break;
  case 'sum':case 'sum_squares':if(args.length!==1||!sameShape(t.shape,[1,1]))fail();break;
  case 'reduce_rows':if(args.length!==1||!sameShape(t.shape,[1,args[0][1]]))fail();break;
  case 'assemble':{
   const blocks=t.attrs.blocks as number[][];
   if(!Array.isArray(blocks)||!args.length||blocks.length!==args.length)fail();
   let area=0;
   blocks.forEach((b,i)=>{
    if(!Array.isArray(b)||!sameShape(regionShape(t.shape,b),args[i]))fail();
    area+=args[i][0]*args[i][1];
    for(let j=0;j<i;j++){const a=blocks[j];if(Math.max(a[0],b[0])<Math.min(a[1],b[1])&&Math.max(a[2],b[2])<Math.min(a[3],b[3]))fail();}
   });
   if(area!==t.shape[0]*t.shape[1])fail();break;
  }
  case 'add_many':if(!args.length)fail();exact(args.length);break;
  case 'cross_entropy':case 'ce_backward':{
   if(args.length!==(t.op==='cross_entropy'?1:2)||!sameShape(t.shape,t.op==='cross_entropy'?[1,1]:args[0]))fail();
   if(t.op==='ce_backward'&&!sameShape(args[1],[1,1]))fail();
   const targets=t.attrs.targets as number[],ignore=t.attrs.ignore_index;
   if(!Array.isArray(targets)||targets.length!==args[0][0]||targets.some(x=>!Number.isSafeInteger(x)||(x!==ignore&&(x<0||x>=args[0][1]))))fail();
   const count=finite(t.attrs,'count',targets.filter(x=>x!==ignore).length);
   const smoothing=finite(t.attrs,'label_smoothing',0);
   if(!Number.isSafeInteger(count)||count<1||smoothing<0||smoothing>1)fail();break;
  }
  case 'scale':broadcast(1);finite(t.attrs,'factor');break;
  case 'inverse_norm':broadcast(1);if(finite(t.attrs,'epsilon')<=0)fail();break;
  case 'adam_m':case 'adam_v':case 'adam_update':{
   exact(t.op==='adam_update'?3:2);
   for(const key of ['beta1','beta2']){const x=finite(t.attrs,key);if(x<0||x>=1)fail();}
   if(t.op==='adam_update'){
    if(finite(t.attrs,'epsilon')<=0||finite(t.attrs,'learning_rate')<=0||!Number.isSafeInteger(t.attrs.step)||Number(t.attrs.step)<1)fail();
   }break;
  }
  case 'clip_gradient':if(args.length!==2||!sameShape(args[0],t.shape)||!sameShape(args[1],[1,1])||finite(t.attrs,'clip_norm')<0)fail();break;
  case 'sgd':broadcast(2);if(finite(t.attrs,'learning_rate')<=0)fail();break;
  case 'copy':case 'relu':broadcast(1);break;
  case 'add':case 'mul':case 'relu_backward':broadcast(2);break;
  default:fail();
 }
}
/** Unsupported operation/size returns false. Malformed supported plans throw. */
export function supportsResidentTraining(request:ResidentTrainingRequest):boolean {
 const uniformBytes=request.steps.reduce((n,s)=>n+s.plan.tasks.reduce((m,t)=>m+5+t.inputs.length,0)*256,0);
 if(uniformBytes>64*1024*1024)return false;
 const supported=new Set<string>([...RESIDENT_OPS,...CONV_OPS,'matmul','assemble','add_many','add_bias','broadcast']);
 if(!request.steps.length||request.steps.length>256||request.steps.reduce((n,s)=>n+s.plan.tasks.length,0)>20000)return false;
 if(request.steps.some(s=>s.plan.tasks.some(t=>!supported.has(t.op))))return false;
 let uploadBytes=0;
 for(const value of [...Object.values(request.weights),...Object.values(request.moments.m),...Object.values(request.moments.v),...request.steps.flatMap(s=>Object.values(s.constants))]){
  if(!shapeOK(value.shape)||value.data.length!==value.shape[0]*value.shape[1]||!value.data.every(x=>Number.isFinite(x)&&Number.isFinite(Math.fround(x))))throw Error('Invalid resident tensor');
  uploadBytes+=value.data.length*8;
 }
 if(uploadBytes>64*1024*1024)return false;
 for(const {plan,constants} of request.steps){
  if(names(plan.parameters)!==names(request.weights))throw Error('Resident model changed');
  const shapes=new Map<string,number[]>();
  const add=(id:string,shape:number[])=>{if(shapes.has(id))throw Error('Duplicate resident value');shapes.set(id,shape);};
  for(const [id,value] of Object.entries(constants))add(id,value.shape);
  for(const [name,id] of Object.entries(plan.parameters))add(id,request.weights[name].shape);
  const refs=stateReferences(plan);
  for(const kind of ['m','v'] as const){
   if(names(refs[kind])!==names(request.moments[kind]))throw Error('Resident state layout changed');
   for(const [name,id] of Object.entries(refs[kind]))add(id,request.moments[kind][name].shape);
  }
  for(const t of plan.tasks){
   if(!shapeOK(t.shape)||Math.ceil(t.shape[0]*t.shape[1]/64)>65535)return false;
   const args=t.inputs.map(ref=>{
    const shape=shapes.get(ref.value);if(!shape)throw Error('Unordered resident plan');return regionShape(shape,ref.region);
   });
   validateResidentTask(t,args);add(t.id,t.shape);
  }
  const loss=shapes.get(plan.loss);if(!loss||!sameShape(loss,[1,1]))throw Error('Invalid resident loss');
  for(const id of [...Object.values(plan.parameters),...Object.values(refs.m),...Object.values(refs.v)]){
   const shape=shapes.get(plan.updates[id]);if(!shape||!sameShape(shape,shapes.get(id)!))throw Error('Invalid resident update');
  }
 }
 return true;
}
