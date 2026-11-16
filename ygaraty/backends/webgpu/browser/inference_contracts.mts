// Schema validation and conservative effects. No GPU work or model topology.
import {INFERENCE_SCHEMA as schema} from './inference_schema.mjs';
import type {ArgumentSpec, Expression, OperationName, OperationArguments} from './inference_schema.mjs';
export type {Expression} from './inference_schema.mjs';
export interface ResourceEffect {
  argument:string; binding:Record<string, Expression>; access:string;
  precision:string; lifetime:string; scope:string; transition:string;
}
export interface OperationContract {schema:number; arguments:string[]; effects:ResourceEffect[]}
const operations:Record<string,readonly ArgumentSpec[]> = schema.operations;
const bindings:Record<string,readonly string[]> = schema.bindings;
const fields:Record<string,Record<string,string>> = schema.options;
export const stateContract = schema.state;
export const loopOperations = new Set<string>(schema.loops);
const resourceTags = new Set(['buffer','matrix','vector','state','layer_buffer','resource']);
function object(value:unknown):Record<string,unknown> {
  if(!value || typeof value!=='object' || Array.isArray(value))throw Error('Invalid contract object');
  return value as Record<string,unknown>;
}
function keys(value:Record<string,unknown>, allowed:string[], required=allowed):void {
  if(Object.keys(value).some(k=>!allowed.includes(k)) || required.some(k=>!Object.hasOwn(value,k)))throw Error('Invalid contract fields');
}
function expression(value:unknown,kind:string,op:string,depth=0):void {
  if(depth>20)throw Error('Plan expression nesting limit');
  if(value && typeof value==='object' && !Array.isArray(value) && '$' in value){
    const v=object(value),tag=String(v.$);
    if(tag==='choose'){
      keys(v,['$','condition','yes','no']);
      if(v.condition!=='batch' && v.condition!=='head_all')throw Error('Invalid choice expression');
      expression(v.yes,kind,op,depth+1);expression(v.no,kind,op,depth+1);return;
    }
    if(tag==='var' && ['integer','number','boolean','precision'].includes(kind)){
      keys(v,['$','name']);
      const expected=(schema.variables as Record<string,string>)[String(v.name)];
      if(expected!==kind && !(kind==='number'&&expected==='integer'))throw Error('Invalid runtime variable');
      return;
    }
    if(tag==='product' && ['integer','number'].includes(kind)){
      keys(v,['$','values']);
      if(!Array.isArray(v.values)||!v.values.length)throw Error('Invalid product expression');
      for(const item of v.values)expression(item,kind,op,depth+1);return;
    }
    if(!bindings[kind]?.includes(tag))throw Error(`Invalid ${kind} binding: ${tag}`);
    keys(v,tag==='buffer'?['$','name','stride','view']:['$','name'],['$','name']);
    if(typeof v.name!=='string'||!v.name)throw Error('Invalid resource binding');
    if(v.stride!==undefined){
      if(!Number.isSafeInteger(v.stride)||Number(v.stride)<=0||!['batch','always','head'].includes(String(v.view)))throw Error('Invalid buffer view');
    }else if(v.view!==undefined)throw Error('Buffer view needs a stride');
    return;
  }
  if(kind==='options'){
    const v=object(value),options=fields[op];
    keys(v,Object.keys(options),op==='rms_norm'?['size']:[]);
    for(const [name,item] of Object.entries(v))expression(item,options[name],op,depth+1);
    return;
  }
  const valid=kind==='boolean'&&typeof value==='boolean' ||
    kind==='integer'&&Number.isSafeInteger(value)&&Number(value)>=0 ||
    kind==='number'&&typeof value==='number'&&Number.isFinite(value) ||
    kind==='precision'&&(value==='f16'||value==='f32');
  if(!valid)throw Error(`Invalid ${kind} argument`);
}
function resources(value:unknown,path:string):[string,Record<string,Expression>][] {
  if(!value || typeof value!=='object'||Array.isArray(value))return [];
  const v=value as Record<string,Expression>;
  if(v.$==='choose')return [...resources(v.yes,`${path}.yes`),...resources(v.no,`${path}.no`)];
  if(resourceTags.has(String(v.$)))return [[path,v]];
  return v.$===undefined?Object.entries(v).flatMap(([k,x])=>resources(x,`${path}.${k}`)):[];
}
/** Named-argument API; positional calls remain the stable kernel ABI. */
export function encodeCall<K extends OperationName>(op:K,args:OperationArguments[K],layer:number|null):Expression[]{
  const specs=operations[op],values=args as Record<string,Expression>;
  keys(values,specs.map(a=>a.name),specs.filter(a=>!a.optional).map(a=>a.name));
  const call=specs.filter(a=>Object.hasOwn(values,a.name)).map(a=>values[a.name]);
  operationContract(op,call,layer);return call;
}
export function operationContract(op:string,call:Expression[],layer:number|null):OperationContract {
  if(layer!==null&&(!Number.isSafeInteger(layer)||layer<0))throw Error('Invalid layer resource');
  if(loopOperations.has(op)){
    if(call.length)throw Error('Loop has primitive arguments');
    return {schema:schema.version,arguments:[],effects:[]};
  }
  const args=Object.hasOwn(operations,op)?operations[op]:undefined;
  if(!args)throw Error(`Unsupported primitive: ${op}`);
  if(call.length<args.filter(a=>!a.optional).length||call.length>args.length)throw Error(`Invalid ${op} arity`);
  const effects:ResourceEffect[]=[];
  for(let i=0;i<call.length;i++){
    const a=args[i],value=call[i];expression(value,a.kind,op);
    for(const [path,binding] of resources(value,a.name)){
      const tag=String(binding.$);
      if(['matrix','vector','layer_buffer','state'].includes(tag)&&layer===null)throw Error('Layer resource without a layer');
      let access=a.kind==='options'?'read':a.access;
      if(op==='matmul'&&a.name==='output'&&call.length===5){
        const options=object(call[4]);
        if(options.$==='choose'||options.accumulate)access='read_write';
      }
      const persistent=tag==='state';
      const lifetime=persistent?'session':tag!=='buffer'||['frequencies','weight'].includes(a.name)?'checkpoint':'invocation';
      effects.push({argument:path,binding:structuredClone(binding),access,
        precision:a.kind==='options'?'f32':a.precision,
        lifetime:access==='scratch'?'operation':lifetime,
        scope:persistent?'layer_sequence':'stage',
        transition:persistent?(op.startsWith('cache_store')?'append':access==='read_write'?'update':'read'):'none'});
    }
  }
  return {schema:schema.version,arguments:args.slice(0,call.length).map(a=>a.name),effects};
}
function canonical(value:unknown):string {
  if(Array.isArray(value))return '['+value.map(canonical).join(',')+']';
  if(value && typeof value==='object')return '{'+Object.entries(value).sort(([a],[b])=>a.localeCompare(b)).map(([k,v])=>JSON.stringify(k)+':'+canonical(v)).join(',')+'}';
  return JSON.stringify(value);
}
export function checkContract(actual:unknown,expected:unknown):void {
  if(canonical(actual)!==canonical(expected))throw Error('Inference resource contract differs from operation');
}
/** Check resolved scalar/options arguments before any GPU command is encoded. */
export function validateResolved(op:string,call:unknown[]):void {
  const args=operations[op];
  for(let i=0;i<call.length;i++){
    const kind=args[i].kind;
    if(['integer','number','boolean','precision'].includes(kind))expression(call[i],kind,op);
    if(bindings[kind] && (!call[i]||typeof call[i]!=='object'))throw Error(`Invalid resolved ${kind} resource`);
    if(kind==='options'){
      const options=object(call[i]);
      for(const [key,value] of Object.entries(options)){
        if(key!=='gate')expression(value,fields[op][key],op);
      }
    }
  }
  const byName=Object.fromEntries(args.map((a,i)=>[a.name,call[i]]));
  for(const name of ['tokens','elements','capacity','heads','kv_heads','head_dim','stride','width','experts','top_k']){
    if(byName[name]!==undefined && Number(byName[name])<=0)throw Error(`Invalid positive ${name}`);
  }
  if(byName.position!==undefined && byName.capacity!==undefined &&
      Number(byName.position)+Number(byName.tokens)>Number(byName.capacity))throw Error('State position exceeds capacity');
  if(op==='rms_norm' && Number(object(byName.options).size)<=0)throw Error('Invalid norm width');
}
