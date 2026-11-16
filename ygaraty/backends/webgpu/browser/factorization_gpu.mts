// Generated output: factorization_gpu.mjs.
// The factorization tile on WebGPU (experiment 48): one tile's data gradients and loss with X
// generated on the GPU from its seeded spec, as runtime/factorization_kernels.py does on CPU workers.
//
// X[r,c] = sum_k L[r,k] R[c,k] / sqrt(latent) + noise * h(row0+r, col0+c), where L and R are the
// tile's rows of the latent factors (models/factorization_sharded.py: X-left, X-right; NumPy's
// generator makes them, so they arrive as inputs, latent x (rows + columns) values instead of
// rows x columns) and h is the splitmix64 noise of hashed_noise, in 32-bit words (WGSL has no u64).
//
// Pass 1 writes the residual E = U V^T - X (f32). Pass 2 reduces it: dU = E V / total and the
// row's sum of squares; pass 3: dV = E^T U / total. The reductions over a tile's width use
// compensated (Kahan) sums. The result packs dU, dV and 0.5 sum(E^2) / total as the CPU kernel's
// [rows + columns + 1, rank]; the loss is summed over rows in float64 here.
import {tensor} from './sdk_tensor.mjs';
import {BufferScope, compileProgram, runReadback, checkedGPU} from './webgpu_kernels.mjs';
import type {Workgroups} from './webgpu_kernels.mjs';
import type {Tensor, LowRankSpec, FactorizationAttrs, FactorizationGroupAttrs, FactorizationResult, Timings} from './webgpu_contracts.mjs';

const MASK64=(1n<<64n)-1n;
const words=(x: bigint)=>[Number(x&0xffffffffn),Number((x>>32n)&0xffffffffn)];
// hashed_noise's salt: (seed * 0x9E3779B97F4A7C15) mod 2^64, as two 32-bit words.
// A seed above 2^53 cannot arrive intact as a JSON number, so browser jobs keep seeds below it.
export const saltWords=(seed: number)=>{
  if(!Number.isSafeInteger(seed)||seed<0)throw Error('seed must be an integer from 0 to 2^53 - 1');
  return words((BigInt(seed)*0x9E3779B97F4A7C15n)&MASK64);
};

export const FACTORIZATION_WGSL=`
struct Params {rows:u32, cols:u32, latent:u32, rank:u32,
                row0:u32, col0:u32, columns:u32, pad:u32,
                salt_lo:u32, salt_hi:u32, noise:f32, inv_sqrt_latent:f32,
                inv_total:f32, z0:u32, z1:u32, z2:u32};
@group(0) @binding(0) var<storage,read> L:array<f32>;
@group(0) @binding(1) var<storage,read> R:array<f32>;
@group(0) @binding(2) var<storage,read> U:array<f32>;
@group(0) @binding(3) var<storage,read> V:array<f32>;
@group(0) @binding(4) var<storage,read_write> E:array<f32>;
@group(0) @binding(5) var<storage,read_write> out:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;

// u64 as vec2<u32>(low, high).
fn mul32(a:u32, b:u32)->vec2<u32>{
  let a0=a&0xffffu; let a1=a>>16u; let b0=b&0xffffu; let b1=b>>16u;
  let p00=a0*b0; let p01=a0*b1; let p10=a1*b0; let p11=a1*b1;
  let mid=(p00>>16u)+(p01&0xffffu)+(p10&0xffffu);
  return vec2<u32>((p00&0xffffu)|(mid<<16u), p11+(p01>>16u)+(p10>>16u)+(mid>>16u));
}
fn add64(a:vec2<u32>, b:vec2<u32>)->vec2<u32>{
  let lo=a.x+b.x; return vec2<u32>(lo, a.y+b.y+select(0u,1u,lo<a.x));
}
fn mul64(a:vec2<u32>, b:vec2<u32>)->vec2<u32>{     // the low 64 bits of a*b
  let low=mul32(a.x,b.x); return vec2<u32>(low.x, low.y+a.x*b.y+a.y*b.x);
}
fn xorshift(a:vec2<u32>, n:u32)->vec2<u32>{        // a ^ (a >> n), 0 < n < 32
  return vec2<u32>(a.x^((a.x>>n)|(a.y<<(32u-n))), a.y^(a.y>>n));
}
// hashed_noise's splitmix64 of the entry's index in the whole matrix.
fn hashed(row:u32, col:u32)->vec2<u32>{
  var z=add64(add64(mul32(row,p.columns), vec2<u32>(col,0u)), vec2<u32>(p.salt_lo,p.salt_hi));
  z=mul64(xorshift(z,30u), vec2<u32>(0x1CE4E5B9u,0xBF58476Du));
  z=mul64(xorshift(z,27u), vec2<u32>(0x133111EBu,0x94D049BBu));
  return xorshift(z,31u);
}
fn noise_at(row:u32, col:u32)->f32{
  let z=hashed(row,col);
  let first=f32(z.y>>6u)/67108864.0;                             // bits 38..63
  let second=f32((z.x>>12u)|((z.y&0x3fu)<<20u))/67108864.0;      // bits 12..37
  return (first+second-1.0)*2.449489742783178;
}
// pad is a runtime zero: the integer round trip stops fast math from deleting Kahan's correction.
fn rounded(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^p.pad);}

@compute @workgroup_size(16,16) fn residual(@builtin(global_invocation_id) id:vec3<u32>){
  let r=id.y; let c=id.x; if(r>=p.rows||c>=p.cols){return;}
  // The latent sum in index order, then the division, as x_values does.
  var x=L[r*p.latent]*R[c*p.latent];
  for(var k=1u;k<p.latent;k++){x=x+L[r*p.latent+k]*R[c*p.latent+k];}
  x=x*p.inv_sqrt_latent;
  if(p.noise!=0.0){x=x+p.noise*noise_at(p.row0+r,p.col0+c);}
  var uv=0.0;
  for(var q=0u;q<p.rank;q++){uv=uv+U[r*p.rank+q]*V[c*p.rank+q];}
  E[r*p.cols+c]=uv-x;
}
@compute @workgroup_size(64) fn grad_u(@builtin(global_invocation_id) id:vec3<u32>){
  // One invocation per (row, q); q == rank: the row's sum of squares.
  let r=id.x/(p.rank+1u); let q=id.x%(p.rank+1u); if(r>=p.rows){return;}
  var s=0.0; var comp=0.0; let vq=min(q,p.rank-1u);
  for(var c=0u;c<p.cols;c++){
    let e=E[r*p.cols+c];
    let term=select(e*V[c*p.rank+vq], e*e, q==p.rank);
    let y=rounded(term-comp); let t=rounded(s+y); comp=rounded(rounded(t-s)-y); s=t;
  }
  if(q==p.rank){out[(p.rows+p.cols)*p.rank+r]=s;} else {out[r*p.rank+q]=s*p.inv_total;}
}
@compute @workgroup_size(64) fn grad_v(@builtin(global_invocation_id) id:vec3<u32>){
  let c=id.x/p.rank; let q=id.x%p.rank; if(c>=p.cols){return;}
  var s=0.0; var comp=0.0;
  for(var r=0u;r<p.rows;r++){
    let y=rounded(E[r*p.cols+c]*U[r*p.rank+q]-comp); let t=rounded(s+y); comp=rounded(rounded(t-s)-y); s=t;
  }
  out[(p.rows+c)*p.rank+q]=s*p.inv_total;
}
`;

// The same splitmix64 for chosen entries, as raw words: the hash must match NumPy bit for bit.
export const HASH_PROBE_WGSL=FACTORIZATION_WGSL.replace(/@compute[\s\S]*$/,'')+`
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>){
  if(id.x>=p.rows){return;}
  let z=hashed(bitcast<u32>(L[2u*id.x]), bitcast<u32>(L[2u*id.x+1u]));
  out[2u*id.x]=bitcast<f32>(z.x); out[2u*id.x+1u]=bitcast<f32>(z.y);
}`;

const validSpec=(s: LowRankSpec)=>s&&s.kind==='lowrank_x'&&(['rows','columns','latent','row_offset','column_offset','seed'] as const).every(k=>Number.isSafeInteger(s[k])&&s[k]>=0)
  &&Array.isArray(s.shape)&&s.shape.length===2&&Number.isFinite(s.noise);

// inputs: [L (rows x latent), R (columns x latent), U (rows x rank), V (columns x rank)]; attrs:
// x_spec and total_elements, as the CPU kernel's. Returns {value, timings}.
async function factorizationTileGPUUnchecked(device: GPUDevice,inputs: Tensor[],attrs: FactorizationAttrs,shape: number[]): Promise<FactorizationResult>{
  const spec=attrs?.x_spec,[L,R,U,V]=inputs;
  if(!validSpec(spec))throw Error('factorization_tile needs a lowrank_x x_spec whose integers (seed included) are below 2^53');
  const [rows,cols]=spec.shape,latent=spec.latent,rank=U.shape[1],total=attrs.total_elements;
  if(L.shape[0]!==rows||R.shape[0]!==cols||L.shape[1]!==latent||R.shape[1]!==latent||U.shape[0]!==rows||V.shape[0]!==cols||V.shape[1]!==rank)
    throw Error('factorization_tile input dimensions do not match its x_spec');
  if(!Number.isSafeInteger(total)||total<rows*cols)throw Error('total_elements must cover the tile');
  if(spec.columns>=2**32||spec.row_offset+rows>=2**32||spec.column_offset+cols>2**32||latent<1)throw Error('x_spec out of range');
  if(shape[0]!==rows+cols+1||shape[1]!==rank)throw Error('output shape must be [rows + columns + 1, rank]');
  const residualBytes=rows*cols*4;
  if(residualBytes>Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize))
    throw Error(`tile residual (${residualBytes} bytes) exceeds this device's buffer limit`);
  const [saltLo,saltHi]=saltWords(spec.seed);
  const params=new ArrayBuffer(64),u32=new Uint32Array(params),f32=new Float32Array(params);
  u32.set([rows,cols,latent,rank,spec.row_offset,spec.column_offset,spec.columns,0,saltLo,saltHi]);
  f32[10]=spec.noise;f32[11]=1/Math.sqrt(latent);f32[12]=1/total;
  // The output holds dU, dV, then one sum of squares per row (not the packed loss row).
  const outLength=(rows+cols)*rank+rows;
  const timings: Timings={upload_ms:0,gpu_ms:0};let t=performance.now();
  const {buffers,run}=await pipelines(device,FACTORIZATION_WGSL,['residual','grad_u','grad_v'],
    [L,R,U,V].map(x=>Float32Array.from(x.data)),residualBytes,outLength*4,params);
  try{
    timings.upload_ms=performance.now()-t;t=performance.now();
    const raw=await run([[Math.ceil(cols/16),Math.ceil(rows/16)],[Math.ceil(rows*(rank+1)/64)],[Math.ceil(cols*rank/64)]]);
    timings.gpu_ms=performance.now()-t;
    const values=new Float64Array(shape[0]*rank);
    for(let i=0;i<(rows+cols)*rank;i++)values[i]=raw[i];
    let loss=0;for(let r=0;r<rows;r++)loss+=raw[(rows+cols)*rank+r];
    values[(rows+cols)*rank]=0.5*loss/total;
    if(!values.every(Number.isFinite))throw Error('Nonfinite GPU output');
    return {value:tensor(shape,values),timings};
  }finally{for(const b of buffers)b.destroy();}
}

// factorization_tiles (a plan's group): the p x q tiles of row_shards x column_shards one at a time
// through factorizationTileGPU (one tile's GPU memory), their gradients summed per shard in float64
// in a fixed order, as runtime/factorization_kernels._tile_group does. Inputs: the shards' latent
// rows (p + q), then U_i (p) and V_j (q); the result packs one band per shard, then the loss.
export async function factorizationTilesGPU(device: GPUDevice,inputs: Tensor[],attrs: FactorizationGroupAttrs,shape: number[]): Promise<FactorizationResult>{
  const rs=attrs?.row_shards,cs=attrs?.column_shards,template=attrs?.x_spec,total=attrs?.total_elements;
  const bounds=(v: number[][])=>Array.isArray(v)&&v.length>=1&&v.length<=64&&v.every(b=>Array.isArray(b)&&b.length===2&&b.every(Number.isSafeInteger)&&b[0]>=0&&b[0]<b[1]);
  if(!bounds(rs)||!bounds(cs)||!template||typeof template!=='object')throw Error('factorization_tiles needs row_shards, column_shards and an x_spec');
  const p=rs.length,q=cs.length;
  if(inputs.length!==2*(p+q))throw Error('factorization_tiles on the GPU needs latent rows, U and V for each shard');
  const L=inputs.slice(0,p),R=inputs.slice(p,p+q),U=inputs.slice(p+q,2*p+q),V=inputs.slice(2*p+q);
  const rank=U[0].shape[1],sizes=[...rs,...cs].map(b=>b[1]-b[0]),rows=sizes.reduce((a,b)=>a+b,0);
  if(shape[0]!==rows+1||shape[1]!==rank)throw Error('output shape must be [sum of the shards + 1, rank]');
  const offsets=[];let at=0;for(const n of sizes){offsets.push(at);at+=n;}
  const out=new Float64Array(shape[0]*rank),timings={upload_ms:0,gpu_ms:0,tiles:0};let loss=0;
  for(let a=0;a<p;a++)for(let b=0;b<q;b++){
    const [r0,r1]=rs[a],[c0,c1]=cs[b],size=r1-r0,width=c1-c0;
    const spec: LowRankSpec={...template,shape:[size,width],row_offset:r0,column_offset:c0};
    const {value,timings:t}=await factorizationTileGPU(device,[L[a],R[b],U[a],V[b]],{x_spec:spec,total_elements:total},[size+width+1,rank]);
    timings.upload_ms+=t.upload_ms;timings.gpu_ms+=t.gpu_ms;timings.tiles++;
    const d=value.data,ro=offsets[a]*rank,co=offsets[p+b]*rank;
    for(let i=0;i<size*rank;i++)out[ro+i]+=d[i];
    for(let i=0;i<width*rank;i++)out[co+i]+=d[size*rank+i];
    loss+=d[(size+width)*rank];
  }
  out[rows*rank]=loss;
  if(!out.every(Number.isFinite))throw Error('Nonfinite GPU output');
  return {value:tensor(shape,out),timings};
}

// hashed_noise's 64-bit word for each [row, col] pair: [low, high] per pair.
async function hashProbeGPUUnchecked(device: GPUDevice,pairs: number[][],columns: number,seed: number){
  const [saltLo,saltHi]=saltWords(seed);
  const params=new ArrayBuffer(64),u32=new Uint32Array(params);
  u32.set([pairs.length,1,1,1,0,0,columns,0,saltLo,saltHi]);
  const packed=new Float32Array(new Uint32Array(pairs.flat()).buffer);
  const {buffers,run}=await pipelines(device,HASH_PROBE_WGSL,['probe'],[packed,new Float32Array(1),new Float32Array(1),new Float32Array(1)],4,pairs.length*8,params);
  try{
    const raw=await run([[Math.ceil(pairs.length/64)]]);
    return Array.from({length:pairs.length},(_,i)=>[new Uint32Array(raw.buffer)[2*i],new Uint32Array(raw.buffer)[2*i+1]]);
  }finally{for(const b of buffers)b.destroy();}
}

async function pipelines(device: GPUDevice,code: string,entries: string[],inputs: Float32Array<ArrayBuffer>[],residualBytes: number,outBytes: number,params: ArrayBuffer){
  const scope=new BufferScope(device);
  try {
    const kinds: GPUBufferBindingType[]=['read-only-storage','read-only-storage','read-only-storage','read-only-storage','storage','storage','uniform'];
    const steps=await compileProgram(device,code,entries,kinds);
    const ins=inputs.map(x=>scope.upload(x,GPUBufferUsage.STORAGE));
    const E=scope.empty(residualBytes,GPUBufferUsage.STORAGE);
    const out=scope.empty(outBytes,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const P=scope.upload(new Uint8Array(params),GPUBufferUsage.UNIFORM);
    return {buffers:scope.buffers,run:(dispatches: Workgroups[])=>runReadback(scope,steps,[...ins,E,out,P],dispatches,out,outBytes)};
  } catch(error) { scope.destroy(); throw error; }
}

export function factorizationTileGPU(device: GPUDevice,inputs: Tensor[],attrs: FactorizationAttrs,shape: number[]): Promise<FactorizationResult> {
  return checkedGPU(device,()=>factorizationTileGPUUnchecked(device,inputs,attrs,shape));
}

export function hashProbeGPU(device: GPUDevice,pairs: number[][],columns: number,seed: number): Promise<number[][]> {
  return checkedGPU(device,()=>hashProbeGPUUnchecked(device,pairs,columns,seed));
}
