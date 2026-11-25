// Generated output: convolution.mjs.
// Small NCHW ConvNet primitives. WGSL uses float32; CPU is a float64 oracle.
// Each invocation owns one output element, including gradient reductions.
import {CONV_OPS} from './webgpu_contracts.mjs';
export {CONV_OPS} from './webgpu_contracts.mjs';
import type {ConvolutionOp, ConvolutionAttrs, Tensor} from './webgpu_contracts.mjs';
import {BufferScope, compileProgram, runReadback, checkedGPU} from './webgpu_kernels.mjs';

export function convolutionShapeContract(op: ConvolutionOp,attrs: ConvolutionAttrs,inputs: number[][],shape: number[]){
  const index=CONV_OPS.indexOf(op);if(index<0)throw Error('Unknown convolution operation');
  const keys=['batch','in_channels','height','width'] as const,limits=[64,256,128,128];
  const dims=keys.map((key,i)=>{const v=attrs[key];if(!Number.isSafeInteger(v)||v<1||v>limits[i])throw Error('Invalid '+key);return v;});
  const [n,c,h,w]=dims,o=attrs.out_channels??c,k=attrs.kernel??3;
  if(!Number.isSafeInteger(o)||o<1||o>256||![1,3].includes(k))throw Error('Invalid output channels or kernel');
  const x=[n,c*h*w],y=[n,o*h*w],weight=[o,c*k*k];let expected,out;
  if(index===0){expected=[x,weight,[1,o]];out=y;}
  else if(index===1){expected=[y,weight];out=x;}
  else if(index===2){expected=[x,y];out=weight;}
  else if(index===3){expected=[y];out=[1,o];}
  else if(index===4||index===5){
    if(h%2||w%2)throw Error('Max pooling requires even spatial dimensions');
    const pooled=[n,c*(h/2)*(w/2)];expected=index===4?[x]:[x,pooled];out=index===4?pooled:x;
  }else{expected=index===6?[x]:[[n,c]];out=index===6?[n,c]:x;}
  const same=(a: number[],b: number[])=>Array.isArray(a)&&a.length===2&&a.every((v,i)=>v===b[i]);
  if(!same(shape,out)||inputs.length!==expected.length)throw Error('Convolution input/output contract mismatch');
  for(let i=0;i<inputs.length;i++){
    const v=inputs[i],s=expected[i];
    if(!v||!same(v,s))throw Error('Invalid convolution input shape');
  }
  if([...expected,out].some(s=>s[0]*s[1]*8>64*1024*1024))throw Error('Convolution tensor exceeds 64 MiB');
  return [n,c,h,w,o,k,index,out[0]*out[1]];
}

export function convolutionContract(op: ConvolutionOp,attrs: ConvolutionAttrs,inputs: Tensor[],shape: number[]){
  const dims=convolutionShapeContract(op,attrs,inputs.map(v=>v.shape),shape);
  for(const v of inputs)if(v.data?.length!==v.shape[0]*v.shape[1]||!v.data.every(Number.isFinite))throw Error('Invalid convolution input');
  return dims;
}

export function convolutionCPU(op: ConvolutionOp,inputs: Tensor[],attrs: ConvolutionAttrs,shape: number[]): Tensor{
  const [N,C,H,W,O,K,mode,total]=convolutionContract(op,attrs,inputs,shape),P=K>>1;
  const [a,b,bias]=inputs.map(v=>v.data),out=new Float64Array(total);
  const xi=(n: number,c: number,y: number,x: number)=>((n*C+c)*H+y)*W+x, yi=(n: number,o: number,y: number,x: number)=>((n*O+o)*H+y)*W+x;
  const wi=(o: number,c: number,y: number,x: number)=>((o*C+c)*K+y)*K+x;
  for(let i=0;i<total;i++){
    let sum=0;
    if(mode===0){
      const x=i%W,y=Math.floor(i/W)%H,o=Math.floor(i/(W*H))%O,n=Math.floor(i/(W*H*O));sum=bias[o];
      for(let c=0;c<C;c++)for(let ky=0;ky<K;ky++)for(let kx=0;kx<K;kx++){
        const yy=y+ky-P,xx=x+kx-P;if(yy>=0&&yy<H&&xx>=0&&xx<W)sum+=a[xi(n,c,yy,xx)]*b[wi(o,c,ky,kx)];
      }
    }else if(mode===1){
      const x=i%W,y=Math.floor(i/W)%H,c=Math.floor(i/(W*H))%C,n=Math.floor(i/(W*H*C));
      for(let o=0;o<O;o++)for(let ky=0;ky<K;ky++)for(let kx=0;kx<K;kx++){
        const yy=y+P-ky,xx=x+P-kx;if(yy>=0&&yy<H&&xx>=0&&xx<W)sum+=a[yi(n,o,yy,xx)]*b[wi(o,c,ky,kx)];
      }
    }else if(mode===2){
      const kx=i%K,ky=Math.floor(i/K)%K,c=Math.floor(i/(K*K))%C,o=Math.floor(i/(K*K*C));
      for(let n=0;n<N;n++)for(let y=0;y<H;y++)for(let x=0;x<W;x++){
        const yy=y+ky-P,xx=x+kx-P;if(yy>=0&&yy<H&&xx>=0&&xx<W)sum+=a[xi(n,c,yy,xx)]*b[yi(n,o,y,x)];
      }
    }else if(mode===3){for(let n=0;n<N;n++)for(let y=0;y<H;y++)for(let x=0;x<W;x++)sum+=a[yi(n,i,y,x)];
    }else if(mode===4||mode===5){
      const width=mode===4?W/2:W,height=mode===4?H/2:H;
      const x=i%width,y=Math.floor(i/width)%height,c=Math.floor(i/(width*height))%C,n=Math.floor(i/(width*height*C));
      const px=mode===4?x:Math.floor(x/2),py=mode===4?y:Math.floor(y/2);
      let winner=xi(n,c,py*2,px*2),best=a[winner];
      for(let t=1;t<4;t++){const at=xi(n,c,py*2+Math.floor(t/2),px*2+t%2);if(a[at]>best){best=a[at];winner=at;}}
      sum=mode===4?best:(i===winner?b[((n*C+c)*(H/2)+py)*(W/2)+px]:0);
    }else if(mode===6){const n=Math.floor(i/C),c=i%C;for(let y=0;y<H;y++)for(let x=0;x<W;x++)sum+=a[xi(n,c,y,x)];sum/=H*W;
    }else{sum=a[Math.floor(i/(H*W))]/(H*W);}
    out[i]=sum;
  }
  if(!out.every(Number.isFinite))throw Error('Nonfinite convolution output');
  return {shape:[...shape],data:out};
}

export const CONV_WGSL=`
struct Dims { n:u32, ci:u32, h:u32, w:u32, co:u32, k:u32, op:u32, total:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@group(0) @binding(4) var<uniform> d:Dims;
fn xi(n:u32,c:u32,y:u32,x:u32)->u32{return ((n*d.ci+c)*d.h+y)*d.w+x;}
fn yi(n:u32,c:u32,y:u32,x:u32)->u32{return ((n*d.co+c)*d.h+y)*d.w+x;}
fn wi(o:u32,c:u32,y:u32,x:u32)->u32{return ((o*d.ci+c)*d.k+y)*d.k+x;}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=d.total){return;}let p=i32(d.k/2u);var sum=0.0;
  if(d.op==0u){
    let x=i%d.w;let y=(i/d.w)%d.h;let o=(i/(d.w*d.h))%d.co;let n=i/(d.w*d.h*d.co);sum=bias[o];
    for(var c=0u;c<d.ci;c++){for(var ky=0u;ky<d.k;ky++){for(var kx=0u;kx<d.k;kx++){
      let yy=i32(y)+i32(ky)-p;let xx=i32(x)+i32(kx)-p;
      if(yy>=0&&yy<i32(d.h)&&xx>=0&&xx<i32(d.w)){sum+=a[xi(n,c,u32(yy),u32(xx))]*b[wi(o,c,ky,kx)];}
    }}}
  }else if(d.op==1u){
    let x=i%d.w;let y=(i/d.w)%d.h;let c=(i/(d.w*d.h))%d.ci;let n=i/(d.w*d.h*d.ci);
    for(var o=0u;o<d.co;o++){for(var ky=0u;ky<d.k;ky++){for(var kx=0u;kx<d.k;kx++){
      let yy=i32(y)+p-i32(ky);let xx=i32(x)+p-i32(kx);
      if(yy>=0&&yy<i32(d.h)&&xx>=0&&xx<i32(d.w)){sum+=a[yi(n,o,u32(yy),u32(xx))]*b[wi(o,c,ky,kx)];}
    }}}
  }else if(d.op==2u){
    let kx=i%d.k;let ky=(i/d.k)%d.k;let c=(i/(d.k*d.k))%d.ci;let o=i/(d.k*d.k*d.ci);
    for(var n=0u;n<d.n;n++){for(var y=0u;y<d.h;y++){for(var x=0u;x<d.w;x++){
      let yy=i32(y)+i32(ky)-p;let xx=i32(x)+i32(kx)-p;
      if(yy>=0&&yy<i32(d.h)&&xx>=0&&xx<i32(d.w)){sum+=a[xi(n,c,u32(yy),u32(xx))]*b[yi(n,o,y,x)];}
    }}}
  }else if(d.op==3u){
    for(var n=0u;n<d.n;n++){for(var y=0u;y<d.h;y++){for(var x=0u;x<d.w;x++){sum+=a[yi(n,i,y,x)];}}}
  }else if(d.op==4u||d.op==5u){
    let width=select(d.w,d.w/2u,d.op==4u);let height=select(d.h,d.h/2u,d.op==4u);
    let x=i%width;let y=(i/width)%height;let c=(i/(width*height))%d.ci;let n=i/(width*height*d.ci);
    let px=select(x/2u,x,d.op==4u);let py=select(y/2u,y,d.op==4u);
    var winner=xi(n,c,py*2u,px*2u);var best=a[winner];
    for(var t=1u;t<4u;t++){let at=xi(n,c,py*2u+t/2u,px*2u+t%2u);if(a[at]>best){best=a[at];winner=at;}}
    if(d.op==4u){sum=best;}else if(i==winner){sum=b[((n*d.ci+c)*(d.h/2u)+py)*(d.w/2u)+px];}
  }else if(d.op==6u){
    let n=i/d.ci;let c=i%d.ci;for(var y=0u;y<d.h;y++){for(var x=0u;x<d.w;x++){sum+=a[xi(n,c,y,x)];}}sum/=f32(d.h*d.w);
  }else{sum=a[i/(d.h*d.w)]/f32(d.h*d.w);}
  out[i]=sum;
}`;

async function convolutionGPUUnchecked(device: GPUDevice,op: ConvolutionOp,inputs: Tensor[],attrs: ConvolutionAttrs,shape: number[]): Promise<Tensor>{
  const dims=convolutionContract(op,attrs,inputs,shape),size=dims[7]*4;
  if(Math.ceil(dims[7]/64)>device.limits.maxComputeWorkgroupsPerDimension)throw Error('Convolution dispatch exceeds device limit');
  if([size,...inputs.map(v=>v.data.length*4)].some(n=>n>Math.min(device.limits.maxStorageBufferBindingSize,device.limits.maxBufferSize)))
    throw Error('Convolution buffer exceeds device limit');
  const scope=new BufferScope(device);
  try {
    const steps=await compileProgram(device,CONV_WGSL,['main']);
    const sources=[0,1,2].map(i=>{
      const data=i<inputs.length?Float32Array.from(inputs[i].data):new Float32Array(1);
      if(!data.every(Number.isFinite))throw Error('Convolution input overflows float32');
      return scope.upload(data,GPUBufferUsage.STORAGE);
    });
    const output=scope.upload(new Float32Array(dims[7]),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const uniform=scope.upload(new Uint32Array(dims),GPUBufferUsage.UNIFORM);
    const raw=await runReadback(scope,steps,[...sources,output,uniform],[[Math.ceil(dims[7]/64)]],output,size);
    const values=Float64Array.from(raw);
    if(!values.every(Number.isFinite))throw Error('Nonfinite WebGPU convolution result');
    return {shape:[...shape],data:values};
  } finally { scope.destroy(); }
}

export function convolutionGPU(device: GPUDevice,op: ConvolutionOp,inputs: Tensor[],attrs: ConvolutionAttrs,shape: number[]): Promise<Tensor> {
  return checkedGPU(device,()=>convolutionGPUUnchecked(device,op,inputs,attrs,shape));
}
