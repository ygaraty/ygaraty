// Generated output: resident_training_kernels.mjs. Edit this TypeScript source.
import {COMPENSATED_ARITHMETIC_WGSL} from './training_gpu.mjs';
import {CONV_WGSL} from './convolution.mjs';

export const RESIDENT_OPS = ['copy','add','mul','scale','relu','relu_backward',
  'transpose','sum','sum_squares','reduce_rows','inverse_norm','adam_m','adam_v',
  'adam_update','cross_entropy','ce_backward','clip_gradient','sgd'] as const;
// Split constants on the host; no tensor computation or shader text from plans.
const pair = (x: number) => `vec2<f32>(${Math.fround(x)},${Math.fround(x-Math.fround(x))})`;
let factorial = 1;
const coefficients = [1];
for (let i=1;i<=12;i++) {factorial *= i; coefficients.push(1/factorial);}
const polynomial = coefficients.slice(0,-1).reverse().map(c=>`p=add_pair(${pair(c)},mul_pair(r,p));`).join('\n');

export const RESIDENT_TRAINING_WGSL = `
struct Dims {
 m:u32,n:u32,k:u32,pad:u32,
 a:vec4<u32>,b:vec4<u32>,c:vec4<u32>,
 p0:vec2<f32>,p1:vec2<f32>,p2:vec2<f32>,p3:vec2<f32>,p4:vec2<f32>,p5:vec2<f32>,
 flags:vec4<u32>,dest:vec4<u32>
};
@group(0) @binding(0) var<storage,read> a:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> b:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> c:array<vec2<f32>>;
@group(0) @binding(3) var<storage,read_write> out:array<vec2<f32>>;
@group(0) @binding(4) var<uniform> d:Dims;
@group(0) @binding(5) var<storage,read_write> fault:atomic<u32>;
@group(0) @binding(6) var<storage,read> targets:array<u32>;
${COMPENSATED_ARITHMETIC_WGSL}
fn sub_pair(x:vec2<f32>,y:vec2<f32>)->vec2<f32>{return add_pair(x,-y);}
fn div_pair(x:vec2<f32>,y:vec2<f32>)->vec2<f32>{
 let q=rounded(x.x/y.x);
 let r=sub_pair(x,mul_pair(y,vec2<f32>(q,0.0)));
 let z=two_sum(q,rounded(r.x/y.x));
 let residual=sub_pair(x,mul_pair(y,z));
 return add_pair(z,vec2<f32>(rounded(residual.x/y.x),0.0));
}
fn sqrt_pair(x:vec2<f32>)->vec2<f32>{
 if(x.x==0.0 && x.y==0.0){return vec2<f32>(0.0);}
 let s=sqrt(x.x);
 let residual=sub_pair(x,mul_pair(vec2<f32>(s,0.0),vec2<f32>(s,0.0)));
 return add_pair(vec2<f32>(s,0.0),div_pair(residual,vec2<f32>(2.0*s,0.0)));
}
fn exp_pair(x:vec2<f32>)->vec2<f32>{
 let power=round((x.x+x.y)*1.4426950408889634);
 if(power < -126.0){return vec2<f32>(0.0);}
 let r=sub_pair(x,mul_pair(vec2<f32>(power,0.0),${pair(Math.LN2)}));
 var p=${pair(coefficients[12])};
 ${polynomial}
 return mul_pair(p,vec2<f32>(bitcast<f32>(u32(i32(power)+127)<<23u),0.0));
}
fn log_pair(x:vec2<f32>)->vec2<f32>{
 let seed=vec2<f32>(log(x.x),0.0);let e=exp_pair(seed);
 let correction=div_pair(sub_pair(x,e),e);
 return add_pair(seed,sub_pair(correction,mul_pair(vec2<f32>(0.5,0.0),mul_pair(correction,correction))));
}
fn at(v:vec4<u32>,r:u32,col:u32)->u32{return v.z+select(r,0u,v.x==1u)*v.w+select(col,0u,v.y==1u);}
fn store_value(r:u32,col:u32,value:vec2<f32>){
 if(!(abs(value.x)<=3.402823466e38 && abs(value.y)<=3.402823466e38)){atomicStore(&fault,1u);}
 out[d.dest.z+r*d.dest.w+col]=value;
}
@compute @workgroup_size(8,8) fn matmul(@builtin(global_invocation_id) id:vec3<u32>){
 let row=id.y;let col=id.x;if(row>=d.m||col>=d.n){return;}
 var sum=vec2<f32>(0.0);
 for(var q=0u;q<d.k;q++){sum=add_pair(sum,mul_pair(a[at(d.a,row,q)],b[at(d.b,q,col)]));}
 store_value(row,col,sum);
}
@compute @workgroup_size(64) fn point(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=d.m*d.n){return;}let row=i/d.n;let col=i%d.n;
 // Reductions and transposes use their own indices: avoid speculative OOB loads.
 var value=vec2<f32>(0.0);let op=d.flags.x;
 if(op==6u){value=a[at(d.a,col,row)];}
 else if(op==7u || op==8u){
  for(var r=0u;r<d.a.x;r++){for(var j=0u;j<d.a.y;j++){
   let x=a[at(d.a,r,j)];value=add_pair(value,select(x,mul_pair(x,x),op==8u));
  }}
 }else if(op==9u){for(var r=0u;r<d.a.x;r++){value=add_pair(value,a[at(d.a,r,col)]);}}
 else if(op==14u || op==15u){
  let first=select(0u,row,op==15u);let last=select(d.a.x,row+1u,op==15u);
  for(var r=first;r<last;r++){
   if(targets[r]==d.flags.z){continue;}
   var highest=a[at(d.a,r,0u)];
   for(var j=1u;j<d.a.y;j++){let x=a[at(d.a,r,j)];if(x.x>highest.x || (x.x==highest.x && x.y>highest.y)){highest=x;}}
   var total=vec2<f32>(0.0);var mean=vec2<f32>(0.0);
   for(var j=0u;j<d.a.y;j++){
    let x=sub_pair(a[at(d.a,r,j)],highest);total=add_pair(total,exp_pair(x));mean=add_pair(mean,x);
   }
   if(op==14u){
    let logtotal=log_pair(total);
    let targetLoss=sub_pair(logtotal,sub_pair(a[at(d.a,r,targets[r])],highest));
    let smoothLoss=sub_pair(logtotal,div_pair(mean,vec2<f32>(f32(d.a.y),0.0)));
    value=add_pair(value,add_pair(mul_pair(sub_pair(vec2<f32>(1.0,0.0),d.p0),targetLoss),mul_pair(d.p0,smoothLoss)));
   }else{
    value=sub_pair(div_pair(exp_pair(sub_pair(a[at(d.a,r,col)],highest)),total),div_pair(d.p0,vec2<f32>(f32(d.a.y),0.0)));
    if(col==targets[r]){value=sub_pair(value,sub_pair(vec2<f32>(1.0,0.0),d.p0));}
    value=mul_pair(value,b[at(d.b,0u,0u)]);
   }
  }
  value=div_pair(value,vec2<f32>(f32(d.flags.y),0.0));
 }else{
  let x=a[at(d.a,row,col)];
  if(op==0u){value=x;}
  else if(op==1u){value=add_pair(x,b[at(d.b,row,col)]);}
  else if(op==2u){value=mul_pair(x,b[at(d.b,row,col)]);}
  else if(op==3u){value=mul_pair(x,d.p0);}
  else if(op==4u){if(x.x>0.0 || (x.x==0.0 && x.y>0.0)){value=x;}}
  else if(op==5u){if(x.x>0.0 || (x.x==0.0 && x.y>0.0)){value=b[at(d.b,row,col)];}}
  else if(op==10u){value=div_pair(vec2<f32>(1.0,0.0),add_pair(sqrt_pair(x),d.p3));}
  else if(op==11u){value=add_pair(mul_pair(d.p1,x),mul_pair(sub_pair(vec2<f32>(1.0,0.0),d.p1),b[at(d.b,row,col)]));}
  else if(op==12u){let y=b[at(d.b,row,col)];value=add_pair(mul_pair(d.p2,x),mul_pair(sub_pair(vec2<f32>(1.0,0.0),d.p2),mul_pair(y,y)));}
  else if(op==13u){
   let mean=div_pair(b[at(d.b,row,col)],d.p4);
   let variance=div_pair(c[at(d.c,row,col)],d.p5);
   value=sub_pair(x,div_pair(mul_pair(d.p0,mean),add_pair(sqrt_pair(variance),d.p3)));
  }else if(op==16u){
   let norm=sqrt_pair(b[at(d.b,0u,0u)]);var factor=vec2<f32>(1.0,0.0);
   if(norm.x>d.p0.x || (norm.x==d.p0.x && norm.y>d.p0.y)){factor=div_pair(d.p0,norm);}
   value=mul_pair(x,factor);
  }else if(op==17u){value=sub_pair(x,mul_pair(d.p0,b[at(d.b,row,col)]));}
 }
 store_value(row,col,value);
}`;

// Reuse the established FP32 convolution equations. Inputs round to FP32 at
// exactly the same boundary as convolutionGPU; only storage stays resident.
export const RESIDENT_CONV_WGSL = CONV_WGSL
 .replaceAll('array<f32>','array<vec2<f32>>')
 .replace(/\b(a|b|bias)\[([^\]]+)\]/g,'$1[$2].x')
 .replace('out[i]=sum;',`out[i]=vec2<f32>(sum,0.0);
 if(!(abs(sum)<=3.402823466e38)){atomicStore(&fault,1u);}`)
 + '\n@group(0) @binding(5) var<storage,read_write> fault:atomic<u32>;\n';
