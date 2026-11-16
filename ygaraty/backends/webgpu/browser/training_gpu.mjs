// Generated output: training_gpu.mjs.
import { tensor } from './sdk_tensor.mjs';
import { BufferScope, compileProgram, runReadback, checkedGPU } from './webgpu_kernels.mjs';
export const COMPENSATED_MATMUL_WGSL = `
          struct Dims {m:u32,n:u32,k:u32,pad:u32};
          @group(0) @binding(0) var<storage,read> a:array<vec2<f32>>;
          @group(0) @binding(1) var<storage,read> b:array<vec2<f32>>;
          @group(0) @binding(2) var<storage,read_write> c:array<vec2<f32>>;
          @group(0) @binding(3) var<uniform> d:Dims;
          // Unevaluated hi+lo pairs preserve the residual discarded by f32.
          // pad is a runtime zero. The integer round-trip prevents GPU fast-
          // math from algebraically deleting the rounding residuals. It does
          // not change a bit; keep the fourth uniform word zero.
          fn rounded(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^d.pad);}
          fn two_sum(a:f32,b:f32)->vec2<f32>{
            let s=rounded(a+b); let v=rounded(s-a);
            let left=rounded(a-rounded(s-v));let right=rounded(b-v);
            return vec2<f32>(s,rounded(left+right));
          }
          fn add_pair(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
            let s=two_sum(a.x,b.x);
            let low=rounded(rounded(s.y+a.y)+b.y);
            return two_sum(s.x,low);
          }
          fn mul_pair(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
            let p=rounded(a.x*b.x);
            let residual=rounded(fma(a.x,b.x,-p));
            let cross=rounded(rounded(a.x*b.y)+rounded(a.y*b.x));
            let e=rounded(rounded(residual+cross)+rounded(a.y*b.y));
            return two_sum(p,e);
          }
          @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
            let row=id.y;let col=id.x;if(row>=d.m||col>=d.n){return;}
            var sum=vec2<f32>(0.0,0.0);
            for(var q:u32=0u;q<d.k;q=q+1u){sum=add_pair(sum,mul_pair(a[row*d.k+q],b[q*d.n+col]));}
            c[row*d.n+col]=sum;
          }`;
export const DENSE_MATMUL_WGSL = `
          struct Dims {m:u32,n:u32,k:u32,pad:u32};
          @group(0) @binding(0) var<storage,read> a:array<f32>;
          @group(0) @binding(1) var<storage,read> b:array<f32>;
          @group(0) @binding(2) var<storage,read_write> c:array<f32>;
          @group(0) @binding(3) var<uniform> d:Dims;
          @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
            let row=id.y;let col=id.x;if(row>=d.m||col>=d.n){return;}
            var sum:f32=0.0;for(var q:u32=0u;q<d.k;q=q+1u){sum=sum+a[row*d.k+q]*b[q*d.n+col];}
            c[row*d.n+col]=sum;
          }`;
async function denseMatmulGPUUnchecked(device, a, b, shape, arithmetic = 'float32') {
    if (arithmetic !== 'float32' && arithmetic !== 'compensated')
        throw Error('Unsupported dense arithmetic');
    if (a.shape.length !== 2 || b.shape.length !== 2 || shape.length !== 2 || a.shape[1] !== b.shape[0] || shape[0] !== a.shape[0] || shape[1] !== b.shape[1])
        throw Error('Dense matmul shape mismatch');
    const scope = new BufferScope(device), compensated = arithmetic === 'compensated';
    try {
        const steps = await compileProgram(device, compensated ? COMPENSATED_MATMUL_WGSL : DENSE_MATMUL_WGSL, ['main']);
        const pack = (values) => {
            if (!compensated)
                return Float32Array.from(values);
            const pairs = new Float32Array(values.length * 2);
            for (let i = 0; i < values.length; i++) {
                pairs[2 * i] = values[i];
                pairs[2 * i + 1] = values[i] - pairs[2 * i];
            }
            return pairs;
        };
        const A = scope.upload(pack(a.data), GPUBufferUsage.STORAGE), B = scope.upload(pack(b.data), GPUBufferUsage.STORAGE);
        const C = scope.upload(new Float32Array(shape[0] * shape[1] * (compensated ? 2 : 1)), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const D = scope.upload(new Uint32Array([shape[0], shape[1], a.shape[1], 0]), GPUBufferUsage.UNIFORM);
        const raw = await runReadback(scope, steps, [A, B, C, D], [[Math.ceil(shape[1] / 8), Math.ceil(shape[0] / 8)]], C, C.size);
        const values = new Float64Array(shape[0] * shape[1]);
        for (let i = 0; i < values.length; i++)
            values[i] = compensated ? raw[2 * i] + raw[2 * i + 1] : raw[i];
        if (!values.every(Number.isFinite))
            throw Error('Nonfinite GPU output');
        return tensor(shape, values);
    }
    finally {
        scope.destroy();
    }
}
export function denseMatmulGPU(device, a, b, shape, arithmetic = 'float32') {
    return checkedGPU(device, () => denseMatmulGPUUnchecked(device, a, b, shape, arithmetic));
}
/** Typed host-tensor entry point. No implicit CPU fallback or precision change. */
export async function executeKernel(device, request) {
    if (request.op === 'matmul')
        return { value: await denseMatmulGPU(device, request.inputs[0], request.inputs[1], request.shape, request.arithmetic) };
    if (request.op === 'factorization_tile' || request.op === 'factorization_tiles') {
        if (request.arithmetic !== 'kahan-f32')
            throw Error('Factorization requires kahan-f32');
        const kernels = await import('./factorization_gpu.mjs');
        return request.op === 'factorization_tile'
            ? kernels.factorizationTileGPU(device, request.inputs, request.attrs, request.shape)
            : kernels.factorizationTilesGPU(device, request.inputs, request.attrs, request.shape);
    }
    if (request.arithmetic !== 'float32')
        throw Error('Convolution requires float32');
    const { convolutionGPU } = await import('./convolution.mjs');
    return { value: await convolutionGPU(device, request.op, request.inputs, request.attrs, request.shape) };
}
