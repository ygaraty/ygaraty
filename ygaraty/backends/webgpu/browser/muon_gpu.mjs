import { tensor } from './sdk_tensor.mjs';
import { BufferScope, checkedGPU, compileProgram } from './webgpu_kernels.mjs';
import { COMPENSATED_MATMUL_WGSL, COMPENSATED_ARITHMETIC_WGSL } from './training_gpu.mjs';
const ELEMENTS = `
struct Dims {m:u32,n:u32,k:u32,pad:u32,factor:vec2<f32>,reserved:vec2<u32>};
@group(0) @binding(0) var<storage,read> a:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read> b:array<vec2<f32>>;
@group(0) @binding(2) var<storage,read_write> c:array<vec2<f32>>;
@group(0) @binding(3) var<uniform> d:Dims;
${COMPENSATED_ARITHMETIC_WGSL}
@compute @workgroup_size(64) fn add(@builtin(global_invocation_id) id:vec3<u32>) {
  let i=id.x;if(i<d.m*d.n){c[i]=add_pair(a[i],b[i]);}
}
@compute @workgroup_size(64) fn scale(@builtin(global_invocation_id) id:vec3<u32>) {
  let i=id.x;if(i<d.m*d.n){c[i]=mul_pair(a[i],d.factor);}
}
@compute @workgroup_size(64) fn transpose(@builtin(global_invocation_id) id:vec3<u32>) {
  let i=id.x;if(i<d.m*d.n){c[i]=a[(i%d.n)*d.m+i/d.n];}
}`;
const same = (a, b) => a.length === b.length && a.every((n, i) => n === b[i]);
/** Return null before allocating if the graph requires the portable fallback. */
export function supportsMuonBlock(tasks, inputs, outputs) {
    if (!tasks.length || tasks.length > 512 || !outputs.length)
        return false;
    const shapes = new Map([...inputs].map(([id, t]) => [id, t.shape]));
    const ids = new Set();
    let bytes = 0;
    for (const t of tasks) {
        if (shapes.has(t.id) || t.shape.length !== 2 || !t.shape.every(n => Number.isSafeInteger(n) && n > 0))
            return false;
        // WebGPU guarantees at least 65535 workgroups per dispatch dimension.
        if (Math.ceil(t.shape[0] * t.shape[1] / 64) > 65535 ||
            Math.ceil(Math.max(...t.shape) / 8) > 65535)
            return false;
        const args = [];
        for (const ref of t.inputs) {
            const shape = shapes.get(ref.value);
            if (!shape || ref.region && !same(ref.region, [0, shape[0], 0, shape[1]]))
                return false;
            args.push(shape);
        }
        const equal = args.every(s => same(s, t.shape));
        switch (t.op) {
            case 'matmul':
                if (args.length !== 2 || args[0][1] !== args[1][0] || !same(t.shape, [args[0][0], args[1][1]]))
                    return false;
                break;
            case 'add':
                if (args.length !== 2 || !equal)
                    return false;
                break;
            case 'scale':
                if (args.length !== 1 || !equal || typeof t.attrs.factor !== 'number' || !Number.isFinite(t.attrs.factor))
                    return false;
                break;
            case 'transpose':
                if (args.length !== 1 || !same(t.shape, [args[0][1], args[0][0]]))
                    return false;
                break;
            case 'assemble': {
                const blocks = t.attrs.blocks;
                if (args.length !== 1 || !equal || !Array.isArray(blocks) || blocks.length !== 1 || !Array.isArray(blocks[0]) || !same(blocks[0], [0, t.shape[0], 0, t.shape[1]]))
                    return false;
                break;
            }
            default: return false;
        }
        bytes += 8 * t.shape[0] * t.shape[1];
        if (bytes > 64 * 1024 * 1024)
            return false;
        shapes.set(t.id, t.shape);
        ids.add(t.id);
    }
    return outputs.every(id => ids.has(id));
}
export async function muonBlockGPU(device, tasks, inputs, outputs, arithmetic) {
    // Preserve the requested precision. The float32 path remains unchanged.
    if (arithmetic !== 'compensated' || !supportsMuonBlock(tasks, inputs, outputs))
        return null;
    return checkedGPU(device, async () => {
        const scope = new BufferScope(device);
        try {
            const [matrix, elements] = await Promise.all([
                compileProgram(device, COMPENSATED_MATMUL_WGSL, ['main']),
                compileProgram(device, ELEMENTS, ['add', 'scale', 'transpose'], ['read-only-storage', 'read-only-storage', 'storage', 'uniform']),
            ]);
            const buffers = new Map(), shapes = new Map([...inputs].map(([id, t]) => [id, t.shape]));
            const get = (id) => {
                const old = buffers.get(id);
                if (old)
                    return old;
                const value = inputs.get(id);
                if (!value)
                    throw Error(`Missing resident input: ${id}`);
                const pairs = new Float32Array(2 * value.data.length);
                for (let i = 0; i < value.data.length; i++) {
                    pairs[2 * i] = value.data[i];
                    pairs[2 * i + 1] = value.data[i] - pairs[2 * i];
                }
                if (!pairs.every(Number.isFinite))
                    throw Error('Nonfinite resident input');
                const buffer = scope.upload(pairs, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
                buffers.set(id, buffer);
                return buffer;
            };
            const encoder = device.createCommandEncoder();
            for (const task of tasks) {
                shapes.set(task.id, task.shape);
                if (task.op === 'assemble') {
                    buffers.set(task.id, get(task.inputs[0].value));
                    continue;
                }
                const a = get(task.inputs[0].value), b = task.inputs.length === 2 ? get(task.inputs[1].value) : a;
                const c = scope.empty(8 * task.shape[0] * task.shape[1], GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
                const uniform = new ArrayBuffer(32), ints = new Uint32Array(uniform), floats = new Float32Array(uniform);
                ints.set([task.shape[0], task.shape[1], shapes.get(task.inputs[0].value)[1], 0]);
                const factor = Number(task.attrs.factor ?? 0);
                floats[4] = factor;
                floats[5] = factor - floats[4];
                const d = scope.upload(new Uint8Array(uniform), GPUBufferUsage.UNIFORM);
                const pipeline = task.op === 'matmul' ? matrix[0] : elements[['add', 'scale', 'transpose'].indexOf(task.op)];
                const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [a, b, c, d].map((buffer, binding) => ({ binding, resource: { buffer } })) });
                const pass = encoder.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bind);
                if (task.op === 'matmul')
                    pass.dispatchWorkgroups(Math.ceil(task.shape[1] / 8), Math.ceil(task.shape[0] / 8));
                else
                    pass.dispatchWorkgroups(Math.ceil(task.shape[0] * task.shape[1] / 64));
                pass.end();
                buffers.set(task.id, c);
            }
            const sizes = outputs.map(id => 8 * shapes.get(id)[0] * shapes.get(id)[1]);
            const read = scope.empty(sizes.reduce((a, b) => a + b, 0), GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
            let offset = 0;
            outputs.forEach((id, i) => { encoder.copyBufferToBuffer(get(id), 0, read, offset, sizes[i]); offset += sizes[i]; });
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const result = new Map();
            try {
                const raw = new Float32Array(read.getMappedRange());
                offset = 0;
                outputs.forEach((id, i) => {
                    const data = new Float64Array(sizes[i] / 8);
                    for (let j = 0; j < data.length; j++)
                        data[j] = raw[offset + 2 * j] + raw[offset + 2 * j + 1];
                    if (!data.every(Number.isFinite))
                        throw Error('Nonfinite resident output');
                    result.set(id, tensor(shapes.get(id), data));
                    offset += sizes[i] / 4;
                });
            }
            finally {
                read.unmap();
            }
            return result;
        }
        finally {
            scope.destroy();
        }
    });
}
