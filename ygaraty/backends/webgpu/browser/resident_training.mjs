// Generated output: resident_training.mjs. Edit this TypeScript source.
// A transaction owns GPU state for H local Tape steps. Only a successful final
// readback can be published by DiLoCoTrainer; device loss/cancellation rolls back.
import { tensor } from './sdk_tensor.mjs';
import { BufferScope, checkedGPU, compileProgram } from './webgpu_kernels.mjs';
import { CONV_OPS, convolutionShapeContract } from './convolution.mjs';
import { RESIDENT_OPS, RESIDENT_TRAINING_WGSL, RESIDENT_CONV_WGSL } from './resident_training_kernels.mjs';
import { supportsResidentTraining, stateReferences, regionShape, residentOp, sameShape } from './resident_training_plan.mjs';
/** Free scratch can be reused after its last encoded reader. Queue ordering
 * protects reuse across submissions; uploads are encoded copies, never writes
 * that could overtake commands still being built on the host. */
class Arena {
    device;
    slots = [];
    free = new Map();
    bytes = 0;
    constructor(device) {
        this.device = device;
    }
    take(shape) {
        const bytes = shape[0] * shape[1] * 8;
        let slot = this.free.get(bytes)?.pop();
        if (!slot) {
            if (bytes > Math.min(this.device.limits.maxBufferSize, this.device.limits.maxStorageBufferBindingSize) || this.bytes + bytes > 256 * 1024 * 1024)
                throw Error('Resident training buffer budget exceeded');
            slot = { buffer: this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }), refs: 0 };
            this.slots.push(slot);
            this.bytes += bytes;
        }
        slot.refs = 1;
        return { slot, shape: [...shape], offset: 0, stride: shape[1] };
    }
    retain(v) { v.slot.refs++; return v; }
    release(v) {
        if (--v.slot.refs < 0)
            throw Error('Resident buffer lifetime error');
        if (!v.slot.refs) {
            const size = v.slot.buffer.size;
            let bucket = this.free.get(size);
            if (!bucket) {
                bucket = [];
                this.free.set(size, bucket);
            }
            bucket.push(v.slot);
        }
    }
    destroy() { for (const s of this.slots)
        s.buffer.destroy(); }
}
const portion = (v, region) => !region ? v : { ...v, shape: regionShape(v.shape, region), offset: v.offset + region[0] * v.stride + region[2] };
const descriptor = (v) => [v.shape[0], v.shape[1], v.offset, v.stride];
const stateViews = (m) => [...Object.values(m.m), ...Object.values(m.v)];
export async function residentTrainingGPU(device, request, arithmetic) {
    if (arithmetic !== 'compensated' || !supportsResidentTraining(request))
        return null;
    request.signal?.throwIfAborted();
    return checkedGPU(device, async () => {
        const arena = new Arena(device), scope = new BufferScope(device);
        const stats = { cpu_tasks: 0, gpu_tasks: 0, matmul_tasks: 0, resident_rounds: 1, gpu_submissions: 0, readbacks: 0, uploaded_bytes: 0, downloaded_bytes: 0 };
        try {
            const [generic, conv] = await Promise.all([
                compileProgram(device, RESIDENT_TRAINING_WGSL, ['point', 'matmul'], ['read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform', 'storage', 'read-only-storage']),
                compileProgram(device, RESIDENT_CONV_WGSL, ['main'], ['read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'uniform', 'storage']),
            ]);
            const zero = arena.take([1, 1]);
            const fault = scope.empty(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const lossBuffer = scope.empty(request.steps.length * 8, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
            let weights = {}, moments = { m: {}, v: {} };
            for (let step = 0; step < request.steps.length; step++) {
                request.signal?.throwIfAborted();
                const { plan, constants } = request.steps[step], refs = stateReferences(plan);
                const encoder = device.createCommandEncoder();
                const upload = (value) => {
                    const v = arena.take(value.shape), packed = new Float32Array(value.data.length * 2);
                    value.data.forEach((x, i) => { packed[2 * i] = x; packed[2 * i + 1] = x - packed[2 * i]; });
                    const staging = scope.upload(packed, GPUBufferUsage.COPY_SRC);
                    encoder.copyBufferToBuffer(staging, 0, v.slot.buffer, 0, packed.byteLength);
                    stats.uploaded_bytes += packed.byteLength;
                    return v;
                };
                if (step === 0) {
                    weights = Object.fromEntries(Object.entries(request.weights).map(([n, v]) => [n, upload(v)]));
                    for (const kind of ['m', 'v'])
                        moments[kind] = Object.fromEntries(Object.entries(request.moments[kind]).map(([n, v]) => [n, upload(v)]));
                }
                const values = new Map();
                for (const [id, value] of Object.entries(constants))
                    values.set(id, upload(value));
                for (const [name, id] of Object.entries(plan.parameters))
                    values.set(id, arena.retain(weights[name]));
                for (const kind of ['m', 'v'])
                    for (const [name, id] of Object.entries(refs[kind]))
                        values.set(id, arena.retain(moments[kind][name]));
                const keep = new Set([plan.loss, ...Object.values(plan.updates)]);
                const remaining = new Map();
                for (const t of plan.tasks)
                    for (const id of new Set(t.inputs.map(r => r.value)))
                        remaining.set(id, (remaining.get(id) || 0) + 1);
                const get = (id) => { const v = values.get(id); if (!v)
                    throw Error(`Missing resident value: ${id}`); return v; };
                const count = plan.tasks.reduce((n, t) => n + 5 + t.inputs.length, 0);
                const uniforms = device.createBuffer({ size: count * 256, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
                scope.buffers.push(uniforms);
                const uniformBytes = new Uint8Array(uniforms.getMappedRange());
                let cursor = 0;
                const uniform = (data) => {
                    if (cursor + 256 > uniformBytes.length)
                        throw Error('Resident uniform budget exceeded');
                    uniformBytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), cursor);
                    const binding = { buffer: uniforms, offset: cursor, size: data.byteLength };
                    cursor += 256;
                    return binding;
                };
                const pass = encoder.beginComputePass();
                const dispatch = (op, args, output, attrs = {}, labels = zero.slot.buffer) => {
                    const buffer = new ArrayBuffer(144), u = new Uint32Array(buffer), f = new Float32Array(buffer);
                    const [a = zero, b = zero, c = zero] = args;
                    u.set([output.shape[0], output.shape[1], a.shape[1], 0]);
                    u.set(descriptor(a), 4);
                    u.set(descriptor(b), 8);
                    u.set(descriptor(c), 12);
                    const parameter = (index, x) => { f[index] = x; f[index + 1] = x - f[index]; };
                    parameter(16, Number(op === 'scale' ? attrs.factor : op === 'clip_gradient' ? attrs.clip_norm : op === 'cross_entropy' || op === 'ce_backward' ? attrs.label_smoothing ?? 0 : attrs.learning_rate ?? 0));
                    parameter(18, Number(attrs.beta1 ?? 0));
                    parameter(20, Number(attrs.beta2 ?? 0));
                    parameter(22, Number(attrs.epsilon ?? 0));
                    parameter(24, 1 - Number(attrs.beta1 ?? 0) ** Number(attrs.step ?? 1));
                    parameter(26, 1 - Number(attrs.beta2 ?? 0) ** Number(attrs.step ?? 1));
                    const targets = attrs.targets;
                    u.set([RESIDENT_OPS.indexOf(op), Number(attrs.count ?? targets?.filter(x => x !== attrs.ignore_index).length ?? 0), Number(attrs.ignore_index ?? 0xffffffff), 0], 28);
                    u.set(descriptor(output), 32);
                    const pipeline = generic[op === 'matmul' ? 1 : 0];
                    const resources = [a.slot.buffer, b.slot.buffer, c.slot.buffer, output.slot.buffer].map(buffer => ({ buffer }));
                    resources.push(uniform(new Uint8Array(buffer)), { buffer: fault }, { buffer: labels });
                    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) });
                    pass.setPipeline(pipeline);
                    pass.setBindGroup(0, bind);
                    if (op === 'matmul') {
                        pass.dispatchWorkgroups(Math.ceil(output.shape[1] / 8), Math.ceil(output.shape[0] / 8));
                        stats.matmul_tasks++;
                    }
                    else
                        pass.dispatchWorkgroups(Math.ceil(output.shape[0] * output.shape[1] / 64));
                    stats.gpu_tasks++;
                };
                const compact = (v) => {
                    if (v.offset === 0 && v.stride === v.shape[1])
                        return arena.retain(v);
                    const result = arena.take(v.shape);
                    dispatch('copy', [v], result);
                    return result;
                };
                for (const task of plan.tasks) {
                    const args = task.inputs.map(r => portion(get(r.value), r.region));
                    let output;
                    if ((task.op === 'assemble' || task.op === 'add_many') && args.length === 1 && sameShape(args[0].shape, task.shape))
                        output = arena.retain(args[0]);
                    else if (task.op === 'assemble') {
                        output = arena.take(task.shape);
                        const blocks = task.attrs.blocks;
                        args.forEach((v, i) => dispatch('copy', [v], portion(output, blocks[i])));
                    }
                    else if (task.op === 'add_many') {
                        output = arena.retain(args[0]);
                        for (const arg of args.slice(1)) {
                            const next = arena.take(task.shape);
                            dispatch('add', [output, arg], next);
                            arena.release(output);
                            output = next;
                        }
                    }
                    else if (CONV_OPS.includes(task.op)) {
                        output = arena.take(task.shape);
                        const packed = args.map(compact), dims = convolutionShapeContract(task.op, task.attrs, args.map(v => v.shape), task.shape);
                        const resources = [0, 1, 2].map(i => ({ buffer: (packed[i] ?? zero).slot.buffer }));
                        resources.push({ buffer: output.slot.buffer }, uniform(new Uint32Array(dims)), { buffer: fault });
                        const pipeline = conv[0], bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: resources.map((resource, binding) => ({ binding, resource })) });
                        pass.setPipeline(pipeline);
                        pass.setBindGroup(0, bind);
                        pass.dispatchWorkgroups(Math.ceil(dims[7] / 64));
                        stats.gpu_tasks++;
                        packed.forEach(v => arena.release(v));
                    }
                    else {
                        output = arena.take(task.shape);
                        const targets = task.attrs.targets;
                        const labels = targets ? scope.upload(new Uint32Array(targets), GPUBufferUsage.STORAGE) : zero.slot.buffer;
                        if (targets)
                            stats.uploaded_bytes += targets.length * 4;
                        dispatch(residentOp(task.op), args, output, task.attrs, labels);
                    }
                    values.set(task.id, output);
                    for (const id of new Set(task.inputs.map(r => r.value))) {
                        remaining.set(id, remaining.get(id) - 1);
                        if (!remaining.get(id) && !keep.has(id)) {
                            arena.release(get(id));
                            values.delete(id);
                        }
                    }
                    if (!remaining.get(task.id) && !keep.has(task.id)) {
                        arena.release(output);
                        values.delete(task.id);
                    }
                }
                pass.end();
                uniforms.unmap();
                const loss = get(plan.loss);
                encoder.copyBufferToBuffer(loss.slot.buffer, loss.offset * 8, lossBuffer, step * 8, 8);
                const nextWeights = Object.fromEntries(Object.entries(plan.parameters).map(([name, id]) => [name, arena.retain(get(plan.updates[id]))]));
                const nextMoments = { m: {}, v: {} };
                for (const kind of ['m', 'v'])
                    nextMoments[kind] = Object.fromEntries(Object.entries(refs[kind]).map(([name, id]) => [name, arena.retain(get(plan.updates[id]))]));
                for (const v of [...Object.values(weights), ...stateViews(moments)])
                    arena.release(v);
                for (const v of values.values())
                    arena.release(v);
                weights = nextWeights;
                moments = nextMoments;
                device.queue.submit([encoder.finish()]);
                stats.gpu_submissions++;
                // Yield to heartbeat/cancellation handlers, without a device synchronization.
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            request.signal?.throwIfAborted();
            const outputs = [...Object.values(weights), ...stateViews(moments)];
            const bytes = outputs.reduce((n, v) => n + v.shape[0] * v.shape[1] * 8, 0) + request.steps.length * 8 + 4;
            const read = scope.empty(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ), encoder = device.createCommandEncoder();
            let offset = 0;
            for (const v of outputs) {
                for (let row = 0; row < v.shape[0]; row++)
                    encoder.copyBufferToBuffer(v.slot.buffer, (v.offset + row * v.stride) * 8, read, offset + row * v.shape[1] * 8, v.shape[1] * 8);
                offset += v.shape[0] * v.shape[1] * 8;
            }
            encoder.copyBufferToBuffer(lossBuffer, 0, read, offset, request.steps.length * 8);
            encoder.copyBufferToBuffer(fault, 0, read, bytes - 4, 4);
            device.queue.submit([encoder.finish()]);
            stats.gpu_submissions++;
            stats.readbacks++;
            stats.downloaded_bytes = bytes;
            await read.mapAsync(GPUMapMode.READ);
            try {
                request.signal?.throwIfAborted();
                const memory = read.getMappedRange();
                if (new DataView(memory).getUint32(bytes - 4, true))
                    throw Error('Nonfinite resident training intermediate');
                const raw = new Float32Array(memory, 0, (bytes - 4) / 4);
                let index = 0;
                const unpack = (shape) => {
                    const data = new Float64Array(shape[0] * shape[1]);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = raw[index] + raw[index + 1];
                        index += 2;
                    }
                    if (!data.every(Number.isFinite))
                        throw Error('Nonfinite resident training output');
                    return tensor(shape, data);
                };
                const finalWeights = Object.fromEntries(Object.entries(weights).map(([name, v]) => [name, unpack(v.shape)]));
                const finalMoments = { m: {}, v: {} };
                for (const kind of ['m', 'v'])
                    finalMoments[kind] = Object.fromEntries(Object.entries(moments[kind]).map(([name, v]) => [name, unpack(v.shape)]));
                const losses = Array.from(unpack([1, request.steps.length]).data);
                stats.gpu_buffer_bytes = arena.bytes + scope.buffers.reduce((n, b) => n + b.size, 0);
                return { weights: finalWeights, moments: finalMoments, losses, stats };
            }
            finally {
                read.unmap();
            }
        }
        finally {
            await device.queue.onSubmittedWorkDone().catch(() => { });
            arena.destroy();
            scope.destroy();
        }
    });
}
