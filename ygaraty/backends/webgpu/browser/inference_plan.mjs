import { validateBufferLayout } from './inference_layout.mjs';
import { operationContract, checkContract, validateResolved, stateContract, loopOperations as loops } from './inference_contracts.mjs';
const admitted = new WeakSet();
function freeze(value) {
    if (value && typeof value === 'object') {
        for (const child of Object.values(value))
            freeze(child);
        Object.freeze(value);
    }
    return value;
}
const lifecycles = new WeakMap();
function lifecycle(owner) {
    let state = lifecycles.get(owner);
    if (!state) {
        state = { dirty: new Set(), busy: false };
        lifecycles.set(owner, state);
    }
    return state;
}
export function assertStateReady(owner, slots) {
    const state = lifecycle(owner);
    if (state.busy)
        throw Error('Inference stage is busy');
    for (const slot of slots)
        if (state.dirty.has(slot))
            throw Error(`Inference slot ${slot} requires reset after failure`);
}
/** Failed resets retain dirty state; reset and inference share the stage lock. */
export async function resetPlanState(owner, slots, reset) {
    const state = lifecycle(owner);
    assertStateIdle(owner);
    state.busy = true;
    for (const slot of slots)
        state.dirty.add(slot);
    try {
        await reset();
        for (const slot of slots)
            state.dirty.delete(slot);
    }
    finally {
        state.busy = false;
    }
}
export function assertStateIdle(owner) {
    if (lifecycle(owner).busy)
        throw Error('Inference stage is busy');
}
function record(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        throw Error('Invalid plan object');
    return value;
}
export function validatePlan(value, manifest, family, hidden) {
    const p = structuredClone(value);
    const legacy = record(value).version === 2;
    if (!p || (!legacy && (p.version !== 3 || p.contract_version !== 1)) || p.family !== family || p.hidden !== hidden || p.rows !== 'tokens' ||
        !Number.isSafeInteger(p.owner) || p.owner < 0 || p.submit_token_layers !== 64 || !Array.isArray(p.tasks) ||
        !Array.isArray(p.required_ops) || !manifest.layers.length)
        throw Error('Missing or unsupported compiled inference plan');
    if (legacy) {
        if (p.contract_version !== undefined || p.state_contract !== undefined)
            throw Error('Legacy plan has unexpected contracts');
        p.state_contract = structuredClone(stateContract);
    }
    else
        checkContract(p.state_contract, stateContract);
    const incoming = manifest.embed ? 'tokens' : `hidden_${manifest.layers[0].index - 1}`;
    if (p.input !== incoming)
        throw Error('Plan input differs from stage');
    const operations = new Set(), ends = [];
    let budget = 10000;
    function validate(tasks, input, depth = 0, sequence = false) {
        if (depth > 4 || !tasks.length)
            throw Error('Invalid plan composition');
        let previous = input;
        const ids = new Set([input]);
        for (const t of tasks) {
            if (--budget < 0 || !t || typeof t.id !== 'string' || ids.has(t.id) || t.attrs?.owner !== p.owner ||
                !Array.isArray(t.inputs) || t.inputs.length !== 1 || t.inputs[0].value !== previous ||
                (t.inputs[0].region !== null && t.inputs[0].region !== undefined) || !Array.isArray(t.attrs.call) ||
                !Array.isArray(t.shape) || t.shape.length !== 2 || t.shape[0] !== 1 || !Number.isSafeInteger(t.shape[1]) || t.shape[1] < 1)
                throw Error('Invalid task dependency/shape/ownership');
            if (t.attrs.when !== undefined && (t.attrs.when !== "head_enabled" || depth || t.attrs.layer !== null))
                throw Error("Invalid execution guard");
            if (t.attrs.layer !== null && !manifest.layers.some(l => l.index === t.attrs.layer))
                throw Error('Unknown layer resource');
            const contract = operationContract(t.op, t.attrs.call, t.attrs.layer);
            if (legacy) {
                if (t.attrs.contract !== undefined)
                    throw Error('Legacy plan has unexpected contracts');
                t.attrs.contract = contract;
            }
            else
                checkContract(t.attrs.contract, contract);
            if (contract.effects.some(e => e.scope === 'layer_sequence') && !sequence)
                throw Error('State access requires a sequence loop');
            if (loops.has(t.op)) {
                if (t.attrs.call.length || !Array.isArray(t.attrs.body))
                    throw Error('Invalid loop body');
                validate(t.attrs.body, previous, depth + 1, sequence || t.op === 'for_each_sequence');
            }
            else {
                if (t.attrs.body !== undefined)
                    throw Error('Primitive has a loop body');
            }
            if (t.attrs.layer_end) {
                if (depth || t.attrs.layer === null)
                    throw Error('Invalid layer boundary');
                ends.push(t.attrs.layer);
            }
            operations.add(t.op);
            ids.add(t.id);
            previous = t.id;
        }
        return previous;
    }
    if (validate(p.tasks, incoming) !== p.output || p.output !== (manifest.head ? 'next_token' : `hidden_${manifest.layers.at(-1).index}`) ||
        JSON.stringify(ends) !== JSON.stringify(manifest.layers.map(l => l.index)) ||
        JSON.stringify([...operations].sort()) !== JSON.stringify(p.required_ops))
        throw Error('Invalid plan output, boundaries or kernel requirements');
    p.version = 3;
    p.contract_version = 1;
    admitted.add(p);
    return freeze(p);
}
export function bindKernel(fn) { return (...args) => fn(...args); }
function resolve(value, ctx, task, part) {
    if (Array.isArray(value))
        return value.map(v => resolve(v, ctx, task, part));
    if (value === null || typeof value !== 'object')
        return value;
    const v = record(value), name = String(v.name);
    const layer = () => { if (task.attrs.layer === null)
        throw Error('No layer resource'); return record(ctx.layer(task.attrs.layer)); };
    const required = (x) => { if (x === undefined || x === null)
        throw Error(`Missing plan binding: ${name}`); return x; };
    switch (v.$) {
        case 'buffer': {
            const b = required(ctx.buffers[name]);
            if (v.stride !== undefined && (v.view === 'always' || (v.view === 'head' ? ctx.variables.head_all : ctx.variables.batch))) {
                if (!Number.isSafeInteger(v.stride) || Number(v.stride) < 1)
                    throw Error('Invalid view stride');
                const offset = part.row * Number(v.stride) * 4;
                if (!Number.isSafeInteger(offset) || offset < 0 || (typeof b.size === 'number' && offset >= b.size))
                    throw Error('Buffer view outside allocation');
                return { buffer: b, offset };
            }
            return b;
        }
        case 'matrix': return required(record(layer().matrices)[name]);
        case 'vector': return required(record(layer().vectors)[name]);
        case 'layer_buffer': return required(layer()[name]);
        case 'state': {
            const slots = layer()[name];
            if (!Array.isArray(slots))
                throw Error('Invalid state resource');
            return required(slots[part.slot]);
        }
        case 'var': return required({ row: part.row, count: part.count, position: part.position }[name] ?? ctx.variables[name]);
        case 'resource': return required(ctx.resources[name]);
        case 'product': {
            if (!Array.isArray(v.values))
                throw Error('Invalid product expression');
            const values = v.values.map(x => resolve(x, ctx, task, part));
            if (!values.every(x => typeof x === 'number' && Number.isFinite(x)))
                throw Error('Invalid numeric binding');
            return values.reduce((a, b) => a * b, 1);
        }
        case 'choose':
            if (v.condition !== 'batch' && v.condition !== 'head_all')
                throw Error('Unknown predicate');
            return resolve(ctx.variables[String(v.condition)] ? v.yes : v.no, ctx, task, part);
        case undefined: return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, resolve(x, ctx, task, part)]));
        default: throw Error('Unknown binding');
    }
}
// Admission freezes and validates every task once. Keep hostCache=false as the
// diagnostic baseline that repeats contract checks without bypassing admission.
export async function executePlan(plan, tokens, gpu, kernels, submitMs, submitLayers, context, upload) {
    if (!plan)
        throw Error('A compiled inference plan is required');
    if (!admitted.has(plan))
        throw Error('An admitted plan from validatePlan is required');
    for (const op of plan.required_ops)
        if (!loops.has(op) && typeof kernels[op] !== 'function')
            throw Error(`Unavailable inference kernel: ${op}`);
    if (!context?.sequences?.length || !Number.isSafeInteger(tokens) || tokens < 1)
        throw Error('Invalid execution context');
    if (!context.stateOwner)
        throw Error("Missing stage state owner");
    const slots = new Set();
    for (const part of context.sequences) {
        if (![part.slot, part.row, part.count, part.position].every(n => Number.isSafeInteger(n) && n >= 0) || part.count < 1 || part.row + part.count > tokens || slots.has(part.slot))
            throw Error('Invalid sequence state scope');
        slots.add(part.slot);
        if (Number.isSafeInteger(context.variables.capacity) && part.position + part.count > Number(context.variables.capacity))
            throw Error('State position exceeds capacity');
    }
    const ordered = [...context.sequences].sort((a, b) => a.row - b.row);
    let nextRow = 0;
    for (const part of ordered) {
        if (part.row !== nextRow)
            throw Error('Sequence rows must cover tokens without overlap');
        nextRow += part.count;
    }
    if (nextRow !== tokens || context.variables.tokens !== tokens)
        throw Error('Sequence token count mismatch');
    assertStateReady(context.stateOwner, [...slots]);
    const expandStart = performance.now();
    const commands = [];
    function expand(tasks, ctx, part, inSequence = false) {
        for (const task of tasks) {
            if (gpu.hostCache === false) {
                const contract = operationContract(task.op, [...task.attrs.call], task.attrs.layer);
                checkContract(task.attrs.contract, contract);
                if (contract.effects.some(e => e.scope === 'layer_sequence') && !inSequence)
                    throw Error('State access requires a sequence loop');
            }
            if (task.attrs.when && !ctx.variables[task.attrs.when])
                continue;
            if (task.op === 'for_each_sequence')
                for (const sequence of ctx.sequences)
                    expand(task.attrs.body, ctx, sequence, true);
            else if (task.op === 'for_each_chunk') {
                const chunks = ctx.resources.output;
                if (!Array.isArray(chunks))
                    throw Error('Missing output chunks');
                for (const chunk of chunks)
                    expand(task.attrs.body, { ...ctx, resources: { ...ctx.resources, chunk }, variables: { ...ctx.variables, chunk_first: record(chunk).first } }, part, inSequence);
            }
            else if (task.op === 'for_each_head_row') {
                for (let row = 0; row < Number(ctx.variables.head_tokens); row++)
                    expand(task.attrs.body, ctx, { ...part, row }, inSequence);
            }
            else {
                const args = task.attrs.call.map(arg => resolve(arg, ctx, task, part));
                validateResolved(task.op, args);
                validateBufferLayout(task.op, args, ctx.geometry);
                commands.push({ kernel: kernels[task.op], args, op: task.op });
            }
            if (task.attrs.layer_end)
                commands.push({ layerEnd: true });
        }
    }
    expand(plan.tasks, context, context.sequences[0]); // Resolve everything before encoding GPU work.
    if (gpu.profiling)
        gpu.pendingExpandMs = performance.now() - expandStart;
    const per = Math.max(1, Math.floor(plan.submit_token_layers / tokens));
    const total = commands.filter(c => c.layerEnd).length;
    const state = lifecycle(context.stateOwner);
    state.busy = true;
    try {
        upload?.();
        let count = 0, encoded = 0, start = performance.now();
        gpu.begin();
        const fuse = gpu.fuse !== false && typeof gpu.normAdd === 'function';
        for (let i = 0; i < commands.length; i++) {
            const command = commands[i];
            // A norm whose output the very next command adds into the residual, over exactly tokens x size (one group,
            // contiguous, no gate or offsets): one fused dispatch instead of two.
            const next = commands[i + 1];
            if (fuse && command.op === 'rms_norm' && next?.op === 'add') {
                const [x, w, y, t, opts] = command.args;
                const [a, res, n] = next.args;
                const plain = Object.keys(opts || {}).every(k => k === 'size' || k === 'eps' || (k === 'groups' && opts.groups === 1));
                if (a === y && plain && typeof opts.size === 'number' && n === t * Number(opts.size) && res !== y && res !== x) {
                    gpu.normAdd(x, w, y, res, t, Number(opts.size), Number(opts.eps ?? 1e-6));
                    i++; // the add is done; its layer boundary (the next command) is still counted
                    continue;
                }
            }
            // Two consecutive grouped norms of the same group size over different buffers (QK-norm): one dispatch.
            if (fuse && typeof gpu.normPair === 'function' && command.op === 'rms_norm' && next?.op === 'rms_norm') {
                const [xa, wa, ya, ta, oa] = command.args;
                const [xb, wb, yb, tb, ob] = next.args;
                const grouped = (o) => Object.keys(o || {}).every(k => k === 'size' || k === 'eps' || k === 'groups') &&
                    Number.isSafeInteger(o.groups) && Number(o.groups) >= 1 && typeof o.size === 'number';
                const distinct = new Set([xa, ya, xb, yb]).size === 4 && ya !== wb && yb !== wa;
                if (grouped(oa) && grouped(ob) && ta === tb && oa.size === ob.size && (oa.eps ?? 1e-6) === (ob.eps ?? 1e-6) && distinct) {
                    gpu.normPair(xa, wa, ya, Number(oa.groups), xb, wb, yb, Number(ob.groups), ta, Number(oa.size), Number(oa.eps ?? 1e-6));
                    i++;
                    continue;
                }
            }
            // WS-36: consecutive one-token products of the same input (plain, or only an output offset) in one dispatch;
            // gate and up followed by silu_mul over the up output as one fused GLU dispatch. WS-49: a plain one-token
            // rms_norm whose output that group reads is folded into it (the norm output is still written).
            if (fuse && typeof gpu.matmulGroup === 'function' && (command.op === 'matmul' || (gpu.normFold && command.op === 'rms_norm'))) {
                const plain = (c) => {
                    if (c?.op !== 'matmul')
                        return null;
                    const [W, x, y, t, opts] = c.args;
                    if (t !== 1 || Object.keys(opts || {}).some(k => k !== 'yOffset' && k !== 'yStride'))
                        return null;
                    if (x === y || (y && typeof y === 'object' && 'buffer' in y))
                        return null;
                    return { W, x, y, yOffset: Number(opts?.yOffset ?? 0) };
                };
                let norm = null, at = i;
                if (command.op === 'rms_norm') {
                    const [nx, nw, ny, nt, no] = command.args;
                    const simple = Object.keys(no || {}).every(k => k === 'size' || k === 'eps' || (k === 'groups' && no.groups === 1));
                    const objs = [nx, nw, ny].every(b => !(b && typeof b === 'object' && 'buffer' in b));
                    const W0 = plain(commands[i + 1])?.W;
                    if (!(nt === 1 && simple && objs && nx !== ny && W0 && no.size === W0.cols && plain(commands[i + 1]).x === ny)) {
                        if (command.kernel)
                            command.kernel(...command.args);
                        if (command.layerEnd) {
                            count++;
                            encoded++;
                        }
                        continue;
                    }
                    norm = { src: nx, weight: nw, eps: Number(no.eps ?? 1e-6) };
                    at = i + 1;
                }
                const first = plain(commands[at]);
                const items = first ? [first] : [];
                const limit = gpu.device?.limits?.maxStorageBuffersPerShaderStage ?? 8;
                while (first && items.length < 4) {
                    const nxt = plain(commands[at + items.length]);
                    if (!nxt || nxt.x !== first.x || items.some(it => it.y === nxt.x))
                        break;
                    const cand = [...items, nxt], ys = new Set(cand.map(it => it.y));
                    if (norm && [...ys].some(y => y === norm.src || y === norm.weight))
                        break;
                    if (cand.length + 1 + ys.size + (norm ? 2 : 0) > limit || !gpu.groupShape(cand.map(it => it.W)))
                        break;
                    items.push(nxt);
                }
                if (items.length >= 2) {
                    const after = commands[at + items.length];
                    const [a, b, n] = (after?.op === 'silu_mul' ? after.args : []);
                    const W0 = items[0].W;
                    const glu = items.length === 2 && after?.op === 'silu_mul' && a === items[0].y && b === items[1].y && n === W0.rows &&
                        items[1].W.rows === W0.rows && items[1].W.kind === items[0].W.kind && items[0].yOffset === 0 && items[1].yOffset === 0;
                    gpu.matmulGroup(items, first.x, glu, norm);
                    i = at + items.length - 1 + (glu ? 1 : 0);
                    continue;
                }
                if (norm) {
                    if (command.kernel)
                        command.kernel(...command.args);
                    continue;
                }
            }
            if (command.kernel)
                command.kernel(...command.args);
            if (command.layerEnd) {
                count++;
                encoded++;
            }
            if (count === per && encoded < total) {
                await gpu.end();
                submitMs.push(performance.now() - start);
                submitLayers.push(count);
                count = 0;
                start = performance.now();
                gpu.begin();
            }
        }
        await gpu.end();
        submitMs.push(performance.now() - start);
        submitLayers.push(count);
    }
    catch (error) {
        for (const slot of slots)
            state.dirty.add(slot);
        throw error;
    }
    finally {
        state.busy = false;
    }
}
export async function residentDecode(stage, gpu, first, n, slot, stop, onToken) {
    const saves = !stage.residentDecode; // WS-43: recurrent state, undone from a snapshot taken before each step
    if (saves && !(stage.snapshot && stage.restore))
        throw new Error("This stage keeps recurrent state it cannot restore: no resident decode");
    const out = [];
    if (n < 1 || stop(first))
        return out;
    const device = gpu.device, pool = [];
    const staging = () => pool.pop() ?? device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const saved = { defer: gpu.defer, profiling: gpu.profiling };
    gpu.defer = true;
    gpu.profiling = false;
    try {
        // Step 0 consumes `first` (already in stage.token); step i consumes token i, read back while step i runs.
        for (let i = 0;; i++) {
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(stage.token, 0, stage.ids, 0, 4);
            const b = i > 0 ? staging() : null;
            if (b)
                enc.copyBufferToBuffer(stage.token, 0, b, 0, 4);
            device.queue.submit([enc.finish()]);
            // Mapped only after the copy into it is submitted (a buffer with a pending map cannot be used in a submit).
            const read = b ? b.mapAsync(GPUMapMode.READ).then(() => { const v = new Uint32Array(b.getMappedRange())[0]; b.unmap(); pool.push(b); return v; }) : null;
            const ran = i < n; // step i produces token i+1; no step past the budget
            if (ran && saves && i > 0)
                stage.snapshot(slot); // queue order: before step i (step 0 is never undone)
            if (ran)
                await stage.run({ tokens: 1, slot, gpuIds: true });
            if (!read)
                continue;
            const token = await read;
            out.push(token);
            onToken?.(token);
            if (stop(token) || out.length >= n) {
                if (ran && saves)
                    stage.restore(slot); // step i ran past the end: its state and position undone
                else if (ran) {
                    stage.position -= 1;
                    stage.positions[slot] = stage.position;
                }
                break;
            }
        }
        await device.queue.onSubmittedWorkDone();
    }
    finally {
        gpu.defer = saved.defer;
        gpu.profiling = saved.profiling;
        for (const b of pool)
            b.destroy();
    }
    return out;
}
