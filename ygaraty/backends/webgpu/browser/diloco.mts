// Generated output: diloco.mjs. Edit this TypeScript source.
// Complete local DiLoCo rounds. All numerical work goes through the existing SDK.
// This module does no networking and evaluates only allowlisted, data-only tasks.
import {stateReferences} from './resident_training_plan.mjs';
import {ComputeSDK} from './sdk_compute.mjs';
import {tensor, type Tensor} from './sdk_tensor.mjs';
import {momentKinds} from './training_contracts.mjs';
import type {Assignment, WireTensor, Named, Moments, OptimizerState, ExecutionOptions, TrainingExecutor, StepPlan, RoundResult, ExecutionStats, ArrayTensor} from './training_contracts.mjs';
import type {DenseArithmetic} from './webgpu_contracts.mjs';
import type {Precision} from './training_contracts.mjs';

// Float32 wire tensors arrive as base64 little-endian bytes ({shape, dtype: 'float32', base64}) and widen to float64.
const float32Data = (text: string) => {
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0));
  if (bytes.length % 4) throw Error('Invalid float32 tensor bytes');
  const view = new DataView(bytes.buffer), out = new Float64Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4*i, true);
  return out;
};
const float32Base64 = (data: Float64Array) => {
  const view = new DataView(new ArrayBuffer(4*data.length));
  for (let i = 0; i < data.length; i++) view.setFloat32(4*i, data[i], true);
  const bytes = new Uint8Array(view.buffer); let text = '';
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
};
// uint8 pixel constants ({shape, dtype: 'uint8', normalize: 'x/127.5-1', base64}) rebuild the recipe's float64 values exactly.
const pixelData = (text: string, normalize: string) => {
  if (normalize !== 'x/127.5-1') throw Error('Unknown pixel normalization');
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0)), out = new Float64Array(bytes.length);
  for (let i = 0; i < out.length; i++) out[i] = bytes[i]/127.5-1;
  return out;
};
const unpack = (x: WireTensor): Tensor => {
  const data = 'data' in x ? x.data
    : x.dtype === 'float32' ? float32Data(x.base64) : pixelData(x.base64, x.normalize);
  if (!x || !Array.isArray(x.shape) || x.shape.length !== 2 ||
      !x.shape.every(n => Number.isSafeInteger(n) && n > 0) ||
      data?.length !== x.shape[0] * x.shape[1]) throw Error('Invalid tensor shape');
  const value = tensor([...x.shape], data);
  if (!value.data.every(Number.isFinite)) throw Error('Nonfinite tensor');
  return value;
};
const slice = (value: Tensor, region?: number[] | null): Tensor => {
  if (!region) return value;
  const [r0, r1, c0, c1] = region, [rows, cols] = value.shape;
  if (!region.every(Number.isSafeInteger) || region.length !== 4 ||
      r0 < 0 || r1 > rows || r0 >= r1 || c0 < 0 || c1 > cols || c0 >= c1)
    throw Error('Invalid input region');
  const data = new Float64Array((r1-r0)*(c1-c0));
  for (let r = r0; r < r1; r++) data.set(value.data.subarray(r*cols+c0, r*cols+c1), (r-r0)*(c1-c0));
  return {shape: [r1-r0, c1-c0], data};
};

export async function executeLocalPlan(plan: StepPlan, inputs: Named<Tensor>, sdk: TrainingExecutor, {precision='float64', arithmetic='compensated', signal}: ExecutionOptions={}) {
  const values = new Map(Object.entries(inputs));
  const keep = new Set([plan.loss, ...Object.values(plan.updates)]);
  const remaining = new Map<string, number>(), known = new Set(values.keys());
  const dependencies = plan.tasks.map(t => new Set(t.inputs.map(x => x.value)));
  for (let i = 0; i < plan.tasks.length; i++) {
    const task = plan.tasks[i];
    if (known.has(task.id) || [...dependencies[i]].some(k => !known.has(k))) throw Error('Invalid or unordered local plan');
    for (const dep of dependencies[i]) remaining.set(dep, (remaining.get(dep) || 0) + 1);
    known.add(task.id);
  }
  if ([...keep].some(k => !known.has(k))) throw Error('Missing local outputs');
  for (const key of values.keys()) if (!remaining.get(key) && !keep.has(key)) values.delete(key);
  const stats: ExecutionStats = {cpu_tasks: 0, gpu_tasks: 0, matmul_tasks: 0, optimizer_seconds: 0};
  let portableUntil = 0;
  for (let i = 0; i < plan.tasks.length; i++) {
    signal?.throwIfAborted();
    const block = plan.tasks[i].attrs.muon_block;
    if (i >= portableUntil && precision === 'webgpu' && typeof block === 'string' && sdk.executeMuonBlock) {
      let end = i + 1;
      while (end < plan.tasks.length && plan.tasks[end].attrs.muon_block === block) end++;
      const tasks = plan.tasks.slice(i, end), uses = new Map<string, number>();
      for (let j = i; j < end; j++) for (const dep of dependencies[j]) uses.set(dep, (uses.get(dep) || 0) + 1);
      const outputs = tasks.filter(t => keep.has(t.id) || (remaining.get(t.id) || 0) > (uses.get(t.id) || 0)).map(t => t.id);
      const start = performance.now();
      const result = await sdk.executeMuonBlock(tasks, values, outputs, arithmetic);
      if (result) {
        signal?.throwIfAborted();
        for (const id of outputs) {
          const value = required(result, id), task = tasks.find(t => t.id === id)!;
          if (value.shape[0] !== task.shape[0] || value.shape[1] !== task.shape[1] || !value.data.every(Number.isFinite)) throw Error('Invalid resident output');
          values.set(id, value);
        }
        stats.optimizer_seconds! += (performance.now() - start) / 1000;
        stats.gpu_tasks += tasks.filter(t => t.op !== 'assemble').length;
        stats.matmul_tasks += tasks.filter(t => t.op === 'matmul').length;
        for (const [dep, used] of uses) {
          remaining.set(dep, (remaining.get(dep) || 0) - used);
          if (!remaining.get(dep) && !keep.has(dep)) values.delete(dep);
        }
        i = end - 1;
        continue;
      }
      portableUntil = end;
    }
    const task = plan.tasks[i], args = task.inputs.map(x => slice(required(values, x.value), x.region));
    const taskStart = performance.now();
    const {value, backend} = await sdk.execute({...task, precision,
      attrs: {...task.attrs, gpu_arithmetic: arithmetic}}, args);
    if (task.attrs.optimizer) stats.optimizer_seconds! += (performance.now() - taskStart) / 1000;
    if (value.shape[0] !== task.shape[0] || value.shape[1] !== task.shape[1] ||
        !value.data.every(Number.isFinite)) throw Error('Invalid local output');
    stats[backend === 'webgpu' ? 'gpu_tasks' : 'cpu_tasks']++;
    if (task.op === 'matmul') stats.matmul_tasks++;
    values.set(task.id, value);
    for (const dep of dependencies[i]) {
      remaining.set(dep, (remaining.get(dep) ?? 0)-1);
      if (!remaining.get(dep) && !keep.has(dep)) values.delete(dep);
    }
    if (!remaining.get(task.id) && !keep.has(task.id)) values.delete(task.id);
  }
  return {values, stats};
}

function required(values: Map<string, Tensor>, name: string): Tensor {
  const value = values.get(name);
  if (!value) throw Error(`Missing local tensor: ${name}`);
  return value;
}
function momentsFrom<T>(build: (kind: 'm' | 'v') => Named<T>): Moments<T> {
  return {m: build('m'), v: build('v')};
}

export class DiLoCoTrainer {
  worker: string;
  sdk: TrainingExecutor;
  precision: Precision;
  arithmetic: DenseArithmetic;
  signal?: AbortSignal;
  step = 0;
  weights: Named<Tensor> | null = null;
  moments: Moments<Tensor> | null = null;
  seenRounds = new Set<string>();
  version = -1;
  busy = false;
  constructor(worker: string, {sdk=new ComputeSDK(), precision='float64', arithmetic='compensated', signal}: ExecutionOptions & {sdk?: TrainingExecutor}={}) {
    if (!['float64', 'webgpu'].includes(precision) || !['float32', 'compensated'].includes(arithmetic))
      throw Error('Invalid precision or GPU arithmetic');
    this.worker = worker; this.sdk = sdk; this.precision = precision; this.arithmetic = arithmetic;
    this.signal = signal;
  }

  checkpoint(): OptimizerState {
    const pack = (value: Tensor): ArrayTensor => ({shape: [...value.shape], data: Array.from(value.data)});
    const moments = this.moments;
    return {step: this.step, moments: moments && momentsFrom(kind =>
      Object.fromEntries(Object.entries(moments[kind]).map(([name, value]) => [name, pack(value)])))};
  }

  restore(state: OptimizerState, weights: Named<WireTensor>) {
    if (this.busy || !Number.isSafeInteger(state.step) || state.step < 0)
      throw Error('Invalid optimizer checkpoint');
    const expected = Object.fromEntries(Object.entries(weights).map(([n, v]) => [n, unpack(v)]));
    let moments: Moments<Tensor> | null = null;
    if (state.moments !== null || state.step !== 0) {
      if (!state.moments || Object.keys(state.moments).sort().join(',') !== 'm,v')
        throw Error('Incomplete optimizer checkpoint');
      const saved = state.moments;
      moments = momentsFrom(kind => {
        const values = saved[kind];
        if (Object.keys(values).some(n => !expected[n]) || kind === 'm' && Object.keys(values).length !== Object.keys(expected).length)
          throw Error('Checkpoint model mismatch');
        return Object.fromEntries(Object.entries(values).map(([name, raw]) => {
          const value = unpack(raw);
          if (value.shape.some((n, i) => n !== expected[name].shape[i]) ||
              kind === 'v' && value.data.some(x => x < 0)) throw Error('Invalid optimizer moment');
          return [name, value];
        }));
      });
    }
    this.step = state.step; this.moments = moments; this.weights = null;
    this.seenRounds = new Set(); this.version = -1;
  }

  async runRound(assignment: Assignment, plans: Iterable<StepPlan> | AsyncIterable<StepPlan>): Promise<RoundResult> {
    if (this.busy) throw Error('A local round is already running');
    this.busy = true;
    try { return await this.trainRound(assignment, plans); }
    finally { this.busy = false; }
  }

  async trainRound(assignment: Assignment, plans: Iterable<StepPlan> | AsyncIterable<StepPlan>): Promise<RoundResult> {
    if (assignment.worker !== this.worker || this.seenRounds.has(assignment.round_id) || assignment.version < this.version ||
        !Number.isSafeInteger(assignment.steps) || assignment.steps < 1) throw Error('Invalid round assignment');
    const base = Object.fromEntries(Object.entries(assignment.weights).map(([k, v]) => [k, unpack(v)]));
    if (!Object.keys(base).length) throw Error('Empty model');
    const previous = this.weights;
    if (previous && (Object.keys(base).length !== Object.keys(previous).length ||
        Object.keys(base).some(n => !previous[n] || base[n].shape.some((s,i) => s !== previous[n].shape[i]))))
      throw Error('Assignment model changed');
    let weights = base, moments = this.moments || momentsFrom(() => Object.fromEntries(Object.entries(base).map(([n,w]) => [n, tensor(w.shape, new Float64Array(w.data.length))])));
    const losses: number[] = [];
    const stats: ExecutionStats = {cpu_tasks: 0, gpu_tasks: 0, matmul_tasks: 0, optimizer_seconds: 0};
    const scheduled: StepPlan[] = [];
    for await (const plan of plans) {
      this.signal?.throwIfAborted();
      if (scheduled.length >= assignment.steps || plan.step !== this.step + scheduled.length + 1)
        throw Error('Wrong local optimizer step');
      scheduled.push(plan);
    }
    if (scheduled.length !== assignment.steps) throw Error('Not enough local plans');
    if (!this.moments) {
      const first = stateReferences(scheduled[0]);
      if (Object.keys(first.m).sort().join(',') !== Object.keys(base).sort().join(',') || Object.keys(first.v).some(n => !base[n]))
        throw Error('Optimizer state layout changed');
      moments = momentsFrom(kind => Object.fromEntries(Object.keys(first[kind]).map(n => [n,moments[kind][n]])));
    }
    const resident = this.precision === 'webgpu' && this.sdk.executeTrainingRound
      ? await this.sdk.executeTrainingRound({steps: scheduled.map(plan => ({plan,
          constants: Object.fromEntries(Object.entries(plan.constants).map(([k,v])=>[k,unpack(v)]))})),
          weights, moments, signal: this.signal}, this.arithmetic) : null;
    if (resident) {
      weights = resident.weights; moments = resident.moments; losses.push(...resident.losses);
      // GPU timestamps are not collected: do not label host encoding time as optimizer time.
      delete (stats as ExecutionStats).optimizer_seconds;
      Object.assign(stats,resident.stats);
    }
    for (const plan of resident ? [] : scheduled) {
      this.signal?.throwIfAborted();
      if (losses.length >= assignment.steps || plan.step !== this.step + losses.length + 1)
        throw Error('Wrong local optimizer step');
      const names = Object.keys(plan.parameters);
      if (names.length !== Object.keys(base).length || names.some(n => !base[n])) throw Error('Plan model mismatch');
      const inputs = Object.fromEntries(Object.entries(plan.constants).map(([k,v]) => [k, unpack(v)]));
      for (const [name, p] of Object.entries(plan.parameters)) {
        inputs[p] = weights[name];
      }
      const stateParameters = plan.state_parameters || momentsFrom(kind => Object.fromEntries(Object.entries(plan.parameters).map(([n,p]) => [n, `adam_${kind}_${p}`])));
      if (!this.moments && losses.length === 0) moments = momentsFrom(kind => Object.fromEntries(Object.keys(stateParameters[kind]).map(n => [n, moments[kind][n]])));
      for (const kind of momentKinds) {
        if (Object.keys(stateParameters[kind]).sort().join(',') !== Object.keys(moments[kind]).sort().join(',')) throw Error('Optimizer state layout changed');
        for (const [name, ref] of Object.entries(stateParameters[kind])) inputs[ref] = moments[kind][name];
      }
      const result = await executeLocalPlan(plan, inputs, this.sdk, this);
      for (const key of (Object.keys(stats) as (keyof ExecutionStats)[])) stats[key] = (stats[key] ?? 0) + (result.stats[key] ?? 0);
      losses.push(required(result.values, plan.loss).data[0]);
      weights = Object.fromEntries(Object.entries(plan.parameters).map(([n,p]) => [n, required(result.values, plan.updates[p])]));
      moments = momentsFrom(kind => Object.fromEntries(Object.entries(stateParameters[kind])
        .map(([n,ref]) => [n, required(result.values, plan.updates[ref])])));
    }
    if (losses.length !== assignment.steps || !losses.every(Number.isFinite)) throw Error('Invalid local losses');
    const finalRefs = stateReferences(scheduled[scheduled.length - 1]);
    for (const [actual, expected, variance] of [[weights, base, false], [moments.m, finalRefs.m, false], [moments.v, finalRefs.v, true]] as const) {
      if (Object.keys(actual).sort().join(',') !== Object.keys(expected).sort().join(',')) throw Error('Invalid local state layout');
      for (const [name,value] of Object.entries(actual)) {
        if (!base[name] || value.shape.length !== base[name].shape.length || value.shape.some((n,i) => n !== base[name].shape[i]) || value.data.length !== base[name].data.length ||
            !value.data.every(x => Number.isFinite(x) && (!variance || x >= 0))) throw Error('Invalid local state output');
      }
    }
    const pseudo_gradient: Named<WireTensor> = Object.fromEntries(Object.keys(base).map(name => {
      const before = base[name], after = weights[name];
      const data = Float64Array.from(before.data, (x, i) => x - after.data[i]);
      if (!data.every(Number.isFinite)) throw Error('Nonfinite pseudo-gradient');
      return [name, assignment.wire_dtype === 'float32'
        ? {shape: [...before.shape], dtype: 'float32', base64: float32Base64(data)}
        : {shape: [...before.shape], data: Array.from(data)}];
    }));
    this.signal?.throwIfAborted();
    this.weights = weights; this.moments = moments; this.step += assignment.steps;
    this.seenRounds.add(assignment.round_id); this.version = assignment.version;
    return {round_id: assignment.round_id, version: assignment.version, worker: this.worker,
      steps: assignment.steps, pseudo_gradient, losses, stats};
  }
}
