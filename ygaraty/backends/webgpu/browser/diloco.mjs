// Complete local DiLoCo rounds. All numerical work goes through the existing SDK.
// This module does no networking and evaluates only allowlisted, data-only tasks.
import {ComputeSDK, tensor} from './sdk.mjs';

// Float32 wire tensors arrive as base64 little-endian bytes ({shape, dtype: 'float32', base64}) and widen to float64.
const float32Data = text => {
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0));
  if (bytes.length % 4) throw Error('Invalid float32 tensor bytes');
  const view = new DataView(bytes.buffer), out = new Float64Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(4*i, true);
  return out;
};
const float32Base64 = data => {
  const view = new DataView(new ArrayBuffer(4*data.length));
  for (let i = 0; i < data.length; i++) view.setFloat32(4*i, data[i], true);
  const bytes = new Uint8Array(view.buffer); let text = '';
  for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(text);
};
// uint8 pixel constants ({shape, dtype: 'uint8', normalize: 'x/127.5-1', base64}) rebuild the recipe's float64 values exactly.
const pixelData = (text, normalize) => {
  if (normalize !== 'x/127.5-1') throw Error('Unknown pixel normalization');
  const bytes = Uint8Array.from(atob(text), c => c.charCodeAt(0)), out = new Float64Array(bytes.length);
  for (let i = 0; i < out.length; i++) out[i] = bytes[i]/127.5-1;
  return out;
};
const unpack = x => {
  const data = typeof x?.base64 !== 'string' ? x?.data
    : x.dtype === 'float32' ? float32Data(x.base64) : x.dtype === 'uint8' ? pixelData(x.base64, x.normalize) : undefined;
  if (!x || !Array.isArray(x.shape) || x.shape.length !== 2 ||
      !x.shape.every(n => Number.isSafeInteger(n) && n > 0) ||
      data?.length !== x.shape[0] * x.shape[1]) throw Error('Invalid tensor shape');
  const value = tensor([...x.shape], data);
  if (!value.data.every(Number.isFinite)) throw Error('Nonfinite tensor');
  return value;
};
const slice = (value, region) => {
  if (!region) return value;
  const [r0, r1, c0, c1] = region, [rows, cols] = value.shape;
  if (!region.every(Number.isSafeInteger) || region.length !== 4 ||
      r0 < 0 || r1 > rows || r0 >= r1 || c0 < 0 || c1 > cols || c0 >= c1)
    throw Error('Invalid input region');
  const data = new Float64Array((r1-r0)*(c1-c0));
  for (let r = r0; r < r1; r++) data.set(value.data.subarray(r*cols+c0, r*cols+c1), (r-r0)*(c1-c0));
  return {shape: [r1-r0, c1-c0], data};
};

export async function executeLocalPlan(plan, inputs, sdk, {precision='float64', arithmetic='compensated'}={}) {
  const values = new Map(Object.entries(inputs));
  const keep = new Set([plan.loss, ...Object.values(plan.updates)]);
  const remaining = new Map(), known = new Set(values.keys());
  const dependencies = plan.tasks.map(t => new Set(t.inputs.map(x => x.value)));
  for (let i = 0; i < plan.tasks.length; i++) {
    const task = plan.tasks[i];
    if (known.has(task.id) || [...dependencies[i]].some(k => !known.has(k))) throw Error('Invalid or unordered local plan');
    for (const dep of dependencies[i]) remaining.set(dep, (remaining.get(dep) || 0) + 1);
    known.add(task.id);
  }
  if ([...keep].some(k => !known.has(k))) throw Error('Missing local outputs');
  for (const key of values.keys()) if (!remaining.get(key) && !keep.has(key)) values.delete(key);
  const stats = {cpu_tasks: 0, gpu_tasks: 0, matmul_tasks: 0};
  for (let i = 0; i < plan.tasks.length; i++) {
    const task = plan.tasks[i], args = task.inputs.map(x => slice(values.get(x.value), x.region));
    const {value, backend} = await sdk.execute({...task, precision,
      attrs: {...task.attrs, gpu_arithmetic: arithmetic}}, args);
    if (value.shape[0] !== task.shape[0] || value.shape[1] !== task.shape[1] ||
        !value.data.every(Number.isFinite)) throw Error('Invalid local output');
    stats[backend === 'webgpu' ? 'gpu_tasks' : 'cpu_tasks']++;
    if (task.op === 'matmul') stats.matmul_tasks++;
    values.set(task.id, value);
    for (const dep of dependencies[i]) {
      remaining.set(dep, remaining.get(dep)-1);
      if (!remaining.get(dep) && !keep.has(dep)) values.delete(dep);
    }
    if (!remaining.get(task.id) && !keep.has(task.id)) values.delete(task.id);
  }
  return {values, stats};
}

export class DiLoCoTrainer {
  constructor(worker, {sdk=new ComputeSDK(), precision='float64', arithmetic='compensated'}={}) {
    if (!['float64', 'webgpu'].includes(precision) || !['float32', 'compensated'].includes(arithmetic))
      throw Error('Invalid precision or GPU arithmetic');
    this.worker = worker; this.sdk = sdk; this.precision = precision; this.arithmetic = arithmetic;
    this.step = 0; this.weights = null; this.moments = null; this.seenRounds = new Set(); this.version = -1;
    this.busy = false;
  }

  async runRound(assignment, plans) {
    if (this.busy) throw Error('A local round is already running');
    this.busy = true;
    try { return await this.trainRound(assignment, plans); }
    finally { this.busy = false; }
  }

  async trainRound(assignment, plans) {
    if (assignment.worker !== this.worker || this.seenRounds.has(assignment.round_id) || assignment.version < this.version ||
        !Number.isSafeInteger(assignment.steps) || assignment.steps < 1) throw Error('Invalid round assignment');
    const base = Object.fromEntries(Object.entries(assignment.weights).map(([k, v]) => [k, unpack(v)]));
    if (!Object.keys(base).length) throw Error('Empty model');
    if (this.weights && (Object.keys(base).length !== Object.keys(this.weights).length ||
        Object.keys(base).some(n => !this.weights[n] || base[n].shape.some((s,i) => s !== this.weights[n].shape[i]))))
      throw Error('Assignment model changed');
    let weights = base, moments = this.moments || Object.fromEntries(['m', 'v'].map(kind =>
      [kind, Object.fromEntries(Object.entries(base).map(([n,w]) => [n, tensor(w.shape, new Float64Array(w.data.length))]))]));
    const losses = [], stats = {cpu_tasks: 0, gpu_tasks: 0, matmul_tasks: 0};
    for await (const plan of plans) {
      if (losses.length >= assignment.steps || plan.step !== this.step + losses.length + 1)
        throw Error('Wrong local optimizer step');
      const names = Object.keys(plan.parameters);
      if (names.length !== Object.keys(base).length || names.some(n => !base[n])) throw Error('Plan model mismatch');
      const inputs = Object.fromEntries(Object.entries(plan.constants).map(([k,v]) => [k, unpack(v)]));
      for (const [name, p] of Object.entries(plan.parameters)) {
        inputs[p] = weights[name];
        for (const kind of ['m', 'v']) inputs[`adam_${kind}_${p}`] = moments[kind][name];
      }
      const result = await executeLocalPlan(plan, inputs, this.sdk, this);
      for (const key of Object.keys(stats)) stats[key] += result.stats[key];
      losses.push(result.values.get(plan.loss).data[0]);
      weights = Object.fromEntries(Object.entries(plan.parameters).map(([n,p]) => [n, result.values.get(plan.updates[p])]));
      moments = Object.fromEntries(['m', 'v'].map(kind => [kind, Object.fromEntries(Object.entries(plan.parameters)
        .map(([n,p]) => [n, result.values.get(plan.updates[`adam_${kind}_${p}`])]))]));
    }
    if (losses.length !== assignment.steps) throw Error('Not enough local plans');
    const pseudo_gradient = Object.fromEntries(Object.keys(base).map(name => {
      const before = base[name], after = weights[name];
      const data = Float64Array.from(before.data, (x, i) => x - after.data[i]);
      if (!data.every(Number.isFinite)) throw Error('Nonfinite pseudo-gradient');
      return [name, assignment.wire_dtype === 'float32'
        ? {shape: [...before.shape], dtype: 'float32', base64: float32Base64(data)}
        : {shape: [...before.shape], data: Array.from(data)}];
    }));
    this.weights = weights; this.moments = moments; this.step += assignment.steps;
    this.seenRounds.add(assignment.round_id); this.version = assignment.version;
    return {round_id: assignment.round_id, version: assignment.version, worker: this.worker,
      steps: assignment.steps, pseudo_gradient, losses, stats};
  }
}
