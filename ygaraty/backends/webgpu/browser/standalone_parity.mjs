// A whole training program, run directly. No workers, scheduler, task claims,
// coordinator, DiLoCo rounds or tiled plans. Reuse only the numerical SDK.
import {ComputeSDK, tensor} from './sdk.mjs';
import {initializeParityGPU} from './standalone_gpu.mjs';

const unpack = value => tensor(value.shape, value.data);
const pack = value => ({shape: [...value.shape], data: Array.from(value.data)});
const packState = state => Object.fromEntries(Object.entries(state).map(([name, value]) => [name, pack(value)]));

export async function runProgram(fixture, mode, sdk=new ComputeSDK()) {
  if (!['float64', 'float32', 'compensated'].includes(mode)) throw Error('Unknown arithmetic mode');
  const gpu = mode !== 'float64', precision = gpu ? 'webgpu' : 'float64';
  if (gpu && !sdk.device) throw Error('WebGPU device required; CPU fallback is not allowed');
  let weights = Object.fromEntries(Object.entries(fixture.weights).map(([name, value]) => [name, unpack(value)]));
  let moments = Object.fromEntries(['m', 'v'].map(kind => [kind, Object.fromEntries(Object.entries(weights)
    .map(([name, value]) => [name, tensor(value.shape, new Float64Array(value.data.length))]))]));
  const records = [], options = fixture.optimizer;
  for (let cursor = 0; cursor < fixture.graphs.length; cursor++) {
    const graph = fixture.graphs[cursor], before = weights, step = cursor + 1;
    const values = new Map(Object.entries(graph.constants).map(([id, value]) => [id, unpack(value)]));
    for (const [name, id] of Object.entries(graph.parameters)) values.set(id, weights[name]);
    const stats = {operations: 0, matmuls: 0, gpu_matmuls: 0}, started = performance.now();
    // Interpret the original tape in order, without task lowering or scheduling.
    for (const node of graph.nodes) {
      const inputs = node.inputs.map(id => {
        if (!values.has(id)) throw Error(`Missing input ${id}`);
        return values.get(id);
      });
      const {value, backend} = await sdk.execute({...node, precision,
        attrs: {...node.attrs, gpu_arithmetic: mode}}, inputs);
      if (value.shape.some((size, i) => size !== node.shape[i]) || !value.data.every(Number.isFinite))
        throw Error(`Invalid output at ${node.id}`);
      values.set(node.id, value);
      stats.operations++;
      if (node.op === 'matmul') {
        stats.matmuls++;
        if (backend === 'webgpu') stats.gpu_matmuls++;
      }
    }
    if (gpu && stats.gpu_matmuls !== stats.matmuls) throw Error('A matrix multiplication did not execute on WebGPU');
    const gradients = {}, after = {}, nextMoments = {m: {}, v: {}};
    for (const [name, id] of Object.entries(graph.parameters)) {
      const gradient = values.get(graph.gradients[id]), shape = weights[name].shape;
      gradients[name] = gradient;
      // Adam is called directly, with exactly the SDK kernels used elsewhere.
      // Moments and the bias-correction clock persist through the whole program.
      for (const kind of ['m', 'v']) nextMoments[kind][name] = (await sdk.execute(
        {op: `adam_${kind}`, shape, precision: 'float64', attrs: options}, [moments[kind][name], gradient])).value;
      after[name] = (await sdk.execute({op: 'adam_update', shape, precision: 'float64',
        attrs: {...options, step}}, [weights[name], nextMoments.m[name], nextMoments.v[name]])).value;
    }
    records.push({step, loss: pack(values.get(graph.loss)), logits: graph.logits.map(id => pack(values.get(id))),
      before: packState(before), gradients: packState(gradients), after: packState(after),
      moments: {m: packState(nextMoments.m), v: packState(nextMoments.v)},
      execution: {...stats, seconds: (performance.now()-started)/1000}});
    weights = after; moments = nextMoments;
  }
  return {mode, records};
}

// HTTP is only a fixture/result bridge for the browser, with no training API.
// Importing this module from Node performs no I/O; runProgram is also usable there.
if (typeof window !== 'undefined') {
  let result;
  try {
    const fixture = await (await fetch('/fixture.json')).json(), sdk = new ComputeSDK();
    const hasGPUWork = fixture.modes.some(mode => mode !== 'float64');
    const gpu = hasGPUWork ? await initializeParityGPU(sdk, {requireNvidia: fixture.require_nvidia === true}) :
      {reason: await sdk.initialize(false), adapter: null};
    const runs = [];
    for (const mode of fixture.modes) runs.push(await runProgram(fixture, mode, sdk));
    result = {browser: navigator.userAgent, gpu_available: Boolean(sdk.device),
      device_status: gpu.reason, adapter: gpu.adapter, runs};
  } catch (error) {result = {error: String(error.stack || error)};}
  await fetch('/result', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(result)});
}
