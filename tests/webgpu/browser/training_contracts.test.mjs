import {test} from 'node:test';
import assert from 'node:assert/strict';
import {decodeJoin, decodeAssignment, decodeTensor} from '../../../ygaraty/backends/webgpu/browser/training_contracts.mjs';
import {DiLoCoTrainer} from '../../../ygaraty/backends/webgpu/browser/diloco.mjs';
import {contribute} from '../../../ygaraty/backends/webgpu/browser/training_client.mjs';

const tensor = {shape: [1, 1], data: [1]};
const bundle = () => ({state: 'assignment', format: 'ygaraty-diloco-round-v1',
  optimizer: {step: 0, moments: null},
  assignment: {worker: 'worker-0', round_id: 'round', version: 0, steps: 1, weights: {w: tensor}},
  plans: [{step: 1, loss: 'loss', constants: {x: tensor}, parameters: {w: 'w'}, updates: {w: 'out'},
    tasks: [{id: 'out', op: 'matmul', shape: [1, 1], attrs: {}, inputs: [{value: 'x', region: [0, 1, 0, 1]}]}]}]});

test('training protocol preserves zero-based regions and rejects malformed network data', () => {
  const decoded = decodeAssignment(bundle());
  assert.deepEqual(decoded.plans[0].tasks[0].inputs[0].region, [0, 1, 0, 1]);
  for (const mutate of [b => b.assignment.steps = -1, b => b.optimizer.step = '0',
    b => b.plans[0].parameters.w = 3, b => b.format = 'unknown',
    b => b.plans[0].tasks[0].inputs[0].region = [0, 'one', 0, 1]]) {
    const invalid = bundle(); mutate(invalid);
    assert.throws(() => decodeAssignment(invalid));
  }
});

test('tensor and contributor contracts reject unknown encodings and invalid credentials', () => {
  for (const value of [{shape: [1], data: [1]}, {shape: [1, 2], data: [1]},
    {shape: [1, 1], data: [NaN]}, {...tensor, dtype: 'unsupported'}])
    assert.throws(() => decodeTensor(value));
  assert.deepEqual(decodeJoin({state: 'complete', version: 2}), {state: 'complete', version: 2});
  assert.throws(() => decodeJoin({worker: 'w', token: 7, precision: 'float64', heartbeat_seconds: 1}));
  assert.throws(() => decodeJoin({worker: 'w', token: 't', precision: 'webgpu', heartbeat_seconds: Infinity}));
});

test('optimizer checkpoints round-trip typed tensors and reject mismatched moments', () => {
  const trainer = new DiLoCoTrainer('worker-0');
  const state = {step: 1, moments: {m: {w: tensor}, v: {w: tensor}}};
  trainer.restore(state, {w: tensor});
  assert.deepEqual(trainer.checkpoint(), state);
  assert.throws(() => trainer.restore({...state, moments: {m: {w: tensor}, v: {w: {...tensor, data: [-1]}}}}, {w: tensor}));
});

test('cancelled training never publishes local optimizer state', async () => {
  const controller = new AbortController(); controller.abort();
  const trainer = new DiLoCoTrainer('worker-0', {signal: controller.signal});
  await assert.rejects(trainer.runRound(bundle().assignment, bundle().plans), {name: 'AbortError'});
  assert.deepEqual(trainer.checkpoint(), {step: 0, moments: null});
});

test('contributor rejects malformed successful HTTP responses at the boundary', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({worker: 'w', token: 7, precision: 'float64', heartbeat_seconds: 1}));
  try {
    await assert.rejects(contribute({baseURL: 'http://localhost', joinToken: 'test',
      sdk: {initialize: async () => 'test'}}), /Expected string/);
  } finally { globalThis.fetch = original; }
});
