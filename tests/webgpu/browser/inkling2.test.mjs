// DO-95: the Inkling 2 adapter's manifest checks and visit order (no GPU). GPU parity against the NumPy reference
// runs from runs/experiments/DO-95/tools (synth.py + gpu_check.mjs, real_ref.py for the GGUF).
import test from 'node:test';
import assert from 'node:assert/strict';
import {parseInkling2Manifest, isInkling2Manifest, visits} from '../../../ygaraty/backends/webgpu/browser/inkling2_gpu.mjs';

const sliding = Array.from({length: 24}, (_, i) => ![5, 11, 17, 23].includes(i));
const config = {family: 'inkling2', layers: 24, hidden: 2304, heads: 24, kv_heads: 8, head_dim: 128, vocab: 201024, vocab_unpadded: 200058,
  experts: 128, top_k: 8, expert_ffn: 1152, expert_scale: 8, eps: 1e-6, window: 1024, sliding, conv: 4, res_block: 6, rel_dim: 16,
  rel_extent: 1024, qk_clip: 22.627, rel_clip: 8, log_floor: 8192, log_alpha: 0.1, situ_beta: 4, situ_linear_beta: 25, logit_cap: 30,
  loop_start: 6, loop_end: 18, passes: 3, gate_rank: 64};
const tensors = {'blk.0.attn_q.weight': {kind: 'Q8_0', shape: [3072, 2304], offset: 0, bytes: 3072 * 2448},
  'blk.0.ffn_gate_up_exps.weight': {kind: 'IQ4_NL', shape: [128, 2304, 2304], offset: 1, bytes: 128 * 2304 * 1296}};

test('Inkling 2 Mini visits its looped core three times: 48 visits', () => {
  const order = visits(config);
  assert.equal(order.length, 48);
  assert.deepEqual(order.slice(0, 7), [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]);
  assert.deepEqual(order[18], [6, 1]);
  assert.deepEqual(order[41], [17, 2]);
  assert.deepEqual(order.at(-1), [23, 0]);
});

test('Inkling 2 manifests are recognized and checked', () => {
  const m = {config, tensors};
  assert.equal(isInkling2Manifest(m), true);
  assert.equal(parseInkling2Manifest(m).config.layers, 24);
  assert.throws(() => parseInkling2Manifest({config: {...config, loop_start: 5}, tensors}), /align/);
  assert.throws(() => parseInkling2Manifest({config: {...config, sliding: sliding.slice(1)}, tensors}), /sliding/);
  assert.throws(() => parseInkling2Manifest({config: {...config, head_dim: 256}, tensors}), /geometry/);
  assert.throws(() => parseInkling2Manifest({config, tensors: {x: {kind: 'Q3_K', shape: [1], offset: 0, bytes: 1}}}), /Invalid tensor/);
});
