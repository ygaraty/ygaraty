// DO-80: the MoE adapter's qwen3moe profile accepts Qwen3-Coder-30B-A3B's manifest shape and rejects Kolibri-only
// options, sliding layers, missing or miscounted expert chunks (no GPU).
import test from 'node:test';
import assert from 'node:assert/strict';
import {parseManifest, isKolibriManifest} from '../../../ygaraty/backends/webgpu/browser/kolibri_gpu.mjs';

const H = 2048, F = 768, E = 128, D = 128;
const config = {family: 'qwen3moe', layers: 48, hidden: H, heads: 32, kv_heads: 4, head_dim: D, vocab: 151936, experts: E, top_k: 8,
  expert_ffn: F, norm_topk: true, rope_theta: 1e7, eps: 1e-6, attention_scale: D ** -0.5, tied_embeddings: false, intermediate: 8 * 2 * F};
const t = (shape, kind = 'Q4_K') => ({file: 'f', kind, shape, row_bytes: shape[shape.length - 1] / 256 * 144});
const f32 = (shape) => ({file: 'f', kind: 'F32', shape, row_bytes: 4 * shape[shape.length - 1]});
const tensors = {'attn_q.weight': t([4096, H]), 'attn_k.weight': t([512, H]), 'attn_v.weight': t([512, H]), 'attn_output.weight': t([H, 4096]),
  'ffn_gate_inp.weight': f32([E, H]), 'attn_norm.weight': f32([H]), 'ffn_norm.weight': f32([H]),
  'attn_q_norm.weight': f32([D]), 'attn_k_norm.weight': f32([D]),
  'ffn_gate_exps.0.weight': t([64 * F, H]), 'ffn_gate_exps.1.weight': t([64 * F, H]),
  'ffn_up_exps.0.weight': t([E * F, H]), 'ffn_down_exps.0.weight': t([E * H, F])};
const manifest = {config, layers: [{index: 0, type: 'full', tensors}]};
const layerWith = (changed) => ({...manifest, layers: [{index: 0, type: 'full', tensors: changed}]});

test('qwen3moe manifests use the MoE adapter', () => {
  assert.equal(parseManifest(manifest).config.family, 'qwen3moe');
  assert.equal(isKolibriManifest(manifest), true);
});

test('qwen3moe manifests reject Kolibri options, sliding layers and bad expert chunks', () => {
  assert.throws(() => parseManifest({...manifest, config: {...config, sliding: [false]}}), /qwen3moe routing or layer options/);
  assert.throws(() => parseManifest({...manifest, config: {...config, norm_topk: undefined}}), /qwen3moe routing or layer options/);
  assert.throws(() => parseManifest({...manifest, layers: [{index: 0, type: 'sliding', tensors}]}), /layer order\/type/);
  const noUp = Object.fromEntries(Object.entries(tensors).filter(([k]) => !k.startsWith('ffn_up_exps')));
  assert.throws(() => parseManifest(layerWith(noUp)), /ffn_up_exps chunks/);
  assert.throws(() => parseManifest(layerWith({...tensors, 'ffn_gate_exps.1.weight': t([63 * F, H])})), /127 experts, expected 128/);
});
