import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PeerTrainingWorker, validatePeerConfig} from '../../../ygaraty/backends/webgpu/browser/peer_training.mjs';
import {encodeFrame, decodeFrame} from '../../../ygaraty/backends/webgpu/browser/tensor_transport.mjs';
import {tensor} from '../../../ygaraty/backends/webgpu/browser/sdk_tensor.mjs';

const scalar = n => tensor([1, 1], [n]);
const token = 'test-only-'.repeat(4), session = 'test';
const gradientPlan = {step: 1, loss: 'loss', tasks: [], constants: {}, parameters: {w: 'w'}, updates: {w: 'gradient'}};
const optimizer = {step: 1, loss: 'out', constants: {}, parameters: {},
  updates: {w: 'out', adam_m_w: 'adam_m_w', adam_v_w: 'adam_v_w'},
  tasks: [{id: 'out', op: 'sgd', shape: [1, 1], inputs: [{value: 'w'}, {value: 'g'}], attrs: {learning_rate: 1}}]};

async function cluster(count, callback) {
  const peers = Array.from({length: count}, (_, i) => `http://worker-${i}:8780`);
  const workers = peers.map((_, rank) => new PeerTrainingWorker(validatePeerConfig({rank, peers, token, session, precision: 'float64', timeoutMs: 1000})));
  const original = globalThis.fetch, edges = [];
  const routes = [];
  for (const worker of workers) {
    const originalPeer = worker.peer.bind(worker);
    worker.peer = (rank, path, ...args) => {
      routes.push({from: worker.config.rank, to: rank, path});
      return originalPeer(rank, path, ...args);
    };
  }
  let failRank = -1;
  globalThis.fetch = async (url, options) => {
    const rank = peers.indexOf(new URL(url).origin), path = new URL(url).pathname;
    edges.push({rank, path});
    if (rank === failRank) throw new TypeError('Disconnected peer');
    return workers[rank].handle(new Request(url, options));
  };
  const call = async (rank, path, metadata = {}, tensors = {}) => {
    const response = await workers[rank].handle(new Request(peers[rank] + path, {method: 'POST',
      headers: {Authorization: `Bearer ${token}`}, body: await encodeFrame({session, ...metadata}, tensors)}));
    const value = await decodeFrame(new Uint8Array(await response.arrayBuffer()));
    return {status: response.status, ...value};
  };
  const prepare = async () => {
    assert.equal((await call(0, '/commit', {version: 0}, {w: scalar(100)})).status, 200);
    for (let rank = 0; rank < count; rank++) {
      assert.equal((await call(rank, '/prepare', {version: 1, counts: Array.from({length: count}, (_, i) => i + 1),
        plan: gradientPlan, ...(rank === 0 ? {optimizer, gradientRefs: {w: 'g'}} : {})},
      {gradient: scalar(rank + 1), loss: scalar(rank + 2)})).status, 200);
    }
  };
  try { await callback({workers, edges, routes, call, prepare, disconnect: rank => { failRank = rank; }}); }
  finally { globalThis.fetch = original; for (const worker of workers) worker.close(); }
}

for (const count of [1, 2, 4, 10]) test(`${count}-peer tree carries unequal sample counts, direct edges and duplicate step receipts`, async () => {
  await cluster(count, async ({call, prepare, edges, routes}) => {
    await prepare(); edges.length = 0;
    const [result, duplicate] = await Promise.all([call(0, '/step', {version: 1}), call(0, '/step', {version: 1})]);
    assert.equal(result.status, 200, JSON.stringify(result.metadata));
    assert.equal(duplicate.status, 200);
    const total = count * (count + 1) / 2;
    const expected = 100 - Array.from({length: count}, (_, i) => (i + 1) ** 2).reduce((a, b) => a + b, 0) / total;
    assert.ok(Math.abs(result.tensors.w.data[0] - expected) < 1e-12);
    assert.equal(result.metadata.samples, total);
    assert.equal(result.metadata.nodes.length, count);
    assert.equal(duplicate.tensors.w.data[0], result.tensors.w.data[0]);
    assert.equal(edges.filter(e => e.path === '/reduce').length, count - 1);
    assert.equal(edges.filter(e => e.path === '/commit').length, count - 1);
    for (const route of routes) assert.equal(route.from, Math.floor((route.to - 1) / 2));
    for (let rank = 0; rank < count; rank++) assert.equal((await call(rank, '/status')).metadata.version, 1);
  });
});

test('lost peer aborts the collective without publishing a root optimizer version', async () => {
  await cluster(4, async ({call, prepare, disconnect}) => {
    await prepare(); disconnect(3);
    assert.equal((await call(0, '/step', {version: 1})).status, 409);
    const status = (await call(0, '/status')).metadata;
    assert.equal(status.failed, true); assert.equal(status.version, 0);
    assert.equal((await call(0, '/step', {version: 1})).status, 409);
  });
});

test('cross-session, stale and conflicting preparation requests are rejected', async () => {
  await cluster(1, async ({call, prepare}) => {
    await prepare();
    assert.equal((await call(0, '/prepare', {session: 'other', version: 1})).status, 409);
    assert.equal((await call(0, '/prepare', {version: 0})).status, 409);
    assert.equal((await call(0, '/prepare', {version: 1, counts: [2]})).status, 409);
    assert.equal((await call(0, '/step', {version: 1})).status, 200);
  });
});

test('binary frames reject corruption, unknown versions, truncation and nonfinite tensors', async () => {
  const good = await encodeFrame({version: 1}, {w: scalar(3)});
  for (const index of [0, good.length - 1]) {
    const corrupt = good.slice(); corrupt[index] ^= 1;
    await assert.rejects(decodeFrame(corrupt));
  }
  await assert.rejects(decodeFrame(good.slice(0, -1)));
  await assert.rejects(encodeFrame({}, {w: scalar(Infinity)}));
});
