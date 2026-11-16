// Exercise the shipped fetch/heartbeat code without a browser or GPU.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source = readFileSync(new URL('../../../ygaraty/backends/webgpu/browser/qwen35_contributor.mjs', import.meta.url), 'utf8');

test('WebRTC requests and background heartbeats retain instance authorization', async () => {
  const requests = [];
  const context = {
    headers: {'X-Key': 'invitation', 'X-Worker-Key': 'instance'},
    connectionGeneration: 0, BASE: '', performance, AbortSignal,
    connectionEvent() {}, contributorContact() {}, beatFailed() {},
    fetch: async (url, options) => {
      requests.push({url, options});
      return {ok: true, status: 200, json: async () => ({ice: []})};
    },
  };
  runInNewContext(source.slice(source.indexOf('async function call('), source.indexOf('const get = async')) + '\nglobalThis.request = call;', context);
  await context.request('ice/worker');
  await context.request('signal/worker', {to: 1, data: {}});
  for (const request of requests) {
    assert.equal(request.options.headers['X-Key'], 'invitation');
    assert.equal(request.options.headers['X-Worker-Key'], 'instance');
  }
  const container = {};
  runInNewContext(source.slice(source.indexOf('const BEAT ='), source.indexOf('function startBeats(')) + '\nglobalThis.worker = BEAT;', container);
  const worker = {
    fetch: context.fetch, AbortSignal, postMessage() {},
    setInterval() {return 1;}, clearInterval() {},
  };
  runInNewContext(container.worker, worker);
  worker.onmessage({data: {url: '/api/heartbeat/worker', headers: context.headers, progress: {}}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.at(-1).options.headers['X-Worker-Key'], 'instance');
});
