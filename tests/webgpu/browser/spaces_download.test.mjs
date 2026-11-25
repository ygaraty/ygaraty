import {test} from 'node:test';
import assert from 'node:assert/strict';
import {downloadPiece} from '../../../ygaraty/backends/webgpu/browser/spaces_download.mjs';

const N = 4000, HASH = 'h';
const describe = () => Promise.resolve({url: 'https://example.test/p', bytes: N, sha256: HASH});
// A response sending `chunk` bytes every `everyMs` (a crawling flow), or everything at once.
function response(chunk, everyMs, signal) {
  let sent = 0, timer;
  const body = new ReadableStream({
    pull(c) { return new Promise((resolve) => { timer = setTimeout(() => { const n = Math.min(chunk, N - sent); sent += n; c.enqueue(new Uint8Array(n)); if (sent >= N) c.close(); resolve(); }, everyMs); }); },
    cancel() { clearTimeout(timer); },
  });
  signal?.addEventListener('abort', () => clearTimeout(timer));
  return new Response(body, {status: 200, headers: {'Content-Length': String(N)}});
}

test('a crawling request below the rate floor is asked for again and the fast one completes', async () => {
  const calls = []; let slow = 0;
  const fetcher = (url, opts) => { calls.push(url); return Promise.resolve(calls.length === 1 ? response(10, 50, opts.signal) : response(N, 1, opts.signal)); };
  const data = await downloadPiece(describe, N, HASH, () => {}, fetcher, {minRate: 2000, rateWindowMs: 200, graceMs: 50, onSlow: () => slow++});
  assert.equal(data.byteLength, N);
  assert.equal(calls.length, 2);
  assert.equal(slow, 1);
});

test('without a floor the crawling request is kept; after maxSlow retries the slow one is kept too', async () => {
  let calls = 0;
  const crawl = (url, opts) => { calls++; return Promise.resolve(response(400, 20, opts.signal)); };  // 20 kB/s
  assert.equal((await downloadPiece(describe, N, HASH, () => {}, crawl, {})).byteLength, N);
  assert.equal(calls, 1);
  calls = 0; let slow = 0;
  const slowest = (url, opts) => { calls++; return Promise.resolve(response(20, 50, opts.signal)); };  // 400 B/s
  const data = await downloadPiece(describe, N, HASH, () => {}, slowest, {minRate: 100000, rateWindowMs: 100, graceMs: 20, maxSlow: 2, onSlow: () => slow++});
  assert.equal(data.byteLength, N);
  assert.equal(slow, 2);
  assert.equal(calls, 3);
});

test('a fast request is never cut by the floor, and invalid floors are refused', async () => {
  let calls = 0;
  const fast = (url, opts) => { calls++; return Promise.resolve(response(1000, 5, opts.signal)); };
  assert.equal((await downloadPiece(describe, N, HASH, () => {}, fast, {minRate: 1000, rateWindowMs: 50, graceMs: 10})).byteLength, N);
  assert.equal(calls, 1);
  await assert.rejects(downloadPiece(describe, N, HASH, () => {}, fast, {minRate: -1}), /rate floor/);
});
