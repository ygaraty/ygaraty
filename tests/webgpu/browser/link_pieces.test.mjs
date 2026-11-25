import {test} from 'node:test';
import assert from 'node:assert/strict';
import {pieces} from '../../../ygaraty/backends/webgpu/browser/qwen35_peer.mjs';

test('link bodies start in small pieces (WS-26), or whole PIECE-sized sends for DO-54 arm B', () => {
  assert.deepEqual(pieces(10 * 1024), [[0, 4096], [4096, 4096], [8192, 2048]]);
  assert.deepEqual(pieces(10 * 1024, true), [[0, 10240]]);
  const big = pieces(200 * 1024);
  assert.deepEqual(big.slice(0, 5), [[0, 4096], [4096, 4096], [8192, 4096], [12288, 4096], [16384, 65536]]);
  assert.equal(big.reduce((n, [, s]) => n + s, 0), 200 * 1024);
  assert.deepEqual(pieces(200 * 1024, true).map(([, s]) => s), [65536, 65536, 65536, 8192]);
});
