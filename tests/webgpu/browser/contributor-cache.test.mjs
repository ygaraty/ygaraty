import {test} from 'node:test';
import assert from 'node:assert/strict';
import {summary, describe, clear, mount, CACHE_NAMES} from '../../../ygaraty/backends/webgpu/browser/contributor_cache.mjs';

function fakeCaches(entries) {
  const stores = new Map([[CACHE_NAMES[0], entries], ['other-site-cache', ['x']]]);
  return {
    stores,
    has: async name => stores.has(name),
    open: async name => ({keys: async () => stores.get(name) || []}),
    delete: async name => stores.delete(name),
  };
}
const storage = {estimate: async () => ({usage: 1.5e9})};
const settle = () => new Promise(resolve => setImmediate(resolve));

test('summary counts saved weight files and reports site usage', async () => {
  assert.deepEqual(await summary({caches: fakeCaches(['a', 'b', 'c']), storage}), {available: true, files: 3, bytes: 1.5e9});
  assert.match(describe({available: true, files: 3, bytes: 1.5e9}), /^3 model files saved in this browser \(1\.50 GB\)/);
  assert.equal(describe({available: true, files: 0, bytes: 0}), 'No model files saved in this browser.');
  assert.match(describe(await summary({caches: undefined, storage})), /does not keep/);
});

test('clear deletes only the contributor weight cache', async () => {
  const caches = fakeCaches(['a']);
  assert.equal(await clear({caches}), 1);
  assert.deepEqual([...caches.stores.keys()], ['other-site-cache']);
  assert.equal(await clear({caches}), 0);
});

test('button asks first, clears, then shows nothing saved', async () => {
  const caches = fakeCaches(['a', 'b']);
  const element = {textContent: ''}, button = {disabled: true};
  const answers = [false, true];
  mount({element, button, caches, storage, confirm: () => answers.shift()});
  await settle();
  assert.match(element.textContent, /^2 model files/);
  assert.equal(button.disabled, false);
  await button.onclick();  // declined
  assert.ok(caches.stores.has(CACHE_NAMES[0]));
  await button.onclick();  // accepted
  assert.ok(!caches.stores.has(CACHE_NAMES[0]));
  assert.equal(element.textContent, 'No model files saved in this browser.');
  assert.equal(button.disabled, true);
});
