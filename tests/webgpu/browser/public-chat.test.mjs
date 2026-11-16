// Minimal browser DOM/fetch harness; no sockets or GPU.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {randomUUID} from 'node:crypto';

const source = readFileSync(new URL('../../../ygaraty/backends/webgpu/browser/public_chat.mjs', import.meta.url), 'utf8');
const settle = () => new Promise(resolve => setImmediate(resolve));
const IDS = ['key', 'api-url', 'model', 'model-name', 'send', 'status', 'clear', 'stop', 'chat', 'message', 'answer', 'empty', 'copy-key', 'copy-url', 'copy-model', 'show-key'];

class Node {
  constructor(id) {
    this.id = id; this.value = ''; this.textContent = ''; this.disabled = false; this.type = id === 'key' ? 'password' : 'text';
    this.className = ''; this.children = []; this.parent = null; this.style = {}; this.scrollHeight = 0;
    const classes = new Set();
    this.classList = {add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c)};
  }
  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  replaceChildren() { this.children = []; }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter(n => n !== this); }
  focus() {}
}

function browser({key = 'owner-key', state = 'unavailable', authorized = true, network = 'relay', stream} = {}) {
  const elements = Object.fromEntries(IDS.map(id => [id, new Node(id)]));
  elements.answer.append(elements.empty);
  elements.chat.requestSubmit = () => elements.chat.onsubmit({preventDefault() {}});
  const requests = [], timers = [], history = [], copied = [];
  const fixture = {state, authorized, network};
  let clock = 0;
  runInNewContext(source, {
    document: {getElementById: id => elements[id], createElement: () => new Node(), execCommand: () => true},
    navigator: {clipboard: {writeText: async text => copied.push(text)}},
    performance: {now: () => (clock += 250)},
    location: {hash: key ? '#key=' + encodeURIComponent(key) : '', origin: 'https://ai.example.com', pathname: '/chat', search: ''},
    history: {replaceState: (...args) => history.push(args)},
    crypto: {randomUUID}, URLSearchParams, AbortController, AbortSignal, TextDecoder,
    setTimeout: callback => { timers.push(callback); return timers.length; },
    clearTimeout: () => {},
    fetch: async (url, options) => {
      requests.push({url, options});
      if (!fixture.authorized) return new Response('{}', {status: 401});
      if (url === '/v1/session') return Response.json({state: fixture.state, network: fixture.network});
      if (url === '/v1/models') return Response.json({data: [{id: 'test-model'}]});
      if (stream) return stream(options);
      return new Response('data: {"choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo"}}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n');
    },
  });
  const thread = () => elements.answer.children.map(row => ({
    role: row.className.replace('message ', ''),
    text: row.children[0].textContent,
    stats: row.children[1]?.textContent,
    error: row.classList.contains('error'),
  }));
  return {elements, requests, timers, history, fixture, copied, thread};
}

test('private link supplies key, removes fragment, waits then enables chat', async () => {
  const b = browser();
  await settle();
  assert.equal(b.elements.key.value, 'owner-key');
  assert.equal(b.history[0][2], '/chat');
  assert.equal(b.elements['api-url'].value, 'https://ai.example.com/v1');
  assert.equal(b.elements.model.value, 'test-model');
  assert.equal(b.elements['model-name'].textContent, 'test-model');
  assert.equal(b.elements.send.disabled, true);
  assert.match(b.elements.status.textContent, /Waiting for workers/);
  assert.ok(b.requests.every(r => r.options.headers.Authorization === 'Bearer owner-key'));
  b.fixture.state = 'ready';
  await b.timers.at(-1)();
  assert.equal(b.elements.send.disabled, false);
  b.fixture.state = 'unavailable';
  await b.timers.at(-1)();
  assert.equal(b.elements.send.disabled, true);
});

test('missing and incorrect owner keys do not enable sending', async () => {
  const missing = browser({key: ''});
  await settle();
  assert.equal(missing.requests.length, 0);
  assert.equal(missing.elements.send.disabled, true);
  const wrong = browser({authorized: false});
  await settle();
  assert.equal(wrong.elements.send.disabled, true);
  assert.match(wrong.elements.status.textContent, /not accepted/);
});

test('copy buttons copy the key, base URL and model; show reveals the key', async () => {
  const b = browser({state: 'ready'});
  await settle();
  await b.elements['copy-key'].onclick();
  await b.elements['copy-url'].onclick();
  await b.elements['copy-model'].onclick();
  assert.deepEqual(b.copied, ['owner-key', 'https://ai.example.com/v1', 'test-model']);
  assert.equal(b.elements['copy-key'].textContent, 'Copied');
  b.elements['show-key'].onclick();
  assert.equal(b.elements.key.type, 'text');
  b.elements['show-key'].onclick();
  assert.equal(b.elements.key.type, 'password');
});

test('messages append as a thread with terminal-style stats and keep history', async () => {
  const b = browser({state: 'ready'});
  await settle();
  b.elements.message.value = 'Hi';
  await b.elements.chat.onsubmit({preventDefault() {}});
  b.elements.message.value = 'Again';
  await b.elements.chat.onsubmit({preventDefault() {}});
  const rows = b.thread();
  assert.deepEqual(rows.map(r => [r.role, r.text]), [['user', 'Hi'], ['assistant', 'Hello'], ['user', 'Again'], ['assistant', 'Hello']]);
  // Fake clock ticks 0.25 s per read: start, first token, end.
  assert.equal(rows[1].stats, 'TTFT 0.25 s · 2 tokens · 4.0 tok/s · relay');
  assert.equal(b.elements.message.value, '');
  const calls = b.requests.filter(r => r.url === '/v1/chat/completions');
  const first = JSON.parse(calls[0].options.body), second = JSON.parse(calls[1].options.body);
  assert.equal(first.conversation_id, second.conversation_id);
  assert.deepEqual(second.messages, [{role: 'user', content: 'Hi'}, {role: 'assistant', content: 'Hello'}, {role: 'user', content: 'Again'}]);
  b.elements.clear.onclick();
  assert.equal(b.thread().length, 0);
  b.elements.message.value = 'Fresh';
  await b.elements.chat.onsubmit({preventDefault() {}});
  const third = JSON.parse(b.requests.filter(r => r.url === '/v1/chat/completions')[2].options.body);
  assert.notEqual(third.conversation_id, first.conversation_id);
  assert.deepEqual(third.messages, [{role: 'user', content: 'Fresh'}]);
});

test('failed replies are marked and left out of the history', async () => {
  const b = browser({state: 'ready', stream: () => new Response('data: {"choices":[{"delta":{"content":"Par"}}]}\n\n')});
  await settle();
  b.elements.message.value = 'Hi';
  await b.elements.chat.onsubmit({preventDefault() {}});
  const [, reply] = b.thread();
  assert.equal(reply.text, 'Par');
  assert.ok(reply.error);
  assert.match(reply.stats, /Response interrupted · not kept/);
  b.elements.message.value = 'Again';
  await b.elements.chat.onsubmit({preventDefault() {}});
  const last = JSON.parse(b.requests.filter(r => r.url === '/v1/chat/completions').at(-1).options.body);
  assert.deepEqual(last.messages, [{role: 'user', content: 'Again'}]);
  assert.equal(b.elements.send.disabled, false);
});
