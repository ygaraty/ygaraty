// Private chat page: one conversation kept in memory only (never stored).
// After each answer it shows the same stats as the terminal chat: time to
// first token, tokens, decode rate after the first token, and the route.
const $ = id => document.getElementById(id);
let messages = [], conversation = crypto.randomUUID(), controller, ready = false, network = null;
const now = () => performance.now() / 1000;
const sharedKey = new URLSearchParams(location.hash.slice(1)).get('key');
if (sharedKey) {
  $('key').value = sharedKey;
  history.replaceState(null, '', location.pathname + location.search);
}
$('api-url').value = location.origin + '/v1';
$('send').disabled = true;

function setStatus(text, kind = '') {
  $('status').textContent = text;
  if ($('status').classList) $('status').className = kind;
}

function formatStats({ttft, tokens, rate}, route) {
  const parts = [];
  if (ttft != null) parts.push(`TTFT ${ttft.toFixed(2)} s`);
  parts.push(`${tokens} token${tokens === 1 ? '' : 's'}`);
  if (rate) parts.push(`${rate.toFixed(1)} tok/s`);
  if (route) parts.push(route);
  return parts.join(' · ');
}

let statusTimer;
async function refreshStatus() {
  clearTimeout(statusTimer);
  const credential = $('key').value;
  if (!credential) {
    ready = false;
    setStatus('Open your private chat link or enter the API key.', 'error');
  } else {
    try {
      const headers = {Authorization: `Bearer ${credential}`};
      const response = await fetch('/v1/session', {headers, signal: AbortSignal.timeout(5000)});
      if (!response.ok) throw Error(response.status === 401 ? 'API key not accepted.' : 'Server unavailable. Retrying…');
      const session = await response.json();
      if (credential !== $('key').value) return;
      ready = session.state === 'ready';
      network = session.network || null;
      if (!controller) setStatus(ready ? 'Ready' : 'Waiting for workers to open the contributor link…', ready ? 'ready' : '');
      const models = await fetch('/v1/models', {headers, signal: AbortSignal.timeout(5000)});
      if (models.ok && credential === $('key').value) {
        $('model').value = (await models.json()).data[0]?.id || '';
        if ($('model-name')) $('model-name').textContent = $('model').value;
      }
    } catch (error) {
      ready = false;
      if (!controller) setStatus(error.name === 'TimeoutError' ? 'Server unavailable. Retrying…' : error.message, 'error');
    }
  }
  $('send').disabled = !ready || !!controller;
  statusTimer = setTimeout(refreshStatus, 2000);
}
$('key').oninput = () => { ready = false; $('send').disabled = true; };
$('key').onchange = refreshStatus;
refreshStatus();

async function copy(id, button) {
  const value = $(id).value;
  try { await navigator.clipboard.writeText(value); }
  catch {  // insecure origin or denied: select it for Ctrl-C instead
    const input = $(id), type = input.type;
    input.type = 'text'; input.select?.(); document.execCommand?.('copy'); input.type = type;
  }
  const label = button.textContent;
  button.textContent = 'Copied'; setTimeout(() => { button.textContent = label; }, 1500);
}
for (const [button, field] of [['copy-key', 'key'], ['copy-url', 'api-url'], ['copy-model', 'model']]) {
  if ($(button)) $(button).onclick = () => copy(field, $(button));
}
if ($('show-key')) $('show-key').onclick = () => {
  const hidden = $('key').type === 'password';
  $('key').type = hidden ? 'text' : 'password';
  $('show-key').textContent = hidden ? 'Hide' : 'Show';
};

function addMessage(role, text = '') {
  $('empty')?.remove?.();
  const row = document.createElement('div');
  row.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.append(bubble);
  $('answer').append(row);
  scrollDown();
  return {row, bubble};
}
function addStats(row, text) {
  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.textContent = text;
  row.append(stats);
}
function scrollDown() {
  const thread = $('answer');
  thread.scrollTop = thread.scrollHeight;
}

$('clear').onclick = () => {
  if (controller) return;
  messages = []; conversation = crypto.randomUUID();
  $('answer').replaceChildren();
  $('message').focus?.();
};
$('stop').onclick = () => controller?.abort();
$('message').onkeydown = event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('chat').requestSubmit?.();
  }
};
$('message').oninput = () => {
  const box = $('message');
  if (!box.style) return;
  box.style.height = 'auto';
  box.style.height = `${box.scrollHeight}px`;
};

$('chat').onsubmit = async event => {
  event.preventDefault();
  const text = $('message').value.trim();
  if (controller || !ready || !text) return;
  controller = new AbortController(); $('send').disabled = true; $('stop').disabled = false;
  const pending = [...messages, {role: 'user', content: text}];
  addMessage('user', text);
  $('message').value = ''; $('message').oninput();
  const reply = addMessage('assistant');
  reply.row.classList?.add('pending');
  setStatus('Generating…', 'ready');
  let answer = '', completed = false, chunks = 0, usage = null, first = null;
  const started = now();
  try {
    const response = await fetch('/v1/chat/completions', {method: 'POST', signal: controller.signal,
      headers: {'Content-Type': 'application/json', Authorization: `Bearer ${$('key').value}`},
      body: JSON.stringify({messages: pending, conversation_id: conversation, stream: true})});
    if (!response.ok) throw Error((await response.json()).detail || 'Request failed');
    const reader = response.body.getReader(), decoder = new TextDecoder(); let buffered = '';
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buffered += decoder.decode(value, {stream: true});
      let boundary;
      while ((boundary = buffered.indexOf('\n\n')) >= 0) {
        const line = buffered.slice(0, boundary); buffered = buffered.slice(boundary + 2);
        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
        const chunk = JSON.parse(line.slice(6)); if (chunk.error) throw Error(chunk.error.message);
        usage = chunk.usage || usage;
        const part = chunk.choices?.[0]?.delta?.content || '';
        if (part) {
          first ??= now(); chunks += 1; answer += part;
          reply.bubble.textContent = answer; scrollDown();
        }
        if (chunk.choices?.[0]?.finish_reason) completed = true;
      }
    }
    if (!completed) throw Error('Response interrupted');
    const ended = now(), tokens = usage?.completion_tokens || chunks;
    const rate = first != null && tokens > 1 && ended > first ? (tokens - 1) / (ended - first) : null;
    addStats(reply.row, formatStats({ttft: first == null ? null : first - started, tokens, rate}, network));
    messages = [...pending, {role: 'assistant', content: answer}];
    setStatus('Ready', 'ready');
  } catch (error) {
    controller.abort();
    const stopped = error.name === 'AbortError';
    // Not added to the history: the next message starts from the last full answer.
    reply.row.classList?.add('error');
    if (!answer) reply.bubble.textContent = stopped ? 'Stopped.' : error.message;
    addStats(reply.row, stopped ? 'Stopped · not kept in the conversation' : `Error: ${error.message} · not kept in the conversation`);
    setStatus(stopped ? 'Stopped' : error.message, stopped ? '' : 'error');
  } finally {
    reply.row.classList?.remove('pending');
    controller = null; $('send').disabled = !ready; $('stop').disabled = true;
    $('message').focus?.();
  }
};
