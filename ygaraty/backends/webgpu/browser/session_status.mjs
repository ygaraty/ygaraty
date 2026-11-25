// Session page (/): model details, workers and live statistics, read with
// the owner key (/#owner=KEY) or the contributor key (/#key=KEY), or without a
// key when the server runs with an open status page. The owner also sees the
// contributor link to share.
import {facts, stateText, statTiles, workerCoordinator, workerLinks, workerState} from './session_stats_view.mjs';

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.hash.slice(1));
const ownerKey = params.get('owner');
const contributorKey = params.get('key');
if (ownerKey) {
  // Keep the owner key out of the address bar and history.
  params.delete('owner');
  const rest = params.toString();
  history.replaceState(null, '', location.pathname + location.search + (rest ? `#${rest}` : ''));
}
const statusKey = ownerKey || contributorKey;
const open = $('session-card').dataset.openStatus === 'true';
if (!contributorKey && !ownerKey) $('join-hint').hidden = false;

function cell(text, kind = '') {
  const td = document.createElement('td');
  td.textContent = text;
  if (kind) td.className = kind;
  return td;
}

function renderWorkers(workers) {
  $('workers-card').hidden = !workers?.length;
  $('workers').querySelector('tbody').replaceChildren(...(workers || []).map(worker => {
    const row = document.createElement('tr');
    row.append(
      cell(worker.stage),
      cell(workerState(worker), worker.state === 'ready' ? 'ok' : worker.state === 'waiting' ? 'warn' : ''),
      cell(workerCoordinator(worker), worker.coordinator === 'late' ? 'warn' : ''),
      cell(workerLinks(worker)),
    );
    return row;
  }));
}

function render(status) {
  const [text, kind] = stateText(status);
  $('session-state').textContent = text;
  $('session-state').className = kind;
  if (!status) return;
  $('session-model').textContent = status.model || '';
  document.title = `Ygaraty · ${status.model || 'Session'}`;
  $('session-facts').replaceChildren(...facts(status).map(value => {
    const item = document.createElement('span'); item.textContent = value; return item;
  }));
  $('session-stats').replaceChildren(...statTiles(status).map(([value, label]) => {
    const tile = document.createElement('div'); tile.className = 'stat';
    const strong = document.createElement('b'); strong.textContent = value;
    const caption = document.createElement('span'); caption.textContent = label;
    tile.append(strong, caption); return tile;
  }));
  renderWorkers(status.workers);
  const last = status.stats?.last;
  $('session-network').textContent = last
    ? `Last answer: ${last.generated_tokens} tokens in ${last.seconds} s` + (last.ttft_s != null ? `, first token after ${last.ttft_s} s` : '') + (last.tokens_per_s ? `, ${last.tokens_per_s} tok/s.` : '.')
    : 'No answers yet.';
}

function locked() {
  $('session-locked').hidden = false;
  $('session-state').textContent = 'Key needed';
  $('session-state').className = '';
}

let timer;
async function refresh() {
  clearTimeout(timer);
  try {
    const headers = statusKey ? {Authorization: `Bearer ${statusKey}`} : {};
    const response = await fetch('/v1/status', {headers, signal: AbortSignal.timeout(5000)});
    if (response.status === 401) { locked(); return; }  // no key, or not this session's
    render(response.ok ? await response.json() : null);
  } catch { render(null); }
  timer = setTimeout(refresh, 2000);
}
if (statusKey || open) refresh(); else locked();

async function showOwner() {
  if (!ownerKey) return;
  try {
    const response = await fetch('/v1/session/contributor-key', {headers: {Authorization: `Bearer ${ownerKey}`}, signal: AbortSignal.timeout(5000)});
    if (!response.ok) return;
    const {key} = await response.json();
    $('contributor-link').value = `${location.origin}/#${new URLSearchParams({key})}`;
    $('chat-link').href = `/chat#${new URLSearchParams({key: ownerKey})}`;
    $('owner').hidden = false;
  } catch { /* not the owner, or the server is away: nothing to show */ }
}
showOwner();

$('copy-contributor').onclick = async () => {
  const input = $('contributor-link');
  try { await navigator.clipboard.writeText(input.value); }
  catch { input.select(); document.execCommand?.('copy'); }
  $('copy-contributor').textContent = 'Copied';
  setTimeout(() => { $('copy-contributor').textContent = 'Copy'; }, 1500);
};
