// Formatting shared by the session page and the chat page's status bar.
// Input: the server's /v1/status object (also `details` in /v1/session).

const number = value => (value == null ? '–' : Number(value).toLocaleString('en-US'));
const seconds = value => (value == null ? '–' : `${value < 10 ? value.toFixed(2) : value.toFixed(1)} s`);
const rate = value => (value == null ? '–' : `${value.toFixed(1)} tok/s`);

export function duration(total) {
  if (total == null) return '–';
  const s = Math.floor(total), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : m ? `${m} min ${s % 60} s` : `${s} s`;
}

export function stateText(status) {
  if (!status) return ['Server unavailable', ''];
  if (status.state === 'ready') return ['Ready', 'ready'];
  if (status.state === 'closed') return ['Closed', ''];
  const need = status.workers_required, ready = status.workers_ready ?? 0;
  return [need ? `Waiting for workers (${ready} of ${need} ready)` : 'Waiting for workers', 'waiting'];
}

export function connectivity(status) {
  if (!status) return '–';
  if (status.network) return status.network;
  return {relay: 'relay', direct: 'WebRTC', private_turn: 'WebRTC (TURN only)'}[status.route] || status.route || '–';
}

// What one worker is doing, in words: "ready", "downloading 45%", ...
export function workerState(worker) {
  const text = {
    ready: 'ready', waiting: 'waiting for someone to open the link', joining: 'joined, preparing',
    checking: 'checking the GPU', downloading: `downloading ${Math.round(worker.percent ?? 0)}%`,
  }[worker.state] || worker.state || '?';
  return text;
}

// "connected", "late (heartbeat 41 s ago)" or "–" for an empty slot.
export function workerCoordinator(worker) {
  if (!worker.coordinator) return '–';
  if (worker.coordinator === 'late') return `late (last heard ${Math.round(worker.heartbeat_age_s ?? 0)} s ago)`;
  return 'connected';
}

// "B: direct udp, 0.4 ms" per link, or the route when there are none.
export function workerLinks(worker) {
  if (!worker.links) return worker.coordinator ? 'none (one worker)' : '–';
  if (!worker.links.length) return worker.route || 'relay';
  return worker.links.map(link => {
    const parts = [link.path || link.state];
    if (link.protocol) parts.push(link.protocol);
    if (link.rtt_ms != null) parts.push(`${link.rtt_ms} ms`);
    if (link.path && !['connected', 'open'].includes(link.state)) parts.push(link.state);
    return `${link.peer ?? '?'}: ${parts.join(' ')}`;
  }).join('; ');
}

// Compact form for the bar: "A ready, B 45%".
export function workersText(workers) {
  return (workers || []).map(worker => {
    let text = worker.state === 'downloading' ? `${Math.round(worker.percent ?? 0)}%` : worker.state;
    if (worker.coordinator === 'late') text += ' (late)';
    return `${worker.stage} ${text}`;
  }).join(', ');
}

// [value, label] pairs for the session page's tiles.
export function statTiles(status) {
  const s = status?.stats || {};
  return [
    [number(s.total_tokens), 'tokens so far'],
    [rate(s.tokens_per_s?.average), 'average decode speed'],
    [seconds(s.ttft_s?.average), 'average time to first token'],
    [`${number(s.active_requests)} / ${number(status?.slots)}`, 'chats now / at a time'],
    [number(s.throughput_tokens_per_s), 'tokens/s, last minute'],
    [number(s.requests?.completed), 'answers'],
    [number(s.peak_active_requests), 'most chats at once'],
    [duration(s.uptime_s), 'uptime'],
  ];
}

// One line for the chat page's top bar.
export function barText(status) {
  const s = status?.stats || {};
  const parts = [
    `${number(s.total_tokens)} tokens`,
    `avg ${rate(s.tokens_per_s?.average)}`,
    `TTFT ${seconds(s.ttft_s?.average)}`,
    `context ${number(status?.context_tokens)}`,
    `${number(s.active_requests)}/${number(status?.slots)} chats`,
  ];
  const workers = workersText(status?.workers);
  if (workers) parts.push(`workers ${workers}`);
  parts.push(connectivity(status));
  return parts.join(' · ');
}

export function facts(status) {
  if (!status) return [];
  const out = [];
  if (status.workers_required) out.push(`${status.workers_ready ?? 0} of ${status.workers_required} worker${status.workers_required === 1 ? '' : 's'} ready`);
  if (status.context_tokens) out.push(`${number(status.context_tokens)}-token context`);
  if (status.max_output_tokens) out.push(`up to ${number(status.max_output_tokens)} output tokens`);
  if (status.slots) out.push(`${status.slots} chat${status.slots === 1 ? '' : 's'} at a time`);
  out.push(`network: ${connectivity(status)}`);
  return out;
}
