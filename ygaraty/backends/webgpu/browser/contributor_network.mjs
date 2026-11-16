// Local-only transport display. Never expose ICE addresses, credentials or SDP.
export function routeState(link, { preparing = false, chat = false } = {}) {
  if (preparing) return { mode: 'waiting', label: 'Waiting for model loading' };
  if (link?.open) {
    if (link.local === 'relay' || link.remote === 'relay') return { mode: 'turn', label: 'WebRTC · TURN' };
    if (['host', 'srflx', 'prflx'].includes(link.local) && ['host', 'srflx', 'prflx'].includes(link.remote))
      return { mode: 'direct', label: 'WebRTC · Direct' };
    return { mode: 'unknown', label: 'WebRTC · Checking route' };
  }
  if (link && !link.dropped && !link.rejected && ['new', 'connecting'].includes(link.state))
    return { mode: 'connecting', label: chat ? 'Connecting · coordinator available' : 'Connecting · relay available' };
  return { mode: 'relay', label: chat ? 'Coordinator fallback · no chat link' : 'Coordinator relay fallback' };
}

export function networkRows({ stage, stages, next, workers = [], chat = null, preparing = false, stopped = false }) {
  const label = n => `Stage ${String.fromCharCode(65 + n)}`;
  const row = (key, direction, endpoint, link, isChat = false) => ({ key, direction, endpoint,
    ...(stopped ? { mode: 'stopped', label: 'Disconnected · contribution ended' } : routeState(link, { preparing, chat: isChat })),
    rttMs: link?.open && Number.isFinite(link.rttMs) ? Math.round(link.rttMs) : null });
  const rows = [];
  if (stage === 0) rows.push(row('chat-in', '↓ In', 'Chat prompts', chat, true));
  if (stages > 1) {
    const previous = (stage + stages - 1) % stages;
    rows.push(row('worker-in', '↓ In', `${label(previous)}${stage === 0 ? ' · token feedback' : ' · activations'}`, workers.find(x => x.peer === previous)));
    rows.push(row('worker-out', '↑ Out', `${label(next)}${stage === stages - 1 ? ' · token feedback' : ' · activations'}`, workers.find(x => x.peer === next)));
  }
  if (stage === stages - 1) rows.push(row('chat-out', '↑ Out', 'Chat tokens', chat, true));
  return rows;
}

export function connectivityDisplay(element) {
  if (!element) return () => {};
  const previous = new Map(), nodes = new Map();
  const list = element.querySelector('[data-network-links]');
  return rows => {
    const keys = new Set(rows.map(row => row.key));
    for (const [key, node] of nodes) if (!keys.has(key)) { node.remove(); nodes.delete(key); previous.delete(key); }
    for (const row of rows) {
      let node = nodes.get(row.key);
      if (!node) {
        node = document.createElement('div'); node.className = 'contributor-network-link';
        for (const cls of ['network-endpoint', 'network-route', 'network-change']) {
          const part = document.createElement('span'); part.className = cls; node.append(part);
        }
        list.append(node); nodes.set(row.key, node);
      }
      node.dataset.mode = row.mode;
      node.children[0].textContent = `${row.direction ? row.direction + ' · ' : ''}${row.endpoint}`;
      node.children[1].textContent = row.label + (row.rttMs === null ? '' : ` · ${row.rttMs} ms RTT`);
      const old = previous.get(row.key);
      if (old && old.mode !== row.mode) node.children[2].textContent = `Was ${old.label} · changed ${new Date().toLocaleTimeString()}`;
      previous.set(row.key, row);
    }
  };
}

export function monitorContributorNetwork({ element, stage, stages, next, getWorkers, getChat, onSample }) {
  if (!element) return { ready() {}, stop() {} };
  element.hidden = false;
  let preparing = true, stopped = false, timer, busy = false;
  const display = connectivityDisplay(element);
  const render = (workers = [], chat = null) => display(networkRows({ stage, stages, next, workers, chat, preparing, stopped }));
  async function sample() {
    if (busy || stopped) return;
    busy = true;
    let deadline;
    try {
      const result = await Promise.race([
        Promise.all([getWorkers()?.stats() ?? [], getChat()?.stats() ?? []]),
        new Promise(resolve => { deadline = setTimeout(() => resolve(null), 900); }),
      ]);
      if (!stopped) {
        if (result) {
          render(result[0], result[1].find(x => x.peer === -1));
          if (!preparing) onSample?.({links: result[0], chat: result[1]});
        }
        else for (const node of element.querySelector('[data-network-links]').children) node.children[1].textContent = 'Connection status unavailable';
      }
    } catch {
      if (!stopped) for (const node of element.querySelector('[data-network-links]').children) node.children[1].textContent = 'Connection status unavailable';
    } finally { clearTimeout(deadline); busy = false; }
  }
  render();
  timer = setInterval(sample, 1000);
  return {
    ready() { preparing = false; void sample(); },
    stop() { stopped = true; clearInterval(timer); render(); },
  };
}

// Use live diagnostics, never the retained pre-close state, for chat route badges.
export function chatNetworkRows(report, mode, stages = 2) {
  const endpoints = stages === 1 ? [[0, 'Chat ↔ sole contributor']] : [[0, '↑ Out · First contributor'], [stages - 1, '↓ In · Last contributor']];
  return endpoints.map(([stage, endpoint]) => {
    const link = report?.links?.find(item => item.stage === stage);
    const pair = link?.statsUnavailable ? null : link?.stats?.pairs?.find(item => item.selected && item.state === 'succeeded');
    const open = mode !== 'fallback' && mode !== 'relay' && !link?.closed && link?.dataChannel === 'open' && link?.state?.connection === 'connected';
    const state = mode === 'connecting' && !link ? 'connecting' : link?.state?.connection;
    const route = routeState({open, state, local:pair?.local, remote:pair?.remote});
    if (mode === 'fallback' || mode === 'relay') Object.assign(route, {mode:'relay', label:'Coordinator fallback'});
    return {key:`chat-${stage}`, direction:'', endpoint, ...route,
      rttMs:open && Number.isFinite(pair?.rttMs) ? Math.round(pair.rttMs) : null};
  });
}
