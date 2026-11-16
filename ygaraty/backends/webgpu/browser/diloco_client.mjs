// HTTP loop shared with offline fake-I/O contracts. Never retries a local round.
import {decodeRound, encodeResult} from './diloco_frames.mjs';

export async function runReplica({origin, job, token, worker, createBackend,
  fetchImpl = globalThis.fetch, signal, onProgress = () => {},
  decode = decodeRound}) {
  const url = new URL(origin);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
      url.pathname !== '/' || url.search || url.hash ||
      !/^[a-f0-9]{32}$/.test(job) || !/^[A-Za-z0-9_-]{1,128}$/.test(worker) ||
      typeof token !== 'string' || !token) throw Error('Invalid connection');
  let backend;
  async function post(action, body, type) {
    const response = await fetchImpl(`${url.origin}/api/jobs/${job}/${action}`, {
      method: 'POST', redirect: 'error', credentials: 'omit', signal,
      headers: {'Authorization': `Bearer ${token}`, 'Content-Type': type}, body,
    });
    if (!response.ok) throw Error(`Coordinator HTTP ${response.status}`);
    const reader = response.body.getReader();
    const chunks = []; let size = 0;
    try {
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 64 * 1024 * 1024) throw Error('Response exceeds adapter limit');
        chunks.push(value);
      }
    } catch (error) { await reader.cancel(); throw error; }
    finally { reader.releaseLock(); }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return response.headers.get('Content-Type')?.split(';')[0] === 'application/octet-stream'
      ? {round: decode(bytes.buffer)}
      : {status: JSON.parse(new TextDecoder().decode(bytes))};
  }
  function wait() {
    return new Promise((resolve, reject) => {
      const abort = () => { clearTimeout(timer); reject(signal.reason); };
      const timer = setTimeout(() => {
        signal?.removeEventListener('abort', abort); resolve();
      }, 500);
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) abort();
    });
  }
  try {
    while (true) {
      signal?.throwIfAborted();
      const reply = await post('round', JSON.stringify({worker}), 'application/json');
      if (reply.status) {
        if (reply.status.state === 'completed') { onProgress({state: 'completed'}); return; }
        if (reply.status.state !== 'waiting') throw Error('Coordinator stopped the job');
        onProgress({state: 'waiting'}); await wait(); continue;
      }
      const round = reply.round;
      if (!backend) {
        backend = await createBackend(worker, round.header.recipe, round.weights);
        if (round.header.checkpoint) {
          if (typeof backend.checkpoint !== 'function' || typeof backend.restore !== 'function')
            throw Error('Backend must implement checkpoint and restore');
          const restored = await post('replica-state', JSON.stringify({worker,
            round_id: round.header.round_id, version: round.header.version}), 'application/json');
          if (restored.status.state !== null) await backend.restore(restored.status.state);
        }
      }
      const result = await backend.runRound(round);
      signal?.throwIfAborted();
      if (round.header.checkpoint) {
        const state = await backend.checkpoint();
        const body = JSON.stringify({worker, round_id: round.header.round_id,
          version: round.header.version, state});
        if (new TextEncoder().encode(body).length > 16 * 1024 * 1024) throw Error('Checkpoint too large');
        const staged = await post('replica-state', body, 'application/json');
        if (staged.status?.accepted !== true) throw Error('Checkpoint rejected');
      }
      const ack = await post('pseudo', encodeResult(result,
        round.header.upload_dtype ?? round.header.wire_dtype), 'application/octet-stream');
      if (!ack.status || ack.status.accepted !== true || ack.status.error || ack.status.state === 'failed') throw Error('Result rejected');
      onProgress({state: 'round', version: result.version, losses: result.losses});
    }
  } finally { backend?.dispose(); }
}
