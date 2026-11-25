import { readSizedResponse } from './contributor_loading.mjs';

// No coordinator ticket, cookies or referrer are sent to Spaces. Refresh once if a
// short-lived URL expires before its GET begins; each retry obtains a new ticket.
// DO-103: `minRate` (bytes/s; off when 0): after `graceMs`, a request whose bytes over the last `rateWindowMs` fall
// below the floor is abandoned and the piece asked for again on a new connection (a fresh descriptor), at most
// `maxSlow` times; after that the slow request is kept (a floor never fails a load by itself).
export async function downloadPiece(describe, expectedBytes, expectedHash, onProgress = () => {}, fetcher = fetch,
                                    {signal, stallMs = 30000, minRate = 0, rateWindowMs = 10000, graceMs = 5000, maxSlow = 3, onSlow = () => {}} = {}) {
  if (!Number.isFinite(stallMs) || stallMs <= 0) throw Error("Invalid stall timeout");
  if (!Number.isFinite(minRate) || minRate < 0 || !(rateWindowMs > 0) || !(graceMs >= 0) || !Number.isInteger(maxSlow) || maxSlow < 0)
    throw Error("Invalid rate floor");
  let slowRetries = 0;
  for (let attempt = 0; attempt < 2; attempt++) {
    const descriptor = await describe();
    if (descriptor.bytes !== expectedBytes || descriptor.sha256 !== expectedHash)
      throw new Error('Download descriptor does not match assigned piece');
    const url = new URL(descriptor.url);
    if (url.protocol !== 'https:' || url.username || url.password || url.hash)
      throw new Error('Invalid Spaces download URL');
    signal?.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal?.addEventListener('abort', abort, {once:true});
    let timer, stalled = false, slow = false, received = 0, rateTimer;
    const floorOn = minRate > 0 && slowRetries < maxSlow;
    const samples = [[performance.now(), 0]];
    if (floorOn) rateTimer = setInterval(() => {
      const now = performance.now();
      samples.push([now, received]);
      while (samples.length > 1 && samples[1][0] <= now - rateWindowMs) samples.shift();
      const [t0, b0] = samples[0];
      if (now - samples.started >= graceMs + rateWindowMs && now - t0 >= rateWindowMs * 0.9 &&
          (received - b0) * 1000 / (now - t0) < minRate && received < expectedBytes) {
        slow = true;
        controller.abort(new Error('Model piece download below the rate floor'));
      }
    }, Math.max(50, Math.min(1000, rateWindowMs / 5)));
    samples.started = performance.now();
    const touch = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        stalled = true;
        controller.abort(new Error('Model piece download stalled'));
      }, stallMs);
    };
    try {
      onProgress(0);
      touch();
      const response = await fetcher(url.href, {
        method: 'GET', credentials: 'omit', referrerPolicy: 'no-referrer',
        mode: 'cors', cache: 'no-store', redirect: 'error', signal: controller.signal,
      });
      if (response.status === 403 && attempt === 0) { await response.body?.cancel(); continue; }
      if (!response.ok) {
        await response.body?.cancel();
        throw new Error(`Model piece download failed (${response.status}); check Spaces access and CORS`);
      }
      return await readSizedResponse(response, expectedBytes, bytes => {
        if (bytes > received) { received = bytes; touch(); }
        onProgress(bytes);
      }, controller.signal);
    } catch (error) {
      signal?.throwIfAborted();
      if (slow) { slowRetries++; onSlow(slowRetries, received); attempt--; continue; }  // not counted as a failure
      if (stalled && attempt === 0) continue;
      if (stalled) throw new Error('Model piece download stalled after retry');
      throw error;
    } finally {
      clearTimeout(timer);
      clearInterval(rateTimer);
      signal?.removeEventListener('abort', abort);
    }
  }
}
