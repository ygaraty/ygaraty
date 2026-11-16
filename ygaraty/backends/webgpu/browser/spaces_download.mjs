import { readSizedResponse } from './contributor_loading.mjs';

// No coordinator ticket, cookies or referrer are sent to Spaces. Refresh once if a
// short-lived URL expires before its GET begins; each retry obtains a new ticket.
export async function downloadPiece(describe, expectedBytes, expectedHash, onProgress = () => {}, fetcher = fetch, {signal, stallMs = 30000} = {}) {
  if (!Number.isFinite(stallMs) || stallMs <= 0) throw Error("Invalid stall timeout");
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
    let timer, stalled = false, received = 0;
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
      if (stalled && attempt === 0) continue;
      if (stalled) throw new Error('Model piece download stalled after retry');
      throw error;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
    }
  }
}
