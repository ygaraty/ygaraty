// Small browser host for the shared contributor, without downloading remote code.
export function contributorURL(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password)
    throw Error('Expected an HTTP(S) contributor link without user information');
  const params = new URLSearchParams(url.hash.slice(1));
  if (!params.get('key')) throw Error('Expected the contributor link with #key=...');
  if (params.has('pool')) throw Error('Use a session contributor link, not a pool link');
  params.set('autostart', '1'); url.hash = params.toString();
  return url;
}

// node-datachannel passes TURN credentials to libdatachannel inside a URL (turn:USER:PASS@host), which it
// URL-decodes; TURN REST usernames (expiry:session:slot) contain colons, so they must be percent-encoded (DO-97).
export function nativePeerConnection(Base) {
  const encode = (s) => s && s.username != null ? {...s, username: encodeURIComponent(s.username), credential: encodeURIComponent(s.credential)} : s;
  return class extends Base {
    constructor(config = {}, ...rest) { super({...config, iceServers: (config.iceServers || []).map(encode)}, ...rest); }
    // libdatachannel's round trip is in milliseconds; WebRTC stats give seconds (DO-100).
    async getStats(...args) {
      const report = await super.getStats(...args);
      report.forEach((s) => {
        if (s.type !== 'candidate-pair') return;
        for (const k of ['currentRoundTripTime', 'totalRoundTripTime']) if (typeof s[k] === 'number') s[k] /= 1000;
      });
      return report;
    }
  };
}

// webrtc: the RTCPeerConnection polyfill module (node-datachannel's), for direct links (DO-97); without it the
// contributor uses the coordinator relay.
// DO-105: Node's built-in fetch (undici) moved large Spaces downloads at about 1 MB/s from a host where core https
// moved 50-66 MB/s (the same object, IPv4): its receive path keeps the TCP window near 128 KB, so throughput is that
// over the round trip. Cross-origin HTTPS GETs (the presigned piece downloads) go through core https instead,
// returned as a standard Response around its stream; redirects are refused like `redirect: 'error'`, and an abort
// signal destroys the request.
export async function httpsGet(target, options = {}) {
  const [{default: https}, {Readable}] = await Promise.all([import('node:https'), import('node:stream')]);
  const signal = options.signal;
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const headers = Object.fromEntries(new Headers(options.headers || {}).entries());
    const request = https.get(target, {headers}, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && options.redirect === 'error') {
        res.resume(); reject(new TypeError('fetch failed: redirect')); return;
      }
      const out = new Headers();
      for (const [k, v] of Object.entries(res.headers)) out.set(k, Array.isArray(v) ? v.join(', ') : String(v));
      const body = [204, 304].includes(res.statusCode) ? null : Readable.toWeb(res);
      resolve(new Response(body, {status: res.statusCode, statusText: res.statusMessage, headers: out}));
    });
    request.on('error', reject);
    signal?.addEventListener('abort', () => { request.destroy(signal.reason); reject(signal.reason); }, {once: true});
  });
}

export function installContributorEnvironment(value, {log = console.log, webrtc = null} = {}) {
  const url = contributorURL(value), events = new EventTarget(), pending = new Set();
  const send = globalThis.fetch.bind(globalThis);
  const intervals = new Set(), interval = globalThis.setInterval;
  const originals = new Map();
  function set(name, val) {
    originals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, {value:val, configurable:true, writable:true});
  }
  const storage = new Map();
  const status = {set textContent(text) {log(String(text));}};
  let reload = false;
  set('location', {origin:url.origin, pathname:url.pathname, search:url.search,
    hash:url.hash, reload() {reload=true;}});
  set('document', {getElementById(id) {return ['status','detail'].includes(id) ? status : null;}});
  set('window', {isSecureContext:true, addEventListener:events.addEventListener.bind(events),
    removeEventListener:events.removeEventListener.bind(events)});
  set('sessionStorage', {getItem:k=>storage.get(k) ?? null, setItem:(k,v)=>storage.set(k,String(v))});
  // Background throttling is browser-only; use the contributor's timer fallback.
  set('Worker', undefined); set('caches', undefined);
  set('RTCPeerConnection', webrtc ? nativePeerConnection(webrtc.RTCPeerConnection) : undefined);
  if (webrtc) {set('RTCIceCandidate', webrtc.RTCIceCandidate); set('RTCSessionDescription', webrtc.RTCSessionDescription);}
  set('setInterval', (...args) => {const id=interval(...args); intervals.add(id); return id;});
  set('fetch', (input, options) => {
    const target = typeof input === 'string' ? new URL(input, url.origin).href : input;
    const parsed = typeof target === 'string' ? new URL(target) : null;
    const direct = !globalThis.Deno && parsed && parsed.protocol === 'https:' && parsed.origin !== url.origin
      && (!options?.method || options.method === 'GET') && !options?.body;
    const request = direct ? httpsGet(target, options) : send(target, options);
    pending.add(request); request.then(()=>pending.delete(request),()=>pending.delete(request));
    return request;
  });
  return {
    get reloadRequested() {return reload;},
    async close() {
      events.dispatchEvent(new Event('pagehide'));
      for (const timer of intervals) clearInterval(timer);
      let deadline;
      try {
        await Promise.race([Promise.allSettled([...pending]),
          new Promise(resolve=>{deadline=setTimeout(resolve,1500);})]);
      } finally {clearTimeout(deadline);}
      for (const [name, desc] of originals) {
        if (desc) Object.defineProperty(globalThis,name,desc); else delete globalThis[name];
      }
    },
  };
}
