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

export function installContributorEnvironment(value, {log = console.log} = {}) {
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
  set('Worker', undefined); set('caches', undefined); set('RTCPeerConnection', undefined);
  set('setInterval', (...args) => {const id=interval(...args); intervals.add(id); return id;});
  set('fetch', (input, options) => {
    const target = typeof input === 'string' ? new URL(input, url.origin).href : input;
    const request = send(target, options);
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
