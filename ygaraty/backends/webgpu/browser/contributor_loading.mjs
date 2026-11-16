// Exact response-body progress, independently testable without WebGPU.
export async function readSizedResponse(response, expected, onProgress = () => {}, signal) {
  if(!Number.isSafeInteger(expected) || expected < 0) throw Error("Invalid server byte count");
  const declared=response.headers.get("Content-Length");
  if(declared !== null && Number(declared) !== expected) throw Error(`unexpected Content-Length`);
  const output=new Uint8Array(expected), reader=response.body.getReader();
  const abort = () => { reader.cancel(signal.reason).catch(() => {}); };
  signal?.addEventListener("abort", abort, {once:true});
  let offset=0;
  try {
    signal?.throwIfAborted();
    for(;;){const {value,done}=await reader.read();signal?.throwIfAborted();if(done)break;
      if(offset+value.byteLength>expected)throw Error(`download exceeds declared size`);
      output.set(value,offset);offset+=value.byteLength;onProgress(offset);
    }
    if(offset!==expected)throw Error(`truncated download (${offset}/${expected})`);
    return output.buffer;
  }catch(error){await reader.cancel();throw error;}
  finally { signal?.removeEventListener("abort", abort); reader.releaseLock(); }
}

// A single stage can reuse its embedding buffers for the output head. Those
// separately named head files are assigned for split stages but never loaded here.
export function requiredWeightSizes(manifest, sizes) {
  if (!manifest?.head?.shared_embedding || !manifest?.embed?.chunks?.length) return {...sizes};
  const used = new Set();
  function visit(value) {
    if (!value || typeof value !== 'object') return;
    if (typeof value.file === 'string') used.add(value.file);
    for (const [key, child] of Object.entries(value)) if (key !== 'plan') visit(child);
  }
  visit({...manifest, head:{...manifest.head, chunks:[]}});
  const skipped = new Set((manifest.head.chunks || []).map(chunk => chunk.file));
  return Object.fromEntries(Object.entries(sizes).filter(([name]) => !skipped.has(name) || used.has(name)));
}

export function weightProgress(sizes, changed) {
  const complete = new Set(), pending = new Map();
  const total = Object.values(sizes).reduce((sum, bytes) => {
    if (!Number.isSafeInteger(bytes) || bytes < 0) throw Error('Invalid weight size');
    return sum + bytes;
  }, 0);
  const emit = () => changed({total,
    loaded:[...complete].reduce((sum,name)=>sum+sizes[name],0)+[...pending.values()].reduce((sum,bytes)=>sum+bytes,0),
    complete:complete.size === Object.keys(sizes).length});
  emit();
  return {
    update(name, bytes, verified=false) {
      if (!Object.hasOwn(sizes,name) || !Number.isSafeInteger(bytes) || bytes < 0 || bytes > sizes[name]) throw Error('Invalid weight progress');
      if (verified) {
        if (bytes !== sizes[name]) throw Error('Incomplete verified weight');
        complete.add(name); pending.delete(name);
      } else if (!complete.has(name)) pending.set(name,bytes);
      emit();
    },
    finish() {
      if (complete.size !== Object.keys(sizes).length) throw Error('Model loading finished with missing weight files');
      emit();
    },
  };
}

// Three speculative pieces plus one demand lane. Completed buffers keep their
// reservations until consumed, so a slow first piece cannot accumulate a model
// in RAM. Oversized/out-of-order demand never waits for speculative capacity.
// Caller consumes/uploads each returned buffer before asking for the next one.
export function prefetchWeights(sizes, load, {maxBytes = 256 * 1024 * 1024, concurrency = 3} = {}) {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0 || !Number.isInteger(concurrency) || concurrency < 1)
    throw Error('Invalid prefetch limits');
  const names = Object.keys(sizes), entries = new Map(), consumed = new Set();
  for (const size of Object.values(sizes))
    if (!Number.isSafeInteger(size) || size < 0) throw Error('Invalid weight size');
  const controller = new AbortController();
  let reserved = 0, count = 0, previous = null, closed = false;
  function start(name, speculative) {
    if (speculative) { reserved += sizes[name]; count++; }
    const entry = {speculative};
    // Store failures as values until consumed; no unhandled background rejections.
    entry.promise = Promise.resolve().then(() => {
      controller.signal.throwIfAborted();
      return load(name, controller.signal);
    }).then(data => ({data}), error => ({error}));
    entries.set(name, entry);
    return entry;
  }
  function fill() {
    for (const name of names) {
      if (count >= concurrency) break;
      if (consumed.has(name) || entries.has(name) || sizes[name] > maxBytes - reserved) continue;
      start(name, true);
    }
  }
  function release() {
    if (previous === null) return;
    const entry = entries.get(previous);
    if (entry?.speculative) { reserved -= sizes[previous]; count--; }
    entries.delete(previous); previous = null;
  }
  return {
    async get(name) {
      if (closed) throw Error('Weight prefetch closed');
      if (!Object.hasOwn(sizes, name)) throw Error('Weight request is not in the assigned manifest');
      // Linear-attention stages reread several tensors for derived buffers.
      // Reload on demand (normally from verified disk cache), rather than retain
      // every consumed weight in RAM or reject legitimate repeated requests.
      release();
      const entry = entries.get(name) || start(name, false);
      consumed.add(name); previous = name;
      fill();
      const result = await entry.promise;
      controller.signal.throwIfAborted();
      if (result.error) throw result.error;
      return result.data;
    },
    close() {
      closed = true; controller.abort(new Error('Weight loading finished'));
      entries.clear();
    },
  };
}
