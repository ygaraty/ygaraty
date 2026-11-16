import {emitCapture,captureEnabled} from './sdk_capture.mjs';
// Tiles a worker has received, by key, so the server can skip resending them (weights are
// read hundreds of times per update). Least recently used tiles go first; additions and
// evictions are reported with the next request. Entries are copies: a view would keep a
// whole response alive.
export class TileCache {
  constructor(maxBytes){this.maxBytes=maxBytes;this.bytes=0;this.entries=new Map();this.added=new Set();this.removed=new Set();this.announced=false;this.pinned=new Set();}
  trace(action,decoded_bytes=0){if(!captureEnabled())return;emitCapture('worker.cache',{action,decoded_bytes,
    resident_bytes:this.bytes,capacity_bytes:this.maxBytes,pinned:this.pinned.size});}
  get(key){const v=this.entries.get(key);if(v===undefined){this.trace('miss');return null;}this.trace('hit',v.byteLength);this.entries.delete(key);this.entries.set(key,v);return v;}
  // Keys a claim reply says this worker holds: not evicted while its tasks run, or an earlier task's
  // inputs could push out a later task's "cached" tile (a cache miss the coordinator cannot foresee).
  pin(keys){this.pinned=new Set(keys);this.trace('pin_set');}
  put(key,data){
    if(this.entries.has(key)||data.byteLength>this.maxBytes){this.trace(this.entries.has(key)?'skip_existing':'skip_oversize',data.byteLength);return;}
    for(const [old,value] of this.entries){
      if(this.bytes+data.byteLength<=this.maxBytes)break;
      if(this.pinned.has(old))continue;
      this.entries.delete(old);this.bytes-=value.byteLength;this.trace('evict',value.byteLength);
      if(!this.added.delete(old))this.removed.add(old);
    }
    const copy=data instanceof Float64Array?Float64Array.from(data):Float64Array.from(data);this.entries.set(key,copy);this.bytes+=copy.byteLength;this.trace('insert',copy.byteLength);
    if(!this.removed.delete(key))this.added.add(key);
  }
  report(){
    const out={cache:true};
    if(!this.announced){out.cached=[...this.entries.keys()];this.announced=true;}
    else{if(this.added.size)out.cache_added=[...this.added];if(this.removed.size)out.cache_removed=[...this.removed];}
    this.added.clear();this.removed.clear();return out;
  }
}
