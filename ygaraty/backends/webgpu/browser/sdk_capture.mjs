// Opt-in, bounded, local metadata capture. No requests, headers, URLs or payloads saved.
// fetch() cannot expose browser-generated CORS OPTIONS: retain a sanitized CDP trace too.
let active=null;
let sequence=0;
const responseIds=new WeakMap();
export function requestId(response){return responseIds.get(response);}
const fields=new Set(['action','hit','resident_bytes','capacity_bytes','decoded_bytes','pinned',
  'method','route','request_bytes','response_bytes','framing_bytes','payload_bytes','status',
  'start_ms','end_ms','failed','blocks','dtype','request']);
export function startCapture(capacity=32768){
  if(!Number.isSafeInteger(capacity)||capacity<1)throw Error('positive capture capacity required');
  active={capacity,rows:[],dropped:0,started_ms:performance.now(),utc_ms:Date.now()};
}
export function captureEnabled(){return active!==null;}
export function emitCapture(event,values={}){
  if(!active)return;
  if(active.rows.length>=active.capacity){active.dropped++;return;}
  const row={event,mono_ms:performance.now()};
  for(const [key,value] of Object.entries(values))if(fields.has(key)&&
    (typeof value==='number'&&Number.isFinite(value)||typeof value==='boolean'||typeof value==='string'))
    row[key]=value;
  active.rows.push(row);
}
export function stopCapture(){
  if(!active)return null;
  const result={schema_version:1,...active,ended_ms:performance.now(),
    warning:'Local metadata; browser preflight and HTTP/TLS bytes require network capture.'};
  active=null;return result;
}
// Wrap only relay fetch headers; callers record body bytes after consumption.
// Disabled mode passes the exact arguments straight through.
export async function relayFetch(url,options,requestBytes=0){
  if(!active)return fetch(url,options);
  const session=active,start_ms=performance.now(),request=++sequence;
  try{
    const response=await fetch(url,options);
    if(active!==session)return response;
    responseIds.set(response,request);
    emitCapture('relay.http_headers',{request,start_ms,end_ms:performance.now(),method:options.method||'GET',
      route:'blocks',request_bytes:requestBytes,status:response.status,failed:false});
    return response;
  }catch(error){
    if(active===session)emitCapture('relay.http_headers',{request,start_ms,end_ms:performance.now(),method:options.method||'GET',
      route:'blocks',request_bytes:requestBytes,failed:true});
    throw error;
  }
}
