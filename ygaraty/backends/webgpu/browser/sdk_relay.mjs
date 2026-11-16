import {relayFetch, emitCapture, requestId} from './sdk_capture.mjs';
import {LITTLE_ENDIAN} from './sdk_frames.mjs';
// Block relay (relay/, experiment 50): blocks by SHA-256, fetched several at a time in one frame
// ({blocks: [{key, size}]} header, bytes after it) and stored with PUT /blocks/{sha}. The same
// client and assembly as storage/relay.py: a block is float64, float32 or bf16, physical shape,
// transposed when the manifest says so; a region of a value is cut from the blocks covering it.
const BLOCK_BYTES={'<f8':8,'<f4':4,'bf16':2};
export class RelayClient {
  // batch: one request per result (PUT /blocks) and per fetch (POST /blocks/get), each at one fixed URL
  // so the browser's CORS preflight is cached; false, or a relay without them, uses a request per block.
  constructor({url,token,block_size=512,dtype='<f8',batch=true}){
    if(!/^https?:\/\//.test(url)||!token)throw Error('Relay needs an http(s) URL and a token');
    if(!(dtype in BLOCK_BYTES))throw Error('Relay wire dtype must be <f8 or <f4');
    Object.assign(this,{url:url.replace(/\/$/,''),token,blockSize:block_size,dtype,batch});
    this.getBytes=0;this.putBytes=0;this.requests=0;
  }
  // A batched call that fails is retried once the old way; if that works, batching stays off (an older relay).
  async batched(fast,slow){
    if(!this.batch)return slow();
    try{return await fast();}
    catch(e){const value=await slow();this.batch=false;return value;}
  }
  headers(extra={}){return {Authorization:`Bearer ${this.token}`,...extra};}
  // Bytes for each key, in order; throws naming a missing key.
  async getMany(shas){
    const out=new Map(),unique=[...new Set(shas)];
    for(let start=0;start<unique.length;start+=256){
      const chunk=unique.slice(start,start+256);
      const requestBody=JSON.stringify({keys:chunk});
      const response=await this.batched(
        ()=>this.ok(relayFetch(`${this.url}/blocks/get`,{method:'POST',headers:this.headers({'Content-Type':'application/json'}),body:requestBody,signal:AbortSignal.timeout(60000)},new TextEncoder().encode(requestBody).byteLength)),
        ()=>relayFetch(`${this.url}/blocks?keys=${chunk.join(',')}`,{headers:this.headers(),signal:AbortSignal.timeout(60000)}));
      this.requests++;
      if(response.status===404){const body=await response.json().catch(()=>({}));throw Error(`relay lacks block ${(body.missing||chunk)[0]}`);}
      if(!response.ok)throw Error(`relay read failed: HTTP ${response.status}`);
      const buffer=await response.arrayBuffer(),size=new DataView(buffer).getUint32(0,true);
      const header=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,size)));let offset=4+size;
      for(const block of header.blocks){out.set(block.key,new Uint8Array(buffer,offset,block.size));offset+=block.size;}
      if(offset!==buffer.byteLength)throw Error('relay frame length mismatch');
      this.getBytes+=buffer.byteLength;
      emitCapture('relay.download_body',{request:requestId(response),response_bytes:buffer.byteLength,framing_bytes:4+size,
        payload_bytes:buffer.byteLength-4-size,blocks:header.blocks.length});
    }
    return shas.map(s=>out.get(s));
  }
  // A batched read's reply: a 404 naming missing blocks is an answer, not a sign of an older relay.
  async ok(pending){const r=await pending;if(r.ok||r.status===404)return r;throw Error(`HTTP ${r.status}`);}
  // Store several blocks in one request: a frame of them (as GET /blocks answers), each entry naming
  // the finite-value check the relay applies. The relay stores none of them if any is bad.
  async putMany(entries){
    const header=new TextEncoder().encode(JSON.stringify({blocks:entries.map(e=>({key:e.sha,size:e.bytes.byteLength,check:e.dtype==='<f4'?'f32':e.dtype==='bf16'?'bf16':'f64'}))}));
    const total=4+header.length+entries.reduce((n,e)=>n+e.bytes.byteLength,0),body=new Uint8Array(total);
    new DataView(body.buffer).setUint32(0,header.length,true);body.set(header,4);
    let at=4+header.length;for(const e of entries){body.set(e.bytes,at);at+=e.bytes.byteLength;}
    const response=await relayFetch(`${this.url}/blocks`,{method:'PUT',headers:this.headers({'Content-Type':'application/octet-stream'}),body,signal:AbortSignal.timeout(60000)},total);
    this.requests++;
    if(!response.ok){const b=await response.json().catch(()=>({}));throw Error(`relay batched write failed: HTTP ${response.status} ${b.error||''}`);}
    this.putBytes+=total;
  }
  // Store one block; the relay checks the hash and, with the flag, that every value is finite.
  async put(sha,bytes,dtype){
    const flag=dtype==='<f4'?'f32':dtype==='bf16'?'bf16':'f64';
    const response=await relayFetch(`${this.url}/blocks/${sha}?${flag}=1`,{method:'PUT',headers:this.headers({'Content-Type':'application/octet-stream'}),body:bytes,signal:AbortSignal.timeout(60000)},bytes.byteLength);
    this.requests++;
    if(!response.ok){const body=await response.json().catch(()=>({}));throw Error(`relay write failed: HTTP ${response.status} ${body.error||''}`);}
    this.putBytes+=bytes.byteLength;
  }
  // A result as relay blocks: full-width bands of `rows` rows each (the task's upload_rows: what its
  // readers want), or blockSize x blockSize tiles. Returns the manifest the coordinator takes.
  async upload(value,rows=null){
    const [n,m]=value.shape,regions=[];
    if(rows){let at=0;for(const r of rows){regions.push([at,at+r,0,m]);at+=r;}if(at!==n)throw Error('upload_rows do not cover the result');}
    else for(let i=0;i<n;i+=this.blockSize)for(let j=0;j<m;j+=this.blockSize)regions.push([i,Math.min(i+this.blockSize,n),j,Math.min(j+this.blockSize,m)]);
    const blocks=await Promise.all(regions.map(async region=>{
      const bytes=encodeBlock(value.data,m,region,this.dtype);
      const sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(b=>b.toString(16).padStart(2,'0')).join('');
      const [i,ei,j,ej]=region;
      emitCapture('relay.upload_block',{decoded_bytes:(ei-i)*(ej-j)*8,payload_bytes:bytes.byteLength,dtype:this.dtype});
      return {sha,bytes,region};
    }));
    await this.batched(()=>this.putMany(blocks.map(b=>({...b,dtype:this.dtype}))),
                       ()=>Promise.all(blocks.map(b=>this.put(b.sha,b.bytes,this.dtype))));
    return blocks.map(({sha,region:[i,ei,j,ej],region})=>({sha,shape:[ei-i,ej-j],region,...(this.dtype!=='<f8'?{dtype:this.dtype}:{})}));
  }
}
// The bytes of region [i, ei, j, ej] of a row-major (rows x width) value in a block dtype, little-endian.
export function encodeBlock(data,width,region,dtype){
  const [i,ei,j,ej]=region,n=(ei-i)*(ej-j),out=dtype==='<f4'?new Float32Array(n):new Float64Array(n);
  if(dtype==='bf16')throw Error('bf16 results are not produced in the browser');
  let k=0;for(let r=i;r<ei;r++)for(let c=j;c<ej;c++)out[k++]=data[r*width+c];
  if(dtype==='<f4')for(let q=0;q<n;q++)if(!Number.isFinite(out[q]))throw Error('value overflows the float32 wire');
  if(!LITTLE_ENDIAN)throw Error('big-endian hosts are not supported for relay blocks');
  return new Uint8Array(out.buffer);
}
// A block's float64 matrix {rows, cols, data} from its bytes: physical shape, then transposed if the manifest says so.
export function decodeBlock(raw,block){
  const dtype=block.dtype||'<f8',[r,c]=block.transposed?[block.shape[1],block.shape[0]]:block.shape,n=r*c;
  if(raw.byteLength!==n*BLOCK_BYTES[dtype])throw Error('relay block size mismatch');
  const aligned=raw.byteOffset%8===0?raw:new Uint8Array(raw),view=new DataView(aligned.buffer,aligned.byteOffset,aligned.byteLength);
  let data;
  if(dtype==='<f8')data=LITTLE_ENDIAN?new Float64Array(aligned.buffer,aligned.byteOffset,n):Float64Array.from({length:n},(_,k)=>view.getFloat64(8*k,true));
  else if(dtype==='<f4')data=LITTLE_ENDIAN?Float64Array.from(new Float32Array(aligned.buffer,aligned.byteOffset,n)):Float64Array.from({length:n},(_,k)=>view.getFloat32(4*k,true));
  else{data=new Float64Array(n);const f=new Float32Array(1),u=new Uint32Array(f.buffer);for(let k=0;k<n;k++){u[0]=view.getUint16(2*k,true)<<16;data[k]=f[0];}}
  if(!block.transposed)return {rows:r,cols:c,data};
  const t=new Float64Array(n);for(let a=0;a<r;a++)for(let b=0;b<c;b++)t[b*r+a]=data[a*c+b];
  return {rows:c,cols:r,data:t};
}
// Region [i, ei, j, ej] of a value from the blocks covering it (regions in the value's coordinates).
export function assembleRegion(region,blocks,raws,decoded=new Map()){
  const full=(b,raw)=>{const key=`${b.sha}|${b.shape}|${!!b.transposed}|${b.dtype||''}`;if(!decoded.has(key))decoded.set(key,decodeBlock(raw,b));return decoded.get(key);};
  const [i,ei,j,ej]=region,out=new Float64Array((ei-i)*(ej-j)),width=ej-j;
  for(let k=0;k<blocks.length;k++){
    const b=blocks[k],[bi,bei,bj,bej]=b.region,si=Math.max(i,bi),se=Math.min(ei,bei),sj=Math.max(j,bj),sje=Math.min(ej,bej);
    if(si>=se||sj>=sje)continue;
    const m=full(b,raws[k]);
    if(m.rows!==bei-bi||m.cols!==bej-bj)throw Error('relay block does not match its region');
    for(let r=si;r<se;r++)for(let c=sj;c<sje;c++)out[(r-i)*width+(c-j)]=m.data[(r-bi)*m.cols+(c-bj)];
  }
  return out;
}
