import {Peers} from './qwen35_peer.mjs';

// Separate channels from the inference ring. Chat is stage -1 and always offers.
export class ChatEdges extends Peers {
  constructor({stage, stages, lease, api, ice, policy, capacity, onMessage, onDiagnostics}) {
    super({id:'chat',stage,stages,ice,policy,onMessage,openWithinMs:30000,
      call:(_path, body)=>api({action:'signal',lease,...body})});
    this.relayPeers=new Set();this.directAttempts=new Map();this.phaseCandidates=new Map();
    this.onDiagnostics=onDiagnostics;this.diagnosticsStopped=false;
    this.neighbours=stage===-1?[...new Set([0,stages-1])]:[-1];
    this.capacity=Number.isSafeInteger(capacity)&&capacity>0?Math.min(capacity,262144):0;
  }
  async iceServers(peer) {
    const servers=await super.iceServers();
    if(this.policy==='relay'||this.relayPeers.has(peer))return servers;
    // Both endpoints try host/STUN paths first. No TURN allocation can win this attempt.
    return (servers||[]).flatMap(server=>{
      const urls=(Array.isArray(server.urls)?server.urls:[server.urls]).filter(url=>/^stuns?:/i.test(url||''));
      return urls.length?[{urls}]:[];
    });
  }
  async apply(item) {
    const relay=item.relay_fallback===true;
    if(this.stage!==-1&&relay&&!this.relayPeers.has(item.sender)&&item.data.candidate){
      const pending=this.phaseCandidates.get(item.sender)||[];
      if(pending.length<64)pending.push(item);
      this.phaseCandidates.set(item.sender,pending);
      return;
    }
    if(this.stage!==-1&&item.data.sdp?.type==='offer'){
      const changed=this.relayPeers.has(item.sender)!==relay;
      if(relay)this.relayPeers.add(item.sender);else this.relayPeers.delete(item.sender);
      // Replace even a not-yet-answered direct attempt with the offerer's TURN retry.
      if(changed)this.unlink(item.sender);
    }else if(relay!==this.relayPeers.has(item.sender))return; // late answer/candidate from the previous attempt
    await super.apply(item);
    if(this.stage!==-1&&item.data.sdp?.type==='offer'&&relay){
      const pending=this.phaseCandidates.get(item.sender)||[];this.phaseCandidates.delete(item.sender);
      for(const candidate of pending)await super.apply(candidate);
    }
  }
  async advanceFallback() {
    if(this.stage!==-1||this.policy==='relay'||this.diagnosticsStopped)return;
    for(const peer of this.neighbours)await this.exclusive(peer,async()=>{
      const link=this.links.get(peer);
      if(!link||this.has(peer)||this.relayPeers.has(peer)||link.rejected)return;
      if(!link.dead&&performance.now()-link.made<8000)return;
      this.directAttempts.set(peer,{state:link.failureState||{ice:link.pc.iceConnectionState,connection:link.pc.connectionState},
        remoteDescriptionReceived:link.remoteSet,localCandidates:link.localCandidates,remoteCandidates:link.remoteCandidates,receivedCandidateTypes:link.receivedTypes,remoteMdnsCandidates:link.remoteMdnsCandidates,stats:link.statsSummary});
      this.relayPeers.add(peer);
      // Keep the original 30-second total connection deadline, including the direct attempt.
      const next=await this.link(peer,true);
      if(!this.diagnosticsStopped)await this.offer(next);
      else this.unlink(peer);
      this.publishDiagnostics();
    });
  }
  async link(peer,initial=false) {
    const link=await super.link(peer,initial);
    link.localCandidates=0;link.remoteCandidates=0;link.candidateErrors=0;
    link.iceErrors=[];link.statsSummary=null;link.receivedTypes={};link.remoteMdnsCandidates=0;
    link.pc.onicecandidateerror=event=>{
      // Deliberately omit errorText, host names, URLs, addresses and credentials.
      const code=Number.isInteger(event.errorCode)?event.errorCode:null;
      const scheme=/^(stun|stuns|turn|turns):/.exec(event.url||'')?.[1]||'unknown';
      if(link.iceErrors.length<16)link.iceErrors.push({code,server:scheme});
      this.publishDiagnostics();
    };
    const add=link.pc.addIceCandidate.bind(link.pc);
    link.pc.addIceCandidate=async candidate=>{
      if(candidate?.candidate){
        link.remoteCandidates++;
        const parts=candidate.candidate.split(/\s+/),kind=parts[parts.indexOf('typ')+1];
        if(['host','srflx','prflx','relay'].includes(kind))link.receivedTypes[kind]=(link.receivedTypes[kind]||0)+1;
        if(kind==='host'&&parts[4]?.toLowerCase().endsWith('.local'))link.remoteMdnsCandidates++;
      }
      try{return await add(candidate);}catch(error){link.candidateErrors++;throw error;}
    };
    return link;
  }
  unlink(peer) {
    const link=this.links.get(peer);
    if(link&&!link.dead){
      link.channelBeforeClose=link.channel?.readyState||'not open';
      void this.sampleDiagnostics(link);
      link.failureState={
      ice:link.pc.iceConnectionState,connection:link.pc.connectionState,
      gathering:link.pc.iceGatheringState,signaling:link.pc.signalingState,
    };
    }
    super.unlink(peer);
    this.publishDiagnostics();
  }
  signal(to,data) {
    if(data.candidate?.candidate){const link=this.links.get(to);if(link)link.localCandidates++;}
    // Keep the base class's ordered, non-throwing signaling contract, but retain a safe diagnostic.
    const relay_fallback=this.relayPeers.has(to);
    const sent=this.signalChain.then(()=>this.call(`signal/${this.id}`,{to,data,relay_fallback})).catch(()=>{
      const link=this.links.get(to);
      if(link)link.signalFailed=true;
      return null;
    });
    this.signalChain=sent;
    return sent;
  }
  async sampleDiagnostics(link) {
    if(!this.onDiagnostics||link.statsPending)return;
    link.statsPending=true;
    try{
      const stats=await link.pc.getStats();
      const type=c=>['host','srflx','prflx','relay'].includes(c?.candidateType)?c.candidateType:'unknown';
      const protocol=c=>['udp','tcp','tls'].includes(c?.protocol)?c.protocol:'unknown';
      const summary={local:{},remote:{},pairs:[],dataChannels:[],dtls:'unknown',sampledAt:new Date().toISOString()};
      let selected;
      for(const r of stats.values())if(r.type==='transport'){
        selected=r.selectedCandidatePairId||selected;
        if(['new','connecting','connected','closed','failed'].includes(r.dtlsState))summary.dtls=r.dtlsState;
      }
      for(const r of stats.values()){
        if(r.type==='local-candidate'||r.type==='remote-candidate'){
          const bucket=r.type==='local-candidate'?summary.local:summary.remote;
          const label=type(r);bucket[label]=(bucket[label]||0)+1;
        }
        if(r.type==='candidate-pair'&&summary.pairs.length<32){
          const local=stats.get(r.localCandidateId),remote=stats.get(r.remoteCandidateId);
          const num=v=>typeof v==='number'&&Number.isFinite(v)&&v>=0?v:null;
          summary.pairs.push({state:['frozen','waiting','in-progress','failed','succeeded'].includes(r.state)?r.state:'unknown',
            selected:r.id===selected,nominated:r.nominated===true,local:type(local),remote:type(remote),protocol:protocol(local),
            requestsSent:num(r.requestsSent),responsesReceived:num(r.responsesReceived),
            bytesSent:num(r.bytesSent),bytesReceived:num(r.bytesReceived),rttMs:num(r.currentRoundTripTime)==null?null:Math.round(r.currentRoundTripTime*1000)});
        }
        if(r.type==='data-channel'&&summary.dataChannels.length<4)
          summary.dataChannels.push({state:['connecting','open','closing','closed'].includes(r.state)?r.state:'unknown'});
      }
      link.statsSummary=summary;link.statsUnavailable=false;
    }catch{link.statsUnavailable=true;}finally{link.statsPending=false;this.publishDiagnostics();}
  }
  publishDiagnostics() {
    if(!this.onDiagnostics)return;
    const links=[...this.links.values()].map(link=>({
      endpoint:link.peer===0?(this.stages===1?'first / last':'first'):link.peer===this.stages-1?'last':'chat',
      stage:link.peer,closed:link.dead,remoteDescriptionReceived:link.remoteSet,
      attempt:this.policy==='relay'?'private-turn':this.relayPeers.has(link.peer)?'turn-fallback':'direct-first',
      directAttempt:this.directAttempts.get(link.peer)||null,
      state:link.failureState||{ice:link.pc.iceConnectionState,connection:link.pc.connectionState,
        gathering:link.pc.iceGatheringState,signaling:link.pc.signalingState},
      dataChannel:link.channelBeforeClose||link.channel?.readyState||'not open',
      localCandidates:link.localCandidates,remoteCandidates:link.remoteCandidates,rejectedCandidates:link.candidateErrors,
      receivedCandidateTypes:link.receivedTypes||{},remoteMdnsCandidates:link.remoteMdnsCandidates||0,
      signalingFailed:link.signalFailed===true,setupTimedOut:link.late===true,
      iceErrors:link.iceErrors||[],stats:link.statsSummary,statsUnavailable:link.statsUnavailable===true,
    }));
    try{this.onDiagnostics({version:1,links});}catch{/* Diagnostics must never disrupt chat. */}
  }
  startDiagnostics() {
    if(!this.onDiagnostics||this.diagnosticsTimer||this.diagnosticsStopped)return;
    const sample=()=>{for(const link of this.links.values())if(!link.dead)void this.sampleDiagnostics(link);this.publishDiagnostics();};
    sample();this.diagnosticsTimer=setInterval(sample,1000);
  }
  close() {
    this.phaseCandidates.clear();
    this.diagnosticsStopped=true;clearInterval(this.diagnosticsTimer);this.diagnosticsTimer=null;
    for(const link of this.links.values())if(!link.dead)this.unlink(link.peer);
    super.close();this.publishDiagnostics();
  }
  failureMessage() {
    const entries=[...this.links.values()];
    const link=entries.find(l=>l.dead||l.dropped||l.rejected);
    if(!link)return 'A contributor connection failed.';
    const endpoint=link.peer===0?'first':link.peer===this.stages-1?'last':'chat';
    const cause=link.signalFailed?'signaling request failed':link.rejected?'data-channel protocol rejected':
      link.late?'data channel did not open within 30 seconds':link.dropped==='failed'?'peer connection failed':
      link.dropped==='disconnected'?'peer connection disconnected':'data channel closed';
    const state=link.failureState;
    const phase=state?` ICE ${state.ice || 'unknown'}, peer ${state.connection || 'unknown'}, gathering ${state.gathering || 'unknown'}; candidates local ${link.localCandidates || 0}, received ${link.remoteCandidates || 0}, rejected ${link.candidateErrors || 0}.`:'';
    return `The ${endpoint} contributor link failed: ${cause} (${link.remoteSet?'remote SDP received':'no remote SDP received'}).${phase}`;
  }
  accepts(peer,kind) {
    if(this.stage===0 && peer===-1 && kind==='prompt')return this.capacity*4;
    if(this.stage===-1 && peer===this.stages-1 && kind==='chat_token')return 0;
    return null;
  }
  send(peer,kind,meta,body,fallback) {
    const link=this.links.get(peer);
    if(!link||link.dead)return Promise.resolve(fallback?.());
    if(link.unacked.length>=256){this.drop(link,'chat acknowledgements stalled');return Promise.resolve(fallback?.());}
    return super.send(peer,kind,meta,body,fallback);
  }
  async open() {
    for(const peer of this.neighbours){
      const link=await this.link(peer,true);
      if(link.offerer)await this.offer(link);
    }
    this.arm();this.startDiagnostics();
  }
}

// Exact supported tokenizer decoder operations. Unsupported decoders keep HTTP text delivery.
export function makeDecoder(config) {
  const supported=d=>d && (d.type==='Sequence'?d.decoders.every(supported):
    ['ByteLevel','ByteFallback','Fuse','Strip','Metaspace'].includes(d.type) ||
    d.type==='Replace' && typeof d.pattern?.String==='string');
  if(!supported(config.decoder))throw Error('This tokenizer uses coordinator text decoding');
  const vocab=new Map(Object.entries({...config.vocab,...config.added}).map(([s,id])=>[id,s]));
  const special=new Set([...(config.special||[]),...(config.stop_ids||[])]);
  const bytes=[...Array(94)].map((_,i)=>i+33).concat([...Array(12)].map((_,i)=>i+161),[...Array(82)].map((_,i)=>i+174));
  const byteMap=new Map(bytes.map(b=>[String.fromCodePoint(b),b]));
  let extra=256;
  for(let b=0;b<256;b++)if(!bytes.includes(b))byteMap.set(String.fromCodePoint(extra++),b);
  const utf8=new TextDecoder();
  function decode(parts,d){
    if(d.type==='Sequence')return d.decoders.reduce((p,step)=>decode(p,step),parts);
    if(d.type==='Fuse')return [parts.join('')];
    if(d.type==='Replace')return parts.map(s=>s.split(d.pattern.String).join(d.content));
    if(d.type==='Strip')return parts.map(s=>{for(let n=0;n<d.start&&s.startsWith(d.content);n++)s=s.slice(d.content.length);for(let n=0;n<d.stop&&s.endsWith(d.content);n++)s=s.slice(0,-d.content.length);return s;});
    if(d.type==='Metaspace')return parts.map((s,i)=>{s=s.split(d.replacement).join(' ');return i===0&&d.prepend_scheme!=='never'&&s.startsWith(' ')?s.slice(1):s;});
    if(d.type==='ByteLevel')return [utf8.decode(Uint8Array.from([...parts.join('')].flatMap(c=>byteMap.has(c)?[byteMap.get(c)]:[...new TextEncoder().encode(c)])))];
    if(d.type==='ByteFallback'){
      const result=[];let pending=[];
      const flush=()=>{if(pending.length){result.push(utf8.decode(Uint8Array.from(pending)));pending=[];}};
      for(const s of parts){if(/^<0x[0-9A-Fa-f]{2}>$/.test(s))pending.push(parseInt(s.slice(3,5),16));else{flush();result.push(s);}}flush();return result;
    }
  }
  return ids=>decode(ids.filter(id=>!special.has(id)).map(id=>{if(!vocab.has(id))throw Error('Unknown token');return vocab.get(id);}),config.decoder).join('');
}

export async function contributorChat({stage,stages,capacity,ice,policy,api,onPrompt}) {
  let edges=null, lease=null, stopped=false, pending=false;
  let outputTurn=null, outputIndex=0;
  const receive=async m=>{
    if(pending||m.kind!=='prompt'||m.body.byteLength%4)return;
    pending=true;
    try{
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',m.body)),b=>b.toString(16).padStart(2,'0')).join('');
      const result=await api({action:'claim',lease,permit:m.meta.permit,digest});
      if(result.accepted)onPrompt({kind:'prompt',meta:{...result,chat_lease:lease},body:m.body,peer:true});
    }catch{/* HTTP prompt delivery remains available. */}finally{pending=false;}
  };
  const poll=async()=>{
    while(!stopped){
      try{
        const r=await api({action:'poll'});
        if(r.lease!==lease){edges?.close();lease=r.lease;edges=null;
          if(lease){edges=new ChatEdges({stage,stages,capacity,ice,policy,lease,api,onMessage:m=>{void receive(m);}});await edges.open();}}
        for(const item of r.signals||[])await edges?.handle(item);
      }catch{/* Keep inference and HTTP fallback independent of chat signaling. */}
      await new Promise(r=>setTimeout(r,750));
    }
  };
  void poll();
  return {
    async stats(){ return edges ? edges.stats() : []; },
    token(meta,token){
      if(meta?.chat_lease!==lease||!Number.isSafeInteger(meta?.chat_turn))return;
      const identity=`${meta.epoch}:${meta.chat_turn}`;
      if(outputTurn!==identity){outputTurn=identity;outputIndex=0;}
      const index=outputIndex++;
      if(edges?.has(-1))void edges.send(-1,'chat_token',{turn:meta.chat_turn,epoch:meta.epoch,index,token},'',()=>false).catch(()=>{});
    },
    close(){stopped=true;edges?.close();},
  };
}

// Reconcile a fast peer stream with the authoritative HTTP stream without duplicated text.
export class ChatReply {
  constructor(decode,grant){this.decode=decode;this.grant=grant;this.ids=[];this.pending=new Map();this.received=new Set();this.text='';this.peer=true;this.done=false;}
  token(meta){
    if(this.done||meta.turn!==this.grant.turn||meta.epoch!==this.grant.epoch||
      !Number.isSafeInteger(meta.index)||meta.index<0||meta.index>=this.grant.limit||
      !Number.isSafeInteger(meta.token)||meta.token<0)return this.text;
    this.received.add(meta.index);
    if(!this.peer||meta.index<this.ids.length)return this.text;
    if(this.pending.size>=256&&!this.pending.has(meta.index)){this.peer=false;return this.text;}
    this.pending.set(meta.index,meta.token);
    while(this.pending.has(this.ids.length)){const i=this.ids.length;this.ids.push(this.pending.get(i));this.pending.delete(i);}
    try{const text=this.decode(this.ids);if(!text.endsWith('\ufffd'))this.text=text;}catch{this.peer=false;}
    return this.text;
  }
  receivedTokens(generated){return [...this.received].filter(index=>index<generated).length;}
  relay(text,done=false){
    if(done||!this.text.startsWith(text)){this.text=text;if(!done)this.peer=false;}
    if(done){this.done=true;this.pending.clear();}
    return this.text;
  }
}

export function formatReplyMetrics(event, received = 0) {
  const positive = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
  const count = Number.isSafeInteger(event.generated) && event.generated >= 0 ? event.generated : null;
  const rate = positive(event.tokens_per_s) ? event.tokens_per_s
    : !('tokens_per_s' in event) && positive(event.ms_per_token) && event.ms_per_token > 0 ? 1000 / event.ms_per_token : null;
  const seconds = (value, digits = 2) => positive(value) ? `${value.toFixed(digits)}s` : '—';
  const webRtc = count === null ? 0 : Math.min(count, Math.max(0, Number.isSafeInteger(received) ? received : 0));
  const prompt = event.prompt_transport === 'webrtc' ? 'prompt + ' : '';
  const transport = `WebRTC: ${prompt}${webRtc}/${count ?? '—'} tokens`
    + (event.prompt_transport === 'coordinator' ? ' · prompt via coordinator' : event.prompt_transport !== 'webrtc' ? ' · prompt route unconfirmed' : '');
  return `${rate === null ? '— tok/s' : `≈${rate.toFixed(1)} tok/s`} · ${count ?? '—'} tokens · first token ${seconds(event.first_token_s)} · total ${seconds(event.total_s, 1)} · ${transport}`
    + ('cached_tokens' in event ? (Number.isSafeInteger(event.cached_tokens) && event.cached_tokens >= 0
      ? ` · Prompt cache: ${event.cached_tokens}/${event.prompt_tokens ?? '—'} tokens` : ' · Prompt cache: not reported') : '')
    + (event.stopped === 'length' ? ` · Output limit reached (${count ?? '—'} tokens)` : '');
}
