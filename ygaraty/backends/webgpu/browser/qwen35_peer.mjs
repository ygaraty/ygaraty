// DO-40: direct links between neighbouring Qwen stages over WebRTC data channels.
//
// The coordinator only relays signaling (SDP offers and answers, ICE candidates) through
// /api/signal/<id>, to neighbouring stages only. Each unordered pair of neighbouring stages (k, k+1 and
// the last stage with the first) gets one RTCPeerConnection, offered by the lower stage, with one ordered,
// reliable data channel, encrypted by DTLS; the fingerprints travel in the SDP through the coordinator, so
// each end is the browser the coordinator assigned. A message is a JSON header {k: kind, m: meta, n: body
// bytes} followed by the body in pieces of at most 64 KiB (SCTP messages above 256 KiB are not portable).
//
// What a stage accepts from a peer (user_webrtc.md, section 5): from the previous stage, hidden states
// only, at most `limits.hidden` bytes; the first stage, from the last, tokens only, with no body. Anything
// else (another kind, an oversized or malformed header, a piece without a header or past the announced
// length) closes that link, and the pair's traffic goes through the coordinator relay.
//
// Nothing sent on a link is lost when it dies: the receiver acknowledges each message ({a: seq}), and when a
// link is closed the sender hands every unacknowledged message, in order, to its relay fallback (with
// `resent`). Every message carries its sequence number (`m.pseq`), so a receiver that did get the original
// drops the copy.

const PIECE = 64 * 1024;
// WS-26: the browser's SCTP sends about four packets (~4.7 KB) per send() call before it waits for an
// acknowledgement, so a 5 KB hidden state sent whole took an extra round trip on every decode step. The start
// of each body goes out in small pieces (more sends, one flight); the rest in PIECE-sized ones, which large
// transfers need.
const HEAD_PIECE = 4 * 1024, HEAD_BYTES = 16 * 1024;
export function pieces(n) {
  const out = [];
  for (let at = 0; at < n;) { const size = Math.min(n - at, at < HEAD_BYTES ? HEAD_PIECE : PIECE); out.push([at, size]); at += size; }
  return out;
}
const HIGH = 4 * 1024 * 1024, LOW = 1024 * 1024;  // sender's buffer limits
const HEADER_BYTES = 1024;
const STALL_MS = 5000;  // a send buffer that does not drain for this long: the link is dropped (DO-44)

export class Peers {
  // ice: an RTCIceServer list, or an async function returning one (fetched again for every new link: the
  // coordinator's TURN credentials are short-lived). call(path, body): the page's API call (relative to
  // /api/). onMessage({kind, meta, body, from}). limits: {hidden: the largest hidden-state body in bytes}.
  // openWithinMs: a link must open within this long of `arm()` (made before it) or of being made (after);
  // one that does not is closed and never used (DO-44: a link that opened 70 s late stalled a turn).
  constructor({ id, stage, stages, ice, policy, call, onMessage, limits, openWithinMs = 30000 }) {
    Object.assign(this, { id, stage, stages, call, onMessage, openWithinMs });
    this.armed = false;
    this.ice = ice || [];
    this.policy = policy || "all";  // "relay": TURN only
    this.limits = limits || {};
    this.links = new Map();  // peer stage -> link
    this.relinks = new Map();  // peer stage -> times the link was made again
    this.signalChain = Promise.resolve();
    this.locks = new Map();  // peer stage -> the last change to that link: changes run one at a time
    this.seqs = new Map();  // peer stage -> the last sequence number sent to it
    this.polling = false;
    this.neighbours = [...new Set([(stage + 1) % stages, (stage - 1 + stages) % stages])].filter((p) => p !== stage);
    this.previous = (stage - 1 + stages) % stages;  // the only stage this one takes data from
  }

  // The kinds this stage takes from `peer`, with the largest body for each.
  accepts(peer, kind) {
    if (peer !== this.previous) return null;
    if (this.stage === 0) return kind === "token" || kind === "tokens" ? 0 : null;  // from the last stage
    return kind === "hidden" || kind === "hiddens" ? (this.limits.hidden ?? 0) : null;
  }

  // Runs fn alone for that peer's link: an incoming offer, a relink and a re-offer each check the link and
  // may make it again (after an await for ICE servers), so two at once would close each other's (DO-43).
  exclusive(peer, fn) {
    const run = (this.locks.get(peer) || Promise.resolve()).then(fn, fn);
    this.locks.set(peer, run.catch(() => {}));
    return run;
  }

  async iceServers() { return typeof this.ice === "function" ? await this.ice() : this.ice; }

  // initial: one of connect()'s first links, held to the deadline counted from arm(); otherwise a link made
  // again (relink, a new offer, a re-offer) gets its own openWithinMs.
  async link(peer, initial = false) {
    const old = this.links.get(peer);
    if (old) { if (!old.dead) this.retire(old); old.dead = true; try { old.pc.close(); } catch {} }
    const pc = new RTCPeerConnection({ iceServers: await this.iceServers(peer), iceTransportPolicy: this.policy });
    const link = { peer, pc, channel: null, pending: [], chain: Promise.resolve(), offerer: this.stage < peer,
                   bytesSent: 0, bytesReceived: 0, messagesSent: 0, messagesReceived: 0, opened: null, remoteSet: false, dead: false, rejected: false,
                   made: performance.now(), openedMs: null, late: false, dropped: null, unacked: [] };
    link.open = new Promise((resolve) => { link.opened = resolve; });
    this.links.set(peer, link);
    if (this.armed) this.deadline(link, initial ? Math.max(0, this.armedAt + this.openWithinMs - performance.now()) : this.openWithinMs);
    // A connection that drops or fails is not waited for: the pair goes to the relay for good.
    pc.onconnectionstatechange = () => {
      if (!link.dead && ["disconnected", "failed"].includes(pc.connectionState)) this.drop(link, pc.connectionState);
    };
    pc.onicecandidate = (e) => { if (e.candidate && !link.dead) this.signal(peer, { candidate: e.candidate.toJSON() }); };
    const attach = (channel) => {
      channel.binaryType = "arraybuffer";
      channel.bufferedAmountLowThreshold = LOW;
      let header = null, buffer = null, at = 0;
      channel.onmessage = (e) => {
        if (link.dead) return;
        if (typeof e.data === "string") {
          if (header !== null) return this.reject(link, "a header before the previous body ended");
          let h = null;
          try { h = e.data.length <= HEADER_BYTES ? JSON.parse(e.data) : null; } catch {}
          if (h && Object.keys(h).length === 1 && Number.isSafeInteger(h.a)) {  // the peer got everything up to a
            link.unacked = link.unacked.filter((u) => u.seq > h.a);
            return;
          }
          if (!h || typeof h.k !== "string" || !Number.isSafeInteger(h.n) || h.n < 0 || typeof h.m !== "object" || h.m === null)
            return this.reject(link, "malformed header");
          const most = this.accepts(peer, h.k);
          if (most === null) return this.reject(link, `kind ${h.k} not accepted from stage ${peer}`);
          if (h.n > most) return this.reject(link, `${h.k}: ${h.n} bytes, at most ${most}`);
          if (h.n === 0) { link.messagesReceived++; this.deliver(peer, h, new ArrayBuffer(0)); this.ack(channel, h); return; }
          header = h; buffer = new Uint8Array(h.n); at = 0;
          return;
        }
        if (header === null) return this.reject(link, "a body piece without a header");
        if (at + e.data.byteLength > header.n) return this.reject(link, "a body longer than its header said");
        buffer.set(new Uint8Array(e.data), at);
        at += e.data.byteLength;
        link.bytesReceived += e.data.byteLength;
        if (at === header.n) { link.messagesReceived++; this.deliver(peer, header, buffer.buffer); this.ack(channel, header); header = null; buffer = null; }
      };
      channel.onopen = () => {
        if (link.dead) return;
        if (link.late) return this.drop(link, "opened late");
        link.channel = channel;
        link.openedMs = Math.round(performance.now() - link.made);
        link.opened(true);
      };
      channel.onclose = () => { if (link.channel === channel) this.drop(link, "disconnected"); };
      if (channel.readyState === "open") channel.onopen();
    };
    if (link.offerer) attach(pc.createDataChannel("hidden", { ordered: true }));
    else pc.ondatachannel = (e) => attach(e.channel);
    return link;
  }

  ack(channel, header) {
    if (Number.isSafeInteger(header.m.pseq)) try { channel.send(JSON.stringify({ a: header.m.pseq })); } catch {}
  }

  // A link closing with messages the peer has not acknowledged: each goes to its relay fallback, in order.
  retire(link) {
    const unacked = link.unacked.splice(0);
    for (const u of unacked) try { u.fallback?.(u.seq, true); } catch {}
  }

  // A peer that broke the rules: its link is closed for good (until the coordinator relinks the stage).
  reject(link, reason) {
    link.rejected = reason;
    this.retire(link);
    this.unlink(link.peer);
  }

  // A link that cannot be trusted to deliver (late, stalled, disconnected): closed, so nothing sent on it
  // can arrive after the relay has taken over.
  drop(link, reason) {
    if (link.dead) return;
    link.dropped = link.dropped || reason;
    this.retire(link);
    if (this.links.get(link.peer) === link) this.unlink(link.peer);
    else { link.dead = true; try { link.pc.close(); } catch {} }
  }

  deadline(link, ms) {
    setTimeout(() => { if (!link.dead && !link.channel) { link.late = true; this.drop(link, "not open in time"); } }, ms);
  }

  // At `go`: from now on every link must open within openWithinMs (each one made later, within that long of
  // being made). Links still negotiating at their deadline are closed and their pair uses the relay.
  arm() {
    if (this.armed) return;
    this.armed = true;
    this.armedAt = performance.now();
    for (const link of this.links.values()) if (!link.dead) this.deadline(link, this.openWithinMs);
  }

  deliver(from, header, body) { this.onMessage({ kind: header.k, meta: header.m, body, from }); }

  // In order, so an offer reaches the other side before its candidates. Resolves to the coordinator's
  // answer ({delivered: false}: no browser held that stage, so the signal was dropped), or null on error.
  signal(to, data) {
    const sent = this.signalChain.then(() => this.call(`signal/${this.id}`, { to, data })).catch(() => null);
    this.signalChain = sent;
    return sent;
  }

  async offer(link) {
    await link.pc.setLocalDescription(await link.pc.createOffer());
    link.offerDropped = false;
    const answer = await this.signal(link.peer, { sdp: link.pc.localDescription.toJSON() });
    if (answer && answer.delivered === false && this.links.get(link.peer) === link) link.offerDropped = true;
  }

  handle(item) {
    if (!this.neighbours.includes(item.sender)) return Promise.resolve();
    return this.exclusive(item.sender, () => this.apply(item));
  }

  async apply({ sender, data }) {
    let link = this.links.get(sender);
    // A new offer on a used or closed link: the other stage's browser was replaced, so the link starts again.
    if (data.sdp?.type === "offer" && (!link || link.dead || link.remoteSet)) {
      this.relinks.set(sender, (this.relinks.get(sender) || 0) + (link ? 1 : 0));
      link = await this.link(sender);
    }
    if (!link || link.dead) return;
    const { pc } = link;
    if (data.sdp) {
      await pc.setRemoteDescription(data.sdp);
      link.remoteSet = true;
      for (const c of link.pending.splice(0)) await pc.addIceCandidate(c).catch(() => {});
      if (data.sdp.type === "offer") {
        await pc.setLocalDescription(await pc.createAnswer());
        this.signal(sender, { sdp: pc.localDescription.toJSON() });
      }
    } else if (data.candidate) {
      if (link.remoteSet) await pc.addIceCandidate(data.candidate).catch(() => {});
      else if (link.pending.length < 64) link.pending.push(data.candidate);
    }
  }

  // Makes every link, then reads signals until `close` (a replaced neighbour links again), so an offer that
  // arrives early finds its link. A link that does not open leaves its traffic to the coordinator relay.
  async connect() {
    for (const peer of this.neighbours) if (!this.links.get(peer)) await this.link(peer, true);
    this.polling = true;
    this.poll = (async () => {
      while (this.polling) {
        const items = await this.call(`signal/${this.id}`).catch(() => null);
        if (items === null) { await new Promise((resolve) => setTimeout(resolve, 2000)); continue; }
        for (const item of items) await this.handle(item).catch(() => {});
      }
    })();
    for (const peer of this.neighbours) {
      await this.exclusive(peer, async () => {
        const link = this.links.get(peer);
        if (link.offerer && !link.remoteSet) await this.offer(link);
      });
    }
  }

  // After `go`, when every stage holds its browser and reads its signals: offers again on each link whose
  // offer the coordinator dropped because the other stage had no browser yet (DO-43: a stage that finished
  // its self-test first offered to nobody). Only those: re-offering a link whose answer is still on its way
  // would pair that answer with the new offer, and the link would never open.
  async reoffer() {
    for (const peer of this.neighbours) {
      await this.exclusive(peer, async () => {
        const link = this.links.get(peer);
        if (!link || !link.offerer || !link.offerDropped || link.remoteSet || link.channel) return;
        await this.offer(await this.link(peer));
      });
    }
  }

  // The coordinator says a neighbouring stage lost its browser: that link is closed.
  unlink(peer) {
    const link = this.links.get(peer);
    if (!link) return;
    link.dead = true;
    link.channel = null;
    try { link.pc.close(); } catch {}
  }

  // The coordinator says a new browser holds a neighbouring stage: a new link (fresh ICE credentials), offered
  // by the lower stage. A link the new browser already started (by its offer) is kept.
  async relink(peer) {
    if (!this.neighbours.includes(peer)) return;
    await this.exclusive(peer, async () => {
      const link = this.links.get(peer);
      if (link && !link.dead && (link.remoteSet || link.channel)) return;  // the new browser's offer came first
      this.relinks.set(peer, (this.relinks.get(peer) || 0) + 1);
      const fresh = await this.link(peer);
      if (fresh.offerer) await this.offer(fresh);
    });
  }

  // Waits until every link is open or `timeoutMs` has passed, and returns each link's state. It watches the
  // current links: one made again meanwhile (a new offer) counts when it opens (DO-43).
  async settle(timeoutMs) {
    const until = performance.now() + timeoutMs;
    while (performance.now() < until && !this.neighbours.every((p) => this.has(p)))
      await new Promise((resolve) => setTimeout(resolve, 50));
    return this.stats();
  }

  has(peer) {
    const link = this.links.get(peer);
    return !!link && !link.dead && link.channel?.readyState === "open" && link.pc.connectionState === "connected";
  }

  // Queued in order on the link; resolves to true when the last piece is handed to the channel. If the link
  // has closed by the time the message's turn comes, `fallback` (the relay) sends it instead, and the
  // promise resolves to what `fallback` returns.
  // fallback(seq, resent): sends the message through the relay instead; `resent` when the link had taken it
  // but died before the peer acknowledged it.
  send(peer, kind, meta, body, fallback) {
    const link = this.links.get(peer);
    const seq = (this.seqs.get(peer) || 0) + 1;
    this.seqs.set(peer, seq);
    meta = { ...meta, pseq: seq };
    const relay = () => fallback?.(seq, false);
    const bytes = body instanceof ArrayBuffer ? new Uint8Array(body) : ArrayBuffer.isView(body)
      ? new Uint8Array(body.buffer, body.byteOffset, body.byteLength) : new Uint8Array(0);
    link.chain = link.chain.then(async () => {
      const ch = link.channel;
      if (link.dead || !ch || ch.readyState !== "open" || link.pc.connectionState !== "connected") return relay();
      // Never wait long on a link: a buffer that does not drain means the path stopped delivering, so the
      // link is dropped (closed, so the part already sent can never complete) and the relay sends it.
      const drained = () => ch.bufferedAmount <= HIGH || new Promise((resolve) => {
        const done = (ok) => { clearTimeout(timer); ch.removeEventListener("bufferedamountlow", low); ch.removeEventListener("close", closed); resolve(ok); };
        const low = () => done(true), closed = () => done(false), timer = setTimeout(() => done(false), STALL_MS);
        ch.addEventListener("bufferedamountlow", low); ch.addEventListener("close", closed);
      });
      if (!(await drained())) { this.drop(link, "stalled"); return relay(); }
      ch.send(JSON.stringify({ k: kind, m: meta, n: bytes.byteLength }));
      for (const [at, size] of pieces(bytes.byteLength)) {
        if (!(await drained())) { this.drop(link, "stalled"); return relay(); }
        ch.send(bytes.subarray(at, at + size));
      }
      link.bytesSent += bytes.byteLength;
      link.messagesSent++;  // a token's message has no body: bytes alone would not show it (DO-51 gate 3)
      link.unacked.push({ seq, fallback });  // until the peer says it has it
      return true;
    }).catch(() => relay());
    return link.chain;
  }

  // Each link: whether it opened, the selected candidate pair's types (host, srflx, prflx, relay) and
  // protocol, and its round-trip time. No addresses: contributors' IPs stay out of reports.
  async stats() {
    const out = [];
    for (const peer of this.neighbours) {
      const link = this.links.get(peer);
      if (!link) { out.push({ peer, open: false, state: "new" }); continue; }
      const entry = { peer, open: this.has(peer), state: link.pc.connectionState, bytesSent: link.bytesSent,
                      bytesReceived: link.bytesReceived, messagesSent: link.messagesSent,
                      messagesReceived: link.messagesReceived, relinks: this.relinks.get(peer) || 0 };
      if (link.rejected) entry.rejected = true;
      if (link.openedMs !== null) entry.openedMs = link.openedMs;  // from the link being made to its opening
      if (link.dropped) entry.dropped = { "opened late": "late", "not open in time": "late", stalled: "stalled",
                                          disconnected: "disconnected", failed: "failed" }[link.dropped] || "other";
      try {
        const report = await link.pc.getStats();
        let pair = null;
        report.forEach((s) => { if (s.type === "transport" && s.selectedCandidatePairId) pair = report.get(s.selectedCandidatePairId); });
        if (!pair) report.forEach((s) => { if (s.type === "candidate-pair" && s.nominated && s.state === "succeeded") pair = s; });
        if (pair) {
          const local = report.get(pair.localCandidateId) || {}, remote = report.get(pair.remoteCandidateId) || {};
          Object.assign(entry, { local: local.candidateType, remote: remote.candidateType, protocol: local.protocol,
                                 relayProtocol: local.relayProtocol, rttMs: pair.currentRoundTripTime != null ? pair.currentRoundTripTime * 1e3 : null });
        }
      } catch (error) { entry.statsError = true; }
      out.push(entry);
    }
    return out;
  }

  close() {
    this.polling = false;  // the long-poll in flight ends within its 20 s
    for (const link of this.links.values()) { link.dead = true; try { link.pc.close(); } catch {} }
  }
}
