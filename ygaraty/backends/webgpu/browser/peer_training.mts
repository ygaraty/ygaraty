// Fixed-membership LAN collective. Each worker owns a model replica; gradients
// travel on child->parent edges and weights on parent->child edges. A failed
// step aborts the experiment: this protocol does not promise crash recovery.
import {ComputeSDK} from './sdk_compute.mjs';
import {executeLocalPlan} from './diloco.mjs';
import {tensor, type Tensor} from './sdk_tensor.mjs';
import {decodeStepPlan, object, type Named, type Precision, type StepPlan} from './training_contracts.mjs';
import {CONTENT_TYPE, decodeFrame, digest, encodeFrame, readFrameBody, type Frame} from './tensor_transport.mjs';

export interface PeerConfig {
  rank: number; peers: string[]; token: string; session: string;
  precision: Precision; timeoutMs: number;
}
interface Job {version: number; samples: number; counts: number[]; plan: StepPlan; constants: Named<Tensor>;
  optimizer: StepPlan | null; gradientRefs: Named<string>}
const integer = (value: unknown, min = 0): number => {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < min) throw Error('Invalid integer');
  return value;
};
function match(actual: Named<Tensor>, expected: Named<Tensor>): void {
  if (Object.keys(actual).length !== Object.keys(expected).length) throw Error('Tensor membership mismatch');
  for (const [name, value] of Object.entries(expected))
    if (!Object.hasOwn(actual, name) || actual[name].shape.join() !== value.shape.join()) throw Error('Tensor shape mismatch');
}
export function validatePeerConfig(raw: unknown): PeerConfig {
  const c = object(raw), rank = integer(c.rank);
  if (!Array.isArray(c.peers) || ![1, 2, 4, 10].includes(c.peers.length) || rank >= c.peers.length) throw Error('Expected 1, 2, 4 or 10 peers');
  const peers = c.peers.map(value => {
    if (typeof value !== 'string') throw Error('Invalid peer URL');
    const url = new URL(value);
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.pathname !== '/' || url.search || url.hash)
      throw Error('Expected peer origin URL');
    return url.origin;
  });
  if (new Set(peers).size !== peers.length || typeof c.token !== 'string' || c.token.length < 32 ||
      typeof c.session !== 'string' || !c.session || !['webgpu', 'float64'].includes(String(c.precision))) throw Error('Invalid peer configuration');
  const timeoutMs = integer(c.timeoutMs, 1000);
  if (timeoutMs > 300000) throw Error('Timeout exceeds five minutes');
  return {rank, peers, token: c.token, session: c.session, precision: c.precision as Precision, timeoutMs};
}
export class PeerTrainingWorker {
  readonly sdk = new ComputeSDK();
  readonly children: number[];
  private weights: Named<Tensor> = {};
  private moments: Named<Tensor> = {};
  private version = 0;
  private job: Job | null = null;
  private preparedHash = '';
  private committedHash = '';
  private reduction: Promise<Frame> | null = null;
  private step: Promise<Frame> | null = null;
  private commit: Promise<Frame> | null = null;
  private committing = false;
  private busy = false;
  private failed = false;
  private sent = 0;
  private received = 0;
  constructor(readonly config: PeerConfig, readonly runtime: Record<string, string> = {}) {
    this.children = [2 * config.rank + 1, 2 * config.rank + 2].filter(n => n < config.peers.length);
  }
  async initialize(): Promise<void> {
    await this.sdk.initialize(this.config.precision === 'webgpu');
    if (this.config.precision === 'webgpu' && (!this.sdk.device || this.sdk.adapterInfo?.isFallbackAdapter !== false))
      throw Error(`Hardware WebGPU required: ${this.sdk.reason}`);
  }
  close(): void { this.sdk.device?.destroy(); }
  private async peer(rank: number, path: string, metadata: Record<string, unknown>, tensors: Named<Tensor> = {}): Promise<Frame> {
    const body = await encodeFrame({...metadata, session: this.config.session}, tensors);
    this.sent += body.length;
    const response = await fetch(this.config.peers[rank] + path, {method: 'POST',
      headers: {Authorization: `Bearer ${this.config.token}`, 'Content-Type': CONTENT_TYPE}, body,
      signal: AbortSignal.timeout(this.config.timeoutMs)});
    const data = await readFrameBody(response.body); this.received += data.length;
    const result = await decodeFrame(data);
    if (!response.ok) throw Error(`Peer ${rank} failed: ${String(result.metadata.error)}`);
    return result;
  }
  private subtreeCount(rank: number, counts: number[]): number {
    if (rank >= counts.length) return 0;
    return counts[rank] + this.subtreeCount(2 * rank + 1, counts) + this.subtreeCount(2 * rank + 2, counts);
  }
  private async local(job: Job): Promise<Frame> {
    const inputs = {...job.constants};
    for (const [name, value] of Object.entries(this.weights)) {
      if (!job.plan.parameters[name]) throw Error('Missing parameter');
      inputs[job.plan.parameters[name]] = value;
    }
    const started = performance.now();
    const result = await executeLocalPlan(job.plan, inputs, this.sdk,
      {precision: this.config.precision, arithmetic: 'compensated', signal: AbortSignal.timeout(this.config.timeoutMs)});
    const gradients = Object.fromEntries(Object.entries(job.plan.updates).map(([name, ref]) => {
      const value = result.values.get(ref); if (!value) throw Error('Missing gradient');
      return [name, value];
    }));
    match(gradients, this.weights);
    const loss = result.values.get(job.plan.loss)?.data[0];
    if (loss === undefined || !Number.isFinite(loss)) throw Error('Invalid loss');
    return {metadata: {samples: job.samples, loss, version: job.version,
      nodes: [{rank: this.config.rank, samples: job.samples, seconds: (performance.now() - started) / 1000, ...result.stats}]}, tensors: gradients};
  }
  private async reduce(job: Job): Promise<Frame> {
    // Start children before local kernels; await in fixed rank order so network
    // arrival order cannot alter floating-point reduction order.
    const parts = await Promise.all([this.local(job), ...this.children.map(rank =>
      this.peer(rank, '/reduce', {version: job.version}))]);
    let samples = 0, loss = 0;
    const gradients: Named<Tensor> = {}, nodes: unknown[] = [];
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i], count = integer(part.metadata.samples, 1);
      const expected = i === 0 ? job.samples : this.subtreeCount(this.children[i - 1], job.counts);
      if (count !== expected || part.metadata.version !== job.version || typeof part.metadata.loss !== 'number' ||
          !Number.isFinite(part.metadata.loss) || !Array.isArray(part.metadata.nodes)) throw Error('Invalid subtree result');
      match(part.tensors, this.weights);
      const total = samples + count, a = samples / total, b = count / total;
      for (const [name, value] of Object.entries(part.tensors)) {
        const previous = gradients[name];
        const data = new Float64Array(value.data.length);
        for (let j = 0; j < data.length; j++) data[j] = (previous ? a * previous.data[j] : 0) + b * value.data[j];
        gradients[name] = tensor(value.shape, data);
      }
      loss = a * loss + b * part.metadata.loss; samples = total; nodes.push(...part.metadata.nodes);
    }
    return {metadata: {version: job.version, samples, loss, nodes}, tensors: gradients};
  }
  private async apply(job: Job): Promise<Frame> {
    if (!job.optimizer) throw Error('Root needs optimizer plan');
    const reduced = await (this.reduction ??= this.reduce(job));
    const inputs = {...this.weights, ...this.moments};
    for (const [name, value] of Object.entries(reduced.tensors)) inputs[job.gradientRefs[name]] = value;
    const result = await executeLocalPlan(job.optimizer, inputs, this.sdk, {precision: 'float64'});
    const next: Named<Tensor> = {}, moments: Named<Tensor> = {};
    for (const [name, ref] of Object.entries(job.optimizer.updates)) {
      const value = result.values.get(ref); if (!value) throw Error('Missing optimizer output');
      (Object.hasOwn(this.weights, name) ? next : moments)[name] = value;
    }
    match(next, this.weights); match(moments, this.moments);
    // Publish version only after all descendant acknowledgements. Any lost node
    // aborts the run; callers must start a fresh session, never skip a rank.
    await this.broadcast(job.version, next);
    this.moments = moments;
    return {metadata: {...reduced.metadata, version: this.version}, tensors: this.weights};
  }
  private async broadcast(version: number, weights: Named<Tensor>): Promise<Frame> {
    if (Object.keys(this.weights).length) match(weights, this.weights);
    await Promise.all(this.children.map(rank => this.peer(rank, '/commit', {version}, weights)));
    this.weights = weights; this.version = version;
    return {metadata: {version, accepted: true}, tensors: {}};
  }
  private async dispatch(path: string, frame: Frame, hash: string): Promise<Frame> {
    const m = frame.metadata;
    if (m.session !== this.config.session) throw Error('Session mismatch');
    if (path === '/status') return {metadata: {rank: this.config.rank, version: this.version, failed: this.failed,
      precision: this.config.precision, runtime: this.runtime, children: this.children, adapter: this.sdk.adapterInfo,
      peer_sent_bytes: this.sent, peer_received_bytes: this.received}, tensors: {}};
    if (this.failed) throw Error('Session failed; restart all workers');
    const version = integer(m.version);
    if (path === '/commit') {
      if (hash === this.committedHash && this.commit) return this.commit;
      if (this.busy || this.committing || (Object.keys(this.weights).length ? version !== this.version + 1 : version !== 0)) throw Error('Unexpected commit version');
      if (!Object.keys(frame.tensors).length) throw Error('Empty weights');
      if (version > 0 && (!this.job || this.job.version !== version || !this.reduction)) throw Error('Commit before reduction');
      this.committedHash = hash;
      this.committing = true;
      return this.commit = (async () => {
        if (version > 0) await this.reduction;
        return this.broadcast(version, frame.tensors);
      })().finally(() => { this.committing = false; });
    }
    if (path === '/prepare') {
      if (hash === this.preparedHash && this.job) return {metadata: {accepted: true}, tensors: {}};
      if (this.busy || this.committing || version !== this.version + 1 || !Object.keys(this.weights).length ||
          (this.job?.version === version)) throw Error('Unexpected prepare version');
      const counts = m.counts;
      if (!Array.isArray(counts) || counts.length !== this.config.peers.length) throw Error('Invalid sample counts');
      const decoded = counts.map(n => integer(n, 1));
      const plan = decodeStepPlan(m.plan), optimizer = m.optimizer ? decodeStepPlan(m.optimizer) : null;
      const refs = object(m.gradientRefs ?? {});
      if (Object.values(refs).some(x => typeof x !== 'string')) throw Error('Invalid gradient references');
      if (this.config.rank === 0 && (!optimizer || Object.keys(refs).length !== Object.keys(this.weights).length)) throw Error('Missing root optimizer');
      if (this.config.rank === 0 && !Object.keys(this.moments).length) {
        for (const [name, value] of Object.entries(this.weights)) for (const kind of ['m', 'v'])
          this.moments[`adam_${kind}_${name}`] = tensor(value.shape, new Float64Array(value.data.length));
      }
      this.job = {version, samples: decoded[this.config.rank], counts: decoded, plan, constants: frame.tensors,
        optimizer, gradientRefs: refs as Named<string>};
      this.preparedHash = hash; this.reduction = null; this.step = null;
      return {metadata: {accepted: true}, tensors: {}};
    }
    if (!this.job || version !== this.job.version) throw Error('Unprepared version');
    if (path === '/reduce') return this.reduction ??= this.reduce(this.job);
    if (path === '/step' && this.config.rank === 0) {
      if (!this.step) { this.busy = true; this.step = this.apply(this.job).finally(() => { this.busy = false; }); }
      return this.step;
    }
    throw Error('Unknown peer operation');
  }
  async handle(request: Request): Promise<Response> {
    if (request.method !== 'POST' || request.headers.get('Authorization') !== `Bearer ${this.config.token}`)
      return new Response(null, {status: 401});
    const path = new URL(request.url).pathname;
    try {
      const body = await readFrameBody(request.body), frame = await decodeFrame(body);
      // State setup is serialized; reductions/steps are promise-cached so a
      // duplicate request never executes a second optimizer update.
      const result = await this.dispatch(path, frame, await digest(body));
      return new Response(await encodeFrame(result.metadata, result.tensors), {headers: {'Content-Type': CONTENT_TYPE}});
    } catch (error: unknown) {
      if (['/reduce', '/step', '/commit'].includes(path)) this.failed = true;
      return new Response(await encodeFrame({error: error instanceof Error ? error.message : 'Worker failed'}),
        {status: 409, headers: {'Content-Type': CONTENT_TYPE}});
    }
  }
}
