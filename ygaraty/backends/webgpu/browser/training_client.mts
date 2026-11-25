// Generated output: training_client.mjs. HTTP transport stays outside the trainer.
import {DiLoCoTrainer} from './diloco.mjs';
import {ComputeSDK} from './sdk_compute.mjs';
import {decodeJoin, decodeAssignment, object} from './training_contracts.mjs';
import type {Credentials, Submission, ContributorSDK, ContributorStatus, SessionStatus} from './training_contracts.mjs';

export interface ContributorOptions {
  baseURL?: string;
  joinToken: string;
  clientId?: string;
  sdk?: ContributorSDK;
  signal?: AbortSignal;
  onStatus?: (status: ContributorStatus) => void;
}
class HTTPError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}
class ContributorError extends Error {}
const errorMessage = (error: unknown): string => error instanceof Error ? error.message : String(error);
const sleep = (ms: number, signal?: AbortSignal): Promise<void> => new Promise(resolve => {
  const finish = () => { clearTimeout(timer); signal?.removeEventListener('abort', finish); resolve(); };
  const timer = setTimeout(finish, ms);
  signal?.addEventListener('abort', finish, {once: true});
  if (signal?.aborted) finish();
});

export async function contribute({baseURL=globalThis.location?.origin, joinToken,
  clientId=crypto.randomUUID(), sdk=new ComputeSDK(), signal, onStatus=()=>{}}: ContributorOptions
): Promise<SessionStatus | {state: 'stopped'}> {
  await sdk.initialize(true);
  let credentials: Credentials | null = null;
  let pending: Submission | null = null;
  let heartbeat: ReturnType<typeof setInterval> | undefined;
  let heartbeatRunning = false;
  const request = async (path: string, token: string, data: unknown={}): Promise<unknown> => {
    const response = await fetch(new URL(path, baseURL), {method: 'POST',
      headers: {'Content-Type': 'application/json', Authorization: `Bearer ${token}`},
      body: JSON.stringify(data), signal: AbortSignal.any(
        [AbortSignal.timeout(30000), ...(signal ? [signal] : [])])});
    const body: unknown = await response.json();
    if (!response.ok) {
      const detail = object(body).error;
      throw new HTTPError(typeof detail === 'string' ? detail : `HTTP ${response.status}`, response.status);
    }
    return body;
  };
  const workerRequest = (action: 'heartbeat' | 'result' | 'assignment', data?: Submission): Promise<unknown> => {
    if (!credentials) throw new ContributorError('No active contributor lease');
    return request(`/api/workers/${credentials.worker}/${action}`, credentials.token, data);
  };
  try {
    while (!signal?.aborted) {
      try {
        if (!credentials) {
          const joined = decodeJoin(await request('/api/join', joinToken, {client_id: clientId}));
          if ('state' in joined) { onStatus(joined); return joined; }
          credentials = joined;
          if (joined.precision === 'webgpu' && !sdk.device)
            throw new ContributorError('WebGPU unavailable. Use a supported browser over HTTPS or localhost.');
          clearInterval(heartbeat);
          heartbeat = setInterval(async () => {
            if (heartbeatRunning || !credentials) return;
            heartbeatRunning = true;
            try { await workerRequest('heartbeat'); }
            catch (error: unknown) { onStatus({state: 'reconnecting', message: errorMessage(error)}); }
            finally { heartbeatRunning = false; }
          }, joined.heartbeat_seconds * 1000);
          onStatus({state: 'joined', worker: joined.worker, backend: sdk.reason});
        }
        // Retain exactly the same submission across transport failures. ACK loss
        // must never execute another local optimizer step or alter the result.
        if (pending) {
          const receipt = object(await workerRequest('result', pending));
          if (receipt.accepted !== true || typeof receipt.version !== 'number' ||
              !Number.isSafeInteger(receipt.version) || receipt.version < 0)
            throw new ContributorError('Invalid result acknowledgement');
          pending = null;
        }
        const bundle = decodeAssignment(await workerRequest('assignment'));
        if (bundle.state !== 'assignment') {
          onStatus(bundle);
          if (bundle.state === 'complete' || bundle.state === 'failed') return bundle;
          await sleep(1000, signal);
          continue;
        }
        const trainer = new DiLoCoTrainer(credentials.worker, {sdk, precision: credentials.precision, signal});
        trainer.restore(bundle.optimizer, bundle.assignment.weights);
        onStatus({state: 'training', version: bundle.assignment.version, worker: credentials.worker});
        const start = performance.now();
        const result = await trainer.runRound(bundle.assignment, bundle.plans);
        pending = {result, optimizer: trainer.checkpoint(), stats: {...result.stats,
          seconds: (performance.now() - start) / 1000, precision: credentials.precision,
          adapter: sdk.adapterInfo}};
      } catch (error: unknown) {
        if (signal?.aborted) break;
        if (error instanceof ContributorError) throw error;
        if (error instanceof HTTPError) {
          if (error.status === 401) { credentials = null; pending = null; clearInterval(heartbeat); }
          // Restore committed state after a rejected/aborted round.
          else if (error.status === 409 && pending) pending = null;
        } else if (!(error instanceof Error) || !['TypeError', 'TimeoutError', 'AbortError'].includes(error.name)) {
          throw error;
        }
        onStatus({state: 'reconnecting', message: errorMessage(error)});
        await sleep(1500, signal);
      }
    }
    return {state: 'stopped'};
  } finally { clearInterval(heartbeat); }
}
