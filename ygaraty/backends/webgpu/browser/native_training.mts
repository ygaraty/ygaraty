// Deno entry point: native wgpu, no browser/DOM shims or npm GPU bindings.
import {PeerTrainingWorker, validatePeerConfig} from './peer_training.mjs';
interface DenoRuntime {
  args: string[];
  version: {deno: string; v8: string; typescript: string};
  readTextFile(path: string): Promise<string>;
  addSignalListener(signal: 'SIGINT' | 'SIGTERM', handler: () => void): void;
  serve(options: {hostname: string; port: number; signal: AbortSignal; onListen: () => void},
    handler: (request: Request) => Promise<Response>): {finished: Promise<void>};
}
const deno = (globalThis as typeof globalThis & {Deno?: DenoRuntime}).Deno;
if (!deno) throw Error('Run this worker with Deno --unstable-webgpu');
if (deno.args.length !== 3) throw Error('Expected CONFIG HOST PORT');
const config = validatePeerConfig(JSON.parse(await deno.readTextFile(deno.args[0])));
const port = Number(deno.args[2]);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw Error('Invalid port');
const worker = new PeerTrainingWorker(config, {...deno.version}), controller = new AbortController();
deno.addSignalListener('SIGINT', () => controller.abort());
deno.addSignalListener('SIGTERM', () => controller.abort());
try {
  await worker.initialize();
  const server = deno.serve({hostname: deno.args[1], port, signal: controller.signal,
    onListen: () => console.log(JSON.stringify({ready: true, rank: config.rank, adapter: worker.sdk.adapterInfo}))},
  request => worker.handle(request));
  await server.finished;
} finally { worker.close(); }
