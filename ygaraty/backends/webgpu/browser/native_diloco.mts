// Native Deno contributor for the same transactional DiLoCo HTTP protocol.
import {contribute} from './training_client.mjs';
import {ComputeSDK} from './sdk_compute.mjs';
interface DenoRuntime {
  args: string[];
  readTextFile(path: string): Promise<string>;
  addSignalListener(signal: 'SIGINT' | 'SIGTERM', handler: () => void): void;
}
const deno = (globalThis as typeof globalThis & {Deno?: DenoRuntime}).Deno;
if (!deno || deno.args.length !== 1) throw Error('Expected Deno native_diloco.mjs CONFIG');
const config: unknown = JSON.parse(await deno.readTextFile(deno.args[0]));
if (!config || typeof config !== 'object') throw Error('Invalid contributor config');
const c = config as Record<string, unknown>;
if (typeof c.base_url !== 'string' || typeof c.join_token !== 'string' || c.join_token.length < 16 ||
    typeof c.client_id !== 'string' || typeof c.timeout_seconds !== 'number' ||
    !Number.isFinite(c.timeout_seconds) || c.timeout_seconds < 1 || c.timeout_seconds > 7200)
  throw Error('Invalid contributor configuration');
const url = new URL(c.base_url);
if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash || url.pathname !== '/')
  throw Error('Expected coordinator origin');
const controller = new AbortController(), sdk = new ComputeSDK();
const timer = setTimeout(() => controller.abort(), c.timeout_seconds * 1000);
for (const signal of ['SIGINT', 'SIGTERM'] as const) deno.addSignalListener(signal, () => controller.abort());
const initialize = sdk.initialize.bind(sdk);
sdk.initialize = async prefer => {
  const result = await initialize(prefer);
  if (!sdk.device || sdk.adapterInfo?.isFallbackAdapter !== false) throw Error('Hardware WebGPU required');
  console.log(JSON.stringify({state: 'gpu-ready', adapter: sdk.adapterInfo}));
  return result;
};
try {
  const result = await contribute({baseURL: url.origin, joinToken: c.join_token, clientId: c.client_id,
    signal: controller.signal, sdk, onStatus: status => console.log(JSON.stringify(status))});
  if (result.state !== 'complete') throw Error(`Training ended with state ${result.state}`);
} finally { clearTimeout(timer); sdk.device?.destroy(); }
