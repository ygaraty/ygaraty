// Generated output: gradient_client.mjs. Experimental gradient-only worker.
import {ComputeSDK} from './sdk_compute.mjs';
import {tensor, type Tensor} from './sdk_tensor.mjs';
import {executeLocalPlan} from './diloco.mjs';
import {decodeStepPlan, decodeTensor, object} from './training_contracts.mjs';
import type {Named, Precision} from './training_contracts.mjs';

export async function contributeGradients({worker, token, baseURL=location.origin,
  precision='webgpu', signal}: {worker: number; token: string; baseURL?: string;
    precision?: Precision; signal?: AbortSignal}): Promise<void> {
  const sdk = new ComputeSDK();
  await sdk.initialize(precision === 'webgpu');
  if (precision === 'webgpu' && (!sdk.device || sdk.adapterInfo?.isFallbackAdapter !== false))
    throw Error('This experiment requires a hardware WebGPU adapter');
  const request = async (path: string, body?: unknown): Promise<Record<string, unknown>> => {
    const response = await fetch(new URL(path, baseURL), {
      method: body === undefined ? 'GET' : 'POST',
      headers: {Authorization: `Bearer ${token}`, 'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.any([AbortSignal.timeout(30000), ...(signal ? [signal] : [])]),
    });
    const raw: unknown = await response.json();
    if (!response.ok) throw Error(`Gradient transport HTTP ${response.status}`);
    return object(raw);
  };
  const unpack = (raw: unknown): Tensor => {
    const value = decodeTensor(raw);
    if (!('data' in value)) throw Error('Gradient experiment requires array wire tensors');
    return tensor(value.shape, value.data);
  };
  let pending: unknown = null;
  try {
    while (!signal?.aborted) {
      try {
        if (pending) {
          const receipt = await request(`/gradient/${worker}/result`, pending);
          if (receipt.accepted !== true) throw Error('Invalid gradient acknowledgement');
          pending = null;
        }
        const job = await request(`/gradient/${worker}/job`);
        if (job.state === 'complete') return;
        if (job.state === 'waiting') {
          await new Promise(resolve => setTimeout(resolve, 10));
          continue;
        }
        if (job.state !== 'gradient' || typeof job.version !== 'number' ||
            !Number.isSafeInteger(job.version) || typeof job.samples !== 'number' ||
            !Number.isSafeInteger(job.samples) || job.samples < 1) throw Error('Invalid gradient job');
        const plan = decodeStepPlan(job.plan);
        const inputs: Named<Tensor> = Object.fromEntries(Object.entries(plan.constants).map(([n, value]) => [n, unpack(value)]));
        for (const [name, raw] of Object.entries(object(job.weights))) {
          const key = plan.parameters[name];
          if (!key) throw Error('Gradient parameter mismatch');
          inputs[key] = unpack(raw);
        }
        const start = performance.now();
        const result = await executeLocalPlan(plan, inputs, sdk, {precision, arithmetic: 'compensated', signal});
        const gradients = Object.fromEntries(Object.entries(plan.updates).map(([name, ref]) => {
          const value = result.values.get(ref);
          if (!value) throw Error('Missing gradient output');
          return [name, {shape: value.shape, data: Array.from(value.data)}];
        }));
        const loss = result.values.get(plan.loss);
        if (!loss) throw Error('Missing loss');
        pending = {version: job.version, samples: job.samples, loss: loss.data[0], gradients,
          stats: {...result.stats, seconds: (performance.now() - start) / 1000, adapter: sdk.adapterInfo}};
      } catch (error: unknown) {
        if (signal?.aborted) return;
        // Preserve the same result across lost ACKs. Protocol/numerical failures
        // stop this bounded experiment rather than silently restarting a step.
        if (!(error instanceof Error) || !['TypeError', 'TimeoutError'].includes(error.name)) throw error;
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
  } finally { sdk.device?.destroy(); }
}
