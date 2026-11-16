// Adapter selection for the standalone experiment, separate from shared kernels.
export function adapterEvidence(info={}, legacyFallback) {
  return {
    vendor: String(info.vendor || ''), architecture: String(info.architecture || ''),
    device: String(info.device || ''), description: String(info.description || ''),
    isFallbackAdapter: info.isFallbackAdapter ?? legacyFallback ?? null,
  };
}

export function requireNvidiaAdapter(info) {
  const vendor = info.vendor.trim().toLowerCase();
  const text = [info.vendor, info.architecture, info.device, info.description].join(' ').toLowerCase();
  if (!/^(nvidia(?:[- ]corporation)?|(?:0x)?10de)$/.test(vendor) || info.isFallbackAdapter !== false ||
      /swiftshader|llvmpipe|lavapipe|software|microsoft basic/.test(text))
    throw Error(`NVIDIA hardware adapter required; refusing unverified/software device: ${JSON.stringify(info)}`);
  return info;
}

export async function initializeParityGPU(sdk, {requireNvidia=false, gpu=globalThis.navigator?.gpu}={}) {
  if (!requireNvidia) {
    const reason = await sdk.initialize(true);
    return {reason, adapter: sdk.device ? adapterEvidence(sdk.device.adapterInfo) : null};
  }
  if (!gpu) throw Error('NVIDIA WebGPU required, but navigator.gpu is unavailable');
  const adapter = await gpu.requestAdapter({powerPreference: 'high-performance', forceFallbackAdapter: false});
  if (!adapter) throw Error('No WebGPU adapter; check NVIDIA Vulkan libraries and Chromium flags');
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  const evidence = requireNvidiaAdapter(adapterEvidence(info, adapter.isFallbackAdapter));
  // The SDK executes on this exact device, from the verified adapter. Do not
  // probe one adapter then let SDK.initialize() silently select another.
  const device = await adapter.requestDevice();
  if (device.adapterInfo) requireNvidiaAdapter(adapterEvidence(device.adapterInfo, evidence.isFallbackAdapter));
  sdk.device = device;
  sdk.reason = 'Verified NVIDIA hardware WebGPU device';
  device.lost.then(() => {sdk.device = null; sdk.reason = 'NVIDIA GPU device lost';});
  return {reason: sdk.reason, adapter: evidence};
}
