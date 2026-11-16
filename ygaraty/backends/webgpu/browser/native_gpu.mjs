// WebGPU initialization extracted from the archived DO-68 native runner.
// Deno supplies wgpu; Node/Bun load the public Dawn binding explicitly.
export async function initializeGPU({modulePath, dawnOptions = []} = {}) {
  if (globalThis.Deno) {
    if (!globalThis.navigator?.gpu) throw Error("Deno has no WebGPU; use a WebGPU-enabled Deno 2 runtime.");
    return navigator.gpu;
  }
  const {createRequire} = await import('node:module');
  const {pathToFileURL} = await import('node:url');
  const {resolve} = await import('node:path');
  const require = createRequire(resolve(process.cwd(), 'package.json'));
  let binding;
  try {
    binding = await import(pathToFileURL(modulePath ? resolve(modulePath) : require.resolve('webgpu')).href);
  } catch (error) {
    throw Error("Cannot load Dawn. Install webgpu@0.6.2 in your current directory, or use --webgpu-module /path/to/webgpu/index.js. " + error.message);
  }
  Object.assign(globalThis, binding.globals);
  const gpu = binding.create(dawnOptions);
  if (!globalThis.navigator) globalThis.navigator = {};
  Object.defineProperty(navigator, 'gpu', {value: gpu, configurable: true});
  return gpu;
}

export async function probeGPU(gpu) {
  const adapter = await gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw Error('No WebGPU adapter available. Check your GPU driver and runtime.');
  const device = await adapter.requestDevice();
  try {
    // A real compute and readback probe, not just adapter enumeration.
    const output = device.createBuffer({size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    const read = device.createBuffer({size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    const module = device.createShaderModule({code: '@group(0) @binding(0) var<storage,read_write> value: array<u32>; @compute @workgroup_size(1) fn main() { value[0] = 42u; }'});
    const pipeline = await device.createComputePipelineAsync({layout:'auto', compute:{module, entryPoint:'main'}});
    const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0), entries:[{binding:0, resource:{buffer:output}}]});
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0,group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,4); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(read.getMappedRange())[0];
    read.unmap(); output.destroy(); read.destroy();
    if (result !== 42) throw Error('WebGPU compute probe failed');
    return {adapter: Object.fromEntries(['vendor','architecture','device','description'].map(k=>[k,adapter.info?.[k]])), features: [...adapter.features], compute: result};
  } finally {device.destroy();}
}
