// DO-68 benchmark runner, exported from mess-ledger's native-only follow-up.
// Run a local harness page under Deno/wgpu or Node/Bun with Dawn.
import {initializeGPU} from './native_gpu.mjs';
const args = globalThis.Deno ? Deno.args : process.argv.slice(2);
const [base, page] = args;
const exit = code => globalThis.Deno ? Deno.exit(code) : process.exit(code);
try {
  if (!base || !page) throw Error('usage: native_runner.mjs BASE PAGE_PATH');
  await initializeGPU();
  const send = globalThis.fetch;
  globalThis.fetch = (url, options) => send(typeof url === 'string' && url.startsWith('/') ? base + url : url, options);
  const {pathToFileURL} = await import('node:url');
  await import(pathToFileURL(page).href);
  exit(0);
} catch (error) {
  console.error(error);
  exit(1);
}
