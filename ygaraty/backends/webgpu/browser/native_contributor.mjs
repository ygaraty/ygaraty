// Live contributor using the same admission, self-test, weights and engine as Chromium.
import {initializeGPU, probeGPU} from './native_gpu.mjs';
import {contributorURL, installContributorEnvironment} from './native_environment.mjs';
const args = globalThis.Deno ? Deno.args : process.argv.slice(2);
const exit = code => globalThis.Deno ? Deno.exit(code) : process.exit(code);
let env, stopping=false;
async function stop(code) {
  if (stopping) return;
  stopping=true;
  await env?.close();
  exit(code);
}
try {
  let url, check=false, modulePath, webrtcPath, dawnOptions=[];
  for (let i=0;i<args.length;i++) {
    if (args[i]==='--check') check=true;
    else if (args[i]==='--webgpu-module') modulePath=args[++i];
    else if (args[i]==='--dawn-option') dawnOptions.push(args[++i]);
    else if (args[i]==='--webrtc-module') webrtcPath=args[++i];
    else if (args[i].startsWith('-') || url) throw Error('Unexpected contributor argument');
    else url=args[i];
  }
  if (!check) contributorURL(url);
  const gpu = await initializeGPU({modulePath,dawnOptions});
  if (check) {console.log(JSON.stringify(await probeGPU(gpu))); exit(0);}
  // DO-97: direct links through node-datachannel's RTCPeerConnection polyfill, when given.
  const webrtc = webrtcPath ? await import(new URL(webrtcPath, `file://${globalThis.Deno ? Deno.cwd() : process.cwd()}/`).href) : null;
  if (webrtcPath && typeof webrtc.RTCPeerConnection !== 'function') throw Error('The WebRTC module has no RTCPeerConnection');
  console.log(webrtc ? 'Native contributor: direct links (WebRTC) when the session routes them; no persistent weight cache.'
                     : 'Native contributor: coordinator relay; no WebRTC or persistent weight cache.');
  for (const signal of ['SIGINT','SIGTERM']) {
    if (globalThis.Deno) Deno.addSignalListener(signal,()=>stop(0));
    else process.on(signal,()=>stop(0));
  }
  env=installContributorEnvironment(url, {webrtc});
  const {contributorError, contributorReloadRequired} = await import('./qwen35_contributor.mjs');
  if (contributorError) throw contributorError;
  if (contributorReloadRequired || env.reloadRequested) throw Error('Loading interrupted by a coordinator restart; rerun this contributor command.');
  await stop(0);
} catch (error) {
  // Do not dump command arguments or share-link credentials.
  console.error('Contributor failed:', error.message);
  await stop(1);
}
