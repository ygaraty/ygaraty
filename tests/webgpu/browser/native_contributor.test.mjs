import {test} from 'node:test';
import assert from 'node:assert/strict';
import {contributorURL, installContributorEnvironment, nativePeerConnection} from '../../../ygaraty/backends/webgpu/browser/native_environment.mjs';

test('native contributor validates invitations without accepting arbitrary protocols', () => {
  for (const url of ['file:///tmp/x#key=x','https://example.test/','https://u:p@example.test/#key=x','https://example.test/#key=x&pool=y'])
    assert.throws(()=>contributorURL(url));
  const url=contributorURL('https://example.test/sessions/demo#key=a%2Bb');
  assert.equal(new URLSearchParams(url.hash.slice(1)).get('key'),'a+b');
  assert.equal(new URLSearchParams(url.hash.slice(1)).get('autostart'),'1');
});

test('native host uses shared admission and reports errors, releases its timers and sends leave', async () => {
  const originalFetch=globalThis.fetch;
  const nav=Object.getOwnPropertyDescriptor(globalThis,'navigator');
  const device={lost:new Promise(()=>{}),addEventListener(){}};
  const adapter={info:{},limits:{maxBufferSize:1024,maxStorageBufferBindingSize:1024,maxStorageBuffersPerShaderStage:8},features:new Set(),requestDevice:async()=>device};
  Object.defineProperty(globalThis,'navigator',{configurable:true,value:{gpu:{requestAdapter:async()=>adapter},userAgent:'test-native'}});
  try {
    for (const scenario of ['full','unauthorized','broken-selftest']) {
      const calls=[];
      globalThis.fetch=async(url, options)=>{
        calls.push({url,options});
        if (url.endsWith('/join')) {
          if(scenario==='unauthorized') return new Response('denied',{status:403});
          return Response.json(scenario==='full' ? {accepted:false,reason:'full'} : {id:'worker',worker_key:'instance-key',selftest:null});
        }
        return Response.json({});
      };
      const env=installContributorEnvironment('https://example.test/sessions/demo/#key=invite',{log(){}});
      const originalTimeout=globalThis.setTimeout, delayed=[];
      if(scenario==='full') {
        // A browser may still receive an old invitation with a pool fragment.
        location.hash += '&pool=retired-pool-key';
        globalThis.setTimeout=(fn,ms,...rest)=>{
          if(ms===5000) {delayed.push(fn);return 0;}
          return originalTimeout(fn,ms,...rest);
        };
      }
      try {
        const result=await import(`../../../ygaraty/backends/webgpu/browser/qwen35_contributor.mjs?native-test=${scenario}`);
        assert.equal(Boolean(result.contributorError),scenario!=='full');
        const join=calls.find(x=>x.url.endsWith('/join'));
        assert.equal(join.url,'https://example.test/sessions/demo/api/join');
        assert.equal(join.options.headers['X-Key'],'invite');
        assert.equal(JSON.parse(join.options.body).userAgent,'test-native');
        assert.equal(globalThis.RTCPeerConnection,undefined);
        assert.equal(globalThis.Worker,undefined);
        assert.equal(delayed.length,0,'must not schedule a hosted pool redirect');
        assert.equal(location.href,undefined);
      } finally {globalThis.setTimeout=originalTimeout;await env.close();}
      if(scenario==='broken-selftest') {
        const leave=calls.find(x=>x.url.endsWith('/leave/worker'));
        assert.equal(leave.options.headers['X-Worker-Key'],'instance-key');
        assert.ok(calls.find(x=>x.url.endsWith('/error/worker')));
      }
      assert.equal(globalThis.document,undefined);
    }
  } finally {
    globalThis.fetch=originalFetch;
    if(nav)Object.defineProperty(globalThis,'navigator',nav);else delete globalThis.navigator;
  }
});

test('closing an idle native host cancels its shutdown deadline', async () => {
  const schedule=globalThis.setTimeout, cancel=globalThis.clearTimeout;
  const deadlines=new Set();
  globalThis.setTimeout=(fn,ms,...args)=>{
    const timer=schedule(fn,ms,...args);
    if(ms===1500)deadlines.add(timer);
    return timer;
  };
  globalThis.clearTimeout=timer=>{deadlines.delete(timer);cancel(timer);};
  try {
    const env=installContributorEnvironment('https://example.test/#key=invite',{log(){}});
    await env.close();
    assert.equal(deadlines.size,0,'shutdown must not leave a live timer');
  } finally {
    for(const timer of deadlines)cancel(timer);
    globalThis.setTimeout=schedule;globalThis.clearTimeout=cancel;
  }
});

test('native host offers WebRTC only when given a polyfill, and restores the globals', async () => {
  const before=globalThis.RTCPeerConnection;
  let env=installContributorEnvironment('https://example.test/#key=invite',{log(){}});
  assert.equal(globalThis.RTCPeerConnection,undefined,'relay-only by default');
  await env.close();
  class FakePeer {constructor(config) {this.config=config;} async getStats() {return new Map([['p',{type:'candidate-pair',currentRoundTripTime:23,totalRoundTripTime:46}],['t',{type:'transport'}]]);}}
  env=installContributorEnvironment('https://example.test/#key=invite',{log(){},webrtc:{RTCPeerConnection:FakePeer,RTCIceCandidate:class {},RTCSessionDescription:class {}}});
  const pc=new globalThis.RTCPeerConnection({iceServers:[{urls:'stun:h:3478'},{urls:['turn:h:3478'],username:'1795:s:0',credential:'a/b+c='}],iceTransportPolicy:'relay'});
  assert.ok(pc instanceof FakePeer);
  assert.deepEqual(pc.config.iceServers,[{urls:'stun:h:3478'},{urls:['turn:h:3478'],username:'1795%3As%3A0',credential:'a%2Fb%2Bc%3D'}]);
  assert.equal(pc.config.iceTransportPolicy,'relay');
  const stats=await pc.getStats();
  assert.equal(stats.get('p').currentRoundTripTime,0.023);
  assert.equal(stats.get('p').totalRoundTripTime,0.046);
  await env.close();
  assert.equal(globalThis.RTCPeerConnection,before);
});
