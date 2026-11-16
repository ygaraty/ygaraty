import test from 'node:test';
import assert from 'node:assert/strict';
import {checkBuffer, validateBufferLayout as check} from '../../../ygaraty/backends/webgpu/browser/inference_layout.mjs';
const b = size => ({size});
const w = (rows=3, cols=4) => ({kind:'F32', rows, cols, rowBytes:cols*4, buffer:b(rows*cols*4)});

test('byte views enforce complete footprints, alignment and explicit size',()=>{
  checkBuffer({buffer:b(32),offset:8,size:16},16);
  for(const [binding,size] of [[{buffer:b(32),offset:8,size:16},20],[{buffer:b(32),offset:2},4],[{},4],[{buffer:b(32),offset:24,size:16},4],[b(32),Infinity]])
    assert.throws(()=>checkBuffer(binding,size));
});
test('matrix strides include last row, offset and chunk row base',()=>{
  const args=[w(),b(40),b(52),2,{xStride:6,yStride:7,rowBase:3}];
  check('matmul',args);
  assert.throws(()=>check('matmul',[...args.slice(0,2),b(48),...args.slice(3)]),/allocation/);
  assert.throws(()=>check('matmul',[{...w(),rowBytes:12},...args.slice(1)]),/row layout/);
  assert.throws(()=>check('matmul',[w(),b(40),b(52),2,{rows:4}]),/row limit/);
});
test('packed row bytes are verified against quantization block geometry',()=>{
  const weight={kind:'Q6_K',rows:2,cols:512,rowBytes:420,buffer:b(840)};
  check('matmul',[weight,b(2048),b(8),1]);
  assert.throws(()=>check('matmul',[{...weight,buffer:b(836)},b(2048),b(8),1]),/allocation/);
});
test('norm checks grouped strides and the optional gate',()=>{
  const args=[b(112),b(16),b(64),2,{size:4,groups:2,inStride:16,inGroup:8,gate:b(64),gateStride:8}];
  check('rms_norm',args);
  assert.throws(()=>check('rms_norm',[...args.slice(0,4),{...args[4],gate:b(60)}]),/allocation/);
});
test('cache and attention bounds include capacity, precision and split scratch',()=>{
  check('cache_store_grouped',[b(32),b(160),1,9,10,'f16',2,4]);
  assert.throws(()=>check('cache_store_grouped',[b(32),b(156),1,9,10,'f16',2,4]),/allocation/);
  const args=[b(64),b(16416),b(16416),b(64),1,512,513,'f16',4,2,8,1,b(256),b(64)];
  // Query/output need 4 heads * 8 floats, not 64 bytes.
  assert.throws(()=>check('grouped_attention',args),/allocation/);
  args[0]=args[3]=b(128);check('grouped_attention',args);
  args[12]=b(252);assert.throws(()=>check('grouped_attention',args),/allocation/);
});
test('argmax requires both index and score, and recurrent state is bounded',()=>{
  check('argmax',[b(16),b(8),4]);
  assert.throws(()=>check('argmax',[b(16),b(4),4]),/allocation/);
  check('gated_delta',[b(32768),b(256),b(256),b(2097152),b(16384),1]);
  assert.throws(()=>check('gated_delta',[b(32768),b(256),b(256),b(2097148),b(16384),1]),/allocation/);
});
