import {denseMatmulGPU} from './training_gpu.mjs';
import {executeCPU} from './sdk_cpu.mjs';
import {CONV_OPS, convolutionGPU} from './convolution.mjs';
export class ComputeSDK {
  constructor(){this.device=null;this.reason='JavaScript float64';this.adapterInfo=null;}
  async initialize(preferGPU=true){
    if(preferGPU && globalThis.navigator?.gpu) try {
      const adapter=await navigator.gpu.requestAdapter();
      if(adapter){
        // The adapter's own buffer limits (the defaults are 128 to 256 MiB): a budget may use them.
        const {maxBufferSize,maxStorageBufferBindingSize}=adapter.limits,info=adapter.info||{};
        this.device=await adapter.requestDevice({requiredLimits:{maxBufferSize,maxStorageBufferBindingSize}});
        this.adapterInfo={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,
          isFallbackAdapter:adapter.isFallbackAdapter??info.isFallbackAdapter??null,maxBufferSize,maxStorageBufferBindingSize};
        this.reason='WebGPU available; arithmetic selected by job';
        this.device.lost.then(()=>{this.device=null;this.reason='GPU lost; JavaScript fallback';});}
    } catch(e){this.reason=`GPU unavailable; JavaScript fallback: ${e.message}`;}
    return this.reason;
  }
  async matmul(a,b,shape,arithmetic='float32'){
    if(!this.device)throw Error('Dense matmul requires a GPU device');
    return denseMatmulGPU(this.device,a,b,shape,arithmetic);
  }
  async execute(task,inputs){
    if(CONV_OPS.includes(task.op)&&task.precision==='webgpu'){
      if(!this.device)throw Error('Convolution requires WebGPU; no CPU fallback for a WebGPU test');
      return {value:await convolutionGPU(this.device,task.op,inputs,task.attrs,task.shape),backend:'webgpu'};
    }
    if((task.op==='factorization_tile'&&inputs.length===4||task.op==='factorization_tiles')&&task.attrs?.x_spec){
      // A tile, or a group of tiles, with its latent rows (a plan's latent_inputs): X is generated on the GPU (experiment 48).
      if(!this.device)throw Error('This factorization tile requires WebGPU');
      const gpu=await import('./factorization_gpu.mjs');
      const run=task.op==='factorization_tiles'?gpu.factorizationTilesGPU:gpu.factorizationTileGPU;
      return {value:(await run(this.device,inputs,task.attrs,task.shape)).value,backend:'webgpu'};
    }
    if(task.precision==='webgpu' && task.op==='matmul' && !this.device)throw Error('This job requires WebGPU; enable GPU or use a compatible browser');
    if(this.device && task.precision!=='float64' && task.op==='matmul')try{return {value:await this.matmul(inputs[0],inputs[1],task.shape,task.attrs?.gpu_arithmetic||'float32'),backend:'webgpu'};}catch(e){this.device=null;this.reason='GPU failed; JavaScript fallback';if(task.precision==='webgpu')throw e;}
    return {value:executeCPU(task.op,inputs,task.attrs,task.shape),backend:'cpu'};
  }
}
