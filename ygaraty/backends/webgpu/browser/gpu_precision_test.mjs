import {ComputeSDK,tensor} from './sdk.mjs';
document.getElementById('run').onclick=async()=>{
  const out=document.getElementById('results');
  try{
    const sdk=new ComputeSDK();await sdk.initialize(true);
    if(!sdk.device)throw Error('WebGPU required; no CPU fallback permitted');
    const cases=[
      ['sum cancellation',[1e8,1,-1e8],[1,1,1]],
      ['input rounding',[1+2**-30,1],[1,-1]],
      ['product rounding',[1+2**-20,1],[1-2**-20,-1]],
      ['long cancellation',Array.from({length:512},(_,i)=>i%4===0?1e6:i%4===2?-1e6:Math.sin(i)),Array(512).fill(1)]
    ];
    const checks=[];
    for(const [name,a,b] of cases){
      const inputs=[tensor([1,a.length],a),tensor([b.length,1],b)];
      // Neumaier reference avoids the naive CPU sum losing the very residual
      // this cancellation test is intended to check.
      let sum=0,correction=0;
      for(let i=0;i<a.length;i++){const x=a[i]*b[i],next=sum+x;correction+=Math.abs(sum)>=Math.abs(x)?(sum-next)+x:(x-next)+sum;sum=next;}
      const reference=sum+correction;
      const raw=await sdk.execute({op:'matmul',shape:[1,1],precision:'webgpu',attrs:{gpu_arithmetic:'float32'}},inputs);
      const corrected=await sdk.execute({op:'matmul',shape:[1,1],precision:'webgpu',attrs:{gpu_arithmetic:'compensated'}},inputs);
      const error=Math.abs(corrected.value.data[0]-reference);
      checks.push({name,reference,raw:raw.value.data[0],corrected:corrected.value.data[0],error,backend:corrected.backend,passed:name==='long cancellation'?error<=1e-10+1e-10*Math.abs(reference):error===0});
    }
    out.textContent=JSON.stringify({passed:checks.every(c=>c.passed)&&checks.filter(c=>c.raw!==c.reference).length>=3,checks},null,2);
  }catch(e){out.textContent=JSON.stringify({passed:false,error:e.message});}
};
