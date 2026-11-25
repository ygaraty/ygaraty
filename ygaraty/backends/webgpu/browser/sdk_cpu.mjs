import {inverseNorm} from './optimizer_math.mjs';
import {tensor} from './sdk_tensor.mjs';
import {CONV_OPS, convolutionCPU} from './convolution.mjs';
const sigmoid=x=>{const z=Math.exp(-Math.abs(x));return x>=0?1/(1+z):z/(1+z);};
// Sum matmul partial products laid out as [[col_start, col_stop, count], ...] (as sequence_kernels.recurrent).
const recurrentSum=(parts,blocks,rows,width)=>{
  const out=new Float64Array(rows*width);let k=0;
  for(const [start,stop,count] of blocks)for(let c=0;c<count;c++){const p=parts[k++],w=stop-start;
    for(let r=0;r<rows;r++)for(let j=0;j<w;j++)out[r*width+start+j]+=p.data[r*w+j];}
  return out;
};
// a (rows x k) @ b (k x cols), row-major.
const matmulFlat=(a,b,rows,k,cols)=>{const out=new Float64Array(rows*cols);
  for(let i=0;i<rows;i++)for(let q=0;q<k;q++){const x=a[i*k+q];if(x===0)continue;for(let j=0;j<cols;j++)out[i*cols+j]+=x*b[q*cols+j];}
  return out;};
// One block's (factorization_block: a row shard with all of V) or tile's (factorization_tile: X_ij with
// U_i and V_j) normalized data gradients and loss. Regularization belongs to the plan, once per
// factor. These fused operations currently use the CPU backend.
function factorizationBlock(inputs,attrs,shape,op='factorization_block'){
  const validShape=dims=>Array.isArray(dims)&&dims.length===2&&dims.every(x=>Number.isSafeInteger(x)&&x>0);
  if(inputs.length!==3)throw Error(op+' requires X, U, and V');
  for(const x of inputs){
    if(!x||!validShape(x.shape)||!x.data||x.data.length!==x.shape[0]*x.shape[1])
      throw Error(op+' requires nonempty 2D matrices');
    for(const value of x.data)if(!Number.isFinite(value))throw Error(op+' inputs must be finite');
  }
  const [x,u,v]=inputs,[rows,columns]=x.shape,rank=u.shape[1],total=attrs.total_elements;
  if(u.shape[0]!==rows||v.shape[0]!==columns||v.shape[1]!==rank)
    throw Error(op+' input dimensions do not match');
  // A block's count is a multiple of its width; a tile's only has to cover the tile.
  if(!Number.isSafeInteger(total)||total<rows*columns||(op==='factorization_block'&&total%columns!==0))
    throw Error(op+(op==='factorization_block'?' total_elements must be an integer multiple of columns covering X':' total_elements must be an integer covering X'));
  if(!validShape(shape)||shape[0]!==rows+columns+1||shape[1]!==rank)
    throw Error(op+' output shape must be [rows + columns + 1, rank]');
  const vt=new Float64Array(rank*columns);
  for(let c=0;c<columns;c++)for(let q=0;q<rank;q++)vt[q*columns+c]=v.data[c*rank+q];
  const residual=matmulFlat(u.data,vt,rows,rank,columns),rt=new Float64Array(columns*rows);
  let loss=0;
  for(let r=0;r<rows;r++)for(let c=0;c<columns;c++){
    const i=r*columns+c,e=residual[i]-x.data[i];residual[i]=e;rt[c*rows+r]=e;loss+=e*e;
  }
  const du=matmulFlat(residual,v.data,rows,columns,rank),dv=matmulFlat(rt,u.data,columns,rows,rank);
  const out=new Float64Array((rows+columns+1)*rank);
  for(let i=0;i<du.length;i++)out[i]=du[i]/total;
  for(let i=0;i<dv.length;i++)out[du.length+i]=dv[i]/total;
  out[(rows+columns)*rank]=0.5*loss/total;
  if(!out.every(Number.isFinite))throw Error('Nonfinite output');
  return tensor(shape,out);
}
// Batched sequences are time-major (row t*batch+b); an LSTM step row is [h_state|c_state|h_out|i|f|g|o].
function executeSequence(op,inputs,attrs,shape,out){
  const [m,n]=shape;
  if(op==='lstm_cell'){
    const [xp,bih,bhh,hp,cp,mask,...parts]=inputs,H=attrs.hidden;
    // The recurrent product inline (the whole W_hh.T as the last input), or as summed partials.
    const hw=attrs.inline?matmulFlat(hp.data,parts[0].data,m,H,4*H):recurrentSum(parts,attrs.blocks,m,4*H);
    for(let r=0;r<m;r++){const live=mask.data[r]>0,row=r*7*H;
      for(let j=0;j<H;j++){
        const gate=k=>(xp.data[r*4*H+k*H+j]+bih.data[k*H+j])+(hw[r*4*H+k*H+j]+bhh.data[k*H+j]);
        const i=sigmoid(gate(0)),f=sigmoid(gate(1)),g=Math.tanh(gate(2)),o=sigmoid(gate(3));
        const cn=f*cp.data[r*H+j]+i*g,hn=o*Math.tanh(cn);
        out[row+j]=live?hn:hp.data[r*H+j];out[row+H+j]=live?cn:cp.data[r*H+j];out[row+2*H+j]=live?hn:0;
        out[row+3*H+j]=i;out[row+4*H+j]=f;out[row+5*H+j]=g;out[row+6*H+j]=o;
      }}
  }else if(op==='lstm_cell_backward'||op==='lstm_cell_carry'||op==='lstm_cell_grad'||op==='lstm_state_grad'){
    const H=attrs.hidden,values=[...inputs];let state,cp,mask;
    if(op!=='lstm_state_grad')[state,cp,mask]=values.splice(0,3);
    const gout=attrs.has_out?values.shift():null,carry=attrs.has_carry?values.shift():null;
    const rec=attrs.inline?matmulFlat(values[0].data,values[1].data,m,4*H,H):(values.length?recurrentSum(values,attrs.blocks,m,H):null);
    for(let r=0;r<m;r++)for(let j=0;j<H;j++){
      const dh=(carry?carry.data[r*2*H+j]:0)+(rec?rec[r*H+j]:0),dc=carry?carry.data[r*2*H+H+j]:0;
      if(op==='lstm_state_grad'){out[r*n+j]=dh;out[r*n+H+j]=dc;continue;}
      const live=mask.data[r]>0,row=r*7*H,at=k=>state.data[row+k*H+j];
      const i=at(3),f=at(4),g=at(5),o=at(6),tc=Math.tanh(at(1));
      const dhn=live?dh+(gout?gout.data[r*H+j]:0):0,dcn=live?dc+dhn*o*(1-tc*tc):0;
      if(op!=='lstm_cell_carry'){const q=r*n;   // gates at the row start (4h); lstm_cell_grad adds the carry after them
        out[q+j]=dcn*g*i*(1-i);out[q+H+j]=dcn*cp.data[r*H+j]*f*(1-f);out[q+2*H+j]=dcn*i*(1-g*g);out[q+3*H+j]=dhn*tc*o*(1-o);
        if(op==='lstm_cell_grad'){out[q+4*H+j]=live?0:dh;out[q+5*H+j]=dcn*f+(live?0:dc);}
      }else{out[r*2*H+j]=live?0:dh;out[r*2*H+H+j]=dcn*f+(live?0:dc);}
    }
  }else if(op==='bqk'){
    const [q,k]=inputs,B=attrs.batch,d=q.shape[1];
    for(let r=0;r<m;r++){const b=r%B;for(let s=0;s<n;s++){let sum=0;const kr=(s*B+b)*d;for(let x=0;x<d;x++)sum+=q.data[r*d+x]*k.data[kr+x];out[r*n+s]=sum;}}
  }else if(op==='bpv'){
    const [p,v]=inputs,B=attrs.batch,S=p.shape[1];
    for(let r=0;r<m;r++){const b=r%B;for(let j=0;j<n;j++){let sum=0;for(let s=0;s<S;s++)sum+=p.data[r*S+s]*v.data[(s*B+b)*n+j];out[r*n+j]=sum;}}
  }else if(op==='bptq'){
    const [p,q]=inputs,B=attrs.batch,S=p.shape[1],T=p.shape[0]/B;
    for(let s=0;s<S;s++)for(let b=0;b<B;b++)for(let j=0;j<n;j++){let sum=0;
      for(let t=0;t<T;t++)sum+=p.data[(t*B+b)*S+s]*q.data[(t*B+b)*n+j];out[(s*B+b)*n+j]=sum;}
  }else if(op==='place_block'){
    inputs.forEach((x,i)=>{const [r0,c0]=attrs.offsets[i];
      for(let r=0;r<x.shape[0];r++)for(let c=0;c<x.shape[1];c++)out[(r0+r)*n+c0+c]+=x.data[r*x.shape[1]+c];});
  }else return false;
  return true;
}
export function executeCPU(op, inputs, attrs = {}, shape) {
  if(CONV_OPS.includes(op))return convolutionCPU(op,inputs,attrs,shape);
  if(op==='factorization_block'||op==='factorization_tile')return factorizationBlock(inputs,attrs,shape,op);
  const a=inputs[0], b=inputs[1], [m,n]=shape, out=new Float64Array(m*n);
  if(executeSequence(op,inputs,attrs,shape,out)) return tensor(shape,out);
  const at=(x,i,j)=>x.data[(x.shape[0]===1?0:i)*x.shape[1]+(x.shape[1]===1?0:j)];
  if(op==='matmul') {
    const k=a.shape[1]; for(let i=0;i<m;i++) for(let j=0;j<n;j++) {
      let sum=0; for(let q=0;q<k;q++) sum+=a.data[i*k+q]*b.data[q*n+j]; out[i*n+j]=sum;
    }
  } else if(op==='transpose') {
    for(let i=0;i<m;i++) for(let j=0;j<n;j++) out[i*n+j]=a.data[j*m+i];
  } else if(op==='sum'||op==='sum_squares') {
    out[0]=a.data.reduce((s,x)=>s+(op==='sum'?x:x*x),0);
  } else if(op==='reduce_rows') {
    for(let i=0;i<a.shape[0];i++) for(let j=0;j<n;j++) out[j]+=a.data[i*n+j];
  } else if(op==='add_many') {
    for(const x of inputs) for(let i=0;i<out.length;i++) out[i]+=x.data[i];
  } else if(op==='gather_rows') {
    let offset=0; for(const x of inputs){out.set(x.data,offset);offset+=x.data.length;}
  } else if(op==='embedding'||op==='embedding_backward'||op==='scatter_rows') {
    for(let i=0;i<attrs.ids.length;i++) {
      const row=attrs.ids[i]; if(op==='embedding_backward'&&row===attrs.padding_idx)continue; if(row<0||row>=(op==='embedding'?a.shape[0]:m)) continue;
      for(let j=0;j<n;j++) if(op==='embedding') out[i*n+j]=a.data[row*n+j]; else out[row*n+j]+=a.data[i*n+j];
    }
  } else if(op==='slice_columns') {
    for(let i=0;i<m;i++) for(let j=0;j<n;j++) out[i*n+j]=a.data[i*a.shape[1]+attrs.start+j];
  } else if(op==='slice_block'||op==='slice_backward') {
    if(a) for(let i=0;i<a.shape[0];i++) for(let j=0;j<a.shape[1];j++) out[i*n+(attrs.offset??attrs.start)+j]=a.data[i*a.shape[1]+j];
  } else if(op==='concat_columns') {
    let start=0;for(const x of inputs){for(let i=0;i<m;i++)out.set(x.data.subarray(i*x.shape[1],(i+1)*x.shape[1]),i*n+start);start+=x.shape[1];}
  } else if(op==='softmax'||op==='softmax_backward') {
    for(let i=0;i<m;i++){
      if(op==='softmax'){
        let max=-Infinity,total=0;
        for(let j=0;j<n;j++)if(!attrs.mask?.[i]?.[j])max=Math.max(max,a.data[i*n+j]);
        for(let j=0;j<n;j++){const v=attrs.mask?.[i]?.[j]?0:Math.exp(a.data[i*n+j]-max);out[i*n+j]=v;total+=v;}
        for(let j=0;j<n;j++)out[i*n+j]/=total;
      }else{
        let dot=0;for(let j=0;j<n;j++)dot+=a.data[i*n+j]*b.data[i*n+j];
        for(let j=0;j<n;j++)out[i*n+j]=a.data[i*n+j]*(b.data[i*n+j]-dot);
      }
    }
  } else if(op==='layer_norm'||op==='layer_norm_backward') {
    const [rows,cols]=a.shape;
    for(let i=0;i<rows;i++){
      let mean=0,variance=0;for(let j=0;j<cols;j++)mean+=a.data[i*cols+j];mean/=cols;
      for(let j=0;j<cols;j++)variance+=(a.data[i*cols+j]-mean)**2;
      const inv=1/Math.sqrt(variance/cols+attrs.epsilon);
      let meanG=0,meanGX=0;
      if(op==='layer_norm_backward'&&attrs.index===0){
        for(let j=0;j<cols;j++){const g=inputs[2].data[i*cols+j]*b.data[j];meanG+=g;meanGX+=g*(a.data[i*cols+j]-mean)*inv;}
        meanG/=cols;meanGX/=cols;
      }
      for(let j=0;j<cols;j++){
        const normalized=(a.data[i*cols+j]-mean)*inv,g=inputs[2].data[i*cols+j];
        if(op==='layer_norm')out[i*cols+j]=normalized*b.data[j]+inputs[2].data[j];
        else if(attrs.index===0)out[i*cols+j]=inv*(g*b.data[j]-meanG-normalized*meanGX);
        else out[j]+=attrs.index===1?g*normalized:g;
      }
    }
  } else if(op==='cross_entropy'||op==='ce_backward') {
    const [rows,cols]=a.shape; let loss=0;const count=attrs.count??attrs.targets.filter(x=>x!==attrs.ignore_index).length, smoothing=attrs.label_smoothing||0;
    for(let i=0;i<rows;i++) {
      let max=-Infinity; for(let j=0;j<cols;j++) max=Math.max(max,a.data[i*cols+j]);
      let total=0; for(let j=0;j<cols;j++) total+=Math.exp(a.data[i*cols+j]-max);
      if(attrs.targets[i]===attrs.ignore_index)continue;
      let average=0;for(let j=0;j<cols;j++)average+=(a.data[i*cols+j]-max)/cols;
      loss+=(1-smoothing)*(Math.log(total)-(a.data[i*cols+attrs.targets[i]]-max))+smoothing*(Math.log(total)-average);
      if(op==='ce_backward') for(let j=0;j<cols;j++) out[i*cols+j]=(Math.exp(a.data[i*cols+j]-max)/total-smoothing/cols-(j===attrs.targets[i]?1-smoothing:0))*b.data[0]/count;
    }
    if(op==='cross_entropy') out[0]=loss/count;
  } else if(op==='assemble') {
    inputs.forEach((x,k)=>{const [r,,c]=attrs.blocks[k];for(let i=0;i<x.shape[0];i++)for(let j=0;j<x.shape[1];j++)out[(r+i)*n+c+j]=x.data[i*x.shape[1]+j];});
  } else {
    const known=new Set(['inverse_norm','add','add_bias','mul','relu','relu_backward','sigmoid','tanh','sigmoid_backward','tanh_backward','scale','broadcast','clip_gradient','sgd','adam_m','adam_v','adam_update']);
    if(!known.has(op)) throw Error(`Unsupported operation: ${op}`);
    for(let i=0;i<m;i++) for(let j=0;j<n;j++) {
      const x=at(a,i,j), y=b?at(b,i,j):0; let v;
      switch(op) {
        case 'add': case 'add_bias': v=x+y;break;
        case 'mul':v=x*y;break;
        case 'relu':v=Math.max(0,x);break;
        case 'relu_backward':v=x>0?y:0;break;
        case 'sigmoid':{const z=Math.exp(-Math.abs(x));v=x>=0?1/(1+z):z/(1+z);break;}
        case 'tanh':v=Math.tanh(x);break;
        case 'sigmoid_backward':v=y*x*(1-x);break;
        case 'tanh_backward':v=y*(1-x*x);break;
        case 'inverse_norm':v=inverseNorm(x,attrs.epsilon);break;
        case 'scale':v=x*attrs.factor;break;
        case 'broadcast':v=x;break;
        case 'clip_gradient':{const norm=Math.sqrt(b.data[0]);v=x*(norm?Math.min(1,attrs.clip_norm/norm):1);break;}
        case 'sgd':v=x-attrs.learning_rate*y;break;
        case 'adam_m':v=attrs.beta1*x+(1-attrs.beta1)*y;break;
        case 'adam_v':v=attrs.beta2*x+(1-attrs.beta2)*y*y;break;
        case 'adam_update':v=x-attrs.learning_rate*(y/(1-attrs.beta1**attrs.step))/(Math.sqrt(at(inputs[2],i,j)/(1-attrs.beta2**attrs.step))+attrs.epsilon);break;
      }
      out[i*n+j]=v;
    }
  }
  if(!out.every(Number.isFinite)) throw Error('Nonfinite output');
  return tensor(shape,out);
}
