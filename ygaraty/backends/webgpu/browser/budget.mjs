// The bytes one task may make this worker allocate (experiment 48's memory budget): the same
// function as ygaraty_cloud/runtime/budget.py, which documents each term; tests/test_memory_budget.py
// checks that both give the same number. Throws for an op without a bound or a malformed task.
export const ELEMENTWISE=['add','add_many','scale','sgd','sum_squares'];
export const BUDGET_OPS=[...ELEMENTWISE,'factorization_tile','factorization_tiles'];
const PARAMS_BYTES=64;
const shapeOf=value=>{
  if(!Array.isArray(value)||value.length!==2||!value.every(n=>Number.isSafeInteger(n)&&n>0&&n<=2**31))
    throw Error('shapes must be two positive integers');
  return value;
};
export function taskBytes(op,inputShapes,shape,attrs={}){
  if(!BUDGET_OPS.includes(op))throw Error(`no memory bound for ${op}`);
  if(!Array.isArray(inputShapes))throw Error('input shapes must be a list');
  const inputs=inputShapes.reduce((n,s)=>{const [r,c]=shapeOf(s);return n+r*c;},0);
  const [outRows,rank]=shapeOf(shape),out=outRows*rank,total=8*inputs+8*out;
  if(ELEMENTWISE.includes(op))return total+8*out;
  let bytes;
  if(op==='factorization_tiles')
    bytes=total+Math.max(...groupTiles(inputShapes,attrs).map(([rows,cols,latent])=>tileExtra(rows,cols,latent,rank)+8*(rows+cols+1)*rank));
  else{
    const spec=attrs?.x_spec;
    if(inputShapes.length!==4||!spec||typeof spec!=='object')
      throw Error('a factorization tile is bounded only on the GPU path: four inputs and an x_spec');
    const [rows,cols]=shapeOf(spec.shape);
    bytes=total+tileExtra(rows,cols,shapeOf(inputShapes[0])[1],rank);
  }
  if(!Number.isSafeInteger(bytes))throw Error('task size is out of range');
  return bytes;
}
// The GPU tile's own allocations beyond its float64 inputs and result (budget.py _tile_extra).
const tileExtra=(rows,cols,latent,rank)=>{const gpuOut=4*((rows+cols)*rank+rows),inputs=(rows+cols)*(latent+rank);return 2*4*inputs+4*rows*cols+3*gpuOut+PARAMS_BYTES;};
const bounds=v=>Array.isArray(v)&&v.length>=1&&v.length<=64&&v.every(b=>Array.isArray(b)&&b.length===2&&b.every(Number.isSafeInteger)&&b[0]>=0&&b[0]<b[1]);
// [rows, cols, latent] of every tile of a factorization_tiles task, from its shards (budget.py _group_tiles).
export function groupTiles(inputShapes,attrs={}){
  const rs=attrs?.row_shards,cs=attrs?.column_shards;
  if(!bounds(rs)||!bounds(cs)||!attrs?.x_spec||typeof attrs.x_spec!=='object')throw Error('a tile group needs row_shards, column_shards and an x_spec');
  const p=rs.length,q=cs.length;
  if(inputShapes.length!==2*(p+q))throw Error('a tile group is bounded only on the GPU path: latent rows, U and V for each shard');
  const latent=shapeOf(inputShapes[0])[1];
  [...rs,...cs].forEach((b,k)=>{if(shapeOf(inputShapes[k])[0]!==b[1]-b[0]||shapeOf(inputShapes[p+q+k])[0]!==b[1]-b[0])throw Error("a tile group's inputs must match its shards");});
  return rs.flatMap(r=>cs.map(c=>[r[1]-r[0],c[1]-c[0],latent]));
}
