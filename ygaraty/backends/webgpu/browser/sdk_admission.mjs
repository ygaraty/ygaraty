import {taskBytes} from './budget.mjs';
// A task's memory admission on a job that enforces budgets (the task carries memory_bytes): this
// worker's own count (budget.mjs) must agree with the coordinator's and fit its budget, checked
// before any input is copied or GPU buffer created. Returns null, or why the task is refused.
export function admitTask(work,budget){
  const task=work?.task;
  if(task?.memory_bytes===undefined)return null;
  if(!Number.isSafeInteger(budget)||budget<=0)return 'this worker has no memory budget';
  let bytes;
  try{bytes=taskBytes(task.op,(work.inputs||[]).map(x=>x?.shape),task.shape,task.attrs);}catch(e){return 'refused before allocating: '+e.message;}
  if(bytes!==task.memory_bytes)return `refused before allocating: memory estimate ${bytes} differs from the coordinator's ${task.memory_bytes}`;
  if(bytes>budget)return `refused before allocating: needs ${bytes} bytes, budget ${budget}`;
  return null;
}
// The default advertised budget: a quarter of what the adapter allows in one buffer, at most 1 GiB.
export const defaultBudget=limits=>limits?Math.min(Math.floor(Math.min(limits.maxBufferSize,limits.maxStorageBufferBindingSize)/4),2**30):null;
