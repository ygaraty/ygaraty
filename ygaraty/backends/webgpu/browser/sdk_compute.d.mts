// Typed boundary for the existing JavaScript SDK implementation.
// Keep this declaration aligned with sdk_compute.mjs; training depends only on
// the ContributorSDK interface and does not bypass kernel runtime validation.
import type {ContributorSDK, AdapterInfo, Task, Precision, ResidentTrainingRequest, ResidentTrainingResult} from './training_contracts.mjs';
import type {Tensor, DenseArithmetic} from './webgpu_contracts.mjs';
export class ComputeSDK implements ContributorSDK {
  device: GPUDevice | null;
  reason: string;
  adapterInfo: AdapterInfo | null;
  initialize(preferGPU?: boolean): Promise<string>;
  matmul(a: Tensor, b: Tensor, shape: number[], arithmetic?: DenseArithmetic): Promise<Tensor>;
  executeTrainingRound(request: ResidentTrainingRequest, arithmetic: DenseArithmetic): Promise<ResidentTrainingResult | null>;
  executeMuonBlock(tasks: Task[], inputs: Map<string, Tensor>, outputs: string[], arithmetic: DenseArithmetic): Promise<Map<string, Tensor> | null>;
  execute(task: Task & {precision: Precision}, inputs: Tensor[]): Promise<{value: Tensor; backend: 'cpu' | 'webgpu'}>;
}
