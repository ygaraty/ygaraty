// Operation arithmetic is distinct from wire/storage precision and CPU fallback.
import type { Tensor } from './sdk_tensor.mjs';
export type { Tensor } from './sdk_tensor.mjs';
export type DenseArithmetic = 'float32' | 'compensated';
export const CONV_OPS = ['conv2d','conv2d_input_grad','conv2d_weight_grad',
  'conv2d_bias_grad','maxpool2d','maxpool2d_grad','avgpool2d','avgpool2d_grad'] as const;
export type ConvolutionOp = typeof CONV_OPS[number];
export interface ConvolutionAttrs {
  batch: number; in_channels: number; height: number; width: number;
  out_channels?: number; kernel?: number;
}
export interface LowRankSpec {
  kind: 'lowrank_x'; rows: number; columns: number; latent: number;
  row_offset: number; column_offset: number; seed: number;
  shape: number[]; noise: number;
}
export interface FactorizationAttrs { x_spec: LowRankSpec; total_elements: number }
export interface FactorizationGroupAttrs extends FactorizationAttrs {
  row_shards: number[][]; column_shards: number[][];
}
export interface Timings { upload_ms: number; gpu_ms: number; tiles?: number }
export interface FactorizationResult { value: Tensor; timings: Timings }
export type KernelRequest =
  | { op: 'matmul'; arithmetic: DenseArithmetic; inputs: Tensor[]; shape: number[] }
  | { op: ConvolutionOp; arithmetic: 'float32'; inputs: Tensor[]; shape: number[]; attrs: ConvolutionAttrs }
  | { op: 'factorization_tile'; arithmetic: 'kahan-f32'; inputs: Tensor[]; shape: number[]; attrs: FactorizationAttrs }
  | { op: 'factorization_tiles'; arithmetic: 'kahan-f32'; inputs: Tensor[]; shape: number[]; attrs: FactorizationGroupAttrs };
export const PRECISION_CONTRACTS = {
  matmul: { arithmetic: ['float32', 'compensated'], output: 'host-f64', reduction: 'ordered' },
  convolution: { arithmetic: ['float32'], output: 'host-f64', reduction: 'ordered' },
  factorization: { arithmetic: ['kahan-f32'], output: 'host-f64', reduction: 'host-f64 loss and shard accumulation' },
  inference: { arithmetic: ['float32'], output: 'device-f32', reduction: 'kernel-specific' },
} as const;

/** GPU-resident inference primitives. Caller owns buffers and submission. */
export type ResidentKernelRequest = { arithmetic: 'float32' } & (
  | { op: 'packed_matmul'; weight: import('./webgpu_kernels.mjs').PackedMatrix; input: GPUBuffer; output: GPUBuffer; tokens: number; options?: import('./webgpu_kernels.mjs').MatmulOptions }
  | { op: 'rms_norm'; input: GPUBuffer; weight: GPUBuffer; output: GPUBuffer; tokens: number; options: import('./webgpu_kernels.mjs').NormOptions }
  | { op: 'embedding'; chunks: import('./webgpu_kernels.mjs').EmbeddingChunk[]; ids: GPUBuffer; output: GPUBuffer; tokens: number }
  | { op: 'silu_mul'; input: GPUBuffer; output: GPUBuffer; elements: number }
  | { op: 'argmax'; input: GPUBuffer; output: GPUBuffer; elements: number }
);
