// Generated output: sdk_tensor.mjs. Host tensors retain float64 storage.
export interface Tensor { shape: number[]; data: Float64Array<ArrayBuffer> }
export const tensor = (shape: number[], data: ArrayLike<number>): Tensor =>
  ({shape, data: Float64Array.from(data)});
