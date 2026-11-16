export const CONV_OPS = ['conv2d', 'conv2d_input_grad', 'conv2d_weight_grad',
    'conv2d_bias_grad', 'maxpool2d', 'maxpool2d_grad', 'avgpool2d', 'avgpool2d_grad'];
export const PRECISION_CONTRACTS = {
    matmul: { arithmetic: ['float32', 'compensated'], output: 'host-f64', reduction: 'ordered' },
    convolution: { arithmetic: ['float32'], output: 'host-f64', reduction: 'ordered' },
    factorization: { arithmetic: ['kahan-f32'], output: 'host-f64', reduction: 'host-f64 loss and shard accumulation' },
    inference: { arithmetic: ['float32'], output: 'device-f32', reduction: 'kernel-specific' },
};
