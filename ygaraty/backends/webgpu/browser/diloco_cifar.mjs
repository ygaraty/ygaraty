// CIFAR example recipe and legacy packed-pixel decoding, outside generic APIs.
import {decodeRound} from './diloco_frames.mjs';
import {TensorFlowBackend} from './diloco_tfjs.mjs';

export function mlpManifest(recipe) {
  const c = recipe?.config;
  if (recipe?.kind !== 'mlp' || recipe.pixels !== 'packed-u8' ||
      c?.inputs !== 3072 || c.classes !== 10 || !Array.isArray(c.hidden) ||
      !c.hidden.every(n => Number.isSafeInteger(n) && n > 0) ||
      !Number.isFinite(recipe.learning_rate) || recipe.learning_rate <= 0)
    throw Error('Expected packed CIFAR MLP recipe');
  const sizes = [3072, ...c.hidden, 10], manifest = {};
  let elements = 0;
  for (let i = 0; i < sizes.length - 1; i++) {
    const [a, b] = sizes.slice(i, i + 2);
    elements += a * b + b;
    if (!Number.isSafeInteger(elements) || elements * 8 > 64 * 1024 * 1024)
      throw Error('Model exceeds adapter limit');
    manifest[`layer${i}.weight`] = [a, b];
    manifest[`layer${i}.bias`] = [1, b];
  }
  return manifest;
}
export function decodeCifarRound(buffer) {
  return decodeRound(buffer, {decodeBatches(header, reader) {
    const manifest = mlpManifest(header.recipe);
    if (header.weights.length !== Object.keys(manifest).length || header.weights.some(w =>
      JSON.stringify(w.shape) !== JSON.stringify(manifest[w.name]))) throw Error('Parameter manifest mismatch');
    return header.batches.map(shapes => {
      const n = shapes?.[1]?.[0];
      if (!Number.isSafeInteger(n) || n < 1 ||
          JSON.stringify(shapes) !== JSON.stringify([[n, 384], [n]]))
        throw Error('Invalid CIFAR batch layout');
      const images = reader.bytes(n * 3072);
      const labels = reader.tensor([n], 'int32').data;
      if (!labels.every(v => v >= 0 && v < 10)) throw Error('Invalid label');
      return {images, labels, count: n};
    });
  }});
}
export class TensorFlowMLPBackend extends TensorFlowBackend {
  constructor(tf, worker, recipe) {
    const manifest = mlpManifest(recipe), layers = recipe.config.hidden.length + 1;
    const variables = Object.fromEntries(Object.entries(manifest).map(([name, shape]) =>
      [name, tf.tidy(() => tf.variable(tf.tensor2d(new Float32Array(shape[0] * shape[1]), shape)))]));
    const optimizer = tf.train.adam(recipe.learning_rate, 0.9, 0.999, 1e-8);
    super(tf, worker, {variables, optimizer, wireShapes: manifest, recipe,
      validateBatch(b) {
        if (!Number.isSafeInteger(b.count) || b.count < 1 ||
            !(b.images instanceof Uint8Array) || b.images.length !== b.count * 3072 ||
            b.labels.length !== b.count || !b.labels.every(n => Number.isInteger(n) && n >= 0 && n < 10))
          throw Error('Invalid CIFAR batch');
      },
      loss(batch) {
        const pixels = tf.tensor2d(Float32Array.from(batch.images), [batch.count, 3072]);
        const mean = tf.tensor1d([0.4914, 0.4822, 0.4465]).reshape([1, 3, 1]);
        const std = tf.tensor1d([0.2470, 0.2435, 0.2616]).reshape([1, 3, 1]);
        let x = pixels.reshape([batch.count, 3, 1024]).div(255).sub(mean).div(std).reshape([batch.count, 3072]);
        const labels = tf.oneHot(tf.tensor1d(batch.labels, 'int32'), 10);
        for (let i = 0; i < layers; i++) {
          x = x.matMul(variables[`layer${i}.weight`]).add(variables[`layer${i}.bias`]);
          if (i < layers - 1) x = x.relu();
        }
        return tf.losses.softmaxCrossEntropy(labels, x).mean();
      },
    });
  }
}
