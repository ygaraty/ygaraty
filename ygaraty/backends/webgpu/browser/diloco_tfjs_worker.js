// Dedicated classic Web Worker. Dependencies are served locally, never by CDN.
let running = false;
self.onmessage = async ({data}) => {
  if (running) return;
  running = true;
  try {
    if (!['cpu', 'webgl'].includes(data.backend)) throw Error('Unsupported TF.js backend');
    importScripts('/tf.min.js');
    if (self.tf.version.tfjs !== '4.22.0') throw Error('Expected TensorFlow.js 4.22.0');
    if (!await self.tf.setBackend(data.backend)) throw Error('Requested backend unavailable');
    await self.tf.ready();
    const {TensorFlowMLPBackend, decodeCifarRound} = await import('./diloco_cifar.mjs');
    const {runReplica} = await import('./diloco_client.mjs');
    await runReplica({decode: decodeCifarRound, origin: self.location.origin, job: data.job, token: data.token,
      worker: data.worker,
      createBackend: (worker, recipe) => new TensorFlowMLPBackend(self.tf, worker, recipe),
      onProgress: progress => self.postMessage(progress)});
  } catch (error) {
    self.postMessage({state: 'error', message: error.message});
  } finally { self.close(); }
};
