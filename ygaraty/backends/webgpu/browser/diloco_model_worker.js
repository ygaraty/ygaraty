// The submitting operator selects model assets. Round data cannot select code.
let running = false;
self.onmessage = async ({data}) => {
  if (running) return;
  running = true;
  try {
    if (!/^[a-f0-9]{32}$/.test(data.job) || !['cpu', 'webgl'].includes(data.backend))
      throw Error('Invalid job or execution backend');
    const prefix = `/diloco-user/${data.job}/`;
    const response = await fetch(prefix + 'entry.json', {redirect: 'error'});
    if (!response.ok) throw Error('This job has no browser model entry');
    const entry = await response.json();
    const moduleURL = new URL(entry.module, self.location.origin);
    if (moduleURL.origin !== self.location.origin || !moduleURL.pathname.startsWith(prefix) ||
        moduleURL.search || moduleURL.hash || !moduleURL.pathname.endsWith('.mjs'))
      throw Error('Invalid local model entry');
    importScripts('/tf.min.js');
    if (entry.tfjs_version !== '4.22.0' || self.tf.version.tfjs !== entry.tfjs_version)
      throw Error('Expected TensorFlow.js 4.22.0');
    if (!await self.tf.setBackend(data.backend)) throw Error('Backend unavailable');
    await self.tf.ready();
    const model = await import(moduleURL.href);
    if (typeof model.createBackend !== 'function') throw Error('Model must export createBackend');
    const {runReplica} = await import('/diloco_client.mjs');
    await runReplica({origin: self.location.origin, job: data.job, token: data.token,
      worker: data.worker,
      createBackend: (worker, recipe, weights) => model.createBackend(self.tf, worker, recipe, weights),
      onProgress: progress => self.postMessage(progress)});
  } catch (error) { self.postMessage({state: 'error', message: error.message}); }
  finally { self.close(); }
};
