// Generic TensorFlow.js replica. Model, variables, optimizer and loss are local.
const shapeSize = shape => {
  if (!Array.isArray(shape) || !shape.every(n => Number.isSafeInteger(n) && n > 0))
    throw Error('Invalid parameter shape');
  const n = shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(n) || n < 1) throw Error('Invalid parameter size');
  return n;
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const variableValue = v => typeof v.assign === 'function' ? v : v.read();

export class TensorFlowBackend {
  constructor(tf, worker, {model, variables, optimizer, loss, prepareBatch = b => b,
    validateBatch = () => {}, wireShapes, recipe, bufferPolicy = 'reject', disposeResources,
    captureState, restoreState} = {}) {
    if (!['reject', 'local'].includes(bufferPolicy)) throw Error('Unknown buffer policy');
    if (model?.nonTrainableWeights?.length && bufferPolicy === 'reject')
      throw Error('Non-trainable model state requires explicit local policy');
    if (Boolean(captureState) !== Boolean(restoreState)) throw Error('Supply both state hooks');
    this.captureState = captureState; this.restoreState = restoreState;
    this.tf = tf; this.worker = worker; this.model = model;
    this.variables = {...(variables ?? Object.fromEntries(
      (model?.trainableWeights ?? []).map(v => [v.originalName, v])))};
    this.names = Object.keys(this.variables).sort();
    if (!this.names.length || this.names.some(n => !n) || typeof loss !== 'function' || !optimizer)
      throw Error('Expected variables, optimizer and scalar loss callback');
    const values = this.names.map(n => variableValue(this.variables[n]));
    if (new Set(values).size !== values.length || values.some(v =>
      v.dtype !== 'float32' || typeof v.assign !== 'function'))
      throw Error('Expected unique float32 variables');
    if (model?.trainableWeights && (values.length !== model.trainableWeights.length ||
        model.trainableWeights.some(v => !values.includes(variableValue(v)))))
      throw Error('Variable mapping must cover model trainable weights');
    this.nativeShapes = Object.fromEntries(this.names.map(n => [n, [...this.variables[n].shape]]));
    this.manifest = Object.fromEntries(this.names.map(n =>
      [n, [...(wireShapes?.[n] ?? [1, shapeSize(this.nativeShapes[n])])]]));
    if ((wireShapes && !same(Object.keys(wireShapes).sort(), this.names)) ||
        this.names.some(n => this.manifest[n].length !== 2 ||
          shapeSize(this.manifest[n]) !== shapeSize(this.nativeShapes[n])))
      throw Error('Invalid wire parameter layout');
    this.optimizer = optimizer; this.loss = loss; this.prepareBatch = prepareBatch;
    this.validateBatch = validateBatch; this.recipe = JSON.stringify(recipe);
    this.disposeResources = disposeResources ?? (() => {
      optimizer.dispose();
      if (model?.dispose) model.dispose();
      else for (const v of Object.values(this.variables)) v.dispose();
    });
    this.version = -1; this.seen = new Set();
    this.busy = false; this.failed = false; this.disposed = false;
  }
  async exportWeights() {
    const weights = Object.create(null);
    for (const name of this.names) {
      const data = Float64Array.from(await variableValue(this.variables[name]).data());
      if (data.length !== shapeSize(this.manifest[name]) || !data.every(Number.isFinite))
        throw Error('Invalid model state');
      weights[name] = {shape: [...this.manifest[name]], data};
    }
    return weights;
  }
  async runRound({header, weights, batches}) {
    if (this.busy || this.failed || this.disposed) throw Error('Backend unavailable');
    if (JSON.stringify(header.recipe) !== this.recipe ||
        !Number.isSafeInteger(header.version) || header.version <= this.version ||
        typeof header.round_id !== 'string' || !header.round_id || this.seen.has(header.round_id) ||
        !Number.isSafeInteger(header.steps) || header.steps < 1 || batches.length !== header.steps)
      throw Error('Invalid or repeated assignment');
    if (Object.keys(weights).length !== this.names.length || this.names.some(n =>
      !Object.hasOwn(weights, n) || !same(weights[n].shape, this.manifest[n]) ||
      !same(this.variables[n].shape, this.nativeShapes[n]) ||
      weights[n].data.length !== shapeSize(this.manifest[n]) ||
      !weights[n].data.every(x => Number.isFinite(x) && Number.isFinite(Math.fround(x)))))
      throw Error('Invalid model parameters');
    for (const batch of batches) this.validateBatch(batch);
    const base = Object.fromEntries(this.names.map(n => [n, Float64Array.from(weights[n].data)]));
    this.busy = true;
    const tf = this.tf;
    try {
      for (const name of this.names) tf.tidy(() => {
        const value = tf.tensor(Float32Array.from(base[name]), this.nativeShapes[name], 'float32');
        const variable = this.variables[name];
        if (typeof variable.assign === 'function') variable.assign(value);
        else variable.write(value);
      });
      const losses = [];
      for (const batch of batches) {
        const loss = tf.tidy(() => this.optimizer.minimize(() => {
          const value = this.loss(this.prepareBatch(batch), this.model);
          if (value.rank !== 0) throw Error('Loss must be a scalar tensor');
          return value;
        }, true, this.names.map(n => variableValue(this.variables[n])).filter(v => v.trainable !== false)));
        try {
          const value = (await loss.data())[0];
          if (!Number.isFinite(value)) throw Error('Nonfinite loss');
          losses.push(value);
        } finally { loss.dispose(); }
      }
      const after = await this.exportWeights(), delta = Object.create(null);
      for (const name of this.names) {
        const data = Float64Array.from(after[name].data, (v, i) => base[name][i] - v);
        if (!data.every(Number.isFinite)) throw Error('Nonfinite delta');
        delta[name] = {shape: [...this.manifest[name]], data};
      }
      this.version = header.version; this.seen.add(header.round_id);
      return {worker: this.worker, round_id: header.round_id, version: header.version,
        steps: header.steps, losses, pseudo_gradient: delta};
    } catch (error) { this.failed = true; throw error; }
    finally { this.busy = false; }
  }
  async checkpoint() {
    if (this.busy || this.failed || this.disposed) throw Error('Backend unavailable');
    const describe = async tensor => ({shape: [...tensor.shape], dtype: tensor.dtype,
      data: Array.from(await tensor.data())});
    const slots = await this.optimizer.getWeights();
    try {
      const state = {format: 'ygaraty.tfjs-state/v1', worker: this.worker,
        recipe: this.recipe, names: this.names, nativeShapes: this.nativeShapes,
        optimizer: this.optimizer.getClassName(), config: this.optimizer.getConfig(),
        weights: await Promise.all(this.names.map(n => describe(variableValue(this.variables[n])))),
        buffers: await Promise.all((this.model?.nonTrainableWeights ?? []).map(v => describe(variableValue(v)))),
        slots: await Promise.all(slots.map(async v => ({name: v.name, tensor: await describe(v.tensor)}))),
        version: this.version, seen: [...this.seen], extra: this.captureState ? await this.captureState() : null};
      const json = JSON.stringify(state, (_key, value) => {
        if (typeof value === 'number' && !Number.isFinite(value)) throw Error('Nonfinite checkpoint');
        return value;
      });
      if (new TextEncoder().encode(json).length > 16 * 1024 * 1024 - 4096) throw Error('Checkpoint exceeds 16 MiB');
      return JSON.parse(json);
    } finally {
      // Standard optimizers return borrowed slot variables and an owned iteration scalar.
      for (const v of slots) if (v.name === 'iter') v.tensor.dispose();
    }
  }
  async restore(state) {
    if (this.busy || this.disposed || this.version !== -1) throw Error('Restore requires a fresh backend');
    const buffers = this.model?.nonTrainableWeights ?? [];
    if (state.format !== 'ygaraty.tfjs-state/v1' || state.worker !== this.worker ||
        state.recipe !== this.recipe || !same(state.names, this.names) ||
        !same(state.nativeShapes, this.nativeShapes) ||
        state.optimizer !== this.optimizer.getClassName() || !same(state.config, this.optimizer.getConfig()) ||
        state.weights.length !== this.names.length || state.buffers.length !== buffers.length)
      throw Error('Checkpoint backend identity mismatch');
    const tensors = [];
    const tensor = spec => {
      if (!['float32', 'int32', 'bool'].includes(spec.dtype) ||
          spec.data.length !== shapeSize(spec.shape) || !spec.data.every(Number.isFinite))
        throw Error('Invalid checkpoint tensor');
      const value = this.tf.tensor(spec.data, spec.shape, spec.dtype); tensors.push(value); return value;
    };
    try {
      const weights = state.weights.map(tensor), local = state.buffers.map(tensor);
      const slots = state.slots.map(v => ({name: v.name, tensor: tensor(v.tensor)}));
      const assign = (v, value) => typeof v.assign === 'function' ? v.assign(value) : v.write(value);
      this.names.forEach((n, i) => assign(this.variables[n], weights[i]));
      buffers.forEach((v, i) => assign(v, local[i]));
      await this.optimizer.setWeights(slots);
      if (this.restoreState) await this.restoreState(state.extra);
      else if (state.extra !== null) throw Error('Missing application restore hook');
      this.version = state.version; this.seen = new Set(state.seen); this.failed = false;
    } catch (error) { this.failed = true; throw error; }
    finally { tensors.forEach(v => v.dispose()); }
  }
  dispose() {
    if (this.busy) throw Error('Cannot dispose an active round');
    if (this.disposed) return;
    this.disposed = true; this.disposeResources();
  }
}
