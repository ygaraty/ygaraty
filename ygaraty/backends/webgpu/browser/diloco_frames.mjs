// Model-independent native DiLoCo frames; numeric batch schema version 1.
const MAX_BYTES = 64 * 1024 * 1024;
const TYPES = new Set(['float64', 'float32', 'int64', 'int32', 'uint8']);
function sizeOf(shape) {
  if (!Array.isArray(shape) || !shape.every(n => Number.isSafeInteger(n) && n > 0))
    throw Error('Invalid tensor shape');
  const n = shape.reduce((a, b) => a * b, 1);
  if (!Number.isSafeInteger(n) || n * 8 > MAX_BYTES) throw Error('Tensor exceeds frame limit');
  return n;
}
export function decodeArrayBatches(header, reader) {
  const schema = header.recipe?.batch_schema, fields = schema?.fields;
  if (schema?.version !== 1 || !Array.isArray(fields) || !fields.length ||
      fields.some(f => typeof f?.name !== 'string' || !f.name || !TYPES.has(f.dtype)) ||
      new Set(fields.map(f => f.name)).size !== fields.length)
    throw Error('Expected numeric batch schema version 1');
  return header.batches.map(shapes => {
    if (!Array.isArray(shapes) || shapes.length !== fields.length) throw Error('Batch field count mismatch');
    return Object.fromEntries(fields.map((field, i) =>
      [field.name, reader.tensor(shapes[i], field.dtype)]));
  });
}
export function decodeRound(buffer, {decodeBatches = decodeArrayBatches} = {}) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4 || buffer.byteLength > MAX_BYTES)
    throw Error('Invalid frame size');
  const view = new DataView(buffer), length = view.getUint32(0, true);
  if (length > buffer.byteLength - 4) throw Error('Truncated header');
  const h = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(new Uint8Array(buffer, 4, length)));
  if (h?.state !== 'round' || typeof h.round_id !== 'string' || !h.round_id ||
      !Number.isSafeInteger(h.version) || h.version < 0 ||
      !Number.isSafeInteger(h.steps) || h.steps < 1 ||
      !['float64', 'float32'].includes(h.wire_dtype) ||
      !['float64', 'float32'].includes(h.upload_dtype ?? h.wire_dtype) ||
      h.wire != null || !Array.isArray(h.weights) || !h.weights.length ||
      !Array.isArray(h.batches) || h.batches.length !== h.steps)
    throw Error('Invalid round header');
  const names = h.weights.map(w => w?.name);
  if (names.some(n => typeof n !== 'string' || !n) || new Set(names).size !== names.length)
    throw Error('Invalid parameter names');
  const elements = h.weights.reduce((n, w) => {
    if (w.shape?.length !== 2) throw Error('Wire model tensors must have rank two');
    return n + sizeOf(w.shape);
  }, 0);
  if (elements * 8 > MAX_BYTES ||
      (h.wire_dtype === 'float32' && h.diloco_float32_values !== elements) ||
      (h.wire_dtype === 'float64' && h.diloco_float32_values != null))
    throw Error('Model encoding mismatch');
  let offset = 4 + length;
  function read(width) {
    if (offset + width > buffer.byteLength) throw Error('Truncated frame');
    const n = width === 4 ? view.getFloat32(offset, true) : view.getFloat64(offset, true);
    offset += width;
    if (!Number.isFinite(n)) throw Error('Nonfinite value');
    return n;
  }
  const weights = Object.create(null);
  for (const {name, shape} of h.weights) {
    const data = new Float64Array(sizeOf(shape));
    for (let i = 0; i < data.length; i++) data[i] = read(h.wire_dtype === 'float32' ? 4 : 8);
    weights[name] = {shape, data};
  }
  const reader = {
    bytes(count) {
      if (!Number.isSafeInteger(count) || count < 0 || offset + count > buffer.byteLength)
        throw Error('Truncated batch bytes');
      const out = new Uint8Array(buffer.slice(offset, offset + count));
      offset += count; return out;
    },
    tensor(shape, dtype = 'float64') {
      if (!TYPES.has(dtype)) throw Error('Unknown field dtype');
      const count = sizeOf(shape);
      if (offset + count * 8 > buffer.byteLength) throw Error('Truncated batch field');
      const data = dtype === 'float32' ? new Float32Array(count)
        : dtype === 'int32' ? new Int32Array(count) : dtype === 'uint8' ? new Uint8Array(count)
        : new Float64Array(count); // int64 is restricted to exact JS integers.
      for (let i = 0; i < count; i++) {
        const n = read(8);
        if ((dtype === 'int64' && !Number.isSafeInteger(n)) ||
            (dtype === 'int32' && (!Number.isInteger(n) || n < -2147483648 || n > 2147483647)) ||
            (dtype === 'uint8' && (!Number.isInteger(n) || n < 0 || n > 255)) ||
            (dtype === 'float32' && !Number.isFinite(Math.fround(n))))
          throw Error('Field value outside dtype range');
        data[i] = n;
      }
      return {shape, dtype, data};
    },
  };
  const batches = decodeBatches(h, reader);
  if (!Array.isArray(batches) || batches.length !== h.steps || offset !== buffer.byteLength)
    throw Error('Batch count mismatch or trailing frame bytes');
  return {header: h, weights, batches};
}
export function encodeResult(result, dtype = 'float32') {
  if (!['float32', 'float64'].includes(dtype)) throw Error('Invalid upload dtype');
  const names = Object.keys(result.pseudo_gradient).sort();
  if (!names.length || !result.losses.every(Number.isFinite)) throw Error('Invalid result');
  const tensors = names.map(name => {
    const t = result.pseudo_gradient[name];
    if (t.shape.length !== 2 || !t.shape.every(n => Number.isSafeInteger(n) && n > 0) ||
        t.shape[0] * t.shape[1] !== t.data.length) throw Error('Invalid result tensor');
    return {name, shape: t.shape};
  });
  const count = names.reduce((n, k) => n + result.pseudo_gradient[k].data.length, 0);
  const header = {...result, pseudo_gradient: tensors};
  if (dtype === 'float32') header.diloco_float32_values = count;
  const bytes = new TextEncoder().encode(JSON.stringify(header));
  const size = bytes.length + ((8 - (4 + bytes.length) % 8) % 8), width = dtype === 'float32' ? 4 : 8;
  if (4 + size + count * width > MAX_BYTES) throw Error('Result exceeds adapter limit');
  const out = new ArrayBuffer(4 + size + count * width), view = new DataView(out);
  view.setUint32(0, size, true);
  new Uint8Array(out, 4, size).fill(32);
  new Uint8Array(out, 4, bytes.length).set(bytes);
  let offset = 4 + size;
  for (const name of names) for (const n of result.pseudo_gradient[name].data) {
    if (!Number.isFinite(n) || (width === 4 && !Number.isFinite(Math.fround(n))))
      throw Error('Nonfinite result or wire overflow');
    if (width === 4) view.setFloat32(offset, n, true);
    else view.setFloat64(offset, n, true);
    offset += width;
  }
  return out;
}
