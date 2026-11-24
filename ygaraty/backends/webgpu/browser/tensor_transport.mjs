// YGT1 binary frames; interoperates with training/tensor_transport.py.
import { tensor } from './sdk_tensor.mjs';
import { object } from './training_contracts.mjs';
export const MAX_FRAME = 64 * 1024 * 1024;
const MAX_HEADER = 4 * 1024 * 1024;
export const CONTENT_TYPE = 'application/vnd.ygaraty.tensor';
export async function digest(data) {
    return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), byte => byte.toString(16).padStart(2, '0')).join('');
}
export async function encodeFrame(metadata, tensors = {}) {
    const entries = Object.entries(tensors);
    let size = 0;
    for (const [name, value] of entries) {
        if (!name || value.shape.length !== 2 || value.shape.some(n => !Number.isSafeInteger(n) || n < 1) ||
            value.shape[0] * value.shape[1] !== value.data.length || value.data.some(n => !Number.isFinite(n)))
            throw Error('Invalid tensor');
        size += value.data.length * 8;
        if (size > MAX_FRAME)
            throw Error('Tensor frame too large');
    }
    const payload = new Uint8Array(size), view = new DataView(payload.buffer);
    let offset = 0;
    for (const [, value] of entries)
        for (const n of value.data) {
            view.setFloat64(offset, n, true);
            offset += 8;
        }
    const header = new TextEncoder().encode(JSON.stringify({ metadata,
        tensors: entries.map(([name, value]) => ({ name, shape: value.shape, dtype: 'f64le' })), sha256: await digest(payload) }));
    if (header.length > MAX_HEADER || 8 + header.length + size > MAX_FRAME)
        throw Error('Tensor frame too large');
    const result = new Uint8Array(8 + header.length + size);
    result.set([89, 71, 84, 49]);
    new DataView(result.buffer).setUint32(4, header.length, true);
    result.set(header, 8);
    result.set(payload, 8 + header.length);
    return result;
}
export async function decodeFrame(bytes) {
    if (bytes.length < 8 || bytes.length > MAX_FRAME || bytes.slice(0, 4).join() !== '89,71,84,49')
        throw Error('Invalid tensor frame');
    const size = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, true);
    if (size < 1 || size > MAX_HEADER || 8 + size > bytes.length)
        throw Error('Invalid tensor header');
    const header = object(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes.slice(8, 8 + size))));
    const payload = bytes.slice(8 + size);
    if (await digest(payload) !== header.sha256)
        throw Error('Tensor checksum mismatch');
    if (!Array.isArray(header.tensors))
        throw Error('Invalid tensor index');
    const tensors = Object.create(null);
    let offset = 0;
    const view = new DataView(payload.buffer);
    for (const raw of header.tensors) {
        const entry = object(raw), shape = entry.shape;
        if (typeof entry.name !== 'string' || !entry.name || Object.hasOwn(tensors, entry.name) || entry.dtype !== 'f64le' ||
            !Array.isArray(shape) || shape.length !== 2 || shape.some(n => typeof n !== 'number' || !Number.isSafeInteger(n) || n < 1))
            throw Error('Invalid tensor descriptor');
        const dimensions = shape, count = dimensions[0] * dimensions[1];
        if (!Number.isSafeInteger(count) || offset + count * 8 > payload.length)
            throw Error('Truncated tensor');
        const values = new Float64Array(count);
        for (let i = 0; i < count; i++, offset += 8) {
            values[i] = view.getFloat64(offset, true);
            if (!Number.isFinite(values[i]))
                throw Error('Nonfinite tensor');
        }
        tensors[entry.name] = tensor(dimensions, values);
    }
    if (offset !== payload.length)
        throw Error('Trailing tensor bytes');
    return { metadata: object(header.metadata), tensors };
}
export async function readFrameBody(body) {
    if (!body)
        throw Error('Missing tensor body');
    const reader = body.getReader(), chunks = [];
    let size = 0;
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done)
                break;
            size += value.length;
            if (size > MAX_FRAME)
                throw Error('Tensor frame too large');
            chunks.push(value);
        }
    }
    catch (error) {
        await reader.cancel();
        throw error;
    }
    finally {
        reader.releaseLock();
    }
    const result = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
        result.set(chunk, offset);
        offset += chunk.length;
    }
    return result;
}
