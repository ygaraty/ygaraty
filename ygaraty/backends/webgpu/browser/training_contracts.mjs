export const momentKinds = ['m', 'v'];
export function object(value) {
    if (typeof value !== 'object' || value === null || Array.isArray(value))
        throw Error('Expected object');
    return value;
}
const text = (value) => {
    if (typeof value !== 'string')
        throw Error('Expected string');
    return value;
};
const integer = (value, minimum = 0) => {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum)
        throw Error('Expected integer');
    return value;
};
function array(value, decode) {
    if (!Array.isArray(value))
        throw Error('Expected array');
    return value.map(item => decode(item));
}
function named(value, decode) {
    return Object.fromEntries(Object.entries(object(value)).map(([key, item]) => [key, decode(item)]));
}
export function decodeTensor(value) {
    const raw = object(value), shape = array(raw.shape, n => integer(n, 1));
    if (shape.length !== 2)
        throw Error('Expected rank-two tensor');
    if (raw.dtype === 'float32')
        return { shape, dtype: 'float32', base64: text(raw.base64) };
    if (raw.dtype === 'uint8' && raw.normalize === 'x/127.5-1')
        return { shape, dtype: 'uint8', normalize: raw.normalize, base64: text(raw.base64) };
    if (raw.dtype !== undefined)
        throw Error('Unknown tensor encoding');
    const data = array(raw.data, x => {
        if (typeof x !== 'number' || !Number.isFinite(x))
            throw Error('Nonfinite tensor');
        return x;
    });
    if (data.length !== shape[0] * shape[1])
        throw Error('Invalid tensor size');
    return { shape, data };
}
function optimizer(value) {
    const raw = object(value), step = integer(raw.step);
    if (raw.moments === null && step === 0)
        return { step, moments: null };
    const moments = object(raw.moments);
    return { step, moments: { m: named(moments.m, decodeTensor), v: named(moments.v, decodeTensor) } };
}
function task(value) {
    const raw = object(value);
    return { id: text(raw.id), op: text(raw.op), shape: array(raw.shape, n => integer(n, 1)),
        attrs: object(raw.attrs), inputs: array(raw.inputs, item => {
            const input = object(item);
            return { value: text(input.value), region: input.region == null ? null : array(input.region, integer) };
        }) };
}
export function decodeStatus(value) {
    const raw = object(value), state = raw.state;
    if (state !== 'waiting' && state !== 'training' && state !== 'submitted' && state !== 'complete' && state !== 'failed')
        throw Error('Invalid training session state');
    return { state, ...(raw.version === undefined ? {} : { version: integer(raw.version) }),
        ...(raw.rounds === undefined ? {} : { rounds: integer(raw.rounds) }),
        ...(raw.connected === undefined ? {} : { connected: integer(raw.connected) }),
        ...(raw.required === undefined ? {} : { required: integer(raw.required) }),
        ...(raw.retries === undefined ? {} : { retries: integer(raw.retries) }),
        ...(raw.pending === undefined ? {} : { pending: array(raw.pending, text) }),
        ...(raw.error == null ? {} : { error: text(raw.error) }),
        ...(raw.metrics === undefined ? {} : { metrics: array(raw.metrics, item => item) }) };
}
export function decodeJoin(value) {
    const raw = object(value);
    if ('state' in raw)
        return decodeStatus(raw);
    if ((raw.precision !== 'float64' && raw.precision !== 'webgpu') ||
        typeof raw.heartbeat_seconds !== 'number' || !Number.isFinite(raw.heartbeat_seconds) || raw.heartbeat_seconds <= 0)
        throw Error('Invalid contributor credentials');
    return { worker: text(raw.worker), token: text(raw.token), precision: raw.precision, heartbeat_seconds: raw.heartbeat_seconds };
}
export function decodeAssignment(value) {
    const raw = object(value);
    if (raw.state !== 'assignment')
        return decodeStatus(raw);
    if (raw.format !== 'ygaraty-diloco-round-v1')
        throw Error('Unknown round format');
    const a = object(raw.assignment);
    if (a.wire_dtype !== undefined && a.wire_dtype !== 'float32')
        throw Error('Unknown wire dtype');
    return { state: 'assignment', format: raw.format, optimizer: optimizer(raw.optimizer),
        assignment: { round_id: text(a.round_id), version: integer(a.version), worker: text(a.worker),
            steps: integer(a.steps, 1), weights: named(a.weights, decodeTensor), wire_dtype: a.wire_dtype },
        plans: array(raw.plans, decodeStepPlan) };
}
export function decodeStepPlan(value) {
    const p = object(value);
    return { step: integer(p.step, 1), loss: text(p.loss), tasks: array(p.tasks, task),
        constants: named(p.constants, decodeTensor), parameters: named(p.parameters, text), updates: named(p.updates, text),
        ...(p.state_parameters === undefined ? {} : { state_parameters: {
                m: named(object(p.state_parameters).m, text), v: named(object(p.state_parameters).v, text)
            } }) };
}
