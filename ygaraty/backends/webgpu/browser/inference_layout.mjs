export const DEFAULT_LAYOUT_GEOMETRY = { heads: 16, kvHeads: 4, linearV: 32, conv: 8192, value: 4096, kv: 1024 };
function object(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        throw Error('Invalid buffer descriptor');
    return value;
}
function integer(value, minimum = 0) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > 0xffffffff)
        throw Error('Invalid layout dimension or offset');
    return value;
}
export function checkBuffer(value, bytes) {
    const binding = object(value), buffer = binding.buffer ? object(binding.buffer) : binding;
    const allocation = integer(buffer.size, 1), offset = integer(binding.buffer ? (binding.offset ?? 0) : 0);
    const size = integer(binding.buffer ? (binding.size ?? allocation - offset) : allocation, 1);
    if (offset % 4 || size % 4 || !Number.isSafeInteger(bytes) || bytes <= 0 || offset + size > allocation || bytes > size)
        throw Error(`Buffer layout outside allocation: need ${bytes} bytes, view ${size} at ${offset}, allocation ${allocation}`);
}
function floats(value, elements) { checkBuffer(value, elements * 4); }
function matrix(value) {
    const m = object(value), rows = integer(m.rows, 1), cols = integer(m.cols, 1), rowBytes = integer(m.rowBytes, 1);
    // *_KR: the exact repacks of experiments 141/142. A row need not be 4-byte aligned (a Q6_K row of 768 columns is
    // 630 bytes, Qwen3-Coder's down experts): the decoders read unaligned words; uploads pad the whole buffer.
    const formats = { F32: [1, 4], F16: [1, 2], Q4_K: [256, 144], Q5_K: [256, 176], Q6_K: [256, 210], Q8_0: [32, 34],
        Q4_KR: [256, 192], Q5_KR: [256, 224], Q6_KR: [256, 256] };
    const format = formats[String(m.kind)];
    if (!format || cols % format[0] || rowBytes !== cols / format[0] * format[1])
        throw Error('Invalid packed matrix row layout');
    checkBuffer(m.buffer, rows * rowBytes);
    return m;
}
function chunks(value) {
    if (!Array.isArray(value) || !value.length)
        throw Error('Missing packed weight chunks');
    return value.map(v => { matrix(v); return object(v); });
}
function span(rows, width, stride, offset = 0) {
    integer(rows, 1);
    integer(width, 1);
    integer(stride);
    integer(offset);
    return (rows - 1) * stride + offset + width;
}
function attention(a, gated, g) {
    const shift = gated ? 1 : 0, T = Number(a[4 + shift]), position = Number(a[5 + shift]), capacity = Number(a[6 + shift]);
    const H = gated ? g.heads : Number(a[8]), KV = gated ? g.kvHeads : Number(a[9]), D = gated ? 256 : Number(a[10]);
    if (H % KV || D % 2)
        throw Error('Invalid attention geometry');
    const splits = Math.ceil((position + T) / 512), bytes = a[7 + shift] === 'f16' ? 2 : 4;
    floats(a[0], T * H * D);
    floats(a[3 + shift], T * H * D);
    // Cache is head-major, with capacity as the head stride.
    checkBuffer(a[1], KV * capacity * D * bytes);
    checkBuffer(a[2], KV * capacity * D * bytes);
    if (gated)
        floats(a[3], T * H * D * 2);
    floats(a[gated ? 9 : 12], T * splits * H * D);
    floats(a[gated ? 10 : 13], T * splits * H * 2);
}
export function validateBufferLayout(op, a, g = DEFAULT_LAYOUT_GEOMETRY) {
    const n = (i) => Number(a[i]);
    switch (op) {
        case 'embedding': {
            const list = chunks(a[0]), width = Number(list[0].cols);
            let end = 0;
            for (const c of list) {
                if (integer(c.first) !== end || c.cols !== width)
                    throw Error('Invalid embedding chunk layout');
                end += Number(c.rows);
            }
            floats(a[1], n(3));
            floats(a[2], n(3) * width);
            return;
        }
        case 'matmul': {
            const w = matrix(a[0]), o = object(a[4] ?? {}), rows = integer(o.rows ?? w.rows, 1), T = n(3);
            if (rows > w.rows)
                throw Error('Matrix row limit exceeds weights');
            floats(a[1], span(T, w.cols, Number(o.xStride ?? w.cols), Number(o.xOffset ?? 0)));
            floats(a[2], span(T, rows, Number(o.yStride ?? rows), Number(o.yOffset ?? 0) + Number(o.rowBase ?? 0)));
            return;
        }
        case 'rms_norm': {
            const o = object(a[4]), width = integer(o.size, 1), groups = integer(o.groups ?? 1, 1), T = n(3);
            const extent = (prefix, gate = false) => span(T, (groups - 1) * integer(o[prefix + 'Group'] || width) + width, integer(o[prefix + 'Stride'] ?? (gate ? 0 : width * groups)), integer(o[prefix + 'Offset'] ?? 0));
            floats(a[0], extent('in'));
            floats(a[1], width);
            floats(a[2], extent('out'));
            if (o.gate)
                floats(o.gate, extent('gate', true));
            return;
        }
        case 'silu_mul':
        case 'add':
            floats(a[0], n(2));
            floats(a[1], n(2));
            return;
        case 'argmax':
            floats(a[0], n(2));
            checkBuffer(a[1], 8);
            return;
        case 'causal_conv1d':
            floats(a[0], n(4) * g.conv);
            floats(a[1], 4 * g.conv);
            floats(a[2], 3 * g.conv);
            floats(a[3], n(4) * g.conv);
            return;
        case 'gated_delta':
            floats(a[0], n(5) * g.conv);
            floats(a[1], n(5) * 2 * g.linearV);
            floats(a[2], 2 * g.linearV);
            floats(a[3], g.linearV * 128 * 128);
            floats(a[4], n(5) * g.value);
            return;
        case 'rotary_partial':
            if (n(5) < 64 || n(4) < n(3) * n(5))
                throw Error('Invalid rotary strides');
            floats(a[0], span(n(2), (n(3) - 1) * n(5) + 64, n(4)));
            floats(a[1], 32);
            return;
        case 'rotary_adjacent':
        case 'rotary_halves':
            if (n(4) % 2)
                throw Error('Rotary width must be even');
            floats(a[0], n(2) * n(3) * n(4));
            floats(a[1], n(4) / 2);
            return;
        case 'cache_store':
        case 'cache_store_grouped': {
            const width = op === 'cache_store' ? g.kv : n(6) * n(7);
            if (width % 2)
                throw Error('Invalid packed cache width');
            floats(a[0], n(2) * width);
            checkBuffer(a[1], n(4) * width * (a[5] === 'f16' ? 2 : 4));
            return;
        }
        case 'grouped_attention':
        case 'windowed_attention':
            attention(a, false, g);
            return;
        case 'gated_attention':
            attention(a, true, g);
            return;
        case 'moe_route':
            if (n(6) > n(5))
                throw Error('top_k exceeds experts');
            floats(a[0], n(4) * n(5));
            floats(a[1], n(5));
            floats(a[2], n(4) * n(6));
            floats(a[3], n(4) * n(6));
            return;
        case 'moe_route_softmax': // DO-80: (logits, ids, weights, tokens, experts, top_k, normalize)
            if (n(5) > n(4))
                throw Error('top_k exceeds experts');
            floats(a[0], n(3) * n(4));
            floats(a[1], n(3) * n(5));
            floats(a[2], n(3) * n(5));
            return;
        case 'moe_matmul': {
            const list = chunks(a[0]);
            let end = 0, rows = 0, cols = 0;
            for (const c of list) {
                const count = integer(c.count, 1), width = Number(c.rows) / count;
                integer(width, 1);
                if (integer(c.first) !== end || (end && (rows !== width || cols !== c.cols)))
                    throw Error('Invalid expert chunks');
                end += count;
                rows = width;
                cols = Number(c.cols);
            }
            floats(a[1], n(4) * (a[6] ? n(5) : 1) * cols);
            floats(a[2], n(4) * n(5) * rows);
            floats(a[3], n(4) * n(5));
            return;
        }
        case 'silu_mul_halves':
            floats(a[0], n(2) * n(3) * 2);
            floats(a[1], n(2) * n(3));
            return;
        case 'moe_combine':
            floats(a[0], n(3) * n(4) * n(5));
            floats(a[1], n(3) * n(4));
            floats(a[2], n(3) * n(5));
            return;
        default: throw Error(`Missing buffer layout for ${op}`);
    }
}
