// Binary frames are little-endian float64 starting on an 8-byte boundary, so a
// little-endian host reads and writes them in place; others convert value by value.
// A frame whose header says wire '<f4' carries float32 (a job with wire 'float32'): half the
// bytes, widened to float64 on arrival, and results go back as float32.
export const LITTLE_ENDIAN = new Uint8Array(new Float64Array([1]).buffer)[7] === 0x3f;
export const WIRE_BYTES={'<f8':8,'<f4':4};
export const readFloats = (buffer, offset, n, wire='<f8') => {
  if (wire === '<f4') {
    if (LITTLE_ENDIAN && offset % 4 === 0) return Float64Array.from(new Float32Array(buffer, offset, n));
    const view = new DataView(buffer), values = new Float64Array(n);
    for (let i = 0; i < n; i++) values[i] = view.getFloat32(offset + 4*i, true);
    return values;
  }
  if (LITTLE_ENDIAN && offset % 8 === 0) return new Float64Array(buffer, offset, n);
  const view = new DataView(buffer), values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = view.getFloat64(offset + 8*i, true);
  return values;
};
export const frameWire=header=>{const wire=header.wire??'<f8';if(!(wire in WIRE_BYTES))throw Error('Frame wire must be <f8 or <f4');return wire;};
