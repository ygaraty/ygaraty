package relay

import (
	"encoding/binary"
	"math"
)

// finiteFloat64s reports whether data is whole little-endian float64 values without NaN or Inf.
func finiteFloat64s(data []byte) bool {
	if len(data)%8 != 0 {
		return false
	}
	for i := 0; i < len(data); i += 8 {
		f := math.Float64frombits(binary.LittleEndian.Uint64(data[i:]))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

// finiteFloat32s reports whether data is whole little-endian float32 values without NaN or Inf.
func finiteFloat32s(data []byte) bool {
	if len(data)%4 != 0 {
		return false
	}
	for i := 0; i < len(data); i += 4 {
		f := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[i:])))
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return false
		}
	}
	return true
}

// finiteBFloat16s reports whether data is whole little-endian bf16 values (the top half of a
// float32) without NaN or Inf: an exponent field of all ones.
func finiteBFloat16s(data []byte) bool {
	if len(data)%2 != 0 {
		return false
	}
	for i := 0; i < len(data); i += 2 {
		if binary.LittleEndian.Uint16(data[i:])&0x7F80 == 0x7F80 {
			return false
		}
	}
	return true
}
