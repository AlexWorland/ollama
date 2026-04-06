package mlx

// #include "generated.h"
// // Wrappers needed because cgo cannot handle __fp16 / __bf16 return types.
// static const void* mlx_array_data_f16_raw(mlx_array a) {
//     return (const void*)mlx_array_data_float16(a);
// }
// static const void* mlx_array_data_bf16_raw(mlx_array a) {
//     return (const void*)mlx_array_data_bfloat16(a);
// }
import "C"

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
	"unsafe"
)

// RawData returns a copy of the array's raw tensor data in Go-managed memory.
// The array must be evaluated (via Eval) before calling this. Returns nil if
// the array is invalid or has no data.
func (t *Array) RawData() []byte {
	if !t.Valid() {
		return nil
	}
	nbytes := t.NumBytes()
	if nbytes == 0 {
		return nil
	}
	var ptr unsafe.Pointer
	switch t.DType() {
	case DTypeBool:
		ptr = unsafe.Pointer(C.mlx_array_data_bool(t.ctx))
	case DTypeUint8:
		ptr = unsafe.Pointer(C.mlx_array_data_uint8(t.ctx))
	case DTypeUint16:
		ptr = unsafe.Pointer(C.mlx_array_data_uint16(t.ctx))
	case DTypeUint32:
		ptr = unsafe.Pointer(C.mlx_array_data_uint32(t.ctx))
	case DTypeUint64:
		ptr = unsafe.Pointer(C.mlx_array_data_uint64(t.ctx))
	case DTypeInt8:
		ptr = unsafe.Pointer(C.mlx_array_data_int8(t.ctx))
	case DTypeInt16:
		ptr = unsafe.Pointer(C.mlx_array_data_int16(t.ctx))
	case DTypeInt32:
		ptr = unsafe.Pointer(C.mlx_array_data_int32(t.ctx))
	case DTypeInt64:
		ptr = unsafe.Pointer(C.mlx_array_data_int64(t.ctx))
	case DTypeFloat16:
		ptr = C.mlx_array_data_f16_raw(t.ctx)
	case DTypeBFloat16:
		ptr = C.mlx_array_data_bf16_raw(t.ctx)
	case DTypeFloat32:
		ptr = unsafe.Pointer(C.mlx_array_data_float32(t.ctx))
	case DTypeFloat64:
		ptr = unsafe.Pointer(C.mlx_array_data_float64(t.ctx))
	case DTypeComplex64:
		ptr = unsafe.Pointer(C.mlx_array_data_complex64(t.ctx))
	default:
		return nil
	}
	if ptr == nil {
		return nil
	}
	return C.GoBytes(ptr, C.int(nbytes))
}

type safetensorsHeader struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// SerializeSafetensors encodes evaluated arrays and metadata into the safetensors
// binary format, returning the complete file contents as a byte slice. All arrays
// must be evaluated before calling this function. The result is independent of
// MLX memory -- arrays can be freed immediately after this returns.
func SerializeSafetensors(arrays map[string]*Array, metadata map[string]string) ([]byte, error) {
	names := make([]string, 0, len(arrays))
	for name, arr := range arrays {
		if arr != nil {
			names = append(names, name)
		}
	}
	sort.Strings(names)

	header := make(map[string]any, len(names)+1)
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}

	var dataChunks [][]byte
	var totalDataBytes int
	for _, name := range names {
		arr := arrays[name]
		raw := arr.RawData()
		if raw == nil {
			return nil, fmt.Errorf("no data for tensor %q", name)
		}
		header[name] = safetensorsHeader{
			DType:       arr.DType().String(),
			Shape:       arr.Dims(),
			DataOffsets: [2]int{totalDataBytes, totalDataBytes + len(raw)},
		}
		dataChunks = append(dataChunks, raw)
		totalDataBytes += len(raw)
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("marshal safetensors header: %w", err)
	}

	buf := make([]byte, 8+len(headerJSON)+totalDataBytes)
	binary.LittleEndian.PutUint64(buf[:8], uint64(len(headerJSON)))
	copy(buf[8:], headerJSON)
	pos := 8 + len(headerJSON)
	for _, chunk := range dataChunks {
		copy(buf[pos:], chunk)
		pos += len(chunk)
	}

	return buf, nil
}
