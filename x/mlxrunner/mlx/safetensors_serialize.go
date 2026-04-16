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
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"unsafe"
)

// rawDataPtr returns the unsafe pointer to the array's tensor data, or nil
// if the array has no data or the dtype is unsupported. The caller must
// keep the array alive for the duration of any use of the pointer.
func (t *Array) rawDataPtr() unsafe.Pointer {
	switch t.DType() {
	case DTypeBool:
		return unsafe.Pointer(C.mlx_array_data_bool(t.ctx))
	case DTypeUint8:
		return unsafe.Pointer(C.mlx_array_data_uint8(t.ctx))
	case DTypeUint16:
		return unsafe.Pointer(C.mlx_array_data_uint16(t.ctx))
	case DTypeUint32:
		return unsafe.Pointer(C.mlx_array_data_uint32(t.ctx))
	case DTypeUint64:
		return unsafe.Pointer(C.mlx_array_data_uint64(t.ctx))
	case DTypeInt8:
		return unsafe.Pointer(C.mlx_array_data_int8(t.ctx))
	case DTypeInt16:
		return unsafe.Pointer(C.mlx_array_data_int16(t.ctx))
	case DTypeInt32:
		return unsafe.Pointer(C.mlx_array_data_int32(t.ctx))
	case DTypeInt64:
		return unsafe.Pointer(C.mlx_array_data_int64(t.ctx))
	case DTypeFloat16:
		return C.mlx_array_data_f16_raw(t.ctx)
	case DTypeBFloat16:
		return C.mlx_array_data_bf16_raw(t.ctx)
	case DTypeFloat32:
		return unsafe.Pointer(C.mlx_array_data_float32(t.ctx))
	case DTypeFloat64:
		return unsafe.Pointer(C.mlx_array_data_float64(t.ctx))
	case DTypeComplex64:
		return unsafe.Pointer(C.mlx_array_data_complex64(t.ctx))
	}
	return nil
}

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
	ptr := t.rawDataPtr()
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

// safetensorsPlan holds the sorted tensor order, per-tensor byte sizes, and
// the marshaled header. It's the pre-data computation shared by all encode
// paths; data can then be written into a buffer or an io.Writer.
type safetensorsPlan struct {
	names      []string
	sizes      []int
	headerJSON []byte
	dataBytes  int
}

func planSafetensors(arrays map[string]*Array, metadata map[string]string) (*safetensorsPlan, error) {
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

	sizes := make([]int, len(names))
	var total int
	for i, name := range names {
		arr := arrays[name]
		n := arr.NumBytes()
		if n == 0 {
			return nil, fmt.Errorf("no data for tensor %q", name)
		}
		sizes[i] = n
		header[name] = safetensorsHeader{
			DType:       arr.DType().String(),
			Shape:       arr.Dims(),
			DataOffsets: [2]int{total, total + n},
		}
		total += n
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("marshal safetensors header: %w", err)
	}
	return &safetensorsPlan{names: names, sizes: sizes, headerJSON: headerJSON, dataBytes: total}, nil
}

// SerializeSafetensorsTo encodes evaluated arrays + metadata directly to w.
// Tensor bytes are streamed one tensor at a time without buffering the full
// output, avoiding the double-buffering overhead of SerializeSafetensors.
// Arrays must be evaluated and remain alive for the duration of the call.
func SerializeSafetensorsTo(w io.Writer, arrays map[string]*Array, metadata map[string]string) error {
	plan, err := planSafetensors(arrays, metadata)
	if err != nil {
		return err
	}

	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(plan.headerJSON)))
	if _, err := w.Write(sizeBuf[:]); err != nil {
		return fmt.Errorf("write safetensors size: %w", err)
	}
	if _, err := w.Write(plan.headerJSON); err != nil {
		return fmt.Errorf("write safetensors header: %w", err)
	}

	for i, name := range plan.names {
		arr := arrays[name]
		ptr := arr.rawDataPtr()
		if ptr == nil {
			return fmt.Errorf("no data pointer for tensor %q (dtype %s)", name, arr.DType())
		}
		view := unsafe.Slice((*byte)(ptr), plan.sizes[i])
		if _, err := w.Write(view); err != nil {
			return fmt.Errorf("write tensor %q: %w", name, err)
		}
	}
	return nil
}

// SerializeSafetensors encodes evaluated arrays and metadata into the safetensors
// binary format, returning the complete file contents as a byte slice. Allocates
// a single buffer sized to the final output; prefer SerializeSafetensorsTo for
// large tensors to avoid holding the full file in memory. All arrays must be
// evaluated before calling. The result is independent of MLX memory -- arrays
// can be freed immediately after this returns.
func SerializeSafetensors(arrays map[string]*Array, metadata map[string]string) ([]byte, error) {
	plan, err := planSafetensors(arrays, metadata)
	if err != nil {
		return nil, err
	}
	buf := bytes.NewBuffer(make([]byte, 0, 8+len(plan.headerJSON)+plan.dataBytes))
	if err := serializeWithPlan(buf, arrays, plan); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func serializeWithPlan(buf *bytes.Buffer, arrays map[string]*Array, plan *safetensorsPlan) error {
	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(plan.headerJSON)))
	buf.Write(sizeBuf[:])
	buf.Write(plan.headerJSON)
	for i, name := range plan.names {
		arr := arrays[name]
		ptr := arr.rawDataPtr()
		if ptr == nil {
			return fmt.Errorf("no data pointer for tensor %q (dtype %s)", name, arr.DType())
		}
		buf.Write(unsafe.Slice((*byte)(ptr), plan.sizes[i]))
	}
	return nil
}
