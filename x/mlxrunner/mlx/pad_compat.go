package mlx

// #include "generated.h"
import "C"

import "unsafe"

// PadConstant pads array a with zeros along the specified axes.
// Subset of axes supported (unlike theirs' Pad which expects entries for all axes).
// Re-expresses the pre-batching API for upstream callers.
func PadConstant(a *Array, axes []int, lowPad, highPad []int) *Array {
	cAxes := make([]C.int, len(axes))
	cLow := make([]C.int, len(lowPad))
	cHigh := make([]C.int, len(highPad))
	for i := range axes {
		cAxes[i] = C.int(axes[i])
		cLow[i] = C.int(lowPad[i])
		cHigh[i] = C.int(highPad[i])
	}

	padValue := C.mlx_array_new_float(C.float(0))
	defer C.mlx_array_free(padValue)

	cMode := C.CString("constant")
	defer C.free(unsafe.Pointer(cMode))

	out := New("PAD")
	C.mlx_pad(
		&out.ctx,
		a.ctx,
		unsafe.SliceData(cAxes), C.size_t(len(cAxes)),
		unsafe.SliceData(cLow), C.size_t(len(cLow)),
		unsafe.SliceData(cHigh), C.size_t(len(cHigh)),
		padValue,
		cMode,
		DefaultStream().ctx,
	)
	return out
}
