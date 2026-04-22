package mlx

// #include "generated.h"
import "C"

// RoPEWithFreqs applies rotary position embeddings with optional custom
// frequencies and per-token positions.
//
// When freqs is non-nil, it is used instead of the base-derived inverse
// frequencies. Note: MLX takes reciprocal(freqs) internally to get inv_freq,
// so pass the actual frequencies (base^(2i/dim)), not the inverse.
//
// For contiguous positions (single sequence), this dispatches to the scalar
// mlx_fast_rope — bit-identical to the old offset-based API. For non-contiguous
// positions (multi-sequence), it dispatches to mlx_fast_rope_dynamic.
func RoPEWithFreqs(x *Array, dims int, traditional bool, base, scale float32, positions *Array, freqs *Array) *Array {
	posData := positions.Ints()
	if len(posData) == 0 {
		return x
	}

	offset := posData[0]
	contiguous := true
	for i := 1; i < len(posData); i++ {
		if posData[i] != posData[i-1]+1 {
			contiguous = false
			break
		}
	}

	var freqsCtx C.mlx_array
	var optBase C.mlx_optional_float
	if freqs != nil {
		freqsCtx = freqs.ctx
		optBase = C.mlx_optional_float{has_value: C.bool(false)}
	} else {
		empty := New("")
		freqsCtx = empty.ctx
		optBase = C.mlx_optional_float{
			value:     C.float(base),
			has_value: C.bool(base != 0),
		}
	}

	if contiguous {
		out := New("FAST_ROPE")
		C.mlx_fast_rope(
			&out.ctx,
			x.ctx,
			C.int(dims),
			C.bool(traditional),
			optBase,
			C.float(scale),
			C.int(offset),
			freqsCtx,
			DefaultStream().ctx,
		)
		return out
	}

	rotIn := Transpose(x, 2, 1, 0, 3)
	rotOut := New("FAST_ROPE_DYNAMIC")
	C.mlx_fast_rope_dynamic(
		&rotOut.ctx,
		rotIn.ctx,
		C.int(dims),
		C.bool(traditional),
		optBase,
		C.float(scale),
		positions.ctx,
		freqsCtx,
		DefaultStream().ctx,
	)
	return Transpose(rotOut, 2, 1, 0, 3)
}
