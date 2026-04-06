package mlx

import (
	"os"
	"testing"
)

func TestRawDataFloat16(t *testing.T) {
	skipIfNoMLX(t)

	arr := Zeros(DTypeFloat16, 2, 2)
	Eval(arr)

	raw := arr.RawData()
	if raw == nil {
		t.Fatal("RawData returned nil")
	}
	// float16: 2 bytes/element, 2x2 = 4 elements = 8 bytes.
	if len(raw) != 8 {
		t.Fatalf("expected 8 bytes, got %d", len(raw))
	}
	for i, b := range raw {
		if b != 0 {
			t.Fatalf("byte %d: expected 0, got %d", i, b)
		}
	}
}

func TestRawDataInvalidArray(t *testing.T) {
	skipIfNoMLX(t)
	arr := New("empty")
	if arr.RawData() != nil {
		t.Fatal("expected nil for invalid array")
	}
}

func TestSerializeSafetensorsRoundTrip(t *testing.T) {
	skipIfNoMLX(t)

	a := Zeros(DTypeFloat16, 1, 4, 1, 8)
	b := Zeros(DTypeFloat16, 1, 4, 1, 8)
	Eval(a, b)

	arrays := map[string]*Array{"keys": a, "values": b}
	metadata := map[string]string{"from_offset": "0", "to_offset": "5"}

	// Serialize with pure Go encoder.
	data, err := SerializeSafetensors(arrays, metadata)
	if err != nil {
		t.Fatal("SerializeSafetensors:", err)
	}
	if len(data) == 0 {
		t.Fatal("empty serialized data")
	}

	// Write to temp file and load with MLX C library to verify compatibility.
	path := t.TempDir() + "/test.safetensors"
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal("write:", err)
	}

	sf, err := LoadSafetensorsNative(path)
	if err != nil {
		t.Fatal("LoadSafetensorsNative:", err)
	}
	defer sf.Free()

	loadedKeys := sf.Get("keys")
	if loadedKeys == nil {
		t.Fatal("keys not found in loaded file")
	}
	if loadedKeys.DType() != DTypeFloat16 {
		t.Fatalf("keys dtype: got %v, want F16", loadedKeys.DType())
	}
	dims := loadedKeys.Dims()
	if len(dims) != 4 || dims[0] != 1 || dims[1] != 4 || dims[2] != 1 || dims[3] != 8 {
		t.Fatalf("keys shape mismatch: %v", dims)
	}

	if sf.GetMetadata("from_offset") != "0" {
		t.Fatalf("from_offset: got %q", sf.GetMetadata("from_offset"))
	}
	if sf.GetMetadata("to_offset") != "5" {
		t.Fatalf("to_offset: got %q", sf.GetMetadata("to_offset"))
	}
}

func TestSerializeSafetensorsEmptyArrays(t *testing.T) {
	skipIfNoMLX(t)
	_, err := SerializeSafetensors(nil, nil)
	if err != nil {
		t.Fatal("should succeed with empty input:", err)
	}
}
