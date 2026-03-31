package cache

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestExportImportKVSnapshot(t *testing.T) {
	skipIfNoMLX(t)

	c := NewKVCache()
	for range 10 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	snap := c.Snapshot(3)
	if snap == nil {
		t.Fatal("expected snapshot, got nil")
	}
	defer snap.Close()

	exp := ExportSnapshot(snap)
	if exp == nil {
		t.Fatal("ExportSnapshot returned nil")
	}
	if exp.Type != SnapshotTypeKV {
		t.Fatalf("expected type kv, got %s", exp.Type)
	}
	if exp.Arrays["keys"] == nil || exp.Arrays["values"] == nil {
		t.Fatal("missing keys or values in export")
	}
	if exp.Metadata["from_offset"] != "3" {
		t.Fatalf("expected from_offset=3, got %s", exp.Metadata["from_offset"])
	}
	if exp.Metadata["to_offset"] != "10" {
		t.Fatalf("expected to_offset=10, got %s", exp.Metadata["to_offset"])
	}

	imported, err := ImportSnapshot(SnapshotTypeKV, exp.Arrays, exp.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	defer imported.Close()

	if imported.Size() != snap.Size() {
		t.Fatalf("size mismatch: original=%d, imported=%d", snap.Size(), imported.Size())
	}
}

func TestExportImportRotatingSnapshot(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRotatingKVCache(8)
	for range 5 {
		k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
		c.Update(k, v)
	}

	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("expected snapshot, got nil")
	}
	defer snap.Close()

	exp := ExportSnapshot(snap)
	if exp == nil {
		t.Fatal("ExportSnapshot returned nil")
	}
	if exp.Type != SnapshotTypeRotating {
		t.Fatalf("expected type rotating, got %s", exp.Type)
	}
	if _, ok := exp.Metadata["idx"]; !ok {
		t.Fatal("rotating snapshot export missing idx metadata")
	}

	imported, err := ImportSnapshot(SnapshotTypeRotating, exp.Arrays, exp.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	defer imported.Close()

	if imported.Size() != snap.Size() {
		t.Fatalf("size mismatch: original=%d, imported=%d", snap.Size(), imported.Size())
	}
}

func TestExportImportRecurrentSnapshot(t *testing.T) {
	skipIfNoMLX(t)

	c := NewRecurrentCache(3, 12, 4, 8, 8)
	// Initialize state by calling ConvState/DeltaState which triggers ensure().
	c.ConvState(1, mlx.DTypeFloat16)
	c.DeltaState(1, mlx.DTypeFloat16)
	c.Advance(5)

	snap := c.Snapshot(0)
	if snap == nil {
		t.Fatal("expected snapshot, got nil")
	}
	defer snap.Close()

	exp := ExportSnapshot(snap)
	if exp == nil {
		t.Fatal("ExportSnapshot returned nil")
	}
	if exp.Type != SnapshotTypeRecurrent {
		t.Fatalf("expected type recurrent, got %s", exp.Type)
	}
	if exp.Metadata["offset"] != "5" {
		t.Fatalf("expected offset=5, got %s", exp.Metadata["offset"])
	}
	if exp.Arrays["conv_state"] == nil || exp.Arrays["delta_state"] == nil {
		t.Fatal("missing conv_state or delta_state in export")
	}

	imported, err := ImportSnapshot(SnapshotTypeRecurrent, exp.Arrays, exp.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	defer imported.Close()

	if imported.Size() != snap.Size() {
		t.Fatalf("size mismatch: original=%d, imported=%d", snap.Size(), imported.Size())
	}
}

func TestExportNilSnapshot(t *testing.T) {
	if exp := ExportSnapshot(nil); exp != nil {
		t.Fatal("expected nil export for nil snapshot")
	}
}

func TestImportUnknownType(t *testing.T) {
	_, err := ImportSnapshot("bogus", nil, nil)
	if err == nil {
		t.Fatal("expected error for unknown type")
	}
}

func TestImportKVMissingArrays(t *testing.T) {
	_, err := ImportSnapshot(SnapshotTypeKV, map[string]*mlx.Array{}, map[string]string{
		"from_offset": "0",
		"to_offset":   "5",
	})
	if err == nil {
		t.Fatal("expected error for missing arrays")
	}
}

func TestTypeOf(t *testing.T) {
	skipIfNoMLX(t)

	kv := NewKVCache()
	k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
	kv.Update(k, v)
	snap := kv.Snapshot(0)
	defer snap.Close()

	if got := TypeOf(snap); got != SnapshotTypeKV {
		t.Fatalf("expected kv, got %s", got)
	}
	if got := TypeOf(nil); got != "" {
		t.Fatalf("expected empty, got %s", got)
	}
}
