package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeSafetensorsHeader writes a fake safetensors file whose header is
// the JSON-encoded `blob`. Used to exercise ReadMetadata across well-formed
// and malformed inputs without needing real model files.
func writeSafetensorsHeader(t *testing.T, dir, name string, blob any) string {
	t.Helper()
	header, err := json.Marshal(blob)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer f.Close()
	if err := binary.Write(f, binary.LittleEndian, uint64(len(header))); err != nil {
		t.Fatalf("write header len: %v", err)
	}
	if _, err := f.Write(header); err != nil {
		t.Fatalf("write header: %v", err)
	}
	return path
}

func TestReadMetadataHappy(t *testing.T) {
	dir := t.TempDir()
	want := map[string]string{"a": "1", "b": "2"}
	path := writeSafetensorsHeader(t, dir, "ok.safetensors", map[string]any{
		"__metadata__": want,
	})
	got, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s: got %q, want %q", k, got[k], v)
		}
	}
}

func TestReadMetadataMissingMetadataReturnsEmpty(t *testing.T) {
	dir := t.TempDir()
	path := writeSafetensorsHeader(t, dir, "nometa.safetensors", map[string]any{
		"some_tensor": map[string]any{"dtype": "F32"},
	})
	got, err := ReadMetadata(path)
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected empty map, got %v", got)
	}
}

func TestReadMetadataMalformed(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name    string
		content []byte
		wantErr string
	}{
		{"empty-file", nil, "read header length"},
		{"zero-length-header", binary.LittleEndian.AppendUint64(nil, 0), "zero-length"},
		{"header-too-large", binary.LittleEndian.AppendUint64(nil, MaxHeaderBytes+1), "exceeds"},
		{"truncated-header", append(binary.LittleEndian.AppendUint64(nil, 100), byte('{')), "read header"},
		{"bad-json", append(binary.LittleEndian.AppendUint64(nil, 5), []byte("notjs")...), "parse header"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name)
			if err := os.WriteFile(path, tc.content, 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := ReadMetadata(path)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}
