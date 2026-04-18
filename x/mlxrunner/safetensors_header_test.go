package mlxrunner

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/safetensors"
)

// writeFakeSafetensors writes a synthetic safetensors file with the given
// __metadata__ map. Used for pure-Go header-parser tests that don't need
// MLX to be available.
func writeFakeSafetensors(t *testing.T, dir string, meta map[string]string) string {
	t.Helper()
	blob := map[string]any{"__metadata__": meta}
	header, err := json.Marshal(blob)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	path := filepath.Join(dir, "fake.safetensors")
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

func TestParseSafetensorsMetadata(t *testing.T) {
	dir := t.TempDir()
	want := map[string]string{
		"cache_format_version": "2",
		"model_digest":         "m",
		"tokens":               "abc",
	}
	path := writeFakeSafetensors(t, dir, want)

	got, err := parseSafetensorsMetadata(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s: got %q, want %q", k, got[k], v)
		}
	}
}

func TestParseSafetensorsMetadataMalformed(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name    string
		content []byte
		wantErr string
	}{
		{"empty", nil, "read header length"},
		{"truncated-header-len", []byte{1, 2, 3}, "read header length"},
		{"zero-length-header", binary.LittleEndian.AppendUint64(nil, 0), "zero-length"},
		{"header-too-large", binary.LittleEndian.AppendUint64(nil, safetensors.MaxHeaderBytes+1), "exceeds"},
		{"truncated-header", append(binary.LittleEndian.AppendUint64(nil, 100), byte('{')), "read header"},
		{"bad-json", append(binary.LittleEndian.AppendUint64(nil, 5), []byte("notj")...), "read header"},
		{"no-metadata", append(
			binary.LittleEndian.AppendUint64(nil, uint64(len(`{"other":1}`))),
			[]byte(`{"other":1}`)...,
		), "missing __metadata__"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(dir, tc.name)
			if err := os.WriteFile(path, tc.content, 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := parseSafetensorsMetadata(path)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}
