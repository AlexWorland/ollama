package mlxrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestDiskWriterWritesFile(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter()
	defer w.shutdown()

	data := []byte("hello safetensors content")
	w.submit(diskWriteJob{
		data:     data,
		filename: "test.safetensors",
		dir:      dir,
	})

	result := <-w.results
	if result.err != nil {
		t.Fatal("write failed:", result.err)
	}
	if result.fileSize != int64(len(data)) {
		t.Fatalf("fileSize: got %d, want %d", result.fileSize, len(data))
	}

	got, err := os.ReadFile(filepath.Join(dir, "test.safetensors"))
	if err != nil {
		t.Fatal("read file:", err)
	}
	if string(got) != string(data) {
		t.Fatal("file contents mismatch")
	}
}

func TestDiskWriterFailure(t *testing.T) {
	w := newDiskWriter()
	defer w.shutdown()

	w.submit(diskWriteJob{
		data:     []byte("data"),
		filename: "test.safetensors",
		dir:      "/nonexistent/path/that/does/not/exist",
	})

	result := <-w.results
	if result.err == nil {
		t.Fatal("expected error for bad directory")
	}
}

func TestDiskWriterWaitForFile(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter()
	defer w.shutdown()

	w.submit(diskWriteJob{
		data:     []byte("content"),
		filename: "wait.safetensors",
		dir:      dir,
	})

	w.waitForFile("wait.safetensors")

	if _, err := os.Stat(filepath.Join(dir, "wait.safetensors")); err != nil {
		t.Fatal("file should exist after waitForFile:", err)
	}

	<-w.results // drain
}

func TestDiskWriterWaitForFileNotInFlight(t *testing.T) {
	w := newDiskWriter()
	defer w.shutdown()

	// Should return immediately for unknown files.
	w.waitForFile("nonexistent.safetensors")
}

func TestDiskWriterShutdown(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter()

	for i := range 5 {
		w.submit(diskWriteJob{
			data:     []byte(fmt.Sprintf("data_%d", i)),
			filename: fmt.Sprintf("file_%d.safetensors", i),
			dir:      dir,
		})
	}

	w.shutdown()

	for i := range 5 {
		path := filepath.Join(dir, fmt.Sprintf("file_%d.safetensors", i))
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("file_%d should exist after shutdown: %v", i, err)
		}
	}

	// Drain all results.
	for range 5 {
		<-w.results
	}
}
