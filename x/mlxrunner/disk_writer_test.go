package mlxrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// waitForResults polls drainResults until at least n completions are seen
// or the deadline is exceeded. Accumulates results across poll rounds.
func waitForResults(t *testing.T, w *diskWriter, n int) []diskWriteResult {
	t.Helper()
	var all []diskWriteResult
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		all = append(all, w.drainResults()...)
		if len(all) >= n {
			return all
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d results, got %d", n, len(all))
	return nil
}

func TestDiskWriterWritesFile(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter(dir)
	defer w.shutdown()

	data := []byte("hello safetensors content")
	w.submit(diskWriteJob{
		data:     data,
		filename: "test.safetensors",
	})

	results := waitForResults(t, w, 1)
	if results[0].err != nil {
		t.Fatal("write failed:", results[0].err)
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
	w := newDiskWriter("/nonexistent/path/that/does/not/exist")
	defer w.shutdown()

	w.submit(diskWriteJob{
		data:     []byte("data"),
		filename: "test.safetensors",
	})

	results := waitForResults(t, w, 1)
	if results[0].err == nil {
		t.Fatal("expected error for bad directory")
	}
}

func TestDiskWriterWaitForFile(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter(dir)
	defer w.shutdown()

	w.submit(diskWriteJob{
		data:     []byte("content"),
		filename: "wait.safetensors",
	})

	w.waitForFile("wait.safetensors")

	if _, err := os.Stat(filepath.Join(dir, "wait.safetensors")); err != nil {
		t.Fatal("file should exist after waitForFile:", err)
	}
}

func TestDiskWriterWaitForFileNotInFlight(t *testing.T) {
	w := newDiskWriter(t.TempDir())
	defer w.shutdown()

	w.waitForFile("nonexistent.safetensors")
}

func TestDiskWriterShutdown(t *testing.T) {
	dir := t.TempDir()
	w := newDiskWriter(dir)

	for i := range 5 {
		w.submit(diskWriteJob{
			data:     []byte(fmt.Sprintf("data_%d", i)),
			filename: fmt.Sprintf("file_%d.safetensors", i),
		})
	}

	w.shutdown()

	for i := range 5 {
		path := filepath.Join(dir, fmt.Sprintf("file_%d.safetensors", i))
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("file_%d should exist after shutdown: %v", i, err)
		}
	}

	if got := len(w.drainResults()); got != 5 {
		t.Fatalf("expected 5 results after shutdown, got %d", got)
	}
}
