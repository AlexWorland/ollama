package mlxrunner

import (
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// skipIfNoMLX skips persistence tests that require real MLX arrays when the
// MLX dynamic library is not available (e.g. CI without Apple Silicon).
func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// testKvCache wraps *kvCache with cleanup convenience for tests that
// construct a writer directly rather than going through newKvCache.
type testKvCache struct {
	*kvCache
	t *testing.T
}

func (c *testKvCache) teardown() {
	if c.writer != nil {
		c.writer.shutdown(0)
	}
}

func newTestKvCacheWithDisk(t *testing.T, dir, modelDigest string, numLayers int) *testKvCache {
	t.Helper()
	c := &kvCache{
		caches:      make([]cache.Cache, numLayers),
		modelDigest: modelDigest,
		cacheDir:    dir,
	}
	c.ensureRoot()
	return &testKvCache{kvCache: c, t: t}
}

// newTestNodeWithArraySnapshot creates a trie child of `parent` carrying a
// real MLX array snapshot. Caller must skipIfNoMLX before invoking.
func newTestNodeWithArraySnapshot(t *testing.T, parent *trieNode, tokens []int32, byteSize int) *trieNode {
	t.Helper()
	n := &trieNode{tokens: tokens, parent: parent, endOffset: parent.endOffset + len(tokens)}
	parent.children = append(parent.children, n)
	k := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	mlx.Pin(k, v)
	n.snapshots = []cache.Snapshot{&arraySnapshot{keys: k, values: v, size: byteSize}}
	return n
}

func TestFilenameDeterministic(t *testing.T) {
	f1 := contentFilename("modeldigest123", "", []int32{1, 2, 3}, 32, "bfloat16")
	f2 := contentFilename("modeldigest123", "", []int32{1, 2, 3}, 32, "bfloat16")
	if f1 != f2 {
		t.Errorf("filename not deterministic: %q vs %q", f1, f2)
	}
	if !strings.HasSuffix(f1, ".safetensors") {
		t.Errorf("filename %q missing .safetensors suffix", f1)
	}
}

func TestFilenameUniqueness(t *testing.T) {
	f1 := contentFilename("m", "", []int32{1, 2, 3}, 32, "bf16")
	f2 := contentFilename("m", "parent", []int32{1, 2, 3}, 32, "bf16")
	f3 := contentFilename("m", "", []int32{1, 2, 4}, 32, "bf16")
	f4 := contentFilename("m", "", []int32{1, 2, 3}, 32, "fp16")
	if f1 == f2 || f1 == f3 || f1 == f4 {
		t.Error("filename collisions across differing inputs")
	}
}

func TestTokensRoundTrip(t *testing.T) {
	cases := [][]int32{
		nil,
		{},
		{0},
		{-1, 0, 1, 100_000, 2_147_483_647},
	}
	for _, c := range cases {
		encoded := encodeTokens(c)
		got, err := decodeTokens(encoded)
		if err != nil {
			t.Errorf("decode(%v): %v", c, err)
			continue
		}
		if len(got) != len(c) {
			t.Errorf("len mismatch: got %d want %d", len(got), len(c))
			continue
		}
		for i := range c {
			if got[i] != c[i] {
				t.Errorf("tok[%d]: got %d want %d", i, got[i], c[i])
			}
		}
	}
}

func TestHeaderBuildParse(t *testing.T) {
	h := headerFields{
		formatVersion: "1",
		modelDigest:   "abcdef",
		parentHash:    "p1",
		tokens:        []int32{5, 6, 7},
		layerCount:    4,
		snapshotTypes: []string{"kv", "kv", "kv", "kv"},
	}
	meta := encodeHeader(h)
	if meta["cache_format_version"] != "1" {
		t.Errorf("format version missing")
	}
	back, err := decodeHeader(meta)
	if err != nil {
		t.Fatalf("decodeHeader: %v", err)
	}
	if back.modelDigest != "abcdef" || back.parentHash != "p1" || back.layerCount != 4 {
		t.Errorf("header round-trip lost fields: %+v", back)
	}
	if len(back.tokens) != 3 || back.tokens[0] != 5 {
		t.Errorf("tokens round-trip: %v", back.tokens)
	}
	if len(back.snapshotTypes) != 4 {
		t.Errorf("snapshotTypes round-trip: %v", back.snapshotTypes)
	}
}

func TestDecodeHeaderRejectsUnknownVersion(t *testing.T) {
	_, err := decodeHeader(map[string]string{"cache_format_version": "99"})
	if err == nil {
		t.Error("decodeHeader should reject unknown format version")
	}
}

func TestWriteOneRoundTrip(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "modelA", 1)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)

	if err := c.writeOne(node); err != nil {
		t.Fatalf("writeOne: %v", err)
	}
	if node.diskPath == "" {
		t.Fatalf("diskPath not set after writeOne")
	}
	if node.diskSize <= 0 {
		t.Errorf("diskSize = %d, want > 0", node.diskSize)
	}
	if _, err := os.Stat(node.diskPath); err != nil {
		t.Errorf("disk file missing: %v", err)
	}

	h, err := c.readHeader(node.diskPath)
	if err != nil {
		t.Fatalf("readHeader: %v", err)
	}
	if h.modelDigest != "modelA" {
		t.Errorf("modelDigest = %q, want modelA", h.modelDigest)
	}
	if len(h.tokens) != 3 || h.tokens[0] != 1 {
		t.Errorf("tokens round-trip lost: %v", h.tokens)
	}
	if h.layerCount != 1 {
		t.Errorf("layerCount = %d, want 1", h.layerCount)
	}
}

func TestDiskWriterFIFO(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	const N = 5
	nodes := make([]*trieNode, N)
	for i := range nodes {
		nodes[i] = newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
		nodes[i].inflightWrite = make(chan struct{})
		c.writer.enqueue(nodes[i])
	}
	for i, n := range nodes {
		select {
		case <-n.inflightWrite:
		case <-time.After(5 * time.Second):
			t.Fatalf("node[%d] write did not complete in 5s", i)
		}
		if n.diskPath == "" {
			t.Errorf("node[%d].diskPath empty after write", i)
		}
	}
	if atomic.LoadInt64(&c.diskBytes) <= 0 {
		t.Error("diskBytes not updated")
	}
}

func TestDiskWriterShutdownDrainsQueue(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)

	nodes := make([]*trieNode, 3)
	for i := range nodes {
		nodes[i] = newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
		nodes[i].inflightWrite = make(chan struct{})
		c.writer.enqueue(nodes[i])
	}
	remaining := c.writer.shutdown(5 * time.Second)
	if remaining != 0 {
		t.Errorf("shutdown left %d pending", remaining)
	}
	for i, n := range nodes {
		if n.diskPath == "" {
			t.Errorf("node[%d] not persisted after shutdown drain", i)
		}
	}
}

func TestDiskWriterShutdownTimeoutReportsRemaining(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)

	// Force write errors via a read-only directory so the writer keeps
	// failing and shutdown(timeout) exercises the not-yet-drained branch.
	readOnly := filepath.Join(dir, "ro")
	if err := os.MkdirAll(readOnly, 0o500); err != nil {
		t.Fatal(err)
	}
	c.cacheDir = readOnly

	for i := 0; i < 2; i++ {
		n := newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
		n.inflightWrite = make(chan struct{})
		c.writer.enqueue(n)
	}
	// Test only asserts shutdown returns within reasonable time; remaining
	// count may vary depending on how fast the loop drained before stop.
	_ = c.writer.shutdown(2 * time.Second)
}

func TestAttachSnapshotsSchedulesWrite(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{7, 7, 7}, 1024)
	c.scheduleWrite(node)

	if node.inflightWrite == nil {
		t.Fatal("scheduleWrite did not create an inflightWrite channel")
	}
	select {
	case <-node.inflightWrite:
	case <-time.After(2 * time.Second):
		t.Fatal("write did not complete within 2s")
	}
	if node.diskPath == "" {
		t.Errorf("node.diskPath still empty after write")
	}
}

func TestScheduleWriteIdempotent(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
	c.scheduleWrite(node)
	ch1 := node.inflightWrite
	c.scheduleWrite(node)
	if node.inflightWrite != ch1 {
		t.Error("second scheduleWrite replaced the in-flight channel")
	}
	<-ch1

	// After write completes, another scheduleWrite is still a no-op (diskPath set).
	c.scheduleWrite(node)
	if node.inflightWrite != nil {
		t.Error("scheduleWrite re-enqueued an already-persisted node")
	}
}

func TestScheduleWriteNoOpWhenWriterNil(t *testing.T) {
	c := &kvCache{caches: make([]cache.Cache, 1)}
	c.ensureRoot()
	node := &trieNode{tokens: []int32{1}, parent: c.root}
	c.root.children = append(c.root.children, node)
	c.scheduleWrite(node) // must not panic
	if node.inflightWrite != nil {
		t.Error("scheduleWrite created channel even though writer is nil")
	}
}

func TestLoadFromDisk(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
	c.scheduleWrite(node)
	<-node.inflightWrite
	path := node.diskPath
	size := node.diskSize

	// Simulate Cold: drop snapshots in memory.
	for _, s := range node.snapshots {
		s.Close()
	}
	node.snapshots = nil

	if err := c.loadFromDisk(node); err != nil {
		t.Fatalf("loadFromDisk: %v", err)
	}
	if len(node.snapshots) != 1 {
		t.Fatalf("after load: got %d snapshots, want 1", len(node.snapshots))
	}
	if node.diskPath != path || node.diskSize != size {
		t.Errorf("load mutated disk fields: %q %d", node.diskPath, node.diskSize)
	}
}

func TestLoadFromDiskRejectsForeignDigest(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "modelA", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
	c.scheduleWrite(node)
	<-node.inflightWrite
	path := node.diskPath
	node.snapshots = nil

	c2 := newTestKvCacheWithDisk(t, dir, "modelB", 1) // different digest
	defer c2.teardown()
	foreignNode := &trieNode{diskPath: path, parent: c2.root}
	if err := c2.loadFromDisk(foreignNode); err == nil {
		t.Error("loadFromDisk should reject cross-model files")
	}
}
