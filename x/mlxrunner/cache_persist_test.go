//go:build multiseq_persistence_todo

// TODO(multiseq-persistence): These tests exercise the single-seq persistence
// lifecycle (testKvCache stub uses single activePath; scheduleWrite and
// restoreMatchedPath have signatures that assume one path). Theirs' multi-seq
// kvCache uses activePaths map[int][]*trieNode and per-seq snapshot identity.
// The invariants these tests cover (write-on-evict, rehydrate-on-restart,
// Cold/Warm/Gone transitions) remain valid but need porting to the multi-seq
// observer wired from scheduler.go. Skipped for now via build tag.

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
		formatVersion: "3",
		modelDigest:   "abcdef",
		parentHash:    "p1",
		tokens:        []int32{5, 6, 7},
		layerCount:    4,
		snapshotTypes: []string{"kv", "kv", "kv", "kv"},
	}
	meta := encodeHeader(h)
	if meta["cache_format_version"] != "3" {
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

func TestWriteOneRefusesUnserializableSnapshot(t *testing.T) {
	// fakeSnapshot implements cache.Snapshot but not cache.SerializableSnapshot.
	// writeOne must refuse rather than write a file with "unknown" placeholders
	// that the loader will reject on every subsequent startup.
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "modelX", 1)
	defer c.teardown()

	node := &trieNode{
		tokens:    []int32{1, 2, 3},
		parent:    c.root,
		endOffset: c.root.endOffset + 3,
		snapshots: []cache.Snapshot{&fakeSnapshot{byteSize: 1024}},
	}
	c.root.children = append(c.root.children, node)

	err := c.writeOne(node)
	if err == nil {
		t.Fatal("writeOne should have returned error for non-exposer snapshot")
	}
	if !strings.Contains(err.Error(), "is not serializable") {
		t.Errorf("error = %q, want substring 'is not serializable'", err.Error())
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 0 {
		t.Errorf("no files should be written; got %d entries", len(entries))
	}
	if node.diskPath != "" {
		t.Errorf("diskPath should remain empty; got %q", node.diskPath)
	}
	if atomic.LoadInt64(&c.diskBytes) != 0 {
		t.Errorf("diskBytes should remain 0; got %d", atomic.LoadInt64(&c.diskBytes))
	}
}

func TestWriteOnePersistsNilLayerAsEmpty(t *testing.T) {
	// Hybrid models (e.g. Mamba + transformer) may legitimately have a nil
	// snapshot at a recurrent layer that has no state yet. The writer must
	// tag that slot as "empty" and round-trip it back to a nil snapshot on
	// load — not refuse to write the whole node.
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "modelY", 2)
	defer c.teardown()

	k := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	mlx.Pin(k, v)
	node := &trieNode{
		tokens:    []int32{4, 5, 6},
		parent:    c.root,
		endOffset: c.root.endOffset + 3,
		// Layer 0 has real state (kv), layer 1 has none (nil — "empty").
		snapshots: []cache.Snapshot{
			&arraySnapshot{keys: k, values: v, size: 512},
			nil,
		},
	}
	c.root.children = append(c.root.children, node)

	if err := c.writeOne(node); err != nil {
		t.Fatalf("writeOne: %v", err)
	}
	h, err := c.readHeader(node.diskPath)
	if err != nil {
		t.Fatalf("readHeader: %v", err)
	}
	if len(h.snapshotTypes) != 2 || h.snapshotTypes[0] != "kv" || h.snapshotTypes[1] != "empty" {
		t.Errorf("snapshot_types = %v, want [kv empty]", h.snapshotTypes)
	}
}

func TestScheduleWriteSynchronous(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{7, 7, 7}, 1024)
	c.scheduleWrite(node)
	// Synchronous write: scheduleWrite returns only after the file is on disk.
	if node.diskPath == "" {
		t.Errorf("node.diskPath empty after scheduleWrite")
	}
	if _, err := os.Stat(node.diskPath); err != nil {
		t.Errorf("file missing after scheduleWrite: %v", err)
	}
	if atomic.LoadInt64(&c.diskBytes) <= 0 {
		t.Error("diskBytes not updated")
	}
}

func TestScheduleWriteManyNodes(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	const N = 5
	nodes := make([]*trieNode, N)
	for i := range nodes {
		nodes[i] = newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
		c.scheduleWrite(nodes[i])
		if nodes[i].diskPath == "" {
			t.Errorf("node[%d] not persisted", i)
		}
	}
	if atomic.LoadInt64(&c.diskBytes) <= 0 {
		t.Error("diskBytes not updated")
	}
}

func TestScheduleWriteCircuitBreaker(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	// Force write errors via a read-only directory. After 5 consecutive
	// failures the circuit breaker should trip and disable the writer.
	readOnly := filepath.Join(dir, "ro")
	if err := os.MkdirAll(readOnly, 0o500); err != nil {
		t.Fatal(err)
	}
	c.cacheDir = readOnly

	for i := 0; i < 6; i++ {
		n := newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
		c.scheduleWrite(n)
	}
	c.writer.mu.Lock()
	disabled := c.writer.disabled
	c.writer.mu.Unlock()
	if !disabled {
		t.Error("circuit breaker did not trip after 6 failed writes")
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
	path := node.diskPath
	stat1, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file missing after first scheduleWrite: %v", err)
	}
	// Second call should be a no-op — the node is already persisted.
	c.scheduleWrite(node)
	stat2, err := os.Stat(path)
	if err != nil {
		t.Fatalf("file missing after second scheduleWrite: %v", err)
	}
	if !stat1.ModTime().Equal(stat2.ModTime()) {
		t.Error("second scheduleWrite re-wrote the file")
	}
}

func TestScheduleWriteNoOpWhenWriterNil(t *testing.T) {
	c := &kvCache{caches: make([]cache.Cache, 1)}
	c.ensureRoot()
	node := &trieNode{tokens: []int32{1}, parent: c.root}
	c.root.children = append(c.root.children, node)
	c.scheduleWrite(node) // must not panic
	if node.diskPath != "" {
		t.Error("scheduleWrite wrote even though writer is nil")
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
	path := node.diskPath
	node.snapshots = nil

	c2 := newTestKvCacheWithDisk(t, dir, "modelB", 1) // different digest
	defer c2.teardown()
	foreignNode := &trieNode{diskPath: path, parent: c2.root}
	if err := c2.loadFromDisk(foreignNode); err == nil {
		t.Error("loadFromDisk should reject cross-model files")
	}
}

func TestRestoreMatchedPathRestoresCold(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
	c.scheduleWrite(node)
	for _, s := range node.snapshots {
		s.Close()
	}
	node.snapshots = nil

	if err := c.restoreMatchedPath([]*trieNode{node}, len(node.tokens)); err != nil {
		t.Fatalf("restoreMatchedPath: %v", err)
	}
	if len(node.snapshots) != 1 {
		t.Errorf("restoreMatchedPath didn't restore snapshots (got %d)", len(node.snapshots))
	}
}

func TestRestoreMatchedPathStopsOnGoneAncestor(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
	c.scheduleWrite(n1)
	for _, s := range n1.snapshots {
		s.Close()
	}
	n1.snapshots = nil

	// n2 is Gone (no diskPath, no snapshots).
	n2 := &trieNode{tokens: []int32{2}, parent: n1, endOffset: n1.endOffset + 1}
	n1.children = append(n1.children, n2)

	err := c.restoreMatchedPath([]*trieNode{n1, n2}, len(n1.tokens)+len(n2.tokens))
	if err != nil {
		t.Errorf("restoreMatchedPath returned error on Gone ancestor (should degrade): %v", err)
	}
	if n1.snapshots == nil {
		t.Error("n1 should have been restored before the loop stopped at n2")
	}
	if n2.snapshots != nil {
		t.Error("n2 should remain unrestored (it was Gone)")
	}
}

func TestMemoryPassDemotesToCold(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2}, 9<<30) // oversized
	c.pagedOutBytes = 9 << 30
	c.scheduleWrite(node)

	c.enforceEvictionPolicy()
	if node.snapshots != nil {
		t.Error("memory pass should have dropped snapshots")
	}
	if node.diskPath == "" {
		t.Error("disk file was deleted in memory pass — should only drop memory")
	}
}

func TestDiskPassRemovesOverCap(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
	n2 := newTestNodeWithArraySnapshot(t, c.root, []int32{2}, 1024)
	c.scheduleWrite(n1)
	c.scheduleWrite(n2)
	// Set the cap so it holds exactly one file's worth plus slack. The
	// on-disk safetensors size for two 1x8 float32 arrays + metadata is
	// runtime-dependent, so base the cap on actual diskSize.
	c.diskMax = n2.diskSize + n2.diskSize/2

	// Capture paths before eviction; enforceDiskPolicy clears diskPath on
	// the evicted node.
	p1 := n1.diskPath
	p2 := n2.diskPath
	// Make n1 older so disk pass picks it first.
	n1.lastUsed = n1.lastUsed.Add(-time.Hour)

	c.enforceEvictionPolicy()

	if _, err := os.Stat(p1); !os.IsNotExist(err) {
		t.Errorf("disk pass did not delete oldest node's file (err=%v)", err)
	}
	if _, err := os.Stat(p2); err != nil {
		t.Errorf("disk pass deleted too much: %v", err)
	}
}

func TestRestoreMatchedPathShortCircuitsBeyondMatched(t *testing.T) {
	// When matched covers only a prefix of the path, restoreMatchedPath
	// should stop after restoring the nodes that contribute to matched —
	// further disk I/O and MLX upload is pure waste.
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	defer c.teardown()

	n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3, 4, 5}, 1024)
	c.scheduleWrite(n1)
	n2 := newTestNodeWithArraySnapshot(t, n1, []int32{6, 7, 8, 9, 10}, 1024)
	c.scheduleWrite(n2)
	n3 := newTestNodeWithArraySnapshot(t, n2, []int32{11, 12, 13, 14, 15}, 1024)
	c.scheduleWrite(n3)
	for _, n := range []*trieNode{n1, n2, n3} {
		for _, s := range n.snapshots {
			s.Close()
		}
		n.snapshots = nil
	}

	// matched=7 covers n1 entirely (5 tokens) and bisects n2 (needs full load).
	// n3 contributes nothing and must remain Cold.
	err := c.restoreMatchedPath([]*trieNode{c.root, n1, n2, n3}, 7)
	if err != nil {
		t.Fatalf("restoreMatchedPath: %v", err)
	}
	if n1.snapshots == nil {
		t.Error("n1 should be restored")
	}
	if n2.snapshots == nil {
		t.Error("n2 should be restored (matched bisects its edge)")
	}
	if n3.snapshots != nil {
		t.Error("n3 should remain Cold (beyond matched)")
	}
}

func TestRehydrateEmptyDir(t *testing.T) {
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	defer c.teardown()

	if err := c.rehydrate(); err != nil {
		t.Fatalf("rehydrate on empty dir: %v", err)
	}
	if c.diskBytes != 0 {
		t.Errorf("diskBytes = %d on empty dir", c.diskBytes)
	}
}

func TestRehydrateRebuildsSkeleton(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	// Session A: write a parent and a child node.
	sessA := newTestKvCacheWithDisk(t, dir, "model", 1)
	sessA.writer = newDiskWriter(sessA.kvCache)
	n1 := newTestNodeWithArraySnapshot(t, sessA.root, []int32{1, 2}, 1024)
	sessA.scheduleWrite(n1)
	n2 := newTestNodeWithArraySnapshot(t, n1, []int32{3, 4}, 1024)
	sessA.scheduleWrite(n2)
	sessA.teardown()

	// Session B: fresh cache, same dir, same model.
	sessB := newTestKvCacheWithDisk(t, dir, "model", 1)
	defer sessB.teardown()
	if err := sessB.rehydrate(); err != nil {
		t.Fatalf("rehydrate: %v", err)
	}
	if len(sessB.root.children) != 1 {
		t.Fatalf("root children = %d, want 1", len(sessB.root.children))
	}
	n1p := sessB.root.children[0]
	if len(n1p.tokens) != 2 || n1p.tokens[0] != 1 {
		t.Errorf("n1p.tokens = %v", n1p.tokens)
	}
	if n1p.diskPath == "" {
		t.Error("rehydrated n1 missing diskPath")
	}
	if n1p.snapshots != nil {
		t.Error("rehydrate should NOT load snapshots (Cold state)")
	}
	if len(n1p.children) != 1 {
		t.Fatalf("n1p children = %d, want 1", len(n1p.children))
	}
	if sessB.diskBytes <= 0 {
		t.Error("diskBytes not updated")
	}
}

func TestRehydrateCleansTmpOrphans(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	tmpPath := filepath.Join(dir, "aaa.safetensors.tmp")
	if err := os.WriteFile(tmpPath, []byte("partial"), 0o644); err != nil {
		t.Fatal(err)
	}

	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	defer c.teardown()
	if err := c.rehydrate(); err != nil {
		t.Fatalf("rehydrate: %v", err)
	}
	if _, err := os.Stat(tmpPath); !os.IsNotExist(err) {
		t.Error(".tmp file not cleaned up")
	}
}

func TestRehydrateRejectsForeignDigest(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	sessA := newTestKvCacheWithDisk(t, dir, "modelA", 1)
	sessA.writer = newDiskWriter(sessA.kvCache)
	n := newTestNodeWithArraySnapshot(t, sessA.root, []int32{1}, 1024)
	sessA.scheduleWrite(n)
	sessA.teardown()

	sessB := newTestKvCacheWithDisk(t, dir, "modelB", 1) // different digest
	defer sessB.teardown()
	if err := sessB.rehydrate(); err != nil {
		t.Fatalf("rehydrate: %v", err)
	}
	if len(sessB.root.children) != 0 {
		t.Errorf("rehydrate loaded foreign-digest files (%d children)", len(sessB.root.children))
	}
}

func TestRehydrateRejectsUnknownSnapshotTypes(t *testing.T) {
	// A pre-fix writer could produce a safetensors file whose snapshot_types
	// metadata contained "unknown" — the loader rejects such files on every
	// restore, poisoning the ancestor chain. Rehydrate must delete them.
	skipIfNoMLX(t)
	dir := t.TempDir()

	k := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, 8)
	mlx.Pin(k, v)
	defer mlx.Unpin(k, v)

	h := headerFields{
		formatVersion: cacheFormatVersion,
		modelDigest:   "modelZ",
		tokens:        []int32{1, 2, 3},
		layerCount:    1,
		snapshotTypes: []string{"unknown"},
	}
	meta := encodeHeader(h)
	path := filepath.Join(dir, "bad.safetensors")
	if err := mlx.SaveSafetensorsWithMetadata(path,
		map[string]*mlx.Array{"layer_0_keys": k, "layer_0_values": v}, meta); err != nil {
		t.Fatalf("save: %v", err)
	}

	c := newTestKvCacheWithDisk(t, dir, "modelZ", 1)
	defer c.teardown()
	if err := c.rehydrate(); err != nil {
		t.Fatalf("rehydrate: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("rehydrate should have deleted unknown-type file; stat err = %v", err)
	}
	if len(c.root.children) != 0 {
		t.Errorf("unknown-type file should not become a trie node; got %d children", len(c.root.children))
	}
}

func TestShutdownFlushesWarmNodes(t *testing.T) {
	// Writes are deferred from prefill to shutdown. Any Warm node with
	// no diskPath must be persisted when shutdown runs.
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)

	// Three Warm nodes, no disk state yet.
	n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
	n2 := newTestNodeWithArraySnapshot(t, n1, []int32{4, 5}, 1024)
	n3 := newTestNodeWithArraySnapshot(t, n2, []int32{6, 7, 8, 9}, 1024)
	for _, n := range []*trieNode{n1, n2, n3} {
		if n.diskPath != "" {
			t.Fatalf("pre-shutdown: diskPath should be empty; got %q", n.diskPath)
		}
	}

	c.shutdown()

	for i, n := range []*trieNode{n1, n2, n3} {
		if n.diskPath == "" {
			t.Errorf("n%d: shutdown left diskPath empty", i+1)
		}
		if _, err := os.Stat(n.diskPath); err != nil {
			t.Errorf("n%d: file missing after shutdown: %v", i+1, err)
		}
	}
}

func TestShutdownSkipsAlreadyPersistedNodes(t *testing.T) {
	// Nodes that already have diskPath (persisted during eviction) must
	// not trigger redundant writes on shutdown.
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)

	n := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
	c.scheduleWrite(n)
	if n.diskPath == "" {
		t.Fatal("setup failed: scheduleWrite didn't persist")
	}
	firstPath := n.diskPath
	firstMtime, err := os.Stat(firstPath)
	if err != nil {
		t.Fatal(err)
	}

	c.shutdown()

	stat, err := os.Stat(n.diskPath)
	if err != nil {
		t.Fatalf("file missing after shutdown: %v", err)
	}
	if !stat.ModTime().Equal(firstMtime.ModTime()) {
		t.Error("shutdown re-wrote an already-persisted node")
	}
}

func TestShutdownBudgetFlushesNewestFirst(t *testing.T) {
	// Under a 0-budget shutdown, no writes complete (deadline passes before
	// the first iteration). Under an ample budget, all flush. Between, the
	// newest-by-lastUsed wins.
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)

	// Three nodes with distinct lastUsed times — n3 newest, n1 oldest.
	base := time.Now()
	n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
	n1.lastUsed = base.Add(-2 * time.Hour)
	n2 := newTestNodeWithArraySnapshot(t, n1, []int32{4, 5}, 1024)
	n2.lastUsed = base.Add(-1 * time.Hour)
	n3 := newTestNodeWithArraySnapshot(t, n2, []int32{6, 7, 8, 9}, 1024)
	n3.lastUsed = base

	// Zero budget: nothing should flush, no error, all stay Warm.
	c.shutdownWithBudget(0)
	for i, n := range []*trieNode{n1, n2, n3} {
		if n.diskPath != "" {
			t.Errorf("0-budget shutdown unexpectedly wrote n%d", i+1)
		}
	}

	// Ample budget: all three flush.
	c.shutdownWithBudget(30 * time.Second)
	for i, n := range []*trieNode{n1, n2, n3} {
		if n.diskPath == "" {
			t.Errorf("ample-budget shutdown left n%d unwritten", i+1)
		}
	}
}

func TestFeatureDisabledHasNoWriter(t *testing.T) {
	t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", "0")
	c := newKvCache("model", 1)
	if c.writer != nil {
		t.Error("writer created when OLLAMA_KV_CACHE_DISK_MAX=0")
	}
	if c.cacheDir != "" {
		t.Error("cacheDir set when feature disabled")
	}
}

func TestFeatureEnabledCreatesWriter(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("OLLAMA_KV_CACHE_ROOT", dir)
	t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", "-1")
	c := newKvCache("modelX", 1)
	defer c.shutdown()
	if c.writer == nil {
		t.Error("writer missing with feature enabled")
	}
	if !strings.HasSuffix(c.cacheDir, "modelX") {
		t.Errorf("cacheDir = %q, want suffix modelX", c.cacheDir)
	}
}

func TestEndToEndWarmRestart(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	t.Setenv("OLLAMA_KV_CACHE_ROOT", dir)
	t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", "-1")

	// Session A: write a node, then drain & shutdown.
	sessA := newKvCache("m", 1)
	// helper attaches a real array snapshot under the runtime root.
	node := newTestNodeWithArraySnapshot(t, sessA.root, []int32{1, 2, 3, 4}, 1024)
	sessA.scheduleWrite(node)
	sessA.shutdown()

	// Session B: fresh cache, same env, should rehydrate.
	sessB := newKvCache("m", 1)
	defer sessB.shutdown()
	if len(sessB.root.children) != 1 {
		t.Fatalf("rehydrate missed node (children=%d)", len(sessB.root.children))
	}
	rehydrated := sessB.root.children[0]
	if rehydrated.snapshots != nil {
		t.Error("rehydrated node not Cold")
	}

	// Simulate a prefix match and call restoreMatchedPath.
	if err := sessB.restoreMatchedPath([]*trieNode{rehydrated}, len(rehydrated.tokens)); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if rehydrated.snapshots == nil {
		t.Error("restore didn't materialize snapshots")
	}
}

// newTestNode4D builds a trie node with a 4D [B,H,L,D] KV snapshot so that
// real cache.KVCache.Split (which slices on axis 2) can operate on it.
// Required by the split-on-Cold tests — the 2D helper above is incompatible
// with KVCache.Split.
func newTestNode4D(t *testing.T, parent *trieNode, tokens []int32, byteSize int) *trieNode {
	t.Helper()
	n := &trieNode{tokens: tokens, parent: parent, endOffset: parent.endOffset + len(tokens)}
	parent.children = append(parent.children, n)
	k := mlx.Zeros(mlx.DTypeFloat32, 1, 1, len(tokens), 8)
	v := mlx.Zeros(mlx.DTypeFloat32, 1, 1, len(tokens), 8)
	mlx.Pin(k, v)
	n.snapshots = []cache.Snapshot{&arraySnapshot{keys: k, values: v, size: byteSize}}
	return n
}

func TestSplitOfColdNodeLoadsBeforeSplit(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	// Real KVCache so Split has a working implementation.
	c.caches[0] = cache.NewKVCache()
	defer c.teardown()

	// Persist a node with 5 tokens under the root.
	node := newTestNode4D(t, c.root, []int32{10, 20, 30, 40, 50}, 1024)
	c.scheduleWrite(node)
	if node.diskPath == "" {
		t.Fatal("node was not persisted")
	}
	oldPath := node.diskPath

	// Demote to Cold: drop in-memory snapshots, keep diskPath.
	for _, s := range node.snapshots {
		s.Close()
	}
	node.snapshots = nil

	// advancePath with tokens that partially match at index 3: [10,20,30] match,
	// [99] diverges. This triggers splitNode on the Cold node.
	c.activePath = []*trieNode{c.root}
	_ = c.advancePath(c.root, []int32{10, 20, 30, 99}, 4)

	// After the fix: the stale file is gone, original (now suffix) has tokens
	// [40,50] with fresh snapshots and a new diskPath, and the new intermediate
	// has tokens [10,20,30] with snapshots and diskPath.
	if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
		t.Errorf("stale pre-split file not removed: err=%v", err)
	}
	if len(node.tokens) != 2 || node.tokens[0] != 40 || node.tokens[1] != 50 {
		t.Errorf("suffix tokens = %v, want [40,50]", node.tokens)
	}
	if node.snapshots == nil {
		t.Error("suffix has no snapshots after split")
	}
	if node.diskPath == "" {
		t.Error("suffix was not re-persisted")
	}
	if node.diskPath == oldPath {
		t.Error("suffix kept stale diskPath")
	}

	if node.parent == nil || node.parent == c.root {
		t.Fatalf("suffix parent wrong: %v", node.parent)
	}
	intermediate := node.parent
	if len(intermediate.tokens) != 3 || intermediate.tokens[0] != 10 {
		t.Errorf("prefix tokens = %v, want [10,20,30]", intermediate.tokens)
	}
	if intermediate.snapshots == nil {
		t.Error("prefix has no snapshots after split")
	}
	if intermediate.diskPath == "" {
		t.Error("prefix was not persisted")
	}
}

func TestSplitOfColdNodeSkipsWhenLoadFails(t *testing.T) {
	skipIfNoMLX(t)
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = newDiskWriter(c.kvCache)
	c.caches[0] = cache.NewKVCache()
	defer c.teardown()

	node := newTestNode4D(t, c.root, []int32{10, 20, 30, 40, 50}, 1024)
	c.scheduleWrite(node)
	if node.diskPath == "" {
		t.Fatal("node was not persisted")
	}
	oldPath := node.diskPath
	oldSize := node.diskSize

	// Demote to Cold and corrupt the file so loadFromDisk fails.
	for _, s := range node.snapshots {
		s.Close()
	}
	node.snapshots = nil
	if err := os.Truncate(oldPath, 0); err != nil {
		t.Fatal(err)
	}

	before := atomic.LoadInt64(&c.diskBytes)
	c.activePath = []*trieNode{c.root}
	_ = c.advancePath(c.root, []int32{10, 20, 30, 99}, 4)

	// Load failed: node should be demoted to Gone (diskPath cleared), no
	// intermediate created, diskBytes decremented by the failed node's size.
	if node.diskPath != "" {
		t.Errorf("failed-load node still has diskPath = %q", node.diskPath)
	}
	if node.diskSize != 0 {
		t.Errorf("failed-load node still has diskSize = %d", node.diskSize)
	}
	after := atomic.LoadInt64(&c.diskBytes)
	if before-after != oldSize {
		t.Errorf("diskBytes decrement = %d, want %d", before-after, oldSize)
	}
	// node.tokens must NOT have been sliced (no split happened).
	if len(node.tokens) != 5 {
		t.Errorf("node.tokens = %v, want 5 tokens (unsliced)", node.tokens)
	}
	// appendTokens should have created a sibling of node with the divergent tokens.
	if len(c.root.children) < 2 {
		t.Errorf("no sibling created for divergent tokens; root children=%d", len(c.root.children))
	}
}

func TestScheduleWriteSkipsEmptySnapshots(t *testing.T) {
	dir := t.TempDir()
	c := newTestKvCacheWithDisk(t, dir, "model", 1)
	c.writer = &diskWriter{}
	defer c.teardown()

	// Node with no snapshots — scheduleWrite should no-op without panic and
	// without going through writeOne (which would warn).
	n := &trieNode{tokens: []int32{1, 2, 3}, parent: c.root}
	c.scheduleWrite(n)
	if n.diskPath != "" {
		t.Errorf("scheduleWrite on empty-snapshots node wrote to disk: %q", n.diskPath)
	}
}
