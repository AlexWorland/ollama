package mlxrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// saveTrieTo is a test helper that sets cacheDir/modelID and calls saveTrie.
func saveTrieTo(c *kvCache, dir, modelID string) error {
	c.cacheDir = dir
	c.modelID = modelID
	return c.saveTrie()
}

func skipIfNoMLXTest(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// makeTestKVCache creates a kvCache with real KVCache layers and a simple
// trie. Returns the cache and a cleanup function.
func makeTestKVCache(t *testing.T, numLayers int) *kvCache {
	t.Helper()

	c := &kvCache{}
	c.caches = make([]cache.Cache, numLayers)
	for i := range c.caches {
		c.caches[i] = cache.NewKVCache()
	}
	c.root = &trieNode{lastUsed: time.Now()}
	c.activePath = []*trieNode{c.root}
	return c
}

// feedTokens simulates the cache receiving tokens by calling Update on each
// layer with zero arrays of the right shape.
func feedTokens(c *kvCache, count int) {
	for range count {
		for _, kv := range c.caches {
			k := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
			v := mlx.Zeros(mlx.DTypeFloat16, 1, 4, 1, 8)
			kv.Update(k, v)
		}
	}
}

func TestSaveLoadTrieRoundTrip(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:test1234"
	numLayers := 2

	// Build a kvCache with a simple trie: root -> [1,2,3] -> [4,5]
	c := makeTestKVCache(t, numLayers)

	// Feed 3 tokens and create a snapshot at the root's child.
	feedTokens(c, 3)
	child := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = append(c.activePath, child)

	// Feed 2 more tokens and create a grandchild.
	feedTokens(c, 2)
	grandchild := child.appendTokens(c.root, []int32{4, 5}, 5)
	snaps2 := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps2[j] = kv.Snapshot(3)
	}
	grandchild.setSnapshots(snaps2, &c.pagedOutBytes)
	c.activePath = append(c.activePath, grandchild)

	if c.pagedOutBytes == 0 {
		t.Fatal("expected non-zero paged out bytes")
	}

	// Save.
	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal("saveTrie:", err)
	}

	// Verify files exist.
	if _, err := os.Stat(filepath.Join(dir, "trie.json")); err != nil {
		t.Fatal("trie.json not found:", err)
	}

	// Verify files are hash-named (no node_N or evicted_N patterns).
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		name := e.Name()
		if name == "trie.json" {
			continue
		}
		// Hash files are 16 hex chars + .safetensors
		if len(name) != len("0123456789abcdef.safetensors") {
			t.Fatalf("unexpected file naming: %s", name)
		}
	}

	// Load into a fresh trie.
	root, pagedOut, _, err := loadTrie(dir, modelID, numLayers)
	if err != nil {
		t.Fatal("loadTrie:", err)
	}
	if root == nil {
		t.Fatal("loadTrie returned nil root")
	}
	if pagedOut == 0 {
		t.Fatal("expected non-zero paged out bytes after load")
	}

	// Verify topology.
	if len(root.children) != 1 {
		t.Fatalf("root should have 1 child, got %d", len(root.children))
	}
	restoredChild := root.children[0]
	if len(restoredChild.tokens) != 3 || restoredChild.tokens[0] != 1 {
		t.Fatalf("child tokens mismatch: %v", restoredChild.tokens)
	}
	if restoredChild.endOffset != 3 {
		t.Fatalf("child endOffset: got %d, want 3", restoredChild.endOffset)
	}
	if !restoredChild.hasAllSnapshots() {
		t.Fatal("restored child should have all snapshots")
	}

	if len(restoredChild.children) != 1 {
		t.Fatalf("child should have 1 grandchild, got %d", len(restoredChild.children))
	}
	restoredGC := restoredChild.children[0]
	if len(restoredGC.tokens) != 2 || restoredGC.tokens[0] != 4 {
		t.Fatalf("grandchild tokens mismatch: %v", restoredGC.tokens)
	}
	if restoredGC.endOffset != 5 {
		t.Fatalf("grandchild endOffset: got %d, want 5", restoredGC.endOffset)
	}
	if !restoredGC.hasAllSnapshots() {
		t.Fatal("restored grandchild should have all snapshots")
	}

	// Parent pointer should be correct.
	if restoredGC.parent != restoredChild {
		t.Fatal("grandchild parent should be child")
	}
	if restoredChild.parent != root {
		t.Fatal("child parent should be root")
	}
}

func TestLoadTrieModelMismatch(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	numLayers := 1

	c := makeTestKVCache(t, numLayers)
	feedTokens(c, 2)
	child := c.root.appendTokens(c.root, []int32{1, 2}, 2)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)

	if err := saveTrieTo(c, dir, "sha256:model_a"); err != nil {
		t.Fatal(err)
	}

	// Loading with a different model ID should return nil root.
	root, _, _, err := loadTrie(dir, "sha256:model_b", numLayers)
	if err != nil {
		t.Fatal(err)
	}
	if root != nil {
		t.Fatal("expected nil root for model mismatch")
	}
}

func TestLoadTrieLayerCountMismatch(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:test"
	numLayers := 2

	c := makeTestKVCache(t, numLayers)
	feedTokens(c, 2)
	child := c.root.appendTokens(c.root, []int32{1, 2}, 2)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)

	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal(err)
	}

	// Loading with different layer count should return nil root.
	root, _, _, err := loadTrie(dir, modelID, 4)
	if err != nil {
		t.Fatal(err)
	}
	if root != nil {
		t.Fatal("expected nil root for layer count mismatch")
	}
}

func TestLoadTrieNoFile(t *testing.T) {
	dir := t.TempDir()
	root, pagedOut, _, err := loadTrie(dir, "sha256:test", 2)
	if err != nil {
		t.Fatal(err)
	}
	if root != nil {
		t.Fatal("expected nil root when no trie.json exists")
	}
	if pagedOut != 0 {
		t.Fatal("expected 0 paged out bytes")
	}
}

func TestSaveTrieEmptyCache(t *testing.T) {
	c := &kvCache{}
	dir := t.TempDir()
	// Should succeed silently with no root.
	if err := saveTrieTo(c, dir, "sha256:test"); err != nil {
		t.Fatal(err)
	}
	// No trie.json should be created.
	if _, err := os.Stat(filepath.Join(dir, "trie.json")); !os.IsNotExist(err) {
		t.Fatal("trie.json should not exist for empty cache")
	}
}

func TestSaveTrieRootOnly(t *testing.T) {
	c := &kvCache{
		root: &trieNode{lastUsed: time.Now()},
	}
	c.activePath = []*trieNode{c.root}
	dir := t.TempDir()
	// Root-only trie should not be saved (nothing useful).
	if err := saveTrieTo(c, dir, "sha256:test"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "trie.json")); !os.IsNotExist(err) {
		t.Fatal("trie.json should not exist for root-only trie")
	}
}

func TestCaptureActiveFrontier(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 2
	c := makeTestKVCache(t, numLayers)
	feedTokens(c, 5)

	// Create a child without snapshots.
	child := c.root.appendTokens(c.root, []int32{1, 2, 3, 4, 5}, 5)
	c.activePath = []*trieNode{c.root, child}

	if child.hasSnapshots() {
		t.Fatal("child should not have snapshots yet")
	}

	c.captureActiveFrontier()

	if !child.hasAllSnapshots() {
		t.Fatal("child should have all snapshots after capture")
	}
	if c.pagedOutBytes == 0 {
		t.Fatal("paged out bytes should be non-zero after capture")
	}
}

func TestSaveLoadBranchingTrie(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:branching"
	numLayers := 1

	c := makeTestKVCache(t, numLayers)

	// Build: root -> [1,2,3] -> { [4,5], [6,7] }
	feedTokens(c, 3)
	shared := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	sharedSnaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		sharedSnaps[j] = kv.Snapshot(0)
	}
	shared.setSnapshots(sharedSnaps, &c.pagedOutBytes)

	feedTokens(c, 2)
	branchA := shared.appendTokens(c.root, []int32{4, 5}, 5)
	branchASnaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		branchASnaps[j] = kv.Snapshot(3)
	}
	branchA.setSnapshots(branchASnaps, &c.pagedOutBytes)

	feedTokens(c, 2)
	branchB := shared.appendTokens(c.root, []int32{6, 7}, 5)
	branchBSnaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		branchBSnaps[j] = kv.Snapshot(3)
	}
	branchB.setSnapshots(branchBSnaps, &c.pagedOutBytes)

	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal(err)
	}

	root, _, _, err := loadTrie(dir, modelID, numLayers)
	if err != nil {
		t.Fatal(err)
	}
	if root == nil {
		t.Fatal("nil root after load")
	}

	if len(root.children) != 1 {
		t.Fatalf("root should have 1 child (shared prefix), got %d", len(root.children))
	}
	sharedRestored := root.children[0]
	if len(sharedRestored.children) != 2 {
		t.Fatalf("shared node should have 2 children, got %d", len(sharedRestored.children))
	}

	// Verify branch tokens.
	tokensSet := map[int32]bool{}
	for _, b := range sharedRestored.children {
		tokensSet[b.tokens[0]] = true
		if b.parent != sharedRestored {
			t.Fatal("branch parent should be shared node")
		}
	}
	if !tokensSet[4] || !tokensSet[6] {
		t.Fatalf("expected branches starting with 4 and 6, got %v", tokensSet)
	}
}

func TestEvictAndReloadNodeRoundTrip(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 2
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = t.TempDir()

	// Build root -> [1,2,3] leaf with snapshots.
	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)

	if c.pagedOutBytes == 0 {
		t.Fatal("expected non-zero pagedOutBytes after snapshot")
	}
	bytesBeforeEvict := c.pagedOutBytes

	// Evict to disk.
	if err := c.evictNodeToDisk(leaf); err != nil {
		t.Fatal("evictNodeToDisk:", err)
	}

	// Verify eviction state — file should be hash-named.
	if leaf.diskFile == "" {
		t.Fatal("diskFile should be set after eviction")
	}
	evictedName := filepath.Base(leaf.diskFile)
	expectedHash := nodeFileHash(leaf)
	if evictedName != expectedHash {
		t.Fatalf("evicted file should be hash-named: got %s, want %s", evictedName, expectedHash)
	}
	if leaf.hasSnapshots() {
		t.Fatal("snapshots should be nil after eviction")
	}
	if leaf.snapTypes == nil {
		t.Fatal("snapTypes should be preserved after eviction")
	}
	if leaf.diskFileSize == 0 {
		t.Fatal("diskFileSize should be non-zero")
	}
	if c.pagedOutBytes != 0 {
		t.Fatalf("pagedOutBytes should be 0 after eviction, got %d", c.pagedOutBytes)
	}
	if c.totalDiskBytes != leaf.diskFileSize {
		t.Fatalf("totalDiskBytes mismatch: got %d, want %d", c.totalDiskBytes, leaf.diskFileSize)
	}

	// Verify file exists on disk.
	if _, err := os.Stat(leaf.diskFile); err != nil {
		t.Fatal("evicted file should exist:", err)
	}

	// Reload from disk.
	if err := c.loadNodeFromDisk(leaf); err != nil {
		t.Fatal("loadNodeFromDisk:", err)
	}

	// Verify reload state.
	if !leaf.hasSnapshots() {
		t.Fatal("snapshots should be restored after reload")
	}
	if leaf.diskFile != "" {
		t.Fatal("diskFile should be cleared after reload")
	}
	if leaf.diskFileSize != 0 {
		t.Fatal("diskFileSize should be 0 after reload")
	}
	if c.pagedOutBytes == 0 {
		t.Fatal("pagedOutBytes should be non-zero after reload")
	}
	if c.totalDiskBytes != 0 {
		t.Fatalf("totalDiskBytes should be 0 after reload, got %d", c.totalDiskBytes)
	}

	// Verify snapshot data is usable (approximately same size as before).
	if c.pagedOutBytes != bytesBeforeEvict {
		t.Fatalf("pagedOutBytes mismatch: got %d, want %d", c.pagedOutBytes, bytesBeforeEvict)
	}
}

func TestEvictNodeDiskFailureFallsBackToDelete(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 1
	c := makeTestKVCache(t, numLayers)
	// Empty cacheDir causes evictNodeToDisk to fail.
	c.cacheDir = ""

	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = []*trieNode{c.root}

	beforeBytes := c.pagedOutBytes
	if beforeBytes == 0 {
		t.Fatal("expected non-zero pagedOutBytes")
	}

	// evictNode should fall back to removeNode when disk write fails.
	c.evictNode(leaf)

	// Leaf should be removed from trie.
	if len(c.root.children) != 0 {
		t.Fatalf("root should have 0 children after fallback delete, got %d", len(c.root.children))
	}
	if c.pagedOutBytes != 0 {
		t.Fatalf("pagedOutBytes should be 0 after removal, got %d", c.pagedOutBytes)
	}
}

func TestEnforceDiskEvictionPolicyDeletesOldest(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	numLayers := 1
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = dir

	// Create 3 leaf nodes with disk-backed files.
	feedTokens(c, 2)
	leafA := c.root.appendTokens(c.root, []int32{1, 2}, 2)
	snapsA := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snapsA[j] = kv.Snapshot(0)
	}
	leafA.setSnapshots(snapsA, &c.pagedOutBytes)

	feedTokens(c, 2)
	leafB := c.root.appendTokens(c.root, []int32{3, 4}, 2)
	snapsB := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snapsB[j] = kv.Snapshot(0)
	}
	leafB.setSnapshots(snapsB, &c.pagedOutBytes)

	feedTokens(c, 2)
	leafC := c.root.appendTokens(c.root, []int32{5, 6}, 2)
	snapsC := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snapsC[j] = kv.Snapshot(0)
	}
	leafC.setSnapshots(snapsC, &c.pagedOutBytes)

	// Evict all three to disk with staggered timestamps.
	c.activePath = []*trieNode{c.root} // none active
	if err := c.evictNodeToDisk(leafA); err != nil {
		t.Fatal(err)
	}
	leafA.lastUsed = time.Now().Add(-3 * time.Hour) // oldest

	if err := c.evictNodeToDisk(leafB); err != nil {
		t.Fatal(err)
	}
	leafB.lastUsed = time.Now().Add(-2 * time.Hour)

	if err := c.evictNodeToDisk(leafC); err != nil {
		t.Fatal(err)
	}
	leafC.lastUsed = time.Now().Add(-1 * time.Hour) // newest

	// Set cap to allow only the largest node (leafC) to survive.
	t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", fmt.Sprintf("%d", leafC.diskFileSize+1))

	c.enforceDiskEvictionPolicy()

	// leafA (oldest) and leafB should be deleted; leafC should remain.
	if leafA.diskFile != "" {
		t.Fatal("leafA should have been evicted from disk")
	}
	if leafB.diskFile != "" {
		t.Fatal("leafB should have been evicted from disk")
	}
	if leafC.diskFile == "" {
		t.Fatal("leafC should still be on disk")
	}
	if c.totalDiskBytes != leafC.diskFileSize {
		t.Fatalf("totalDiskBytes mismatch: got %d, want %d", c.totalDiskBytes, leafC.diskFileSize)
	}
}

func TestSaveTrieWithColdNodes(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:cold-test"
	numLayers := 1

	c := makeTestKVCache(t, numLayers)
	c.cacheDir = dir

	// Build root -> [1,2,3] with snapshots, then evict to disk.
	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = []*trieNode{c.root}

	if err := c.evictNodeToDisk(leaf); err != nil {
		t.Fatal(err)
	}

	// Verify the evicted file exists with a hash name.
	evictedPath := leaf.diskFile
	if _, err := os.Stat(evictedPath); err != nil {
		t.Fatal("evicted file should exist:", err)
	}

	// Save trie — cold node file is already hash-named, no rename needed.
	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal("saveTrie:", err)
	}

	// Evicted file should still exist (same hash name, not renamed).
	if _, err := os.Stat(evictedPath); err != nil {
		t.Fatal("evicted file should still exist (same hash name):", err)
	}

	// Load and verify the cold node round-trips.
	root, _, _, err := loadTrie(dir, modelID, numLayers)
	if err != nil {
		t.Fatal("loadTrie:", err)
	}
	if root == nil {
		t.Fatal("nil root")
	}
	if len(root.children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(root.children))
	}

	restored := root.children[0]
	if len(restored.tokens) != 3 || restored.tokens[0] != 1 {
		t.Fatalf("token mismatch: %v", restored.tokens)
	}
	if !restored.hasAllSnapshots() {
		t.Fatal("cold node should have snapshots after load")
	}
}

func TestCleanUnreferencedFiles(t *testing.T) {
	dir := t.TempDir()

	// Create a mix of referenced and unreferenced files.
	referenced := map[string]bool{
		"abc123def456789a.safetensors": true,
		"fedcba9876543210.safetensors": true,
	}
	orphaned := []string{
		"0000000000000000.safetensors",
		"1111111111111111.safetensors",
	}
	for name := range referenced {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("keep"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, name := range orphaned {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("orphan"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// Non-safetensors files should be untouched.
	if err := os.WriteFile(filepath.Join(dir, "trie.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}

	cleanUnreferencedFiles(dir, referenced)

	// Referenced files should still exist.
	for name := range referenced {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("referenced file %s should still exist", name)
		}
	}
	// Orphaned files should be removed.
	for _, name := range orphaned {
		if _, err := os.Stat(filepath.Join(dir, name)); !os.IsNotExist(err) {
			t.Fatalf("orphaned file %s should have been removed", name)
		}
	}
	// trie.json should be untouched.
	if _, err := os.Stat(filepath.Join(dir, "trie.json")); err != nil {
		t.Fatal("trie.json should still exist")
	}
}

func TestNodeFileHashDeterministic(t *testing.T) {
	root := &trieNode{}
	child := &trieNode{tokens: []int32{10, 20, 30}, parent: root, endOffset: 3}

	h1 := nodeFileHash(child)
	h2 := nodeFileHash(child)
	if h1 != h2 {
		t.Fatalf("hash should be deterministic: got %s and %s", h1, h2)
	}

	// A structurally identical node should produce the same hash.
	child2 := &trieNode{tokens: []int32{10, 20, 30}, parent: root, endOffset: 3}
	h3 := nodeFileHash(child2)
	if h1 != h3 {
		t.Fatalf("identical token paths should hash equally: got %s and %s", h1, h3)
	}
}

func TestNodeFileHashUnique(t *testing.T) {
	root := &trieNode{}

	// Sibling branches with different tokens.
	branchA := &trieNode{tokens: []int32{1, 2, 3}, parent: root, endOffset: 3}
	branchB := &trieNode{tokens: []int32{4, 5, 6}, parent: root, endOffset: 3}

	hA := nodeFileHash(branchA)
	hB := nodeFileHash(branchB)
	if hA == hB {
		t.Fatalf("different token paths should produce different hashes: both %s", hA)
	}

	// Same tokens but different depth (different cumulative path).
	mid := &trieNode{tokens: []int32{1, 2}, parent: root, endOffset: 2}
	deep := &trieNode{tokens: []int32{3}, parent: mid, endOffset: 3}
	shallow := &trieNode{tokens: []int32{1, 2, 3}, parent: root, endOffset: 3}

	hDeep := nodeFileHash(deep)
	hShallow := nodeFileHash(shallow)
	// Same cumulative path [1,2,3] → should be equal regardless of trie structure.
	if hDeep != hShallow {
		t.Fatalf("same cumulative path should hash equally: deep=%s, shallow=%s", hDeep, hShallow)
	}
}

func TestNodeFileHashStableAfterSplit(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 1
	c := makeTestKVCache(t, numLayers)

	// Build root -> [1,2,3,4] with snapshots.
	feedTokens(c, 4)
	child := c.root.appendTokens(c.root, []int32{1, 2, 3, 4}, 4)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)

	// Hash of the full [1,2,3,4] node before split.
	hashBefore := nodeFileHash(child)

	// Split at position 2: [1,2,3,4] → [1,2] → [3,4]
	// splitNode returns the new prefix parent; the original child variable
	// is modified in-place to become the tail with tokens [3,4].
	_ = splitNode(child, 2, c.caches, &c.pagedOutBytes)

	// child is now the tail [3,4] with parent [1,2] — cumulative path
	// is still [1,2,3,4], so hash should be unchanged.
	hashAfter := nodeFileHash(child)
	if hashBefore != hashAfter {
		t.Fatalf("split tail should retain cumulative path hash: before=%s, after=%s", hashBefore, hashAfter)
	}
}

func TestIncrementalSaveSkipsColdNodes(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:incremental"
	numLayers := 1

	c := makeTestKVCache(t, numLayers)
	c.cacheDir = dir

	// Build root -> [1,2,3] with snapshots, evict to disk.
	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = []*trieNode{c.root}

	if err := c.evictNodeToDisk(leaf); err != nil {
		t.Fatal(err)
	}

	coldFile := leaf.diskFile
	info1, err := os.Stat(coldFile)
	if err != nil {
		t.Fatal("cold file should exist:", err)
	}

	// Save trie — cold node should NOT be rewritten.
	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal(err)
	}

	info2, err := os.Stat(coldFile)
	if err != nil {
		t.Fatal("cold file should still exist after save:", err)
	}

	// ModTime should be unchanged (file was not rewritten).
	if !info1.ModTime().Equal(info2.ModTime()) {
		t.Fatal("cold node file should not have been rewritten during save")
	}
}

func TestCrashSafetyPartialSave(t *testing.T) {
	skipIfNoMLXTest(t)

	dir := t.TempDir()
	modelID := "sha256:crash-test"
	numLayers := 1

	// First: create a valid save.
	c := makeTestKVCache(t, numLayers)
	feedTokens(c, 3)
	child := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	child.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = []*trieNode{c.root, child}

	if err := saveTrieTo(c, dir, modelID); err != nil {
		t.Fatal(err)
	}

	// Simulate a "crash" by writing an orphaned hash file (as if a new
	// save started writing node files but crashed before writing trie.json).
	orphanName := "deadbeefdeadbeef.safetensors"
	if err := os.WriteFile(filepath.Join(dir, orphanName), []byte("orphan"), 0o600); err != nil {
		t.Fatal(err)
	}

	// Load should succeed (old trie.json + old hash files are consistent).
	root, pagedOut, referenced, err := loadTrie(dir, modelID, numLayers)
	if err != nil {
		t.Fatal("loadTrie:", err)
	}
	if root == nil {
		t.Fatal("nil root — crash should not corrupt existing save")
	}
	if pagedOut == 0 {
		t.Fatal("expected non-zero paged out bytes")
	}

	// Cleanup should remove the orphan.
	cleanUnreferencedFiles(dir, referenced)
	if _, err := os.Stat(filepath.Join(dir, orphanName)); !os.IsNotExist(err) {
		t.Fatal("orphaned file should have been cleaned up")
	}

	// Original data should still be loadable.
	if len(root.children) != 1 {
		t.Fatalf("expected 1 child, got %d", len(root.children))
	}
	if !root.children[0].hasAllSnapshots() {
		t.Fatal("child should have snapshots")
	}
}

func TestProcessDiskCompletionsOnFailure(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 1
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = t.TempDir()
	c.diskWriter = newDiskWriter()
	defer c.diskWriter.shutdown()

	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)
	c.activePath = []*trieNode{c.root}

	// Simulate a failed async write by injecting a failed result.
	leaf.diskFile = filepath.Join(c.cacheDir, "fake.safetensors")
	leaf.diskFileSize = 1000
	leaf.snapTypes = []cache.SnapshotType{cache.SnapshotTypeKV}
	leaf.setSnapshots(nil, &c.pagedOutBytes)
	c.totalDiskBytes = 1000

	c.diskWriter.results <- diskWriteResult{
		node:     leaf,
		fileSize: 1000,
		err:      fmt.Errorf("simulated write failure"),
	}

	c.processDiskCompletions()

	if leaf.diskFile != "" {
		t.Fatal("diskFile should be cleared after failure")
	}
	if c.totalDiskBytes != 0 {
		t.Fatalf("totalDiskBytes should be 0, got %d", c.totalDiskBytes)
	}
	if len(c.root.children) != 0 {
		t.Fatal("leaf should be removed from trie after failure")
	}
}

func TestAsyncEvictNodeToDisk(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 2
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = t.TempDir()
	c.diskWriter = newDiskWriter()
	defer c.diskWriter.shutdown()

	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)

	if err := c.evictNodeToDisk(leaf); err != nil {
		t.Fatal("evictNodeToDisk:", err)
	}

	if leaf.diskFile == "" {
		t.Fatal("diskFile should be set immediately")
	}
	if leaf.hasSnapshots() {
		t.Fatal("snapshots should be nil")
	}

	// Wait for background write.
	c.diskWriter.waitForFile(filepath.Base(leaf.diskFile))
	<-c.diskWriter.results

	if _, err := os.Stat(leaf.diskFile); err != nil {
		t.Fatal("file should exist after async write:", err)
	}
}

func TestLoadNodeFromDiskWaitsForInFlight(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 2
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = t.TempDir()
	c.diskWriter = newDiskWriter()
	defer c.diskWriter.shutdown()

	feedTokens(c, 3)
	leaf := c.root.appendTokens(c.root, []int32{1, 2, 3}, 3)
	snaps := make([]cache.Snapshot, numLayers)
	for j, kv := range c.caches {
		snaps[j] = kv.Snapshot(0)
	}
	leaf.setSnapshots(snaps, &c.pagedOutBytes)

	// Evict (async write is now in-flight).
	if err := c.evictNodeToDisk(leaf); err != nil {
		t.Fatal(err)
	}

	// Immediately load -- should block until write completes, then succeed.
	if err := c.loadNodeFromDisk(leaf); err != nil {
		t.Fatal("loadNodeFromDisk should succeed after waiting:", err)
	}

	if !leaf.hasSnapshots() {
		t.Fatal("snapshots should be restored")
	}

	<-c.diskWriter.results // drain
}

func TestShutdownDrainsAsyncWrites(t *testing.T) {
	skipIfNoMLXTest(t)

	numLayers := 1
	c := makeTestKVCache(t, numLayers)
	c.cacheDir = t.TempDir()
	c.diskWriter = newDiskWriter()

	var leaves []*trieNode
	for i := range 3 {
		feedTokens(c, 2)
		leaf := c.root.appendTokens(c.root, []int32{int32(i*2 + 1), int32(i*2 + 2)}, 2)
		snaps := make([]cache.Snapshot, numLayers)
		for j, kv := range c.caches {
			snaps[j] = kv.Snapshot(0)
		}
		leaf.setSnapshots(snaps, &c.pagedOutBytes)
		leaves = append(leaves, leaf)
	}

	c.activePath = []*trieNode{c.root}
	for _, leaf := range leaves {
		if err := c.evictNodeToDisk(leaf); err != nil {
			t.Fatal(err)
		}
	}

	// Shutdown: close -> wait -> drain.
	c.diskWriter.shutdown()
	c.processDiskCompletions()

	for _, leaf := range leaves {
		if _, err := os.Stat(leaf.diskFile); err != nil {
			t.Fatalf("file should exist after shutdown: %v", err)
		}
	}
}
