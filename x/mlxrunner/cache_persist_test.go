package mlxrunner

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

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
	if err := c.saveTrie(dir, modelID); err != nil {
		t.Fatal("saveTrie:", err)
	}

	// Verify files exist.
	if _, err := os.Stat(filepath.Join(dir, "trie.json")); err != nil {
		t.Fatal("trie.json not found:", err)
	}

	// Load into a fresh trie.
	root, pagedOut, err := loadTrie(dir, modelID, numLayers)
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

	if err := c.saveTrie(dir, "sha256:model_a"); err != nil {
		t.Fatal(err)
	}

	// Loading with a different model ID should return nil root.
	root, _, err := loadTrie(dir, "sha256:model_b", numLayers)
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

	if err := c.saveTrie(dir, modelID); err != nil {
		t.Fatal(err)
	}

	// Loading with different layer count should return nil root.
	root, _, err := loadTrie(dir, modelID, 4)
	if err != nil {
		t.Fatal(err)
	}
	if root != nil {
		t.Fatal("expected nil root for layer count mismatch")
	}
}

func TestLoadTrieNoFile(t *testing.T) {
	dir := t.TempDir()
	root, pagedOut, err := loadTrie(dir, "sha256:test", 2)
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
	if err := c.saveTrie(dir, "sha256:test"); err != nil {
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
	if err := c.saveTrie(dir, "sha256:test"); err != nil {
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

	if err := c.saveTrie(dir, modelID); err != nil {
		t.Fatal(err)
	}

	root, _, err := loadTrie(dir, modelID, numLayers)
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
