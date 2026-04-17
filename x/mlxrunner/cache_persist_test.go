package mlxrunner

import (
	"os"
	"strings"
	"testing"

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
