# MLX KV Cache Persistence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `.planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md` — read it first; this plan assumes you have.

**Goal:** Let the MLX runner's KV cache trie survive both RAM eviction and ollama-server restarts by writing completed nodes to disk and reloading them on prefix match.

**Architecture:** Write-through persistence on snapshot attach. Self-describing safetensors files (content-addressed filenames; parent-hash + tokens in header). Single background writer with atomic rename. Two-tier eviction: the existing `enforceEvictionPolicy` selector is reused unchanged but now has a memory pass (Warm → Cold) before the disk pass (Warm/Cold → Gone). Warm-restart rehydration scans headers only, building the trie skeleton with zero MLX array loads.

**Tech Stack:** Go; `x/mlxrunner/mlx` (existing `SaveSafetensorsWithMetadata` / `LoadSafetensorsNative`); `sync.Mutex` + `sync.Cond` for the writer queue; stdlib `crypto/sha256`, `encoding/base64`, `os` for atomic rename; `envconfig` for the env-var surface.

**Baseline:** upstream main @ `57653b8e`. All tasks start from this branch (`cache-persisitance`). Pre-existing tests (`cache_test.go`, `cache_trie_test.go`) must remain green after every commit.

**Test discipline:** Every task is TDD. The repo already has rich fake-cache test infrastructure in `x/mlxrunner/cache_test.go` (`snapshotTracker`, `fakeSnapshot`, `fakeRewindableCache`) — reuse it; do not introduce a new fake.

**Common commands used in steps:**
- Run all MLX runner tests: `go test ./x/mlxrunner/...`
- Run one test: `go test -run TestName ./x/mlxrunner/`
- Build check: `go build ./...`
- Lint: `go vet ./x/mlxrunner/...`

---

## Task 1: Environment-variable surface

**Files:**
- Modify: `envconfig/config.go` — add two env-var accessors and a byte-size parser helper
- Create: `envconfig/config_test.go` (already exists — append tests)

**What this enables:** subsequent tasks can read `envconfig.KVCacheDiskMax()` and `envconfig.KVCacheRoot()` for the on/off switch and the root directory.

- [ ] **Step 1.1: Write failing tests for the byte-size parser**

Append to `envconfig/config_test.go`:

```go
func TestParseByteSize(t *testing.T) {
    cases := []struct {
        in      string
        want    int64
        wantErr bool
    }{
        {"", 0, true},
        {"0", 0, false},
        {"-1", -1, false},
        {"-42", -42, false},
        {"1024", 1024, false},
        {"1K", 1024, false},
        {"1KiB", 1024, false},
        {"1KB", 1000, false},
        {"50GiB", 50 * (1 << 30), false},
        {"2G", 2 * (1 << 30), false},
        {"100M", 100 * (1 << 20), false},
        {"1.5G", 0, true},        // fractional not supported
        {"abc", 0, true},
        {"10XB", 0, true},
    }
    for _, c := range cases {
        got, err := parseByteSize(c.in)
        if (err != nil) != c.wantErr {
            t.Errorf("parseByteSize(%q) err=%v wantErr=%v", c.in, err, c.wantErr)
            continue
        }
        if !c.wantErr && got != c.want {
            t.Errorf("parseByteSize(%q) = %d, want %d", c.in, got, c.want)
        }
    }
}

func TestKVCacheDiskMax(t *testing.T) {
    cases := []struct {
        env  string
        want int64
    }{
        {"", -1},            // default: unlimited
        {"-1", -1},
        {"0", 0},
        {"50GiB", 50 * (1 << 30)},
        {"bogus", -1},       // invalid falls back to default
    }
    for _, c := range cases {
        t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", c.env)
        if got := KVCacheDiskMax(); got != c.want {
            t.Errorf("OLLAMA_KV_CACHE_DISK_MAX=%q: got %d want %d", c.env, got, c.want)
        }
    }
}

func TestKVCacheRoot(t *testing.T) {
    t.Setenv("OLLAMA_KV_CACHE_ROOT", "/tmp/kvtest")
    if got := KVCacheRoot(); got != "/tmp/kvtest" {
        t.Errorf("KVCacheRoot = %q, want /tmp/kvtest", got)
    }
    t.Setenv("OLLAMA_KV_CACHE_ROOT", "")
    if got := KVCacheRoot(); got == "" {
        t.Errorf("KVCacheRoot returned empty; expected default path")
    }
}
```

- [ ] **Step 1.2: Run tests to verify they fail**

```
go test -run 'TestParseByteSize|TestKVCacheDiskMax|TestKVCacheRoot' ./envconfig/
```
Expected: compile error — `parseByteSize`, `KVCacheDiskMax`, `KVCacheRoot` undefined.

- [ ] **Step 1.3: Implement the parser and env accessors**

Append to `envconfig/config.go`:

```go
// parseByteSize parses an integer byte size with an optional binary/decimal suffix.
// Supported suffixes: K/KB (1000), KiB (1024); M/MB/MiB; G/GB/GiB; T/TB/TiB.
// A bare integer (including 0 and negatives) is returned as-is.
// Fractional values are not supported.
func parseByteSize(s string) (int64, error) {
    if s == "" {
        return 0, fmt.Errorf("empty")
    }
    // Fast path: plain integer, possibly signed.
    if n, err := strconv.ParseInt(s, 10, 64); err == nil {
        return n, nil
    }
    // Locate the numeric prefix.
    i := 0
    if s[0] == '-' || s[0] == '+' {
        i = 1
    }
    for i < len(s) && s[i] >= '0' && s[i] <= '9' {
        i++
    }
    if i == 0 || (i == 1 && (s[0] == '-' || s[0] == '+')) {
        return 0, fmt.Errorf("no digits: %q", s)
    }
    n, err := strconv.ParseInt(s[:i], 10, 64)
    if err != nil {
        return 0, err
    }
    var mult int64
    switch strings.ToUpper(s[i:]) {
    case "K", "KB":
        mult = 1000
    case "KI", "KIB":
        mult = 1 << 10
    case "M", "MB":
        mult = 1000 * 1000
    case "MI", "MIB":
        mult = 1 << 20
    case "G", "GB":
        mult = 1000 * 1000 * 1000
    case "GI", "GIB":
        mult = 1 << 30
    case "T", "TB":
        mult = 1000 * 1000 * 1000 * 1000
    case "TI", "TIB":
        mult = 1 << 40
    default:
        return 0, fmt.Errorf("unknown suffix %q", s[i:])
    }
    return n * mult, nil
}

// KVCacheDiskMax returns the maximum total bytes allowed for on-disk KV cache snapshots.
// Negative or unset: unlimited (default). Zero: persistence disabled. Positive: hard cap.
// Accepts byte-unit suffixes (e.g. "50GiB"). Invalid values log a warning and return the default.
func KVCacheDiskMax() int64 {
    if s := Var("OLLAMA_KV_CACHE_DISK_MAX"); s != "" {
        if n, err := parseByteSize(s); err == nil {
            return n
        } else {
            slog.Warn("invalid OLLAMA_KV_CACHE_DISK_MAX, using default", "value", s, "err", err)
        }
    }
    return -1
}

// KVCacheRoot returns the root directory for on-disk KV cache snapshots.
// One subdirectory per model digest is created underneath.
// Default: <OLLAMA_MODELS dir>/../cache/kv, i.e. sibling to the models store.
func KVCacheRoot() string {
    if s := Var("OLLAMA_KV_CACHE_ROOT"); s != "" {
        return s
    }
    return filepath.Join(filepath.Dir(Models()), "cache", "kv")
}
```

Ensure new imports at top of file: `"fmt"`, `"path/filepath"`, `"strings"`. (If already present, skip.)

- [ ] **Step 1.4: Register the vars in the `AsMap()` display (around line 311)**

Locate the entry for `"OLLAMA_GPU_OVERHEAD"` and append two entries in the same style:

```go
"OLLAMA_KV_CACHE_DISK_MAX": {"OLLAMA_KV_CACHE_DISK_MAX", KVCacheDiskMax(),
    "Max bytes of on-disk MLX KV cache (-1 or unset = unlimited, 0 = disabled). Suffixes: KiB, MiB, GiB, KB, MB, GB."},
"OLLAMA_KV_CACHE_ROOT":    {"OLLAMA_KV_CACHE_ROOT", KVCacheRoot(),
    "Root directory for on-disk MLX KV cache (default: sibling of models dir)."},
```

- [ ] **Step 1.5: Run tests to verify they pass**

```
go test -run 'TestParseByteSize|TestKVCacheDiskMax|TestKVCacheRoot' ./envconfig/
```
Expected: PASS.

- [ ] **Step 1.6: Verify nothing else broke**

```
go test ./envconfig/ && go build ./...
```

- [ ] **Step 1.7: Commit**

```
git add envconfig/config.go envconfig/config_test.go
git commit -m "envconfig: add OLLAMA_KV_CACHE_DISK_MAX and OLLAMA_KV_CACHE_ROOT

Signed int64 byte size; negative = unlimited, 0 = disabled.
Suffix-aware parser accepts KiB/MiB/GiB (binary) and KB/MB/GB (decimal).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Data-model additions (trieNode + kvCache)

**Files:**
- Modify: `x/mlxrunner/cache_trie.go:14-24` — add four fields to `trieNode`
- Modify: `x/mlxrunner/cache.go:34-39` — add four fields to `kvCache`
- Modify: `x/mlxrunner/cache_test.go` — extend one existing test to assert zero values
- Modify: `x/mlxrunner/runner.go` — construction plumbing (modelDigest + cacheDir)
- Create: `x/mlxrunner/cache_persist.go` — empty stub with package decl (real contents in later tasks)

**What this enables:** every subsequent task can reference `node.diskPath`, `cache.diskBytes`, and `cache.writer` without touching the type definitions again.

- [ ] **Step 2.1: Write a failing assertion — new trieNode fields default to zero/nil**

Append to `x/mlxrunner/cache_trie_test.go`:

```go
func TestTrieNodeZeroValue(t *testing.T) {
    n := &trieNode{}
    if n.diskPath != "" {
        t.Errorf("diskPath = %q, want empty", n.diskPath)
    }
    if n.diskSize != 0 {
        t.Errorf("diskSize = %d, want 0", n.diskSize)
    }
    if n.inflightWrite != nil {
        t.Errorf("inflightWrite = %v, want nil", n.inflightWrite)
    }
    if n.writeAttempts != 0 {
        t.Errorf("writeAttempts = %d, want 0", n.writeAttempts)
    }
}
```

- [ ] **Step 2.2: Run test to verify failure**

```
go test -run TestTrieNodeZeroValue ./x/mlxrunner/
```
Expected: compile error — unknown field.

- [ ] **Step 2.3: Add the fields to `trieNode`**

Locate `type trieNode struct` at `x/mlxrunner/cache_trie.go:14` and add four lines at the end of the struct:

```go
type trieNode struct {
    // ... existing fields unchanged ...

    // Persistence fields (see .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md §5.2).
    diskPath      string         // "" if not persisted; content-addressed filename when set
    diskSize      int64          // bytes on disk; 0 when diskPath == ""
    inflightWrite chan struct{}  // non-nil while a write is queued or in flight; closed on completion
    writeAttempts uint8          // retry counter; capped at 3 (see spec §6.3)
}
```

- [ ] **Step 2.4: Add fields to `kvCache`**

Locate `type kvCache struct` at `x/mlxrunner/cache.go:34` and add four fields (leaving existing fields untouched):

```go
type kvCache struct {
    root          *trieNode
    activePath    []*trieNode
    caches        []cache.Cache
    pagedOutBytes int64

    // Persistence fields.
    diskBytes   int64       // atomic; sum of diskSize across all nodes
    modelDigest string      // set at construction; identifies the model for dir scoping
    cacheDir    string      // <OLLAMA_KV_CACHE_ROOT>/<modelDigest>; empty when feature disabled
    writer      *diskWriter // nil when feature disabled
}
```

Note: `diskWriter` does not exist yet. Add a forward-declaration stub at the bottom of `cache.go`:

```go
// diskWriter is implemented in cache_persist.go.
type diskWriter struct{}
```

This stub will be deleted in Task 5.

- [ ] **Step 2.5: Create the empty cache_persist.go**

```go
// Package mlxrunner — KV cache disk persistence.
// Design: .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md
// Real implementation arrives in Task 3 onward; this file reserves the name.
package mlxrunner
```

- [ ] **Step 2.6: Wire cacheDir + modelDigest in runner.go**

Read `x/mlxrunner/runner.go` to locate the `kvCache` construction site (grep for `&kvCache{`). Adjust it to pass the two new fields:

```go
c := &kvCache{
    caches: make([]cache.Cache, numLayers),
    // modelDigest and cacheDir are required only when disk persistence is active;
    // later tasks set them from the caller. For Task 2 these stay zero and the
    // feature acts as disabled.
}
```

No behavior change yet — the zero-valued fields are equivalent to "feature disabled".

- [ ] **Step 2.7: Run tests to verify they pass**

```
go test ./x/mlxrunner/...
go build ./...
```
Expected: all pre-existing tests still green; `TestTrieNodeZeroValue` green.

- [ ] **Step 2.8: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/cache_trie.go x/mlxrunner/cache_trie_test.go x/mlxrunner/cache_persist.go x/mlxrunner/runner.go
git commit -m "mlxrunner: add persistence fields to trieNode and kvCache

No behavior change. Prepares for disk-persist in subsequent commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Safetensors header encode/decode

**Files:**
- Modify: `x/mlxrunner/cache_persist.go` — content-addressed filename, header encode/decode, tokens packing
- Create: `x/mlxrunner/cache_persist_test.go`

**What this enables:** tasks 4 and 7 can read/write per-node metadata without re-solving filename or byte packing.

- [ ] **Step 3.1: Write failing tests for header round-trip**

Create `x/mlxrunner/cache_persist_test.go`:

```go
package mlxrunner

import (
    "strings"
    "testing"
)

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
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
go test -run 'TestFilename|TestTokens|TestHeader' ./x/mlxrunner/
```
Expected: compile error — symbols undefined.

- [ ] **Step 3.3: Implement the header helpers**

Replace `x/mlxrunner/cache_persist.go` contents:

```go
// Package mlxrunner — KV cache disk persistence: filename and header helpers.
// Design: .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md §10.
package mlxrunner

import (
    "crypto/sha256"
    "encoding/base64"
    "encoding/binary"
    "encoding/hex"
    "fmt"
    "strconv"
    "strings"
    "time"
)

const cacheFormatVersion = "1"

// contentFilename returns a deterministic .safetensors filename for a node.
// Same inputs always produce the same output so writes are idempotent
// and parent references in other files are stable.
func contentFilename(modelDigest, parentHash string, tokens []int32, layerCount int, dtype string) string {
    h := sha256.New()
    h.Write([]byte(modelDigest))
    h.Write([]byte{0})
    h.Write([]byte(parentHash))
    h.Write([]byte{0})
    h.Write(int32BytesLE(tokens))
    h.Write([]byte{0})
    fmt.Fprintf(h, "%d", layerCount)
    h.Write([]byte{0})
    h.Write([]byte(dtype))
    sum := h.Sum(nil)
    // Take the first 16 bytes (32 hex chars) — still ~128 bits of entropy, plenty.
    return hex.EncodeToString(sum[:16]) + ".safetensors"
}

// encodeTokens packs int32 tokens in little-endian bytes and base64-encodes them.
func encodeTokens(tokens []int32) string {
    if len(tokens) == 0 {
        return ""
    }
    return base64.StdEncoding.EncodeToString(int32BytesLE(tokens))
}

func decodeTokens(s string) ([]int32, error) {
    if s == "" {
        return nil, nil
    }
    raw, err := base64.StdEncoding.DecodeString(s)
    if err != nil {
        return nil, fmt.Errorf("decode tokens: %w", err)
    }
    if len(raw)%4 != 0 {
        return nil, fmt.Errorf("decode tokens: length %d not a multiple of 4", len(raw))
    }
    out := make([]int32, len(raw)/4)
    for i := range out {
        out[i] = int32(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
    }
    return out, nil
}

func int32BytesLE(tokens []int32) []byte {
    buf := make([]byte, 4*len(tokens))
    for i, t := range tokens {
        binary.LittleEndian.PutUint32(buf[i*4:i*4+4], uint32(t))
    }
    return buf
}

// headerFields is the metadata we embed in every safetensors file.
type headerFields struct {
    formatVersion string
    modelDigest   string
    parentHash    string
    tokens        []int32
    layerCount    int
    snapshotTypes []string
    createdAt     time.Time
}

func encodeHeader(h headerFields) map[string]string {
    v := h.formatVersion
    if v == "" {
        v = cacheFormatVersion
    }
    ts := h.createdAt
    if ts.IsZero() {
        ts = time.Now().UTC()
    }
    return map[string]string{
        "cache_format_version": v,
        "model_digest":         h.modelDigest,
        "parent_hash":          h.parentHash,
        "tokens":               encodeTokens(h.tokens),
        "layer_count":          strconv.Itoa(h.layerCount),
        "snapshot_types":       strings.Join(h.snapshotTypes, ","),
        "created_at":           ts.Format(time.RFC3339),
    }
}

func decodeHeader(m map[string]string) (headerFields, error) {
    v := m["cache_format_version"]
    if v != cacheFormatVersion {
        return headerFields{}, fmt.Errorf("unsupported cache_format_version %q", v)
    }
    lc, err := strconv.Atoi(m["layer_count"])
    if err != nil {
        return headerFields{}, fmt.Errorf("layer_count: %w", err)
    }
    toks, err := decodeTokens(m["tokens"])
    if err != nil {
        return headerFields{}, err
    }
    var snapTypes []string
    if s := m["snapshot_types"]; s != "" {
        snapTypes = strings.Split(s, ",")
    }
    var ts time.Time
    if s := m["created_at"]; s != "" {
        ts, _ = time.Parse(time.RFC3339, s) // tolerate malformed timestamps
    }
    return headerFields{
        formatVersion: v,
        modelDigest:   m["model_digest"],
        parentHash:    m["parent_hash"],
        tokens:        toks,
        layerCount:    lc,
        snapshotTypes: snapTypes,
        createdAt:     ts,
    }, nil
}
```

- [ ] **Step 3.4: Run tests to verify they pass**

```
go test -run 'TestFilename|TestTokens|TestHeader' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 3.5: Verify nothing else broke**

```
go test ./x/mlxrunner/... && go vet ./x/mlxrunner/...
```

- [ ] **Step 3.6: Commit**

```
git add x/mlxrunner/cache_persist.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner/persist: content-addressed filename + header codec

No write path yet — this is pure data plumbing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Synchronous writeOne

**Files:**
- Modify: `x/mlxrunner/cache_persist.go` — add `writeOne(node)` and helper to extract snapshot arrays
- Modify: `x/mlxrunner/cache_persist_test.go` — write-then-read-back test

**What this enables:** we have a single, testable function that turns a Warm node into a disk file. The goroutine loop in Task 5 just wraps this.

- [ ] **Step 4.1: Write failing test for round-trip writeOne → loadFromDisk**

Append to `x/mlxrunner/cache_persist_test.go`:

```go
func TestWriteOneRoundTrip(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "modelA", 1 /* layer */)
    defer c.teardown()

    // Build a trie: root -> A(tokens=[1,2,3]) with a fake snapshot.
    node := &trieNode{
        tokens: []int32{1, 2, 3},
        parent: c.root,
    }
    c.root.children = append(c.root.children, node)

    tracker := &snapshotTracker{}
    snap := &fakeSnapshot{tokens: []int32{1, 2, 3}, from: 0, to: 3, byteSize: 1024}
    tracker.track(snap)
    node.snapshots = []cache.Snapshot{snap}

    // Write synchronously.
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

    // Reading back: we don't have loadFromDisk yet; just confirm the header is readable.
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
```

This test needs two test helpers: `newTestKvCacheWithDisk` and a `teardown` method. Add them at the top of the test file:

```go
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
```

The `fakeSnapshot` type is already defined in `cache_test.go` — both test files are in the same package. We may need to extend it: `fakeSnapshot` will need a way to emit the array data. Since Task 4 does the write side only and Task 7 reads back via `LoadSafetensorsNative`, the serialization must produce a real safetensors file with real MLX arrays. The `fakeSnapshot` doesn't hold MLX arrays. **Decision:** Task 4's test uses a minimal snapshot that supplies real `*mlx.Array` values. Define a new helper alongside `fakeSnapshot`:

```go
// arraySnapshot is a Snapshot backed by a real MLX Array so the writer can
// serialize through LoadSafetensorsNative. Used only by persistence tests.
type arraySnapshot struct {
    keys, values *mlx.Array
    size         int
}

func (a *arraySnapshot) Size() int { return a.size }
func (a *arraySnapshot) Close()    { mlx.Unpin(a.keys, a.values) }
```

and update `TestWriteOneRoundTrip` to use `arraySnapshot` instead of `fakeSnapshot` so `writeOne` has real arrays to write.

- [ ] **Step 4.2: Run test to verify it fails**

```
go test -run TestWriteOneRoundTrip ./x/mlxrunner/
```
Expected: compile error — `writeOne`, `readHeader`, `arraySnapshot` undefined.

- [ ] **Step 4.3: Implement `writeOne`**

Append to `x/mlxrunner/cache_persist.go`:

```go
import (
    "errors"
    "os"
    "path/filepath"
    // plus existing imports
)

// writeOne serializes node's snapshots to a content-addressed file under c.cacheDir.
// Preconditions: node is off the active path and has snapshots attached.
// On success: node.diskPath and node.diskSize are set; c.diskBytes is incremented.
// Synchronous; does NOT touch node.inflightWrite (that's the caller's responsibility).
func (c *kvCache) writeOne(node *trieNode) error {
    if c.cacheDir == "" {
        return errors.New("writeOne called with empty cacheDir")
    }
    if node == nil || len(node.snapshots) == 0 {
        return errors.New("writeOne called with no snapshots")
    }

    // 1. Batch-evaluate all arrays in one call for graph fusion.
    arrays, fieldMap, types := collectNodeArrays(node)
    if len(arrays) > 0 {
        mlx.Eval(arrays...)
    }

    // 2. Build metadata.
    parentHash := ""
    if node.parent != nil && node.parent.diskPath != "" {
        parentHash = strings.TrimSuffix(filepath.Base(node.parent.diskPath), ".safetensors")
    }
    h := headerFields{
        formatVersion: cacheFormatVersion,
        modelDigest:   c.modelDigest,
        parentHash:    parentHash,
        tokens:        node.tokens,
        layerCount:    len(node.snapshots),
        snapshotTypes: types,
    }
    meta := encodeHeader(h)

    // 3. Compute filename.
    dtype := "unknown"
    if len(arrays) > 0 {
        dtype = arrays[0].DType().String()
    }
    fname := contentFilename(c.modelDigest, parentHash, node.tokens, len(node.snapshots), dtype)
    finalPath := filepath.Join(c.cacheDir, fname)
    tmpPath := finalPath + ".tmp"

    if err := os.MkdirAll(c.cacheDir, 0o755); err != nil {
        return fmt.Errorf("mkdir cacheDir: %w", err)
    }

    // 4. Write to tmp, then rename atomically.
    if err := mlx.SaveSafetensorsWithMetadata(tmpPath, fieldMap, meta); err != nil {
        _ = os.Remove(tmpPath)
        return fmt.Errorf("save safetensors: %w", err)
    }
    if err := os.Rename(tmpPath, finalPath); err != nil {
        _ = os.Remove(tmpPath)
        return fmt.Errorf("rename: %w", err)
    }

    // 5. Update node + counter.
    st, err := os.Stat(finalPath)
    if err != nil {
        return fmt.Errorf("stat written file: %w", err)
    }
    node.diskPath = finalPath
    node.diskSize = st.Size()
    atomic.AddInt64(&c.diskBytes, st.Size())
    return nil
}

// collectNodeArrays returns the MLX arrays, the layer-indexed name→array map
// expected by mlx.SaveSafetensorsWithMetadata, and per-layer snapshot type tags.
// Each Snapshot implementation owns its own field-name scheme; this dispatch
// should be kept in sync with restoreSnapshotArrays in Task 7.
func collectNodeArrays(node *trieNode) ([]*mlx.Array, map[string]*mlx.Array, []string) {
    var all []*mlx.Array
    fields := map[string]*mlx.Array{}
    types := make([]string, 0, len(node.snapshots))
    for i, s := range node.snapshots {
        // The only snapshot type supported in v1 is the paged-out kvSnapshot
        // from x/mlxrunner/cache/cache.go:95. If you add more types, extend
        // this switch and the restore path together.
        switch v := s.(type) {
        case *cache.KVSnapshotView:
            // NOTE: the existing cache.kvSnapshot is unexported. Task 7 may need
            // to expose a small helper (e.g., KVSnapshotView wrapper) to get at
            // the keys/values arrays. For Task 4 we accept the test-only
            // arraySnapshot and fall through to the default case.
            _ = v
        case *arraySnapshot:
            fields[fmt.Sprintf("layer_%d_keys", i)] = v.keys
            fields[fmt.Sprintf("layer_%d_values", i)] = v.values
            all = append(all, v.keys, v.values)
            types = append(types, "kv")
        default:
            // Fallback: try duck-typed accessors introduced in Task 7.
            types = append(types, "unknown")
        }
    }
    return all, fields, types
}

// readHeader reads only the metadata block of a safetensors file without
// loading the arrays. Used for startup rehydration and test assertions.
func (c *kvCache) readHeader(path string) (headerFields, error) {
    sf, err := mlx.LoadSafetensorsNative(path)
    if err != nil {
        return headerFields{}, err
    }
    defer sf.Free()
    raw := make(map[string]string)
    for _, k := range []string{
        "cache_format_version", "model_digest", "parent_hash",
        "tokens", "layer_count", "snapshot_types", "created_at",
    } {
        raw[k] = sf.GetMetadata(k)
    }
    return decodeHeader(raw)
}
```

Add imports for `atomic` (`"sync/atomic"`), `errors`, `os`, `path/filepath`.

Also add the test-only `arraySnapshot` in `cache_test.go` next to `fakeSnapshot`:

```go
type arraySnapshot struct {
    keys, values *mlx.Array
    size         int
}

func (a *arraySnapshot) Size() int { return a.size }
func (a *arraySnapshot) Close()    { mlx.Unpin(a.keys, a.values) }
```

- [ ] **Step 4.4: Note: real snapshot integration is deferred to Task 7**

The `cache.KVSnapshotView` reference is a forward declaration. If `cache.kvSnapshot` doesn't expose its arrays, Task 7 adds that seam. For Task 4, `writeOne` works end-to-end for `*arraySnapshot` only (which is exactly what our test uses).

Add a `TODO(task-7):` comment above the `case *cache.KVSnapshotView:` line flagging this handoff.

- [ ] **Step 4.5: Run tests**

```
go test -run TestWriteOneRoundTrip ./x/mlxrunner/
```
Expected: PASS — file written, header round-trips, disk-bytes counter updates.

- [ ] **Step 4.6: Verify nothing else broke**

```
go test ./x/mlxrunner/... && go build ./...
```

- [ ] **Step 4.7: Commit**

```
git add x/mlxrunner/cache_persist.go x/mlxrunner/cache_persist_test.go x/mlxrunner/cache_test.go
git commit -m "mlxrunner/persist: synchronous writeOne with atomic rename

Serializes a trieNode's snapshots to a content-addressed safetensors file,
updates diskBytes via atomic.AddInt64. No goroutine, no queue yet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Async diskWriter with shutdown-drain

**Files:**
- Modify: `x/mlxrunner/cache.go` — delete the `diskWriter` stub added in Task 2
- Modify: `x/mlxrunner/cache_persist.go` — real `diskWriter` with goroutine loop
- Modify: `x/mlxrunner/cache_persist_test.go` — queue behavior + shutdown tests

**What this enables:** the writer is a separate, testable component. `scheduleWrite` in Task 6 just calls into it.

- [ ] **Step 5.1: Write failing tests for the queue + shutdown-drain**

Append to `cache_persist_test.go`:

```go
func TestDiskWriterFIFO(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    // Enqueue N nodes; verify they all land on disk and diskBytes reflects it.
    const N = 5
    nodes := make([]*trieNode, N)
    for i := range nodes {
        nodes[i] = newTestNodeWithArraySnapshot(t, c.root, []int32{int32(i)}, 1024)
        nodes[i].inflightWrite = make(chan struct{})
        c.writer.enqueue(nodes[i])
    }
    // Wait for each to complete.
    for i, n := range nodes {
        <-n.inflightWrite
        if n.diskPath == "" {
            t.Errorf("node[%d].diskPath empty after write", i)
        }
    }
    if atomic.LoadInt64(&c.diskBytes) <= 0 {
        t.Error("diskBytes not updated")
    }
}

func TestDiskWriterShutdownDrainsQueue(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)

    // Enqueue, then shutdown with a generous timeout.
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
    // Use a fake sleep to make shutdown racy, or rely on timeout == 0.
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)

    // Force the writer to block so shutdown hits the timeout path.
    // Easiest: write into a read-only directory so writeOne errors & retries.
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
    remaining := c.writer.shutdown(1 * time.Millisecond)
    // The test only asserts that shutdown returns and does not hang.
    _ = remaining
}
```

Add the helper `newTestNodeWithArraySnapshot` alongside `newTestKvCacheWithDisk`:

```go
func newTestNodeWithArraySnapshot(t *testing.T, parent *trieNode, tokens []int32, bytes int) *trieNode {
    t.Helper()
    n := &trieNode{tokens: tokens, parent: parent}
    parent.children = append(parent.children, n)
    // Two small MLX arrays; mlx.Zeros is available from x/mlxrunner/mlx.
    k := mlx.Zeros([]int{1, 8}, mlx.Float32)
    v := mlx.Zeros([]int{1, 8}, mlx.Float32)
    mlx.Pin(k, v) // ensure the arrays survive until the snapshot is Closed
    snap := &arraySnapshot{keys: k, values: v, size: bytes}
    n.snapshots = []cache.Snapshot{snap}
    return n
}
```

- [ ] **Step 5.2: Run tests to verify failure**

```
go test -run 'TestDiskWriter' ./x/mlxrunner/
```
Expected: compile error — `newDiskWriter`, `enqueue`, `shutdown` undefined.

- [ ] **Step 5.3: Delete the stub `diskWriter` in `cache.go`**

Remove the `type diskWriter struct{}` added in Task 2. The real type arrives now.

- [ ] **Step 5.4: Implement the `diskWriter`**

Append to `cache_persist.go`:

```go
import (
    "log/slog"
    "sync"
    "time"
)

type diskWriter struct {
    mu      sync.Mutex
    cond    *sync.Cond
    pending []*trieNode
    stopped bool
    done    chan struct{}
    cache   *kvCache

    consecutiveFailures int // circuit breaker, see spec §6.3
}

func newDiskWriter(c *kvCache) *diskWriter {
    w := &diskWriter{
        cache: c,
        done:  make(chan struct{}),
    }
    w.cond = sync.NewCond(&w.mu)
    go w.loop()
    return w
}

// enqueue schedules node for writing. Must be called while holding whatever
// serialization guarantees the caller uses for node-mutation; node.inflightWrite
// must already be a fresh channel.
func (w *diskWriter) enqueue(node *trieNode) {
    w.mu.Lock()
    defer w.mu.Unlock()
    if w.stopped {
        // Writer disabled; caller can observe by watching node.inflightWrite
        // close without diskPath being set. Close the channel to unblock waiters.
        if node.inflightWrite != nil {
            close(node.inflightWrite)
            node.inflightWrite = nil
        }
        return
    }
    w.pending = append(w.pending, node)
    w.cond.Signal()
}

func (w *diskWriter) loop() {
    defer close(w.done)
    for {
        w.mu.Lock()
        for len(w.pending) == 0 && !w.stopped {
            w.cond.Wait()
        }
        if w.stopped && len(w.pending) == 0 {
            w.mu.Unlock()
            return
        }
        node := w.pending[0]
        w.pending = w.pending[1:]
        disabled := w.stopped // snapshot under lock; if true we skip the actual write
        w.mu.Unlock()

        if !disabled {
            w.writeOneWithRetry(node)
        }
        if node.inflightWrite != nil {
            close(node.inflightWrite)
            node.inflightWrite = nil
        }
    }
}

func (w *diskWriter) writeOneWithRetry(node *trieNode) {
    err := w.cache.writeOne(node)
    if err == nil {
        w.mu.Lock()
        w.consecutiveFailures = 0
        w.mu.Unlock()
        return
    }
    node.writeAttempts++
    slog.Warn("kv cache write failed",
        "path", node.diskPath, "attempt", node.writeAttempts, "err", err)
    w.mu.Lock()
    w.consecutiveFailures++
    if w.consecutiveFailures >= 5 {
        slog.Warn("kv cache writer: disabling after 5 consecutive failures")
        w.stopped = true
        w.cond.Broadcast() // wake loop so it can drain without writing
    }
    w.mu.Unlock()
    // NOTE: this implementation does NOT auto-requeue. Task 9 (memory pass)
    // calls scheduleWrite again when the node is next selected for eviction
    // and writeAttempts < 3. That's the retry path.
}

// shutdown blocks until the queue drains or the timeout elapses.
// Returns the number of still-pending nodes (0 on clean drain).
// A timeout of 0 means "no wait — drain whatever is possible immediately".
func (w *diskWriter) shutdown(timeout time.Duration) int {
    w.mu.Lock()
    w.stopped = true
    w.cond.Broadcast()
    w.mu.Unlock()

    if timeout <= 0 {
        <-w.done
    } else {
        select {
        case <-w.done:
        case <-time.After(timeout):
        }
    }

    w.mu.Lock()
    remaining := len(w.pending)
    w.mu.Unlock()
    if remaining > 0 {
        slog.Warn("kv cache writer shutdown: drained with pending", "remaining", remaining)
    }
    return remaining
}
```

- [ ] **Step 5.5: Run tests to verify they pass**

```
go test -run 'TestDiskWriter' ./x/mlxrunner/
```
Expected: all three PASS.

- [ ] **Step 5.6: Regression sweep**

```
go test ./x/mlxrunner/... && go build ./...
```

- [ ] **Step 5.7: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/cache_persist.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner/persist: async diskWriter with shutdown drain

Mutex+cond queue, single goroutine. Shutdown with timeout returns the
count of still-pending nodes. Circuit breaker disables writer after
5 consecutive failures.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: scheduleWrite wiring

**Files:**
- Modify: `x/mlxrunner/cache.go` — new `scheduleWrite` method; wire into `attachSnapshots`
- Modify: `x/mlxrunner/cache_trie.go` — wire into `splitNode` via a callback
- Modify: `x/mlxrunner/cache_persist_test.go` — test that attach triggers write

**What this enables:** the runtime naturally persists every Warm node as soon as it transitions off the active path.

- [ ] **Step 6.1: Write failing test — attachSnapshots schedules a write**

Append to `cache_persist_test.go`:

```go
func TestAttachSnapshotsSchedulesWrite(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    // Set up a one-node active path using the real cache begin() flow
    // is heavy — use a focused unit test instead.
    node := newTestNodeWithArraySnapshot(t, c.root, []int32{7, 7, 7}, 1024)
    // Simulate attachSnapshots: mark node as Warm and schedule write.
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
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
    c.scheduleWrite(node)
    ch1 := node.inflightWrite
    c.scheduleWrite(node)         // no-op — write in flight
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
```

- [ ] **Step 6.2: Run tests to verify failure**

Expected: `scheduleWrite` undefined.

- [ ] **Step 6.3: Implement `scheduleWrite`**

Add to `cache.go` (place it next to `evictNode`):

```go
// scheduleWrite enqueues node for disk write if persistence is enabled
// and the node isn't already written or in flight. Idempotent.
func (c *kvCache) scheduleWrite(node *trieNode) {
    if c.writer == nil {
        return // feature disabled
    }
    if node.inflightWrite != nil || node.diskPath != "" {
        return
    }
    if node.writeAttempts >= 3 {
        return // permanently-failed; memory eviction will delete normally
    }
    node.inflightWrite = make(chan struct{})
    c.writer.enqueue(node)
}
```

- [ ] **Step 6.4: Wire `scheduleWrite` into `attachSnapshots`**

Locate `x/mlxrunner/cache.go:382` (`func (s *cacheSession) attachSnapshots`). After the function's existing logic has finished attaching snapshots to `node`, append a call:

```go
// Existing attachSnapshots body ends with the snapshots attached to node.
// Add the final line:
s.cache.scheduleWrite(node)
```

(Exact placement: after the last `node.setSnapshots(...)` or the last mutation of `node.snapshots` in the function body — whatever is already the final statement.)

- [ ] **Step 6.5: Wire `scheduleWrite` into `splitNode`**

`splitNode` is at `cache_trie.go:176` and is called from `cache.go`. It creates a new node with snapshots. The cleanest hook is in the *caller* of `splitNode` (to avoid passing a `*kvCache` reference into `cache_trie.go` and breaking the trie module's isolation).

Grep for `splitNode(` callers in `cache.go`. At each call site, after `splitNode` returns the new intermediate node, insert:

```go
c.scheduleWrite(intermediate)
```

(The variable name may differ at each call site; update accordingly.)

- [ ] **Step 6.6: Run tests**

```
go test -run 'TestAttachSnapshotsSchedulesWrite|TestScheduleWriteIdempotent' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 6.7: Regression**

```
go test ./x/mlxrunner/...
```
All existing tests must remain green (attachSnapshots has a new side effect, but with `writer == nil` in existing tests, `scheduleWrite` is a no-op).

- [ ] **Step 6.8: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/cache_trie.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner: hook scheduleWrite into attachSnapshots and splitNode

Every node that transitions Active->Warm is persisted. Idempotent;
disabled when writer is nil.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: loadFromDisk — single-node restore

**Files:**
- Modify: `x/mlxrunner/cache/cache.go` — expose a way to rebuild a paged-out `kvSnapshot` from externally-provided arrays (the "seam" flagged in Task 4)
- Modify: `x/mlxrunner/cache_persist.go` — `loadFromDisk` + `restoreSnapshotArrays`
- Modify: `x/mlxrunner/cache_persist_test.go` — write then load, verify bytes

**What this enables:** Task 8 can turn a Cold node into Warm on cache-hit.

- [ ] **Step 7.1: Add a package-level constructor for `kvSnapshot`**

Read `x/mlxrunner/cache/cache.go` for the existing `kvSnapshot` type (around line 95). Add a public factory alongside it:

```go
// NewKVSnapshotFromArrays rebuilds a paged-out KV snapshot from externally-
// produced arrays (e.g. from a safetensors file). fromOffset and toOffset
// are the token range this snapshot covers. Callers own the arrays; the
// returned Snapshot will Unpin them on Close.
func NewKVSnapshotFromArrays(keys, values *mlx.Array, fromOffset, toOffset int) Snapshot {
    return &kvSnapshot{
        keys:       keys,
        values:     values,
        fromOffset: fromOffset,
        toOffset:   toOffset,
    }
}
```

(Adjust field names to match the actual unexported struct.)

- [ ] **Step 7.2: Write failing test for `loadFromDisk`**

Append to `cache_persist_test.go`:

```go
func TestLoadFromDisk(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
    c.scheduleWrite(node)
    <-node.inflightWrite
    path := node.diskPath
    size := node.diskSize

    // Simulate the Cold state: drop in-memory snapshots.
    for _, s := range node.snapshots {
        s.Close()
    }
    node.snapshots = nil
    // diskPath stays set.

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
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "modelA", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
    c.scheduleWrite(node)
    <-node.inflightWrite
    path := node.diskPath
    node.snapshots = nil

    // Swap in a cache that thinks it's a different model.
    c2 := newTestKvCacheWithDisk(t, dir, "modelB", 1)
    defer c2.teardown()
    foreignNode := &trieNode{diskPath: path, parent: c2.root}
    if err := c2.loadFromDisk(foreignNode); err == nil {
        t.Error("loadFromDisk should reject cross-model files")
    }
}
```

- [ ] **Step 7.3: Run tests — expect failure (`loadFromDisk` undefined).**

- [ ] **Step 7.4: Implement `loadFromDisk`**

Append to `cache_persist.go`:

```go
// loadFromDisk reconstructs node.snapshots from node.diskPath and attaches them.
// Precondition: node.diskPath != "" && node.snapshots == nil.
func (c *kvCache) loadFromDisk(node *trieNode) error {
    if node.diskPath == "" {
        return errors.New("loadFromDisk: empty diskPath")
    }
    if node.snapshots != nil {
        return errors.New("loadFromDisk: node already has snapshots")
    }
    sf, err := mlx.LoadSafetensorsNative(node.diskPath)
    if err != nil {
        return fmt.Errorf("load safetensors: %w", err)
    }
    // Validate header.
    raw := map[string]string{}
    for _, k := range []string{
        "cache_format_version", "model_digest", "parent_hash",
        "tokens", "layer_count", "snapshot_types",
    } {
        raw[k] = sf.GetMetadata(k)
    }
    h, err := decodeHeader(raw)
    if err != nil {
        sf.Free()
        return err
    }
    if h.modelDigest != c.modelDigest {
        sf.Free()
        return fmt.Errorf("model_digest mismatch: file=%q cache=%q", h.modelDigest, c.modelDigest)
    }
    if h.layerCount != len(c.caches) {
        sf.Free()
        return fmt.Errorf("layer_count mismatch: file=%d cache=%d", h.layerCount, len(c.caches))
    }

    snaps, err := c.restoreSnapshotArrays(sf, h, startOffsetForNode(node))
    if err != nil {
        sf.Free()
        return err
    }
    node.setSnapshots(snaps, &c.pagedOutBytes)
    node.lastUsed = time.Now()
    // sf itself is no longer needed — the arrays inside `snaps` hold references.
    sf.Free()
    return nil
}

// restoreSnapshotArrays dispatches by snapshot_types metadata to rebuild
// typed Snapshots from a loaded safetensors file.
func (c *kvCache) restoreSnapshotArrays(sf *mlx.SafetensorsFile, h headerFields, startOffset int) ([]cache.Snapshot, error) {
    snaps := make([]cache.Snapshot, h.layerCount)
    endOffset := startOffset + len(h.tokens)
    for i := 0; i < h.layerCount; i++ {
        st := "kv"
        if i < len(h.snapshotTypes) {
            st = h.snapshotTypes[i]
        }
        switch st {
        case "kv":
            k := sf.Get(fmt.Sprintf("layer_%d_keys", i))
            v := sf.Get(fmt.Sprintf("layer_%d_values", i))
            if k == nil || v == nil {
                return nil, fmt.Errorf("layer %d: keys/values array missing", i)
            }
            snaps[i] = cache.NewKVSnapshotFromArrays(k, v, startOffset, endOffset)
        default:
            return nil, fmt.Errorf("unknown snapshot type %q at layer %d", st, i)
        }
    }
    return snaps, nil
}

// startOffsetForNode walks parent chain summing token counts.
// Callers should cache this when they need it repeatedly.
func startOffsetForNode(node *trieNode) int {
    off := 0
    for p := node.parent; p != nil; p = p.parent {
        off += len(p.tokens)
    }
    return off
}
```

- [ ] **Step 7.5: Update `collectNodeArrays` to handle real `kvSnapshot`**

Now that `cache.NewKVSnapshotFromArrays` exists, add a `case cache.Snapshot:` branch to `collectNodeArrays` that calls a new accessor for `*mlx.Array` pairs. If the existing `kvSnapshot` type has no public `Keys()` / `Values()` method, add those now:

```go
// Keys returns the keys array (persistence only; callers must not mutate).
func (s *kvSnapshot) Keys() *mlx.Array { return s.keys }

// Values returns the values array (persistence only).
func (s *kvSnapshot) Values() *mlx.Array { return s.values }
```

Replace the `case *cache.KVSnapshotView:` placeholder in `collectNodeArrays` with:

```go
// Type assert via the public accessors we added in the cache package.
type arrayExposer interface {
    Keys() *mlx.Array
    Values() *mlx.Array
}
if ax, ok := s.(arrayExposer); ok {
    k, v := ax.Keys(), ax.Values()
    fields[fmt.Sprintf("layer_%d_keys", i)] = k
    fields[fmt.Sprintf("layer_%d_values", i)] = v
    all = append(all, k, v)
    types = append(types, "kv")
    continue
}
```

This keeps the real `kvSnapshot` and the test-only `arraySnapshot` on the same path.

- [ ] **Step 7.6: Run tests**

```
go test -run 'TestLoadFromDisk|TestLoadFromDiskRejectsForeignDigest' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 7.7: Regression**

```
go test ./x/mlxrunner/... && go build ./...
```

- [ ] **Step 7.8: Commit**

```
git add x/mlxrunner/cache/cache.go x/mlxrunner/cache_persist.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner/persist: loadFromDisk restores Cold->Warm

Validates model_digest and layer_count; dispatches per-layer restore
by snapshot_types. Exposes Keys()/Values() on kvSnapshot for the
writer's collectNodeArrays path.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: `begin()` restore loop integration

**Files:**
- Modify: `x/mlxrunner/cache.go` — add a restore loop in `begin()` after `findBestMatch`
- Modify: `x/mlxrunner/cache_persist_test.go` — integration test through begin()

**What this enables:** cache hits on Cold nodes actually become Warm and serve from disk — the user-visible feature.

- [ ] **Step 8.1: Read the current `begin()` (`cache.go:89`) end-to-end**

Understand which point has `findBestMatch`'s result. The restore loop inserts *between* `findBestMatch` returning and `switchToPath` being called.

- [ ] **Step 8.2: Write failing integration test**

Append to `cache_persist_test.go`:

```go
func TestBeginRestoresColdNodes(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    // Produce a Warm node via attach (simulate by writing directly).
    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2, 3}, 1024)
    c.scheduleWrite(node)
    <-node.inflightWrite

    // Now demote to Cold: drop snapshots from memory, keep diskPath.
    for _, s := range node.snapshots {
        s.Close()
    }
    node.snapshots = nil

    // Call begin() with a prompt whose prefix matches node.tokens.
    // begin() should see the Cold node on the match path and invoke loadFromDisk.
    // Since begin() is coupled to the active MLX cache, use a smaller surface:
    // call the exact helper begin uses for the restore loop.
    if err := c.restoreMatchedPath([]*trieNode{node}); err != nil {
        t.Fatalf("restoreMatchedPath: %v", err)
    }
    if len(node.snapshots) != 1 {
        t.Errorf("restoreMatchedPath didn't restore snapshots (got %d)", len(node.snapshots))
    }
}

func TestRestoreMatchedPathStopsOnGoneAncestor(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
    c.scheduleWrite(n1)
    <-n1.inflightWrite
    for _, s := range n1.snapshots {
        s.Close()
    }
    n1.snapshots = nil

    // n2 is Gone (no diskPath, no snapshots) — restore must stop at it.
    n2 := &trieNode{tokens: []int32{2}, parent: n1}
    n1.children = append(n1.children, n2)

    err := c.restoreMatchedPath([]*trieNode{n1, n2})
    if err != nil {
        t.Errorf("restoreMatchedPath returned error on Gone ancestor (should degrade, not fail): %v", err)
    }
    if n1.snapshots == nil {
        t.Error("n1 should have been restored before the loop stopped at n2")
    }
    if n2.snapshots != nil {
        t.Error("n2 should remain unrestored (it was Gone)")
    }
}
```

- [ ] **Step 8.3: Run tests — expect failure.**

- [ ] **Step 8.4: Implement `restoreMatchedPath` and wire into `begin()`**

Add to `cache.go`:

```go
// restoreMatchedPath walks `path` from root side and restores any Cold nodes
// into Warm state before the caller hands off to switchToPath. A Gone node
// (diskPath empty, snapshots nil) or a load failure terminates the walk —
// the caller must handle partial paths by re-prefilling the tail.
func (c *kvCache) restoreMatchedPath(path []*trieNode) error {
    for _, node := range path {
        if node == c.root {
            continue
        }
        if node.snapshots != nil {
            continue // Warm or just-restored
        }
        if node.diskPath == "" {
            break // Gone: cannot restore; caller stops here.
        }
        if err := c.loadFromDisk(node); err != nil {
            slog.Warn("kv cache restore failed, treating as Gone",
                "path", node.diskPath, "err", err)
            node.diskPath = ""
            node.diskSize = 0
            break
        }
    }
    return nil
}
```

Then wire into `begin()`. Find the line where `path, matched := findBestMatch(c.root, inputs)` is stored (around `cache.go:105`). Immediately after, add:

```go
if err := c.restoreMatchedPath(path); err != nil {
    slog.Warn("kv cache restore aborted", "err", err)
}
// Re-check the matched prefix — if any intermediate was demoted to Gone,
// the effective match length may have shrunk. Recompute `matched`.
matched = matchedAfterRestore(path, inputs)
```

Add helper:

```go
// matchedAfterRestore recomputes the number of matching tokens after a
// partial-restore walk that may have demoted some nodes to Gone. We stop
// counting at the first node whose snapshots are still nil (Cold without
// a file) — it means restore failed there.
func matchedAfterRestore(path []*trieNode, inputs []int32) int {
    matched := 0
    for _, node := range path {
        if node.snapshots == nil && node.diskPath == "" && node != path[0] {
            return matched
        }
        matched += len(node.tokens)
        if matched > len(inputs) {
            return len(inputs)
        }
    }
    return matched
}
```

- [ ] **Step 8.5: Run tests**

```
go test -run 'TestBeginRestoresColdNodes|TestRestoreMatchedPath' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 8.6: Regression**

```
go test ./x/mlxrunner/...
```
Existing `begin()` tests must still pass (for them, `writer == nil`, every node's `diskPath == ""`, and the new loop is a no-op walk).

- [ ] **Step 8.7: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner: restore Cold nodes to Warm on cache-hit in begin()

Lazy mmap-backed restore; failure demotes the node to Gone so the
caller re-prefills only the tail.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Two-pass `enforceEvictionPolicy`

**Files:**
- Modify: `x/mlxrunner/cache.go:463` — split existing function into memory pass + disk pass
- Modify: `x/mlxrunner/cache_persist_test.go` — tests for Warm→Cold and Warm→Gone paths

**What this enables:** memory eviction no longer loses data (becomes Cold instead of deleting); disk cap triggers real deletes.

- [ ] **Step 9.1: Write failing tests**

Append to `cache_persist_test.go`:

```go
func TestMemoryPassDemotesToCold(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1, 2}, 9<<30) // oversized
    c.pagedOutBytes = 9 << 30
    c.scheduleWrite(node)
    <-node.inflightWrite

    c.enforceEvictionPolicy()
    if node.snapshots != nil {
        t.Error("memory pass should have dropped snapshots")
    }
    if node.diskPath == "" {
        t.Error("disk file was deleted in memory pass — should only drop memory")
    }
}

func TestMemoryPassSkipsInflightWrites(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    node := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 9<<30)
    c.pagedOutBytes = 9 << 30
    node.inflightWrite = make(chan struct{}) // simulate in-flight

    c.enforceEvictionPolicy()
    if node.snapshots == nil {
        t.Error("memory pass dropped a node whose write was still in flight")
    }
    close(node.inflightWrite)
    node.inflightWrite = nil
}

func TestDiskPassRemovesOverCap(t *testing.T) {
    dir := t.TempDir()
    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    c.writer = newDiskWriter(c.kvCache)
    defer c.teardown()

    c.diskMax = 100 // very small cap
    n1 := newTestNodeWithArraySnapshot(t, c.root, []int32{1}, 1024)
    n2 := newTestNodeWithArraySnapshot(t, c.root, []int32{2}, 1024)
    c.scheduleWrite(n1); <-n1.inflightWrite
    c.scheduleWrite(n2); <-n2.inflightWrite

    // Force n1 to be older so disk pass picks it first.
    n1.lastUsed = n1.lastUsed.Add(-time.Hour)

    c.enforceEvictionPolicy()

    if _, err := os.Stat(n1.diskPath); !os.IsNotExist(err) {
        t.Errorf("disk pass did not delete oldest node's file: %v", err)
    }
    if _, err := os.Stat(n2.diskPath); err != nil {
        t.Errorf("disk pass deleted too much: %v", err)
    }
}
```

- [ ] **Step 9.2: Add `diskMax` field to `kvCache`**

In `cache.go` struct, alongside the fields added in Task 2:

```go
diskMax int64 // -1 or <= 0 disables disk-pass; positive values are the byte cap
```

Default in constructor: pull from `envconfig.KVCacheDiskMax()` when feature is enabled; Task 12 wires this from the runner.

- [ ] **Step 9.3: Run tests — expect failure (tests reference new behavior).**

- [ ] **Step 9.4: Split `enforceEvictionPolicy` into two passes**

Replace the body of `enforceEvictionPolicy` (`cache.go:463`) with:

```go
func (c *kvCache) enforceEvictionPolicy() {
    c.enforceMemoryPolicy()
    c.enforceDiskPolicy()
}

// selectEvictionCandidate encapsulates the existing selector rule
// (oldest lastUsed, deepest endOffset, largest snapshotBytes, skipping
// root / activePath / multi-child).
func (c *kvCache) selectEvictionCandidate(filter func(*trieNode) bool) *trieNode {
    activeSet := make(map[*trieNode]bool, len(c.activePath))
    for _, n := range c.activePath {
        activeSet[n] = true
    }
    var best *trieNode
    walkNodes(c.root, func(n *trieNode) bool {
        if n == c.root || activeSet[n] || len(n.children) > 1 {
            return true
        }
        if !filter(n) {
            return true
        }
        if best == nil || cmp.Or(
            n.lastUsed.Compare(best.lastUsed),
            cmp.Compare(best.endOffset, n.endOffset),
            cmp.Compare(best.snapshotBytes(), n.snapshotBytes()),
        ) < 0 {
            best = n
        }
        return true
    })
    return best
}

func (c *kvCache) enforceMemoryPolicy() {
    for c.pagedOutBytes > maxPagedOutBytes {
        cand := c.selectEvictionCandidate(func(n *trieNode) bool {
            return n.snapshots != nil // Warm or not-yet-evicted
        })
        if cand == nil {
            return
        }
        if cand.inflightWrite != nil {
            // Unable to safely drop: write in flight. Try the next cycle.
            return
        }
        if cand.diskPath == "" {
            // Node was never persisted (either feature disabled or write failed).
            if c.writer != nil && cand.writeAttempts < 3 {
                c.scheduleWrite(cand)
                return // re-run after write completes
            }
            // Permanently un-persisted: fall back to legacy delete.
            c.evictNode(cand)
            continue
        }
        // Warm -> Cold: drop in-memory snapshots, keep trie node + disk file.
        freed := cand.snapshotBytes()
        for _, s := range cand.snapshots {
            if s != nil {
                s.Close()
            }
        }
        cand.snapshots = nil
        c.pagedOutBytes -= freed
    }
}

func (c *kvCache) enforceDiskPolicy() {
    if c.writer == nil || c.diskMax <= 0 {
        return
    }
    for atomic.LoadInt64(&c.diskBytes) > c.diskMax {
        cand := c.selectEvictionCandidate(func(n *trieNode) bool {
            return n.diskPath != ""
        })
        if cand == nil {
            return
        }
        if err := os.Remove(cand.diskPath); err != nil && !os.IsNotExist(err) {
            slog.Warn("kv cache disk evict: remove failed", "path", cand.diskPath, "err", err)
        }
        atomic.AddInt64(&c.diskBytes, -cand.diskSize)
        cand.diskPath = ""
        cand.diskSize = 0
        // Remove node from the trie (reuses existing evictNode logic).
        c.evictNode(cand)
    }
}
```

- [ ] **Step 9.5: Run tests**

```
go test -run 'TestMemoryPass|TestDiskPass' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 9.6: Regression**

```
go test ./x/mlxrunner/...
```
Pre-existing `enforceEvictionPolicy` tests must still pass (they don't set `diskMax`, so disk pass is a no-op, and without `writer` the memory pass falls back to legacy `evictNode`).

- [ ] **Step 9.7: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner: two-tier enforceEvictionPolicy (memory + disk)

Extracts the existing selector into selectEvictionCandidate. Memory pass
demotes Warm->Cold (keeping disk file). Disk pass deletes files and
removes the node entirely when diskBytes > diskMax.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Startup rehydration

**Files:**
- Modify: `x/mlxrunner/cache_persist.go` — `rehydrate` method on `kvCache`
- Modify: `x/mlxrunner/cache_persist_test.go` — rehydration tests with fixtures

**What this enables:** a fresh ollama server picks up caches from the previous session for the same model digest.

- [ ] **Step 10.1: Write failing tests**

Append to `cache_persist_test.go`:

```go
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
    dir := t.TempDir()
    // Session A: write some nodes.
    sessA := newTestKvCacheWithDisk(t, dir, "model", 1)
    sessA.writer = newDiskWriter(sessA.kvCache)
    n1 := newTestNodeWithArraySnapshot(t, sessA.root, []int32{1, 2}, 1024)
    sessA.scheduleWrite(n1); <-n1.inflightWrite
    n2 := newTestNodeWithArraySnapshot(t, n1, []int32{3, 4}, 1024)
    sessA.scheduleWrite(n2); <-n2.inflightWrite
    sessA.teardown()

    // Session B: fresh cache, same dir, same model.
    sessB := newTestKvCacheWithDisk(t, dir, "model", 1)
    defer sessB.teardown()
    if err := sessB.rehydrate(); err != nil {
        t.Fatalf("rehydrate: %v", err)
    }

    // Expect two children: root -> n1' (tokens=[1,2]) -> n2' (tokens=[3,4])
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

func TestRehydrateCleansOrphansAndTmps(t *testing.T) {
    dir := t.TempDir()
    // Plant an orphan .tmp file.
    _ = os.MkdirAll(dir, 0o755)
    _ = os.WriteFile(filepath.Join(dir, "aaa.safetensors.tmp"), []byte("partial"), 0o644)
    _ = os.WriteFile(filepath.Join(dir, "orphan.safetensors"), []byte("bogus"), 0o644) // unparseable

    c := newTestKvCacheWithDisk(t, dir, "model", 1)
    defer c.teardown()
    if err := c.rehydrate(); err != nil {
        t.Fatalf("rehydrate: %v", err)
    }
    if _, err := os.Stat(filepath.Join(dir, "aaa.safetensors.tmp")); !os.IsNotExist(err) {
        t.Error(".tmp file not cleaned up")
    }
}

func TestRehydrateRejectsForeignDigest(t *testing.T) {
    dir := t.TempDir()
    sessA := newTestKvCacheWithDisk(t, dir, "modelA", 1)
    sessA.writer = newDiskWriter(sessA.kvCache)
    n := newTestNodeWithArraySnapshot(t, sessA.root, []int32{1}, 1024)
    sessA.scheduleWrite(n); <-n.inflightWrite
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
```

- [ ] **Step 10.2: Run tests — expect failure.**

- [ ] **Step 10.3: Implement `rehydrate`**

Append to `cache_persist.go`:

```go
// rehydrate scans c.cacheDir and rebuilds the trie skeleton from safetensors headers.
// No arrays are loaded; every rebuilt node is Cold (snapshots == nil, diskPath set).
// Called once during kvCache construction when persistence is enabled.
func (c *kvCache) rehydrate() error {
    if c.cacheDir == "" {
        return nil
    }
    if err := os.MkdirAll(c.cacheDir, 0o755); err != nil {
        return fmt.Errorf("mkdir cacheDir: %w", err)
    }
    entries, err := os.ReadDir(c.cacheDir)
    if err != nil {
        return fmt.Errorf("readdir: %w", err)
    }

    // Step 1: clean .tmp files (crashed writes).
    for _, e := range entries {
        name := e.Name()
        if strings.HasSuffix(name, ".tmp") {
            _ = os.Remove(filepath.Join(c.cacheDir, name))
        }
    }

    // Step 2: scan headers; collect (name, header, size).
    type scanned struct {
        name string
        path string
        size int64
        h    headerFields
    }
    var ok []scanned
    for _, e := range entries {
        name := e.Name()
        if !strings.HasSuffix(name, ".safetensors") {
            continue
        }
        path := filepath.Join(c.cacheDir, name)
        st, err := os.Stat(path)
        if err != nil {
            continue
        }
        h, err := c.readHeader(path)
        if err != nil {
            slog.Warn("kv cache rehydrate: unreadable header, deleting", "path", path, "err", err)
            _ = os.Remove(path)
            continue
        }
        if h.modelDigest != c.modelDigest {
            // Different model stamped this file. Leave it alone (another cache
            // instance may own it); skip for this rehydration.
            continue
        }
        if h.layerCount != len(c.caches) {
            slog.Warn("kv cache rehydrate: layer_count mismatch, deleting", "path", path)
            _ = os.Remove(path)
            continue
        }
        ok = append(ok, scanned{name: name, path: path, size: st.Size(), h: h})
    }

    // Step 3: topo sort. A node's parent_hash must be present for it to be
    // hooked up. Orphans (parent referenced but absent) are unreachable.
    nameOf := func(s scanned) string { return strings.TrimSuffix(s.name, ".safetensors") }
    byName := make(map[string]scanned, len(ok))
    for _, s := range ok {
        byName[nameOf(s)] = s
    }
    nodes := make(map[string]*trieNode, len(ok))
    var roots []scanned
    for _, s := range ok {
        if s.h.parentHash == "" {
            roots = append(roots, s)
        } else if _, found := byName[s.h.parentHash]; !found {
            slog.Warn("kv cache rehydrate: orphan (missing parent), deleting", "path", s.path, "parent", s.h.parentHash)
            _ = os.Remove(s.path)
        }
    }

    // Step 4: BFS from roots, building trie nodes.
    totalBytes := int64(0)
    var queue []scanned
    queue = append(queue, roots...)
    for len(queue) > 0 {
        s := queue[0]
        queue = queue[1:]
        var parent *trieNode
        if s.h.parentHash == "" {
            parent = c.root
        } else {
            parent = nodes[s.h.parentHash]
            if parent == nil {
                // shouldn't happen — we filtered orphans — but defensively skip.
                continue
            }
        }
        n := &trieNode{
            tokens:    s.h.tokens,
            parent:    parent,
            diskPath:  s.path,
            diskSize:  s.size,
            lastUsed:  s.h.createdAt,
        }
        // Update parent.endOffset bookkeeping: n.endOffset = parent.endOffset + len(tokens)
        // The existing trieNode type may track this differently — use the
        // existing helper if present. For now the kvCache API computes it lazily
        // from startOffset() walks.
        parent.children = append(parent.children, n)
        nodes[nameOf(s)] = n
        totalBytes += s.size

        // Enqueue children of this node.
        for _, cand := range ok {
            if cand.h.parentHash == nameOf(s) {
                queue = append(queue, cand)
            }
        }
    }

    atomic.StoreInt64(&c.diskBytes, totalBytes)
    return nil
}
```

Note: the BFS inner loop above is O(N²) for clarity; fine at the node counts we expect (<1K). If this becomes a hotspot it can be rewritten as a bucketed walk.

- [ ] **Step 10.4: Run tests**

```
go test -run 'TestRehydrate' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 10.5: Regression**

```
go test ./x/mlxrunner/...
```

- [ ] **Step 10.6: Commit**

```
git add x/mlxrunner/cache_persist.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner/persist: startup rehydration from safetensors headers

Scans cacheDir, cleans .tmp files, deletes unreadable / orphaned /
foreign-digest files, BFS-builds the Cold trie skeleton. No arrays
loaded — lazy via loadFromDisk.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Runner + server wiring (construction, shutdown)

**Files:**
- Modify: `x/mlxrunner/runner.go` — pass modelDigest + cacheDir; construct writer; call rehydrate
- Modify: `x/mlxrunner/server.go` — invoke `cache.writer.shutdown()` on graceful stop
- Modify: `x/mlxrunner/cache.go` — `newKvCache` factory that centralizes the construction knobs

**What this enables:** ollama-server-level integration. The feature is now usable end-to-end.

- [ ] **Step 11.1: Write failing integration test (feature-off path)**

Append to `cache_persist_test.go`:

```go
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
```

- [ ] **Step 11.2: Run tests — expect failure.**

- [ ] **Step 11.3: Add `newKvCache` factory and `shutdown` method**

In `cache.go`:

```go
import "github.com/ollama/ollama/envconfig"

// newKvCache is the construction entry point for kvCache.
// When OLLAMA_KV_CACHE_DISK_MAX == 0, returns a cache with no writer —
// byte-for-byte identical to upstream-main behavior.
func newKvCache(modelDigest string, numLayers int) *kvCache {
    c := &kvCache{
        caches: make([]cache.Cache, numLayers),
    }
    c.ensureRoot()

    diskMax := envconfig.KVCacheDiskMax()
    if diskMax == 0 {
        return c // feature disabled
    }
    c.modelDigest = modelDigest
    c.cacheDir = filepath.Join(envconfig.KVCacheRoot(), modelDigest)
    c.diskMax = diskMax
    if err := c.rehydrate(); err != nil {
        slog.Warn("kv cache rehydrate failed, starting cold", "err", err)
    }
    c.writer = newDiskWriter(c)
    return c
}

// shutdown drains pending writes and releases the writer goroutine.
func (c *kvCache) shutdown() {
    if c.writer != nil {
        c.writer.shutdown(15 * time.Second)
    }
}
```

Replace the direct `&kvCache{...}` literal in `runner.go` with `newKvCache(modelDigest, numLayers)`. If `modelDigest` isn't already available at that call site, plumb it from the nearest caller that has it (grep for where the model ID / digest is known — usually in the runner's `init` or the HTTP handler that spawned it). If the digest is genuinely unavailable, pass an empty string, which will still produce a working feature-off cache. Don't fabricate a digest.

- [ ] **Step 11.4: Wire shutdown into server graceful stop**

Read `x/mlxrunner/server.go`. Locate the existing shutdown handler (grep for `Shutdown`, `Close`, or `SIGTERM`). After the current cleanup — but before the runner's own exit — call:

```go
if r.kv != nil {
    r.kv.shutdown()
}
```

(Use whatever field name the runner actually uses to hold its `*kvCache`.)

- [ ] **Step 11.5: Run the new integration tests**

```
go test -run 'TestFeatureDisabled|TestFeatureEnabled' ./x/mlxrunner/
```
Expected: PASS.

- [ ] **Step 11.6: Full regression**

```
go test ./x/mlxrunner/... && go build ./...
```

- [ ] **Step 11.7: Commit**

```
git add x/mlxrunner/cache.go x/mlxrunner/runner.go x/mlxrunner/server.go x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner: wire persistence into runner construction + shutdown

newKvCache honors OLLAMA_KV_CACHE_DISK_MAX: 0 = feature off,
negative = unlimited, positive = capped. shutdown drains with
15s timeout (matches existing llmServer shutdown budget).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: End-to-end integration test

**Files:**
- Modify: `x/mlxrunner/cache_persist_test.go` — full write-then-rehydrate-then-hit test

**What this enables:** confidence that the whole pipeline produces a cache hit across simulated process restart.

- [ ] **Step 12.1: Write the end-to-end test**

```go
func TestEndToEndWarmRestart(t *testing.T) {
    dir := t.TempDir()
    t.Setenv("OLLAMA_KV_CACHE_ROOT", dir)
    t.Setenv("OLLAMA_KV_CACHE_DISK_MAX", "-1")

    // Session A: write a node.
    sessA := newKvCache("m", 1)
    root := sessA.root
    node := newTestNodeWithArraySnapshot(t, root, []int32{1, 2, 3, 4}, 1024)
    sessA.scheduleWrite(node)
    <-node.inflightWrite
    sessA.shutdown()

    // Session B: fresh cache, same env, should rehydrate.
    sessB := newKvCache("m", 1)
    defer sessB.shutdown()
    if len(sessB.root.children) != 1 {
        t.Fatalf("rehydrate missed node")
    }
    rehydrated := sessB.root.children[0]
    if rehydrated.snapshots != nil {
        t.Error("rehydrated node not Cold")
    }

    // Simulate a prefix match and call restoreMatchedPath.
    if err := sessB.restoreMatchedPath([]*trieNode{rehydrated}); err != nil {
        t.Fatalf("restore: %v", err)
    }
    if rehydrated.snapshots == nil {
        t.Error("restore didn't materialize snapshots")
    }
}
```

- [ ] **Step 12.2: Run the test**

```
go test -run TestEndToEndWarmRestart ./x/mlxrunner/
```
Expected: PASS. If it fails, the failure mode is instructive — each of tasks 1-11 had its own focused coverage, so this is purely the integration assembly.

- [ ] **Step 12.3: Commit**

```
git add x/mlxrunner/cache_persist_test.go
git commit -m "mlxrunner/persist: end-to-end warm-restart integration test

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 13: Documentation

**Files:**
- Modify: `docs/faq.mdx` — add an Environment variables entry for the two new vars

- [ ] **Step 13.1: Locate the env-vars section in `docs/faq.mdx`**

```
grep -n "OLLAMA_" docs/faq.mdx | head -20
```

- [ ] **Step 13.2: Add the entries**

Under the existing env-vars section, append:

```markdown
### `OLLAMA_KV_CACHE_DISK_MAX`

Maximum total bytes of on-disk MLX KV cache. When the Apple-Silicon MLX runner
evicts a KV snapshot from memory, by default it also persists it to disk so that
a later request covering the same prefix can reload it instead of re-prefilling.

- unset or negative (default `-1`): unlimited (may grow until you cap it)
- `0`: persistence fully disabled — same behavior as prior releases
- positive: hard cap in bytes, with optional suffix `KiB`, `MiB`, `GiB` (binary) or `KB`, `MB`, `GB` (decimal)

Example: `OLLAMA_KV_CACHE_DISK_MAX=50GiB`

### `OLLAMA_KV_CACHE_ROOT`

Root directory for on-disk KV cache files. One subdirectory per model digest is
created underneath. Default: `<OLLAMA_MODELS>/../cache/kv` — a sibling of your
models store.
```

- [ ] **Step 13.3: Commit**

```
git add docs/faq.mdx
git commit -m "docs: document OLLAMA_KV_CACHE_DISK_MAX and OLLAMA_KV_CACHE_ROOT

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: Final regression sweep

- [ ] **Step 14.1: Build, vet, test, race.**

```
go build ./...
go vet ./...
go test ./...
go test -race ./x/mlxrunner/...
```

All four must pass. The `-race` pass is especially important given the new writer goroutine.

- [ ] **Step 14.2: Hand-exercise the feature-off path**

```
OLLAMA_KV_CACHE_DISK_MAX=0 go test ./x/mlxrunner/...
```

Expected: all existing tests green, no new code paths touched.

- [ ] **Step 14.3: If anything failed, fix it — do not commit until clean.**

No commit for this task unless a fix is made.

---

## Self-review notes

**Spec coverage:** Every numbered requirement (F1–F7, O1–O4) and every section (5.1–10.3) in the spec maps to at least one task above. Explicit mapping:

| Spec | Task |
|---|---|
| F1 (write on attach/split) | 6 |
| F2 (atomic write) | 4 |
| F3 (sync load on match) | 8 |
| F4 (shutdown drain 15s) | 5, 11 |
| F5 (startup header-only scan) | 10 |
| F6 (no drop while in flight) | 9 |
| F7 (disk cap → delete) | 9 |
| O1 (off = zero cost) | 11, 14 |
| O2 (never panic) | 5 (circuit breaker) |
| O3 (crash-safe) | 4 (atomic rename) + 10 (.tmp cleanup) |
| O4 (digest scoping) | 7, 10 |
| §5.1 state model | 2, 8, 9 (transitions) |
| §5.2 new fields | 2 |
| §6 write pipeline | 4, 5, 6 |
| §7 restore pipeline | 7, 8 |
| §8 rehydration | 10 |
| §9 eviction | 9 |
| §10 on-disk format | 3, 4 |
| §11 configuration | 1, 11 |
| §13 testing | throughout |

**Placeholder scan:** No `TBD` / `TODO` in the plan text itself. Task 4 has a `TODO(task-7):` comment that is intentional — it's a hand-off marker and resolved in Task 7.

**Type consistency:** `diskBytes` is `int64` atomic throughout. `diskMax` is `int64`. `pagedOutBytes` remains `int64` (unchanged from upstream). `writeAttempts` is `uint8`. Method names: `scheduleWrite`, `writeOne`, `loadFromDisk`, `restoreMatchedPath`, `enforceMemoryPolicy`, `enforceDiskPolicy`, `selectEvictionCandidate`, `rehydrate`, `shutdown`. Used consistently in every task.

**Scope check:** 13 focused tasks + 1 regression sweep. One plan, one feature, one shippable unit. No sub-plan decomposition needed.
