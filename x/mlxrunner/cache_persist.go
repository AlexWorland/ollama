// Package mlxrunner — KV cache disk persistence.
package mlxrunner

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const cacheFormatVersion = "3"

// diskWriter separates circuit-breaker state from the lock-free kvCache
// invariant. It tracks consecutive failures and trips a breaker that
// self-heals after circuitBreakerTTL, so one transient SSD hiccup doesn't
// disable persistence for the process lifetime.
//
// An earlier design ran writes on a background goroutine, but MLX's Metal
// backend is not thread-safe: any MLX call (Eval, SaveSafetensors, etc.)
// touches a shared default Metal command encoder, and concurrent calls from
// the pipeline goroutine and the writer goroutine triggered
//
//	-[IOGPUMetalCommandBuffer validate]:215:
//	failed assertion `commit command buffer with uncommitted encoder'
//
// Since the MLX runner is single-goroutine serial, writeOne is called
// directly from scheduleWrite on the main goroutine. diskWriter stays a
// separate type because kvCache is intentionally mutex-free, and the
// breaker fields do need a mutex for test-goroutine setup/teardown.
type diskWriter struct {
	mu                  sync.Mutex
	disabled            bool // circuit breaker tripped (5+ consecutive failures)
	consecutiveFailures int
	disabledAt          time.Time // when the breaker tripped; used for TTL decay
}

// circuitBreakerTTL defines how long a tripped circuit breaker stays tripped
// before the next write attempt resets it. A transient SSD hiccup shouldn't
// disable persistence for the process lifetime.
const circuitBreakerTTL = 5 * time.Minute

func newDiskWriter(c *kvCache) *diskWriter {
	return &diskWriter{}
}

// shutdown is a no-op in the synchronous design; there's no goroutine to
// drain. Kept for API compatibility with the older async writer.
func (w *diskWriter) shutdown(_ time.Duration) int {
	return 0
}

// writeNode writes a single node synchronously. Must run on the same
// goroutine that owns MLX access (the pipeline goroutine). Returns true on
// success. Tracks consecutive failures and trips the circuit breaker after
// 5 in a row; a tripped breaker self-heals after circuitBreakerTTL.
func (w *diskWriter) writeNode(c *kvCache, node *trieNode) bool {
	w.mu.Lock()
	if w.disabled {
		if time.Since(w.disabledAt) >= circuitBreakerTTL {
			w.disabled = false
			w.consecutiveFailures = 0
		} else {
			w.mu.Unlock()
			return false
		}
	}
	w.mu.Unlock()

	if err := c.writeOne(node); err != nil {
		slog.Warn("kv cache write failed", "err", err)
		w.mu.Lock()
		w.consecutiveFailures++
		if w.consecutiveFailures >= 5 {
			slog.Warn("kv cache writer: disabling after 5 consecutive failures",
				"ttl", circuitBreakerTTL)
			w.disabled = true
			w.disabledAt = time.Now()
		}
		w.mu.Unlock()
		return false
	}
	w.mu.Lock()
	w.consecutiveFailures = 0
	w.mu.Unlock()
	return true
}

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

// writeOne serializes node's snapshots to a content-addressed file under c.cacheDir.
// Preconditions: node is off the active path and has snapshots attached.
// On success: node.diskPath and node.diskSize are set; c.diskBytes is incremented.
func (c *kvCache) writeOne(node *trieNode) error {
	if c.cacheDir == "" {
		return errors.New("writeOne called with empty cacheDir")
	}
	if node == nil || len(node.snapshots) == 0 {
		return errors.New("writeOne called with no snapshots")
	}

	arrays, fieldMap, types, cleanup, err := collectNodeArrays(node)
	if err != nil {
		return fmt.Errorf("collect arrays: %w", err)
	}
	defer cleanup()
	mlx.Eval(arrays...)

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

	dtype := arrays[0].DType().String()
	fname := contentFilename(c.modelDigest, parentHash, node.tokens, len(node.snapshots), dtype)
	finalPath := filepath.Join(c.cacheDir, fname)
	// MLX's safetensors writer appends ".safetensors" if the path doesn't
	// already end with it, so the tmp suffix has to be embedded BEFORE the
	// extension. Otherwise the on-disk file ends up at "<name>.tmp.safetensors"
	// and the rename of "<name>.tmp" fails with ENOENT.
	tmpPath := strings.TrimSuffix(finalPath, ".safetensors") + ".tmp.safetensors"

	if err := os.MkdirAll(c.cacheDir, 0o755); err != nil {
		return fmt.Errorf("mkdir cacheDir: %w", err)
	}

	if err := mlx.SaveSafetensorsWithMetadata(tmpPath, fieldMap, meta); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("save safetensors: %w", err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("rename: %w", err)
	}

	st, err := os.Stat(finalPath)
	if err != nil {
		return fmt.Errorf("stat written file: %w", err)
	}
	node.diskPath = finalPath
	node.diskSize = st.Size()
	totalDisk := atomic.AddInt64(&c.diskBytes, st.Size())
	slog.Debug("kv cache wrote node",
		"file", filepath.Base(finalPath),
		"parent_hash", parentHash,
		"tokens", len(node.tokens),
		"layers", len(node.snapshots),
		"dtype", dtype,
		"size_bytes", st.Size(),
		"total_disk_bytes", totalDisk)
	return nil
}

// collectNodeArrays extracts the MLX arrays for each layer's snapshot, the
// layer-indexed name→array map for SaveSafetensorsWithMetadata, the
// snapshot-type tags used by the loader, and a cleanup func that releases
// any arrays snapshots allocated solely for this call (e.g. scalars packed
// into 1-element arrays). Callers must invoke cleanup once the write
// completes. Returns an error when any layer's snapshot is nil-mapped-to
// non-empty type or is not a SerializableSnapshot — writing a partial file
// with "unknown" placeholders would produce a file the loader cannot
// restore, poisoning the cache for future startups.
func collectNodeArrays(node *trieNode) (all []*mlx.Array, fields map[string]*mlx.Array, types []string, cleanup func(), err error) {
	all = make([]*mlx.Array, 0, 2*len(node.snapshots))
	fields = make(map[string]*mlx.Array, 2*len(node.snapshots))
	types = make([]string, 0, len(node.snapshots))
	cleanups := make([]func(), 0, len(node.snapshots))
	cleanup = func() {
		for _, fn := range cleanups {
			fn()
		}
	}
	for i, s := range node.snapshots {
		if s == nil {
			// A layer that legitimately has no state yet (e.g. a recurrent
			// layer at the start of a conversation). Tag as "empty" — the
			// loader reconstructs it as nil, and the cache's Restore() knows
			// how to handle a nil snapshot.
			types = append(types, cache.SnapshotTypeEmpty)
			continue
		}
		ser, ok := s.(cache.SerializableSnapshot)
		if !ok {
			cleanup()
			return nil, nil, nil, nil, fmt.Errorf("layer %d: snapshot type %T is not serializable", i, s)
		}
		st := ser.SnapshotType()
		if _, known := cache.SnapshotFieldNames(st); !known {
			// Guards against method-promotion surprises: a snapshot that embeds
			// another serializable type silently inherits its SnapshotType() —
			// which would round-trip with the wrong factory on load. Reject
			// unregistered types at write time so the bug is loud, not latent.
			cleanup()
			return nil, nil, nil, nil, fmt.Errorf("layer %d: snapshot %T reports unregistered type %q", i, s, st)
		}
		arrs, cfn := ser.CollectArrays()
		cleanups = append(cleanups, cfn)
		for name, arr := range arrs {
			fields[fmt.Sprintf("layer_%d_%s", i, name)] = arr
			all = append(all, arr)
		}
		types = append(types, st)
	}
	return all, fields, types, cleanup, nil
}

// readHeader reads only the metadata block of a safetensors file without
// loading the arrays. Used for startup rehydration and test assertions.
// On a populated cache dir this is 10-50× faster than loading the file via
// MLX, and avoids the MLX allocation for tensors we immediately discard.
func (c *kvCache) readHeader(path string) (headerFields, error) {
	meta, err := parseSafetensorsMetadata(path)
	if err != nil {
		return headerFields{}, err
	}
	return decodeHeader(meta)
}

// loadFromDisk reconstructs node.snapshots from node.diskPath and attaches them.
// Precondition: node.diskPath != "" && node.snapshots == nil.
// On error the node is unchanged (still Cold).
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
	raw := make(map[string]string)
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

	startOffset := node.startOffset()
	endOffset := startOffset + len(h.tokens)
	if node.endOffset > 0 {
		// Trust the node's own endOffset when set (e.g., not from rehydration).
		endOffset = node.endOffset
	}
	startedAt := time.Now()
	snaps, err := c.restoreSnapshotArrays(sf, h, startOffset, endOffset)
	if err != nil {
		sf.Free()
		return err
	}
	node.setSnapshots(snaps, &c.pagedOutBytes)
	node.lastUsed = time.Now()
	sf.Free()
	slog.Debug("kv cache loaded from disk",
		"file", filepath.Base(node.diskPath),
		"start_offset", startOffset,
		"end_offset", endOffset,
		"layers", h.layerCount,
		"size_bytes", node.diskSize,
		"took", time.Since(startedAt).Truncate(time.Microsecond))
	return nil
}

// rehydrate scans c.cacheDir and rebuilds the trie skeleton from safetensors
// headers. No arrays are loaded; every rebuilt node is Cold (snapshots == nil,
// diskPath set). Called once during kvCache construction when persistence is
// enabled. Foreign-digest files are left alone (they may belong to another
// model's cache instance running concurrently); orphaned and unreadable files
// are deleted.
func (c *kvCache) rehydrate() error {
	if c.cacheDir == "" {
		return nil
	}
	startedAt := time.Now()
	if err := os.MkdirAll(c.cacheDir, 0o755); err != nil {
		return fmt.Errorf("mkdir cacheDir: %w", err)
	}
	c.ensureRoot()
	entries, err := os.ReadDir(c.cacheDir)
	if err != nil {
		return fmt.Errorf("readdir: %w", err)
	}
	// Step 1: clean tmp files (crashed writes). MLX-written tmp files end
	// in ".tmp.safetensors"; legacy ".tmp" files (from earlier writers, or
	// other tools) are also removed defensively.
	tmpDeleted := 0
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".tmp.safetensors") || strings.HasSuffix(name, ".tmp") {
			if err := os.Remove(filepath.Join(c.cacheDir, name)); err == nil {
				tmpDeleted++
			}
		}
	}
	if tmpDeleted > 0 {
		slog.Debug("kv cache rehydrate cleaned tmp files", "count", tmpDeleted)
	}

	// Step 2: scan headers; collect (name, header, size) for our model.
	type scanned struct {
		name string
		path string
		size int64
		h    headerFields
	}
	var ok []scanned
	for _, e := range entries {
		if !e.Type().IsRegular() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".safetensors") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		path := filepath.Join(c.cacheDir, name)
		h, err := c.readHeader(path)
		if err != nil {
			slog.Warn("kv cache rehydrate: unreadable header, deleting", "path", path, "err", err)
			_ = os.Remove(path)
			continue
		}
		if h.modelDigest != c.modelDigest {
			// Different model stamped this file. Leave it for another cache
			// instance; skip for this rehydration.
			continue
		}
		if h.layerCount != len(c.caches) {
			slog.Warn("kv cache rehydrate: layer_count mismatch, deleting", "path", path, "file_layers", h.layerCount, "expected", len(c.caches))
			_ = os.Remove(path)
			continue
		}
		if slices.Contains(h.snapshotTypes, cache.SnapshotTypeUnknown) {
			// Written by a pre-fix writer; the loader can't restore these and
			// they'd poison the ancestor chain on every startup. Delete now.
			slog.Warn("kv cache rehydrate: unloadable snapshot_types, deleting", "path", path)
			_ = os.Remove(path)
			continue
		}
		ok = append(ok, scanned{name: name, path: path, size: info.Size(), h: h})
	}

	// Step 3: index by hash, identify orphans (missing parent).
	nameOf := func(s scanned) string { return strings.TrimSuffix(s.name, ".safetensors") }
	byName := make(map[string]scanned, len(ok))
	for _, s := range ok {
		byName[nameOf(s)] = s
	}
	var roots []scanned
	survivors := make([]scanned, 0, len(ok))
	for _, s := range ok {
		if s.h.parentHash == "" {
			roots = append(roots, s)
			survivors = append(survivors, s)
			continue
		}
		if _, found := byName[s.h.parentHash]; !found {
			slog.Warn("kv cache rehydrate: orphan (missing parent), deleting",
				"path", s.path, "parent", s.h.parentHash)
			_ = os.Remove(s.path)
			continue
		}
		survivors = append(survivors, s)
	}

	// Step 4: BFS from roots, building trie nodes. Index children by parent
	// hash so each step is O(children) rather than O(N).
	childrenOf := make(map[string][]scanned, len(survivors))
	for _, s := range survivors {
		if s.h.parentHash != "" {
			childrenOf[s.h.parentHash] = append(childrenOf[s.h.parentHash], s)
		}
	}

	totalBytes := int64(0)
	nodes := make(map[string]*trieNode, len(survivors))
	queue := append([]scanned{}, roots...)
	for len(queue) > 0 {
		s := queue[0]
		queue = queue[1:]
		var parent *trieNode
		if s.h.parentHash == "" {
			parent = c.root
		} else {
			parent = nodes[s.h.parentHash]
			if parent == nil {
				continue // defensively skip; orphans were filtered above
			}
		}
		n := &trieNode{
			tokens:    s.h.tokens,
			endOffset: parent.endOffset + len(s.h.tokens),
			parent:    parent,
			diskPath:  s.path,
			diskSize:  s.size,
			lastUsed:  s.h.createdAt,
		}
		if n.lastUsed.IsZero() {
			n.lastUsed = time.Now()
		}
		parent.children = append(parent.children, n)
		nodes[nameOf(s)] = n
		totalBytes += s.size
		queue = append(queue, childrenOf[nameOf(s)]...)
	}

	atomic.StoreInt64(&c.diskBytes, totalBytes)
	if len(nodes) > 0 || tmpDeleted > 0 {
		slog.Info("kv cache rehydrated",
			"nodes", len(nodes),
			"roots", len(roots),
			"total_bytes", totalBytes,
			"tmp_cleaned", tmpDeleted,
			"took", time.Since(startedAt).Truncate(time.Millisecond))
	}
	return nil
}

// restoreSnapshotArrays rebuilds typed Snapshots from a loaded safetensors
// file, dispatching by snapshot_types metadata. Field-name and type
// dispatch live in the cache package so adding a new snapshot variant
// doesn't require touching the persistence layer.
func (c *kvCache) restoreSnapshotArrays(sf *mlx.SafetensorsFile, h headerFields, startOffset, endOffset int) ([]cache.Snapshot, error) {
	snaps := make([]cache.Snapshot, h.layerCount)
	for i := 0; i < h.layerCount; i++ {
		st := cache.SnapshotTypeKV
		if i < len(h.snapshotTypes) {
			st = h.snapshotTypes[i]
		}
		names, ok := cache.SnapshotFieldNames(st)
		if !ok {
			return nil, fmt.Errorf("layer %d: unknown snapshot type %q", i, st)
		}
		fields := make(map[string]*mlx.Array, len(names))
		pinned := make([]*mlx.Array, 0, len(names))
		for _, name := range names {
			full := fmt.Sprintf("layer_%d_%s", i, name)
			arr := sf.Get(full)
			if arr == nil {
				mlx.Unpin(pinned...)
				return nil, fmt.Errorf("layer %d: missing field %q", i, full)
			}
			mlx.Pin(arr)
			pinned = append(pinned, arr)
			fields[name] = arr
		}
		snap, err := cache.NewSnapshotFromArrays(st, fields, startOffset, endOffset)
		if err != nil {
			mlx.Unpin(pinned...)
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		snaps[i] = snap
	}
	return snaps, nil
}

