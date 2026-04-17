// Package mlxrunner — KV cache disk persistence.
// Design: .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md
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
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const cacheFormatVersion = "1"

// diskWriter serializes trieNode writes to disk on a single goroutine.
// Queue is a mutex-guarded slice (channels caused a shutdown deadlock in
// the prior branch — see commit 4f6e3111 in the historical persist branch).
//
// stopped vs disabled:
//   - stopped: shutdown was called. Reject new enqueues, drain pending
//     queue WITH writes, exit loop when queue is empty.
//   - disabled: circuit breaker tripped (5+ consecutive failures). Drain
//     pending queue WITHOUT writes, then exit. Stays disabled for the
//     lifetime of the writer.
type diskWriter struct {
	mu       sync.Mutex
	cond     *sync.Cond
	pending  []*trieNode
	stopped  bool
	disabled bool
	done     chan struct{}
	cache    *kvCache

	consecutiveFailures int // circuit breaker per spec §6.3
}

func newDiskWriter(c *kvCache) *diskWriter {
	w := &diskWriter{
		cache: c,
		done:  make(chan struct{}),
	}
	w.cond = sync.NewCond(&w.mu)
	slog.Debug("kv cache writer started", "cache_dir", c.cacheDir, "model_digest", c.modelDigest)
	go w.loop()
	return w
}

// enqueue schedules node for writing. node.inflightWrite must already be a
// fresh channel — the loop closes it when this node is done.
func (w *diskWriter) enqueue(node *trieNode) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped || w.disabled {
		// Writer shut down or tripped the circuit breaker: close the
		// channel so callers waiting on it aren't stranded. Do NOT clear
		// the field (see loop() comment about race conditions).
		reason := "stopped"
		if w.disabled {
			reason = "disabled"
		}
		slog.Debug("kv cache enqueue rejected",
			"reason", reason,
			"start_offset", node.startOffset(),
			"tokens", len(node.tokens))
		if node.inflightWrite != nil {
			close(node.inflightWrite)
		}
		return
	}
	w.pending = append(w.pending, node)
	slog.Debug("kv cache write enqueued",
		"start_offset", node.startOffset(),
		"tokens", len(node.tokens),
		"queue_depth", len(w.pending))
	w.cond.Signal()
}

func (w *diskWriter) loop() {
	defer close(w.done)
	for {
		w.mu.Lock()
		for len(w.pending) == 0 && !w.stopped {
			// Note: we don't exit on `disabled` alone because the cache
			// may still be running and scheduleWrite could race — we just
			// stay here until either (a) stopped fires, or (b) some
			// no-op enqueue wakes us (only to be dropped by the skipWrite
			// branch below).
			w.cond.Wait()
		}
		if len(w.pending) == 0 {
			// Only reachable when stopped == true.
			w.mu.Unlock()
			return
		}
		node := w.pending[0]
		w.pending = w.pending[1:]
		// Circuit breaker: if the writer has been disabled, drain the
		// queue without writing. Plain shutdown still drains WITH writes.
		skipWrite := w.disabled
		w.mu.Unlock()

		if !skipWrite {
			w.writeOneWithRetry(node)
		}
		// Close the channel so waiters unblock. Do NOT clear the field:
		// main-goroutine code uses isWriteInFlight() (closed-channel check)
		// to distinguish "done" from "queued". Clearing here would race
		// with concurrent reads of node.inflightWrite.
		if node.inflightWrite != nil {
			close(node.inflightWrite)
		}
	}
}

func (w *diskWriter) writeOneWithRetry(node *trieNode) {
	startedAt := time.Now()
	err := w.cache.writeOne(node)
	if err == nil {
		w.mu.Lock()
		brokeStreak := w.consecutiveFailures > 0
		w.consecutiveFailures = 0
		w.mu.Unlock()
		if brokeStreak {
			slog.Debug("kv cache writer: failure streak broken")
		}
		slog.Debug("kv cache write completed",
			"path", node.diskPath,
			"size_bytes", node.diskSize,
			"tokens", len(node.tokens),
			"took", time.Since(startedAt).Truncate(time.Microsecond))
		return
	}
	node.writeAttempts++
	slog.Warn("kv cache write failed",
		"path", node.diskPath, "attempt", node.writeAttempts, "err", err)
	w.mu.Lock()
	w.consecutiveFailures++
	failures := w.consecutiveFailures
	if failures >= 5 {
		slog.Warn("kv cache writer: disabling after 5 consecutive failures")
		w.disabled = true
		w.cond.Broadcast() // wake loop so it can drain without writing
	}
	w.mu.Unlock()
	if failures < 5 {
		slog.Debug("kv cache writer failure streak", "consecutive", failures)
	}
	// No auto-requeue: the memory eviction pass calls scheduleWrite again when
	// the node next becomes a candidate, capped at writeAttempts < 3.
}

// shutdown blocks until the queue drains or the timeout elapses.
// Returns the number of still-pending nodes (0 on clean drain).
// timeout <= 0 means "wait until done".
func (w *diskWriter) shutdown(timeout time.Duration) int {
	w.mu.Lock()
	queued := len(w.pending)
	w.stopped = true
	w.cond.Broadcast()
	w.mu.Unlock()
	slog.Debug("kv cache writer shutdown initiated", "queued", queued, "timeout", timeout)
	startedAt := time.Now()

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
	took := time.Since(startedAt).Truncate(time.Millisecond)
	if remaining > 0 {
		slog.Warn("kv cache writer shutdown: drained with pending", "remaining", remaining, "took", took)
	} else {
		slog.Debug("kv cache writer shutdown complete", "drained", queued, "took", took)
	}
	return remaining
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

// arrayExposer lets writeOne extract the underlying mlx arrays from a Snapshot
// without depending on a concrete type. Both the test-only arraySnapshot and
// the cache.kvSnapshot (after Task 7 adds Keys/Values) implement it.
type arrayExposer interface {
	Keys() *mlx.Array
	Values() *mlx.Array
}

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

	arrays, fieldMap, types := collectNodeArrays(node)
	if len(arrays) > 0 {
		mlx.Eval(arrays...)
	}

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

	dtype := "unknown"
	if len(arrays) > 0 {
		dtype = arrays[0].DType().String()
	}
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
// layer-indexed name→array map for SaveSafetensorsWithMetadata, and the
// snapshot-type tags used by the loader to dispatch by type.
func collectNodeArrays(node *trieNode) ([]*mlx.Array, map[string]*mlx.Array, []string) {
	all := make([]*mlx.Array, 0, 2*len(node.snapshots))
	fields := make(map[string]*mlx.Array, 2*len(node.snapshots))
	types := make([]string, 0, len(node.snapshots))
	for i, s := range node.snapshots {
		if s == nil {
			types = append(types, "unknown")
			continue
		}
		if ax, ok := s.(arrayExposer); ok {
			k, v := ax.Keys(), ax.Values()
			fields[fmt.Sprintf("layer_%d_keys", i)] = k
			fields[fmt.Sprintf("layer_%d_values", i)] = v
			all = append(all, k, v)
			types = append(types, "kv")
			continue
		}
		// Unknown snapshot type: tag and skip its arrays. Task 7 will add
		// support for the real cache.kvSnapshot via the same arrayExposer
		// interface; until then, real-snapshot writes silently drop arrays.
		types = append(types, "unknown")
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
	if h.layerCount != c.layerCount() {
		sf.Free()
		return fmt.Errorf("layer_count mismatch: file=%d cache=%d", h.layerCount, c.layerCount())
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
	slog.Debug("kv cache rehydrate scanning",
		"cache_dir", c.cacheDir, "entries", len(entries))

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
			// Different model stamped this file. Leave it for another cache
			// instance; skip for this rehydration.
			continue
		}
		if h.layerCount != c.layerCount() {
			slog.Warn("kv cache rehydrate: layer_count mismatch, deleting", "path", path, "file_layers", h.layerCount, "expected", c.layerCount())
			_ = os.Remove(path)
			continue
		}
		ok = append(ok, scanned{name: name, path: path, size: st.Size(), h: h})
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

// restoreSnapshotArrays dispatches by snapshot_types metadata to rebuild
// typed Snapshots from a loaded safetensors file. Unknown types are
// treated as corrupt (per spec §10.3).
func (c *kvCache) restoreSnapshotArrays(sf *mlx.SafetensorsFile, h headerFields, startOffset, endOffset int) ([]cache.Snapshot, error) {
	snaps := make([]cache.Snapshot, h.layerCount)
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
			mlx.Pin(k, v)
			snaps[i] = cache.NewKVSnapshotFromArrays(k, v, startOffset, endOffset)
		default:
			return nil, fmt.Errorf("unknown snapshot type %q at layer %d", st, i)
		}
	}
	return snaps, nil
}

