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

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const cacheFormatVersion = "1"

// diskWriter serializes trieNode writes to disk on a single goroutine.
// Queue is a mutex-guarded slice (channels caused a shutdown deadlock in
// the prior branch — see commit 4f6e3111 in the historical persist branch).
type diskWriter struct {
	mu      sync.Mutex
	cond    *sync.Cond
	pending []*trieNode
	stopped bool
	done    chan struct{}
	cache   *kvCache

	consecutiveFailures int // circuit breaker per spec §6.3
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

// enqueue schedules node for writing. node.inflightWrite must already be a
// fresh channel — the loop closes it when this node is done.
func (w *diskWriter) enqueue(node *trieNode) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped {
		// Writer disabled; close the channel to unblock anyone waiting.
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
		disabled := w.stopped // snapshot under lock
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
	// No auto-requeue: the memory eviction pass (Task 9) calls scheduleWrite
	// again when the node next becomes a candidate, capped at writeAttempts < 3.
}

// shutdown blocks until the queue drains or the timeout elapses.
// Returns the number of still-pending nodes (0 on clean drain).
// timeout <= 0 means "wait until done".
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
	tmpPath := finalPath + ".tmp"

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
	atomic.AddInt64(&c.diskBytes, st.Size())
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

