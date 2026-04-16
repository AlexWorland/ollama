package mlxrunner

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const trieFormatVersion = 2

// nodeFileHash returns a content-addressable filename for a trie node based
// on its cumulative token path from root. Two nodes with the same path
// always produce the same hash; different paths never collide in practice.
func nodeFileHash(node *trieNode) string {
	// Walk parent chain to collect full token path root→node.
	var segments [][]int32
	total := 0
	for n := node; n != nil && len(n.tokens) > 0; n = n.parent {
		segments = append(segments, n.tokens)
		total += len(n.tokens)
	}

	buf := make([]byte, total*4)
	off := len(buf) // fill from end (segments are child→root order)
	for _, seg := range segments {
		off -= len(seg) * 4
		for i, tok := range seg {
			binary.LittleEndian.PutUint32(buf[off+i*4:], uint32(tok))
		}
	}

	sum := sha256.Sum256(buf)
	return fmt.Sprintf("%x.safetensors", sum[:8])
}

func layerKey(layer int, field string) string {
	return fmt.Sprintf("layer_%d_%s", layer, field)
}

// persistedNode is the JSON-serializable form of a trieNode.
type persistedNode struct {
	Tokens    []int32              `json:"tokens"`
	EndOffset int                  `json:"end_offset"`
	User      bool                 `json:"user,omitempty"`
	Children  []int                `json:"children,omitempty"`
	SnapTypes []cache.SnapshotType `json:"snap_types,omitempty"` // per-layer: "kv","rotating","recurrent",""
	File      string               `json:"file,omitempty"`       // content-addressed filename
}

// persistedTrie is the JSON-serializable form of the full trie.
type persistedTrie struct {
	Version   int             `json:"version"`
	ModelID   string          `json:"model_id"`
	NumLayers int             `json:"num_layers"`
	Nodes     []persistedNode `json:"nodes"`
	SavedAt   string          `json:"saved_at"`
}

// kvCacheDir returns the directory for persisting KV cache for a given model,
// creating it if necessary. The base is derived from envconfig.Models() so it
// respects OLLAMA_MODELS overrides (e.g. <models-parent>/cache/kv/<digest>/).
func kvCacheDir(modelDigest string) (string, error) {
	base := filepath.Dir(envconfig.Models()) // e.g. ~/.ollama
	digest := strings.ReplaceAll(modelDigest, ":", "-")
	dir := filepath.Join(base, "cache", "kv", digest)
	return dir, os.MkdirAll(dir, 0o700)
}

func (c *kvCache) captureActiveFrontier() {
	if len(c.activePath) <= 1 || len(c.caches) == 0 {
		return
	}
	frontier := c.activePath[len(c.activePath)-1]
	if frontier == c.root || frontier.hasAllSnapshots() {
		return
	}
	fromOffset := frontier.startOffset()
	snaps := make([]cache.Snapshot, len(c.caches))
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		snaps[j] = kv.Snapshot(fromOffset)
	}
	frontier.setSnapshots(snaps, &c.pagedOutBytes)
}

func (c *kvCache) saveTrie() error {
	cacheDir := c.cacheDir
	modelID := c.modelID
	if c.root == nil {
		return nil
	}

	c.captureActiveFrontier()

	nodes, nodeMap := indexNodes(c.root)

	if len(nodes) <= 1 {
		return nil
	}

	persisted := persistedTrie{
		Version:   trieFormatVersion,
		ModelID:   modelID,
		NumLayers: len(c.caches),
		SavedAt:   time.Now().UTC().Format(time.RFC3339),
	}

	referenced := make(map[string]bool)

	for _, node := range nodes {
		pn := persistedNode{
			Tokens:    node.tokens,
			EndOffset: node.endOffset,
			User:      node.user,
		}

		for _, child := range node.children {
			pn.Children = append(pn.Children, nodeMap[child])
		}

		if node.hasSnapshots() {
			arrays, metadata, types := exportNodeSnapshots(node.snapshots)
			pn.SnapTypes = types

			if len(arrays) > 0 {
				hash := nodeFileHash(node)
				path := filepath.Join(cacheDir, hash)
				if _, err := os.Stat(path); err != nil {
					// Content-addressed file doesn't exist yet — write it.
					if _, err := atomicSaveSafetensors(cacheDir, hash, arrays, metadata); err != nil {
						return fmt.Errorf("save node %d: %w", nodeMap[node], err)
					}
				}
				pn.File = hash
				referenced[hash] = true
			}
		} else if node.diskFile != "" {
			// Cold node: already on disk with a content-addressed name
			// from evictNodeToDisk. No rename or copy needed.
			name := filepath.Base(node.diskFile)
			pn.File = name
			pn.SnapTypes = node.snapTypes
			referenced[name] = true
		}

		persisted.Nodes = append(persisted.Nodes, pn)
	}

	data, err := json.Marshal(persisted)
	if err != nil {
		return fmt.Errorf("marshal trie: %w", err)
	}

	tmpPath := filepath.Join(cacheDir, "trie.json.tmp")
	finalPath := filepath.Join(cacheDir, "trie.json")
	if err := atomicWriteFile(tmpPath, finalPath, data); err != nil {
		return err
	}

	cleanUnreferencedFiles(cacheDir, referenced)

	slog.Info("KV cache saved", "nodes", len(nodes), "dir", cacheDir)
	return nil
}

// loadTrie deserializes the trie from cacheDir. Nodes with disk-backed files
// are restored lazily: diskFile/diskFileSize/snapTypes are set from metadata,
// but no safetensors files are opened. Actual loading happens on demand via
// loadNodeFromDisk.
func loadTrie(cacheDir, modelID string, numLayers int) (root *trieNode, totalDiskBytes int64, referenced map[string]bool, err error) {
	triePath := filepath.Join(cacheDir, "trie.json")
	data, err := os.ReadFile(triePath)
	if os.IsNotExist(err) {
		return nil, 0, nil, nil
	}
	if err != nil {
		return nil, 0, nil, fmt.Errorf("read trie.json: %w", err)
	}

	var persisted persistedTrie
	if err := json.Unmarshal(data, &persisted); err != nil {
		return nil, 0, nil, fmt.Errorf("parse trie.json: %w", err)
	}

	if persisted.Version != trieFormatVersion {
		slog.Info("ignoring cached trie: version mismatch", "file", persisted.Version, "expected", trieFormatVersion)
		return nil, 0, nil, nil
	}
	if persisted.ModelID != modelID {
		slog.Info("ignoring cached trie: model mismatch", "cached", persisted.ModelID, "current", modelID)
		return nil, 0, nil, nil
	}
	if persisted.NumLayers != numLayers {
		slog.Info("ignoring cached trie: layer count mismatch", "cached", persisted.NumLayers, "current", numLayers)
		return nil, 0, nil, nil
	}

	if len(persisted.Nodes) == 0 {
		return nil, 0, nil, nil
	}

	trieNodes := make([]*trieNode, len(persisted.Nodes))
	for i, pn := range persisted.Nodes {
		trieNodes[i] = &trieNode{
			tokens:    pn.Tokens,
			endOffset: pn.EndOffset,
			user:      pn.User,
			lastUsed:  time.Now(),
		}
	}

	for i, pn := range persisted.Nodes {
		for _, childID := range pn.Children {
			if childID < 0 || childID >= len(trieNodes) {
				return nil, 0, nil, fmt.Errorf("node %d references invalid child %d", i, childID)
			}
			child := trieNodes[childID]
			child.parent = trieNodes[i]
			trieNodes[i].children = append(trieNodes[i].children, child)
		}
	}

	referenced = make(map[string]bool)
	for i, pn := range persisted.Nodes {
		if pn.File == "" || len(pn.SnapTypes) == 0 {
			continue
		}

		expectedHash := nodeFileHash(trieNodes[i])
		if pn.File != expectedHash {
			slog.Warn("node file hash mismatch, skipping", "node", i,
				"stored", pn.File, "expected", expectedHash)
			continue
		}

		nodePath := filepath.Join(cacheDir, pn.File)
		info, err := os.Stat(nodePath)
		if err != nil {
			slog.Warn("persisted node file missing, skipping", "node", i, "file", pn.File)
			continue
		}

		referenced[pn.File] = true
		trieNodes[i].diskFile = nodePath
		trieNodes[i].diskFileSize = info.Size()
		trieNodes[i].snapTypes = pn.SnapTypes
		totalDiskBytes += info.Size()
	}

	return trieNodes[0], totalDiskBytes, referenced, nil
}

func closeSnapshots(snaps []cache.Snapshot) {
	for _, s := range snaps {
		if s != nil {
			s.Close()
		}
	}
}

func extractLayerData(sf *mlx.SafetensorsFile, layer int, snapType cache.SnapshotType) (map[string]*mlx.Array, map[string]string) {
	arrays := make(map[string]*mlx.Array)
	meta := make(map[string]string)

	arrayNames, metaNames := snapType.FieldNames()

	for _, name := range arrayNames {
		arr := sf.Get(layerKey(layer, name))
		if arr != nil {
			arrays[name] = arr
		}
	}

	for _, name := range metaNames {
		val := sf.GetMetadata(layerKey(layer, name))
		if val != "" {
			meta[name] = val
		}
	}

	return arrays, meta
}

func exportNodeSnapshots(snapshots []cache.Snapshot) (map[string]*mlx.Array, map[string]string, []cache.SnapshotType) {
	arrays := make(map[string]*mlx.Array)
	metadata := make(map[string]string)
	snapTypes := make([]cache.SnapshotType, len(snapshots))

	for li, snap := range snapshots {
		exp := cache.ExportSnapshot(snap)
		if exp == nil {
			continue
		}
		snapTypes[li] = exp.Type

		for name, arr := range exp.Arrays {
			arrays[layerKey(li, name)] = arr
		}
		for key, val := range exp.Metadata {
			metadata[layerKey(li, key)] = val
		}
	}

	return arrays, metadata, snapTypes
}

// The node remains in the trie with diskFile set so it can be reloaded on
// demand. The file is content-addressed so saveTrie can reference it without
// renaming.
func (c *kvCache) evictNodeToDisk(node *trieNode) error {
	if c.cacheDir == "" {
		return fmt.Errorf("cacheDir not set")
	}

	filename := nodeFileHash(node)
	warmPath := filepath.Join(c.cacheDir, filename)
	if info, err := os.Stat(warmPath); err == nil {
		slog.Info("fast re-eviction via warm cache", "offset", node.startOffset(),
			"tokens", len(node.tokens), "path", filename)
		node.snapTypes = snapshotTypes(node.snapshots)
		node.diskFile = warmPath
		node.diskFileSize = info.Size()
		node.setSnapshots(nil, &c.pagedOutBytes)
		c.totalDiskBytes += info.Size()
		c.emitEvent(EventEvictToDisk, node, info.Size(), filename)
		return nil
	}

	arrays, metadata, snapTypes := exportNodeSnapshots(node.snapshots)
	if len(arrays) == 0 {
		return fmt.Errorf("no arrays to evict")
	}

	mlx.Eval(slices.Collect(maps.Values(arrays))...)

	data, err := mlx.SerializeSafetensors(arrays, metadata)
	if err != nil {
		return fmt.Errorf("serialize evicted node: %w", err)
	}

	slog.Info("evicting node to disk", "offset", node.startOffset(),
		"tokens", len(node.tokens), "bytes", node.snapshotBytes(), "path", filename)

	// Disk state is optimistic; processDiskCompletions corrects it on failure.
	node.snapTypes = snapTypes
	node.diskFile = filepath.Join(c.cacheDir, filename)
	node.diskFileSize = int64(len(data))
	node.setSnapshots(nil, &c.pagedOutBytes)
	c.totalDiskBytes += int64(len(data))

	c.emitEvent(EventEvictToDisk, node, int64(len(data)), filename)

	if c.diskWriter != nil {
		c.diskWriter.submit(diskWriteJob{
			data:     data,
			filename: filename,
			node:     node,
		})
	} else {
		// Synchronous fallback for tests without a diskWriter.
		tmpPath := filepath.Join(c.cacheDir, ".tmp_"+filename)
		finalPath := filepath.Join(c.cacheDir, filename)
		if err := atomicWriteFile(tmpPath, finalPath, data); err != nil {
			c.clearNodeDiskState(node)
			return fmt.Errorf("write evicted node: %w", err)
		}
	}

	return nil
}

func (c *kvCache) loadNodeFromDisk(node *trieNode) error {
	// Wait for in-flight async write before opening the file.
	if c.diskWriter != nil {
		c.diskWriter.waitForFile(filepath.Base(node.diskFile))
	}

	sf, err := mlx.LoadSafetensorsNative(node.diskFile)
	if err != nil {
		return fmt.Errorf("load %s: %w", node.diskFile, err)
	}
	defer sf.Free()

	snaps := make([]cache.Snapshot, len(c.caches))
	for layer := 0; layer < len(c.caches); layer++ {
		if layer >= len(node.snapTypes) || node.snapTypes[layer] == "" {
			continue
		}

		snapType := node.snapTypes[layer]
		arrays, meta := extractLayerData(sf, layer, snapType)
		if len(arrays) == 0 {
			closeSnapshots(snaps)
			return fmt.Errorf("missing arrays for layer %d", layer)
		}

		snap, err := cache.ImportSnapshot(snapType, arrays, meta)
		if err != nil {
			closeSnapshots(snaps)
			return fmt.Errorf("import layer %d: %w", layer, err)
		}
		snaps[layer] = snap
	}

	slog.Info("loaded node from disk", "offset", node.startOffset(),
		"tokens", len(node.tokens), "path", filepath.Base(node.diskFile))

	// Keep file on disk as warm cache for fast re-eviction.
	c.totalDiskBytes -= node.diskFileSize
	node.diskFile = ""
	node.diskFileSize = 0

	node.setSnapshots(snaps, &c.pagedOutBytes)
	return nil
}

func (c *kvCache) enforceDiskEvictionPolicy() {
	diskCap := int64(envconfig.KvCacheDiskMax())
	if diskCap <= 0 {
		return
	}

	// Fast path: totalDiskBytes is maintained incrementally, so we can
	// skip the full trie walk when under cap (the common case).
	if c.totalDiskBytes <= diskCap {
		return
	}

	activeSet := c.activeSet()

	// Walk to find eviction candidates (only needed when over cap).
	type diskEntry struct {
		node *trieNode
		size int64
	}
	var entries []diskEntry
	walkNodes(c.root, func(n *trieNode) bool {
		if n.diskFile != "" && !n.hasSnapshots() && !activeSet[n] {
			entries = append(entries, diskEntry{n, n.diskFileSize})
		}
		return true
	})

	// Sort oldest-first and delete until under cap.
	slices.SortFunc(entries, func(a, b diskEntry) int {
		return a.node.lastUsed.Compare(b.node.lastUsed)
	})

	for _, e := range entries {
		if c.totalDiskBytes <= diskCap {
			break
		}
		slog.Info("disk eviction cap exceeded, deleting oldest",
			"offset", e.node.startOffset(), "tokens", len(e.node.tokens),
			"path", filepath.Base(e.node.diskFile))
		c.emitEvent(EventEvictFromDisk, e.node, e.size, filepath.Base(e.node.diskFile))
		os.Remove(e.node.diskFile)
		c.clearNodeDiskState(e.node)

		if len(e.node.children) == 0 {
			removeNode(e.node, &c.pagedOutBytes)
		}
	}
}

// processDiskCompletions drains completed background writes, undoing
// optimistic state for any failures. Called at the top of begin() on the
// inference goroutine -- no concurrency concerns.
func (c *kvCache) processDiskCompletions() {
	if c.diskWriter == nil {
		return
	}
	for _, result := range c.diskWriter.drainResults() {
		if result.err == nil {
			continue
		}
		slog.Warn("async disk write failed", "error", result.err,
			"file", filepath.Base(result.node.diskFile))
		c.clearNodeDiskState(result.node)
		if len(result.node.children) == 0 {
			removeNode(result.node, &c.pagedOutBytes)
		}
	}
}

// clearNodeDiskState undoes the optimistic disk state set by evictNodeToDisk.
func (c *kvCache) clearNodeDiskState(node *trieNode) {
	c.totalDiskBytes -= node.diskFileSize
	node.diskFile = ""
	node.diskFileSize = 0
	node.snapTypes = nil
}

// snapshotTypes extracts the per-layer SnapshotType from live snapshots.
func snapshotTypes(snapshots []cache.Snapshot) []cache.SnapshotType {
	types := make([]cache.SnapshotType, len(snapshots))
	for i, snap := range snapshots {
		if exp := cache.ExportSnapshot(snap); exp != nil {
			types[i] = exp.Type
		}
	}
	return types
}

func cleanUnreferencedFiles(dir string, referenced map[string]bool) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, ".safetensors") && !referenced[name] {
			os.Remove(filepath.Join(dir, name))
		}
	}
}

func atomicSaveSafetensors(dir, filename string, arrays map[string]*mlx.Array, metadata map[string]string) (int64, error) {
	// Use a .tmp_ prefix (not suffix) so the file retains the .safetensors
	// extension that the MLX C library expects.
	tmpPath := filepath.Join(dir, ".tmp_"+filename)
	finalPath := filepath.Join(dir, filename)

	if err := mlx.SaveSafetensorsWithMetadata(tmpPath, arrays, metadata); err != nil {
		os.Remove(tmpPath)
		return 0, err
	}

	info, err := os.Stat(tmpPath)
	if err != nil {
		os.Remove(tmpPath)
		return 0, fmt.Errorf("stat %s: %w", tmpPath, err)
	}
	fileSize := info.Size()

	if err := os.Rename(tmpPath, finalPath); err != nil {
		os.Remove(tmpPath)
		return 0, fmt.Errorf("rename %s: %w", finalPath, err)
	}

	return fileSize, nil
}

// atomicWriteFile writes data via tmp+rename for crash safety.
func atomicWriteFile(tmpPath, finalPath string, data []byte) error {
	f, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("create %s: %w", tmpPath, err)
	}

	cleanup := true
	defer func() {
		if cleanup {
			os.Remove(tmpPath)
		}
	}()

	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("write %s: %w", tmpPath, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("fsync %s: %w", tmpPath, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", tmpPath, err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return fmt.Errorf("rename %s: %w", finalPath, err)
	}
	cleanup = false
	return nil
}
