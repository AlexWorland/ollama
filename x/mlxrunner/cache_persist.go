package mlxrunner

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const trieFormatVersion = 1

const (
	nodeFilePrefix = "node_"
	nodeFileSuffix = ".safetensors"
)

func nodeFileName(i int) string {
	return fmt.Sprintf("%s%d%s", nodeFilePrefix, i, nodeFileSuffix)
}

// persistedNode is the JSON-serializable form of a trieNode.
// The node's ID is its index in the persistedTrie.Nodes slice.
type persistedNode struct {
	Tokens    []int32  `json:"tokens"`
	EndOffset int      `json:"end_offset"`
	User      bool     `json:"user,omitempty"`
	Children  []int    `json:"children,omitempty"`
	SnapTypes []string `json:"snap_types,omitempty"` // per-layer: "kv","rotating","recurrent",""
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

// captureActiveFrontier snapshots the active path's leaf node so its live
// cache state is preserved before saving.
func (c *kvCache) captureActiveFrontier() {
	if len(c.activePath) <= 1 || len(c.caches) == 0 {
		return
	}
	frontier := c.activePath[len(c.activePath)-1]
	if frontier == c.root || frontier.hasAllSnapshots() {
		return
	}
	fromOffset := frontier.startOffset()
	slog.Debug("capturing active frontier snapshot", "offset", fromOffset, "tokens", len(frontier.tokens))
	snaps := make([]cache.Snapshot, len(c.caches))
	for j, kv := range c.caches {
		if kv == nil {
			continue
		}
		snaps[j] = kv.Snapshot(fromOffset)
	}
	frontier.setSnapshots(snaps, &c.pagedOutBytes)
}

// saveTrie serializes the trie topology and all snapshot data to cacheDir.
func (c *kvCache) saveTrie(cacheDir, modelID string) error {
	if c.root == nil {
		return nil
	}

	c.captureActiveFrontier()

	// Assign sequential IDs via depth-first walk.
	nodeMap := make(map[*trieNode]int)
	var nodes []*trieNode
	walkNodes(c.root, func(n *trieNode) bool {
		nodeMap[n] = len(nodes)
		nodes = append(nodes, n)
		return true
	})

	if len(nodes) <= 1 {
		slog.Debug("skipping trie save, only root node")
		return nil
	}
	slog.Debug("saving trie", "nodes", len(nodes), "layers", len(c.caches))

	persisted := persistedTrie{
		Version:   trieFormatVersion,
		ModelID:   modelID,
		NumLayers: len(c.caches),
		SavedAt:   time.Now().UTC().Format(time.RFC3339),
	}

	// Write new node files first (overwriting any with matching indices).
	// This avoids a crash-vulnerability window where old files are deleted
	// but new ones haven't been written yet.
	for i, node := range nodes {
		pn := persistedNode{
			Tokens:    node.tokens,
			EndOffset: node.endOffset,
			User:      node.user,
		}

		for _, child := range node.children {
			pn.Children = append(pn.Children, nodeMap[child])
		}

		if node.hasSnapshots() {
			pn.SnapTypes = make([]string, len(c.caches))
			arrays := make(map[string]*mlx.Array)
			metadata := make(map[string]string)

			for li, snap := range node.snapshots {
				exp := cache.ExportSnapshot(snap)
				if exp == nil {
					continue
				}
				pn.SnapTypes[li] = string(exp.Type)

				for name, arr := range exp.Arrays {
					arrays[fmt.Sprintf("layer_%d_%s", li, name)] = arr
				}
				for key, val := range exp.Metadata {
					metadata[fmt.Sprintf("layer_%d_%s", li, key)] = val
				}
			}

			if len(arrays) > 0 {
				path := filepath.Join(cacheDir, nodeFileName(i))
				slog.Debug("saving node snapshot", "node", i, "arrays", len(arrays))
				if err := mlx.SaveSafetensorsWithMetadata(path, arrays, metadata); err != nil {
					return fmt.Errorf("save node %d: %w", i, err)
				}
			}
		}

		persisted.Nodes = append(persisted.Nodes, pn)
	}

	// Write trie.json atomically, then clean up stale node files from a
	// previous save that had more nodes than the current one.
	data, err := json.Marshal(persisted)
	if err != nil {
		return fmt.Errorf("marshal trie: %w", err)
	}

	tmpPath := filepath.Join(cacheDir, "trie.json.tmp")
	finalPath := filepath.Join(cacheDir, "trie.json")
	if err := atomicWriteFile(tmpPath, finalPath, data); err != nil {
		return err
	}

	cleanStaleNodeFiles(cacheDir, len(nodes))

	slog.Info("KV cache saved", "nodes", len(nodes), "dir", cacheDir)
	return nil
}

// loadTrie deserializes the trie from cacheDir. Returns (nil, 0, nil) if
// no cache exists or the version/model doesn't match.
func loadTrie(cacheDir, modelID string, numLayers int) (*trieNode, int64, error) {
	triePath := filepath.Join(cacheDir, "trie.json")
	slog.Debug("loading trie", "path", triePath)
	data, err := os.ReadFile(triePath)
	if os.IsNotExist(err) {
		slog.Debug("no cached trie found")
		return nil, 0, nil
	}
	if err != nil {
		return nil, 0, fmt.Errorf("read trie.json: %w", err)
	}

	var persisted persistedTrie
	if err := json.Unmarshal(data, &persisted); err != nil {
		return nil, 0, fmt.Errorf("parse trie.json: %w", err)
	}

	if persisted.Version != trieFormatVersion {
		slog.Info("ignoring cached trie: version mismatch", "file", persisted.Version, "expected", trieFormatVersion)
		return nil, 0, nil
	}
	if persisted.ModelID != modelID {
		slog.Info("ignoring cached trie: model mismatch", "cached", persisted.ModelID, "current", modelID)
		return nil, 0, nil
	}
	if persisted.NumLayers != numLayers {
		slog.Info("ignoring cached trie: layer count mismatch", "cached", persisted.NumLayers, "current", numLayers)
		return nil, 0, nil
	}

	if len(persisted.Nodes) == 0 {
		return nil, 0, nil
	}

	// First pass: create all trieNode objects.
	trieNodes := make([]*trieNode, len(persisted.Nodes))
	for i, pn := range persisted.Nodes {
		trieNodes[i] = &trieNode{
			tokens:    pn.Tokens,
			endOffset: pn.EndOffset,
			user:      pn.User,
			lastUsed:  time.Now(), // treat all restored nodes as recently used
		}
	}

	// Second pass: wire parent/child pointers.
	for i, pn := range persisted.Nodes {
		for _, childID := range pn.Children {
			if childID < 0 || childID >= len(trieNodes) {
				return nil, 0, fmt.Errorf("node %d references invalid child %d", i, childID)
			}
			child := trieNodes[childID]
			child.parent = trieNodes[i]
			trieNodes[i].children = append(trieNodes[i].children, child)
		}
	}

	// Third pass: load snapshots.
	var pagedOutBytes int64
	for i, pn := range persisted.Nodes {
		if len(pn.SnapTypes) == 0 {
			continue
		}

		nodePath := filepath.Join(cacheDir, nodeFileName(i))
		sf, err := mlx.LoadSafetensorsNative(nodePath)
		if err != nil {
			slog.Warn("failed to load node snapshot, skipping", "node", i, "error", err)
			continue
		}

		snaps := make([]cache.Snapshot, numLayers)
		var nodeBytes int64
		allLoaded := true

		for layer := 0; layer < numLayers; layer++ {
			if layer >= len(pn.SnapTypes) || pn.SnapTypes[layer] == "" {
				continue
			}

			snapType := cache.SnapshotType(pn.SnapTypes[layer])
			arrays, meta := extractLayerData(sf, layer, snapType)
			if len(arrays) == 0 {
				slog.Warn("missing arrays for layer", "node", i, "layer", layer)
				allLoaded = false
				continue
			}

			snap, err := cache.ImportSnapshot(snapType, arrays, meta)
			if err != nil {
				slog.Warn("failed to import snapshot", "node", i, "layer", layer, "error", err)
				allLoaded = false
				continue
			}

			snaps[layer] = snap
			nodeBytes += int64(snap.Size())
		}

		sf.Free()

		if !allLoaded {
			// Close any successfully loaded snapshots for this node.
			for _, s := range snaps {
				if s != nil {
					s.Close()
				}
			}
			continue
		}

		trieNodes[i].snapshots = snaps
		pagedOutBytes += nodeBytes
		slog.Debug("loaded node snapshot", "node", i, "bytes", nodeBytes)
	}

	slog.Debug("trie loaded", "nodes", len(trieNodes), "paged_out_bytes", pagedOutBytes)
	root := trieNodes[0]
	return root, pagedOutBytes, nil
}

// extractLayerData pulls arrays and metadata for a specific layer from a
// loaded safetensors file. Array names are expected to have the format
// "layer_{i}_{name}".
func extractLayerData(sf *mlx.SafetensorsFile, layer int, snapType cache.SnapshotType) (map[string]*mlx.Array, map[string]string) {
	prefix := fmt.Sprintf("layer_%d_", layer)
	arrays := make(map[string]*mlx.Array)
	meta := make(map[string]string)

	arrayNames, metaNames := snapType.FieldNames()

	for _, name := range arrayNames {
		arr := sf.Get(prefix + name)
		if arr != nil {
			arrays[name] = arr
		}
	}

	for _, name := range metaNames {
		val := sf.GetMetadata(prefix + name)
		if val != "" {
			meta[name] = val
		}
	}

	return arrays, meta
}

// cleanStaleNodeFiles removes node_*.safetensors files with indices >= nodeCount.
// These are leftovers from a previous save that had more nodes than the current one.
// Best-effort: logs and continues on individual removal failures.
func cleanStaleNodeFiles(dir string, nodeCount int) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, nodeFilePrefix) || !strings.HasSuffix(name, nodeFileSuffix) {
			continue
		}
		idxStr := strings.TrimPrefix(strings.TrimSuffix(name, nodeFileSuffix), nodeFilePrefix)
		idx, err := strconv.Atoi(idxStr)
		if err != nil {
			continue
		}
		if idx >= nodeCount {
			if err := os.Remove(filepath.Join(dir, name)); err != nil && !os.IsNotExist(err) {
				slog.Warn("failed to remove stale cache file", "file", name, "error", err)
			}
		}
	}
}

// atomicWriteFile writes data to tmpPath, fsyncs it, then renames to finalPath.
// The tmp file is cleaned up on any error.
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
