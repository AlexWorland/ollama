package mlxrunner

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

const trieFormatVersion = 1

// persistedNode is the JSON-serializable form of a trieNode.
type persistedNode struct {
	ID        int      `json:"id"`
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
// creating it if necessary. The path is ~/.ollama/cache/kv/<sanitized-digest>/.
func kvCacheDir(modelDigest string) (string, error) {
	// Models() returns ~/.ollama/models; go one level up for ~/.ollama/
	base := filepath.Dir(envconfig.Models())
	digest := strings.Replace(modelDigest, ":", "-", 1)
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
		// Only root with no children — nothing worth saving.
		return nil
	}

	// Clean out any old node files before writing new ones.
	if err := cleanNodeFiles(cacheDir); err != nil {
		slog.Warn("failed to clean old cache files", "error", err)
	}

	persisted := persistedTrie{
		Version:   trieFormatVersion,
		ModelID:   modelID,
		NumLayers: len(c.caches),
		SavedAt:   time.Now().UTC().Format(time.RFC3339),
	}

	for _, node := range nodes {
		id := nodeMap[node]
		pn := persistedNode{
			ID:        id,
			Tokens:    node.tokens,
			EndOffset: node.endOffset,
			User:      node.user,
		}

		// Record children IDs.
		for _, child := range node.children {
			pn.Children = append(pn.Children, nodeMap[child])
		}

		// Export and save snapshots.
		if node.hasSnapshots() {
			pn.SnapTypes = make([]string, len(c.caches))
			arrays := make(map[string]*mlx.Array)
			metadata := make(map[string]string)

			for i, snap := range node.snapshots {
				exp := cache.ExportSnapshot(snap)
				if exp == nil {
					continue
				}
				pn.SnapTypes[i] = string(exp.Type)

				// Prefix array names with layer index.
				for name, arr := range exp.Arrays {
					arrays[fmt.Sprintf("layer_%d_%s", i, name)] = arr
				}
				for key, val := range exp.Metadata {
					metadata[fmt.Sprintf("layer_%d_%s", i, key)] = val
				}
			}

			if len(arrays) > 0 {
				path := filepath.Join(cacheDir, fmt.Sprintf("node_%d.safetensors", id))
				if err := mlx.SaveSafetensorsWithMetadata(path, arrays, metadata); err != nil {
					return fmt.Errorf("save node %d: %w", id, err)
				}
			}
		}

		persisted.Nodes = append(persisted.Nodes, pn)
	}

	// Write trie.json atomically.
	data, err := json.MarshalIndent(persisted, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal trie: %w", err)
	}

	tmpPath := filepath.Join(cacheDir, "trie.json.tmp")
	finalPath := filepath.Join(cacheDir, "trie.json")
	if err := os.WriteFile(tmpPath, data, 0o600); err != nil {
		return fmt.Errorf("write trie.json.tmp: %w", err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return fmt.Errorf("rename trie.json: %w", err)
	}

	slog.Info("KV cache saved", "nodes", len(nodes), "dir", cacheDir)
	return nil
}

// loadTrie deserializes the trie from cacheDir. Returns (nil, 0, nil) if
// no cache exists or the version/model doesn't match.
func loadTrie(cacheDir, modelID string, numLayers int) (*trieNode, int64, error) {
	triePath := filepath.Join(cacheDir, "trie.json")
	data, err := os.ReadFile(triePath)
	if os.IsNotExist(err) {
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

		nodePath := filepath.Join(cacheDir, fmt.Sprintf("node_%d.safetensors", pn.ID))
		sf, err := mlx.LoadSafetensorsNative(nodePath)
		if err != nil {
			slog.Warn("failed to load node snapshot, skipping", "node", pn.ID, "error", err)
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
				slog.Warn("missing arrays for layer", "node", pn.ID, "layer", layer)
				allLoaded = false
				continue
			}

			snap, err := cache.ImportSnapshot(snapType, arrays, meta)
			if err != nil {
				slog.Warn("failed to import snapshot", "node", pn.ID, "layer", layer, "error", err)
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
	}

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

	// Determine which array names to look for based on type.
	var arrayNames []string
	var metaNames []string
	switch snapType {
	case cache.SnapshotTypeKV:
		arrayNames = []string{"keys", "values"}
		metaNames = []string{"from_offset", "to_offset"}
	case cache.SnapshotTypeRotating:
		arrayNames = []string{"keys", "values"}
		metaNames = []string{"from_offset", "to_offset", "idx"}
	case cache.SnapshotTypeRecurrent:
		arrayNames = []string{"conv_state", "delta_state"}
		metaNames = []string{"offset"}
	}

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

// cleanNodeFiles removes old node_*.safetensors files from the cache directory.
func cleanNodeFiles(dir string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "node_") && strings.HasSuffix(e.Name(), ".safetensors") {
			if err := os.Remove(filepath.Join(dir, e.Name())); err != nil {
				return err
			}
		}
	}
	return nil
}
