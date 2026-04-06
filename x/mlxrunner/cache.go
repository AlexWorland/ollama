// cache.go manages a shared KV cache across conversations using a compressed
// prefix trie. Each trie node stores a token sequence (edge) and optional
// per-layer snapshots that can be paged in/out of the live MLX cache arrays.
//
// Key properties:
//   - Only one path through the trie is "active" (backed by live MLX arrays)
//     at a time. Switching paths pages out the frontier node and pages in the
//     new path.
//   - Snapshots are only captured at the frontier (end) of the active path.
//     Intermediate node snapshots come from split prefill.
//   - All cache layers must stay at the same token offset.
//   - Sibling edges must not share a common token prefix (compressed trie
//     invariant).
//   - begin() always re-evaluates at least one token so the pipeline can seed
//     generation, even on a full prefix match.

package mlxrunner

import (
	"cmp"
	"fmt"
	"log/slog"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

const maxPagedOutBytes int64 = 8 << 30 // 8 GiB eviction threshold for paged-out snapshot memory

// All methods are called from the single request-processing goroutine in
// Runner.Run (via the Requests channel). No concurrent access protection
// is needed.
type kvCache struct {
	root          *trieNode   // root of the prefix trie
	activePath    []*trieNode // current root→leaf path with live MLX arrays
	caches        []cache.Cache
	pagedOutBytes int64 // total bytes in paged-out snapshots across the trie

	// Disk-backed eviction state.
	cacheDir       string // directory for persisted/evicted safetensors files
	modelID        string // for validation on reload
	totalDiskBytes int64  // running total of disk-backed node file sizes

	// Visualization (safe for cross-goroutine reads via atomic/bus).
	events       *cacheEventBus
	trieSnapshot atomic.Pointer[TrieSnapshot]
}

func (c *kvCache) activeSet() map[*trieNode]bool {
	s := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		s[n] = true
	}
	return s
}

// pendingSnapshot is a snapshot scheduled to be taken during prefill.
type pendingSnapshot struct {
	offset int
	user   bool
}

// cacheSession manages caches for a single pipeline run.
// Callers should append generated tokens to outputs and
// defer close to save the cache state.
type cacheSession struct {
	cache   *kvCache
	inputs  []int32
	outputs []int32

	caches    []cache.Cache
	remaining []int32

	// pendingSnapshots lists offsets where snapshots should be captured
	// during prefill, sorted by offset. Entries are consumed as the
	// cache advances past them.
	pendingSnapshots []pendingSnapshot
}

func (c *kvCache) ensureCaches(m base.Model) {
	if len(c.caches) != 0 {
		return
	}
	if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
		c.caches = cacheFactory.NewCaches()
		return
	}
	c.caches = make([]cache.Cache, m.NumLayers())
	for i := range c.caches {
		c.caches[i] = cache.NewKVCache()
	}
}

func (c *kvCache) ensureRoot() {
	if c.root == nil {
		c.root = &trieNode{
			lastUsed: time.Now(),
		}
		c.activePath = []*trieNode{c.root}
	}
}

// begin prepares caches for a new request. It finds the nearest
// matching cache or creates new caches if none match.
func (c *kvCache) begin(m base.Model, inputs []int32) *cacheSession {
	c.ensureCaches(m)
	c.ensureRoot()

	matchPath, matched := findBestMatch(c.root, inputs)
	originalMatched := matched

	// Always keep at least one token to re-evaluate so the
	// pipeline can seed token generation from it.
	if matched == len(inputs) && matched > 0 {
		matchPath, matched = findBestMatch(c.root, inputs[:len(inputs)-1])
	}

	// Switch to the matched path, paging in/out as needed.
	c.switchToPath(matchPath, matched)

	// switchToPath aligns caches to a common offset
	prefix := c.minCacheOffset()
	remaining := inputs[prefix:]

	session := &cacheSession{
		cache:     c,
		inputs:    inputs,
		caches:    c.caches,
		remaining: remaining,
	}

	// Schedule a snapshot at the branch point during prefill so future
	// requests diverging here can restore instead of re-evaluating.
	if prefix < matched {
		session.pendingSnapshots = append(session.pendingSnapshots, pendingSnapshot{offset: matched, user: false})
	}

	msg := "cache hit"
	if prefix == 0 {
		msg = "cache miss"
	}
	slog.Info(msg, "total", len(inputs), "matched", originalMatched, "cached", prefix, "left", len(remaining))

	return session
}

// switchToPath transitions from the current active path to a new path,
// paging out diverging segments and paging in the new path.
func (c *kvCache) switchToPath(newPath []*trieNode, matched int) {
	defer c.enforceEvictionPolicy()

	// Find common ancestor index.
	commonLen := 0
	for commonLen < len(c.activePath) && commonLen < len(newPath) {
		if c.activePath[commonLen] != newPath[commonLen] {
			break
		}
		commonLen++
	}

	ancestorOffset := 0
	if commonLen > 0 {
		ancestorOffset = c.activePath[commonLen-1].endOffset
	}

	var pageOutCount, pageInCount int

	// Page out the leaf of the old path. Only the leaf's live cache
	// state is correct — intermediate nodes already have snapshots
	// captured during their creation (splitNode + prefill). Snapshotting
	// non-leaf nodes here would produce wrong results for non-rewindable
	// caches (e.g. RecurrentCache) whose state reflects the leaf, not
	// the intermediate boundary.
	leaf := len(c.activePath) - 1
	leafDiverges := leaf >= commonLen
	leafNeedsRewind := matched < c.activePath[leaf].endOffset
	if leafDiverges || leafNeedsRewind {
		node := c.activePath[leaf]
		if !node.hasAllSnapshots() {
			fromOffset := node.startOffset()
			snaps := make([]cache.Snapshot, len(c.caches))
			for j, kv := range c.caches {
				if kv == nil {
					continue
				}
				snaps[j] = kv.Snapshot(fromOffset)
			}
			node.setSnapshots(snaps, &c.pagedOutBytes)
			c.emitEvent(EventPageOut, node, node.snapshotBytes(), "")
			pageOutCount++
			logutil.Trace(fmt.Sprintf("page out: [%d, %d)", fromOffset, node.endOffset))
		}
	}

	// Rewind each cache to the target offset or free it. When matched
	// falls within the ancestor's range (same-path case), we rewind
	// directly to the match point. Otherwise we rewind to the ancestor
	// and let page-in bring us forward to matched.
	rewindTarget := min(ancestorOffset, matched)
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if !kv.Restore(nil, rewindTarget) {
			kv.Free()
		}
	}

	// Page in — walk the full new path, restoring from snapshots.
	// Freed caches naturally pick up the first available snapshot.
	// Caches already past a node skip it via offset check.
	var pageInFromDisk int
pageIn:
	for _, node := range newPath {
		// Load from disk if this node was evicted.
		if !node.hasSnapshots() && node.diskFile != "" {
			diskBase := filepath.Base(node.diskFile) // capture before loadNodeFromDisk clears it
			slog.Debug("switchToPath: loading evicted node from disk",
				"offset", node.startOffset(), "tokens", len(node.tokens),
				"diskFile", diskBase)
			if err := c.loadNodeFromDisk(node); err != nil {
				slog.Warn("failed to load node from disk", "error", err,
					"path", node.diskFile)
				break pageIn
			}
			c.emitEvent(EventPageInDisk, node, node.snapshotBytes(), diskBase)
			pageInFromDisk++
		}
		if !node.hasSnapshots() {
			continue
		}
		nodeTarget := min(node.endOffset, matched)
		for j, kv := range c.caches {
			if kv == nil {
				continue
			}
			if j >= len(node.snapshots) || node.snapshots[j] == nil {
				continue
			}
			if kv.Offset() >= nodeTarget {
				continue
			}
			if !kv.Restore(node.snapshots[j], nodeTarget) {
				// Restore failed — stop page-in and let alignment
				// bring all caches to a consistent offset.
				break pageIn
			}
		}
		if node.endOffset > ancestorOffset {
			c.emitEvent(EventPageIn, node, node.snapshotBytes(), "")
			pageInCount++
			logutil.Trace(fmt.Sprintf("page in: [%d, %d)", node.startOffset(), nodeTarget))
		}
	}

	// Align all caches to the minimum offset.
	c.activePath = newPath
	minOff := c.minCacheOffset()
	for _, kv := range c.caches {
		if kv != nil && kv.Offset() != minOff {
			if !kv.Restore(nil, minOff) {
				slog.Warn("failed to restore cache, freeing all caches", "offset", minOff)
				c.freeAll()
				break
			}
		}
	}
	for i := len(c.activePath) - 1; i >= 0; i-- {
		if c.activePath[i].endOffset <= minOff {
			c.activePath = c.activePath[:i+1]
			break
		}
	}

	// Update last-used time on only the final used node. For recurrent
	// caches we don't need the intermediate snapshots and for KV caches
	// we can reslice the data out of merged edges.
	if len(c.activePath) > 0 {
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}

	// Disk page-in is noteworthy (recovered context from SSD) — log at Info.
	// In-memory page-in/out is routine per-request work — log at Debug.
	if pageInFromDisk > 0 {
		slog.Info("switching cache path", "page_out", pageOutCount, "page_in", pageInCount, "page_in_disk", pageInFromDisk)
	} else if pageOutCount > 0 || pageInCount > 0 {
		slog.Debug("switching cache path", "page_out", pageOutCount, "page_in", pageInCount)
	}
}

// requestSnapshot schedules a user snapshot at the given absolute token
// offset. The snapshot will be captured during prefill when the cache
// reaches this offset.
func (s *cacheSession) requestSnapshot(offset int) {
	baseOffset := len(s.inputs) - len(s.remaining)
	if offset <= baseOffset || offset > len(s.inputs) {
		return
	}
	// Deduplicate: if this offset already exists, upgrade to user.
	for i := range s.pendingSnapshots {
		if s.pendingSnapshots[i].offset == offset {
			s.pendingSnapshots[i].user = true
			return
		}
	}
	s.pendingSnapshots = append(s.pendingSnapshots, pendingSnapshot{offset: offset, user: true})
	slices.SortFunc(s.pendingSnapshots, func(a, b pendingSnapshot) int {
		return a.offset - b.offset
	})
}

// nextPendingSnapshot returns the offset of the next pending snapshot,
// or 0 if there are none.
func (s *cacheSession) nextPendingSnapshot() int {
	if len(s.pendingSnapshots) == 0 {
		return 0
	}
	return s.pendingSnapshots[0].offset
}

// snapshot creates a snapshot at the current cache position. It determines
// whether this is a user snapshot by consuming pending entries whose offset
// has been reached.
func (s *cacheSession) snapshot() {
	c := s.cache
	cacheOffset := c.minCacheOffset()
	if cacheOffset <= 0 {
		return
	}

	// Consume pending snapshots up to the current offset and derive
	// the user flag from them.
	user := false
	for len(s.pendingSnapshots) > 0 && cacheOffset >= s.pendingSnapshots[0].offset {
		if s.pendingSnapshots[0].user {
			user = true
		}
		s.pendingSnapshots = s.pendingSnapshots[1:]
	}

	// The last node in activePath is the frontier where caches are advancing.
	// cacheOffset is always >= its endOffset: begin() restores caches to this
	// boundary and prefill advances monotonically forward.
	frontier := c.activePath[len(c.activePath)-1]

	// If the frontier already ends at cacheOffset, just ensure it has snapshots.
	if frontier.endOffset == cacheOffset {
		if user {
			frontier.user = true
		}
		if !frontier.hasAllSnapshots() {
			s.attachSnapshots(frontier, cacheOffset)
		}
		return
	}

	if frontier.endOffset > cacheOffset {
		slog.Warn("snapshot skipped: cacheOffset is behind frontier", "cacheOffset", cacheOffset, "frontierEndOffset", frontier.endOffset)
		return
	}

	// Advance the trie to cacheOffset — find or create a node there.
	edgeTokens := append(s.inputs, s.outputs...)[frontier.endOffset:cacheOffset]
	frontier = c.advancePath(frontier, edgeTokens, cacheOffset)

	// Attach fresh snapshots from the live caches. Always use fresh
	// snapshots even if the node already has some (e.g. from splitNode's
	// Cache.Split which may be incomplete for non-splittable caches
	// like RecurrentCache).
	if user {
		frontier.user = true
	}
	s.attachSnapshots(frontier, cacheOffset)
}

// advancePath advances the active path from the current frontier by matching
// tokens against existing trie children, splitting partial matches, and
// appending any remaining tokens as new nodes. Returns the new frontier.
func (c *kvCache) advancePath(frontier *trieNode, tokens []int32, endOffset int) *trieNode {
	// Check if existing children already cover some or all of tokens.
	// tokens may span multiple trie nodes when extending a previous run's
	// leaf and this snapshot now overlaps that same range.
	matchPath, matched := findBestMatch(frontier, tokens)
	// matchPath[0] is frontier itself; the rest are newly traversed nodes.
	remaining := tokens[matched:]

	// Check for a partial match within the last node's edge — if so, split it.
	if len(matchPath) > 1 {
		lastNode := matchPath[len(matchPath)-1]
		matchedInEdge := frontier.endOffset + matched - lastNode.startOffset()
		if matchedInEdge > 0 && matchedInEdge < len(lastNode.tokens) {
			matchPath[len(matchPath)-1] = splitNode(lastNode, matchedInEdge, c.caches, &c.pagedOutBytes)
		}
	}

	// Append traversed nodes (excluding frontier) to the active path.
	c.activePath = append(c.activePath, matchPath[1:]...)
	dest := matchPath[len(matchPath)-1]

	if len(remaining) > 0 {
		// Drop non-user snapshots so appendTokens can extend in-place
		// rather than creating a new child node.
		if len(dest.children) == 0 && !dest.user {
			dest.setSnapshots(nil, &c.pagedOutBytes)
		}
		newDest := dest.appendTokens(c.root, remaining, endOffset)
		if newDest != dest {
			c.activePath = append(c.activePath, newDest)
		}
		dest = newDest
	}
	return dest
}

// attachSnapshots attaches cache snapshots to a trie node at the given offset.
// The node must be on the active path (and thus protected from eviction;
// lastUsed is updated in close()). All non-nil caches must be at the same
// offset (cacheOffset); a mismatch indicates a bug in the caller.
func (s *cacheSession) attachSnapshots(node *trieNode, cacheOffset int) {
	c := s.cache

	if c.activePath[len(c.activePath)-1] != node {
		slog.Warn("attachSnapshots skipped: node is not the active frontier", "nodeEndOffset", node.endOffset)
		return
	}

	snaps := make([]cache.Snapshot, len(c.caches))
	for i, kv := range c.caches {
		if kv != nil {
			if kv.Offset() != cacheOffset {
				panic(fmt.Sprintf("attachSnapshots: cache offset mismatch layer %d: expected %d, got %d", i, cacheOffset, kv.Offset()))
			}
			snaps[i] = kv.Snapshot(node.startOffset())
		}
	}
	node.setSnapshots(snaps, &c.pagedOutBytes)
	c.emitEvent(EventSnapshot, node, node.snapshotBytes(), "")
	node.lastUsed = time.Now()
	slog.Debug("created snapshot", "offset", cacheOffset)
	c.enforceEvictionPolicy()
}

// freeAll releases all cache layers.
func (c *kvCache) freeAll() {
	for _, kv := range c.caches {
		if kv != nil {
			kv.Free()
		}
	}
}

func (c *kvCache) minCacheOffset() int {
	offset := 0
	found := false
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		if off := kv.Offset(); !found || off < offset {
			offset = off
			found = true
		}
	}
	return offset
}

// close saves the token state if the forward pass ran.
func (s *cacheSession) close() {
	offset := s.cache.minCacheOffset()
	if offset <= 0 {
		return
	}

	arrays := make([]*mlx.Array, 0, 2*len(s.caches))
	for _, kv := range s.caches {
		if kv == nil {
			continue
		}
		arrays = append(arrays, kv.State()...)
	}

	// Ensure that if we have run the forward pass and set the metadata
	// that we also actually have the data.
	mlx.AsyncEval(arrays...)

	// Advance the trie frontier with any newly generated tokens.
	c := s.cache
	if len(c.activePath) > 0 {
		frontier := c.activePath[len(c.activePath)-1]
		stored := append(s.inputs, s.outputs...)

		if offset > frontier.endOffset {
			newTokens := stored[frontier.endOffset:offset]
			c.advancePath(frontier, newTokens, offset)
		}
		c.activePath[len(c.activePath)-1].lastUsed = time.Now()
	}

	// Skip the full trie walk when no SSE subscribers are connected —
	// the common case during normal inference.
	if c.events != nil && c.events.count.Load() > 0 {
		c.rebuildSnapshot()
	}
}

// enforceEvictionPolicy evicts eligible nodes until paged-out memory is within limits.
func (c *kvCache) enforceEvictionPolicy() {
	if c.pagedOutBytes <= maxPagedOutBytes {
		return
	}

	activeSet := c.activeSet()

	// Collect all candidates in one walk and sort once (oldest, deepest,
	// largest) instead of re-walking per eviction.
	var candidates []*trieNode
	walkNodes(c.root, func(n *trieNode) bool {
		if n == c.root || activeSet[n] || len(n.children) > 1 {
			return true
		}
		if !n.hasSnapshots() {
			return true
		}
		candidates = append(candidates, n)
		return true
	})

	slices.SortFunc(candidates, func(a, b *trieNode) int {
		return cmp.Or(
			a.lastUsed.Compare(b.lastUsed),
			cmp.Compare(b.endOffset, a.endOffset),
			cmp.Compare(b.snapshotBytes(), a.snapshotBytes()),
		)
	})

	for _, node := range candidates {
		if c.pagedOutBytes <= maxPagedOutBytes {
			break
		}
		// Re-validate: evictNode may have structurally changed the trie
		// (mergeWithChild can alter child counts of surrounding nodes).
		if node.parent == nil || activeSet[node] || !node.hasSnapshots() || len(node.children) > 1 {
			continue
		}
		c.evictNode(node)
	}

	c.enforceDiskEvictionPolicy()
}

// evictNode evicts a single node from the trie, freeing its snapshot memory.
// Leaf nodes are saved to disk first so they can be reloaded later; if the
// disk write fails, the node is removed entirely (destructive fallback).
func (c *kvCache) evictNode(node *trieNode) {
	if len(node.children) == 0 {
		// Leaf: save to disk, keep node in trie.
		slog.Debug("evictNode: attempting disk eviction for leaf", "offset", node.startOffset(),
			"tokens", len(node.tokens), "bytes", node.snapshotBytes())
		if err := c.evictNodeToDisk(node); err != nil {
			slog.Warn("failed to evict to disk, falling back to delete",
				"error", err, "offset", node.startOffset())
			removeNode(node, &c.pagedOutBytes)
		}
	} else if len(node.children) == 1 {
		// Interior node with one child: merge with child.
		before := c.pagedOutBytes
		tokens := len(node.tokens)
		mergeWithChild(node, c.caches, &c.pagedOutBytes)
		slog.Debug("evicting interior node", "offset", node.startOffset(), "tokens", tokens, "freed", mlx.PrettyBytes(int(before-c.pagedOutBytes)))
	} else {
		panic("evictNode called on multi-child branch point")
	}
}

func (c *kvCache) dumpTree() {
	snap := c.buildTrieSnapshot()
	for _, line := range strings.Split(renderTrieText(snap), "\n") {
		if line != "" {
			logutil.Trace(line)
		}
	}
}
