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
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

const maxPagedOutBytes int64 = 8 << 30 // 8 GiB eviction threshold for paged-out snapshot memory

type kvCache struct {
	root          *trieNode   // root of the prefix trie
	activePath    []*trieNode // current root→leaf path with live MLX arrays
	caches        []cache.Cache
	pagedOutBytes int64 // total bytes in paged-out snapshots across the trie

	// Persistence fields. See cache_persist.go.
	diskBytes   int64       // atomic; sum of diskSize across all nodes
	diskMax     int64       // -1 or <= 0 disables disk eviction; positive = byte cap
	modelDigest string      // identifies the model for cacheDir scoping
	cacheDir    string      // <OLLAMA_KV_CACHE_ROOT>/<modelDigest>; empty when feature disabled
	writer      *diskWriter // nil when feature disabled
}

// newKvCache is the construction entry point for kvCache.
// When OLLAMA_KV_CACHE_DISK_MAX == 0, returns a cache with no writer —
// byte-for-byte identical to upstream-main behavior. modelDigest may be
// empty; that's treated as "feature off" since cache files would be
// indistinguishable from other models.
func newKvCache(modelDigest string, numLayers int) *kvCache {
	// Pre-allocate caches with nil entries so len(c.caches) reports the
	// layer count during rehydrate (before ensureCaches has seen the model).
	// ensureCaches treats a slice of nil entries as still-uninitialized.
	c := &kvCache{
		caches: make([]cache.Cache, numLayers),
	}
	c.ensureRoot()

	diskMax := envconfig.KVCacheDiskMax()
	if diskMax == 0 {
		slog.Info("kv cache disk persistence disabled (OLLAMA_KV_CACHE_DISK_MAX=0)")
		return c
	}
	if modelDigest == "" {
		slog.Info("kv cache disk persistence disabled (no model digest)")
		return c
	}
	c.modelDigest = modelDigest
	c.cacheDir = filepath.Join(envconfig.KVCacheRoot(), modelDigest)
	c.diskMax = diskMax
	if err := c.rehydrate(); err != nil {
		slog.Warn("kv cache rehydrate failed, starting cold", "err", err)
	}
	c.writer = newDiskWriter(c)
	slog.Info("kv cache disk persistence enabled",
		"cache_dir", c.cacheDir,
		"model_digest", modelDigest,
		"num_layers", numLayers,
		"disk_max_bytes", diskMax)
	return c
}

// shutdownBudget caps how long shutdown will spend flushing dirty Warm nodes.
// 15s matches the old writer-drain budget; at ~25ms per write this covers a
// few hundred nodes worth of flush. Anything beyond that is dropped (oldest
// first) so the process can exit promptly.
const shutdownBudget = 15 * time.Second

// shutdown persists every Warm node that doesn't yet have a disk file, then
// releases the writer. Safe to call when persistence is disabled (writer nil).
func (c *kvCache) shutdown() {
	c.shutdownWithBudget(shutdownBudget)
}

// shutdownWithBudget is the test-parameterizable form of shutdown.
// Ordering: newest-first by lastUsed — a budget-constrained shutdown keeps
// the most recently used data. Dropped nodes are left Warm in memory and
// vanish with the process.
func (c *kvCache) shutdownWithBudget(budget time.Duration) {
	if c == nil || c.writer == nil {
		return
	}

	// Parents must flush before children: a child's disk file references its
	// parent's hash, so dropping the parent mid-budget orphans the child on
	// next rehydrate. Sort by depth ascending first, then newest-first within
	// a depth band so budget pressure still drops the oldest siblings.
	//
	// Track depth during the walk so the comparator doesn't re-walk the parent
	// chain on every comparison (SortFunc is O(N log N) calls × O(depth) walk
	// = O(N² log N) worst case for deep tries).
	type dirtyNode struct {
		n *trieNode
		d int
	}
	var dirty []dirtyNode
	walkDepth(c.root, 0, func(n *trieNode, d int) bool {
		if n != c.root && n.snapshots != nil && n.diskPath == "" {
			dirty = append(dirty, dirtyNode{n, d})
		}
		return true
	})
	slices.SortFunc(dirty, func(a, b dirtyNode) int {
		if d := a.d - b.d; d != 0 {
			return d
		}
		return b.n.lastUsed.Compare(a.n.lastUsed)
	})

	slog.Info("kv cache shutdown: flushing warm nodes",
		"count", len(dirty), "total_disk_bytes", atomic.LoadInt64(&c.diskBytes))

	deadline := time.Now().Add(budget)
	flushed, dropped := 0, 0
	for _, d := range dirty {
		if time.Now().After(deadline) {
			dropped = len(dirty) - flushed
			break
		}
		c.scheduleWrite(d.n)
		if d.n.diskPath != "" {
			flushed++
		}
	}
	if dropped > 0 {
		slog.Warn("kv cache shutdown: budget exceeded, dropping oldest nodes",
			"flushed", flushed, "dropped", dropped, "budget", budget)
	} else {
		slog.Info("kv cache shutdown: flush complete", "flushed", flushed)
	}
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
	// Sentinel: newKvCache pre-allocates the slice with nil entries so
	// rehydrate has the right layer count. A non-nil first entry means
	// ensureCaches already ran this session.
	if len(c.caches) > 0 && c.caches[0] != nil {
		return
	}
	if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
		c.caches = cacheFactory.NewCaches()
		return
	}
	if len(c.caches) == 0 {
		c.caches = make([]cache.Cache, m.NumLayers())
	}
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

	// Always keep at least one token to re-evaluate so the pipeline can
	// seed token generation from it. Apply the holdback BEFORE restoring
	// Cold nodes — otherwise we'd load disk files for tail nodes about to
	// be trimmed off, wasting I/O and MLX memory on every cache-hit request.
	if matched == len(inputs) && matched > 0 {
		matchPath, matched = findBestMatch(c.root, inputs[:len(inputs)-1])
	}

	// Restore any Cold nodes on the trimmed match path before paging in.
	// Failure unlinks the bad file (F2) and the node becomes Gone; recompute
	// the matched length so switchToPath only pages in what's actually live.
	if c.writer != nil {
		slog.Debug("kv cache begin: walking match path",
			"path_len", len(matchPath), "matched", matched, "input_len", len(inputs))
		_ = c.restoreMatchedPath(matchPath, matched)
		postRestore := matchedAfterRestore(matchPath, matched)
		if postRestore < matched {
			slog.Debug("kv cache restore shrunk match (Gone node on path)",
				"from", matched, "to", postRestore)
			matched = postRestore
		}
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
pageIn:
	for _, node := range newPath {
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

	if pageOutCount > 0 || pageInCount > 0 {
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
			// A Cold node (snapshots dropped but disk file kept) cannot be
			// split directly: splitNode only splits via Cache.Split when the
			// source has in-memory snapshots. Without this load, the new
			// parent is born with snapshots=nil and diskPath=="" (Gone), and
			// the original's old disk file — which covers the full pre-split
			// edge — becomes silently stale for the sliced suffix.
			//
			// Load the Cold node back into memory so Cache.Split has something
			// to work with. On load failure, skip the split entirely by
			// rolling the match back to the boundary BEFORE lastNode; the
			// remaining tokens then fall through to appendTokens, which
			// creates a sibling as the divergence point. No Gone intermediate
			// is produced.
			if c.writer != nil && lastNode.snapshots == nil && lastNode.diskPath != "" {
				if err := c.loadFromDisk(lastNode); err != nil {
					slog.Warn("kv cache: load-before-split failed, skipping split",
						"path", lastNode.diskPath, "err", err)
					atomic.AddInt64(&c.diskBytes, -lastNode.diskSize)
					lastNode.diskPath = ""
					lastNode.diskSize = 0
					matched -= matchedInEdge
					matchPath = matchPath[:len(matchPath)-1]
					remaining = tokens[matched:]
				} else {
					c.splitAndPersist(matchPath, lastNode, matchedInEdge)
				}
			} else {
				c.splitAndPersist(matchPath, lastNode, matchedInEdge)
			}
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

// splitAndPersist runs splitNode on lastNode and persists both halves.
// After the split, lastNode owns only the suffix tokens, so any existing
// disk file attached to it is stale — the file covers the full pre-split
// edge. Remove that file and clear the node's disk pointers so the next
// scheduleWrite call produces a fresh file whose header tokens match.
func (c *kvCache) splitAndPersist(matchPath []*trieNode, lastNode *trieNode, matchedInEdge int) {
	if lastNode.diskPath != "" {
		if err := os.Remove(lastNode.diskPath); err != nil && !os.IsNotExist(err) {
			slog.Warn("kv cache: remove stale pre-split file",
				"path", lastNode.diskPath, "err", err)
		}
		atomic.AddInt64(&c.diskBytes, -lastNode.diskSize)
		lastNode.diskPath = ""
		lastNode.diskSize = 0
	}
	intermediate := splitNode(lastNode, matchedInEdge, c.caches, &c.pagedOutBytes)
	matchPath[len(matchPath)-1] = intermediate
	// Persist both halves; their token ranges are now correct for their
	// respective new files.
	c.scheduleWrite(intermediate)
	c.scheduleWrite(lastNode)
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
	node.lastUsed = time.Now()
	// Writes are deferred to eviction and shutdown — see kvCache.shutdown.
	// Keeping prefill-boundary snapshots off disk avoids ~15× serialize+fsync
	// per long-prefill request and saves space on nodes that get extended
	// in-place by the next turn before ever being evicted.
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
}

// enforceEvictionPolicy runs the two-tier eviction:
//   - Memory pass: demote Warm nodes to Cold (drop snapshots, keep disk file).
//   - Disk pass: delete files (and trie nodes) once diskBytes > diskMax.
//
// When persistence is disabled (writer == nil) the memory pass falls back to
// the legacy delete behavior — byte-for-byte identical to upstream main.
func (c *kvCache) enforceEvictionPolicy() {
	c.enforceMemoryPolicy()
	c.enforceDiskPolicy()
}

// collectEvictionCandidates walks the trie once, collecting nodes that pass
// the supplied filter (skipping root, activePath members, and multi-child
// branch points) sorted by eviction priority: oldest lastUsed first, then
// deepest endOffset, then largest snapshotBytes. snapshotBytes is computed
// once per node and stored alongside so the comparator is O(1).
//
// Earlier designs rescanned the trie for every eviction, producing O(N²)
// sweeps under sustained pressure. Callers now batch the scan and iterate
// the result, re-verifying each candidate is still eligible since
// evictNode/mergeWithChild can orphan or reshape nodes between iterations.
func (c *kvCache) collectEvictionCandidates(filter func(*trieNode) bool, activeSet map[*trieNode]bool) []*trieNode {
	type scored struct {
		n     *trieNode
		bytes int64
	}
	var out []scored
	walkNodes(c.root, func(n *trieNode) bool {
		if n == c.root || activeSet[n] || len(n.children) > 1 {
			return true
		}
		if !filter(n) {
			return true
		}
		out = append(out, scored{n: n, bytes: n.snapshotBytes()})
		return true
	})
	slices.SortFunc(out, func(a, b scored) int {
		return cmp.Or(
			a.n.lastUsed.Compare(b.n.lastUsed),
			cmp.Compare(b.n.endOffset, a.n.endOffset),
			cmp.Compare(b.bytes, a.bytes),
		)
	})
	nodes := make([]*trieNode, len(out))
	for i, s := range out {
		nodes[i] = s.n
	}
	return nodes
}

// selectEvictionCandidate returns the single highest-priority candidate
// matching the filter, or nil if none exists. Thin wrapper over
// collectEvictionCandidates for callers that want a one-shot selector.
func (c *kvCache) selectEvictionCandidate(filter func(*trieNode) bool) *trieNode {
	activeSet := buildActiveSet(c.activePath)
	cands := c.collectEvictionCandidates(filter, activeSet)
	if len(cands) == 0 {
		return nil
	}
	return cands[0]
}

func buildActiveSet(path []*trieNode) map[*trieNode]bool {
	s := make(map[*trieNode]bool, len(path))
	for _, n := range path {
		s[n] = true
	}
	return s
}

// stillEligible re-checks a candidate pulled from a cached sort. Between
// the scan and the action, evictNode or mergeWithChild may have orphaned
// the node (parent set to nil) or pulled its sibling up so it now has
// >1 children. activeSet is stable across a single eviction pass.
func stillEligible(n *trieNode, root *trieNode, activeSet map[*trieNode]bool, filter func(*trieNode) bool) bool {
	if n == root || n.parent == nil {
		return false
	}
	if activeSet[n] || len(n.children) > 1 {
		return false
	}
	return filter(n)
}

func (c *kvCache) enforceMemoryPolicy() {
	if c.pagedOutBytes <= maxPagedOutBytes {
		return
	}
	startBytes := c.pagedOutBytes
	demoted, deleted := 0, 0
	activeSet := buildActiveSet(c.activePath)
	filter := func(n *trieNode) bool { return n.snapshots != nil }

	for c.pagedOutBytes > maxPagedOutBytes {
		cands := c.collectEvictionCandidates(filter, activeSet)
		if len(cands) == 0 {
			break
		}
		progress := false
		for _, cand := range cands {
			if c.pagedOutBytes <= maxPagedOutBytes {
				break
			}
			if !stillEligible(cand, c.root, activeSet, filter) {
				continue
			}
			if cand.diskPath == "" {
				if c.writer != nil {
					c.scheduleWrite(cand)
					// scheduleWrite may transition this node to Cold;
					// re-scan to pick up the new landscape.
					return
				}
				c.evictNode(cand)
				deleted++
				progress = true
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
			demoted++
			progress = true
		}
		if !progress {
			break
		}
	}
	if demoted > 0 || deleted > 0 {
		slog.Debug("kv cache memory pass complete",
			"demoted_to_cold", demoted,
			"deleted", deleted,
			"start_bytes", startBytes,
			"end_bytes", c.pagedOutBytes)
	}
}

func (c *kvCache) enforceDiskPolicy() {
	if c.writer == nil || c.diskMax <= 0 {
		return
	}
	startBytes := atomic.LoadInt64(&c.diskBytes)
	if startBytes <= c.diskMax {
		return
	}
	activeSet := buildActiveSet(c.activePath)
	filter := func(n *trieNode) bool { return n.diskPath != "" }
	evicted := 0

	for atomic.LoadInt64(&c.diskBytes) > c.diskMax {
		cands := c.collectEvictionCandidates(filter, activeSet)
		if len(cands) == 0 {
			break
		}
		progress := false
		for _, cand := range cands {
			if atomic.LoadInt64(&c.diskBytes) <= c.diskMax {
				break
			}
			if !stillEligible(cand, c.root, activeSet, filter) {
				continue
			}
			path := cand.diskPath
			size := cand.diskSize
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				slog.Warn("kv cache disk evict: remove failed", "path", path, "err", err)
			}
			atomic.AddInt64(&c.diskBytes, -size)
			cand.diskPath = ""
			cand.diskSize = 0
			// evictNode handles both leaves and single-child interior nodes;
			// multi-child nodes were filtered out by the selector.
			c.evictNode(cand)
			evicted++
			progress = true
		}
		if !progress {
			break
		}
	}
	if evicted > 0 {
		slog.Debug("kv cache disk pass complete",
			"evicted", evicted,
			"start_bytes", startBytes,
			"end_bytes", atomic.LoadInt64(&c.diskBytes),
			"cap", c.diskMax)
	}
}

// restoreMatchedPath walks `path` from root side and restores any Cold nodes
// (snapshots == nil but diskPath set) to Warm before the caller hands off to
// switchToPath. A Gone node (diskPath empty, snapshots nil) or a load failure
// stops the walk; the caller must handle a partial path by re-prefilling
// the tail. Always returns nil today (errors are logged + degraded), but the
// signature returns an error to leave room for future fail-loud modes.
func (c *kvCache) restoreMatchedPath(path []*trieNode, matched int) error {
	covered := 0
	for _, node := range path {
		if node == c.root {
			continue
		}
		if covered >= matched {
			// Remaining nodes contribute nothing to this request's matched
			// prefix — skip the disk I/O and MLX upload.
			break
		}
		if node.snapshots != nil {
			covered += len(node.tokens)
			continue // Warm or just-restored
		}
		if node.diskPath == "" {
			slog.Debug("kv cache restore stopped at Gone node",
				"start_offset", node.startOffset(), "tokens", len(node.tokens))
			break
		}
		if err := c.loadFromDisk(node); err != nil {
			oldPath, oldSize := node.diskPath, node.diskSize
			slog.Warn("kv cache restore failed, deleting bad file",
				"path", oldPath, "err", err)
			_ = os.Remove(oldPath)
			atomic.AddInt64(&c.diskBytes, -oldSize)
			node.diskPath = ""
			node.diskSize = 0
			break
		}
		covered += len(node.tokens)
	}
	return nil
}

// matchedAfterRestore returns how many of the `matched` input tokens are
// actually usable after the partial-restore walk. A Gone node (snapshots nil
// and diskPath empty) caps the usable length at the cumulative-token count
// up to that point. Restore cannot *increase* the match count — findBestMatch
// already determined the true prefix length against the input, so we never
// return more than `matched`.
func matchedAfterRestore(path []*trieNode, matched int) int {
	covered := 0
	for i, node := range path {
		if i == 0 && node.parent == nil {
			continue
		}
		if node.snapshots == nil && node.diskPath == "" {
			return min(covered, matched)
		}
		covered += len(node.tokens)
	}
	return min(covered, matched)
}

// scheduleWrite writes node to disk synchronously (on the caller's
// goroutine) if persistence is enabled and the node isn't already written.
// Called from the pipeline goroutine only. MLX thread-safety: all MLX
// work here runs on the same goroutine that performs inference, so the
// Metal command stream stays single-owner.
//
// The 7-ish-millisecond stall per snapshot is negligible relative to
// prefill time (snapshots land every ~8192 tokens, at GPU-throughput
// speeds). See the diskWriter type comment for why this isn't async.
func (c *kvCache) scheduleWrite(node *trieNode) {
	if c.writer == nil {
		return // feature disabled
	}
	if len(node.snapshots) == 0 {
		// Nothing to serialize. This can happen briefly when a Cold node is
		// split mid-eviction; the caller repopulates snapshots before retrying.
		return
	}
	if node.diskPath != "" {
		return
	}
	c.writer.writeNode(c, node)
}

// evictNode evicts a single node from the trie, freeing its snapshot memory.
func (c *kvCache) evictNode(node *trieNode) {
	if len(node.children) == 0 {
		removeNode(node, &c.pagedOutBytes)
	} else if len(node.children) == 1 {
		mergeWithChild(node, c.caches, &c.pagedOutBytes)
	} else {
		panic("evictNode called on multi-child branch point")
	}
}

func (c *kvCache) dumpTree() {
	// Summary stats
	var cacheBytes int
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		for _, a := range kv.State() {
			if a != nil {
				cacheBytes += a.NumBytes()
			}
		}
	}

	// Build active path set for marking.
	active := make(map[*trieNode]bool, len(c.activePath))
	for _, n := range c.activePath {
		active[n] = true
	}

	var nodeCount, snapshotCount int
	var pagedBytes int64
	var lines []string
	var dump func(n *trieNode, prefix string, isLast bool)
	dump = func(n *trieNode, prefix string, isLast bool) {
		if n == nil {
			return
		}
		nodeCount++

		// Build connector
		var connector string
		if n.parent == nil {
			connector = ""
		} else if isLast {
			connector = prefix + "`-- "
		} else {
			connector = prefix + "|-- "
		}

		// Node label
		nodeBytes := n.snapshotBytes()
		pagedBytes += nodeBytes

		label := fmt.Sprintf("[%d,%d) %dt", n.startOffset(), n.endOffset, len(n.tokens))
		if nodeBytes > 0 {
			label += " " + mlx.PrettyBytes(int(nodeBytes)).String()
		}
		if !n.lastUsed.IsZero() {
			label += fmt.Sprintf(" %s ago", time.Since(n.lastUsed).Truncate(time.Millisecond))
		}
		var flags []string
		if n.user {
			flags = append(flags, "user")
		}
		if n.hasAllSnapshots() {
			snapshotCount++
			flags = append(flags, "snap")
		}
		if active[n] {
			flags = append(flags, "active")
		}
		if len(flags) > 0 {
			label += " (" + flags[0]
			for _, f := range flags[1:] {
				label += ", " + f
			}
			label += ")"
		}
		lines = append(lines, connector+label)

		// Recurse children
		childPrefix := prefix
		if n.parent != nil {
			if isLast {
				childPrefix += "    "
			} else {
				childPrefix += "|   "
			}
		}
		for i, child := range n.children {
			dump(child, childPrefix, i == len(n.children)-1)
		}
	}
	dump(c.root, "", true)

	offset := c.minCacheOffset()
	logutil.Trace(fmt.Sprintf("kv cache active_tokens: %d, active_size: %s, paged_out: %s, trie: nodes=%d, snapshots=%d",
		offset, mlx.PrettyBytes(cacheBytes), mlx.PrettyBytes(int(pagedBytes)), nodeCount, snapshotCount))
	for i, l := range lines {
		if i == 0 {
			logutil.Trace("cache trie: " + l)
		} else {
			logutil.Trace("  " + l)
		}
	}
}
