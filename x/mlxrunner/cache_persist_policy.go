package mlxrunner

// Two-tier eviction + per-(node,seqID) persistence methods on kvCache.
// Split from the legacy single-seq cache.go after the batching merge —
// these hooks need a multi-seq redesign. For now they carry the minimum
// signatures tests and callers depend on; bodies are intentionally
// conservative so the merge lands green and the port is a targeted
// follow-up (see CONFLICT_RESOLUTION_PLAN.md group E).

// scheduleWrite queues a trie node's snapshots for disk persistence.
// Post-merge TODO: route through the scheduler observer so per-seqID
// snapshot files are materialized on seq completion.
func (c *kvCache) scheduleWrite(node *trieNode) {
	if c.writer == nil || c.cacheDir == "" || node == nil {
		return
	}
	c.writer.writeNode(c, node)
}

// restoreMatchedPath walks the active path's cold nodes and pages their
// snapshots back into live MLX arrays. Post-merge TODO: teach it the
// per-seq activePaths map and the lazy-on-admission hook point.
func (c *kvCache) restoreMatchedPath(path []*trieNode, matched int) error {
	for i, node := range path {
		if i >= matched {
			break
		}
		if node == nil || node.diskPath == "" || node.snapshots != nil {
			continue
		}
		if err := c.loadFromDisk(node); err != nil {
			return err
		}
	}
	return nil
}
