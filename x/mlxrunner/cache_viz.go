package mlxrunner

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/internal/httputil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// ---------------- JSON response types ----------------

// TrieSnapshot is a point-in-time capture of the full trie state.
// Built by the trie goroutine, read lock-free by HTTP handlers via atomic.Pointer.
type TrieSnapshot struct {
	Timestamp string         `json:"timestamp"`
	Stats     TrieStats      `json:"stats"`
	Nodes     []TrieNodeInfo `json:"nodes"`
}

type TrieStats struct {
	ActiveTokens     int   `json:"active_tokens"`
	ActiveCacheBytes int64 `json:"active_cache_bytes"`
	PagedOutBytes    int64 `json:"paged_out_bytes"`
	TotalDiskBytes   int64 `json:"total_disk_bytes"`
	NodeCount        int   `json:"node_count"`
	SnapshotCount    int   `json:"snapshot_count"`
	DiskNodeCount    int   `json:"disk_node_count"`
}

type TrieNodeInfo struct {
	ID            int      `json:"id"`
	OffsetRange   [2]int   `json:"offset_range"`
	TokenCount    int      `json:"token_count"`
	Children      []int    `json:"children,omitempty"`
	Flags         []string `json:"flags,omitempty"`
	SnapshotBytes int64    `json:"snapshot_bytes,omitempty"`
	DiskFile      string   `json:"disk_file,omitempty"`
	DiskFileSize  int64    `json:"disk_file_size,omitempty"`
	LastUsed      string   `json:"last_used"`
	Tokens        []int32  `json:"tokens,omitempty"`
}

// ---------------- Event types ----------------

type CacheEventType string

const (
	EventPageOut       CacheEventType = "page_out"
	EventPageIn        CacheEventType = "page_in"
	EventPageInDisk    CacheEventType = "page_in_disk"
	EventEvictToDisk   CacheEventType = "evict_to_disk"
	EventEvictFromDisk CacheEventType = "evict_from_disk"
	EventSnapshot      CacheEventType = "snapshot"
)

type CacheEvent struct {
	Type        CacheEventType `json:"type"`
	Timestamp   string         `json:"timestamp"`
	OffsetRange [2]int         `json:"offset_range"`
	TokenCount  int            `json:"token_count"`
	Bytes       int64          `json:"bytes"`
	DiskFile    string         `json:"disk_file,omitempty"`
}

// ---------------- Event bus ----------------

// cacheEventBus fans out CacheEvents to SSE subscribers. The trie goroutine
// is the sole publisher; HTTP handler goroutines are subscribers. Non-blocking
// sends ensure the inference hot path never blocks on a slow client.
type cacheEventBus struct {
	mu          sync.Mutex
	subscribers map[chan CacheEvent]struct{}
	count       atomic.Int32 // fast path: skip mutex when 0 subscribers
}

func newCacheEventBus() *cacheEventBus {
	return &cacheEventBus{
		subscribers: make(map[chan CacheEvent]struct{}),
	}
}

func (b *cacheEventBus) subscribe() (<-chan CacheEvent, func()) {
	ch := make(chan CacheEvent, 64)
	b.mu.Lock()
	b.subscribers[ch] = struct{}{}
	b.count.Add(1)
	b.mu.Unlock()
	unsub := func() {
		b.mu.Lock()
		delete(b.subscribers, ch)
		b.count.Add(-1)
		close(ch)
		b.mu.Unlock()
	}
	return ch, unsub
}

func (b *cacheEventBus) emit(e CacheEvent) {
	if b == nil || b.count.Load() == 0 {
		return
	}
	b.mu.Lock()
	for ch := range b.subscribers {
		select {
		case ch <- e:
		default: // drop if subscriber is slow
		}
	}
	b.mu.Unlock()
}

func (c *kvCache) emitEvent(typ CacheEventType, node *trieNode, bytes int64, diskFile string) {
	if c.events == nil || c.events.count.Load() == 0 {
		return
	}
	c.events.emit(CacheEvent{
		Type:        typ,
		Timestamp:   time.Now().UTC().Format(time.RFC3339Nano),
		OffsetRange: [2]int{node.startOffset(), node.endOffset},
		TokenCount:  len(node.tokens),
		Bytes:       bytes,
		DiskFile:    diskFile,
	})
}

// Called from the trie goroutine after mutations.
func (c *kvCache) rebuildSnapshot() {
	if c.events == nil {
		return
	}
	c.trieSnapshot.Store(c.buildTrieSnapshot())
}

// ---------------- Snapshot builder ----------------

// buildTrieSnapshot walks the trie and produces a structured snapshot.
func (c *kvCache) buildTrieSnapshot() *TrieSnapshot {
	// Compute active cache bytes from live arrays.
	var cacheBytes int64
	for _, kv := range c.caches {
		if kv == nil {
			continue
		}
		for _, a := range kv.State() {
			if a != nil {
				cacheBytes += int64(a.NumBytes())
			}
		}
	}

	activeSet := c.activeSet()

	ordered, nodeMap := indexNodes(c.root)

	var snapshotCount, diskNodeCount int
	nodes := make([]TrieNodeInfo, len(ordered))
	for i, n := range ordered {
		info := TrieNodeInfo{
			ID:            i,
			OffsetRange:   [2]int{n.startOffset(), n.endOffset},
			TokenCount:    len(n.tokens),
			SnapshotBytes: n.snapshotBytes(),
			Tokens:        n.tokens,
		}

		if !n.lastUsed.IsZero() {
			info.LastUsed = n.lastUsed.UTC().Format(time.RFC3339)
		}

		for _, child := range n.children {
			info.Children = append(info.Children, nodeMap[child])
		}

		if n.user {
			info.Flags = append(info.Flags, "user")
		}
		if n.hasAllSnapshots() {
			info.Flags = append(info.Flags, "snap")
			snapshotCount++
		}
		if activeSet[n] {
			info.Flags = append(info.Flags, "active")
		}
		if n.diskFile != "" {
			info.Flags = append(info.Flags, "disk")
			info.DiskFile = filepath.Base(n.diskFile)
			info.DiskFileSize = n.diskFileSize
			diskNodeCount++
		}

		nodes[i] = info
	}

	return &TrieSnapshot{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Stats: TrieStats{
			ActiveTokens:     c.minCacheOffset(),
			ActiveCacheBytes: cacheBytes,
			PagedOutBytes:    c.pagedOutBytes,
			TotalDiskBytes:   c.totalDiskBytes,
			NodeCount:        len(ordered),
			SnapshotCount:    snapshotCount,
			DiskNodeCount:    diskNodeCount,
		},
		Nodes: nodes,
	}
}

// ---------------- Text renderer ----------------

// renderTrieText produces an ASCII tree from a TrieSnapshot, matching the
// format used by dumpTree() for the ?format=text endpoint.
func renderTrieText(snap *TrieSnapshot) string {
	if snap == nil || len(snap.Nodes) == 0 {
		return "(empty trie)\n"
	}

	var b strings.Builder

	// Header line matching dumpTree format.
	fmt.Fprintf(&b, "kv cache active_tokens: %d, active_size: %s, paged_out: %s, trie: nodes=%d, snapshots=%d\n",
		snap.Stats.ActiveTokens,
		mlx.PrettyBytes(int(snap.Stats.ActiveCacheBytes)),
		mlx.PrettyBytes(int(snap.Stats.PagedOutBytes)),
		snap.Stats.NodeCount,
		snap.Stats.SnapshotCount,
	)

	var dump func(id int, prefix string, isLast bool)
	dump = func(id int, prefix string, isLast bool) {
		n := snap.Nodes[id]

		var connector string
		if id == 0 {
			connector = ""
		} else if isLast {
			connector = prefix + "`-- "
		} else {
			connector = prefix + "|-- "
		}

		label := fmt.Sprintf("[%d,%d) %dt", n.OffsetRange[0], n.OffsetRange[1], n.TokenCount)
		if n.SnapshotBytes > 0 {
			label += " " + mlx.PrettyBytes(int(n.SnapshotBytes)).String()
		}
		if len(n.Flags) > 0 {
			label += " (" + strings.Join(n.Flags, ", ") + ")"
		}

		if id == 0 {
			fmt.Fprintf(&b, "cache trie: %s\n", label)
		} else {
			fmt.Fprintf(&b, "  %s%s\n", connector, label)
		}

		childPrefix := prefix
		if id != 0 {
			if isLast {
				childPrefix += "    "
			} else {
				childPrefix += "|   "
			}
		}
		for i, childID := range n.Children {
			dump(childID, childPrefix, i == len(n.Children)-1)
		}
	}
	dump(0, "", true)

	return b.String()
}

// ---------------- HTTP handlers ----------------

func (c *kvCache) handleTrieSnapshot(w http.ResponseWriter, r *http.Request) {
	snap := c.trieSnapshot.Load()
	if snap == nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TrieSnapshot{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		})
		return
	}

	if r.URL.Query().Get("format") == "text" {
		w.Header().Set("Content-Type", "text/plain")
		io.WriteString(w, renderTrieText(snap))
		return
	}

	// Strip tokens by default (include only with ?tokens=true).
	if r.URL.Query().Get("tokens") != "true" {
		stripped := *snap
		stripped.Nodes = make([]TrieNodeInfo, len(snap.Nodes))
		for i, n := range snap.Nodes {
			n.Tokens = nil
			stripped.Nodes[i] = n
		}
		snap = &stripped
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(snap)
}

func (c *kvCache) handleCacheEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Send current snapshot as initial state. Uses "init" (not "snapshot")
	// because EventSnapshot emits CacheEvent payloads through the bus,
	// which have a different schema than the TrieSnapshot sent here.
	if snap := c.trieSnapshot.Load(); snap != nil {
		httputil.WriteSSE(w, "init", snap)
	}

	ch, unsub := c.events.subscribe()
	defer unsub()

	for {
		select {
		case <-r.Context().Done():
			return
		case event, ok := <-ch:
			if !ok {
				return
			}
			if err := httputil.WriteSSE(w, string(event.Type), event); err != nil {
				return
			}
		}
	}
}


// cacheTrieHTML is a self-contained dashboard that visualizes the trie state
// in real time. Uses only DOM-safe methods (createElement/textContent) for
// rendering dynamic content.
//
//go:embed cache_viz_dashboard.html
var cacheTrieHTML string
