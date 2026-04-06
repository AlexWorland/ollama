package mlxrunner

import (
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

// ---------------- Embedded HTML dashboard ----------------

// cacheTrieHTML is a self-contained dashboard that visualizes the trie state
// in real time. Uses only DOM-safe methods (createElement/textContent) for
// rendering dynamic content.
const cacheTrieHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KV Cache Trie</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Menlo', 'Monaco', monospace; font-size: 13px; background: #0d1117; color: #c9d1d9; padding: 16px; }
  h1 { font-size: 16px; color: #58a6ff; margin-bottom: 12px; }
  .stats { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; padding: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; }
  .stat { display: flex; flex-direction: column; }
  .stat-label { color: #8b949e; font-size: 11px; text-transform: uppercase; }
  .stat-value { color: #f0f6fc; font-size: 15px; font-weight: 600; }
  .trie { padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; overflow-x: auto; }
  .node { padding: 2px 0; white-space: nowrap; transition: background 0.3s; }
  .node.flash-pageout { background: rgba(248, 81, 73, 0.2); }
  .node.flash-pagein { background: rgba(63, 185, 80, 0.2); }
  .node.flash-disk { background: rgba(136, 98, 235, 0.2); }
  .node.flash-snap { background: rgba(88, 166, 255, 0.2); }
  .flag { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 11px; margin-left: 4px; }
  .flag-active { background: #1f6feb33; color: #58a6ff; }
  .flag-snap { background: #23863633; color: #3fb950; }
  .flag-disk { background: #8957e533; color: #bc8cff; }
  .flag-user { background: #d2992233; color: #d29922; }
  .bytes { color: #8b949e; }
  .connector { color: #484f58; }
  .events-panel { margin-top: 16px; padding: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; max-height: 200px; overflow-y: auto; }
  .events-panel h2 { font-size: 13px; color: #8b949e; margin-bottom: 8px; }
  .event { padding: 2px 0; font-size: 12px; border-bottom: 1px solid #21262d; animation: fadeIn 0.3s; }
  .event-type { font-weight: 600; display: inline-block; min-width: 110px; }
  .et-page_out { color: #f85149; }
  .et-page_in, .et-page_in_disk { color: #3fb950; }
  .et-evict_to_disk { color: #bc8cff; }
  .et-evict_from_disk { color: #f85149; }
  .et-snapshot { color: #58a6ff; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  .status { font-size: 11px; color: #3fb950; margin-bottom: 12px; }
  .status.disconnected { color: #f85149; }
</style>
</head>
<body>
<h1>KV Cache Trie Visualizer</h1>
<div class="status" id="status">Connecting...</div>
<div class="stats" id="stats"></div>
<div class="trie" id="trie"></div>
<div class="events-panel">
  <h2>Live Events</h2>
  <div id="event-log"></div>
</div>

<script>
const $ = function(id) { return document.getElementById(id); };

function fmtBytes(b) {
  if (b === 0) return '';
  if (b < 1024) return b + 'B';
  if (b < 1048576) return (b/1024).toFixed(1) + 'KiB';
  if (b < 1073741824) return (b/1048576).toFixed(1) + 'MiB';
  return (b/1073741824).toFixed(2) + 'GiB';
}

function clearChildren(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
}

function renderStats(s) {
  var container = $('stats');
  clearChildren(container);
  var items = [
    ['Active Tokens', s.active_tokens],
    ['Active Cache', fmtBytes(s.active_cache_bytes)],
    ['Paged Out', fmtBytes(s.paged_out_bytes)],
    ['Disk Used', fmtBytes(s.total_disk_bytes)],
    ['Nodes', s.node_count],
    ['Snapshots', s.snapshot_count],
    ['Disk Nodes', s.disk_node_count]
  ];
  items.forEach(function(pair) {
    var div = document.createElement('div');
    div.className = 'stat';
    var lbl = document.createElement('span');
    lbl.className = 'stat-label';
    lbl.textContent = pair[0];
    var val = document.createElement('span');
    val.className = 'stat-value';
    val.textContent = String(pair[1]);
    div.appendChild(lbl);
    div.appendChild(val);
    container.appendChild(div);
  });
}

function createNodeEl(n, connector) {
  var div = document.createElement('div');
  div.className = 'node';
  div.setAttribute('data-range', n.offset_range[0] + '-' + n.offset_range[1]);

  if (connector) {
    var connSpan = document.createElement('span');
    connSpan.className = 'connector';
    connSpan.textContent = connector;
    div.appendChild(connSpan);
  }

  var text = '[' + n.offset_range[0] + ',' + n.offset_range[1] + ') ' + n.token_count + 't';
  div.appendChild(document.createTextNode(text));

  if (n.snapshot_bytes > 0) {
    var bs = document.createElement('span');
    bs.className = 'bytes';
    bs.textContent = ' ' + fmtBytes(n.snapshot_bytes);
    div.appendChild(bs);
  }

  if (n.disk_file) {
    var df = document.createElement('span');
    df.className = 'bytes';
    df.textContent = ' [' + n.disk_file + ' ' + fmtBytes(n.disk_file_size) + ']';
    div.appendChild(df);
  }

  if (n.flags && n.flags.length > 0) {
    n.flags.forEach(function(f) {
      var flag = document.createElement('span');
      flag.className = 'flag flag-' + f;
      flag.textContent = f;
      div.appendChild(flag);
    });
  }

  return div;
}

function renderTrie(snap) {
  var container = $('trie');
  clearChildren(container);
  if (!snap || !snap.nodes || snap.nodes.length === 0) {
    container.textContent = '(empty trie)';
    return;
  }
  var nodes = snap.nodes;

  function render(id, prefix, isLast) {
    var n = nodes[id];
    var conn = '';
    if (id !== 0) {
      conn = isLast ? prefix + '` + "`" + `-- ' : prefix + '|-- ';
    }
    container.appendChild(createNodeEl(n, conn));

    var childPrefix = prefix;
    if (id !== 0) {
      childPrefix += isLast ? '    ' : '|   ';
    }
    var ch = n.children || [];
    for (var i = 0; i < ch.length; i++) {
      render(ch[i], childPrefix, i === ch.length - 1);
    }
  }

  render(0, '', true);
}

function flashNode(range, cls) {
  var el = document.querySelector('[data-range="' + range + '"]');
  if (!el) return;
  el.classList.add(cls);
  setTimeout(function() { el.classList.remove(cls); }, 1500);
}

function addEvent(evt) {
  var log = $('event-log');
  var time = new Date(evt.timestamp).toLocaleTimeString();
  var range = '[' + evt.offset_range[0] + ',' + evt.offset_range[1] + ')';
  var detail = evt.token_count + 't';
  if (evt.bytes > 0) detail += ' ' + fmtBytes(evt.bytes);
  if (evt.disk_file) detail += ' ' + evt.disk_file;

  var div = document.createElement('div');
  div.className = 'event';

  var typeSpan = document.createElement('span');
  typeSpan.className = 'event-type et-' + evt.type;
  typeSpan.textContent = evt.type;
  div.appendChild(typeSpan);

  div.appendChild(document.createTextNode(' ' + range + ' ' + detail + ' '));

  var timeSpan = document.createElement('span');
  timeSpan.className = 'bytes';
  timeSpan.textContent = time;
  div.appendChild(timeSpan);

  log.insertBefore(div, log.firstChild);

  while (log.children.length > 100) log.removeChild(log.lastChild);
}

function handleEvent(evt) {
  addEvent(evt);
  var range = evt.offset_range[0] + '-' + evt.offset_range[1];
  var flashMap = {
    page_out: 'flash-pageout', page_in: 'flash-pagein', page_in_disk: 'flash-pagein',
    evict_to_disk: 'flash-disk', evict_from_disk: 'flash-disk', snapshot: 'flash-snap'
  };
  flashNode(range, flashMap[evt.type] || 'flash-snap');
}

var es = new EventSource('/v1/cache/events');
es.addEventListener('init', function(e) {
  var snap = JSON.parse(e.data);
  renderStats(snap.stats || {});
  renderTrie(snap);
});
['page_out','page_in','page_in_disk','evict_to_disk','evict_from_disk','snapshot'].forEach(function(t) {
  es.addEventListener(t, function(e) { handleEvent(JSON.parse(e.data)); });
});

es.onopen = function() { $('status').textContent = 'Connected'; $('status').className = 'status'; };
es.onerror = function() { $('status').textContent = 'Disconnected'; $('status').className = 'status disconnected'; };
</script>
</body>
</html>`
