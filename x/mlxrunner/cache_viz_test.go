package mlxrunner

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// ---------- Snapshot builder tests ----------

func TestBuildTrieSnapshot(t *testing.T) {
	// Build a trie: root -> [1,2,3] -> [4,5]
	//                    -> [1,2,6]
	root := &trieNode{lastUsed: time.Now()}
	child1 := &trieNode{
		tokens:    []int32{1, 2, 3},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	child2 := &trieNode{
		tokens:    []int32{1, 2, 6},
		endOffset: 3,
		parent:    root,
		lastUsed:  time.Now(),
	}
	grandchild := &trieNode{
		tokens:    []int32{4, 5},
		endOffset: 5,
		parent:    child1,
		lastUsed:  time.Now(),
	}
	root.children = []*trieNode{child1, child2}
	child1.children = []*trieNode{grandchild}

	c := &kvCache{
		root:       root,
		activePath: []*trieNode{root, child1, grandchild},
	}

	snap := c.buildTrieSnapshot()

	if snap.Stats.NodeCount != 4 {
		t.Errorf("expected 4 nodes, got %d", snap.Stats.NodeCount)
	}

	// Root node (id=0) should have 2 children and be "active".
	n0 := snap.Nodes[0]
	if len(n0.Children) != 2 {
		t.Errorf("root should have 2 children, got %d", len(n0.Children))
	}
	if !containsFlag(n0.Flags, "active") {
		t.Error("root should have 'active' flag")
	}

	// child1 (id=1) should be active.
	n1 := snap.Nodes[1]
	if n1.OffsetRange != [2]int{0, 3} {
		t.Errorf("child1 offset range: got %v, want [0,3]", n1.OffsetRange)
	}
	if n1.TokenCount != 3 {
		t.Errorf("child1 token count: got %d, want 3", n1.TokenCount)
	}
	if !containsFlag(n1.Flags, "active") {
		t.Error("child1 should have 'active' flag")
	}

	// child2 (id=3 in DFS: root=0, child1=1, grandchild=2, child2=3)
	n3 := snap.Nodes[3]
	if containsFlag(n3.Flags, "active") {
		t.Error("child2 should NOT have 'active' flag")
	}

	// grandchild (id=2) should be active.
	n2 := snap.Nodes[2]
	if n2.OffsetRange != [2]int{3, 5} {
		t.Errorf("grandchild offset range: got %v, want [3,5]", n2.OffsetRange)
	}
	if !containsFlag(n2.Flags, "active") {
		t.Error("grandchild should have 'active' flag")
	}
}

func TestBuildTrieSnapshotWithDiskNode(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	child := &trieNode{
		tokens:       []int32{1, 2},
		endOffset:    2,
		parent:       root,
		lastUsed:     time.Now(),
		diskFile:     "/some/path/evicted_0.safetensors",
		diskFileSize: 1024,
	}
	root.children = []*trieNode{child}

	c := &kvCache{
		root:           root,
		activePath:     []*trieNode{root},
		totalDiskBytes: 1024,
	}

	snap := c.buildTrieSnapshot()

	if snap.Stats.DiskNodeCount != 1 {
		t.Errorf("expected 1 disk node, got %d", snap.Stats.DiskNodeCount)
	}
	if snap.Stats.TotalDiskBytes != 1024 {
		t.Errorf("expected 1024 total disk bytes, got %d", snap.Stats.TotalDiskBytes)
	}

	n1 := snap.Nodes[1]
	if !containsFlag(n1.Flags, "disk") {
		t.Error("child should have 'disk' flag")
	}
	if n1.DiskFile != "evicted_0.safetensors" {
		t.Errorf("disk file: got %q, want %q", n1.DiskFile, "evicted_0.safetensors")
	}
	if n1.DiskFileSize != 1024 {
		t.Errorf("disk file size: got %d, want 1024", n1.DiskFileSize)
	}
}

func TestBuildTrieSnapshotEmpty(t *testing.T) {
	c := &kvCache{}
	snap := c.buildTrieSnapshot()

	if snap.Stats.NodeCount != 0 {
		t.Errorf("expected 0 nodes for nil root, got %d", snap.Stats.NodeCount)
	}
}

// ---------- Text renderer tests ----------

func TestRenderTrieText(t *testing.T) {
	snap := &TrieSnapshot{
		Stats: TrieStats{
			ActiveTokens: 5,
			NodeCount:    3,
		},
		Nodes: []TrieNodeInfo{
			{ID: 0, OffsetRange: [2]int{0, 0}, TokenCount: 0, Children: []int{1}, Flags: []string{"active"}},
			{ID: 1, OffsetRange: [2]int{0, 3}, TokenCount: 3, Children: []int{2}, Flags: []string{"active", "snap"}},
			{ID: 2, OffsetRange: [2]int{3, 5}, TokenCount: 2, Flags: []string{"active"}},
		},
	}

	text := renderTrieText(snap)

	if !strings.Contains(text, "active_tokens: 5") {
		t.Error("text should contain active_tokens stat")
	}
	if !strings.Contains(text, "[0,3)") {
		t.Error("text should contain child offset range [0,3)")
	}
	if !strings.Contains(text, "[3,5)") {
		t.Error("text should contain grandchild offset range [3,5)")
	}
	if !strings.Contains(text, "snap") {
		t.Error("text should contain 'snap' flag")
	}
}

func TestRenderTrieTextEmpty(t *testing.T) {
	text := renderTrieText(nil)
	if text != "(empty trie)\n" {
		t.Errorf("expected empty trie text, got %q", text)
	}
}

// ---------- Event bus tests ----------

func TestEventBusSubscribeEmit(t *testing.T) {
	bus := newCacheEventBus()
	ch, unsub := bus.subscribe()
	defer unsub()

	event := CacheEvent{
		Type:        EventPageOut,
		OffsetRange: [2]int{0, 10},
		TokenCount:  10,
		Bytes:       1024,
	}
	bus.emit(event)

	select {
	case got := <-ch:
		if got.Type != EventPageOut {
			t.Errorf("expected page_out, got %s", got.Type)
		}
		if got.TokenCount != 10 {
			t.Errorf("expected 10 tokens, got %d", got.TokenCount)
		}
	default:
		t.Fatal("expected event on channel, got nothing")
	}
}

func TestEventBusUnsubscribe(t *testing.T) {
	bus := newCacheEventBus()
	ch, unsub := bus.subscribe()

	if bus.count.Load() != 1 {
		t.Fatalf("expected 1 subscriber, got %d", bus.count.Load())
	}

	unsub()

	if bus.count.Load() != 0 {
		t.Fatalf("expected 0 subscribers after unsub, got %d", bus.count.Load())
	}

	// Channel should be closed.
	_, ok := <-ch
	if ok {
		t.Fatal("expected channel to be closed after unsubscribe")
	}
}

func TestEventBusDropOnFull(t *testing.T) {
	bus := newCacheEventBus()
	ch, unsub := bus.subscribe()
	defer unsub()

	// Fill the buffer (cap=64).
	for i := range 64 {
		bus.emit(CacheEvent{Type: EventPageIn, TokenCount: i})
	}

	// This should not block — event is dropped.
	bus.emit(CacheEvent{Type: EventPageOut, TokenCount: 999})

	// Drain and verify the dropped event is not in there.
	count := 0
	for {
		select {
		case e := <-ch:
			if e.TokenCount == 999 {
				t.Error("overflow event should have been dropped")
			}
			count++
		default:
			goto done
		}
	}
done:
	if count != 64 {
		t.Errorf("expected 64 buffered events, got %d", count)
	}
}

func TestEventBusNoSubscribers(t *testing.T) {
	bus := newCacheEventBus()
	// Should not panic.
	bus.emit(CacheEvent{Type: EventPageOut})
}

func TestEventBusNilSafe(t *testing.T) {
	// emitEvent on kvCache with nil events should be a no-op.
	c := &kvCache{}
	node := &trieNode{tokens: []int32{1, 2}, endOffset: 2}
	c.emitEvent(EventPageOut, node, 100, "") // should not panic
}

// ---------- HTTP handler tests ----------

func TestHandleTrieSnapshotJSON(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	c := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		events:     newCacheEventBus(),
	}
	c.rebuildSnapshot()

	req := httptest.NewRequest("GET", "/v1/cache/trie", nil)
	w := httptest.NewRecorder()
	c.handleTrieSnapshot(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json, got %s", ct)
	}

	var snap TrieSnapshot
	if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if snap.Stats.NodeCount != 1 {
		t.Errorf("expected 1 node, got %d", snap.Stats.NodeCount)
	}
}

func TestHandleTrieSnapshotText(t *testing.T) {
	root := &trieNode{lastUsed: time.Now()}
	c := &kvCache{
		root:       root,
		activePath: []*trieNode{root},
		events:     newCacheEventBus(),
	}
	c.rebuildSnapshot()

	req := httptest.NewRequest("GET", "/v1/cache/trie?format=text", nil)
	w := httptest.NewRecorder()
	c.handleTrieSnapshot(w, req)

	if ct := w.Header().Get("Content-Type"); ct != "text/plain" {
		t.Errorf("expected text/plain, got %s", ct)
	}
	body := w.Body.String()
	if !strings.Contains(body, "cache trie:") {
		t.Error("text response should contain 'cache trie:' header")
	}
}

func TestHandleTrieSnapshotStripTokens(t *testing.T) {
	root := &trieNode{tokens: nil, lastUsed: time.Now()}
	child := &trieNode{tokens: []int32{1, 2, 3}, endOffset: 3, parent: root, lastUsed: time.Now()}
	root.children = []*trieNode{child}

	c := &kvCache{
		root:       root,
		activePath: []*trieNode{root, child},
		events:     newCacheEventBus(),
	}
	c.rebuildSnapshot()

	// Without ?tokens=true, tokens should be nil.
	req := httptest.NewRequest("GET", "/v1/cache/trie", nil)
	w := httptest.NewRecorder()
	c.handleTrieSnapshot(w, req)

	var snap TrieSnapshot
	json.Unmarshal(w.Body.Bytes(), &snap)
	for _, n := range snap.Nodes {
		if n.Tokens != nil {
			t.Errorf("tokens should be nil without ?tokens=true, got %v for node %d", n.Tokens, n.ID)
		}
	}

	// With ?tokens=true, tokens should be present.
	req2 := httptest.NewRequest("GET", "/v1/cache/trie?tokens=true", nil)
	w2 := httptest.NewRecorder()
	c.handleTrieSnapshot(w2, req2)

	var snap2 TrieSnapshot
	json.Unmarshal(w2.Body.Bytes(), &snap2)
	if snap2.Nodes[1].Tokens == nil {
		t.Error("tokens should be present with ?tokens=true")
	}
}

func TestHandleTrieSnapshotNilSnapshot(t *testing.T) {
	c := &kvCache{events: newCacheEventBus()}

	req := httptest.NewRequest("GET", "/v1/cache/trie", nil)
	w := httptest.NewRecorder()
	c.handleTrieSnapshot(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var snap TrieSnapshot
	if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}
	if snap.Timestamp == "" {
		t.Error("expected non-empty timestamp in fallback response")
	}
}

func TestHandleCacheEventsSSE(t *testing.T) {
	c := &kvCache{events: newCacheEventBus()}

	// Build a snapshot so the initial event has something.
	root := &trieNode{lastUsed: time.Now()}
	c.root = root
	c.activePath = []*trieNode{root}
	c.rebuildSnapshot()

	// Start the handler in a goroutine since it blocks.
	req := httptest.NewRequest("GET", "/v1/cache/events", nil)
	ctx, cancel := newTestContext()
	defer cancel()
	req = req.WithContext(ctx)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		c.handleCacheEvents(w, req)
		close(done)
	}()

	// Wait for the handler to subscribe before emitting.
	deadline := time.Now().Add(time.Second)
	for c.events.count.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for SSE handler to subscribe")
		}
		time.Sleep(time.Millisecond)
	}

	// Emit an event.
	c.events.emit(CacheEvent{
		Type:        EventPageOut,
		Timestamp:   time.Now().UTC().Format(time.RFC3339Nano),
		OffsetRange: [2]int{0, 10},
		TokenCount:  10,
		Bytes:       2048,
	})

	// Give handler time to write, then cancel.
	time.Sleep(50 * time.Millisecond)
	cancel()
	<-done

	body := w.Body.String()
	if !strings.Contains(body, "event: snapshot") {
		t.Error("SSE stream should start with initial snapshot event")
	}
	if !strings.Contains(body, "event: page_out") {
		t.Error("SSE stream should contain page_out event")
	}
	if !strings.Contains(body, "data: ") {
		t.Error("SSE stream should contain data lines")
	}
}

// ---------- Helpers ----------

func containsFlag(flags []string, flag string) bool {
	for _, f := range flags {
		if f == flag {
			return true
		}
	}
	return false
}

// newTestContext creates a cancellable context for SSE tests.
func newTestContext() (ctx interface{ Done() <-chan struct{}; Err() error; Deadline() (time.Time, bool); Value(any) any }, cancel func()) {
	type contextLike interface {
		Done() <-chan struct{}
		Err() error
		Deadline() (time.Time, bool)
		Value(any) any
	}
	ch := make(chan struct{})
	cancelled := false
	cancelFn := func() {
		if !cancelled {
			cancelled = true
			close(ch)
		}
	}
	return &testCtx{done: ch}, cancelFn
}

type testCtx struct{ done chan struct{} }

func (c *testCtx) Done() <-chan struct{}        { return c.done }
func (c *testCtx) Err() error                   { select { case <-c.done: return http.ErrAbortHandler; default: return nil } }
func (c *testCtx) Deadline() (time.Time, bool)  { return time.Time{}, false }
func (c *testCtx) Value(any) any                { return nil }
