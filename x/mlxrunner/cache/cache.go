package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

type Cache interface {
	Update(b *batch.ForwardBatch, keys, values *mlx.Array) (newKeys, newValues *mlx.Array, kv mlx.KVHistory)
	State() []*mlx.Array
	Free()
	Offsets(seqIDs ...int) []int32

	Snapshot(seqID int, fromOffset int) Snapshot
	Restore(seqID int, snapshot Snapshot, target int) bool
	Merge(parent, child Snapshot) Snapshot
	Split(snapshot Snapshot, at int) (parent, child Snapshot)

	SetSeqs(seqIDs []int)
}

type Snapshot interface {
	Size() int
	Close()
}

// SerializableSnapshot is a Snapshot whose backing arrays can be written to
// and read back from a safetensors file. Snapshot types that don't implement
// this interface are skipped by the persistence layer (the writer refuses to
// produce files with partial coverage).
type SerializableSnapshot interface {
	Snapshot
	// SnapshotType returns the tag stored in safetensors snapshot_types
	// metadata (e.g. "kv"). The same tag is later passed to
	// NewSnapshotFromArrays to reconstruct the snapshot on load.
	SnapshotType() string
	// CollectArrays returns the field-name -> array map the persistence
	// layer serializes, along with a cleanup that releases any arrays the
	// snapshot allocated solely for serialization (e.g. scalars packed
	// into 1-element arrays). The caller must invoke cleanup after the
	// write completes; snapshot-owned arrays (keys/values) remain owned
	// by the snapshot and are released by Close.
	CollectArrays() (arrays map[string]*mlx.Array, cleanup func())
}

// SnapshotType* are the canonical tags stored in safetensors snapshot_types
// metadata. Callers that read or write these tags should use these constants
// rather than literals — a typo produces a mis-tagged file that rehydrate
// silently invalidates.
const (
	SnapshotTypeKV        = "kv"
	SnapshotTypeRecurrent = "recurrent"
	SnapshotTypeRotating  = "rotating"
	SnapshotTypeEmpty     = "empty"
	SnapshotTypeUnknown   = "unknown"
)

// snapshotTypeFields lists the field names each snapshot type writes via
// CollectArrays. Persistence layers use this to know which safetensors
// keys to read per layer, without having to enumerate the file. The
// "empty" type is a valid sentinel for layers that have no state yet
// (e.g. a recurrent layer at the start of a conversation) — it carries
// no fields, takes no disk space, and restores to a nil Snapshot.
var snapshotTypeFields = map[string][]string{
	SnapshotTypeKV:        {"keys", "values"},
	SnapshotTypeRecurrent: {"conv", "delta"},
	SnapshotTypeRotating:  {"keys", "values"},
	SnapshotTypeEmpty:     {},
}

// SnapshotFieldNames returns the per-type field names a SerializableSnapshot
// produces. The ok return distinguishes "valid type with no fields" (e.g.
// "empty") from "unknown type".
func SnapshotFieldNames(snapshotType string) (names []string, ok bool) {
	names, ok = snapshotTypeFields[snapshotType]
	return
}

// NewSnapshotFromArrays reconstructs a snapshot of the given type from
// arrays previously returned by CollectArrays. Ownership of the arrays
// transfers to the returned Snapshot, which unpins them on Close. The
// "empty" type returns (nil, nil) — downstream restore logic already
// handles nil snapshots as "this layer had no state".
func NewSnapshotFromArrays(snapshotType string, fields map[string]*mlx.Array, fromOffset, toOffset int) (Snapshot, error) {
	switch snapshotType {
	case SnapshotTypeEmpty:
		return nil, nil
	case SnapshotTypeKV:
		k, v := fields["keys"], fields["values"]
		if k == nil || v == nil {
			return nil, fmt.Errorf("kv snapshot missing keys/values")
		}
		return &kvSnapshot{keys: k, values: v, fromOffset: fromOffset, toOffset: toOffset}, nil
	case SnapshotTypeRecurrent:
		conv, delta := fields["conv"], fields["delta"]
		if conv == nil || delta == nil {
			return nil, fmt.Errorf("recurrent snapshot missing conv/delta")
		}
		// Recurrent state is cumulative up to the end of the trie edge;
		// Restore() will only succeed when target == offset, which is toOffset.
		return &recurrentSnapshot{convState: conv, deltaState: delta, offset: toOffset}, nil
	case SnapshotTypeRotating:
		k, v := fields["keys"], fields["values"]
		if k == nil || v == nil {
			return nil, fmt.Errorf("rotating snapshot missing keys/values")
		}
		// The rotating cache stores canonical oldest→newest data and uses
		// offset = toOffset to reconstruct per-slot positions on restore.
		return &rotatingSnapshot{
			keys:   k,
			values: v,
			offset: toOffset,
		}, nil
	default:
		return nil, fmt.Errorf("unknown snapshot type %q", snapshotType)
	}
}

// KVCache stores key/value pairs in a contiguous buffer with per-sequence
// regions separated by gaps. All operations go through regions — there is
// no separate single-sequence mode.
type KVCache struct {
	keys, values *mlx.Array
	step         int

	seqOrder []int
	regions  map[int]*seqRegion
}

type seqRegion struct {
	start    int
	length   int
	capacity int
}

func NewKVCache() *KVCache {
	return &KVCache{step: 256}
}

func (c *KVCache) SetSeqs(seqIDs []int) {
	if c.regions == nil {
		c.regions = make(map[int]*seqRegion)
	}

	wanted := make(map[int]bool, len(seqIDs))
	for _, id := range seqIDs {
		wanted[id] = true
	}

	// Skip rebuild if the set hasn't changed.
	changed := len(seqIDs) != len(c.seqOrder)
	if !changed {
		for _, id := range c.seqOrder {
			if !wanted[id] {
				changed = true
				break
			}
		}
	}
	if !changed {
		return
	}

	// Remove sequences not in the new set
	for _, id := range c.seqOrder {
		if !wanted[id] {
			delete(c.regions, id)
		}
	}

	// Build new order preserving existing order, then appending new sequences
	newOrder := make([]int, 0, len(seqIDs))
	for _, id := range c.seqOrder {
		if wanted[id] {
			newOrder = append(newOrder, id)
		}
	}
	for _, id := range seqIDs {
		if _, ok := c.regions[id]; !ok {
			newOrder = append(newOrder, id)
			c.regions[id] = &seqRegion{}
		}
	}
	c.seqOrder = newOrder

	if c.keys != nil {
		c.rebuild(c.keys.DType(), c.values.DType(), c.keys.Dim(1), c.keys.Dim(3), c.values.Dim(3), nil)
	}
}

func (c *KVCache) Update(b *batch.ForwardBatch, keys, values *mlx.Array) (*mlx.Array, *mlx.Array, mlx.KVHistory) {
	for _, seqID := range b.SeqIDs {
		if _, ok := c.regions[seqID]; !ok {
			panic(fmt.Sprintf("KVCache.Update: sequence %d not found in cache", seqID))
		}
	}

	// Check if any sequence will exhaust its gap
	needsRebuild := false
	for i, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		if r.length+b.SeqLens[i] > r.capacity {
			needsRebuild = true
			break
		}
	}

	if needsRebuild {
		oldLengths := make(map[int]int, len(b.SeqIDs))
		for _, seqID := range b.SeqIDs {
			oldLengths[seqID] = c.regions[seqID].length
		}
		for i, seqID := range b.SeqIDs {
			c.regions[seqID].length += b.SeqLens[i]
		}
		c.rebuild(keys.DType(), values.DType(), keys.Dim(1), keys.Dim(3), values.Dim(3), oldLengths)
	}

	// Write K/V per sequence using contiguous SliceUpdate
	tokenIdx := 0
	for i, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		writeStart := r.length
		if needsRebuild {
			writeStart = r.length - b.SeqLens[i]
		}
		start := r.start + writeStart
		end := start + b.SeqLens[i]
		kSlice := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(tokenIdx, tokenIdx+b.SeqLens[i]), mlx.Slice())
		vSlice := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(tokenIdx, tokenIdx+b.SeqLens[i]), mlx.Slice())
		c.keys.Set(c.keys.SliceUpdate(kSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()))
		c.values.Set(c.values.SliceUpdate(vSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice()))
		tokenIdx += b.SeqLens[i]
	}

	if !needsRebuild {
		for i, seqID := range b.SeqIDs {
			c.regions[seqID].length += b.SeqLens[i]
		}
	}

	return c.sliceForBatch(b)
}

// sliceForBatch returns the smallest contiguous K/V slice covering all batch
// sequences, with page table indices remapped relative to the slice start.
func (c *KVCache) sliceForBatch(b *batch.ForwardBatch) (*mlx.Array, *mlx.Array, mlx.KVHistory) {
	sliceStart, sliceEnd := c.batchExtent(b)
	kv := c.buildKVHistory(b, sliceStart)
	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(sliceStart, sliceEnd), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(sliceStart, sliceEnd), mlx.Slice()),
		kv
}

// batchExtent returns the smallest [start, end) range covering all batch sequences' regions.
func (c *KVCache) batchExtent(b *batch.ForwardBatch) (int, int) {
	minStart := -1
	maxEnd := 0
	for _, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		if minStart < 0 || r.start < minStart {
			minStart = r.start
		}
		if end := r.start + r.length; end > maxEnd {
			maxEnd = end
		}
	}
	if minStart < 0 {
		minStart = 0
	}
	return minStart, maxEnd
}

func putAlongDim2(target, indices, values *mlx.Array) *mlx.Array {
	H := int32(target.Dim(1))
	D := int32(target.Dim(3))
	totalSlots := int32(target.Dim(2))
	N := int32(values.Dim(2))

	t := mlx.Reshape(target, H, totalSlots, D)
	v := mlx.Reshape(values, H, N, D)
	result := t.PutAlongAxis(indices, v, 1)
	return mlx.Reshape(result, 1, H, totalSlots, D)
}

func (c *KVCache) buildKVHistory(b *batch.ForwardBatch, sliceStart int) mlx.KVHistory {
	numSeqs := len(b.SeqIDs)
	if numSeqs == 0 {
		return mlx.KVHistory{}
	}

	maxLen := 0
	seqLens := make([]int, numSeqs)
	for i, seqID := range b.SeqIDs {
		seqLens[i] = c.regions[seqID].length
		if seqLens[i] > maxLen {
			maxLen = seqLens[i]
		}
	}

	pt := make([]int32, numSeqs*maxLen)
	for i, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		for j := range r.length {
			pt[i*maxLen+j] = int32(r.start + j - sliceStart)
		}
	}

	return mlx.KVHistory{
		PageTable: mlx.NewArrayInt32(pt, []int32{int32(numSeqs), int32(maxLen)}),
		SeqLens:   seqLens,
	}
}

// rebuild reallocates the buffer with new region layout. oldLengths maps
// seqID to the number of valid tokens in the OLD buffer (before any
// pre-increment for pending writes). If nil, uses current r.length.
func (c *KVCache) rebuild(kDType, vDType mlx.DType, H, Dk, Dv int, oldLengths map[int]int) {
	// Single pass: save old positions, reassign regions.
	oldStarts := make([]int, len(c.seqOrder))
	copyLens := make([]int, len(c.seqOrder))
	totalSlots := 0
	for i, seqID := range c.seqOrder {
		r := c.regions[seqID]
		oldStarts[i] = r.start
		copyLens[i] = r.length
		if oldLengths != nil {
			if ol, ok := oldLengths[seqID]; ok {
				copyLens[i] = ol
			}
		}

		newCap := roundUpStep(r.length, c.step) + c.step
		r.start = totalSlots
		r.capacity = newCap
		totalSlots += newCap
	}

	if totalSlots == 0 {
		if c.keys != nil {
			mlx.Unpin(c.keys, c.values)
			c.keys, c.values = nil, nil
		}
		return
	}

	newKeys := mlx.Zeros(kDType, 1, H, totalSlots, Dk)
	newValues := mlx.Zeros(vDType, 1, H, totalSlots, Dv)

	if c.keys != nil {
		for i, seqID := range c.seqOrder {
			if copyLens[i] == 0 {
				continue
			}
			oldStart := oldStarts[i]
			newStart := c.regions[seqID].start
			kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(oldStart, oldStart+copyLens[i]), mlx.Slice())
			vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(oldStart, oldStart+copyLens[i]), mlx.Slice())
			newKeys.Set(newKeys.SliceUpdate(kSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(newStart, newStart+copyLens[i]), mlx.Slice()))
			newValues.Set(newValues.SliceUpdate(vSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(newStart, newStart+copyLens[i]), mlx.Slice()))
		}
		mlx.Unpin(c.keys, c.values)
	}

	c.keys, c.values = newKeys, newValues
	mlx.Pin(c.keys, c.values)
}

func roundUpStep(n, step int) int {
	return ((n + step - 1) / step) * step
}

func (c *KVCache) usedSlots() int {
	maxUsed := 0
	for _, seqID := range c.seqOrder {
		if r, ok := c.regions[seqID]; ok {
			if used := r.start + r.length; used > maxUsed {
				maxUsed = used
			}
		}
	}
	return maxUsed
}

func (c *KVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	end := c.usedSlots()
	if end == 0 {
		return nil
	}
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, end), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, end), mlx.Slice()),
	}
}

// kvSnapshot holds paged-out KV data for a range [fromOffset, toOffset).
type kvSnapshot struct {
	keys, values         *mlx.Array
	fromOffset, toOffset int
}

func (s *kvSnapshot) Size() int           { return s.keys.NumBytes() + s.values.NumBytes() }
func (s *kvSnapshot) Close()               { mlx.Unpin(s.keys, s.values) }
func (s *kvSnapshot) SnapshotType() string { return SnapshotTypeKV }
func (s *kvSnapshot) CollectArrays() (map[string]*mlx.Array, func()) {
	return map[string]*mlx.Array{"keys": s.keys, "values": s.values}, func() {}
}

func (c *KVCache) Snapshot(seqID int, fromOffset int) Snapshot {
	if c.keys == nil {
		return nil
	}
	r, ok := c.regions[seqID]
	if !ok {
		return nil
	}

	from := max(0, fromOffset)
	to := r.length
	if to <= from {
		return nil
	}

	start := r.start + from
	end := r.start + to

	kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(start, end), mlx.Slice())
	kCopy := mlx.Contiguous(kSlice, false)
	vCopy := mlx.Contiguous(vSlice, false)
	mlx.Pin(kCopy, vCopy)
	mlx.AsyncEval(kCopy, vCopy)

	return &kvSnapshot{
		keys:       kCopy,
		values:     vCopy,
		fromOffset: from,
		toOffset:   to,
	}
}

func (c *KVCache) Restore(seqID int, snapshot Snapshot, target int) bool {
	r, ok := c.regions[seqID]
	if !ok || target < 0 {
		return false
	}

	if snapshot == nil {
		if target > r.length {
			return false
		}
		r.length = target
		return true
	}

	// Soft type assertion: a cross-type snapshot (e.g. a *rotatingSnapshot fed
	// to a KVCache because a rehydrated file's per-layer type tag doesn't
	// match the runtime cache architecture) must degrade to a miss, not panic
	// the subprocess. The caller treats false as "unrestorable" and purges.
	snap, ok := snapshot.(*kvSnapshot)
	if !ok {
		return false
	}
	if target > snap.toOffset {
		return false
	}
	// Can't bridge a gap — cache must have data up to the snapshot start
	if snap.fromOffset > r.length {
		return false
	}

	// Check if snapshot fits in the region's capacity
	if snap.toOffset > r.capacity || c.keys == nil {
		oldLen := r.length
		r.length = snap.toOffset
		oldLengths := map[int]int{seqID: oldLen}
		c.rebuild(snap.keys.DType(), snap.values.DType(), snap.keys.Dim(1), snap.keys.Dim(3), snap.values.Dim(3), oldLengths)
	}

	writeStart := r.start + snap.fromOffset
	writeEnd := r.start + snap.toOffset
	c.keys.Set(c.keys.SliceUpdate(snap.keys,
		mlx.Slice(), mlx.Slice(), mlx.Slice(writeStart, writeEnd), mlx.Slice()))
	c.values.Set(c.values.SliceUpdate(snap.values,
		mlx.Slice(), mlx.Slice(), mlx.Slice(writeStart, writeEnd), mlx.Slice()))

	r.length = target
	return true
}

func (c *KVCache) Merge(parent, child Snapshot) Snapshot {
	if parent == nil || child == nil {
		if parent != nil {
			parent.Close()
		}
		if child != nil {
			child.Close()
		}
		return nil
	}
	// Soft type assertions: a cross-type snapshot would otherwise panic.
	// Returning nil after closing both inputs drops the merge result; the
	// caller treats that as a trie-level reconciliation signal.
	p, pok := parent.(*kvSnapshot)
	ch, cok := child.(*kvSnapshot)
	if !pok || !cok {
		parent.Close()
		child.Close()
		return nil
	}

	mk := p.keys.Concatenate(2, ch.keys)
	mv := p.values.Concatenate(2, ch.values)
	mlx.Pin(mk, mv)
	mlx.AsyncEval(mk, mv)

	p.Close()
	ch.Close()

	return &kvSnapshot{
		keys:       mk,
		values:     mv,
		fromOffset: p.fromOffset,
		toOffset:   ch.toOffset,
	}
}

func (c *KVCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	if snapshot == nil {
		return nil, nil
	}
	// Soft type assertion: on cross-type mismatch, return the snapshot as
	// the child half (same shape as the "can't split" fallback for other
	// cache types) instead of panicking.
	snap, ok := snapshot.(*kvSnapshot)
	if !ok {
		return nil, snapshot
	}
	splitIdx := at - snap.fromOffset
	seqLen := snap.toOffset - snap.fromOffset
	if splitIdx <= 0 {
		return nil, snapshot
	}
	if splitIdx >= seqLen {
		return snapshot, nil
	}

	pk := mlx.Contiguous(snap.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	pv := mlx.Contiguous(snap.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, splitIdx), mlx.Slice()), false)
	ck := mlx.Contiguous(snap.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	cv := mlx.Contiguous(snap.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(splitIdx, seqLen), mlx.Slice()), false)
	mlx.Pin(pk, pv, ck, cv)
	mlx.AsyncEval(pk, pv, ck, cv)

	snap.Close()

	return &kvSnapshot{
			keys: pk, values: pv,
			fromOffset: snap.fromOffset, toOffset: at,
		}, &kvSnapshot{
			keys: ck, values: cv,
			fromOffset: at, toOffset: snap.toOffset,
		}
}

func (c *KVCache) Free() {
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = nil, nil
	// Preserve sequence registration — callers may Restore after Free.
	for _, seqID := range c.seqOrder {
		if r, ok := c.regions[seqID]; ok {
			r.length = 0
			r.capacity = 0
			r.start = 0
		}
	}
}

func (c *KVCache) Offsets(seqIDs ...int) []int32 {
	offsets := make([]int32, len(seqIDs))
	for i, seqID := range seqIDs {
		if r, ok := c.regions[seqID]; ok {
			offsets[i] = int32(r.length)
		}
	}
	return offsets
}

// RotatingKVCache implements sliding-window KV caching with a fixed-size
// ring buffer per sequence. Each sequence gets maxSize slots. New tokens
// overwrite the oldest slot when the ring wraps. The page table maps
// visible positions to the physical (possibly wrapped) buffer slots.
type RotatingKVCache struct {
	keys, values *mlx.Array
	maxSize      int
	step         int

	seqOrder []int
	regions  map[int]*ringRegion
}

type ringRegion struct {
	start  int // offset into shared buffer
	offset int // total tokens written (wraps for ring position)
}

func NewRotatingKVCache(maxSize int) *RotatingKVCache {
	return &RotatingKVCache{maxSize: maxSize, step: 256}
}

func (c *RotatingKVCache) visible(r *ringRegion) int {
	return min(r.offset, c.maxSize)
}

func (c *RotatingKVCache) SetSeqs(seqIDs []int) {
	if c.regions == nil {
		c.regions = make(map[int]*ringRegion)
	}

	wanted := make(map[int]bool, len(seqIDs))
	for _, id := range seqIDs {
		wanted[id] = true
	}

	changed := len(seqIDs) != len(c.seqOrder)
	if !changed {
		for _, id := range c.seqOrder {
			if !wanted[id] {
				changed = true
				break
			}
		}
	}
	if !changed {
		return
	}

	for _, id := range c.seqOrder {
		if !wanted[id] {
			delete(c.regions, id)
		}
	}

	newOrder := make([]int, 0, len(seqIDs))
	for _, id := range c.seqOrder {
		if wanted[id] {
			newOrder = append(newOrder, id)
		}
	}
	for _, id := range seqIDs {
		if _, ok := c.regions[id]; !ok {
			newOrder = append(newOrder, id)
			c.regions[id] = &ringRegion{}
		}
	}
	c.seqOrder = newOrder

	if c.keys != nil {
		c.rebuild(c.keys.DType(), c.values.DType(), c.keys.Dim(1), c.keys.Dim(3), c.values.Dim(3))
	}
}

func (c *RotatingKVCache) Update(b *batch.ForwardBatch, keys, values *mlx.Array) (*mlx.Array, *mlx.Array, mlx.KVHistory) {
	if c.keys == nil {
		c.rebuild(keys.DType(), values.DType(), keys.Dim(1), keys.Dim(3), values.Dim(3))
	}

	for _, seqID := range b.SeqIDs {
		if _, ok := c.regions[seqID]; !ok {
			panic(fmt.Sprintf("RotatingKVCache.Update: sequence %d not found in cache", seqID))
		}
	}

	// Write K/V into the ring. When n > maxSize, only the last maxSize tokens
	// matter (the rest would be immediately overwritten). At most one wrap
	// per sequence, producing at most 2 SliceUpdates.
	tokenIdx := 0
	for i, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		n := b.SeqLens[i]
		writeTokenIdx := tokenIdx
		writeN := n

		// If more tokens than ring capacity, skip the oldest ones
		if writeN > c.maxSize {
			skip := writeN - c.maxSize
			writeTokenIdx += skip
			writeN = c.maxSize
		}

		ringPos := (r.offset + (n - writeN)) % c.maxSize

		if ringPos+writeN <= c.maxSize {
			slot := r.start + ringPos
			kSlice := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx, writeTokenIdx+writeN), mlx.Slice())
			vSlice := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx, writeTokenIdx+writeN), mlx.Slice())
			c.keys.Set(c.keys.SliceUpdate(kSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(slot, slot+writeN), mlx.Slice()))
			c.values.Set(c.values.SliceUpdate(vSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(slot, slot+writeN), mlx.Slice()))
		} else {
			part1 := c.maxSize - ringPos
			part2 := writeN - part1

			slot1 := r.start + ringPos
			k1 := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx, writeTokenIdx+part1), mlx.Slice())
			v1 := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx, writeTokenIdx+part1), mlx.Slice())
			c.keys.Set(c.keys.SliceUpdate(k1, mlx.Slice(), mlx.Slice(), mlx.Slice(slot1, slot1+part1), mlx.Slice()))
			c.values.Set(c.values.SliceUpdate(v1, mlx.Slice(), mlx.Slice(), mlx.Slice(slot1, slot1+part1), mlx.Slice()))

			slot2 := r.start
			k2 := keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx+part1, writeTokenIdx+writeN), mlx.Slice())
			v2 := values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(writeTokenIdx+part1, writeTokenIdx+writeN), mlx.Slice())
			c.keys.Set(c.keys.SliceUpdate(k2, mlx.Slice(), mlx.Slice(), mlx.Slice(slot2, slot2+part2), mlx.Slice()))
			c.values.Set(c.values.SliceUpdate(v2, mlx.Slice(), mlx.Slice(), mlx.Slice(slot2, slot2+part2), mlx.Slice()))
		}
		tokenIdx += n
	}

	for i, seqID := range b.SeqIDs {
		c.regions[seqID].offset += b.SeqLens[i]
	}

	sliceStart, sliceEnd := c.batchExtent(b)
	return c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(sliceStart, sliceEnd), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(sliceStart, sliceEnd), mlx.Slice()),
		c.buildRotatingKVHistory(b, sliceStart)
}

// batchExtent returns the smallest [start, end) range covering all batch sequences' rings.
func (c *RotatingKVCache) batchExtent(b *batch.ForwardBatch) (int, int) {
	minStart := -1
	maxEnd := 0
	for _, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		if minStart < 0 || r.start < minStart {
			minStart = r.start
		}
		if end := r.start + min(r.offset, c.maxSize); end > maxEnd {
			maxEnd = end
		}
	}
	if minStart < 0 {
		minStart = 0
	}
	return minStart, maxEnd
}

func (c *RotatingKVCache) buildRotatingKVHistory(b *batch.ForwardBatch, sliceStart int) mlx.KVHistory {
	numSeqs := len(b.SeqIDs)
	if numSeqs == 0 {
		return mlx.KVHistory{}
	}

	maxSeqLen := 0
	seqLens := make([]int, numSeqs)
	for i, seqID := range b.SeqIDs {
		seqLens[i] = c.visible(c.regions[seqID])
		if seqLens[i] > maxSeqLen {
			maxSeqLen = seqLens[i]
		}
	}

	pt := make([]int32, numSeqs*maxSeqLen)
	for i, seqID := range b.SeqIDs {
		r := c.regions[seqID]
		vis := seqLens[i]
		oldestSlot := (r.offset - vis) % c.maxSize
		if oldestSlot < 0 {
			oldestSlot += c.maxSize
		}
		for j := range vis {
			pt[i*maxSeqLen+j] = int32(r.start + (oldestSlot+j)%c.maxSize - sliceStart)
		}
	}

	return mlx.KVHistory{
		PageTable: mlx.NewArrayInt32(pt, []int32{int32(numSeqs), int32(maxSeqLen)}),
		SeqLens:   seqLens,
	}
}

func (c *RotatingKVCache) rebuild(kDType, vDType mlx.DType, H, Dk, Dv int) {
	// Save old positions before reassigning
	type saved struct {
		start  int
		offset int
	}
	savedRegions := make(map[int]saved, len(c.seqOrder))
	for _, seqID := range c.seqOrder {
		r := c.regions[seqID]
		savedRegions[seqID] = saved{start: r.start, offset: r.offset}
	}

	totalSlots := len(c.seqOrder) * c.maxSize
	for i, seqID := range c.seqOrder {
		c.regions[seqID].start = i * c.maxSize
	}

	if totalSlots == 0 {
		if c.keys != nil {
			mlx.Unpin(c.keys, c.values)
		}
		c.keys, c.values = nil, nil
		return
	}

	newKeys := mlx.Zeros(kDType, 1, H, totalSlots, Dk)
	newValues := mlx.Zeros(vDType, 1, H, totalSlots, Dv)

	if c.keys != nil {
		// Copy each surviving sequence's ring data to its new position.
		// Offset is unchanged during rebuild, so the ring layout within
		// maxSize slots is identical — only the base address shifts.
		copySlice := func(srcStart, dstStart, count int) {
			kSlice := c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(srcStart, srcStart+count), mlx.Slice())
			vSlice := c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(srcStart, srcStart+count), mlx.Slice())
			newKeys.Set(newKeys.SliceUpdate(kSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(dstStart, dstStart+count), mlx.Slice()))
			newValues.Set(newValues.SliceUpdate(vSlice, mlx.Slice(), mlx.Slice(), mlx.Slice(dstStart, dstStart+count), mlx.Slice()))
		}

		for _, seqID := range c.seqOrder {
			s, ok := savedRegions[seqID]
			if !ok {
				continue
			}
			r := c.regions[seqID]
			vis := min(s.offset, c.maxSize)
			if vis == 0 {
				continue
			}
			if s.offset <= c.maxSize {
				// Ring hasn't wrapped — data is contiguous.
				copySlice(s.start, r.start, vis)
			} else {
				// Ring has wrapped — two contiguous chunks.
				wrapPoint := s.offset % c.maxSize
				copySlice(s.start+wrapPoint, r.start+wrapPoint, c.maxSize-wrapPoint)
				if wrapPoint > 0 {
					copySlice(s.start, r.start, wrapPoint)
				}
			}
		}

		mlx.Unpin(c.keys, c.values)
	}

	c.keys, c.values = newKeys, newValues
	mlx.Pin(c.keys, c.values)
}

func (c *RotatingKVCache) State() []*mlx.Array {
	if c.keys == nil || c.values == nil {
		return nil
	}
	end := 0
	for _, seqID := range c.seqOrder {
		r := c.regions[seqID]
		used := r.start + min(r.offset, c.maxSize)
		if used > end {
			end = used
		}
	}
	if end == 0 {
		return nil
	}
	return []*mlx.Array{
		c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, end), mlx.Slice()),
		c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(0, end), mlx.Slice()),
	}
}

func (c *RotatingKVCache) Offsets(seqIDs ...int) []int32 {
	offsets := make([]int32, len(seqIDs))
	for i, seqID := range seqIDs {
		if r, ok := c.regions[seqID]; ok {
			offsets[i] = int32(r.offset)
		}
	}
	return offsets
}

func (c *RotatingKVCache) Free() {
	mlx.Unpin(c.keys, c.values)
	c.keys, c.values = nil, nil
	for _, seqID := range c.seqOrder {
		if r, ok := c.regions[seqID]; ok {
			r.offset = 0
		}
	}
}

type rotatingSnapshot struct {
	keys, values *mlx.Array
	offset       int
}

func (s *rotatingSnapshot) Size() int {
	n := 0
	if s.keys != nil {
		n += s.keys.NumBytes()
	}
	if s.values != nil {
		n += s.values.NumBytes()
	}
	return n
}
func (s *rotatingSnapshot) Close() { mlx.Unpin(s.keys, s.values) }

// SnapshotType tags serialized rotating snapshots as "rotating" so they
// round-trip back to *rotatingSnapshot on restore. A typo here would
// mis-tag files and feed a *kvSnapshot to RotatingKVCache.Restore.
func (s *rotatingSnapshot) SnapshotType() string { return SnapshotTypeRotating }

// CollectArrays returns the snapshot's serializable fields. offset is
// recovered from the trie node's toOffset on load, so it is not written
// to the safetensors payload.
func (s *rotatingSnapshot) CollectArrays() (map[string]*mlx.Array, func()) {
	return map[string]*mlx.Array{"keys": s.keys, "values": s.values}, func() {}
}

func (c *RotatingKVCache) Snapshot(seqID int, fromOffset int) Snapshot {
	r, ok := c.regions[seqID]
	if !ok || c.keys == nil || r.offset == 0 {
		return nil
	}
	vis := c.visible(r)
	// Read ring data in logical order (oldest→newest)
	var kSlices, vSlices []*mlx.Array
	if r.offset <= c.maxSize {
		// Ring hasn't wrapped — data is contiguous from start
		kSlices = []*mlx.Array{c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start, r.start+vis), mlx.Slice())}
		vSlices = []*mlx.Array{c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start, r.start+vis), mlx.Slice())}
	} else {
		// Ring has wrapped — oldest is at (offset % maxSize), read in two chunks
		wrapPoint := r.offset % c.maxSize
		kSlices = []*mlx.Array{
			c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start+wrapPoint, r.start+c.maxSize), mlx.Slice()),
			c.keys.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start, r.start+wrapPoint), mlx.Slice()),
		}
		vSlices = []*mlx.Array{
			c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start+wrapPoint, r.start+c.maxSize), mlx.Slice()),
			c.values.Slice(mlx.Slice(), mlx.Slice(), mlx.Slice(r.start, r.start+wrapPoint), mlx.Slice()),
		}
	}

	var kData, vData *mlx.Array
	if len(kSlices) == 1 {
		kData = kSlices[0]
		vData = vSlices[0]
	} else {
		kData = kSlices[0].Concatenate(2, kSlices[1])
		vData = vSlices[0].Concatenate(2, vSlices[1])
	}

	kCopy := mlx.Contiguous(kData, false)
	vCopy := mlx.Contiguous(vData, false)
	mlx.Pin(kCopy, vCopy)
	mlx.AsyncEval(kCopy, vCopy)

	return &rotatingSnapshot{keys: kCopy, values: vCopy, offset: r.offset}
}

func (c *RotatingKVCache) Restore(seqID int, snapshot Snapshot, target int) bool {
	r, ok := c.regions[seqID]
	if !ok || target < 0 {
		return false
	}

	if snapshot == nil {
		if target > r.offset {
			return false
		}
		// Can only rewind when ring hasn't wrapped
		if r.offset > c.maxSize {
			return false
		}
		r.offset = target
		return true
	}

	// Soft type assertion: cross-type snapshot degrades to a miss, not panic.
	snap, ok := snapshot.(*rotatingSnapshot)
	if !ok {
		return false
	}
	if target > snap.offset {
		return false
	}
	// Once the ring has wrapped, only exact restore is valid —
	// rewinding would leave an inconsistent window.
	if target < snap.offset && snap.offset > c.maxSize {
		return false
	}

	if c.keys == nil {
		c.rebuild(snap.keys.DType(), snap.values.DType(), snap.keys.Dim(1), snap.keys.Dim(3), snap.values.Dim(3))
	}

	// Write snapshot data into the ring at the correct wrapped positions.
	// Snapshot is oldest→newest. Physical slot for logical position j:
	// (snap.offset - snapLen + j) % maxSize
	snapLen := snap.keys.Dim(2)
	wp := make([]int32, snapLen)
	base := snap.offset - snapLen
	for j := range snapLen {
		slot := (base + j) % c.maxSize
		if slot < 0 {
			slot += c.maxSize
		}
		wp[j] = int32(r.start + slot)
	}
	wpTensor := mlx.NewArrayInt32(wp, []int32{int32(snapLen)})
	c.keys.Set(putAlongDim2(c.keys, wpTensor, snap.keys))
	c.values.Set(putAlongDim2(c.values, wpTensor, snap.values))

	r.offset = target
	return true
}

func (c *RotatingKVCache) Merge(parent, child Snapshot) Snapshot {
	if parent != nil {
		parent.Close()
	}
	return child
}

func (c *RotatingKVCache) Split(snapshot Snapshot, at int) (Snapshot, Snapshot) {
	return nil, snapshot
}
