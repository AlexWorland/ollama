package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SerializableSnapshot is a Snapshot whose backing arrays can be written to
// and read back from a safetensors file. Snapshot types that don't implement
// this interface are skipped by the persistence layer (the writer refuses to
// persist them — partial persistence would break restore semantics).
type SerializableSnapshot interface {
	Snapshot
	// SnapshotType returns the tag stored in the safetensors snapshot_types
	// header field, used at load time to pick the right constructor.
	SnapshotType() string
	// CollectArrays returns the named arrays to persist. The returned cleanup
	// function must be called after serialization to release any temporaries.
	CollectArrays() (map[string]*mlx.Array, func())
	// FromOffset / ToOffset bracket the persisted range.
	FromOffset() int
	ToOffset() int
}

// SnapshotType* are the canonical tags stored in safetensors snapshot_types
// header fields. Reading a snapshot whose type isn't one of these falls back
// to SnapshotTypeUnknown, which callers handle as a miss.
const (
	SnapshotTypeKV        = "kv"
	SnapshotTypeRecurrent = "recurrent"
	SnapshotTypeRotating  = "rotating"
	SnapshotTypeEmpty     = "empty"
	SnapshotTypeUnknown   = "unknown"
)

// snapshotFieldNames is the per-type array field layout for safetensors.
// Post-multi-seq, rotating snapshots store offset as header metadata rather
// than as an `idx` array, so the field list matches kv.
var snapshotFieldNames = map[string][]string{
	SnapshotTypeKV:        {"keys", "values"},
	SnapshotTypeRecurrent: {"conv", "delta"},
	SnapshotTypeRotating:  {"keys", "values"},
	SnapshotTypeEmpty:     {},
}

// SnapshotFieldNames returns the per-type field names a SerializableSnapshot
// serializes. Unknown types return (nil, false).
func SnapshotFieldNames(snapshotType string) (names []string, ok bool) {
	names, ok = snapshotFieldNames[snapshotType]
	return
}

// NewSnapshotFromArrays reconstructs a snapshot of the given type from loaded
// safetensors arrays and offset metadata. Rotating snapshots ignore toOffset;
// they reconstruct from offset stored in the header via fromOffset.
func NewSnapshotFromArrays(snapshotType string, fields map[string]*mlx.Array, fromOffset, toOffset int) (Snapshot, error) {
	switch snapshotType {
	case SnapshotTypeEmpty:
		return nil, nil
	case SnapshotTypeKV:
		keys, kOK := fields["keys"]
		values, vOK := fields["values"]
		if !kOK || !vOK {
			return nil, fmt.Errorf("kv snapshot: missing keys/values")
		}
		return &kvSnapshot{keys: keys, values: values, fromOffset: fromOffset, toOffset: toOffset}, nil
	case SnapshotTypeRecurrent:
		conv, cOK := fields["conv"]
		delta, dOK := fields["delta"]
		if !cOK || !dOK {
			return nil, fmt.Errorf("recurrent snapshot: missing conv/delta")
		}
		return &recurrentSnapshot{convState: conv, deltaState: delta, offset: toOffset}, nil
	case SnapshotTypeRotating:
		keys, kOK := fields["keys"]
		values, vOK := fields["values"]
		if !kOK || !vOK {
			return nil, fmt.Errorf("rotating snapshot: missing keys/values")
		}
		return &rotatingSnapshot{keys: keys, values: values, offset: toOffset}, nil
	default:
		return nil, fmt.Errorf("unknown snapshot type %q", snapshotType)
	}
}

// --- SnapshotType() methods on existing types -------------------------------

func (s *kvSnapshot) SnapshotType() string        { return SnapshotTypeKV }
func (s *rotatingSnapshot) SnapshotType() string  { return SnapshotTypeRotating }
func (s *recurrentSnapshot) SnapshotType() string { return SnapshotTypeRecurrent }

// --- CollectArrays() methods -----------------------------------------------

func (s *kvSnapshot) CollectArrays() (map[string]*mlx.Array, func()) {
	return map[string]*mlx.Array{"keys": s.keys, "values": s.values}, func() {}
}

func (s *rotatingSnapshot) CollectArrays() (map[string]*mlx.Array, func()) {
	return map[string]*mlx.Array{"keys": s.keys, "values": s.values}, func() {}
}

func (s *recurrentSnapshot) CollectArrays() (map[string]*mlx.Array, func()) {
	return map[string]*mlx.Array{"conv": s.convState, "delta": s.deltaState}, func() {}
}

// --- FromOffset / ToOffset accessors ---------------------------------------

func (s *kvSnapshot) FromOffset() int        { return s.fromOffset }
func (s *kvSnapshot) ToOffset() int          { return s.toOffset }
func (s *rotatingSnapshot) FromOffset() int  { return 0 }
func (s *rotatingSnapshot) ToOffset() int    { return s.offset }
func (s *recurrentSnapshot) FromOffset() int { return 0 }
func (s *recurrentSnapshot) ToOffset() int   { return s.offset }
