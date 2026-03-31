package cache

import (
	"fmt"
	"strconv"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SnapshotType identifies the concrete snapshot type for serialization.
type SnapshotType string

const (
	SnapshotTypeKV        SnapshotType = "kv"
	SnapshotTypeRotating  SnapshotType = "rotating"
	SnapshotTypeRecurrent SnapshotType = "recurrent"
)

// FieldNames returns the expected array and metadata key names for this
// snapshot type. This is the single source of truth for the serialization
// schema — used by both export (save) and import (load) paths.
func (t SnapshotType) FieldNames() (arrayNames, metaNames []string) {
	switch t {
	case SnapshotTypeKV:
		return []string{"keys", "values"}, []string{"from_offset", "to_offset"}
	case SnapshotTypeRotating:
		return []string{"keys", "values"}, []string{"from_offset", "to_offset", "idx"}
	case SnapshotTypeRecurrent:
		return []string{"conv_state", "delta_state"}, []string{"offset"}
	default:
		return nil, nil
	}
}

// SnapshotExport holds all data needed to reconstruct a snapshot from disk.
type SnapshotExport struct {
	Type     SnapshotType
	Arrays   map[string]*mlx.Array
	Metadata map[string]string
}

// ExportSnapshot extracts serializable data from a Snapshot.
// Returns nil if the snapshot is nil.
func ExportSnapshot(s Snapshot) *SnapshotExport {
	if s == nil {
		return nil
	}

	switch snap := s.(type) {
	case *kvSnapshot:
		return &SnapshotExport{
			Type: SnapshotTypeKV,
			Arrays: map[string]*mlx.Array{
				"keys":   snap.keys,
				"values": snap.values,
			},
			Metadata: map[string]string{
				"from_offset": strconv.Itoa(snap.fromOffset),
				"to_offset":   strconv.Itoa(snap.toOffset),
			},
		}

	case *rotatingSnapshot:
		return &SnapshotExport{
			Type: SnapshotTypeRotating,
			Arrays: map[string]*mlx.Array{
				"keys":   snap.keys,
				"values": snap.values,
			},
			Metadata: map[string]string{
				"from_offset": strconv.Itoa(snap.fromOffset),
				"to_offset":   strconv.Itoa(snap.toOffset),
				"idx":         strconv.Itoa(snap.idx),
			},
		}

	case *recurrentSnapshot:
		return &SnapshotExport{
			Type: SnapshotTypeRecurrent,
			Arrays: map[string]*mlx.Array{
				"conv_state":  snap.convState,
				"delta_state": snap.deltaState,
			},
			Metadata: map[string]string{
				"offset": strconv.Itoa(snap.offset),
			},
		}

	default:
		return nil
	}
}

// ImportSnapshot reconstructs a Snapshot from loaded arrays and metadata.
// The caller must have already loaded the arrays from a safetensors file.
// Imported arrays are pinned and async-evaluated to match the state
// produced by Snapshot() methods.
func ImportSnapshot(typ SnapshotType, arrays map[string]*mlx.Array, meta map[string]string) (Snapshot, error) {
	switch typ {
	case SnapshotTypeKV:
		return importKVSnapshot(arrays, meta)
	case SnapshotTypeRotating:
		return importRotatingSnapshot(arrays, meta)
	case SnapshotTypeRecurrent:
		return importRecurrentSnapshot(arrays, meta)
	default:
		return nil, fmt.Errorf("unknown snapshot type: %q", typ)
	}
}

func importKVSnapshot(arrays map[string]*mlx.Array, meta map[string]string) (Snapshot, error) {
	keys := arrays["keys"]
	values := arrays["values"]
	if keys == nil || values == nil {
		return nil, fmt.Errorf("kv snapshot missing keys or values array")
	}

	fromOffset, err := strconv.Atoi(meta["from_offset"])
	if err != nil {
		return nil, fmt.Errorf("kv snapshot invalid from_offset: %w", err)
	}
	toOffset, err := strconv.Atoi(meta["to_offset"])
	if err != nil {
		return nil, fmt.Errorf("kv snapshot invalid to_offset: %w", err)
	}

	mlx.Pin(keys, values)
	mlx.AsyncEval(keys, values)

	return &kvSnapshot{
		keys:       keys,
		values:     values,
		fromOffset: fromOffset,
		toOffset:   toOffset,
	}, nil
}

func importRotatingSnapshot(arrays map[string]*mlx.Array, meta map[string]string) (Snapshot, error) {
	keys := arrays["keys"]
	values := arrays["values"]
	if keys == nil || values == nil {
		return nil, fmt.Errorf("rotating snapshot missing keys or values array")
	}

	fromOffset, err := strconv.Atoi(meta["from_offset"])
	if err != nil {
		return nil, fmt.Errorf("rotating snapshot invalid from_offset: %w", err)
	}
	toOffset, err := strconv.Atoi(meta["to_offset"])
	if err != nil {
		return nil, fmt.Errorf("rotating snapshot invalid to_offset: %w", err)
	}
	idx, err := strconv.Atoi(meta["idx"])
	if err != nil {
		return nil, fmt.Errorf("rotating snapshot invalid idx: %w", err)
	}

	mlx.Pin(keys, values)
	mlx.AsyncEval(keys, values)

	return &rotatingSnapshot{
		kvSnapshot: kvSnapshot{
			keys:       keys,
			values:     values,
			fromOffset: fromOffset,
			toOffset:   toOffset,
		},
		idx: idx,
	}, nil
}

func importRecurrentSnapshot(arrays map[string]*mlx.Array, meta map[string]string) (Snapshot, error) {
	convState := arrays["conv_state"]
	deltaState := arrays["delta_state"]
	if convState == nil || deltaState == nil {
		return nil, fmt.Errorf("recurrent snapshot missing conv_state or delta_state array")
	}

	offset, err := strconv.Atoi(meta["offset"])
	if err != nil {
		return nil, fmt.Errorf("recurrent snapshot invalid offset: %w", err)
	}

	mlx.Pin(convState, deltaState)
	mlx.AsyncEval(convState, deltaState)

	return &recurrentSnapshot{
		convState:  convState,
		deltaState: deltaState,
		offset:     offset,
	}, nil
}
