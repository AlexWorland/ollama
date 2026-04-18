package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// MaxHeaderBytes caps the JSON header safetensors readers in this package
// will accept. Legitimate model and cache headers are well under this limit;
// a larger value usually means a corrupt or hostile file asking the reader
// to allocate gigabytes.
const MaxHeaderBytes = 1 << 20 // 1 MiB

// ReadMetadata returns the user-supplied __metadata__ string-to-string map
// from a safetensors file without loading any tensor data. The safetensors
// format begins with an 8-byte little-endian u64 header length followed by
// a JSON blob whose top-level "__metadata__" key holds the map.
//
// A file that parses cleanly but contains no __metadata__ key returns an
// empty map and nil error; callers needing to distinguish "absent" from
// "empty" should verify their own invariants after the call.
func ReadMetadata(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerLen uint64
	if err := binary.Read(f, binary.LittleEndian, &headerLen); err != nil {
		return nil, fmt.Errorf("read header length: %w", err)
	}
	if headerLen == 0 {
		return nil, fmt.Errorf("zero-length header")
	}
	if headerLen > MaxHeaderBytes {
		return nil, fmt.Errorf("header length %d exceeds %d-byte limit", headerLen, MaxHeaderBytes)
	}

	header := make([]byte, headerLen)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var parsed struct {
		Metadata map[string]string `json:"__metadata__"`
	}
	if err := json.Unmarshal(header, &parsed); err != nil {
		return nil, fmt.Errorf("parse header json: %w", err)
	}
	if parsed.Metadata == nil {
		return map[string]string{}, nil
	}
	return parsed.Metadata, nil
}
