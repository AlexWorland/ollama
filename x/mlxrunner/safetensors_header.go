package mlxrunner

import (
	"fmt"

	"github.com/ollama/ollama/x/safetensors"
)

// parseSafetensorsMetadata reads the user-supplied __metadata__ map from a
// safetensors file without loading any tensor data. Used on the rehydrate
// hot path at startup to avoid materializing every tensor through MLX just
// to read a handful of header strings.
//
// Thin wrapper over safetensors.ReadMetadata that preserves the strict
// "missing __metadata__ is an error" contract this package requires — we
// always write metadata, so its absence indicates a corrupt file.
func parseSafetensorsMetadata(path string) (map[string]string, error) {
	meta, err := safetensors.ReadMetadata(path)
	if err != nil {
		return nil, err
	}
	if len(meta) == 0 {
		return nil, fmt.Errorf("missing __metadata__ in safetensors header")
	}
	return meta, nil
}
