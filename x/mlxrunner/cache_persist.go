// Package mlxrunner — KV cache disk persistence: filename and header helpers.
// Design: .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md §10.
package mlxrunner

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const cacheFormatVersion = "1"

// diskWriter is the writer goroutine handle on kvCache.
// The full implementation lands in Task 5.
type diskWriter struct{}

// contentFilename returns a deterministic .safetensors filename for a node.
// Same inputs always produce the same output so writes are idempotent
// and parent references in other files are stable.
func contentFilename(modelDigest, parentHash string, tokens []int32, layerCount int, dtype string) string {
	h := sha256.New()
	h.Write([]byte(modelDigest))
	h.Write([]byte{0})
	h.Write([]byte(parentHash))
	h.Write([]byte{0})
	h.Write(int32BytesLE(tokens))
	h.Write([]byte{0})
	fmt.Fprintf(h, "%d", layerCount)
	h.Write([]byte{0})
	h.Write([]byte(dtype))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum[:16]) + ".safetensors"
}

// encodeTokens packs int32 tokens in little-endian bytes and base64-encodes them.
func encodeTokens(tokens []int32) string {
	if len(tokens) == 0 {
		return ""
	}
	return base64.StdEncoding.EncodeToString(int32BytesLE(tokens))
}

func decodeTokens(s string) ([]int32, error) {
	if s == "" {
		return nil, nil
	}
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("decode tokens: %w", err)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("decode tokens: length %d not a multiple of 4", len(raw))
	}
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4 : i*4+4]))
	}
	return out, nil
}

func int32BytesLE(tokens []int32) []byte {
	buf := make([]byte, 4*len(tokens))
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(buf[i*4:i*4+4], uint32(t))
	}
	return buf
}

// headerFields is the metadata we embed in every safetensors file.
type headerFields struct {
	formatVersion string
	modelDigest   string
	parentHash    string
	tokens        []int32
	layerCount    int
	snapshotTypes []string
	createdAt     time.Time
}

func encodeHeader(h headerFields) map[string]string {
	v := h.formatVersion
	if v == "" {
		v = cacheFormatVersion
	}
	ts := h.createdAt
	if ts.IsZero() {
		ts = time.Now().UTC()
	}
	return map[string]string{
		"cache_format_version": v,
		"model_digest":         h.modelDigest,
		"parent_hash":          h.parentHash,
		"tokens":               encodeTokens(h.tokens),
		"layer_count":          strconv.Itoa(h.layerCount),
		"snapshot_types":       strings.Join(h.snapshotTypes, ","),
		"created_at":           ts.Format(time.RFC3339),
	}
}

func decodeHeader(m map[string]string) (headerFields, error) {
	v := m["cache_format_version"]
	if v != cacheFormatVersion {
		return headerFields{}, fmt.Errorf("unsupported cache_format_version %q", v)
	}
	lc, err := strconv.Atoi(m["layer_count"])
	if err != nil {
		return headerFields{}, fmt.Errorf("layer_count: %w", err)
	}
	toks, err := decodeTokens(m["tokens"])
	if err != nil {
		return headerFields{}, err
	}
	var snapTypes []string
	if s := m["snapshot_types"]; s != "" {
		snapTypes = strings.Split(s, ",")
	}
	var ts time.Time
	if s := m["created_at"]; s != "" {
		ts, _ = time.Parse(time.RFC3339, s) // tolerate malformed timestamps
	}
	return headerFields{
		formatVersion: v,
		modelDigest:   m["model_digest"],
		parentHash:    m["parent_hash"],
		tokens:        toks,
		layerCount:    lc,
		snapshotTypes: snapTypes,
		createdAt:     ts,
	}, nil
}
