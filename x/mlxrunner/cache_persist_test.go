package mlxrunner

import (
	"strings"
	"testing"
)

func TestFilenameDeterministic(t *testing.T) {
	f1 := contentFilename("modeldigest123", "", []int32{1, 2, 3}, 32, "bfloat16")
	f2 := contentFilename("modeldigest123", "", []int32{1, 2, 3}, 32, "bfloat16")
	if f1 != f2 {
		t.Errorf("filename not deterministic: %q vs %q", f1, f2)
	}
	if !strings.HasSuffix(f1, ".safetensors") {
		t.Errorf("filename %q missing .safetensors suffix", f1)
	}
}

func TestFilenameUniqueness(t *testing.T) {
	f1 := contentFilename("m", "", []int32{1, 2, 3}, 32, "bf16")
	f2 := contentFilename("m", "parent", []int32{1, 2, 3}, 32, "bf16")
	f3 := contentFilename("m", "", []int32{1, 2, 4}, 32, "bf16")
	f4 := contentFilename("m", "", []int32{1, 2, 3}, 32, "fp16")
	if f1 == f2 || f1 == f3 || f1 == f4 {
		t.Error("filename collisions across differing inputs")
	}
}

func TestTokensRoundTrip(t *testing.T) {
	cases := [][]int32{
		nil,
		{},
		{0},
		{-1, 0, 1, 100_000, 2_147_483_647},
	}
	for _, c := range cases {
		encoded := encodeTokens(c)
		got, err := decodeTokens(encoded)
		if err != nil {
			t.Errorf("decode(%v): %v", c, err)
			continue
		}
		if len(got) != len(c) {
			t.Errorf("len mismatch: got %d want %d", len(got), len(c))
			continue
		}
		for i := range c {
			if got[i] != c[i] {
				t.Errorf("tok[%d]: got %d want %d", i, got[i], c[i])
			}
		}
	}
}

func TestHeaderBuildParse(t *testing.T) {
	h := headerFields{
		formatVersion: "1",
		modelDigest:   "abcdef",
		parentHash:    "p1",
		tokens:        []int32{5, 6, 7},
		layerCount:    4,
		snapshotTypes: []string{"kv", "kv", "kv", "kv"},
	}
	meta := encodeHeader(h)
	if meta["cache_format_version"] != "1" {
		t.Errorf("format version missing")
	}
	back, err := decodeHeader(meta)
	if err != nil {
		t.Fatalf("decodeHeader: %v", err)
	}
	if back.modelDigest != "abcdef" || back.parentHash != "p1" || back.layerCount != 4 {
		t.Errorf("header round-trip lost fields: %+v", back)
	}
	if len(back.tokens) != 3 || back.tokens[0] != 5 {
		t.Errorf("tokens round-trip: %v", back.tokens)
	}
	if len(back.snapshotTypes) != 4 {
		t.Errorf("snapshotTypes round-trip: %v", back.snapshotTypes)
	}
}

func TestDecodeHeaderRejectsUnknownVersion(t *testing.T) {
	_, err := decodeHeader(map[string]string{"cache_format_version": "99"})
	if err == nil {
		t.Error("decodeHeader should reject unknown format version")
	}
}
