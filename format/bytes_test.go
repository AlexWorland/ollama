package format

import (
	"testing"
)

func TestHumanBytes(t *testing.T) {
	type testCase struct {
		input    int64
		expected string
	}

	tests := []testCase{
		// Test bytes (B)
		{0, "0 B"},
		{1, "1 B"},
		{999, "999 B"},

		// Test kilobytes (KB)
		{1000, "1 KB"},
		{1500, "1.5 KB"},
		{999999, "999 KB"},

		// Test megabytes (MB)
		{1000000, "1 MB"},
		{1500000, "1.5 MB"},
		{999999999, "999 MB"},

		// Test gigabytes (GB)
		{1000000000, "1 GB"},
		{1500000000, "1.5 GB"},
		{999999999999, "999 GB"},

		// Test terabytes (TB)
		{1000000000000, "1 TB"},
		{1500000000000, "1.5 TB"},
		{1999999999999, "2.0 TB"},

		// Test fractional values
		{1234, "1.2 KB"},
		{1234567, "1.2 MB"},
		{1234567890, "1.2 GB"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			result := HumanBytes(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}

func TestParseBytes(t *testing.T) {
	cases := []struct {
		in      string
		want    int64
		wantErr bool
	}{
		{"", 0, true},
		{"0", 0, false},
		{"-1", -1, false},
		{"-42", -42, false},
		{"1024", 1024, false},
		{"1K", 1024, false},
		{"1KiB", 1024, false},
		{"1KB", 1000, false},
		{"50GiB", 50 * GibiByte, false},
		{"2G", 2 * GibiByte, false},
		{"100M", 100 * MebiByte, false},
		{"1TiB", TebiByte, false},
		{"1TB", TeraByte, false},
		{"1.5G", 0, true},
		{"abc", 0, true},
		{"10XB", 0, true},
	}
	for _, c := range cases {
		got, err := ParseBytes(c.in)
		if (err != nil) != c.wantErr {
			t.Errorf("ParseBytes(%q) err=%v wantErr=%v", c.in, err, c.wantErr)
			continue
		}
		if !c.wantErr && got != c.want {
			t.Errorf("ParseBytes(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestHumanBytes2(t *testing.T) {
	type testCase struct {
		input    uint64
		expected string
	}

	tests := []testCase{
		// Test bytes (B)
		{0, "0 B"},
		{1, "1 B"},
		{1023, "1023 B"},

		// Test kibibytes (KiB)
		{1024, "1.0 KiB"},
		{1536, "1.5 KiB"},
		{1048575, "1024.0 KiB"},

		// Test mebibytes (MiB)
		{1048576, "1.0 MiB"},
		{1572864, "1.5 MiB"},
		{1073741823, "1024.0 MiB"},

		// Test gibibytes (GiB)
		{1073741824, "1.0 GiB"},
		{1610612736, "1.5 GiB"},
		{2147483648, "2.0 GiB"},
	}

	for _, tc := range tests {
		t.Run(tc.expected, func(t *testing.T) {
			result := HumanBytes2(tc.input)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}
