package format

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

const (
	Byte = 1

	KiloByte = Byte * 1000
	MegaByte = KiloByte * 1000
	GigaByte = MegaByte * 1000
	TeraByte = GigaByte * 1000

	KibiByte = Byte * 1024
	MebiByte = KibiByte * 1024
	GibiByte = MebiByte * 1024
	TebiByte = GibiByte * 1024
)

// ParseBytes parses an integer byte size with an optional binary/decimal suffix.
// Convention: bare letter (K, M, G, T) and "iB" suffix (KiB, MiB, GiB, TiB) are
// binary (powers of 2). "B" suffix (KB, MB, GB, TB) is decimal (powers of 10).
// A bare integer (including 0 and negatives) is returned as-is.
// Fractional values are not supported.
func ParseBytes(s string) (int64, error) {
	if s == "" {
		return 0, fmt.Errorf("parse byte size: empty string")
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return n, nil
	}
	i := 0
	if s[0] == '-' || s[0] == '+' {
		i = 1
	}
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 || (i == 1 && (s[0] == '-' || s[0] == '+')) {
		return 0, fmt.Errorf("parse byte size %q: missing digits", s)
	}
	n, err := strconv.ParseInt(s[:i], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parse byte size %q: %w", s, err)
	}
	var mult int64
	switch strings.ToUpper(s[i:]) {
	case "K", "KIB":
		mult = KibiByte
	case "KB":
		mult = KiloByte
	case "M", "MIB":
		mult = MebiByte
	case "MB":
		mult = MegaByte
	case "G", "GIB":
		mult = GibiByte
	case "GB":
		mult = GigaByte
	case "T", "TIB":
		mult = TebiByte
	case "TB":
		mult = TeraByte
	default:
		return 0, fmt.Errorf("parse byte size %q: unknown suffix %q", s, s[i:])
	}
	return n * mult, nil
}

func HumanBytes(b int64) string {
	var value float64
	var unit string

	switch {
	case b >= TeraByte:
		value = float64(b) / TeraByte
		unit = "TB"
	case b >= GigaByte:
		value = float64(b) / GigaByte
		unit = "GB"
	case b >= MegaByte:
		value = float64(b) / MegaByte
		unit = "MB"
	case b >= KiloByte:
		value = float64(b) / KiloByte
		unit = "KB"
	default:
		return fmt.Sprintf("%d B", b)
	}

	switch {
	case value >= 10:
		return fmt.Sprintf("%d %s", int(value), unit)
	case value != math.Trunc(value):
		return fmt.Sprintf("%.1f %s", value, unit)
	default:
		return fmt.Sprintf("%d %s", int(value), unit)
	}
}

func HumanBytes2(b uint64) string {
	switch {
	case b >= GibiByte:
		return fmt.Sprintf("%.1f GiB", float64(b)/GibiByte)
	case b >= MebiByte:
		return fmt.Sprintf("%.1f MiB", float64(b)/MebiByte)
	case b >= KibiByte:
		return fmt.Sprintf("%.1f KiB", float64(b)/KibiByte)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
