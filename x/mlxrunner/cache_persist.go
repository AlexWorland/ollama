// Package mlxrunner — KV cache disk persistence.
// Design: .planning/specs/2026-04-16-mlx-kv-cache-persistence-design.md
// Real implementation arrives in Task 3 onward; this file reserves the name.
package mlxrunner

// diskWriter is the writer goroutine handle on kvCache.
// The full implementation lands in Task 5.
type diskWriter struct{}
