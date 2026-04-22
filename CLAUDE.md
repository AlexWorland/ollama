# Ollama (cache-persisitance branch) — Project Notes

## Stack
- **Language:** Go (primary), with MLX backend on Apple Silicon for `x/imagegen` and `x/mlxrunner`.
- **Build:** `./scripts/build_darwin.sh` for the macOS binary; `go build ./...` for everything.
- **Test:** `go test ./...`. MLX-dependent packages require `libmlxc.dylib` — `TestMain` should `InitMLX()` and skip gracefully if absent (see `x/imagegen/mlx/mlx_test.go` or `x/imagegen/nn/nn_test.go` for the pattern).

## Workflow Conventions
- After modifying `scheduler.go`, `server.go`, or VAE / precision-sensitive code: rebuild the `ollama` binary and run regression tests before committing.
- For `/simplify` passes: run in a loop until convergence, then commit and push without intermediate commentary.
- The Ollama daemon listens on `localhost:11434`. Sub-agents cannot reach it (sandbox blocks loopback TCP) — run any daemon checks from the primary session.

## Active Work
- Branch: `cache-persisitance` — KV cache persistence work (see `~/.claude/projects/-Users-alexworland-git-repos-ollama/memory/project_mlx_cache_persistence.md`).
