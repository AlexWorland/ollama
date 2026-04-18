# MLX KV Cache Disk Persistence — Branch Summary

> **Branch:** `cache-persisitance`
> **Commits ahead of main:** 20
> **Files changed vs main:** 18 (14 source, 4 new)
> **Net delta:** +4,623 / −490 lines

---

## Overview

This branch implements **on-disk KV cache persistence** for the MLX runner, enabling Ollama models to survive process restarts without re-running their full prefill. The feature is gated by a new environment variable `OLLAMA_KV_CACHE_DISK_MAX` (defaults: unlimited). When disabled (`=0` or unset), behavior is byte-for-byte identical to upstream main.

The implementation consists of four architectural layers:
1. **Environment configuration** — `KVCacheDiskMax()` / `KVCacheRoot()` env vars with a `parseByteSize()` helper
2. **Disk persistence** — `cache_persist.go`: content-addressed `.safetensors` files with binary header metadata, atomic writes, retry + circuit-breaker
3. **Startup rehydration** — scans the cache directory on load, rebuilds the trie skeleton from headers, restores Cold nodes onto disk-backed snapshots
4. **Eviction & lifecycle** — two-tier eviction (memory → demote Warm→Cold; disk → delete files), signal handler shutdown drain

---

## Changed Files

### New Files (2)

| File | Lines | Purpose |
|------|-------|---------|
| `x/mlxrunner/cache_persist.go` | 588 | Core persistence engine: `writeOne()`, `loadFromDisk()`, `rehydrate()`, `contentFilename()` codec, token encoding/decoding, snapshot array restoration |
| `x/mlxrunner/cache/cache.go` | +18 | Exposes `Keys()`, `Values()` on `kvSnapshot` and adds `NewKVSnapshotFromArrays()` for persistence-layer replay |

### Modified Files (12)

#### MLX Runner — Core Cache (`x/mlxrunner/`)

| File | Net Lines | Key Changes |
|------|-----------|-------------|
| `cache.go` | +268 | `newKvCache()` constructor replaces direct struct literal; `shutdown()` for writer drain; persistence fields on `kvCache`; `begin()` calls `restoreMatchedPath()` before match evaluation; `splitNode()` + `attachSnapshots()` call `scheduleWrite()`; two-tier `enforceEvictionPolicy()` → `enforceMemoryPolicy()` + `enforceDiskPolicy()` |
| `cache_test.go` | +22 | `arraySnapshot` type mirroring `cache.kvSnapshot` interface for test-layer snapshot writes; `newTestNodeWithArraySnapshot()` helper |
| `cache_persist_test.go` | 585 (new) | `writeOne()` round-trip tests, atomic rename verification; integration: rehydrate + Cold→Warm restore on cache-hit; zero-value correctness tests |
| `cache_trie.go` | +7 | Persistence fields on `trieNode`: `diskPath`, `diskSize`, `writeAttempts` |
| `cache_trie_test.go` | +13 | Zero-value assertion for new trieNode persistence fields |
| `runner.go` | +15 | `Runner.cache` → `*kvCache` pointer; `Shutdown()` method; `newKvCache(modelName, numLayers)` call in `Load()` |
| `server.go` | +16 | SIGTERM/SIGINT handler spawns goroutine calling `runner.Shutdown()`, then re-raises signal for clean exit |

#### Environment Config (`envconfig/`)

| File | Net Lines | Key Changes |
|------|-----------|-------------|
| `config.go` | +77 | `parseByteSize()` helper (K/M/G/T, KiB/MiB/GiB/TiB, KB/MB/GB/TB, bare integers); `KVCacheDiskMax()` env var reader; `KVCacheRoot()` env var reader |
| `config_test.go` | +63 | Tests for `parseByteSize()`, `KVCacheDiskMax()`, `KVCacheRoot()` including invalid values and defaults |

#### Launch / Hermes (`cmd/launch/`)

| File | Net Lines | Key Changes |
|------|-----------|-------------|
| `hermes.go` | −326 (net) | **Refactor:** removes `hermesConfigBackend` struct, `runWindows()`, `runWSL()`, `ensureInstalledWindows()`; replaces with simpler `binary()` + `os.ReadFile()`/`os.MkdirAll()`/`fileutil.WriteWithBackup()`; shorter code path for config read/write |
| `hermes_test.go` | −130 (net) | **Aligned:** removes Windows/WSL-specific test branches matching the hermes.go simplification |
| `launch.go` | +1 | Managed-single rewrite now skips when saved config already matches target model (`!savedMatchesModels(saved, ...)` condition) |
| `launch_test.go` | +35 / −35 (net ~0) | **Renamed & refactored:** tests for "repairs using saved model" → "skips rewrite when saved matches"; "configure-only repairs" → "rewrites when saved differs" |

#### Docs (`docs/`)

| File | Net Lines | Key Changes |
|------|-----------|-------------|
| `faq.mdx` | +28 | Section on context window size with `OLLAMA_CONTEXT_LENGTH` env var examples and `/set parameter num_ctx` CLI usage |

---

## Architecture at a Glance

```
envconfig/config.go                          ← OLLAMA_KV_CACHE_DISK_MAX (int64, default -1 = unlimited)
    ↓
envconfig/config.go                          ← OLLAMA_KV_CACHE_ROOT    (string, default: <models>/../cache/kv)
    ↓
x/mlxrunner/runner.go:Load()                ← newKvCache(modelDigest, numLayers)
    ↓
  ├── ensureRoot()                            → builds empty root trieNode
  ├── rehydrate()                             → scans cacheDir, rebuilds trie skeleton from .safetensors headers
  │   ├── cleanup tmp files                 → removes .tmp / .tmp.safetensors
  │   ├── read headers                      → validates model_digest + layer_count
  │   ├── build parent→children index       → BFS from roots to leaf
  │   └── set atomic diskBytes              ← total on-disk size
  └── newDiskWriter(c)                      → synchronous writer with circuit-breaker
    ↓
x/mlxrunner/cache.go:begin()                 → restoreMatchedPath() on hit, then prefill
    ↓
  ├── scheduleWrite(node)                   → writeNode(n) synchronously (disk path, atomic rename)
  └── enforceEvictionPolicy()
       ├── enforceMemoryPolicy()            → demote Warm→Cold (drop snapshots, keep disk files)
       │   └── evictNode(cand)             → legacy delete if un-persistable
       └── enforceDiskPolicy()              → remove files when diskBytes > diskMax
    ↓
x/mlxrunner/cache_persist.go                ← writeOne(), loadFromDisk(), rehydrate()
    ↓
  ├── contentFilename(modelDigest, parentHash, tokens) → SHA-256 content-addressed .safetensors
  ├── encodeHeader / decodeHeader           → binary token encoding + metadata (format version, model digest, parent hash)
  ├── writeOne(node)                        → in-memory → safetensors via atomic rename (.tmp.safetensors → final)
  ├── loadFromDisk(node)                    → read header, validate model/layer, restoreSnapshotArrays → setSnapshots()
  └── rehydrate()                           → full trie rebuild from directory scan (see flow above)
```

---

## Key Design Decisions

### 1. Synchronous writes over async goroutine
An earlier design used a background writer goroutine, but **MLX's Metal backend is not thread-safe** — concurrent calls to MLX operations hit `IOGPUMetalCommandBuffer` assertions. Since the MLX runner already serializes requests on a single goroutine, async writes add complexity with no throughput benefit. The `diskWriter` type remains as an enable/disable flag and circuit-breaker.

### 2. Content-addressed filenames
SHA-256 of `modelDigest || parentHash || tokens || layerCount || dtype` produces deterministic filenames. Same inputs always produce the same output, making writes idempotent and enabling safe cache directory scans.

### 3. Two-tier eviction
- **Memory pass:** Demotes Warm → Cold (free in-memory snapshots, keep disk files). Falls back to legacy delete for un-persistable nodes.
- **Disk pass:** When `diskBytes > diskMax`, evicts the least-recently-used Cold node from disk entirely.

### 4. Startup rehydration from headers only
On startup, `rehydrate()` scans `.safetensors` files, reads **only headers** (no array loading), and rebuilds a trie skeleton with every node in Cold state. Snapshots are materialized on first cache-hit via `loadFromDisk()`. This avoids expensive disk I/O during startup for nodes that won't be hit.

### 5. Foreign-digest safety
Files stamped with a different model's digest are left alone during rehydration, allowing concurrent model instances to coexist without corrupting each other's caches.

---

## State Machine (Node Lifecycle)

```
           Cold             Warm             Gone
         ┌──────┐       ┌────────┐       ┌───────┐
         │ diskPath  │       │ snapshots   │       │ diskPath="" │
         │ diskSize  │       │ = nil / set │       │ diskSize=0  │
         │          │       └────────┘       │           │
         │     scheduleWrite()              │
         │     loadFromDisk() ────────────→ │
         └──────┘                          └───────┘
```

- **Cold → Warm:** `loadFromDisk()` reads `.safetensors` arrays, calls `setSnapshots()` (Cold is transitional)
- **Warm → Cold:** `enforceMemoryPolicy()` drops snapshots in-memory; disk file survives
- **Any → Gone:** `enforceDiskPolicy()` removes the `.safetensors` file + resets `diskPath/diskSize`

---

## Commit History

| # | Commit | Description |
|---|--------|-------------|
| 17 | `d0c6ccef` | design: MLX runner KV cache persistence (design doc) |
| 18 | `45e3b33f` | plan: MLX runner KV cache persistence (plan/spec) |
| 19 | `58d0995f` | envconfig: add OLLAMA_KV_CACHE_DISK_MAX and OLLAMA_KV_CACHE_ROOT |
| 20 | `55786ebe` | mlxrunner: add persistence fields to trieNode and kvCache |
| 21 | `dd93432c` | mlxrunner/persist: content-addressed filename + header codec |
| 22 | `9c125d09` | mlxrunner/persist: synchronous writeOne with atomic rename |
| 23 | `8012ebda` | mlxrunner/persist: async diskWriter with shutdown drain (later simplified to sync) |
| 24 | `b2712690` | mlxrunner: hook scheduleWrite into attachSnapshots and splitNode |
| 25 | `4a408874` | mlxrunner/persist: loadFromDisk restores Cold→Warm |
| 26 | `d5975522` | mlxrunner: restore Cold nodes to Warm on cache-hit in begin() |
| 27 | `a576d8d3` | mlxrunner: two-tier enforceEvictionPolicy (memory + disk) |
| 28 | `346c6037` | mlxrunner/persist: startup rehydration from safetensors headers |
| 29 | `01564ec0` | mlxrunner: wire persistence into runner construction + shutdown |
| 30 | `2b675fd3` | mlxrunner/persist: end-to-end warm-restart integration test |
| 31 | `674b64af` | mlxrunner/persist: comprehensive debug logging on persistence flows |
| 32 | `6faf4f35` | mlxrunner/persist: fix three runtime issues caught by Apple Silicon testing |
| 33 | `44b241c4` | mlxrunner/persist: remove async writer — MLX is not thread-safe (final design) |
| * | `a50ce61c` | launch: skip unchanged managed-single rewrite (#15633) |
| * | `57653b8e` | cmd/launch: show WSL guidance on Windows instead of handing off (#15637) |

---

## Testing Summary

| Test File | Tests Added | Coverage |
|-----------|-------------|----------|
| `cache_persist_test.go` | writeOne round-trip, atomic rename; rehydrate + loadFromDisk + Cold→Warm restore on cache-hit | Persistence engine end-to-end |
| `cache_test.go` | `arraySnapshot` helper type for real MLX-backed snapshot writes | Test infrastructure |
| `cache_trie_test.go` | Zero-value correctness for persistence fields on trieNode | New field invariants |
| `config_test.go` | `parseByteSize()` binary/decimal suffixes, invalid inputs; `KVCacheDiskMax()` and `KVCacheRoot()` env overrides | Env config |

---

## Known Gaps / Risks

- **Not-tested:** Concurrent model restarts (two ollama instances reading same cache dir)
- **Not-tested:** Disk full scenarios — `enforceDiskPolicy()` may loop if the filesystem returns I/O errors silently
- **Not-tested:** Very large caches (100K+ nodes) — BFS rebuild from directory scan could be slow
- **Not-tested:** Cache corruption recovery beyond orphan deletion (e.g., partial `.safetensors` files)
- The `writeAttempts` field on `trieNode` is incremented per-call but never has a maximum cap that forces removal of corrupt entries — writes will keep being scheduled indefinitely for broken nodes
