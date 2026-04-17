# MLX Runner KV Cache Persistence — Design Spec

- **Date:** 2026-04-16
- **Branch:** `cache-persisitance` (worktree; base = upstream main @ `57653b8e`)
- **Scope:** `x/mlxrunner/` only. Does not affect the ggml runner or its `kvcache/` package at the repo root.
- **Status:** Approved for planning.

---

## 1. Problem

Today's MLX runner KV cache (`x/mlxrunner/cache.go`) maintains a compressed prefix trie of per-layer K/V snapshots alongside a single "active path" of live MLX arrays. When paged-out snapshot memory exceeds a fixed threshold (`maxPagedOutBytes = 8 GiB`, `cache.go:32`), `enforceEvictionPolicy` **deletes** evicted nodes entirely. Any subsequent request that would have matched an evicted prefix must re-prefill from an earlier ancestor or from scratch. Re-prefill is orders of magnitude more expensive than restoring a saved snapshot, so repeated long-context conversations incur a large and avoidable TTFT cost.

## 2. Goals

1. **Once a node is computed, it can be reloaded later** (the primary ask) — even after eviction from RAM and even after ollama restarts.
2. Cache hits on previously-evicted prefixes skip re-prefill; restore is bounded by disk read cost, not compute cost.
3. The feature is **opt-in-friendly**: ships with unlimited disk by default (best cache-hit behavior), with an environment knob to cap or disable. A misconfigured filesystem never crashes the runner.
4. Warm-restart: a fresh ollama server can pick up caches produced by the previous session for the same model digest.
5. **Reuse existing mechanisms wherever they exist**: the trie traversal, eviction selector, snapshot type system, and safetensors I/O are all already in the codebase and should not be reinvented.

## 3. Non-goals

- Cross-node / distributed cache. Single local host only.
- Compression beyond what MLX safetensors already provides.
- Visualization, dashboard, or observability UI — that lives on the separate `feature/kv-cache-dashboard` branch.
- The ggml runner's `kvcache/` package. Out of scope.
- A new serialization format. Use `mlx.SaveSafetensorsWithMetadata` / `mlx.LoadSafetensorsNative` (already in `x/mlxrunner/mlx/io.go`). A streaming variant may be added later only if measured peak memory during save is prohibitive.
- Model-lifecycle GC (cleaning the cache dir when a model is `ollama rm`'d). Ollama's existing model-GC path is the correct home; out of scope here.

## 4. Requirements

### Functional

| # | Requirement |
|---|---|
| F1 | Every node that receives snapshots (`attachSnapshots` or `splitNode` call site) is scheduled to be written to disk. |
| F2 | Writes are atomic on successful completion (POSIX `rename(tmp, final)`). Partial files are never visible under their final name. |
| F3 | On prefix-match, any matched node that is disk-only is synchronously loaded back into memory before prefill proceeds. |
| F4 | On graceful shutdown (SIGTERM from the ollama server), the writer drains its pending queue up to a 15-second timeout before exit. |
| F5 | On startup, each model cache directory is scanned; the trie skeleton is rebuilt from safetensors headers without loading any MLX arrays. |
| F6 | Memory eviction never drops a node whose write is still in flight. |
| F7 | Disk eviction deletes files and removes trie nodes when `diskBytes` exceeds a user-set positive cap. Negative or unset cap means unlimited. |

### Operational

| # | Requirement |
|---|---|
| O1 | Disabling the feature (`OLLAMA_KV_CACHE_DISK_MAX=0`) produces byte-for-byte identical runtime behavior to upstream main: no writer goroutine, no I/O, no bookkeeping cost. |
| O2 | A persistent I/O error (permissions, filesystem fault) degrades to "log-and-continue". The runner never panics on disk problems. |
| O3 | A crash at any point during a write (kill -9, power loss, panic) leaves the cache directory in a state that the next startup can clean up without data loss to unrelated nodes. |
| O4 | Cache files from a prior model digest are not loaded by a runner serving a different digest. |

## 5. Architecture

### 5.1 Node state model

Every `trieNode` is in exactly one of four states, observable from its fields:

| State | `snapshots` | `diskPath` | On `activePath`? |
|---|---|---|---|
| **Active** | nil | "" | yes |
| **Warm** | non-nil | non-"" | no |
| **Cold** | nil | non-"" | no |
| **Gone** | — | — | removed from trie |

Transitions:

- `Active → Warm`: `attachSnapshots` attaches snapshots and fires `scheduleWrite`. Write completes → `diskPath` set.
- `Warm → Cold`: memory eviction pass drops in-memory snapshots once write has landed.
- `Cold → Warm`: prefix match hit calls `loadFromDisk` before prefill.
- `Warm|Cold → Gone`: disk eviction pass deletes file and removes the node from the trie.

Invariants (preserved from base code, formalized here):

- A node on `activePath` never has snapshots attached and never has a disk file.
- `inflightWrite != nil` blocks the Warm → Cold transition for that node.
- Compressed-trie invariants (`sibling edges never share prefix`, `all layers at same offset`, `one active path`) are untouched by this feature.

### 5.2 Data model additions

**`trieNode`** (`x/mlxrunner/cache_trie.go`) — four new fields:

```go
diskPath       string         // "" if not persisted
diskSize       int64          // 0 unless written; authoritative for disk accounting
inflightWrite  chan struct{}  // non-nil while writer holds this node; closed on write completion
writeAttempts  uint8          // retry counter; capped at 3
```

**`kvCache`** (`x/mlxrunner/cache.go`) — new fields:

```go
diskBytes    int64        // sum of diskSize across all nodes in this cache
modelDigest  string       // passed in at construction; identifies the model
cacheDir     string       // <OLLAMA_KV_CACHE_ROOT>/<modelDigest>
writer       *diskWriter  // nil when feature is disabled (OLLAMA_KV_CACHE_DISK_MAX=0)
```

**`diskWriter`** (`x/mlxrunner/cache_persist.go`, new file):

```go
type diskWriter struct {
    mu      sync.Mutex
    cond    *sync.Cond          // signals loop when pending grows or stopped flips
    pending []*trieNode         // FIFO; guarded by mu
    stopped bool
    done    chan struct{}       // closed when loop exits
    cache   *kvCache            // back-reference for diskBytes updates
}
```

Queue implementation uses a mutex-guarded slice, not a channel. The prior implementation found channel-based queues hit a shutdown deadlock and a goroutine leak (fix landed as commit `4f6e3111`, "simplify disk writer — fix deadlock + goroutine leak").

## 6. Write pipeline

### 6.1 Trigger points

The main goroutine calls `scheduleWrite(node)` exactly when a node transitions from Active to Warm:

- In `attachSnapshots` (`cache.go:382`), after snapshots are set on the frontier.
- In `splitNode` (`cache_trie.go:176`), after the new interior node's snapshots are populated by `Split`.

`scheduleWrite` is idempotent: it's a no-op if `inflightWrite != nil` or `diskPath != ""`.

### 6.2 Writer loop

Single goroutine. On each iteration:

1. Wait on `cond` until `pending` is non-empty or `stopped` is true.
2. Pop front of `pending`. If `stopped && pending == 0`, exit.
3. Call `writeOne(node)`.
4. Close `node.inflightWrite`; set `node.inflightWrite = nil`.
5. Loop.

`writeOne` steps:

1. **Batch-evaluate** all MLX arrays across the node's layers in a single `mlx.Eval` call. Memory ID 19129 established that per-node `Eval` inside an eviction loop prevents graph fusion; the batch form cuts kernel launch overhead. (Retained here because writer pipeline is hot.)
2. **Collect arrays and build metadata** (see §10.2 for header format).
3. **Compute filename** = `hex(sha256(model_digest ‖ parent_hash ‖ tokens_bytes ‖ layer_count ‖ dtype))[:32] + ".safetensors"`.
4. **Atomic write**:
   - `mlx.SaveSafetensorsWithMetadata(cacheDir + "/" + filename + ".tmp", arrays, meta)`
   - `os.Rename(tmp, cacheDir + "/" + filename)`
5. **Update counters**: `node.diskPath = final; node.diskSize = fileSize` are written by the writer before it closes `node.inflightWrite`; the main goroutine must not read these fields without first observing `inflightWrite == nil`, which the channel close guarantees (Go memory model: close-before-receive-of-zero-value). `cache.diskBytes` is updated under the same `diskWriter.mu` used to guard `pending`; all main-goroutine reads of `diskBytes` for policy decisions take the same mutex.
6. **Enforce disk policy** (may evict old nodes down to the cap).

**No fsync.** KV caches are reconstructable from re-prefill; the correctness guarantee is "any file visible under its final name is a complete snapshot", provided by atomic rename. Power-loss durability is explicitly out of scope.

### 6.3 Write errors

On I/O error from `SaveSafetensorsWithMetadata` or `Rename`:

- `node.writeAttempts++`
- `node.inflightWrite` is closed and nilled (the node returns to the regular Warm pool, still un-persisted).
- Log at `slog.Warn` with path and error.
- If `writeAttempts >= 3`, the node is marked non-retryable: `scheduleWrite` short-circuits on it. The node stays in memory as Warm-without-disk; memory eviction will eventually delete it like in upstream main.
- The writer tracks a separate `consecutiveFailures` counter that resets to zero on any successful write. If it reaches 5, the writer disables itself for the lifetime of this cache instance (`stopped = true`, `pending` drained without writes), after logging once. No per-error classification is performed — this is a simple circuit breaker for pathological filesystems (read-only mount, permission issues, disk removed).

### 6.4 Shutdown

Triggered by the existing `llmServer.Close` SIGTERM→timeout ladder (memory ID 19082). On shutdown:

1. Runner invokes `cache.writer.shutdown()`.
2. `shutdown` sets `stopped = true`, broadcasts on `cond`.
3. Main goroutine waits on `done` (or a timer, max 15 seconds — matches the existing shutdown budget).
4. On timeout: log the count of un-drained nodes; proceed to exit. Orphan `.tmp` files will be cleaned on next startup.

## 7. Restore pipeline

Restoration is driven by the match path in `begin()` (`cache.go:89`). After `findBestMatch` produces the matched ancestor path, a new loop runs before the existing `switchToPath`:

```
for _, node := range matched:
    if node.snapshots != nil:
        continue                  // Warm or Active — usable as-is
    if node.diskPath == "":
        break                     // Gone ancestor — cannot restore from here
    err := cache.loadFromDisk(node)
    if err != nil:
        slog.Warn(...)
        node.diskPath = ""
        node.diskSize = 0
        break                     // treat as Gone from this point
    // node is now Warm
```

`loadFromDisk`:

1. `sf := mlx.LoadSafetensorsNative(node.diskPath)` — mmap-backed, zero-copy on Apple unified memory.
2. Validate header: `cache_format_version`, `model_digest` must match the runner's model. On mismatch: return error (caller treats as load failure).
3. Reconstruct `[]cache.Snapshot` from the file's arrays, one per layer, using `cache.SnapshotType` tags encoded in metadata. The existing `cache.Cache.Restore` contract still applies downstream.
4. `node.setSnapshots(snaps, &c.pagedOutBytes)` — this is the existing helper, preserves the `pagedOutBytes` counter invariant.
5. `node.lastUsed = time.Now()` — LRU bump.

After the loop, existing `switchToPath` logic takes over; it calls `Restore` on each layer to move KV state from snapshots into the live MLX cache. This side of the pipeline is unchanged from upstream.

## 8. Startup rehydration

Invoked once during `kvCache` construction, gated on `OLLAMA_KV_CACHE_DISK_MAX != 0`:

1. **Ensure `cacheDir` exists**; create if missing.
2. **Scan `cacheDir`** for files matching `*.safetensors` and `*.tmp`.
   - `.tmp` files are unconditionally deleted (crashed writes).
3. **Read headers only** from each `.safetensors` file (mmap + parse metadata, no array load):
   - Validate `cache_format_version`. Unknown → skip, log warn.
   - Validate `model_digest` matches the runner's. Mismatch → skip (this dir is right but a prior model stamp leaked — defensive).
   - Extract `parent_hash`, `tokens`, `layer_count`, `snapshot_types`.
4. **Topological sort** by `parent_hash` chains:
   - Nodes with `parent_hash == ""` become children of the runtime root.
   - Orphaned nodes (parent hash present but not found in scan) are unreachable → delete files, log count.
5. **Build `trieNode` skeleton**: every persisted node has `snapshots == nil`, `diskPath` set, `diskSize` populated from `os.Stat`. Edge tokens and parent/child links restore compressed-trie shape.
6. **Update counters**: `diskBytes = sum(diskSize)`.

Startup cost is dominated by header reads: O(n) ≈ a few ms for up to ~1000 nodes on SSD. No MLX array loads happen on startup — those are lazy via `loadFromDisk`.

## 9. Eviction policy

The existing `enforceEvictionPolicy` selector (`cache.go:463`) is **unchanged in its selection rule**: oldest `lastUsed`, then deepest `endOffset`, then largest `snapshotBytes`, with the exclusion `root || activePath || multi-child`. The function becomes two passes:

```go
func (c *kvCache) enforceEvictionPolicy() {
    c.enforceMemoryPolicy()
    c.enforceDiskPolicy()
}
```

**Memory pass** (runs when `pagedOutBytes > maxPagedOutBytes`):

For each selected candidate:
- If `candidate.inflightWrite != nil`: skip, move on to next candidate this round. Never lose an un-persisted node.
- If `candidate.diskPath != ""` (Warm): drop snapshots (call `Close` on each, nil the slice), subtract from `pagedOutBytes`. Node becomes Cold.
- If `candidate.diskPath == ""` and `candidate.writeAttempts < 3`: call `scheduleWrite`, skip this round (will re-evaluate after write completes).
- If `candidate.diskPath == ""` and `candidate.writeAttempts >= 3` (write permanently failed): fall through to the legacy delete behavior — call `removeNode`.

**Disk pass** (runs when `c.diskMax > 0 && c.diskBytes > c.diskMax`):

Selector is the same, but target set is `Warm | Cold` leaves. For each candidate:
- Delete the file (ignore ENOENT).
- Remove the node from the trie (existing `removeNode`).
- Subtract `diskSize` from `diskBytes`.

Disk pass is a no-op when the cap is negative or unset — which is the default.

## 10. On-disk format

### 10.1 Directory layout

```
$OLLAMA_KV_CACHE_ROOT/
└── <model_digest>/
    ├── <sha256-hash-32chars>.safetensors
    ├── <sha256-hash-32chars>.safetensors
    └── …
```

No index file. Parent/child relationships are self-describing via header metadata.

### 10.2 Safetensors header metadata (required keys)

| Key | Type | Description |
|---|---|---|
| `cache_format_version` | string | `"1"`. Any other value → skip file. |
| `model_digest` | string | Runner's current model digest. Must match. |
| `parent_hash` | string | Empty for root-level nodes, else parent file's hash (without `.safetensors` suffix). |
| `tokens` | string | Base64 of int32-packed edge tokens. |
| `layer_count` | string | Decimal string; cross-check against array count. |
| `snapshot_types` | string | Comma-joined `cache.SnapshotType` tags, one per layer. Required for `Restore` dispatch. |
| `created_at` | string | RFC 3339 UTC timestamp. Informational only. |

### 10.3 Array naming inside the safetensors file

`layer_<idx>_<field>` — mirrors the convention the prior branch used in `extractLayerData` (e.g., `layer_0_keys`, `layer_0_values`). The set of fields is determined by the `cache.SnapshotType` for each layer:

- **Paged-out K/V** (`kvSnapshot`): `keys`, `values`.
- **Recurrent**, **rotating-multiturn**, and any future snapshot type: each type defines its own required array names, declared alongside its `Snapshot` implementation in `x/mlxrunner/cache/`.

Writer and loader both dispatch by the `snapshot_types` header field: one `SnapshotType` token per layer, comma-joined, identifying which fields to write/read for that layer. Unknown snapshot-type tokens on load cause the file to be treated as corrupt (skip + warn).

## 11. Configuration

```
OLLAMA_KV_CACHE_DISK_MAX   default: -1
   Values: negative or unset → unlimited (may grow until user caps)
           0                  → persistence fully disabled
           positive bytes     → hard cap; triggers disk eviction when exceeded
   Accepts byte-unit suffixes (e.g., "50GiB", "100M").

OLLAMA_KV_CACHE_ROOT       default: $OLLAMA_MODELS/../cache/kv
   Root directory; one subdirectory per model digest is created underneath.
   Must be on a filesystem that supports atomic rename.
```

These are registered in `envconfig/config.go` alongside existing model-related env vars, and documented in `docs/faq.mdx` under an "Environment variables" section.

## 12. Files touched

| File | New? | Purpose | Approx LOC |
|---|---|---|---|
| `x/mlxrunner/cache_persist.go` | new | `diskWriter`, `writeOne`, `loadFromDisk`, orphan GC, startup rehydration | ~300 |
| `x/mlxrunner/cache.go` | modify | add counters, call `scheduleWrite` from `attachSnapshots`, load path in `begin`, two-pass `enforceEvictionPolicy`, shutdown hook | ~80 delta |
| `x/mlxrunner/cache_trie.go` | modify | add `diskPath`, `diskSize`, `inflightWrite`, `writeAttempts` to `trieNode` | ~20 delta |
| `x/mlxrunner/cache_persist_test.go` | new | write/read/restart/crash/error tests | ~400 |
| `x/mlxrunner/runner.go` | modify | pass `modelDigest` and cache root into `kvCache` init | ~10 delta |
| `x/mlxrunner/server.go` | modify | invoke `cache.writer.shutdown()` from graceful-stop | ~10 delta |
| `envconfig/config.go` | modify | register `OLLAMA_KV_CACHE_DISK_MAX`, `OLLAMA_KV_CACHE_ROOT` | ~15 delta |
| `docs/faq.mdx` | modify | document the two env vars | ~30 |

Total: ~865 lines added, ~135 modified. Prior branch implemented this functionality in ~2,500 lines of new code; the reduction is mostly from reusing the existing eviction selector unchanged, using the existing `mlx/io.go` safetensors support unchanged, and dropping the sidecar-index rehydration path in favor of self-describing files.

## 13. Testing strategy

1. **Writer unit tests** — enqueue N nodes, assert FIFO ordering, assert `inflightWrite` closes exactly once per node, assert shutdown-drain with both "clean" and "timeout" paths, assert write-error retry and permanent-failure paths.
2. **Round-trip unit tests** — write a node, load it back, assert snapshot bytes are bit-identical. Cover all supported snapshot types (paged-out K/V, rotating, recurrent). Adapt fixtures from the prior branch's `trie_persist_test.go` (~1,199 lines; reuse test vectors, not structure).
3. **Startup rehydration tests** — write several nodes with known topology, tear down, rehydrate in a fresh `kvCache`, assert trie skeleton matches. Include cases with `.tmp` orphans, missing parent hashes, and mismatched model digests.
4. **End-to-end integration** — spin up a runner, send a prompt, stop, start a new runner, send the same prompt, assert zero prefill tokens (all matched from disk).
5. **Crash injection** — in-test kill of the writer mid-`SaveSafetensorsWithMetadata` (simulate via fault-injected `os.Rename`); restart; assert cleanup correctness.
6. **Disk-full simulation** — set cap to 1 MiB, generate overflow, assert disk pass evicts correct leaves and `diskBytes` tracks accurately.
7. **Feature-off regression** — with `OLLAMA_KV_CACHE_DISK_MAX=0`, assert zero writer goroutines and zero I/O (cross-check via `runtime.NumGoroutine` and a syscall tracer).

## 14. Explicit follow-ups (out of scope v1)

- Soft warning on unbounded growth past some implicit threshold.
- Streaming variant of `SaveSafetensorsWithMetadata` if peak memory during save is measured to be prohibitive.
- Automatic cleanup of cache directories for models no longer in the library (hand-off to ollama model GC).
- Compaction / defragmentation of orphaned nodes after heavy mutation.
- Per-model disk caps (currently only the global cap).
- Prefetching sibling branches on match to warm likely next turns.
