# Conflict Resolution Plan: `origin/jessegross/batching` → `cache-persisitance`

_Generated 2026-04-22 from merge-tree dry-run. Merge base: `4bc27280`. Incoming tip: `2beb5445`. Our tip: `2bb68d1d`._

**Scope:** 25 conflict markers across 8 files. 4 model files, 3 test/support files, and 2 new files auto-merge or adopt cleanly.

**Budget governor:** ≤ 20 net-added lines inside their files combined (see `MERGE_PLAN_BATCHING.md` §0). All per-file resolutions below record their inside-their-files impact.

## Summary Table

| File | Conflicts | Resolution mode | Inside-theirs lines | Notes |
|---|---:|---|---:|---|
| `x/mlxrunner/mlx/array.go` | 3 | Take theirs | 0 | Thread-safety edits + DType check removals — all accepted |
| `x/mlxrunner/mlx/ops_extra.go` | 2 | Take theirs | 0 | New SiLU, multi-seq `RoPEWithBase` signature |
| `x/mlxrunner/cache/recurrent.go` | 2 | Take theirs; serialization in new sibling | 0 | `recurrentSnapshot` methods move to `cache/serializable.go` |
| `x/mlxrunner/cache/cache.go` | 7 | Take theirs; our soft type-asserts wrapped via `recover()` at call-site | 0 | Largest conflict count but purely a "take theirs" file |
| `x/mlxrunner/pipeline.go` | 4 | Take theirs wholesale | 0 | `NumPredict` → `MaxTokens` rename; our `TextGenerationPipeline` is obsolete |
| `x/mlxrunner/runner.go` | 2 | Take theirs; shutdown + persistence init via new `runner_persist.go` | 1 | Our goroutine loop is replaced by `sched.run(ctx)`; one added init line |
| `x/mlxrunner/server.go` | 1 | Take theirs; our richer sampler API adapts in a new `server_debug.go` | 1 | One added line to register debug handlers; Sampler struct→positional API requires caller adapt |
| `x/mlxrunner/cache.go` | 4 | Take theirs baseline; persistence hooks re-route through new sibling files | 4 | Eviction two-tier logic lands in `cache_persist_policy.go`, called via one hook line in `evictNode` |

**Projected inside-theirs total: 6 lines.** Well under the 20-line ceiling.

**Auto-merged (no action):** `cache_test.go`, `cache_trie_test.go`, `mlx/fast.go`, 4 model files (`llama`, `qwen3`, `qwen3_5`, `glm4_moe_lite`).

**New files from theirs (adopt wholesale):** `batch/batch.go`, `batch/positions.go`, `scheduler.go`, `mlx/sdpa.go`, `mlx/recurrent.go`.

---

## File-by-File Resolution

### `x/mlxrunner/mlx/array.go` — 3 conflicts, take theirs

| # | Our hunk | Their hunk | Resolution | Rationale |
|---|---|---|---|---|
| 1 | Tracing branch + conditional `arrays = append(...)` | `arraysMu.Lock()`/`Unlock()` + `arrays = append(...)` | **Take theirs** | Theirs is the thread-safety commit (`d137b850`). Ours already merged it from fork/main — the conflict is cosmetic; verify post-merge our tracing hooks still exist or are preserved elsewhere. |
| 2 | `Ints()` has `DType` panic check | Check removed | **Take theirs** | Theirs' multi-seq code drives Ints/Floats on more varied dtypes; defensive panic is not worth the maintenance cost. If we later regret this, it's a separate change. |
| 3 | `Floats()` has `DType` panic check | Check removed | **Take theirs** | Same as #2. |

**Action:** `git checkout --theirs -- x/mlxrunner/mlx/array.go` then verify our tracing still works.
**Inside-theirs lines added: 0.**

### `x/mlxrunner/mlx/ops_extra.go` — 2 conflicts, take theirs

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | Old `RoPEWithBase(x, dims, traditional, base, scale, offset int)` returning via `RoPEWithFreqs` | New `RoPEWithBase(..., positions *Array)` with contiguous fast-path + dynamic multi-seq path; adds `SiLU` | **Take theirs** |
| 2 | `C.int(offset)` / `freqsCtx` in `mlx_fast_rope` call | `positions.ctx` / `freqs.ctx` | **Take theirs** (same call site) |

**Action:** `git checkout --theirs -- x/mlxrunner/mlx/ops_extra.go`. All our callers of `RoPEWithBase(..., offset int)` must switch to the positions-based signature — fix those on our side. Model files (`llama`, `qwen3`, etc.) already got theirs' version via auto-merge, so callers inside those are already updated.
**Inside-theirs lines added: 0.**

### `x/mlxrunner/cache/recurrent.go` — 2 conflicts, take theirs + sibling file

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | `recurrentSnapshot` with `Size()`, `Close()`, `SnapshotType()` methods | Multi-seq `recurrentSnapshot` with seqID-keyed `convState`/`deltaState` maps | **Take theirs** |
| 2 | Soft type-assert in restore: `snap, ok := snapshot.(*recurrentSnapshot); if !ok { return miss }` | Hard type-assert: `snap := snapshot.(*recurrentSnapshot)` | **Take theirs; wrap at call site** |

**Action:**
1. `git checkout --theirs -- x/mlxrunner/cache/recurrent.go`.
2. Move `SnapshotType()`, `CollectArrays()`, and (re-designed) `SerializableSnapshot` for `recurrentSnapshot` into new **`x/mlxrunner/cache/serializable.go`**.
3. The soft type-assertion pattern moves to our persistence call-site in `cache_persist.go`'s `loadFromDisk` / `restoreSnapshotArrays` — wrap the `Restore()` call in `defer recover()` so a type mismatch surfaces as a miss rather than a panic.

**Inside-theirs lines added: 0.**

### `x/mlxrunner/cache/cache.go` — 7 conflicts, take theirs + sibling file

This is the highest-conflict-count file but the resolution shape is uniform: theirs' multi-seq refactor wins every hunk; our additions move to a sibling file.

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | `import "github.com/ollama/ollama/logutil"` | `import "github.com/ollama/ollama/x/mlxrunner/batch"` | **Take theirs**; `logutil` is no longer used here (verify by post-resolution build). |
| 2 | `SerializableSnapshot` interface definition + doc | `KVCache` regions struct doc + multi-seq API | **Take theirs**; `SerializableSnapshot` moves to `cache/serializable.go` |
| 3 | `kvSnapshot.Restore` with soft type-assert | Hard type-assert + region gap check | **Take theirs**; soft-assert pattern moves to caller via `recover()` |
| 4 | Cache `Update` body without regions | Regions init / panic-if-missing-seq | **Take theirs**; this is the multi-seq core |
| 5 | `CollectArrays` using single-seq `liveLen` | Multi-seq `seqOrder` iteration + new `Offsets()` / `Free()` methods | **Take theirs** |
| 6 | `rotatingSnapshot.Size()` delegating to `kvSnapshot` | Inlined sum of `keys.NumBytes()` + `values.NumBytes()` | **Take theirs** |
| 7 | `rotatingSnapshot.Restore` soft type-assert | Hard assert with secondary ok check | **Take theirs**; soft-assert moves to caller |

**Action:**
1. `git checkout --theirs -- x/mlxrunner/cache/cache.go`.
2. Create **`x/mlxrunner/cache/serializable.go`** containing:
   - `SnapshotType*` constants (ours, preserved)
   - `SerializableSnapshot` interface (ours, adapted to theirs' shapes — may need redesign for per-seq slices; see `MERGE_PLAN_BATCHING.md` §3.1 — chosen: per-(node, seqID) files)
   - Methods implementing `SerializableSnapshot` on `*kvSnapshot`, `*rotatingSnapshot`, `*recurrentSnapshot` (Go allows methods on same-package types from any file)
   - `SnapshotType()` methods on all three types
3. In our `cache_persist.go`, wrap `Restore()` calls to recover from type-assertion panics:
   ```go
   func safeRestore(c cache.Cache, target int, snap cache.Snapshot) (ok bool) {
       defer func() { if recover() != nil { ok = false } }()
       return c.Restore(target, snap)
   }
   ```

**Inside-theirs lines added: 0.**

### `x/mlxrunner/pipeline.go` — 4 conflicts, take theirs wholesale

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | Extra imports (`sort`, `sampler`) | Theirs' import set | **Take theirs** |
| 2 | Doc comment referring to `NumPredict` | Same comment with `MaxTokens` | **Take theirs** |
| 3 | `request.Options.NumPredict` field usage | `request.Options.MaxTokens` | **Take theirs** |
| 4 | Our `TextGenerationPipeline(ctx, request)` function body | Function removed (replaced by `scheduler.run`) | **Take theirs** |

**Action:** `git checkout --theirs -- x/mlxrunner/pipeline.go`. Audit any remaining references to `NumPredict` or `TextGenerationPipeline` on our side — most are already gone post-auto-merge in the model files.
**Inside-theirs lines added: 0.**

### `x/mlxrunner/runner.go` — 2 conflicts, take theirs + 1 hook line

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | `Pipeline func(context.Context, Request) error` field + lint comment | Field removed | **Take theirs** — `Pipeline` is dead code post-scheduler |
| 2 | Large goroutine loop that drains `r.Requests` and calls `r.Pipeline` | `return sched.run(ctx)` | **Take theirs** — scheduler owns the request loop |

**Action:**
1. `git checkout --theirs -- x/mlxrunner/runner.go`.
2. Preserve our graceful-shutdown scaffolding (`signal.NotifyContext` + `errgroup`) — verify it's outside these conflict hunks and still present; if lost, re-add in a new **`x/mlxrunner/runner_persist.go`** file.
3. Add **one line** inside `runner.go` (likely in the runner constructor or `Run()` entry) to initialize our persistence: e.g., `r.persist = newPersistObserver(r)`. That registers the observer that wires into `scheduler.admitRequest` and `scheduler.finishSeq` from our side.

**Inside-theirs lines added: 1.**

### `x/mlxrunner/server.go` — 1 conflict, take theirs + 1 debug-handler hook

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | `request.Pipeline = runner.TextGenerationPipeline` + `request.Sampler = sample.New(sample.Options{...})` | `request.Options.MaxTokens = cmp.Or(...)` + positional `sample.New(...)` | **Take theirs**; **adapt our sampler** |

Our sampler diverged from theirs' — we added logprobs (`24e038d5`), fused top-P/top-K (`22d6c817`), and `MaxAxis` min-P (`ca01373b`). Theirs uses a simpler positional API.

**Action:**
1. `git checkout --theirs -- x/mlxrunner/server.go` — accept the `cmp.Or` + `Pipeline` removal + positional `sample.New`.
2. Update our `x/mlxrunner/sample/sample.go` to satisfy both APIs (expose positional `New(...)` that internally constructs an options struct; keep our logprobs hooks callable via a separate method).
3. Migrate our debug HTTP handlers to a new **`x/mlxrunner/server_debug.go`** with a single added line in `server.go`'s handler registration: `registerDebugHandlers(mux, r)`.

**Inside-theirs lines added: 1** (the `registerDebugHandlers` call, if placed in `server.go`'s handler setup; 0 if we intercept from a different layer).

### `x/mlxrunner/cache.go` — 4 conflicts, take theirs baseline + 4 hook lines

This is the persistence core. Our additions must preserve but re-route through sibling files.

| # | Our hunk | Their hunk | Resolution |
|---|---|---|---|
| 1 | `ensureCaches` inline init | `else` branch for caches slice init | **Take theirs** — cleaner structure |
| 2 | Comment documenting deferred-write policy | `slog.Debug("created snapshot", ...)` | **Take theirs**; preserve our deferred-write behavior by controlling *whether* a write is queued from our persistence observer, not from this snapshot-creation site |
| 3 | Our `enforceEvictionPolicy` with two-tier comment | Theirs' `activeNodeSet()` helper + their own eviction | **Merge** — keep `activeNodeSet()` from theirs (it's useful); relocate two-tier eviction logic to new **`x/mlxrunner/cache_persist_policy.go`** |
| 4 | Loop body referencing our `maxPagedOutBytes`, `collectEvictionCandidates` | `activeSet := c.activeNodeSet()` | **Take theirs for the helper call**; our two-tier loop moves to the sibling file |

**Action:**
1. `git checkout --theirs -- x/mlxrunner/cache.go`.
2. Create **`x/mlxrunner/cache_persist_policy.go`** housing:
   - `enforceTwoTierEviction(c *kvCache, active map[*trieNode]bool)` — the memory → disk → delete loop, using `c.activeNodeSet()` from theirs.
   - `collectEvictionCandidates` (ours, unchanged).
   - Cold/Warm/Gone state table keyed by `*trieNode` (Union semantics — see `MERGE_PLAN_BATCHING.md` §3.2).
3. Add **one hook line** in theirs' `evictNode` function inside `cache.go`: `persistHookBeforeEvict(c, node)` — intercept for tier-1→tier-2 migration.
4. Add **one hook line** in theirs' `advancePath` or equivalent split site: `persistHookOnSplit(c, node)` — rebuild snapshot per-file identity after split.
5. Add **two more small calls** at the `snapshot created` site and at `cacheSession.close()` if observer cannot cover them from scheduler-level hooks — budget keeps each under 1 line.

**Inside-theirs lines added: ≤ 4.**

---

## Execution Order (dependency-safe)

Within each group, files can be resolved in parallel; between groups, earlier groups must finish first.

**Group A — foundation (take theirs, zero logic change):**
1. `x/mlxrunner/mlx/array.go`
2. `x/mlxrunner/mlx/ops_extra.go`
3. `x/mlxrunner/pipeline.go`
4. `x/mlxrunner/cache/recurrent.go`

Commands: `git checkout --theirs -- <file>` × 4. Build check: `go build ./x/mlxrunner/mlx/... ./x/mlxrunner/cache/...`.

**Group B — core cache layer:**
5. `x/mlxrunner/cache/cache.go` — `git checkout --theirs`, then create `cache/serializable.go` with our types.
6. `x/mlxrunner/cache.go` — `git checkout --theirs`, then create `cache_persist_policy.go` and add 4 hook calls.

Build check: `go build ./x/mlxrunner/...`. Expect compile errors about `kvCache` methods and state; resolve by moving them to sibling files.

**Group C — runner/server glue:**
7. `x/mlxrunner/runner.go` — `git checkout --theirs`, create `runner_persist.go`, add 1 init line.
8. `x/mlxrunner/server.go` — `git checkout --theirs`, adapt sampler caller on our side, create `server_debug.go`, add 1 registration line.

Build check: full `go build ./...`.

**Group D — verification:**
- `go vet ./x/mlxrunner/... ./x/models/...`
- `go test ./x/mlxrunner/cache/... -count=1`
- `go test ./x/mlxrunner/... -run Persist -count=1`
- `go test ./x/mlxrunner/... -run WarmRestart -count=1`
- `go test -race ./x/mlxrunner/...`
- Line-budget gate: `git diff origin/jessegross/batching..HEAD --shortstat -- x/mlxrunner/scheduler.go x/mlxrunner/cache/ x/mlxrunner/cache.go x/mlxrunner/pipeline.go x/mlxrunner/runner.go x/mlxrunner/server.go x/models/` must show ≤ 20 net-added lines.

**Group E — observer wiring (new code, no conflicts):**
9. `x/mlxrunner/scheduler_persist.go` — defines observer with `OnSeqAdmit(ctx, seqID, request)` and `OnSeqFinish(seqID)`; wires into `scheduler.admitRequest` and `scheduler.finishSeq` from outside. Zero lines inside `scheduler.go` if we use a callback field (1 struct-field add, already in budget).

---

## Known Risks

1. **Soft type-assertion via `recover()`** adds runtime overhead on every `Restore()` call. If measurable, promote to a typed check (`if _, ok := snap.(*kvSnapshot); !ok { return false }`) in `cache_persist.go` — costs 2 lines in *our* file, 0 in theirs.

2. **`SerializableSnapshot` redesign for per-(node, seqID) files** — our filename-hash basis changes. Existing on-disk snapshots become unreadable; rehydrate must tolerate missing-file → cold-start, not error.

3. **Deferred-write semantics**: theirs adds `slog.Debug("created snapshot")` implying snapshots fire eagerly. Our deferred-write is a behavior our observer must enforce by simply *not writing* on snapshot creation — write on evict/close only.

4. **Lost behaviors to audit post-merge:**
   - MLX array tracing hook in `New(name)` — verify still called when `tracing` is true.
   - Our graceful shutdown (`signal.NotifyContext` + errgroup) — verify still present in `runner.go` outside the conflicted hunks; re-add in `runner_persist.go` if lost.
   - Sampler extensions (logprobs, fused top-P/K, MaxAxis min-P) — server.go takes theirs' positional API; our extensions must still work via helper methods.

5. **Budget overrun triggers**: if any single file exceeds its allocation, stop and redesign — typically means more logic needs to move to a sibling file, not that the budget is wrong.

---

## What This Plan Does NOT Cover

- **Gemma4 model code** (1520+231+503 lines in the prior abandoned attempt) — out of scope per user decision. Lands separately.
- **Rebase onto future `main`** — deferred; the rebase-dry-run gate in §6 of `MERGE_PLAN_BATCHING.md` proves future syncs will be clean.
- **Multi-seq persistence performance profiling** — follow-up issue per `MERGE_PLAN_BATCHING.md` §7.
