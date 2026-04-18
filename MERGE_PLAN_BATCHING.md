# Merge Plan: `origin/jessegross/batching` → `cache-persisitance`

_Generated 2026-04-18. Merge base: `4bc27280`. Incoming tip: `2beb5445`. Our tip: `b4b7a9b3`._

## 1. Situation

| | |
|--|--|
| Incoming commits | **9** (merge-base..`origin/jessegross/batching`) |
| Our commits since base | **86** (includes fork/main merge + 20-commit persistence series) |
| Files changed on both sides | **14** (all will conflict) |
| New files from incoming | `x/mlxrunner/scheduler.go`, `x/mlxrunner/batch/batch.go`, `x/mlxrunner/batch/positions.go` |
| Textual conflict hunks (`merge-tree` preview) | **125** |
| Working tree state | **Dirty** — 7 modified files + 4 untracked paths (`.claude/`, `.omc/`, `.omx/`, `CACHE_PERSISTENCE_SUMMARY.md`) |

### Incoming commit series (oldest → newest)

```
d137b850 mlx: make array management thread-safe
30915b6b mlxrunner: tokenize prompts in request handler goroutines
987f74c8 mlxrunner: introduce ForwardBatch for model forward pass
b7b2aa5d mlxrunner: Cache.Update takes ForwardBatch and returns KVHistory   ← API break
1ea8e70d mlxrunner: positions tensor and RoPEWithBase
02fe50c9 mlxrunner: SDPA, GatedDelta, and RecurrentConv1d with KVHistory
d8067801 mlxrunner: multi-sequence KVCache, RotatingKVCache, and RecurrentCache
98615b86 mlxrunner: per-sequence trie paths and seqID-parameterized cache ops   ← load-bearing
2beb5445 mlxrunner: replace TextGenerationPipeline with scheduler   ← pipeline.go rewrite
```

## 2. Why This Is Hard

Two architecturally opposed efforts intersect on the same types:

- **Theirs** re-shapes the runtime around multi-sequence continuous batching: `ForwardBatch` flows through model `Forward`, caches are keyed by `seqID`, the trie gets per-sequence paths, and the pipeline.go control loop is replaced by `scheduler.go`.
- **Ours** layers disk persistence onto the pre-batching trie: content-addressed safetensors snapshots, Cold/Warm/Gone node states, sync writer, two-tier eviction, startup rehydration, `SerializableSnapshot` interface on the three cache types.

Their refactor touches the exact types we extended for persistence. Specifically:

- `Cache.Update` signature change (`b7b2aa5d`) invalidates every persistence call site that reads or writes `KVCache` state mid-step.
- Per-sequence trie paths (`98615b86`) change the semantics our content-addressed filenames were built on — one snapshot per node assumed one owning path.
- Scheduler replacement (`2beb5445`) removes the `pipeline.go` hooks where we attached persistence lifecycle (attach-snapshots, split-node, enforce-eviction-policy).

## 3. File-by-File Risk

| File | Conflict type | Ours adds | Theirs adds | Reconciliation |
|---|---|---|---|---|
| `x/mlxrunner/cache.go` | **Severe** | persistence fields on `trieNode`/`kvCache`, `scheduleWrite`, Cold/Warm/Gone transitions, two-tier eviction | multi-seq trie paths, seqID-keyed ops, LRU rework | Manual. Start from **theirs**, re-introduce persistence fields + hooks atop new per-seq structure. |
| `x/mlxrunner/cache/cache.go` | **Severe** (+873/–??) | `SnapshotType()` constants, `SerializableSnapshot` interface, `idxArr` serialization artifact | multi-seq `KVCache` / `RotatingKVCache`, new snapshot shape for batch | Manual. Re-implement `SerializableSnapshot` on top of multi-seq snapshot struct. Audit `snapshotBytes` cost under multi-seq. |
| `x/mlxrunner/cache/recurrent.go` | High | `SnapshotType()` consts on `RecurrentCache` | rewritten recurrent cache for multi-seq | Manual. Port snapshot type constants forward. |
| `x/mlxrunner/pipeline.go` | High (shrinks −163) | persistence call sites | pipeline largely removed in favor of `scheduler.go` | **Take theirs**; move our hooks into `scheduler.go`. |
| `x/mlxrunner/runner.go` | Medium | persistence wiring + graceful shutdown via `signal.NotifyContext` + errgroup | scheduler construction, tokenize-in-goroutine | Manual. Keep ours' shutdown scaffolding, adopt theirs' scheduler wiring. |
| `x/mlxrunner/server.go` | Low | header handlers for debug | small scheduler-facing edits | Manual, cheap. |
| `x/mlxrunner/cache_test.go` | Medium | persistence tests | multi-seq test fixtures | Manual; expect some of our tests to need redesign. |
| `x/mlxrunner/cache_trie_test.go` | Low | one-line addition | trivial edit | Almost clean. |
| `x/mlxrunner/mlx/array.go` | Low | minor | minor | Likely mechanical. |
| `x/mlxrunner/mlx/ops_extra.go` | Medium | — | adds SDPA / gated-delta / recurrent / positions helpers | Mostly take theirs; verify our usages still compile. |
| `x/models/llama/llama.go` | Medium | — | `Forward` signature takes `ForwardBatch` | Take theirs. |
| `x/models/qwen3/qwen3.go` | Medium | — | same | Take theirs. |
| `x/models/qwen3_5/qwen3_5.go` | High (−83) | — | recurrent path rewired for multi-seq | Take theirs. |
| `x/models/glm4_moe_lite/glm4_moe_lite.go` | Medium | — | same | Take theirs. |

**New-from-theirs (no conflict, adopt wholesale):** `scheduler.go`, `batch/batch.go`, `batch/positions.go`, `mlx/sdpa.go`, `mlx/recurrent.go`.

**Our files that only exist on ours (must survive merge untouched):** `cache_persist.go`, `cache_persist_test.go`, `cache_trie.go`, `safetensors_header.go`, `safetensors_header_test.go`, `cache/rotating_multiturn_test.go`, `mlx/compile.go`, `mlx/compile_test.go`, `mlx/dynamic_darwin.go`, `mlx/dynamic_other.go`.

## 4. Strategy

### Step 0 — Clean the working tree (blocker)
Before merging, resolve the dirty tree.

- Either **commit** the current simplify-pass changes (they look staged for commit per observation 19570), or **stash** with `git stash push -u -m "WIP pre-batching merge"`.
- Rule out `.claude/`, `.omc/`, `.omx/` — those are agent scratch dirs; add to `.gitignore` rather than commit.
- `CACHE_PERSISTENCE_SUMMARY.md`: commit (it's intentional doc) or move to `docs/`.

### Step 1 — Create a throwaway reconciliation branch
```
git switch -c merge/batching-into-persistence
```
All reconciliation work happens here. If it goes wrong, abandon without touching `cache-persisitance`.

### Step 2 — Pre-merge baseline capture
Snapshot current green state so we can diff against it post-merge:
```
go build ./x/mlxrunner/... ./x/models/...
go test ./x/mlxrunner/... -run TestCache -count=1 > /tmp/pre-merge-cache-tests.txt
go test ./x/mlxrunner/... -run TestPersist -count=1 > /tmp/pre-merge-persist-tests.txt
```

### Step 3 — Decide merge shape (requires your input — see §6)
Two viable shapes; they produce different conflict experiences:

- **(A) Single merge commit, manual conflict resolution** — one big `git merge origin/jessegross/batching`, fix 125 hunks across 14 files in one sitting. Preserves history of both sides cleanly.
- **(B) Cherry-pick-by-commit replay** — cherry-pick the 9 incoming commits onto our tip one at a time. Each cherry-pick is smaller and easier to review, but produces 9 merge-ish commits instead of one, and our history becomes non-linear vs. `origin/jessegross/batching`.

Default recommendation: **(A)** with aggressive use of `git checkout --theirs` / `--ours` per-file followed by manual reconciliation, because their refactor is coherent and needs to land together — splitting it by commit doesn't give meaningful intermediate compile states.

### Step 4 — Resolve in dependency order
When conflicts materialize, resolve in this order (each unblocks the next):

1. **`x/mlxrunner/mlx/ops_extra.go`, `mlx/array.go`** — foundation helpers; take theirs, verify ours-side callers still compile.
2. **Model files** (`llama`, `qwen3`, `qwen3_5`, `glm4_moe_lite`) — take theirs; `Forward` signature is a pure widening for us.
3. **`x/mlxrunner/cache/cache.go` + `cache/recurrent.go`** — the most important semantic step. Start from theirs, then re-apply:
   - `SnapshotType()` method on each cache type (returning our existing constants from `cache/cache.go`).
   - Whatever `SerializableSnapshot` means on a multi-sequence snapshot (it may need to serialize per-seq slices instead of one flat buffer — design decision).
4. **`x/mlxrunner/cache.go`** — start from theirs, re-introduce persistence fields (`snapshotState`, `lastFlushedAt`, etc.), re-wire Cold/Warm/Gone transitions to per-seq paths.
5. **`x/mlxrunner/pipeline.go`** — take theirs (it's largely deleted).
6. **`x/mlxrunner/scheduler.go` (new)** — after the merge lands as-is, add the persistence hooks it needs: `attachSnapshots`, `splitNode`, `scheduleWrite`, `enforceEvictionPolicy`. These used to live in `pipeline.go`.
7. **`x/mlxrunner/runner.go`** — reconcile ctor wiring: theirs constructs scheduler, ours wires persistence root/budget and shutdown drain.
8. **`x/mlxrunner/server.go`** — tiny, trivial.
9. **Tests** — `cache_test.go`, `cache_trie_test.go`, then re-run our persistence suites. Expect fixtures to need seqID.

### Step 5 — Validation gates (must pass before merging back)

| Gate | Command | Blocker? |
|---|---|---|
| Full build | `go build ./...` | Yes |
| Go vet | `go vet ./x/mlxrunner/... ./x/models/...` | Yes |
| Cache unit tests | `go test ./x/mlxrunner/cache/... -count=1` | Yes |
| Persistence tests | `go test ./x/mlxrunner/... -run Persist -count=1` | Yes |
| Warm-restart integration | `go test ./x/mlxrunner/... -run WarmRestart -count=1` | Yes |
| Multi-turn rotating cache | `go test ./x/mlxrunner/cache/... -run Multiturn -count=1` | Yes |
| End-to-end smoke (local MLX) | runner up + `curl` generate + restart + generate again | Yes |
| Race detector on cache tests | `go test -race ./x/mlxrunner/...` | **Yes** — theirs explicitly makes array management thread-safe; ours removed the async writer because MLX wasn't thread-safe. Recheck that assumption. |

### Step 6 — Post-merge audit items
Even after all gates pass, open follow-up issues for:

1. **Content-addressed filename semantics under multi-seq** — if per-seq trie paths mean the same node can be reached by multiple sequences, the filename hash basis changes.
2. **`snapshotBytes` cost** — prior review flagged O(30N²) under eviction loops; theirs' multi-seq trie could amplify this.
3. **MLX thread-safety revisit** — commit `d137b850` means the "MLX not thread-safe" justification for dropping the async writer may no longer hold. Worth revisiting once batching lands.
4. **`SerializableSnapshot` under multi-seq** — the contract predates per-seq caches; verify restore semantics still make sense.

### Step 7 — Completion options
Once gates pass on `merge/batching-into-persistence`:

- Fast-forward or squash-merge back into `cache-persisitance`.
- Or open a PR from `merge/batching-into-persistence` → `cache-persisitance` for review.

## 5. Rollback Plan

If reconciliation proves unworkable mid-merge:
```
git merge --abort                    # during conflict resolution
git switch cache-persisitance         # after merge commit — branch is throwaway
git branch -D merge/batching-into-persistence
```

No destructive action on `cache-persisitance` at any point.

## 6. Decision Point — Your Input Needed

Before I execute this, one strategic call is yours:

**Question:** Should persistence adapt to the scheduler model, or should we treat batching as a non-goal for this branch and *skip the merge*?

- **Option A: Adapt persistence to scheduler** (this plan). Large reconciliation effort. Estimated 1–2 focused days. Result: one branch that has both features. High-risk step is re-designing `SerializableSnapshot` for multi-seq snapshots.
- **Option B: Merge but keep persistence single-seq-only for now.** Land theirs wholesale, gate persistence behind a check that only writes snapshots when `len(batch.Sequences) == 1`. Ship faster; degrade gracefully under batching. Follow-up work tracks multi-seq persistence.
- **Option C: Don't merge yet.** Wait for `jessegross/batching` to land upstream, then rebase our persistence work onto post-batching main. Smallest engineering risk but longest calendar delay and biggest eventual rebase.

Default recommendation: **B**, because it minimizes time-to-merge while keeping persistence working for the common case (single-sequence generation). Multi-seq persistence is a real design question, not a mechanical port.
