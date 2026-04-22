# Merge Plan: `origin/jessegross/batching` → `cache-persisitance`

_Generated 2026-04-18; rewritten 2026-04-22 for multi-seq persistence + minimal-diff strategy._
_Merge base: `4bc27280`. Incoming tip: `2beb5445`. Our tip: `6924577e`._

## 0. Goals (ordered)

1. **Merge `origin/jessegross/batching`** into `cache-persisitance`.
2. **Persist all sequences.** Disk persistence must remain a full-fidelity feature under multi-sequence continuous batching — no single-seq gate, no regression.
3. **Minimal diff against theirs.** Future syncs from `main` (once batching lands upstream) must be trivial. Hard budget: **≤ 20 net-added lines inside their files combined.** Everything else lives in our sibling files.

These goals are in tension. Goal 2 forces real integration with their multi-seq cache types; goal 3 forbids modifying those types. Resolution: all adaptation work lives in *our own files*; their files expose only the thinnest possible hook surface — ideally one callback/observer registration and zero behavioral edits.

### Design constraints that fall out of the goals

| Rule | Why |
|---|---|
| No refactors of their code | Every edited line is a future conflict |
| Additive, not invasive | Prefer a new file (`cache_persist_hooks.go`, `serializable.go`, `scheduler_persist.go`) over editing one of theirs |
| Wrap, don't modify | Extend their types from outside via free functions / type assertions / sibling types |
| Interface-boundary edits only | If an edit inside their file is unavoidable, it must be a single-line hook invocation or interface satisfaction — no logic |
| Line budget ≤ 20 | Hard gate; overage means redesign before merge |
| Our files stay on our side | `cache_persist*.go`, `cache_trie*.go`, new `*_persist.go` carry forward untouched across main-syncs |

Exit test: `git diff origin/jessegross/batching..HEAD -- x/mlxrunner/scheduler.go x/mlxrunner/cache/ x/mlxrunner/pipeline.go x/mlxrunner/runner.go x/mlxrunner/server.go x/models/` should be short, boring, and re-appliable.

## 1. Situation

| | |
|--|--|
| Incoming commits | **9** (merge-base..`origin/jessegross/batching`) |
| Our commits since base | **~86** (includes fork/main merges + 20-commit persistence series) |
| Files touched by both sides | **14** (all will conflict) |
| New files from incoming | `scheduler.go`, `batch/batch.go`, `batch/positions.go`, `mlx/sdpa.go`, `mlx/recurrent.go` |
| Textual conflict hunks (merge-tree preview) | **125** |
| Working-tree state | **Dirty** — untracked PNGs, `solve_primes.py`, `x/imagegen/models/zimage/scheduler_test.go` |

### Incoming commits (oldest → newest)
```
d137b850 mlx: make array management thread-safe
30915b6b mlxrunner: tokenize prompts in request handler goroutines
987f74c8 mlxrunner: introduce ForwardBatch for model forward pass
b7b2aa5d mlxrunner: Cache.Update takes ForwardBatch and returns KVHistory   ← API break
1ea8e70d mlxrunner: positions tensor and RoPEWithBase
02fe50c9 mlxrunner: SDPA, GatedDelta, and RecurrentConv1d with KVHistory
d8067801 mlxrunner: multi-sequence KVCache, RotatingKVCache, and RecurrentCache   ← cache reshape
98615b86 mlxrunner: per-sequence trie paths and seqID-parameterized cache ops   ← trie reshape
2beb5445 mlxrunner: replace TextGenerationPipeline with scheduler   ← pipeline deletion
```

## 2. Why This Is Hard

Two architecturally opposed efforts intersect on the same types.

- **Theirs:** multi-sequence continuous batching. `ForwardBatch` flows through model `Forward`; caches are keyed by `seqID`; trie has per-sequence paths; `pipeline.go` is replaced by `scheduler.go`.
- **Ours:** disk persistence on top of the pre-batching trie. Content-addressed safetensors snapshots, Cold/Warm/Gone node states, sync writer with atomic rename, two-tier eviction, startup rehydration, `SerializableSnapshot` interface on the three cache types.

Their refactor *and* our persistence both want to own the shape of these types:
- `Cache.Update` signature change (`b7b2aa5d`) invalidates every persistence call site that reads or writes cache state mid-step.
- Per-sequence trie paths (`98615b86`) change the semantics our content-addressed filenames were built on — one snapshot per node assumed one owning path.
- Scheduler replacement (`2beb5445`) removes the `pipeline.go` hooks where we attached persistence lifecycle (attach-snapshots, split-node, enforce-eviction).

The non-shortcut path: **re-express persistence as an observer over their types**, not as modifications to their types. That's the plan.

## 3. Design Decisions (blockers — answer before execution)

Multi-seq persistence is a real semantic shift, not a mechanical port. Three questions must be answered and their rationale recorded here before code changes start. Defaults shown; record actual choice inline.

### 3.1 Snapshot unit: per-node or per-(node, seq)?

- **(a) Per-node, multi-seq payload.** One file per trie node, containing `map[seqID][]byte` of per-seq KV slices. Content-addressed hash includes the set of seqIDs present.
  - Pro: preserves trie node as the persistence unit; fewer files.
  - Con: hash basis now depends on seq-membership, so the file identity churns every time a seq joins/leaves; eviction semantics get subtle.
- **(b) Per-(node, seqID).** One file per (trie node × seqID) pair. Filename hash basis = `hash(node content || seqID)`.
  - Pro: filename scheme is a strict extension of today's scheme (just one extra hash input); seq lifecycle → file lifecycle is 1:1; eviction trivially per-seq.
  - Con: more files; possible content duplication when two seqs share a prefix. Deduplication can be added later as an optimization.

**Default: (b).** Rationale: keeps our existing content-addressing scheme intact with minimal change, and decouples seq lifecycle from file identity — important because seqs admit/evict frequently under batching. Duplication is tolerable and can be addressed with a content-hash side-index later if it becomes a cost.

**Chosen: (b) — per-(node, seqID) files.** _Confirmed 2026-04-22._

### 3.2 Cold/Warm/Gone semantics across seqs

Does node state apply to the node as a whole, or per-(node, seq)?

- **Union (node-level).** Node is Warm if *any* seq holds it in memory; Gone only once *all* seqs release and it is evicted. State tracked per trie node.
  - Pro: simpler bookkeeping; one state per node.
  - Con: memory accounting is coarser — we can't distinguish "this node has 1 seq using it" from "5 seqs using it" for eviction-priority purposes.
- **Per-seq (intersection).** State tracked per (node, seqID). Node is effectively a histogram.
  - Pro: fine-grained eviction priority ("evict nodes used by fewer live seqs first").
  - Con: more bookkeeping; state table grows with seq count.

**Default: Union.** Rationale: matches the choice in 3.1 when we consider that `(b)` already gives us per-seq *file* lifecycle; per-seq *in-memory* state duplicates that with no additional signal. Union keeps our existing eviction policy usable unchanged.

**Chosen: Union (node-level).** _Confirmed 2026-04-22._

### 3.3 Restore timing: eager or lazy?

When does a Cold node get hydrated back into memory on warm-start or during operation?

- **Eager.** All Cold nodes rehydrated at startup (current behavior under single-seq).
  - Pro: predictable latency during serving; no cold-start surprise.
  - Con: peak memory equals full cache; doesn't play well with scheduler admission control.
- **Lazy, keyed on seq admission.** When the scheduler admits a seq, we walk its prefix path and hydrate only the nodes that seq needs.
  - Pro: idle memory stays low; hydration is proportional to active seqs.
  - Con: one more scheduler hook; first-token latency for cold seqs includes a load from disk.

**Default: Lazy.** Rationale: scheduler already has a sequence-admission path that is a natural hook point; eager hydration under multi-seq could blow memory budgets on restart. Load latency is bounded and amortized.

**Chosen: Lazy on seq admission.** _Confirmed 2026-04-22._

### Implications of the defaults on minimal-diff

With **(b) + Union + Lazy**, their code needs only:
- **One observer interface** on `scheduler` — single method, called on seq-admit and seq-complete. Satisfied from our side.
- **Optional method on their cache types** — `SnapshotSlice(seqID)` returning `([]byte, error)`. If their types already expose enough to compute this from outside (e.g., a `Slice(seqID)` accessor), we need zero new methods on their side.

Total projected surface inside their files: ~10 lines. Comfortable under the 20-line budget.

## 4. File-by-File Reconciliation (with line budgets)

| File | Strategy | Max edits allowed inside theirs |
|---|---|---|
| `x/mlxrunner/mlx/ops_extra.go` | `git checkout --theirs` | 0 |
| `x/mlxrunner/mlx/array.go` | `git checkout --theirs` | 0 |
| `x/mlxrunner/mlx/fast.go`, `nn.go` | `git checkout --theirs` | 0 |
| `x/mlxrunner/cache/cache.go` | `git checkout --theirs` + new sibling `cache/serializable.go` defining `SerializableSnapshot` over their types | **0** |
| `x/mlxrunner/cache/recurrent.go` | same as above — serializable methods live in `cache/serializable.go` | **0** |
| `x/mlxrunner/cache.go` | `git checkout --theirs`; persistence state moves to new `cache_persist_state.go` keyed by node identity; hook call on transition | **≤ 2** |
| `x/mlxrunner/pipeline.go` | `git checkout --theirs` (file is largely deleted) | 0 |
| `x/mlxrunner/scheduler.go` (new, from theirs) | adopt verbatim; register our observer from new `scheduler_persist.go`; if no extension point exists, add **one** callback field + one call site | **≤ 3** |
| `x/mlxrunner/runner.go` | keep their scheduler construction; our persistence wiring moves to new `runner_persist.go` called from one added line | **≤ 1** |
| `x/mlxrunner/server.go` | `git checkout --theirs`; our debug handlers move to new `server_debug.go` | **≤ 1** |
| `x/models/llama/llama.go`, `qwen3`, `qwen3_5`, `glm4_moe_lite` | `git checkout --theirs` — `Forward` signature widens; our callers adapt | **0** |
| `x/mlxrunner/cache_test.go`, `cache_trie_test.go` | `git checkout --theirs`; our persistence tests stay in our own `*_test.go` files | 0 |

**Running budget:** sum of the "Max edits" column = **≤ 7**. Safety margin under the 20-line ceiling.

**New files on our side** (absorb all the multi-seq adaptation):
- `x/mlxrunner/cache/serializable.go` — `SerializableSnapshot`, `SnapshotType`, per-seq slice encode/decode
- `x/mlxrunner/cache_persist_state.go` — node-id → persist state table, Cold/Warm/Gone transitions (Union semantics)
- `x/mlxrunner/scheduler_persist.go` — observer that wires seq-admit / seq-complete to hydrate / snapshot
- `x/mlxrunner/runner_persist.go` — persistence root + budget wiring, shutdown drain
- `x/mlxrunner/server_debug.go` — our existing debug header handlers

**Files that must survive unmodified** (our-side-only, no conflict surface): `cache_persist.go`, `cache_persist_test.go`, `cache_trie.go`, `cache_trie_test.go`, `safetensors_header.go`, `safetensors_header_test.go`, `cache/rotating_multiturn_test.go`, `mlx/compile.go`, `mlx/compile_test.go`, `mlx/dynamic_darwin.go`, `mlx/dynamic_other.go`.

## 5. Execution Strategy

### Step 0 — Clean working tree
- Decide disposition of untracked PNGs, `solve_primes.py`, `x/imagegen/models/zimage/scheduler_test.go`: commit to main, stash, or `.gitignore`.
- `.claude/`, `.omc/`, `.omx/` already in `.gitignore` per commit `bc9b7c02`.

### Step 1 — Record design decisions in §3
Fill in the three "Chosen:" slots above. This plan does not execute until those are answered.

### Step 2 — Safety snapshot
```
git branch cache-persist-premerge            # untouched rollback target
git switch -c merge/batching-into-persistence
```
All reconciliation happens on the throwaway branch.

### Step 3 — Pre-merge baseline
```
./scripts/build_darwin.sh
go test ./x/mlxrunner/... -count=1 > /tmp/pre-merge-tests.txt
go test -race ./x/mlxrunner/... -count=1 > /tmp/pre-merge-race.txt
```

### Step 4 — Merge and resolve in dependency order
```
git merge --no-commit --no-ff origin/jessegross/batching
```
Resolve files in this order; after each, commit-amend or stage and move on:

1. Foundation: `mlx/ops_extra.go`, `mlx/array.go`, `mlx/fast.go`, `mlx/nn.go` — `git checkout --theirs`, fix our callers.
2. Models: `llama`, `qwen3`, `qwen3_5`, `glm4_moe_lite` — `git checkout --theirs`.
3. `cache/cache.go` + `cache/recurrent.go` — `git checkout --theirs`. Add our new `cache/serializable.go`.
4. `cache.go` — `git checkout --theirs`. Add `cache_persist_state.go`. Confirm ≤ 2 edits inside `cache.go`.
5. `pipeline.go` — `git checkout --theirs`.
6. `scheduler.go` is new — accept. Add `scheduler_persist.go`. If hook needed, add one callback field + call site in `scheduler.go` (≤ 3 lines).
7. `runner.go` — reconcile; persistence init via one added line calling `runner_persist.go`.
8. `server.go` — `git checkout --theirs`; our handlers go to `server_debug.go`.
9. Tests: take theirs for `cache_test.go`, `cache_trie_test.go`. Our persistence tests adapt to multi-seq via fixtures in our own files.

### Step 5 — Line-budget checkpoint (blocker)
```
git diff origin/jessegross/batching..HEAD --shortstat -- \
  x/mlxrunner/scheduler.go x/mlxrunner/cache/ x/mlxrunner/cache.go \
  x/mlxrunner/pipeline.go x/mlxrunner/runner.go x/mlxrunner/server.go x/models/
```
Net added lines must be ≤ 20. Over-budget ⇒ stop, redesign the offending integration (almost always means pulling more logic into our sibling files).

### Step 6 — Validation gates (all must pass)

| Gate | Command |
|---|---|
| Full build | `go build ./...` |
| macOS binary | `./scripts/build_darwin.sh` |
| Vet | `go vet ./x/mlxrunner/... ./x/models/...` |
| Cache unit tests | `go test ./x/mlxrunner/cache/... -count=1` |
| Persistence tests | `go test ./x/mlxrunner/... -run Persist -count=1` |
| Warm-restart integration | `go test ./x/mlxrunner/... -run WarmRestart -count=1` |
| Multi-turn rotating cache | `go test ./x/mlxrunner/cache/... -run Multiturn -count=1` |
| **Multi-seq persistence** (new) | new test: 2+ concurrent seqs → cold-evict → restart runner → verify per-seq state restored |
| Race detector | `go test -race ./x/mlxrunner/...` |
| E2E smoke | runner up, `curl` generate, restart, generate again |
| Line-budget gate | §5 command shows ≤ 20 net-added lines inside their files |
| **Rebase dry-run** | on throwaway branch: `git rebase origin/jessegross/batching` succeeds with zero manual conflicts — proves future main-sync is clean |

### Step 7 — Land on `cache-persisitance`
Once all gates pass: fast-forward `cache-persisitance` to the reconciled tip, or open a self-review PR if you want a second pass on the diff.

## 6. Rollback

```
git merge --abort                              # during conflict resolution
git switch cache-persisitance                   # post-merge-commit — throwaway branch
git branch -D merge/batching-into-persistence
```
Safety branch `cache-persist-premerge` is never touched; rollback is always `git reset --hard cache-persist-premerge` if something lands that shouldn't have.

## 7. Post-merge follow-ups (track as issues, don't let them block the merge)

1. **Content-hash de-duplication.** Under design 3.1(b), two seqs with a shared prefix produce duplicate files. A content-hash side-index could reclaim space — low priority, additive.
2. **MLX thread-safety revisit.** `d137b850` made array management thread-safe. Our async writer was dropped because MLX wasn't. Revisit whether the async writer can return under the new guarantees.
3. **`snapshotBytes` cost under multi-seq.** Prior review flagged O(N²) under eviction loops; per-(node, seq) files amplify node count. Profile once multi-seq persistence is landed.
4. **Deferred: per-seq eviction priority.** If design 3.2 Union turns out to leave memory on the table, upgrade to per-seq state tracking as a follow-up without touching their files.
