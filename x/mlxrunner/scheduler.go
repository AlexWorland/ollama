package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// activeSeq tracks a single sequence in the decode batch.
type activeSeq struct {
	seqID   int
	session *cacheSession
	request Request

	// Decode state — pinned arrays from the previous step.
	sample, logprobs *mlx.Array

	buf        bytes.Buffer
	generated  int
	final      CompletionResponse
	admittedAt time.Time // set at admitRequest entry; used to log prefill duration
	decodeAt   time.Time // set after prefill completes
}

func (s *activeSeq) cleanup() {
	if s.request.Sampler != nil {
		s.request.Sampler.Free()
	}
	mlx.Unpin(s.sample, s.logprobs)
}

const maxParallel = 4

// decodeLogInterval throttles the per-iteration decode log. The scheduler
// runs one decode step per generated token per active sequence; logging
// every step overwhelms the journal under sustained load. We log the first
// step (so "decode started" is visible) and every Nth step thereafter.
const decodeLogInterval = 100

// scheduler manages prefill and decode for all active sequences.
type scheduler struct {
	runner    *Runner
	active    []*activeSeq
	used      [maxParallel]bool // seqID slot allocation
	stepCount uint64            // monotonic decode step counter; drives decodeLogInterval
}

func (r *Runner) newScheduler() *scheduler {
	return &scheduler{runner: r}
}

// allocSeqID returns the lowest free seqID slot.
func (s *scheduler) allocSeqID() int {
	for i, used := range s.used {
		if !used {
			s.used[i] = true
			slog.Debug("scheduler: allocated seq slot", "seq", i, "used", s.countUsed(), "max_parallel", maxParallel)
			return i
		}
	}
	panic("no free sequence slots")
}

// freeSeqID returns a seqID slot to the pool.
func (s *scheduler) freeSeqID(seqID int) {
	s.used[seqID] = false
	slog.Debug("scheduler: freed seq slot", "seq", seqID, "used", s.countUsed(), "max_parallel", maxParallel)
}

// countUsed returns the number of currently allocated seqID slots.
func (s *scheduler) countUsed() int {
	n := 0
	for _, u := range s.used {
		if u {
			n++
		}
	}
	return n
}

// activeSeqIDs returns the seqIDs currently in the decode batch, for logging.
func (s *scheduler) activeSeqIDs() []int {
	ids := make([]int, len(s.active))
	for i, a := range s.active {
		ids[i] = a.seqID
	}
	return ids
}

func (s *scheduler) run(ctx context.Context) error {
	r := s.runner

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}

	slog.Info("scheduler: started", "max_parallel", maxParallel)
	defer func() {
		slog.Info("scheduler: stopped", "active_at_shutdown", len(s.active))
	}()

	for {
		if len(s.active) == 0 {
			// No active sequences — block waiting for a request. A free slot
			// is guaranteed here, so admission cannot exhaust the pool.
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				s.admitRequest(ctx, request)
			}
			continue
		}

		// Active sequences decoding. Only poll r.Requests when a slot is
		// free; otherwise leave pending requests queued in the channel (and
		// blocking the HTTP handler's send) until a decode completes and
		// frees a slot. This is the backpressure that keeps allocSeqID's
		// pool-exhaustion sentinel from firing under bursty concurrent load.
		if len(s.active) < maxParallel {
			select {
			case <-ctx.Done():
				s.finishAll()
				return nil
			case request := <-r.Requests:
				s.admitRequest(ctx, request)
			default:
			}
		} else {
			select {
			case <-ctx.Done():
				s.finishAll()
				return nil
			default:
			}
		}

		s.decodeStep(ctx)
	}
}

// admitRequest prefills a new request and adds it to the decode batch.
func (s *scheduler) admitRequest(ctx context.Context, request Request) {
	mlx.ResetPeakMemory()

	seqID := s.allocSeqID()

	admitStart := time.Now()
	slog.Info("scheduler: admitting request",
		"seq", seqID,
		"active_before", len(s.active),
		"prompt_tokens", len(request.Tokens))

	seq := &activeSeq{
		seqID:      seqID,
		request:    request,
		admittedAt: admitStart,
		final: CompletionResponse{
			Done:            true,
			PromptEvalCount: len(request.Tokens),
			EvalCount:       request.Options.MaxTokens,
			DoneReason:      1,
		},
	}

	// Ensure caches exist with all pool slots registered. SetSeqs is
	// a no-op after the first call since the slot set never changes.
	s.runner.cache.ensureCaches(s.runner.Model)
	allSlots := make([]int, maxParallel)
	for i := range allSlots {
		allSlots[i] = i
	}
	for _, kv := range s.runner.cache.caches {
		if kv != nil {
			kv.SetSeqs(allSlots)
		}
	}

	if err := s.prefill(ctx, seq); err != nil {
		slog.Info("Prefill failed", "seq", seqID, "error", err)
		seq.cleanup()
		s.freeSeqID(seqID)
		s.sendError(request, err)
		return
	}

	// Materialize all cache state so existing sequences' decode steps
	// see clean buffer data (not lazy graphs from prefill/restore).
	s.materializeCaches()

	slog.Debug("scheduler: admitted",
		"seq", seq.seqID,
		"active_after", len(s.active)+1,
		"prefill_ms", time.Since(admitStart).Milliseconds())

	s.active = append(s.active, seq)
}

func (s *scheduler) prefill(ctx context.Context, seq *activeSeq) error {
	r := s.runner
	inputs := seq.request.Tokens
	seq.request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(seq.seqID, r.Model, inputs)
	seq.session = session

	caches := session.caches
	tokens := session.remaining

	// Schedule periodic snapshots during prefill.
	const snapshotInterval = 8192
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		session.requestSnapshot(offset)
	}
	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		session.requestSnapshot(end)
	}

	prefillChunk := prefillChunkSize()
	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := seq.request.Ctx.Err(); err != nil {
			return err
		}

		n := min(prefillChunk, total-processed-1)

		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			tokensUntilSnapshot := snapOffset - (baseOffset + processed)
			if tokensUntilSnapshot > 0 && tokensUntilSnapshot < n {
				n = tokensUntilSnapshot
			}
		}

		r.Model.Forward(&batch.ForwardBatch{
			InputIDs: mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0),
			SeqIDs:   []int{seq.seqID},
			SeqLens:  []int{n},
		}, caches)
		mlx.Sweep()
		s.materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "seq", seq.seqID, "processed", processed, "total", total)

		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			if baseOffset+processed >= snapOffset {
				session.snapshot()
			}
		}

		mlx.ClearCache()
	}

	// First decode step: process final token(s) and get initial sample.
	// Eval the sample AND the cache state so everything is materialized
	// before any cache transitions (snapshot/restore/rebuild).
	seq.sample, seq.logprobs = s.singleStep(seq, mlx.FromValues(tokens[processed:], total-processed))
	evalArrays := []*mlx.Array{seq.sample, seq.logprobs}
	for _, c := range caches {
		evalArrays = append(evalArrays, c.State()...)
	}
	mlx.Eval(evalArrays...)
	seq.decodeAt = time.Now()

	slog.Debug("scheduler: prefill complete, first decode ready",
		"seq", seq.seqID,
		"prompt_tokens", len(inputs),
		"prefill_ms", time.Since(seq.admittedAt).Milliseconds())

	return nil
}

// singleStep runs a single-sequence forward+sample (used during prefill's
// final token and as fallback).
func (s *scheduler) singleStep(seq *activeSeq, token *mlx.Array) (*mlx.Array, *mlx.Array) {
	r := s.runner
	caches := seq.session.caches

	fwd := r.Model.Forward(&batch.ForwardBatch{
		InputIDs: token.ExpandDims(0),
		SeqIDs:   []int{seq.seqID},
		SeqLens:  []int{1},
	}, caches)
	logits := r.Model.Unembed(fwd)
	logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

	logprobs := logits.Subtract(logits.Logsumexp(true))
	sample := seq.request.Sampler.Sample(logprobs)

	mlx.Pin(sample, logprobs)
	mlx.Sweep()
	mlx.AsyncEval(sample, logprobs)

	return sample, logprobs
}

// decodeStep runs one batched decode iteration for all active sequences.
func (s *scheduler) decodeStep(ctx context.Context) {
	r := s.runner

	// Check for cancelled sequences and remove them.
	s.reapCancelled(ctx)
	if len(s.active) == 0 {
		return
	}

	s.stepCount++
	if s.stepCount == 1 || s.stepCount%decodeLogInterval == 0 {
		slog.Debug("scheduler: decode step",
			"active", len(s.active),
			"seq_ids", s.activeSeqIDs(),
			"step", s.stepCount)
	}

	// Read token values from previous step's samples. This forces
	// evaluation of the lazy computation from the prior step.
	inputTokens := make([]int32, len(s.active))
	for i, seq := range s.active {
		if seq.generated == 0 {
			mlx.Eval(seq.sample)
			seq.final.PromptEvalDuration = time.Since(seq.decodeAt)
			seq.decodeAt = time.Now()
		}
		inputTokens[i] = int32(seq.sample.Int())
	}

	// Process previous step's outputs: stream tokens, check EOS.
	var completed []*activeSeq
	for i, seq := range s.active {
		output := inputTokens[i]
		seq.session.outputs = append(seq.session.outputs, output)
		seq.generated++

		if r.Tokenizer.IsEOS(output) {
			seq.final.DoneReason = 0
			seq.final.EvalCount = seq.generated - 1
			completed = append(completed, seq)
			continue
		}

		if seq.generated >= seq.request.Options.MaxTokens {
			seq.final.EvalCount = seq.generated
			completed = append(completed, seq)
			continue
		}

		// Stream token to client.
		select {
		case <-seq.request.Ctx.Done():
			completed = append(completed, seq)
		case seq.request.Responses <- CompletionResponse{
			Content: r.Decode(output, &seq.buf),
		}:
		}
	}

	// Finish completed sequences and remove from active list.
	if len(completed) > 0 {
		completedSet := make(map[int]bool, len(completed))
		for _, seq := range completed {
			s.finishSeq(seq)
			completedSet[seq.seqID] = true
		}
		alive := s.active[:0]
		for _, seq := range s.active {
			if !completedSet[seq.seqID] {
				alive = append(alive, seq)
			}
		}
		s.active = alive
		mlx.ClearCache()
	}

	if len(s.active) == 0 {
		return
	}

	// Batched forward pass: one token per sequence.
	seqIDs := make([]int, len(s.active))
	seqLens := make([]int, len(s.active))
	nextTokens := make([]int32, len(s.active))
	for i, seq := range s.active {
		seq.request.Sampler.AppendToken(seq.sample)
		nextTokens[i] = int32(seq.sample.Int())
		seqIDs[i] = seq.seqID
		seqLens[i] = 1
		mlx.Unpin(seq.sample, seq.logprobs)
		seq.sample, seq.logprobs = nil, nil
	}

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		logutil.Trace(fmt.Sprintf("scheduler: forward batch seq_ids=%v seq_lens=%v total=%d",
			seqIDs, seqLens, len(nextTokens)))
	}

	fwd := r.Model.Forward(&batch.ForwardBatch{
		InputIDs: mlx.FromValues(nextTokens, len(nextTokens)).ExpandDims(0),
		SeqIDs:   seqIDs,
		SeqLens:  seqLens,
	}, r.cache.caches)
	logits := r.Model.Unembed(fwd)

	for i, seq := range s.active {
		seqLogits := logits.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()).Squeeze(1)
		lp := seqLogits.Subtract(seqLogits.Logsumexp(true))
		sample := seq.request.Sampler.Sample(lp)
		mlx.Pin(sample, lp)
		seq.sample = sample
		seq.logprobs = lp
	}

	mlx.Sweep()

	evalArrays := make([]*mlx.Array, 0, 2*len(s.active))
	for _, seq := range s.active {
		evalArrays = append(evalArrays, seq.sample, seq.logprobs)
	}
	mlx.AsyncEval(evalArrays...)
}

// reapCancelled removes sequences whose request context has been cancelled.
func (s *scheduler) reapCancelled(ctx context.Context) {
	var alive []*activeSeq
	for _, seq := range s.active {
		if ctx.Err() != nil || seq.request.Ctx.Err() != nil {
			s.finishSeq(seq)
		} else {
			alive = append(alive, seq)
		}
	}
	if reaped := len(s.active) - len(alive); reaped > 0 {
		slog.Info("scheduler: reaped cancelled sequences",
			"count", reaped,
			"active_remaining", len(alive))
		s.active = alive
	}
}

// finishSeq sends the final response, saves to trie, and cleans up.
// It does NOT remove from s.active — the caller is responsible for that.
func (s *scheduler) finishSeq(seq *activeSeq) {
	seq.final.EvalDuration = time.Since(seq.decodeAt)

	reason := "max_tokens"
	switch seq.final.DoneReason {
	case 0:
		reason = "eos"
	case 1:
		reason = "max_tokens"
	}
	if seq.request.Ctx.Err() != nil {
		reason = "cancelled"
	}
	slog.Debug("scheduler: sequence finishing",
		"seq", seq.seqID,
		"reason", reason,
		"generated", seq.generated,
		"eval_ms", seq.final.EvalDuration.Milliseconds())

	// Send final response.
	if seq.request.Ctx.Err() == nil {
		select {
		case seq.request.Responses <- seq.final:
		case <-seq.request.Ctx.Done():
		}
	}

	// Save to trie and clean up.
	if seq.session != nil && seq.generated > 0 {
		seq.session.close()
	}
	s.freeSeqID(seq.seqID)
	seq.cleanup()
	close(seq.request.Responses)

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		s.runner.cache.dumpTree()
	}
	slog.Info("sequence complete", "seq", seq.seqID, "generated", seq.generated,
		"peak_memory", mlx.PrettyBytes(mlx.PeakMemory()))
}

func (s *scheduler) sendError(request Request, err error) {
	slog.Info("Request terminated", "error", err)
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		statusErr = api.StatusError{
			StatusCode:   http.StatusInternalServerError,
			ErrorMessage: err.Error(),
		}
	}
	select {
	case request.Responses <- CompletionResponse{Error: &statusErr}:
	case <-request.Ctx.Done():
	}
	close(request.Responses)
}

func (s *scheduler) finishAll() {
	for _, seq := range s.active {
		s.finishSeq(seq)
	}
	s.active = nil
}

func (s *scheduler) materializeCaches() {
	state := make([]*mlx.Array, 0, 2*len(s.runner.cache.caches))
	for _, c := range s.runner.cache.caches {
		state = append(state, c.State()...)
	}
	if len(state) == 0 {
		return
	}
	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		logutil.Trace(fmt.Sprintf("scheduler: materialize %d cache arrays", len(state)))
	}
	mlx.Eval(state...)
}
