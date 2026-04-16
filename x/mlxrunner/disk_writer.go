package mlxrunner

import (
	"path/filepath"
	"sync"
)

type diskWriteJob struct {
	data     []byte
	filename string
	node     *trieNode
}

type diskWriteResult struct {
	node *trieNode
	err  error
}

// diskWriter runs background goroutines that write pre-serialized bytes to
// content-addressed files under a fixed directory. Content-addressed names
// guarantee no write conflicts between workers. The inference goroutine
// serializes arrays and submits jobs; workers handle only file I/O.
type diskWriter struct {
	dir  string
	jobs chan diskWriteJob
	done sync.WaitGroup

	mu       sync.Mutex
	inFlight map[string]*sync.WaitGroup
	results  []diskWriteResult
}

func newDiskWriter(dir string) *diskWriter {
	w := &diskWriter{
		dir:      dir,
		jobs:     make(chan diskWriteJob, 8),
		inFlight: make(map[string]*sync.WaitGroup),
	}
	w.done.Add(2)
	for range 2 {
		go w.worker()
	}
	return w
}

func (w *diskWriter) worker() {
	defer w.done.Done()
	for job := range w.jobs {
		tmpPath := filepath.Join(w.dir, ".tmp_"+job.filename)
		finalPath := filepath.Join(w.dir, job.filename)
		err := atomicWriteFile(tmpPath, finalPath, job.data)

		w.mu.Lock()
		if wg, ok := w.inFlight[job.filename]; ok {
			delete(w.inFlight, job.filename)
			wg.Done()
		}
		w.results = append(w.results, diskWriteResult{node: job.node, err: err})
		w.mu.Unlock()
	}
}

// submit enqueues a write job. The inFlight WaitGroup is registered before
// enqueueing so waitForFile is correct even if called before a worker picks
// up the job. Blocks briefly if the jobs buffer is full.
func (w *diskWriter) submit(job diskWriteJob) {
	wg := &sync.WaitGroup{}
	wg.Add(1)
	w.mu.Lock()
	w.inFlight[job.filename] = wg
	w.mu.Unlock()
	w.jobs <- job
}

// waitForFile blocks until the named file's async write completes. Returns
// immediately if the file is not in-flight.
func (w *diskWriter) waitForFile(filename string) {
	w.mu.Lock()
	wg := w.inFlight[filename]
	w.mu.Unlock()
	if wg != nil {
		wg.Wait()
	}
}

// drainResults returns all completed writes accumulated since the last call.
func (w *diskWriter) drainResults() []diskWriteResult {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := w.results
	w.results = nil
	return out
}

// shutdown closes the jobs channel and waits for all workers to finish.
// Remaining results can be read with drainResults.
func (w *diskWriter) shutdown() {
	close(w.jobs)
	w.done.Wait()
}
