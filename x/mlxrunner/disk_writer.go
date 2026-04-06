package mlxrunner

import (
	"path/filepath"
	"sync"
)

type diskWriteJob struct {
	data     []byte    // pre-serialized safetensors bytes
	filename string    // content-addressed name (e.g. "abc123.safetensors")
	dir      string    // cache directory path
	node     *trieNode // passed through to result (not dereferenced by worker)
}

type diskWriteResult struct {
	node     *trieNode
	fileSize int64
	err      error
}

// diskWriter runs background goroutines that write pre-serialized bytes to
// content-addressed files. The inference goroutine serializes arrays and submits
// jobs; workers handle only pure Go file I/O. Content-addressed filenames
// guarantee no write conflicts between workers.
type diskWriter struct {
	jobs     chan diskWriteJob    // buffered: absorbs burst evictions without blocking inference
	results  chan diskWriteResult // buffered: same size as jobs
	inFlight sync.Map            // filename -> *sync.WaitGroup (for load-race handling)
	done     sync.WaitGroup      // tracks all workers for graceful shutdown
}

func newDiskWriter() *diskWriter {
	w := &diskWriter{
		jobs:    make(chan diskWriteJob, 8),
		results: make(chan diskWriteResult, 8),
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
		tmpPath := filepath.Join(job.dir, ".tmp_"+job.filename)
		finalPath := filepath.Join(job.dir, job.filename)
		err := atomicWriteFile(tmpPath, finalPath, job.data)

		// Signal waiters and clean up inFlight entry.
		if v, ok := w.inFlight.LoadAndDelete(job.filename); ok {
			v.(*sync.WaitGroup).Done()
		}

		w.results <- diskWriteResult{
			node:     job.node,
			fileSize: int64(len(job.data)),
			err:      err,
		}
	}
}

// submit enqueues a write job. The inFlight WaitGroup is registered before
// sending to the channel so waitForFile is correct even if called before
// the worker picks up the job.
func (w *diskWriter) submit(job diskWriteJob) {
	var wg sync.WaitGroup
	wg.Add(1)
	w.inFlight.Store(job.filename, &wg)
	w.jobs <- job
}

// waitForFile blocks until the named file's async write completes. Returns
// immediately if the file is not in-flight (sync.Map lookup miss -- fast path).
func (w *diskWriter) waitForFile(filename string) {
	if v, ok := w.inFlight.Load(filename); ok {
		v.(*sync.WaitGroup).Wait()
	}
}

// shutdown closes the jobs channel and waits for all workers to finish.
// Results remain in the channel for the caller to drain via processDiskCompletions.
func (w *diskWriter) shutdown() {
	close(w.jobs)
	w.done.Wait()
}
