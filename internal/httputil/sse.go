package httputil

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// WriteSSE writes a Server-Sent Event with the given type and JSON-encoded data,
// flushing the response if the writer supports it.
func WriteSSE(w http.ResponseWriter, eventType string, data any) error {
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, d); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}
