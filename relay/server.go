package relay

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"
)

// Server is the HTTP front of a Store.
type Server struct {
	Store         *Store
	Token         string // required on every request
	MaxBlockBytes int64  // largest PUT body accepted
	MaxBatchBytes int64  // largest batched PUT /blocks body (default 256 MiB)
	// Origins of browser pages allowed to call the relay (experiment 50: the coordinator's origin,
	// whose contributor page runs the SDK); "*" allows any. Empty refuses cross-origin browsers:
	// no CORS headers, so a page on another origin cannot read the replies or send a PUT.
	AllowedOrigins []string
	started        time.Time
	Handoffs       Handoffs // named results a waiting worker reads as soon as they land (handoff.go)
	// Requests by kind, for the request patterns clients make (DO-2): preflights, single and batched
	// reads and writes.
	preflights, reads, batchReads, writes, batchWrites, handoffReads atomic.Int64
}

// Handler routes the relay's endpoints. Go 1.22+ method patterns, no framework needed.
func (rs *Server) Handler() http.Handler {
	if rs.started.IsZero() {
		rs.started = time.Now()
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /blocks/{key}", rs.getBlock)
	mux.HandleFunc("GET /blocks", rs.getBlocks)
	mux.HandleFunc("POST /blocks/get", rs.postGetBlocks)
	mux.HandleFunc("PUT /blocks", rs.putBlocks)
	mux.HandleFunc("PUT /blocks/{key}", rs.putBlock)
	mux.HandleFunc("POST /handoff/get", rs.postHandoff)
	mux.HandleFunc("DELETE /blocks", rs.deleteBlocks)
	mux.HandleFunc("GET /status", rs.status)
	return rs.cors(rs.authenticate(mux))
}

func (rs *Server) deleteBlocks(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Keys []string `json:"keys"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<20)).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "JSON body {\"keys\": [...]} required"})
		return
	}
	present, freed := rs.Store.Delete(body.Keys)
	writeJSON(w, http.StatusOK, map[string]int{"deleted": present, "freed": freed})
}

func (rs *Server) status(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, struct {
		Status
		Requests map[string]int64 `json:"requests"`
	}{rs.Store.Status(rs.started), map[string]int64{"preflights": rs.preflights.Load(), "reads": rs.reads.Load(),
		"batch_reads": rs.batchReads.Load(), "writes": rs.writes.Load(), "batch_writes": rs.batchWrites.Load(), "handoff_reads": rs.handoffReads.Load()}})
}
