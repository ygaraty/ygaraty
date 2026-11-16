package relay

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Handoffs (experiment 58, L3): a worker's result goes up as one batched PUT /blocks frame whose
// header also names it ("name", the hash the coordinator derives from the job, task and attempt)
// and carries its block list ("manifest", opaque here). A worker the coordinator has already
// handed the next task waits on POST /handoff/get for that name and gets the manifest and the
// blocks' bytes in the reply as soon as the PUT is stored: the producer's result post, the
// coordinator's commit and the consumer's claim reply are off the hop. The coordinator still
// commits every result and checks that the consumer read the attempt it committed.

const handoffTTL = 10 * time.Minute

type handoff struct {
	manifest json.RawMessage
	keys     []string
	regions  map[string][]int // block regions, when the manifest lists [{"sha", "region"}, ...]
	at       time.Time
}

// covers reports whether a block's region overlaps any of the wanted regions; a block whose
// region the manifest does not give is always sent.
func (e handoff) covers(key string, wanted [][]int) bool {
	r, ok := e.regions[key]
	if !ok || len(r) != 4 {
		return true
	}
	for _, w := range wanted {
		if len(w) == 4 && r[0] < w[1] && w[0] < r[1] && r[2] < w[3] && w[2] < r[3] {
			return true
		}
	}
	return false
}

// Handoffs maps names to published manifests and wakes the requests waiting for them.
type Handoffs struct {
	mu      sync.Mutex
	names   map[string]handoff
	changed chan struct{} // closed and replaced on every publish
}

func (h *Handoffs) init() {
	if h.names == nil {
		h.names, h.changed = make(map[string]handoff), make(chan struct{})
	}
}

// Publish records a name's manifest and the keys of its blocks (stored already). Entries older than
// handoffTTL are dropped here: a name is read once, right after it is published.
func (h *Handoffs) Publish(name string, manifest json.RawMessage, keys []string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.init()
	now := time.Now()
	for n, e := range h.names {
		if now.Sub(e.at) > handoffTTL {
			delete(h.names, n)
		}
	}
	var listed []struct {
		Sha    string `json:"sha"`
		Region []int  `json:"region"`
	}
	regions := map[string][]int{}
	if json.Unmarshal(manifest, &listed) == nil {
		for _, b := range listed {
			regions[b.Sha] = b.Region
		}
	}
	h.names[name] = handoff{manifest: append(json.RawMessage(nil), manifest...), keys: keys, regions: regions, at: now}
	close(h.changed)
	h.changed = make(chan struct{})
}

// lookup returns the entries of every name, or the names still missing and a channel closed at
// the next publish.
func (h *Handoffs) lookup(names []string) ([]handoff, []string, <-chan struct{}) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.init()
	found, missing := make([]handoff, 0, len(names)), []string{}
	for _, n := range names {
		if e, ok := h.names[n]; ok {
			found = append(found, e)
		} else {
			missing = append(missing, n)
		}
	}
	return found, missing, h.changed
}

// postHandoff answers {"names": [...], "keys": [...], "wait_ms": N[, "regions": {name: [[i, ei, j, ej], ...]}]}
// once every name is published (or 404 with the missing names after wait_ms, at most 60 s): a
// frame whose JSON header holds
// {"manifests": [...] (in the order of names), "blocks": [{key, size}]} followed by the bytes of
// every named result's blocks (only those overlapping its regions, when given) and of the extra
// keys, each key once.
func (rs *Server) postHandoff(w http.ResponseWriter, r *http.Request) {
	rs.handoffReads.Add(1)
	var body struct {
		Names   []string           `json:"names"`
		Keys    []string           `json:"keys"`
		WaitMS  int                `json:"wait_ms"`
		Regions map[string][][]int `json:"regions"` // optional: per name, only blocks overlapping these
	}
	// A worker with --gzip-frames gzips its JSON requests too (transports/http/compression.py).
	in := io.Reader(http.MaxBytesReader(w, r.Body, 16<<20))
	switch enc := strings.TrimSpace(r.Header.Get("Content-Encoding")); enc {
	case "", "identity":
	case "gzip":
		zr, err := gzip.NewReader(in)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid gzip body: " + err.Error()})
			return
		}
		defer zr.Close()
		in = http.MaxBytesReader(w, io.NopCloser(zr), 16<<20)
	default:
		writeJSON(w, http.StatusUnsupportedMediaType, map[string]string{"error": "handoff reads take an uncompressed or gzipped body"})
		return
	}
	if err := json.NewDecoder(in).Decode(&body); err != nil || len(body.Names) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "JSON body {\"names\": [...]} required"})
		return
	}
	for _, n := range body.Names {
		if !validKey(n) {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "names must be 64 lowercase hex characters"})
			return
		}
	}
	wait := time.Duration(min(max(body.WaitMS, 0), 60000)) * time.Millisecond
	timer := time.NewTimer(wait)
	defer timer.Stop()
	var found []handoff
	for {
		var missing []string
		var changed <-chan struct{}
		found, missing, changed = rs.Handoffs.lookup(body.Names)
		if len(missing) == 0 {
			break
		}
		select {
		case <-changed:
		case <-timer.C:
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "handoff not published", "missing": missing})
			return
		case <-r.Context().Done():
			return
		}
	}
	manifests, keys, seen := make([]json.RawMessage, len(found)), []string{}, map[string]bool{}
	for i, e := range found {
		manifests[i] = e.manifest
		wanted, filter := body.Regions[body.Names[i]]
		for _, k := range e.keys {
			if filter && !e.covers(k, wanted) {
				continue
			}
			if !seen[k] {
				seen[k] = true
				keys = append(keys, k)
			}
		}
	}
	for _, k := range body.Keys {
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	blocks, header, missing := make([][]byte, 0, len(keys)), make([]frameBlock, 0, len(keys)), []string{}
	for _, key := range keys {
		data, ok := rs.Store.Get(key)
		if !ok {
			missing = append(missing, key)
			continue
		}
		blocks = append(blocks, data)
		header = append(header, frameBlock{Key: key, Size: len(data)})
	}
	if len(missing) > 0 {
		writeJSON(w, http.StatusNotFound, map[string]any{"error": "unknown blocks", "missing": missing})
		return
	}
	encoded, _ := json.Marshal(map[string]any{"manifests": manifests, "blocks": header})
	pad := (-(4 + len(encoded))) & 7
	encoded = append(encoded, strings.Repeat(" ", pad)...)
	prefix := make([]byte, 4)
	binary.LittleEndian.PutUint32(prefix, uint32(len(encoded)))
	writeBlocks(w, r, append([][]byte{prefix, encoded}, blocks...)...)
}
