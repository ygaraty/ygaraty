package relay

import (
	"compress/gzip"
	"errors"
	"io"
	"net/http"
	"strings"
)

func (rs *Server) putBlock(w http.ResponseWriter, r *http.Request) {
	rs.writes.Add(1)
	key := r.PathValue("key")
	if !validKey(key) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": ErrBadKey.Error()})
		return
	}
	limit := rs.MaxBlockBytes
	if limit <= 0 {
		limit = 64 << 20
	}
	if r.ContentLength > limit {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "block exceeds the size limit"})
		return
	}
	var buf []byte
	if r.ContentLength > 0 {
		buf = make([]byte, 0, r.ContentLength)
	}
	body := io.Reader(http.MaxBytesReader(w, r.Body, limit))
	switch encoding := strings.ToLower(strings.TrimSpace(r.Header.Get("Content-Encoding"))); encoding {
	case "", "identity":
	case "gzip":
		// A gzipped block: the key is the hash of the decoded bytes, which the limit also bounds.
		zr, err := gzip.NewReader(body)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid gzip body"})
			return
		}
		defer zr.Close()
		body, buf = io.LimitReader(zr, limit+1), nil
	default:
		writeJSON(w, http.StatusUnsupportedMediaType, map[string]string{"error": "unsupported Content-Encoding " + encoding})
		return
	}
	data, err := readAll(body, buf)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "block exceeds the size limit"})
		} else {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid body: " + err.Error()})
		}
		return
	}
	if int64(len(data)) > limit {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "block exceeds the size limit"})
		return
	}
	if len(data) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "empty block"})
		return
	}
	// ?f64=1: the block must be little-endian float64 values, all finite. Workers that upload
	// results directly ask for this, so the coordinator need not read the bytes to check them.
	// ?f32=1 is the same check for float32 blocks (a job with a float32 wire).
	if r.URL.Query().Get("f64") == "1" && !finiteFloat64s(data) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "block is not finite float64 data"})
		return
	}
	if r.URL.Query().Get("f32") == "1" && !finiteFloat32s(data) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "block is not finite float32 data"})
		return
	}
	if r.URL.Query().Get("bf16") == "1" && !finiteBFloat16s(data) {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "block is not finite bf16 data"})
		return
	}
	created, err := rs.Store.Put(key, data)
	switch {
	case errors.Is(err, ErrHashMismatch):
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
	case errors.Is(err, ErrOverBudget):
		writeJSON(w, http.StatusInsufficientStorage, map[string]string{"error": err.Error()})
	case err != nil:
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
	case created:
		writeJSON(w, http.StatusCreated, map[string]any{"key": key, "size": len(data)})
	default:
		writeJSON(w, http.StatusOK, map[string]any{"key": key, "size": len(data), "duplicate": true})
	}
}

// readAll reads r into buf (grown as needed); io.ReadAll's growth strategy copies large bodies
// several times, so a Content-Length-sized buffer is read in place when it is known.
func readAll(r io.Reader, buf []byte) ([]byte, error) {
	if cap(buf) == 0 {
		return io.ReadAll(r)
	}
	for {
		n, err := r.Read(buf[len(buf):cap(buf)])
		buf = buf[:len(buf)+n]
		if err == io.EOF {
			return buf, nil
		}
		if err != nil {
			return nil, err
		}
		if len(buf) == cap(buf) {
			// Body longer than announced: fall back to a growing read of the rest.
			rest, err := io.ReadAll(r)
			if err != nil {
				return nil, err
			}
			return append(buf, rest...), nil
		}
	}
}
