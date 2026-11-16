package relay

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

// gzipMinBytes: below this the gzip header and the CPU are not worth it.
const gzipMinBytes = 1024

// acceptsGzip reports whether an Accept-Encoding header allows gzip; a q=0 entry refuses it.
func acceptsGzip(header string) bool {
	for _, item := range strings.Split(header, ",") {
		name, params, _ := strings.Cut(strings.TrimSpace(item), ";")
		name = strings.ToLower(strings.TrimSpace(name))
		if name != "gzip" && name != "*" {
			continue
		}
		q, ok := strings.CutPrefix(strings.ToLower(strings.TrimSpace(params)), "q=")
		if !ok {
			return true
		}
		value, err := strconv.ParseFloat(q, 64)
		return err == nil && value > 0
	}
	return false
}

// writeBlocks sends parts as one octet-stream body, gzipped (fastest level) when the client
// accepts it and it comes out smaller. Float blocks shrink little; the client chooses.
func writeBlocks(w http.ResponseWriter, r *http.Request, parts ...[]byte) {
	h := w.Header()
	h.Set("Content-Type", "application/octet-stream")
	h.Set("Cache-Control", "no-store")
	h.Set("Vary", "Accept-Encoding")
	total := 0
	for _, part := range parts {
		total += len(part)
	}
	if total >= gzipMinBytes && acceptsGzip(r.Header.Get("Accept-Encoding")) {
		var packed bytes.Buffer
		zw, _ := gzip.NewWriterLevel(&packed, gzip.BestSpeed)
		for _, part := range parts {
			zw.Write(part)
		}
		if zw.Close() == nil && packed.Len() < total {
			h.Set("Content-Encoding", "gzip")
			h.Set("Content-Length", itoa(packed.Len()))
			w.Write(packed.Bytes())
			return
		}
	}
	h.Set("Content-Length", itoa(total))
	for _, part := range parts {
		w.Write(part)
	}
}

func itoa(n int) string {
	b := make([]byte, 0, 20)
	if n == 0 {
		return "0"
	}
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
