package relay

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
)

type batchBlock struct {
	Key   string `json:"key"`
	Size  int    `json:"size"`
	Check string `json:"check"`
}

// putBlocks stores every block of one frame. Each is checked (key form, hash, size limit, the
// finite-value check its entry names) before any is stored, so a bad block stores nothing; a
// store over budget can still leave the earlier blocks stored (as separate PUTs would), and the
// reply says so. A repeat adds a reference per block, as PUT /blocks/{key} does.
func (rs *Server) putBlocks(w http.ResponseWriter, r *http.Request) {
	rs.batchWrites.Add(1)
	limit := rs.MaxBatchBytes
	if limit <= 0 {
		limit = 256 << 20
	}
	blockLimit := rs.MaxBlockBytes
	if blockLimit <= 0 {
		blockLimit = 64 << 20
	}
	if r.ContentLength > limit {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "batch exceeds the size limit"})
		return
	}
	// A gzipped frame (a slow-link worker's --gzip-frames) is inflated here, its inflated size
	// held to the same limit; blocks are stored and hashed as their raw bytes either way.
	body := io.Reader(http.MaxBytesReader(w, r.Body, limit))
	switch enc := strings.TrimSpace(r.Header.Get("Content-Encoding")); enc {
	case "", "identity":
	case "gzip":
		zr, err := gzip.NewReader(body)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid gzip body: " + err.Error()})
			return
		}
		defer zr.Close()
		body = http.MaxBytesReader(w, io.NopCloser(zr), limit)
	default:
		writeJSON(w, http.StatusUnsupportedMediaType, map[string]string{"error": "batched PUT takes an uncompressed or gzipped frame"})
		return
	}
	var buf []byte
	if r.ContentLength > 0 {
		buf = make([]byte, 0, r.ContentLength)
	}
	data, err := readAll(body, buf)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "batch exceeds the size limit"})
		} else {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid body: " + err.Error()})
		}
		return
	}
	bad := func(message string) { writeJSON(w, http.StatusBadRequest, map[string]string{"error": message}) }
	if len(data) < 4 {
		bad("frame too short")
		return
	}
	size := int(binary.LittleEndian.Uint32(data[:4]))
	if size > 1<<20 || 4+size > len(data) {
		bad("invalid frame header length")
		return
	}
	var header struct {
		Blocks   []batchBlock    `json:"blocks"`
		Name     string          `json:"name"`     // optional: publish the frame as a handoff (handoff.go)
		Manifest json.RawMessage `json:"manifest"` // with name: the result's block list, opaque here
	}
	if err := json.Unmarshal(data[4:4+size], &header); err != nil || len(header.Blocks) == 0 {
		bad("frame header must be JSON {\"blocks\": [...]}")
		return
	}
	offset, parts := 4+size, make([][]byte, len(header.Blocks))
	for i, b := range header.Blocks {
		if !validKey(b.Key) {
			bad(ErrBadKey.Error())
			return
		}
		if b.Size <= 0 || int64(b.Size) > blockLimit || offset+b.Size > len(data) {
			bad("block size invalid or past the end of the frame: " + b.Key)
			return
		}
		part := data[offset : offset+b.Size]
		offset += b.Size
		ok := true
		switch b.Check {
		case "":
		case "f64":
			ok = finiteFloat64s(part)
		case "f32":
			ok = finiteFloat32s(part)
		case "bf16":
			ok = finiteBFloat16s(part)
		default:
			bad("unknown check " + b.Check)
			return
		}
		if !ok {
			bad("block is not finite " + b.Check + " data: " + b.Key)
			return
		}
		sum := sha256.Sum256(part)
		if hex.EncodeToString(sum[:]) != b.Key {
			bad(ErrHashMismatch.Error() + ": " + b.Key)
			return
		}
		parts[i] = part
	}
	if offset != len(data) {
		bad("frame length does not match its blocks")
		return
	}
	if header.Name != "" && (!validKey(header.Name) || len(header.Manifest) == 0 || len(header.Manifest) > 1<<20) {
		bad("a handoff name must be 64 lowercase hex characters with a manifest")
		return
	}
	created, duplicates := 0, 0
	for i, b := range header.Blocks {
		// Copy out of the request buffer: each stored block owns its bytes.
		isNew, err := rs.Store.Put(b.Key, append([]byte(nil), parts[i]...))
		if err != nil {
			code := http.StatusBadRequest
			if errors.Is(err, ErrOverBudget) {
				code = http.StatusInsufficientStorage
			}
			writeJSON(w, code, map[string]any{"error": err.Error(), "stored": i})
			return
		}
		if isNew {
			created++
		} else {
			duplicates++
		}
	}
	if header.Name != "" {
		keys := make([]string, len(header.Blocks))
		for i, b := range header.Blocks {
			keys[i] = b.Key
		}
		rs.Handoffs.Publish(header.Name, header.Manifest, keys)
	}
	writeJSON(w, http.StatusOK, map[string]int{"created": created, "duplicates": duplicates})
}
