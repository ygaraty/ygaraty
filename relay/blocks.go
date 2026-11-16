package relay

import (
	"encoding/binary"
	"encoding/json"
	"net/http"
	"strings"
)

func (rs *Server) getBlock(w http.ResponseWriter, r *http.Request) {
	data, ok := rs.Store.Get(r.PathValue("key"))
	if !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "unknown block"})
		return
	}
	writeBlocks(w, r, data)
}

type frameBlock struct {
	Key  string `json:"key"`
	Size int    `json:"size"`
}

// getBlocks answers ?keys=a,b,c with one binary frame holding every block, in order.
func (rs *Server) getBlocks(w http.ResponseWriter, r *http.Request) {
	rs.reads.Add(1)
	raw := r.URL.Query().Get("keys")
	if raw == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "keys query parameter required"})
		return
	}
	rs.serveBlocks(w, r, strings.Split(raw, ","))
}

// postGetBlocks is getBlocks with the keys in a JSON body, at one URL for every request.
func (rs *Server) postGetBlocks(w http.ResponseWriter, r *http.Request) {
	rs.batchReads.Add(1)
	var body struct {
		Keys []string `json:"keys"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<20)).Decode(&body); err != nil || len(body.Keys) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "JSON body {\"keys\": [...]} required"})
		return
	}
	rs.serveBlocks(w, r, body.Keys)
}

func (rs *Server) serveBlocks(w http.ResponseWriter, r *http.Request, keys []string) {
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
	encoded, _ := json.Marshal(map[string]any{"blocks": header})
	pad := (-(4 + len(encoded))) & 7 // pad to an 8-byte boundary, as frames.py does
	encoded = append(encoded, strings.Repeat(" ", pad)...)
	prefix := make([]byte, 4)
	binary.LittleEndian.PutUint32(prefix, uint32(len(encoded)))
	writeBlocks(w, r, append([][]byte{prefix, encoded}, blocks...)...)
}
