package relay

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const token = "test-token"

func newTestServer(t *testing.T, maxBytes int64) (*httptest.Server, *Store) {
	t.Helper()
	store := NewStore(maxBytes)
	rs := &Server{Store: store, Token: token, MaxBlockBytes: 4 << 20}
	ts := httptest.NewServer(rs.Handler())
	t.Cleanup(ts.Close)
	return ts, store
}

func do(t *testing.T, method, url string, body []byte, auth bool) (*http.Response, []byte) {
	t.Helper()
	req, _ := http.NewRequest(method, url, bytes.NewReader(body))
	if auth {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := io.ReadAll(res.Body)
	res.Body.Close()
	return res, data
}

func block(n int, seed byte) (string, []byte) {
	data := make([]byte, n)
	for i := range data {
		data[i] = byte(i) ^ seed
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), data
}

func TestPutGetRoundTrip(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	key, data := block(4096, 7)
	res, _ := do(t, "PUT", ts.URL+"/blocks/"+key, data, true)
	if res.StatusCode != http.StatusCreated {
		t.Fatalf("put: %d", res.StatusCode)
	}
	res, _ = do(t, "PUT", ts.URL+"/blocks/"+key, data, true)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("duplicate put should be 200, got %d", res.StatusCode)
	}
	res, got := do(t, "GET", ts.URL+"/blocks/"+key, nil, true)
	if res.StatusCode != http.StatusOK || !bytes.Equal(got, data) || res.Header.Get("Content-Type") != "application/octet-stream" {
		t.Fatalf("get: %d, %d bytes", res.StatusCode, len(got))
	}
}

func TestRejectsWrongHashBadKeyAndEmptyBody(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	key, data := block(100, 1)
	data[0] ^= 0xff
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+key, data, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("hash mismatch should be 400, got %d", res.StatusCode)
	}
	if res, _ := do(t, "PUT", ts.URL+"/blocks/not-a-hash", data, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad key should be 400, got %d", res.StatusCode)
	}
	key, _ = block(0, 0)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+key, nil, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("empty body should be 400, got %d", res.StatusCode)
	}
}

func TestUnknownBlockAndAuth(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	key, _ := block(10, 3)
	if res, _ := do(t, "GET", ts.URL+"/blocks/"+key, nil, true); res.StatusCode != http.StatusNotFound {
		t.Fatalf("missing should be 404, got %d", res.StatusCode)
	}
	if res, _ := do(t, "GET", ts.URL+"/status", nil, false); res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token should be 401, got %d", res.StatusCode)
	}
}

func TestMultiGetFrame(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	k1, d1 := block(24, 1)
	k2, d2 := block(4000, 2)
	do(t, "PUT", ts.URL+"/blocks/"+k1, d1, true)
	do(t, "PUT", ts.URL+"/blocks/"+k2, d2, true)
	res, frame := do(t, "GET", ts.URL+"/blocks?keys="+k2+","+k1, nil, true)
	if res.StatusCode != http.StatusOK {
		t.Fatalf("multi get: %d %s", res.StatusCode, frame)
	}
	n := int(binary.LittleEndian.Uint32(frame[:4]))
	if (4+n)%8 != 0 {
		t.Fatalf("floats must start on an 8-byte boundary, header length %d", n)
	}
	var header struct {
		Blocks []frameBlock `json:"blocks"`
	}
	if err := json.Unmarshal(frame[4:4+n], &header); err != nil {
		t.Fatal(err)
	}
	if len(header.Blocks) != 2 || header.Blocks[0].Key != k2 || header.Blocks[1].Key != k1 {
		t.Fatalf("header order: %+v", header.Blocks)
	}
	rest := frame[4+n:]
	if !bytes.Equal(rest[:4000], d2) || !bytes.Equal(rest[4000:], d1) {
		t.Fatal("frame payload mismatch")
	}
	missing, _ := block(1, 9)
	res, body := do(t, "GET", ts.URL+"/blocks?keys="+k1+","+missing, nil, true)
	if res.StatusCode != http.StatusNotFound || !strings.Contains(string(body), missing) {
		t.Fatalf("missing key should be 404 listing it: %d %s", res.StatusCode, body)
	}
}

func TestBudgetDeleteAndStatus(t *testing.T) {
	ts, store := newTestServer(t, 6000)
	k1, d1 := block(4000, 1)
	k2, d2 := block(4000, 2)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+k1, d1, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("first put: %d", res.StatusCode)
	}
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+k2, d2, true); res.StatusCode != http.StatusInsufficientStorage {
		t.Fatalf("over budget should be 507, got %d", res.StatusCode)
	}
	body, _ := json.Marshal(map[string][]string{"keys": {k1, k2}})
	res, out := do(t, "DELETE", ts.URL+"/blocks", body, true)
	if res.StatusCode != http.StatusOK || string(bytes.TrimSpace(out)) != `{"deleted":1,"freed":1}` {
		t.Fatalf("delete: %d %s", res.StatusCode, out)
	}
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+k2, d2, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("put after delete: %d", res.StatusCode)
	}
	_, out = do(t, "GET", ts.URL+"/status", nil, true)
	var status Status
	json.Unmarshal(out, &status)
	if status.Blocks != 1 || status.Bytes != 4000 || status.Puts != 2 || status.Deletes != 1 || store.MaxBytes != 6000 {
		t.Fatalf("status: %+v", status)
	}
}

func TestReferenceCounts(t *testing.T) {
	ts, store := newTestServer(t, 0)
	key, data := block(512, 6)
	do(t, "PUT", ts.URL+"/blocks/"+key, data, true)
	do(t, "PUT", ts.URL+"/blocks/"+key, data, true) // a second version with identical bytes
	body, _ := json.Marshal(map[string][]string{"keys": {key}})
	if _, out := do(t, "DELETE", ts.URL+"/blocks", body, true); string(bytes.TrimSpace(out)) != `{"deleted":1,"freed":0}` {
		t.Fatalf("first release should keep the block: %s", out)
	}
	if res, _ := do(t, "GET", ts.URL+"/blocks/"+key, nil, true); res.StatusCode != http.StatusOK {
		t.Fatalf("block should survive one release, got %d", res.StatusCode)
	}
	if _, out := do(t, "DELETE", ts.URL+"/blocks", body, true); string(bytes.TrimSpace(out)) != `{"deleted":1,"freed":1}` {
		t.Fatalf("second release should free the block: %s", out)
	}
	if res, _ := do(t, "GET", ts.URL+"/blocks/"+key, nil, true); res.StatusCode != http.StatusNotFound || store.Status(time.Now()).Blocks != 0 {
		t.Fatalf("block should be gone, got %d", res.StatusCode)
	}
}

func TestFloat64Check(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	good := make([]byte, 16)
	binary.LittleEndian.PutUint64(good[8:], math.Float64bits(2.5))
	sum := sha256.Sum256(good)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f64=1", good, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("finite floats should be accepted, got %d", res.StatusCode)
	}
	bad := make([]byte, 16)
	binary.LittleEndian.PutUint64(bad[:], math.Float64bits(math.NaN()))
	sum = sha256.Sum256(bad)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f64=1", bad, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("NaN should be rejected, got %d", res.StatusCode)
	}
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:]), bad, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("without the check any bytes are accepted, got %d", res.StatusCode)
	}
	odd := []byte{1, 2, 3}
	sum = sha256.Sum256(odd)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f64=1", odd, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("a length that is not a multiple of 8 should be rejected, got %d", res.StatusCode)
	}
}

func TestFloat32Check(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	good := make([]byte, 8)
	binary.LittleEndian.PutUint32(good[4:], math.Float32bits(2.5))
	sum := sha256.Sum256(good)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f32=1", good, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("finite float32 values should be accepted, got %d", res.StatusCode)
	}
	bad := make([]byte, 8)
	binary.LittleEndian.PutUint32(bad[:], math.Float32bits(float32(math.Inf(1))))
	sum = sha256.Sum256(bad)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f32=1", bad, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("Inf should be rejected, got %d", res.StatusCode)
	}
	odd := []byte{1, 2, 3, 4, 5, 6}
	sum = sha256.Sum256(odd)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?f32=1", odd, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("a length that is not a multiple of 4 should be rejected, got %d", res.StatusCode)
	}
}

func TestBFloat16Check(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	good := []byte{0x00, 0x00, 0x20, 0x40} // 0 and 2.5 as bf16 (the top halves of their float32 bits)
	sum := sha256.Sum256(good)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?bf16=1", good, true); res.StatusCode != http.StatusCreated {
		t.Fatalf("finite bf16 values should be accepted, got %d", res.StatusCode)
	}
	bad := []byte{0x80, 0x7F} // +Inf
	sum = sha256.Sum256(bad)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?bf16=1", bad, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("Inf should be rejected, got %d", res.StatusCode)
	}
	odd := []byte{1, 2, 3}
	sum = sha256.Sum256(odd)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+hex.EncodeToString(sum[:])+"?bf16=1", odd, true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("an odd length should be rejected, got %d", res.StatusCode)
	}
}

func TestTooLargeBlock(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	key, data := block(5<<20, 4) // the test server accepts 4 MiB
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+key, data, true); res.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("too large should be 413, got %d", res.StatusCode)
	}
}

// BenchmarkGet2MiB measures loopback throughput for a typical tile (512x512 float64 = 2 MiB),
// with keep-alive connections and parallel clients: go test -bench . -benchmem
func BenchmarkGet2MiB(b *testing.B) {
	store := NewStore(0)
	rs := &Server{Store: store, Token: token}
	ts := httptest.NewServer(rs.Handler())
	defer ts.Close()
	key, data := block(2<<20, 5)
	store.Put(key, data)
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 64}}
	b.SetBytes(int64(len(data)))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, 2<<20)
		for pb.Next() {
			req, _ := http.NewRequest("GET", ts.URL+"/blocks/"+key, nil)
			req.Header.Set("Authorization", "Bearer "+token)
			res, err := client.Do(req)
			if err != nil {
				b.Fatal(err)
			}
			if _, err := io.ReadFull(res.Body, buf); err != nil {
				b.Fatal(fmt.Errorf("read: %w", err))
			}
			res.Body.Close()
		}
	})
}

func TestGzipNegotiation(t *testing.T) {
	cases := map[string]bool{"": false, "gzip": true, "deflate, gzip;q=0.5": true, "gzip;q=0": false,
		"GZIP": true, "*": true, "identity": false, "gzip;q=x": false, "br, *;q=0": false}
	for header, want := range cases {
		if got := acceptsGzip(header); got != want {
			t.Errorf("acceptsGzip(%q) = %v, want %v", header, got, want)
		}
	}
	ts, _ := newTestServer(t, 0)
	key, data := block(8192, 5) // byte(i)^seed repeats every 256 bytes: compresses well
	do(t, "PUT", ts.URL+"/blocks/"+key, data, true)
	small, tiny := block(100, 6)
	do(t, "PUT", ts.URL+"/blocks/"+small, tiny, true)
	get := func(path, accept string) (*http.Response, []byte) {
		t.Helper()
		req, _ := http.NewRequest("GET", ts.URL+path, nil)
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Accept-Encoding", accept) // set by hand, so Go's transport does not decode it
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer res.Body.Close()
		body, _ := io.ReadAll(res.Body)
		return res, body
	}
	gunzip := func(body []byte) []byte {
		t.Helper()
		zr, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(zr)
		return out
	}
	res, body := get("/blocks/"+key, "gzip")
	if res.Header.Get("Content-Encoding") != "gzip" || len(body) >= len(data) || !bytes.Equal(gunzip(body), data) ||
		res.Header.Get("Content-Length") != fmt.Sprint(len(body)) || res.Header.Get("Vary") != "Accept-Encoding" {
		t.Fatalf("gzip block: encoding %q, %d bytes", res.Header.Get("Content-Encoding"), len(body))
	}
	if res, body = get("/blocks/"+key, "identity"); res.Header.Get("Content-Encoding") != "" || !bytes.Equal(body, data) {
		t.Fatal("a client that does not accept gzip must get the raw bytes")
	}
	if res, body = get("/blocks/"+small, "gzip"); res.Header.Get("Content-Encoding") != "" || !bytes.Equal(body, tiny) {
		t.Fatal("bodies under 1 KiB must stay raw")
	}
	_, raw := get("/blocks?keys="+key+","+small, "identity")
	if res, body = get("/blocks?keys="+key+","+small, "gzip"); res.Header.Get("Content-Encoding") != "gzip" || !bytes.Equal(gunzip(body), raw) {
		t.Fatal("a gzipped frame must decode to the raw frame")
	}
	noise := make([]byte, 4096) // incompressible: sent raw even though gzip was accepted
	for i := range noise {
		noise[i] = byte(sha256.Sum256([]byte{byte(i), byte(i >> 8)})[0])
	}
	sum := sha256.Sum256(noise)
	noisy := hex.EncodeToString(sum[:])
	do(t, "PUT", ts.URL+"/blocks/"+noisy, noise, true)
	if res, body = get("/blocks/"+noisy, "gzip"); res.Header.Get("Content-Encoding") != "" || !bytes.Equal(body, noise) {
		t.Fatal("a body gzip does not shrink must stay raw")
	}
}

func TestGzippedPut(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	put := func(key string, body []byte, encoding string) int {
		t.Helper()
		req, _ := http.NewRequest("PUT", ts.URL+"/blocks/"+key, bytes.NewReader(body))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Encoding", encoding)
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
		return res.StatusCode
	}
	packed := func(data []byte) []byte {
		var buf bytes.Buffer
		zw := gzip.NewWriter(&buf)
		zw.Write(data)
		zw.Close()
		return buf.Bytes()
	}
	key, data := block(8192, 3)
	if code := put(key, packed(data), "gzip"); code != http.StatusCreated {
		t.Fatalf("gzipped put: %d", code)
	}
	if res, got := do(t, "GET", ts.URL+"/blocks/"+key, nil, true); res.StatusCode != http.StatusOK || !bytes.Equal(got, data) {
		t.Fatal("a gzipped put must store the decoded bytes under their hash")
	}
	if code := put(key, packed(append([]byte{1}, data...)), "gzip"); code != http.StatusBadRequest {
		t.Fatalf("the key is the hash of the decoded bytes: %d", code)
	}
	big, bomb := block(5<<20, 4) // 5 MiB of a repeating pattern: a few KiB gzipped, over the 4 MiB limit decoded
	if code := put(big, packed(bomb), "gzip"); code != http.StatusRequestEntityTooLarge {
		t.Fatalf("decoded size must respect the limit: %d", code)
	}
	if code := put(key, []byte("not gzip at all"), "gzip"); code != http.StatusBadRequest {
		t.Fatalf("corrupt gzip: %d", code)
	}
	if code := put(key, packed(data)[:100], "gzip"); code != http.StatusBadRequest {
		t.Fatalf("truncated gzip: %d", code)
	}
	if code := put(key, data, "br"); code != http.StatusUnsupportedMediaType {
		t.Fatalf("unknown encoding: %d", code)
	}
}

// A browser page on an allowed origin gets a preflight answer and CORS headers on every reply; other
// origins get nothing (and a 403 preflight); the token is still required on real requests.
func TestCORS(t *testing.T) {
	store := NewStore(1 << 20)
	rs := &Server{Store: store, Token: "t", AllowedOrigins: []string{"http://127.0.0.1:8090"}}
	ts := httptest.NewServer(rs.Handler())
	defer ts.Close()
	preflight := func(origin string) *http.Response {
		req, _ := http.NewRequest(http.MethodOptions, ts.URL+"/blocks/abc", nil)
		req.Header.Set("Origin", origin)
		req.Header.Set("Access-Control-Request-Method", "PUT")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp
	}
	if resp := preflight("http://127.0.0.1:8090"); resp.StatusCode != http.StatusNoContent ||
		resp.Header.Get("Access-Control-Allow-Origin") != "http://127.0.0.1:8090" ||
		!strings.Contains(resp.Header.Get("Access-Control-Allow-Headers"), "Authorization") ||
		resp.Header.Get("Access-Control-Max-Age") == "" {
		t.Fatalf("preflight for the allowed origin: %d %v", resp.StatusCode, resp.Header)
	}
	if resp := preflight("http://evil.example"); resp.StatusCode != http.StatusForbidden || resp.Header.Get("Access-Control-Allow-Origin") != "" {
		t.Fatalf("preflight for another origin: %d %v", resp.StatusCode, resp.Header)
	}
	key, data := block(64, 1)
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/blocks/"+key, bytes.NewReader(data))
	req.Header.Set("Origin", "http://127.0.0.1:8090")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized || resp.Header.Get("Access-Control-Allow-Origin") != "http://127.0.0.1:8090" {
		t.Fatalf("a PUT without a token from the allowed origin: %d %v", resp.StatusCode, resp.Header)
	}
	req, _ = http.NewRequest(http.MethodPut, ts.URL+"/blocks/"+key, bytes.NewReader(data))
	req.Header.Set("Origin", "http://127.0.0.1:8090")
	req.Header.Set("Authorization", "Bearer t")
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusCreated && resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT with the token: %d", resp.StatusCode)
	}
	if resp.Header.Get("Access-Control-Allow-Origin") != "http://127.0.0.1:8090" || !strings.Contains(resp.Header.Get("Vary"), "Origin") {
		t.Fatalf("CORS headers on a real reply: %v", resp.Header)
	}
}

func batchFrame(t *testing.T, entries []batchBlock, parts ...[]byte) []byte {
	t.Helper()
	header, _ := json.Marshal(map[string]any{"blocks": entries})
	frame := binary.LittleEndian.AppendUint32(nil, uint32(len(header)))
	frame = append(frame, header...)
	for _, p := range parts {
		frame = append(frame, p...)
	}
	return frame
}

func TestBatchedPutAndPostGet(t *testing.T) {
	ts, store := newTestServer(t, 0)
	k1, d1 := block(24, 1)
	k2, d2 := block(4000, 2)
	frame := batchFrame(t, []batchBlock{{Key: k1, Size: len(d1)}, {Key: k2, Size: len(d2)}}, d1, d2)
	res, body := do(t, "PUT", ts.URL+"/blocks", frame, true)
	if res.StatusCode != http.StatusOK || !strings.Contains(string(body), `"created":2`) {
		t.Fatalf("batched put: %d %s", res.StatusCode, body)
	}
	res, body = do(t, "PUT", ts.URL+"/blocks", frame, true) // a repeat adds references
	if res.StatusCode != http.StatusOK || !strings.Contains(string(body), `"duplicates":2`) {
		t.Fatalf("repeat: %d %s", res.StatusCode, body)
	}
	if got, ok := store.Get(k2); !ok || !bytes.Equal(got, d2) {
		t.Fatal("stored bytes differ")
	}
	keys, _ := json.Marshal(map[string]any{"keys": []string{k2, k1}})
	res, got := do(t, "POST", ts.URL+"/blocks/get", keys, true)
	_, want := do(t, "GET", ts.URL+"/blocks?keys="+k2+","+k1, nil, true)
	if res.StatusCode != http.StatusOK || !bytes.Equal(got, want) {
		t.Fatalf("POST /blocks/get must answer as GET /blocks: %d", res.StatusCode)
	}
	missing, _ := block(1, 9)
	keys, _ = json.Marshal(map[string]any{"keys": []string{k1, missing}})
	if res, body := do(t, "POST", ts.URL+"/blocks/get", keys, true); res.StatusCode != http.StatusNotFound || !strings.Contains(string(body), missing) {
		t.Fatalf("missing key: %d %s", res.StatusCode, body)
	}
	if res, _ := do(t, "PUT", ts.URL+"/blocks", frame, false); res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("batched put needs the token: %d", res.StatusCode)
	}
}

func TestBatchedPutStoresNothingWhenOneBlockIsBad(t *testing.T) {
	good, d1 := block(16, 3)
	nan := make([]byte, 8)
	binary.LittleEndian.PutUint64(nan, math.Float64bits(math.NaN()))
	sum := sha256.Sum256(nan)
	nanKey := hex.EncodeToString(sum[:])
	other, _ := block(16, 4)
	cases := map[string][]byte{
		"hash mismatch": batchFrame(t, []batchBlock{{Key: good, Size: 16}, {Key: other, Size: 16}}, d1, d1),
		"not finite":    batchFrame(t, []batchBlock{{Key: good, Size: 16, Check: "f64"}, {Key: nanKey, Size: 8, Check: "f64"}}, d1, nan),
		"short frame":   batchFrame(t, []batchBlock{{Key: good, Size: 16}, {Key: other, Size: 16}}, d1),
		"extra bytes":   batchFrame(t, []batchBlock{{Key: good, Size: 16}}, d1, []byte{0}),
		"bad check":     batchFrame(t, []batchBlock{{Key: good, Size: 16, Check: "f16"}}, d1),
		"bad key":       batchFrame(t, []batchBlock{{Key: "abc", Size: 16}}, d1),
	}
	for name, frame := range cases {
		ts, store := newTestServer(t, 0)
		res, body := do(t, "PUT", ts.URL+"/blocks", frame, true)
		if res.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s: want 400, got %d %s", name, res.StatusCode, body)
		}
		if _, ok := store.Get(good); ok {
			t.Fatalf("%s: a good block was stored from a rejected batch", name)
		}
	}
}

func TestCORSAllowsBatchedMethods(t *testing.T) {
	rs := &Server{Store: NewStore(0), Token: token, AllowedOrigins: []string{"http://page.test"}}
	ts := httptest.NewServer(rs.Handler())
	defer ts.Close()
	req, _ := http.NewRequest("OPTIONS", ts.URL+"/blocks/get", nil)
	req.Header.Set("Origin", "http://page.test")
	req.Header.Set("Access-Control-Request-Method", "POST")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()
	if res.StatusCode != http.StatusNoContent || !strings.Contains(res.Header.Get("Access-Control-Allow-Methods"), "POST") {
		t.Fatalf("preflight: %d %q", res.StatusCode, res.Header.Get("Access-Control-Allow-Methods"))
	}
}

func TestBatchedPutAcceptsGzip(t *testing.T) {
	ts, store := newTestServer(t, 0)
	k1, d1 := block(4000, 5)
	var zipped bytes.Buffer
	zw := gzip.NewWriter(&zipped)
	zw.Write(batchFrame(t, []batchBlock{{Key: k1, Size: len(d1), Check: "f64"}}, d1))
	zw.Close()
	req, _ := http.NewRequest("PUT", ts.URL+"/blocks", bytes.NewReader(zipped.Bytes()))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Encoding", "gzip")
	res, err := http.DefaultClient.Do(req)
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("gzipped batch: %v %v", err, res)
	}
	if got, ok := store.Get(k1); !ok || !bytes.Equal(got, d1) {
		t.Fatal("gzipped batch stored different bytes")
	}
	req, _ = http.NewRequest("PUT", ts.URL+"/blocks", bytes.NewReader([]byte("not gzip")))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Encoding", "gzip")
	if res, _ := http.DefaultClient.Do(req); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("bad gzip: %d", res.StatusCode)
	}
}

func TestHandoffWaitsForTheNamedBatch(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	k1, d1 := block(24, 4)
	extra, d2 := block(16, 5)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+extra, d2, true); res.StatusCode != http.StatusCreated && res.StatusCode != http.StatusOK {
		t.Fatalf("put extra: %d", res.StatusCode)
	}
	name := strings.Repeat("ab", 32)
	type reply struct {
		code int
		body []byte
	}
	done := make(chan reply, 1)
	go func() { // waits before the name exists
		req, _ := json.Marshal(map[string]any{"names": []string{name}, "keys": []string{extra}, "wait_ms": 5000})
		res, body := do(t, "POST", ts.URL+"/handoff/get", req, true)
		done <- reply{res.StatusCode, body}
	}()
	time.Sleep(50 * time.Millisecond)
	header, _ := json.Marshal(map[string]any{"blocks": []batchBlock{{Key: k1, Size: len(d1)}}, "name": name,
		"manifest": []map[string]any{{"sha": k1, "shape": []int{1, 3}, "region": []int{0, 1, 0, 3}}}})
	frame := append(binary.LittleEndian.AppendUint32(nil, uint32(len(header))), header...)
	frame = append(frame, d1...)
	if res, body := do(t, "PUT", ts.URL+"/blocks", frame, true); res.StatusCode != http.StatusOK {
		t.Fatalf("named put: %d %s", res.StatusCode, body)
	}
	var got reply
	select {
	case got = <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the waiting handoff read was not woken by the publish")
	}
	if got.code != http.StatusOK {
		t.Fatalf("handoff read: %d %s", got.code, got.body)
	}
	size := int(binary.LittleEndian.Uint32(got.body[:4]))
	var h struct {
		Manifests []json.RawMessage `json:"manifests"`
		Blocks    []frameBlock      `json:"blocks"`
	}
	if err := json.Unmarshal(got.body[4:4+size], &h); err != nil || len(h.Manifests) != 1 || len(h.Blocks) != 2 ||
		h.Blocks[0].Key != k1 || h.Blocks[1].Key != extra || !strings.Contains(string(h.Manifests[0]), k1) {
		t.Fatalf("handoff frame header: %v %s", err, got.body[4:4+size])
	}
	if !bytes.Equal(got.body[4+size:], append(append([]byte(nil), d1...), d2...)) {
		t.Fatal("handoff frame bytes differ from the blocks")
	}
	req, _ := json.Marshal(map[string]any{"names": []string{strings.Repeat("cd", 32)}, "wait_ms": 20})
	if res, body := do(t, "POST", ts.URL+"/handoff/get", req, true); res.StatusCode != http.StatusNotFound || !strings.Contains(string(body), "missing") {
		t.Fatalf("unpublished name must time out with 404: %d %s", res.StatusCode, body)
	}
	bad, _ := json.Marshal(map[string]any{"blocks": []batchBlock{{Key: k1, Size: len(d1)}}, "name": "short", "manifest": []int{1}})
	frame = append(binary.LittleEndian.AppendUint32(nil, uint32(len(bad))), bad...)
	if res, _ := do(t, "PUT", ts.URL+"/blocks", append(frame, d1...), true); res.StatusCode != http.StatusBadRequest {
		t.Fatalf("a bad handoff name must be refused: %d", res.StatusCode)
	}
}

func TestHandoffSendsOnlyTheBlocksTheRegionsCover(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	k1, d1 := block(16, 6)
	k2, d2 := block(16, 7)
	name := strings.Repeat("ef", 32)
	header, _ := json.Marshal(map[string]any{"blocks": []batchBlock{{Key: k1, Size: len(d1)}, {Key: k2, Size: len(d2)}}, "name": name,
		"manifest": []map[string]any{{"sha": k1, "region": []int{0, 2, 0, 1}}, {"sha": k2, "region": []int{2, 4, 0, 1}}}})
	frame := append(binary.LittleEndian.AppendUint32(nil, uint32(len(header))), header...)
	frame = append(append(frame, d1...), d2...)
	if res, body := do(t, "PUT", ts.URL+"/blocks", frame, true); res.StatusCode != http.StatusOK {
		t.Fatalf("named put: %d %s", res.StatusCode, body)
	}
	for _, c := range []struct {
		regions any
		want    []string
	}{{map[string]any{name: [][]int{{3, 4, 0, 1}}}, []string{k2}}, {map[string]any{name: [][]int{}}, []string{}}, {nil, []string{k1, k2}}} {
		req, _ := json.Marshal(map[string]any{"names": []string{name}, "regions": c.regions})
		res, body := do(t, "POST", ts.URL+"/handoff/get", req, true)
		if res.StatusCode != http.StatusOK {
			t.Fatalf("handoff read: %d %s", res.StatusCode, body)
		}
		size := int(binary.LittleEndian.Uint32(body[:4]))
		var h struct {
			Blocks []frameBlock `json:"blocks"`
		}
		json.Unmarshal(body[4:4+size], &h)
		got := []string{}
		for _, b := range h.Blocks {
			got = append(got, b.Key)
		}
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Fatalf("regions %v: blocks %v, want %v", c.regions, got, c.want)
		}
	}
}

func TestHandoffAcceptsGzippedRequest(t *testing.T) {
	ts, _ := newTestServer(t, 0)
	extra, d2 := block(16, 5)
	if res, _ := do(t, "PUT", ts.URL+"/blocks/"+extra, d2, true); res.StatusCode != http.StatusCreated && res.StatusCode != http.StatusOK {
		t.Fatalf("put extra: %d", res.StatusCode)
	}
	name := strings.Repeat("cd", 32)
	k1, d1 := block(24, 4)
	head, _ := json.Marshal(map[string]any{"name": name, "manifest": []any{}, "blocks": []batchBlock{{Key: k1, Size: len(d1), Check: "f64"}}})
	frame := binary.LittleEndian.AppendUint32(nil, uint32(len(head)))
	frame = append(append(frame, head...), d1...)
	if res, _ := do(t, "PUT", ts.URL+"/blocks", frame, true); res.StatusCode != http.StatusOK {
		t.Fatalf("publish: %d", res.StatusCode)
	}
	var zipped bytes.Buffer
	zw := gzip.NewWriter(&zipped)
	q, _ := json.Marshal(map[string]any{"names": []string{name}, "keys": []string{extra}, "wait_ms": 1000})
	zw.Write(q)
	zw.Close()
	req, _ := http.NewRequest("POST", ts.URL+"/handoff/get", bytes.NewReader(zipped.Bytes()))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Encoding", "gzip")
	res, err := http.DefaultClient.Do(req)
	if err != nil || res.StatusCode != http.StatusOK {
		t.Fatalf("gzipped handoff read: %v %v", err, res)
	}
}
