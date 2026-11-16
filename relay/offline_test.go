package relay

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// offlineRequest invokes routing and middleware without a listener or HTTP client.
func offlineRequest(handler http.Handler, method, path string, body []byte, headers map[string]string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, bytes.NewReader(body))
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	return response
}

func TestOfflineAuthorizationAndCORS(t *testing.T) {
	handler := (&Server{Store: NewStore(1024), Token: "offline", AllowedOrigins: []string{"https://coordinator.example"}}).Handler()
	for _, tc := range []struct {
		method  string
		headers map[string]string
		status  int
	}{
		{"GET", nil, http.StatusUnauthorized},
		{"GET", map[string]string{"Authorization": "Bearer offline"}, http.StatusOK},
		{"OPTIONS", map[string]string{"Origin": "https://coordinator.example"}, http.StatusNoContent},
		{"OPTIONS", map[string]string{"Origin": "https://other.example"}, http.StatusForbidden},
	} {
		response := offlineRequest(handler, tc.method, "/status", nil, tc.headers)
		if response.Code != tc.status {
			t.Fatalf("%s: got %d want %d", tc.method, response.Code, tc.status)
		}
		if tc.status == http.StatusNoContent && response.Header().Get("Access-Control-Allow-Origin") != "https://coordinator.example" {
			t.Fatal("missing CORS origin")
		}
	}
}

func TestOfflineUploadReferencesAndBudget(t *testing.T) {
	handler := (&Server{Store: NewStore(8), Token: "offline", MaxBlockBytes: 16}).Handler()
	headers := map[string]string{"Authorization": "Bearer offline"}
	data := make([]byte, 8)
	sum := sha256.Sum256(data)
	key := hex.EncodeToString(sum[:])
	for _, status := range []int{http.StatusCreated, http.StatusOK} {
		if got := offlineRequest(handler, "PUT", "/blocks/"+key+"?f64=1", data, headers).Code; got != status {
			t.Fatalf("put: %d want %d", got, status)
		}
	}
	other := []byte("12345678")
	digest := sha256.Sum256(other)
	if got := offlineRequest(handler, "PUT", "/blocks/"+hex.EncodeToString(digest[:]), other, headers).Code; got != http.StatusInsufficientStorage {
		t.Fatalf("budget: %d", got)
	}
	body := []byte(`{"keys":["` + key + `"]}`)
	for _, expected := range []int{http.StatusOK, http.StatusNotFound} {
		if got := offlineRequest(handler, "DELETE", "/blocks", body, headers).Code; got != http.StatusOK {
			t.Fatalf("delete: %d", got)
		}
		if got := offlineRequest(handler, "GET", "/blocks/"+key, nil, headers).Code; got != expected {
			t.Fatalf("reference retention: %d want %d", got, expected)
		}
	}
}

func TestOfflineCompressedUploadLimitAndHash(t *testing.T) {
	handler := (&Server{Store: NewStore(4096), Token: "offline", MaxBlockBytes: 128}).Handler()
	data := []byte(strings.Repeat("x", 512))
	sum := sha256.Sum256(data)
	var body bytes.Buffer
	writer := gzip.NewWriter(&body)
	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	headers := map[string]string{"Authorization": "Bearer offline", "Content-Encoding": "gzip"}
	response := offlineRequest(handler, "PUT", "/blocks/"+hex.EncodeToString(sum[:]), body.Bytes(), headers)
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("decoded limit: %d", response.Code)
	}
	delete(headers, "Content-Encoding")
	response = offlineRequest(handler, "PUT", "/blocks/"+hex.EncodeToString(sum[:]), []byte("wrong"), headers)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("digest: %d", response.Code)
	}
}

func TestOfflineFiniteWireValidation(t *testing.T) {
	for _, test := range []struct {
		name              string
		check             func([]byte) bool
		finite, nonfinite []byte
	}{
		{"f64", finiteFloat64s, make([]byte, 8), []byte{0, 0, 0, 0, 0, 0, 240, 127}},
		{"f32", finiteFloat32s, make([]byte, 4), []byte{0, 0, 128, 127}},
		{"bf16", finiteBFloat16s, make([]byte, 2), []byte{128, 127}},
	} {
		if !test.check(test.finite) || test.check(test.nonfinite) || test.check([]byte{0}) {
			t.Fatal("finite/alignment contract: " + test.name)
		}
	}
}
