package relay

import (
    "bytes"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "net/http/httptest"
    "os"
    "strings"
    "testing"
)

// TestCompatibilityFixture uses Handler directly: no listening sockets or external services.
func TestCompatibilityFixture(t *testing.T) {
    raw, err := os.ReadFile("testdata/compatibility-v1.json")
    if err != nil { t.Fatal(err) }
    var fixture struct {
        Blocks []struct { Dtype string `json:"dtype"`; SHA string `json:"sha256"`; Hex string `json:"hex"` } `json:"blocks"`
        Frame string `json:"frame_hex"`
        SHA string `json:"frame_sha256"`
    }
    if err := json.Unmarshal(raw, &fixture); err != nil { t.Fatal(err) }
    server := (&Server{Store: NewStore(1 << 20), Token: "fixture-only", MaxBlockBytes: 1 << 20}).Handler()
    keys := []string{}
    flags := map[string]string{"<f8":"f64", "<f4":"f32", "bf16":"bf16"}
    for _, block := range fixture.Blocks {
        data, err := hex.DecodeString(block.Hex)
        if err != nil { t.Fatal(err) }
        sum := sha256.Sum256(data)
        if hex.EncodeToString(sum[:]) != block.SHA { t.Fatal("block digest mismatch") }
        request := httptest.NewRequest("PUT", "/blocks/"+block.SHA+"?"+flags[block.Dtype]+"=1", bytes.NewReader(data))
        request.Header.Set("Authorization", "Bearer fixture-only")
        response := httptest.NewRecorder()
        server.ServeHTTP(response, request)
        if response.Code != 201 { t.Fatalf("put: %d %s",response.Code,response.Body.String()) }
        keys = append(keys, block.SHA)
    }
    request := httptest.NewRequest("GET", "/blocks?keys="+strings.Join(keys,","), nil)
    request.Header.Set("Authorization", "Bearer fixture-only")
    response := httptest.NewRecorder()
    server.ServeHTTP(response, request)
    expected, err := hex.DecodeString(fixture.Frame)
    if err != nil { t.Fatal(err) }
    if response.Code != 200 || !bytes.Equal(response.Body.Bytes(), expected) { t.Fatal("Go frame differs from shared fixture") }
    sum := sha256.Sum256(expected)
    if hex.EncodeToString(sum[:]) != fixture.SHA { t.Fatal("frame digest mismatch") }
}
