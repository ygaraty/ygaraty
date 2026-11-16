# Relay compatibility fixtures

`compatibility-v1.json` is the shared byte contract consumed by the Go relay, Python relay client and browser SDK tests. It contains little-endian float64, float32 and bf16 blocks (including signed zero), block SHA-256 digests, and the complete multi-get frame with its digest. The frame uses a four-byte little-endian header length, compact JSON padded with spaces to an eight-byte boundary, then block bytes in request order.

The fixture is checked in, not regenerated during tests. Changes to framing or stored bytes must deliberately update this contract and review all three implementations. Exact frame output is a regression fixture for the Go encoder; clients may still accept other valid JSON whitespace. This is a bounded compatibility check, not exhaustive protocol or browser-network coverage.

Offline checks, from the repository root with existing local tools:

```bash
python -m unittest discover -s tests -p test_relay_compatibility.py
node tests/test_relay_compatibility.mjs
cd relay
GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off go test -run '^TestCompatibilityFixture$' .
```

The Python test replaces HTTP, the browser SDK runs in Node with a fake `fetch`, and the Go test invokes the real handler using `httptest.NewRecorder`. None opens a listening socket or contacts a service. The Go command requires Go 1.27+ already installed; it does not download a toolchain or modules. Running all relay tests is different: other tests start loopback servers and require authorization.

Python and browser checks compare the same bytes and hashes, preserve requested duplicate-key ordering, and reject a truncated frame. Python also verifies typed encoding against the block bytes. Go validates typed uploads and must emit the exact fixture frame. Running the SDK in Node checks shared JavaScript behavior, not browser CORS, TLS or WebGPU execution.
