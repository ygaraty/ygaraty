# Block relay

A second server next to the coordinator that serves tensor blocks to workers, so the
coordinator's Python process stops moving bytes. The design is in
[docs/research/areas.md](../docs/research/areas.md) (area D) and the motivation in
[experiment 27](../public/research/027-chain-steps-per-task.md): at 4 workers the coordinator
spends 10 s of thread time on tensor reads and 6.7 s on socket writes per 12.5 s update.

The relay is a dictionary from a block's SHA-256 to its bytes, with a memory budget, over
HTTP. It knows nothing about tensors, tasks or versions. Keys are content hashes, the same
`sha256` the tensor store records for every block (`ygaraty/storage/tensors.py`), so a read
cannot be stale and a write cannot corrupt.

| Method and path | Who | What |
|---|---|---|
| `GET /blocks/{key}` | workers | one block's bytes; 404 if unknown |
| `GET /blocks?keys=a,b,c` | workers | several blocks in one Ygaraty binary frame (`frames.py` layout); 404 listing the missing keys |
| `POST /blocks/get` | workers | the same frame for JSON `{"keys": [...]}`: one fixed URL, so a browser's CORS preflight is cached instead of paid for every key list |
| `PUT /blocks/{key}` | coordinator, workers | store bytes whose SHA-256 is the key; 400 on mismatch, 507 over budget, 413 over the block limit; a repeated key adds a reference; `?f64=1` also requires finite float64 values, `?f32=1` finite float32 values (a job with `YGARATY_FLOAT32_WIRE`), `?bf16=1` finite bf16 values (`YGARATY_ACTIVATION_WIRE`) |
| `PUT /blocks` | workers | several blocks in one frame (the layout `GET /blocks` answers); each header entry is `{"key", "size", "check"}` with `check` `f64`, `f32`, `bf16` or empty. Every block is verified (key, hash, size, finite values) before any is stored, so a bad one stores nothing; 507 over budget can leave the earlier ones stored. Uncompressed, up to 256 MiB. One request per result instead of one per block |
| `POST /handoff/get` | workers | handoffs ([experiment 58](../public/research/058-coordinator-off-the-hop.md)): JSON `{"names": [...], "keys": [...], "wait_ms": N}` waits up to 60 s until every name is published by a `PUT /blocks` frame whose header carries `name` and `manifest`, then answers one frame with the manifests and the bytes of their blocks and of the extra keys; 404 with the names still missing at the deadline. A worker handed a task before its inputs commit waits here, so its input arrives as soon as the producer's upload lands |
| `DELETE /blocks` | coordinator | `{"keys": [...]}` drops one reference per key; a block is freed at zero (identical bytes shared by two versions survive the first release) |
| `GET /status` | operator | counters, including `requests` by kind: preflights, single and batched reads and writes |
| `OPTIONS /...` | browsers | a CORS preflight, answered without a token for an origin named by `-cors-origins` (403 otherwise) |

Authenticated operations carry `Authorization: Bearer <token>`; CORS preflight OPTIONS requests are the exception.

## Browsers

Browser contributors (the SDK, `ygaraty/webapp/static/sdk.mjs`) call the relay directly from the
coordinator's contributor page, so the relay must allow that page's origin: `-cors-origins
https://coordinator.example` (comma-separated; `*` allows any). Replies to an allowed origin carry
`Access-Control-Allow-Origin` and `Vary: Origin`; a preflight is answered with the allowed methods
(`GET, POST, PUT`) and headers (`Authorization, Content-Type, Accept-Encoding, Content-Encoding`) and may
be cached for a day. Without the flag no CORS header is sent and browsers cannot read the relay's
replies. The coordinator's page policy (`connect-src`) names the relay's origin when
`YGARATY_RELAY_URL` is set. A page served over HTTPS needs the relay over HTTPS too (mixed content).
[Experiment 50](../public/research/050-browser-workers-on-the-relay.md) measures the browser path. Browsers preflight every request that carries the token and cache the answer per URL, so the SDK uses the two fixed-URL batched calls (`POST /blocks/get`, `PUT /blocks`): one preflight per tab instead of one per request (DO-2).

Both GETs answer with `Content-Encoding: gzip` (fastest level) when the request's `Accept-Encoding` allows it, the body is at least 1 KiB and gzip makes it smaller. A `PUT` may be gzipped too (`Content-Encoding: gzip`); the key and the size limit apply to the decoded bytes. The Python `RelayClient` does both only with `gzip=True` (a worker's `--gzip-frames`): float blocks shrink by only 4-8%, so it pays mainly on slow links or for sparse or quantized blocks.

## Build and operate

The relay remains an independently runnable Go module in this repository. It requires
**Go 1.27 or newer**, matching `go.mod`. Follow the single
[coordinator and relay operator guide](../docs/internal/operator-guide.md) for builds,
`ygaraty relay serve`, credentials, HTTPS, browser CORS, monitoring and shutdown.
The latency proxy is a separate experiment utility, not a deployment dependency.

Go tests and benchmarks may start loopback servers. Run them only on an authorized
execution host; they are not offline metadata checks.

## Coordinator integration

With `YGARATY_RELAY_URL` set, the coordinator's scratch store pushes every block it writes to
the relay and drops a reference for every block it frees; claim replies name each input's
blocks (`relay: {region, blocks: [{sha, shape, region, transposed}]}`) and carry
`relay: {url, token, block_size}`; `ygaraty http-worker` fetches the blocks, assembles the
regions, and stores its results in the relay as tiles, sending the coordinator
`blocks: [{sha, shape, region}]` instead of bytes. The coordinator reads result bytes back
only for the 1% it recomputes, the loss, and the step end. `storage/relay.py` is the client.
[Experiment 28](../public/research/028-block-relay.md) measured it: coordinator traffic 18 GB →
21 MB per update.

## Not done yet

The token in a claim reply allows writes as well as reads; a public
pool needs a read-only token for contributors. Each task through the relay costs two
extra requests (fetch, upload), 5 s per worker per update on loopback and several round trips over a
network; see experiment 28.

## Cross-language compatibility

The [shared byte fixtures](testdata/README.md) are consumed by the Go relay handler,
Python `RelayClient`, and browser SDK `RelayClient`. They cover little-endian float64,
float32 and bf16 blocks, SHA-256 identities, and padded multi-block framing. The focused
checks use in-memory or fake HTTP and do not start services. Run all three before changing
the relay wire contract; see the fixture guide for exact offline commands and limits.

## Package organization

The relay remains the `github.com/ygaraty/ygaraty/relay` Go module in this repository.
`NewStore`, `Store`, `Status`, `Server` and `Server.Handler` keep their public contracts.
The standard build command and `ygaraty relay serve` launcher in the operator guide are unchanged.

| File | Responsibility |
|---|---|
| `doc.go` | Package/protocol overview |
| `store.go` | Content hashes, reference counts, byte budget and counters |
| `server.go` | Server configuration, route registration, deletion and status endpoints |
| `middleware.go` | Authentication and browser CORS/preflight ordering |
| `responses.go` | JSON replies, gzip negotiation and block response writing |
| `blocks.go` | Single/multi-block downloads and padded binary framing |
| `upload.go` | Bounded request reading, decompression, upload validation and error mapping |
| `validation.go` | Finite little-endian float64, float32 and bf16 checks |
| `cmd/relay/` | Relay executable configuration and listener lifecycle |
| `cmd/latency-proxy/` | Experiment-only network utility; not a deployment dependency |

All files remain in one package: this split introduces no additional service, dependency,
protocol version or startup step. Stored slices retain the existing immutability contract;
repeated PUTs add a reference, and DELETE releases one reference. CORS preflight still
runs before authentication; block operations still require the bearer token. Hashes,
limits, compression, frame padding and status/error mapping are unchanged.

### Focused offline checks

With Go 1.27+ already installed, from `relay/`:

```bash
GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off go test -run '^(TestOffline.*|TestCompatibilityFixture)$' .
```

These tests invoke the real handler with `httptest.NewRecorder` and in-memory requests;
they do not open listeners or contact a server. They cover authentication, CORS,
reference retention/deletion, budget enforcement, decompressed size limits, hash
mismatch, finite wire validation and the shared Python/browser byte fixture.
The rest of `relay_test.go` includes live loopback tests and benchmarks and is not part
of this offline command. Go was unavailable on the cleanup host, so the new Go checks
were added but not executed there. Python/browser fixture checks were rerun separately;
those passes do not substitute for Go compilation or execution.
