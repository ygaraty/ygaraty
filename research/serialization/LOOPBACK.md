# Binary tensor uploads and loopback evaluation

This report preserves measurements from a retired upload implementation. Its
`CoordinatorClient`, upload server and benchmark CLI are not supported APIs in
this public release. Operational commands are archived outside the repository;
methods, results and evidence below remain available for reuse.

## Historical implementation

`CoordinatorClient.submit()` (and the `ygaraty submit` workflow) automatically sends external tensor chunks as raw bytes when upload creation advertises `chunk_encodings: ["json-base64", "raw"]`. Older servers and custom transports without binary support keep the JSON/base64 path. Plan uploads, API metadata, exported models and worker frames retain their existing formats. This change covers external float64 tensor uploads, not every weight/update path.

The production client still requires HTTPS, authenticates with the same bearer token, rejects redirects and bounds responses. No new dependency is needed: raw HTTP bodies suffice because upload sessions already carry tensor metadata. MessagePack remains an optional serialization-benchmark dependency.

Protocol:

- Create and finalize use the existing JSON endpoints and idempotency keys.
- `POST /api/v1/tensors/{id}/chunks/{index}` accepts `Content-Type: application/octet-stream`, an exact `Content-Length` up to the upload's chunk size (256 KiB by default, up to 4 MiB when negotiated; see below), and `X-Ygaraty-Chunk-SHA256` containing the lowercase SHA-256 hex digest of the body.
- Raw bodies contain the existing little-endian float64 values. Authentication and same-origin checks precede body reading. Compressed and transfer-encoded raw requests are rejected.
- Both encodings enter the same tensor validator: exact chunk length, checksum, finite values, duplicate/conflicting chunks, expiry, disk quotas and whole-tensor finalization digest. The raw endpoint is distinct from the worker binary-frame decoder.
- Raw requests closed the connection when this was first measured, so both variants below use a new connection per request. Since the follow-up below, a raw chunk whose body was read in full keeps the connection open; rejected or unread bodies still close it.

## Historical measurement setup

The experiment used a loopback HTTP server and an upload-only fixture from the
retired application, without TLS. Each run created and cleaned up temporary
uploads. The implementation and execution commands are archived separately.

## Measured results

Each mode gets a warm-up; five measured repetitions alternate order. Every upload uses a fresh session and file. Timing includes create, client encoding, HTTP transfer, checksums, finite-value checks, disk writes and finalization. Source generation and the final byte-for-byte read-back are outside timing. OS file caching is not disabled. Float64 precision is identical between modes.

| Tensor size | Encoding | Median upload ms | Payload MiB/s | Request body bytes | Peak traced MiB |
|---|---|---:|---:|---:|---:|
| 0.25 MiB | base64 | 4.92 | 50.8 | 349,709 | 1.94 |
| 0.25 MiB | raw | 2.24 | 111.8 | 262,238 | 0.31 |
| 8 MiB | base64 | 146.04 | 54.8 | 11,187,775 | 8.03 |
| 8 MiB | raw | 47.08 | 169.9 | 8,388,703 | 8.02 |

All uploads passed exact byte comparison, including warm-up and separate memory runs. [Full results and samples](loopback-2026-10-30.json) record the environment. The 8 MiB raw upload was approximately **3.1× faster**, with approximately **25% fewer request-body bytes** than base64.

Request-body totals include create/finalize JSON, but exclude HTTP headers, responses and TCP overhead. Throughput divides original tensor size by end-to-end upload time; it is not link bandwidth. Peak memory is a separate `tracemalloc` pass across client and server in one process, including read-back verification; it excludes the pre-existing source bytes, is not RSS, and may miss native allocations. For 8 MiB, full-file verification dominates the peak in both modes, so this experiment does **not** demonstrate a large overall memory reduction.

This is loopback **HTTP without TLS**, using an experiment-only client adapter. It exercises the production upload helper, HTTP handler, routing and disk manager; the production HTTPS transport is covered by fake-I/O tests. Browser behavior, TLS overhead, persistent connections, remote latency, concurrent uploads and training throughput remain unmeasured. No production-readiness or training-correctness conclusion follows from upload speed.

## Validation and next changes

Offline tests cover binary routing, authentication, origin checks, body limits, truncation, encoding rejection, digest mismatch, nonfinite tensors, duplicate/conflicting chunks, expiry, raw negotiation, old-server fallback, and HTTPS transport headers. Existing submission and external tensor tests also remain applicable.

The experiment evaluated negotiated raw uploads for external tensors. Retain JSON for metadata and reports. Evaluate HTTPS and browser implementations before extending this to other weight/update protocols; preserve their independent precision and frame contracts. Do not replace export formats, canonical hashes or all JSON messages based on this experiment.

## Follow-up: one connection and 4 MiB chunks

The home-link run of the measurement above (50 ms round trip, 100 Mb/s down, 20 Mb/s up, through `latency-proxy`) put the raw 8 MiB upload at 5.15 s, of which 3.36 s is the body at 20 Mb/s. The rest was 34 sequential requests (create, 32 chunks of 256 KiB, finalize), each a new connection. Two changes, each compatible with older peers:

- **Kept-alive connections.** `HTTPTransport` keeps one HTTPS connection (`http.client`) for all its requests instead of urllib's connection per request. Redirects are still refused (a 3xx raises `HTTPError`), responses are still bounded, errors still raise `HTTPError`. If a reused connection turns out to have been closed by the server, the request is resent once on a new connection, but only when repeating it is harmless: chunks, finalize, keyed requests and GETs. The server's raw chunk route no longer forces the connection closed after a body it read in full.
- **Negotiated chunk size.** A tensor create may carry `chunk_bytes`; the server accepts powers of two from 256 KiB to 4 MiB for that upload (`tensor_chunk_bytes` in capabilities) and answers with the size it will check. A repeated create keeps the first size. `submit` asks for 4 MiB. An older server ignores the field and answers 256 KiB, which the client follows; an older client never asks and gets 256 KiB. Plan uploads are unchanged.

`latency-proxy` accepted each connection locally, so opening one cost nothing there and the first home run could not see what a kept-alive connection saves: the same 34 requests took 5.14 s on one connection and 5.15 s on 34 ([no-handshake run](keepalive-home-no-handshake-2026-10-30.json)). Its new `-handshake-rtts N` flag holds each new connection for N round trips before the first byte moves; 2 stands for TCP plus TLS 1.3. The default 0 keeps the earlier behaviour. The benchmark passes 2 on `--link home` unless told otherwise:



`raw-reuse` and `raw-reuse-4m` go through the production `HTTPTransport` (over plain HTTP to the loopback port); `base64` and `raw` are the first measurement's fresh-connection client. Medians of 5, 8 MiB float64 tensor, Ryzen 9 PRO 6950H, all uploads byte-identical:

| Mode | Requests | Connections | Loopback | Home, 2 handshake RTTs |
|---|---:|---:|---:|---:|
| base64, 256 KiB | 34 | 34 | 87.9 ms | 9.75 s |
| raw, 256 KiB | 34 | 34 | 39.8 ms | 8.56 s |
| raw, one connection | 34 | 1 | 26.9 ms | 5.25 s |
| raw, one connection, 4 MiB | 4 | 1 | 19.8 ms | 3.68 s |

The body alone takes 3.36 s at 20 Mb/s, so the last row is within 10% of the link. Against base64 on fresh connections it is 2.6× faster at home; against raw on fresh connections, 2.3×. For a 256 KiB tensor (3 requests) one connection cuts 564 ms to 361 ms; the chunk size does not matter there. Samples varied by under 0.2%. Peak traced memory is 8.0 MiB for 256 KiB chunks and 8.8 MiB for 4 MiB ones (the read-back verification dominates). Runs: [loopback](keepalive-loopback-2026-10-30.json), [home](keepalive-home-2026-10-30.json).

Still emulated, not measured: connection setup is modelled as two round trips (TCP, TLS 1.3), with no TLS computation; a real network adds loss and congestion-window growth, which favour fewer, longer transfers further. The uploads remain sequential; with one connection, the next lever would be overlapping chunk requests or HTTP/2, and at 4 MiB chunks the 20 Mb/s link already dominates.


## Upload deadlines and experiment cleanup

Raw upload HTTP framing is owned by `submission/http_upload.py`; the webapp router delegates to it, while `TensorUploads` retains tensor validation and storage. Raw bodies have a 30-second total read deadline, including slow trickle delivery. A timed-out body receives HTTP 408 and closes the connection; incomplete bodies are never dispatched to storage. The server also uses a 30-second idle/header socket timeout. Successfully read bodies retain the existing keep-alive behavior. The total deadline applies to raw tensor bodies; it is not an overall training or job timeout.

The loopback experiment owns its server and optional proxy through a context manager. Cleanup is registered immediately after each resource starts, including proxy launch/readiness failures. Proxy readiness has a five-second deadline and checks early process exit. Shutdown waits two seconds for proxy termination, then kills it with a further bounded wait. Server polling is 50 ms; handler threads are daemonized so an incomplete client cannot block listener cleanup. No new measurements were taken for this refactor. Offline fault-injection tests cover startup failures, cleanup, stalled/trickled/disconnected bodies, HTTP 408, and preservation of normal keep-alive behavior.

## Session isolation, retries and controlled comparisons

`UploadSession` now owns tensor chunk sizing, progress, checksum validation and finalization. `TensorUploads` reserves quotas and idempotency keys under its manager lock, then performs chunk writes and finalization hashing under a per-session lock. The application lock is used briefly to initialize the manager, not held during tensor upload I/O. Concurrent requests for one session serialize; independent sessions can proceed. Active operations and admission-time tensor publication pin sessions against expiry. Files are allocated on the first accepted chunk; quotas are reserved at creation. Plan uploads retain their existing synchronization.

`HTTPTransport` serializes each complete exchange, including retries, on a reentrant lock. Sharing a client across threads is supported but does not provide parallel requests on its single connection. Closing waits for the current exchange. Automatic retry after a stale reused connection is restricted to GETs, upload chunks/finalization, and keyed creates at `/api/v1/tensors`, `/api/v1/uploads`, and `/api/v1/jobs`. An arbitrary endpoint does not become retryable merely by carrying a key.

For controlled future measurements, compare base64 and raw encoding with the same connection strategy and chunk size:



Each pair uses the same connection strategy and chunk size. Compare base64/raw within a pair for encoding cost, then compare reused 256 KiB/4 MiB modes for chunk-size effects. Schema version 2 records `transfer_peak_traced_bytes` and `verification_peak_traced_bytes` separately, excluding source generation and session-directory setup. Memory passes remain separate from timing and cover client/server traced allocations, not RSS. Existing version 1 reports are historical and have not been rewritten. These new modes and locking changes have offline regression coverage; no new server measurements were run for this refactor.

## Client connection cleanup

The historical client used a context manager to release its connection:



`CoordinatorClient.close()` releases the transport connection, and the context manager does so on both normal return and exceptions. It also closes an explicitly supplied transport if that transport exposes `close()`; callers sharing an injected transport should manage that lifetime deliberately. `HTTPTransport` supports the same context-manager pattern. Closing releases the current connection; a later request can open a new one.

The loopback CLI rejects duplicate modes, duplicate tensor sizes, and nonfinite or negative handshake delays before starting any resources. These checks prevent accidental repeated measurements or invalid link settings. No benchmark results were regenerated for this polish pass.
