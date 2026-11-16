# Browser workers on the block relay

| | |
|---|---|
| **Date** | 2026-10-29 (plan; stages A and B built and gated the same evening; stage C run the same night) |
| **Path** | The sharded factorization of experiment 48 (WebGPU tabs, tile groups, float32 wire) with the Go block relay of experiment 28 carrying every tensor byte: the browser SDK (`webapp/static/sdk.mjs`) fetches its inputs from the relay by hash, assembles them, uploads its results as blocks and sends the coordinator hashes, as `ygaraty http-worker` already does. The relay gets CORS; the contributor page's policy admits the relay's origin |
| **Result** | Yes. At 100,000 proteins (4 WebGPU tabs, 5 x 5 tile groups, float32 wire, 30 updates) the coordinator's traffic fell from 26.2 GB down and 21.5 GB up to 47 MB and 13 MB (99.8% and 99.9%), its own relay pushes (the initial factors, 102 MB) and fetches (the losses, 3 KB) included; the relay served 26.2 GB and took 21.5 GB. Losses bit-identical to the run without the relay, 0 rejected results, every tile on the GPU. 47.7 s per update against 45.9 (median) without the relay: 3.9% slower, within the 10% gate, on a GPU-bound machine. On emulated home links (one 100/20 Mb/s, 50 ms link per tab) the relay costs 18%: 157.5 against 133.5 s per update, about 880 extra round trips per tab per update (one relay fetch per claim, about three block uploads per result, each with a CORS preflight), with the same bytes on the link. A batched upload (`PUT /blocks`) and uploads overlapped with compute are the fix, not built here. Stages A and B (the SDK's relay path, CORS on the relay) gated by tests under Node and Chromium |
| **Evidence** | `runs/experiments/048/stage-d-float32-relay/20261029-203914/report.json` (relay) and `runs/experiments/048/stage-d-float32/20261029-201308/report.json` (the same run without it, the same SDK); home links: `runs/experiments/048/stage-d-float32-home/20261029-213515/report.json` and `runs/experiments/048/stage-d-float32-relay-home/20261029-215743/report.json`; all local, not in git; `tests/test_browser_relay.py`, `tests/test_factorization_webgpu.py: BrowserRelayTrainingTests`, `relay/relay_test.go: TestCORS` |

## Question

Can browser workers take the coordinator out of the byte path, as Python workers did in experiment 28 (coordinator traffic 18 GB to 21 MB per update), with the losses unchanged, and what does it cost a tab in requests and time per update?

## Why this problem

After tile groups and the float32 wire (experiment 48), the scaling list's next item is the coordinator: one Python process every input and result passes through. During experiment 48's single-tile runs it moved at most about 227 MB/s, so at 100,000 proteins it caps an update at about 14 s however many GPUs join, and at 1,000,000 with 5 x 5 groups on a float64 wire it would move about 320 GB per update, 24 minutes, against 75 GPU-minutes of tiles. Halving bytes (the float32 wire) halves that; only moving the bytes off the coordinator removes it. The relay exists and does this for Python workers; experiment 28's "not done" line and the historical HTTP API notes both record that the browser SDK ignores the `relay` field. Stage E of experiment 48 (volunteer browsers on real links) needs this first: a public coordinator that moves every byte itself is the design the relay was built to replace.

## What exists

- **The relay** (`relay/`): `GET /blocks?keys=a,b,c` returns several blocks in one binary frame; `PUT /blocks/{key}?f64=1` (or `f32=1`, `bf16=1`) stores bytes whose SHA-256 is the key and refuses non-finite values; bearer token on every request; gzip both ways when asked. No CORS headers, no `OPTIONS`, so a browser on another origin cannot call it today.
- **The coordinator's side** (`coordination/jobs.py`, `storage/memory.py`, `storage/relay.py`): with `YGARATY_RELAY_URL`, the store pushes every block it writes to the relay; a claim reply carries `relay: {url, token, block_size, dtype}` and, for each input the worker must fetch, `relay: {region, blocks: [{sha, shape, region, transposed, dtype}]}` instead of `data`; a result may arrive as `blocks: [{sha, shape, region, dtype}]` and the store publishes it by reference (`put_remote`), fetching bytes only for a sampled recheck, the loss and persisted weights. `push` claims attach the relay blocks to the reply itself. All of this is exercised by `tests/test_http.PushedInputsTests` with Python workers.
- **The Python worker** (`transports/http/worker.py`, `storage/relay.py`): `assemble_region` builds an input region from blocks (transposed blocks included); `upload_many` cuts a result into the full-width bands its readers want (`attrs.upload_rows`, the factorization's per-shard gradient bands) or square tiles, hashes, and PUTs them concurrently.
- **The browser SDK**: binary frames (float64 or float32), a tile cache keyed as the coordinator's hints, memory budgets, GPU tiles and groups. No relay path: an input without `data` and not `cached` is an error.
- **The contributor page** is served with `Content-Security-Policy: connect-src 'self'`, so even a CORS-enabled relay on another origin is blocked until the page's policy names it.

## Stages A and B (2026-10-29): built

- **Relay:** `-cors-origins` names the page origins allowed; a preflight (`OPTIONS`, no token) is answered with `GET, PUT` and the `Authorization`, `Content-Type` and encoding headers, cacheable for a day; every reply to an allowed origin carries `Access-Control-Allow-Origin` and `Vary: Origin`; other origins get no CORS header and a 403 preflight; the token is still required on every real request (`relay_test.go: TestCORS`).
- **Coordinator:** the contributor page's `Content-Security-Policy` adds the relay's origin to `connect-src` when `YGARATY_RELAY_URL` is set. The factorization job's relay blocks follow the job's wire (float32 on a float32 wire).
- **SDK (`sdk.mjs`):** `RelayClient` (one `GET /blocks?keys=` per claim for every block the reply's inputs name, pushed blocks taken from the reply's tail first; `PUT /blocks/{sha}?f64=1|f32=1` with the SHA-256 from `crypto.subtle.digest`); `decodeBlock` (float64, float32 or bf16, transposed when the manifest says) and `assembleRegion` as `storage/relay.py`'s; results cut into the task's `upload_rows` bands or `block_size` tiles, uploaded concurrently, posted as `blocks` in one JSON request; an upload that fails falls back to inline bytes (counted, reported as a `relay_fallback` event). The tile cache also gained pinning: the keys a claim reply says the tab holds are not evicted while its tasks run, which is the mechanism behind experiment 48's cache misses (an earlier task's inputs evicting a later task's "cached" tile); its default grew from 256 to 512 MiB, the Python worker's.
- **Gates met** (`tests/test_browser_relay.py`, `tests/test_factorization_webgpu.py: BrowserRelayTrainingTests`): the Node worker on the relay reaches losses equal to the inline run's and the Python worker's on both wires, with every input named as relay blocks and every result arriving as blocks, every sampled recheck passing, no fallback; a Chromium tab trains 4,096 x 4,096 with 2 x 2 groups on the float32 wire through the relay, 0 rejected, every tile on the GPU, loss within 1e-5 of the exact reference, every result as relay blocks, and the coordinator receiving less than a tenth of the result bytes.

## Stage C (2026-10-29): 100,000 proteins through the relay

Experiment 48's harness with `--relay` starts a `relay-server` on loopback allowing the coordinator's origin and gives the job a runtime with the relay; otherwise the same run as the float32 5 x 5 one: 100,000 Swiss-Prot proteins, 4 headless Chromium tabs on one AMD Radeon 680M, 5 x 5 tile groups, float32 wire, 30 updates, cache pinning and 512 MiB caches in both.

| | Without the relay | With the relay |
|---|---:|---:|
| Coordinator bytes sent / received, 30 updates | 26,208 / 21,512 MB | 47.2 / 13.4 MB |
| Coordinator's own relay traffic | — | 102.4 MB pushed (the initial U, V and latent rows), 3.2 KB fetched (30 losses) |
| Relay served / took | — | 26,154 / 21,505 MB; 53,538 block reads, 44,787 puts, 0 misses |
| Per tab per update, down / up | 210 to 230 / 175 to 184 MB | 0.4 / 0.1 MB from the coordinator; the rest from and to the relay |
| Seconds per update, median (mean) | 45.9 (51.1, two stalled updates) | 47.7 (47.7) |
| Losses | reference | bit-identical |
| Rejected results | 0 | 0 |
| Explained share of ||X||^2 at 30 | 0.783476 | 0.783476 |

- **The coordinator left the byte path:** 99.82% fewer bytes out and 99.94% fewer in, the same order as experiment 28's Python workers (99.9%). What it still moves is headers, manifests (64-character hashes with shapes and regions) and the 102 MB of initial factors it pushes once; the new factors each update are blocks the tabs stored, which the coordinator never reads back. At 1,000,000 proteins this is the difference between a coordinator moving about 160 GB per update (float32, 5 x 5 groups) and one moving about 150 MB of manifests and headers (the 47 MB per 30 updates here, scaled by the task count).
- **Time:** 3.9% slower on the median, inside the 10% gate. The tabs saturate this GPU either way; the extra 1.8 s per update is the relay round trips that do not hide behind compute (a group's bands are uploaded after its result is ready and before the next claim). The control's two stalled updates (143 and 99 s) at the 10-minute mark did not occur in the relay run; the stall is unexplained (experiment 48's write-up).
- **Requests:** 53,538 block reads and 44,787 puts for 15,240 results over 30 updates, about 3.3 relay requests per task in all (one `GET` of several blocks per claim, one `PUT` per result band). On real links each is a round trip (experiment 51: a hop on home links is bytes, not round trips, so these should cost little).
- **Not shown:** anything over a real network, the public-pool token question, and the 1,000,000-protein plan (stage D, optional, not run).

## Emulated home links (2026-10-29): the relay's round trips show

The first further test: each tab reaches the coordinator and the relay through its own `latency-proxy` (experiment 47's home link: 100 Mb/s down, 20 Mb/s up, 50 ms round trip, shared by that tab's connections; the harness's `--links home`), so the link rather than the GPU sets the time. Same problem, 4 tabs, 10 updates.

| Home links, 100,000 proteins, 4 tabs, 5 x 5 groups, float32 | Without the relay | With the relay |
|---|---:|---:|
| Seconds per update, median | 133.5 | 157.5 |
| Per tab per update on the link, down / up | 215 to 230 / 178 to 181 MB | the same bytes, now to and from the relay |
| Coordinator bytes per update, sent / received | 888 / 717 MB | 1.6 / 0.4 MB |
| Relay requests per tab per update | — | about 127 block fetches (one per claim), 376 uploads (2.96 per result), 376 preflights |
| Losses | reference | bit-identical |
| Rejected | 0 | 0 |

- **The link, not the GPU, sets the time:** 133.5 s against 45.9 on loopback. A tab's 179 MB upload alone is 72 s at 20 Mb/s and its 222 MB download 18 s at 100 Mb/s; the rest is that uploads, downloads and compute do not fully overlap in the SDK (one task in flight at a time).
- **The relay costs 18% here, 24 s per update per tab,** with the same bytes on the link. The difference is round trips: the SDK makes one relay fetch per claim before it can compute (the contributor page claims one task at a time), about three `PUT`s per result (a 5 x 5 group's result is 11 bands, the reductions' and updates' results one each), and the browser sends a CORS preflight before each `PUT` because every block has its own URL, so `Access-Control-Max-Age` never applies. About 880 round trips per tab per update at 50 ms is 44 s if serialized; the browser's six connections per host overlap some of it. Experiment 51's rule (a hop on home links is bytes, not round trips) held for the Python worker, which uploads concurrently and posts results while it computes; the SDK does neither yet.
- **What would remove it,** in order: a batched upload (`PUT /blocks`, several blocks in one frame: one request and one cacheable preflight per result); uploading result i while computing result i + 1, as the Python worker's `StreamPoster` does; claiming several tasks per request on the contributor page (one relay fetch for all of them). None is built here; the relay's own property, the coordinator out of the byte path, holds on the link too (1.6 / 0.4 MB per update against 888 / 717).
- **Not shown:** real distance (the proxy adds latency and rate limits, not loss or jitter) and more than four tabs.

## Plan

**Stage A: the read path.** The relay answers `OPTIONS` and sets `Access-Control-Allow-Origin` (the coordinator's origin, from a flag), `-Allow-Methods: GET, PUT`, `-Allow-Headers: Authorization, Content-Type, Accept-Encoding`, `-Max-Age` so preflights are cached. The coordinator's page policy gets `connect-src 'self' <relay origin>` when a relay is configured. The SDK reads the claim reply's `relay` field, fetches the blocks its inputs name in one `GET /blocks?keys=` per claim (with `push` claims, takes them from the reply), decodes `<f8` / `<f4` / `bf16` blocks, assembles regions (`assemble_region` ported, transposed blocks included), and puts assembled leaves in its tile cache under the same keys as inline ones. Gate: under Node (`node_worker.mjs`), the small relay job of `PushedInputsTests` reaches the same losses as the Python worker, with and without `push`, and the coordinator's `sent_bytes` falls as it does for Python workers.

**Stage B: the write path.** The SDK hashes each result band with `crypto.subtle.digest('SHA-256')`, cuts results as `upload_rows` says (bands) or into `block_size` tiles, PUTs them concurrently (`f32=1` on a float32 wire, `f64=1` otherwise), and posts `blocks: [...]` in place of `data`. A PUT that fails falls back to posting the bytes inline (the coordinator accepts both), so a relay outage costs bytes, not results. Gate: the same job's losses with results through the relay equal the inline run's; the coordinator's `received_bytes` falls to headers; `relay_stats()['results']` counts every result.

**Stage C: the factorization at 100,000 proteins.** Experiment 48's harness (`factorization_browser_training.py`) gets `--relay` (starts `relay-server` on loopback, sets the runtime's relay URL and token). 4 tabs, 5 x 5 groups, float32 wire, 30 updates, as the matching experiment 48 run, with and without the relay. Record per update: coordinator bytes each way, relay bytes served and taken, requests per tab (claims, results, relay GETs and PUTs), seconds per update, the coordinator's CPU time.

**Stage D (optional, if C is clean): the 1,000,000-protein plan on the wire.** Not a training run: one update's worth of claims and results through the relay at 1,000,000 x 1,000,000 with 11 x 11 groups (the plan builds; 9,604 tile tasks), to see whether the coordinator can drive 30,000 tasks per update when it moves no bytes. This is also the question the historical task-store notes leaves open about a SQL task store.

## Gates (fixed before the first run)

- Stage A and B: losses equal to the non-relay run (bit-identical on a float64 wire; on a float32 wire identical too, since the relay rounds exactly where the frame did).
- Stage C: the coordinator's bytes per update fall by at least 99% against the same run without the relay (experiment 28's 18 GB to 21 MB was 99.9%); 0 rejected results; every tab joined; every tile on the GPU; loss within 1e-5 of the exact reference at every update, as in experiment 48.
- Stage C: seconds per update on loopback within 10% of the run without the relay. The GPU bounds this machine (2,401 tiles x 19 ms), so the relay's extra requests must hide behind it; a larger slowdown means the SDK's uploads are synchronous with its compute and need the Python worker's concurrency.
- Report the requests per tab per update and the bytes the relay served. Do not claim a speedup: on one machine none is expected (experiment 28).

## Needs first

- Experiment 48's float32-wire runs committed (in progress at the time of this plan), so stage C compares like with like.
- A built `relay-server` (`cd relay && go build -o ../relay-server ./cmd/relay`); the tests skip without it.
- A decision on the token: the claim reply's relay token allows writes as well as reads. For loopback runs this is fine; for stage E of experiment 48 the relay needs a contributor token that can `GET` and `PUT` but not `DELETE`, or per-job tokens. Note it in the write-up; do not build it here unless stage C is quick.

## Risks

- **Preflights.** A cross-origin `PUT` with `Authorization` triggers an `OPTIONS` round trip per distinct URL unless the browser caches it (`Access-Control-Max-Age`); each result band is a distinct URL. If caching does not hold, one request per band becomes two, and over a real link that is the cost. Batching PUTs is not in the relay's API; add `PUT /blocks` (several blocks in one frame) if stage C shows it.
- **Hashing in the browser.** `SubtleCrypto` runs at roughly 1 GB/s; a 5 x 5 group's 5 MiB float32 result is milliseconds. Fine unless it serializes with the GPU readback.
- **HTTPS.** A page served over HTTPS cannot fetch an `http://` relay (mixed content). Loopback runs are HTTP both ways; stage E needs the relay behind TLS as the coordinator is.
- **The tab cache and relay blocks.** Cached inputs are keyed by the coordinator's hints; relayed inputs arrive as blocks and are assembled. The keys must coincide or the coordinator's cache accounting (`cache_hits`) will be wrong, as experiment 28's first attempt found with its ordering bug between relayed and cached tiles in one reply.
- **Memory.** Assembling a region holds the blocks and the region at once; `runtime/budget.py` and `budget.mjs` count inputs as received, so a relayed input must be counted the same way (it is the same float64 array once assembled) and the blocks freed before the kernel runs.
- **Two extra requests per task** (experiment 28: 5 s per Python worker per update on loopback with one-step tasks). Groups make tasks few (100 per update at 100,000 proteins; 9,604 at 1,000,000), so this should be small here, but it is the number to watch on real links.

## Limits

One machine, loopback: the relay's advantage, bytes off the coordinator, is measured; its cost on real links (the extra round trips) is not, and belongs to stage E of experiment 48. One GPU shared by all tabs, so time per update is GPU-bound and cannot show the coordinator's relief directly; the coordinator's bytes and CPU time are the measurement.

## Reproduce

```bash
cd relay && go build -o ../relay-server ./cmd/relay && cd ..
python -m unittest tests.test_browser_relay tests.test_factorization_webgpu.BrowserRelayTrainingTests   # stages A and B
ygaraty experiment factorization_browser_training --embeddings runs/datasets/protein-sprot/sample-100000.npz \
  --group 5 --wire float32 --updates 30 --tabs 4            # the control (about 25 minutes)
ygaraty experiment factorization_browser_training --embeddings runs/datasets/protein-sprot/sample-100000.npz \
  --group 5 --wire float32 --updates 30 --tabs 4 --relay    # stage C (about 25 minutes)
```

The protein sample comes from experiment 48 (`protein_embeddings.py`). Reports land under `runs/experiments/048/stage-d-float32[-relay]/`.
