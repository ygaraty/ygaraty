# Node's built-in fetch held contributor downloads near 1 MB/s; core https loads the 16.5 GB model in 3.5 minutes

| | |
|---|---|
| **Conclusion** | Works: core https instead of Node's fetch loads 16.5 GB in 211 s at 65 MB/s, against 1.2 MB/s (measured, US host) |
| **Status** | executed |
| **ID** | DO-105 (report 180) |
| **Category** | browser |
| **Where** | remote (a Vast.ai Tesla V100 32 GB in Colorado running `ygaraty launch` of the public `qwen3.8-27b` preset and a native contributor); a local check on this machine |
| **Priority** | high |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | Per-piece load timeline (`?loadTrace=1`, new); then `native_environment.mjs` sends cross-origin HTTPS GETs (the presigned Spaces downloads) through Node's core `https` (`httpsGet`, new) instead of the built-in `fetch` |
| **Result** | Item 1 (Found): before the fix, a 10-minute window loaded 0.73 GB (1.15 MB/s); after the first seconds one 134 MB piece was in flight at a time, each taking about 138 s (0.98 MB/s per connection); descriptor calls 32 ms, first byte 340 ms, hashing 1.8 ms per MB. The same host with plain `curl` moved the same pieces in the same order at 44.8 MB/s (4 streams, 120 s) and 42 MB/s on one connection. Node's `fetch` alone on that host: 0.93–0.95 MB/s (any way of reading the body, IPv4 forced); core `https`: 49 MB/s, and 66 MB/s wrapped as a web stream. Item 2 (fix): the contributor with `httpsGet` loaded all 863 pieces (16.5 GB) in 211 s and was ready (self-test passed) at 256 s: 65 MB/s, 54× the unfixed load (bound 5×); large pieces at 57 MB/s each, about 2 in flight. Peak resident memory rose from 1.2 to 16.4 GiB |
| **Evidence** | measured-remote: `runs/experiments/DO-105/vast-do105b-20261125/` (`loads/1-T.log` and `loads-fix/1-T.log` with `LOAD` lines, `curl4.txt`), `tools/timeline.py`, `tools/node_fetch.mjs`; local: `omarchy-20261125-local/` |

## Question

Report 177 found that a US host downloads the public `qwen3.8-27b` pieces at 20–25 MB/s with plain parallel `curl` (about 16.5 GB in 12–14 minutes), while native contributors on such hosts loaded at about 1 MB/s (reports 169 and 172, from the coordinator's byte counter), whether one piece at a time, in three lanes, or with a rate floor. Where does a contributor's time per piece go, and what change brings a cold load near the host's rate?

## Prediction

TBD. Half the 863 pieces are under 1 MB, each paying a descriptor call and about 1.1–1.5 s to the first byte; large pieces may also be slowed by hashing or by uploads that wait for the GPU.

## Gate (frozen 2026-11-25, before running; budget $1, teardown 1.5 h)

1. **Diagnostic (`Found`):** with a per-piece timeline (`?loadTrace=1`: descriptor call, first byte, body, hash, then the time until the next piece starts), the share of a 10-minute load window spent in each part, on one US host, against plain parallel `curl` of the same pieces from that host (report 177's method).
2. **Fix (only if the timeline points to one that can be made and checked within this budget):** cold-load rate over a 10-minute window at least 5× the unfixed contributor's on the same host, the loaded stage passing the coordinator's self-test. Otherwise the fix is a new experiment.

## Protocol

1. Timestamps per piece in the contributor's loading path (off by default), checked locally against a coordinator serving the public bundle.
2. One US host (budget from the user first), `ygaraty launch` of the public 27B preset on the host, a native contributor with the timeline on.

Related: [177](177-spaces-cdn-and-tiny-objects.md), [169](169-spaces-parallel-pieces-crawling-flows.md), [172](172-spaces-rate-floor-crawling.md).

## Log

- 2026-11-25: Added at the user's request, from report 177.
- 2026-11-25: User: "run 105 and 106 next"; budget $1, teardown 1.5 h. Gate frozen above, before running.
- 2026-11-25: Local check (public `qwen3.5-0.8b` preset, this machine): 0.52 GB in 104.5 s (5.0 MB/s), 1.4 pieces in flight on average, large pieces 4.4 MB/s; this machine's link limits it, so the comparison was made on a US host.
- 2026-11-25: Remote: `do105b` (Vast.ai 59185732, Tesla V100 32 GB, Colorado, $0.211/h; an RTX 3090 offer was taken before renting). `host.sh setup`, `launch`; load T (timeline) for 10 minutes; `tools/curl_order.sh` (the pieces in the bundle's order, 4 streams, 120 s): 312 objects, 5,372 MB, 44.8 MB/s. Node checks on the host (`tools/node_fetch.mjs` and one-liners): `fetch` 0.93–0.95 MB/s for a 134 MB piece (`arrayBuffer`, the body reader, `readSizedResponse`; `--dns-result-order=ipv4first`); `curl -4` 46 MB/s (the host has no working IPv6); core `https` 49 MB/s; core `https` through `Readable.toWeb` 66 MB/s. The figures fit a receive window held near 128 KB over this host's round trip to fra1 (and 4–6 MB/s over this machine's shorter one).
- 2026-11-25: Fix: `httpsGet` in `native_environment.mjs` for cross-origin HTTPS GETs without a body (Node only; Deno and browsers unchanged; coordinator calls unchanged); refuses redirects when asked (`redirect: 'error'`), destroys the request on abort. Test in `tests/webgpu/browser/native_contributor.test.mjs` (5 pass). Rerun on the same host: see Result. Host deleted by its recorded ID after 0.54 h, about **$0.11**.

## Conclusion

The slow cold loads were the native contributor's HTTP client. Node's built-in `fetch` delivered large Spaces downloads at about 1 MB/s on a long route, while the host and Node's own core `https` module did 40–66 MB/s; with the downloads routed through core `https`, a contributor loaded the whole 16.5 GB model in 3.5 minutes instead of about 5 hours. This corrects the picture of reports 169, 172 and 177: neither the bucket, nor the route, nor the number of parallel pieces was the limit. One cost appeared: peak resident memory went from about 1.2 GiB to 16.4 GiB, because pieces now arrive faster than they are released (corrected by report 182: Dawn kept a staging copy of every upload until the queue was flushed; fixed there); that should be bounded before this becomes the default for contributors with less RAM.

## Limits

- One host and one route (Colorado to fra1); browsers' fetch was not measured (Chrome's does not share undici's code).
- Peak memory not bounded by this change.
- The undici cause (a receive window near 128 KB) is inferred from the throughput over round trip, not traced in undici itself.

## Reproduce

On a US host with this tree: `runs/experiments/DO-62/tools/host.sh setup`, `host.sh launch`, `LOADS_ARGS="--window 600" host.sh loads T`; `tools/curl_order.sh BASE order.txt 4 120 OUT`; `node tools/node_fetch.mjs URL BYTES REPO`; `python3 tools/timeline.py LOG`.
