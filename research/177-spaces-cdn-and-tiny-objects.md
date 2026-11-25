# The Spaces CDN gave no gain because direct downloads were not slow: the "crawling flows" of reports 169 and 172 were tiny objects

| | |
|---|---|
| **Conclusion** | Rejected: CDN 25.3 vs origin 21.6 MB/s; direct downloads were never slow, tiny objects had been read as crawling flows (measured, US host) |
| **Status** | executed |
| **ID** | DO-104 (report 177) |
| **Category** | browser |
| **Where** | remote (a Vast.ai RTX 3060 in New York; a temporary test Spaces bucket in fra1 with a CDN endpoint, all deleted after the run) |
| **Priority** | high |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | Test bucket `ygaraty-dev-cdntest-8b5044` (fra1) holding server-side copies of 98 public `qwen3.8-27b` pieces (2.03 GB, public-read) behind a DigitalOcean CDN endpoint; 120 s windows of 4 parallel downloads from the host, origin against CDN |
| **Result** | Origin 22.4 (half A) and 20.7 MB/s (half B); CDN cold 25.3 MB/s (half A; the cold window on B left no output, cause not captured); CDN warm 19.6 and 27.3 MB/s (half B). Each half (about 1 GB) finished in 40–52 s in every window. Cold CDN median 25.3 against origin median 21.6 MB/s: 1.17 ×, against the 3 × needed, so **Rejected**. **Found:** every large object ran at 3–15 MB/s per connection on all three hosts checked; the "crawling" flows in this experiment's pre-check and in reports 169 and 172 were small objects (192 B to 160 KB) downloaded completely, whose rate only reflects about 1.1–1.5 s to the first byte |
| **Evidence** | measured-remote: `runs/experiments/DO-104/remote/` (window JSON per object: size, first byte, seconds), pre-checks `runs/experiments/DO-104/precheck-*.txt`, `state.json` (what was created and deleted) |

## Question

Reports 169 and 172 found that from some US hosts most connections to the fra1 bucket crawl (bytes to a few hundred KB/s) while a few run at about 12 MB/s, so a 16.5 GB cold load takes about 5 hours, and neither parallel pieces nor re-requests help. Does serving the same objects through the Spaces CDN (edges near the host) make those downloads fast?

## Prediction

TBD. The CDN's US edge should fetch from fra1 over the provider's backbone once and serve the host locally; a cold pull is limited by the edge's fetch, a warm one by the host's link.

## Gate (frozen 2026-11-25, before creating anything)

- Objects: about 2 GB of pieces in two disjoint halves A and B. Host: a rented US GPU whose pre-check shows crawling direct flows (as report 172: of four 20 s `curl`s from the bucket, at least two below 1 MB/s); otherwise another host.
- Windows of 120 s, 4 parallel `curl` streams taking objects of a half in turn; throughput = bytes received / 120 s. Order: O on A, C on B (cold), C on B (warm), C on A (cold), O on B.
- **Works** if the median of the two cold CDN windows is at least 3 × the median of the two origin windows. **Rejected** otherwise. Reported: warm CDN window, first-byte times, error counts.
- Production stays untouched: the test bucket and CDN endpoint are created under recorded names and deleted after the run.

## Protocol

`runs/experiments/DO-104/tools/cdn_bucket.py create|delete` (bucket, server-side copies, CDN endpoint; state in a local file), `tools/windows.sh` on the host. Budget: under $0.50 for the GPU host (user's choice), plus cents of Spaces transfer.

Related: [169 (DO-62)](169-spaces-parallel-pieces-crawling-flows.md), [172 (DO-103)](172-spaces-rate-floor-crawling.md).

## Log

- 2026-11-25: Added at the user's request ("Spaces CDN test"); the production bucket has no CDN endpoint (its `.cdn.` name does not resolve), and the user chose a separate test bucket. Gate frozen above.
- 2026-11-25: The production Spaces keys are scoped to that bucket and could not create one; with the user's choice, a temporary full-access Spaces key was created through the DigitalOcean API (`tools/cdn_bucket.py key`), used to create the test bucket and copies, and deleted with them. Bucket and CDN endpoint created in 15 s (98 objects, 2.03 GB; CDN host resolves to Cloudflare addresses).
- 2026-11-25: Hosts by the frozen pre-check: `do104a` (RTX 4090, California, 192.234.50.153) one of four "slow", released; a second 4090 there priced above the cap; `do104h1` (RTX 3060, Oregon) one of four, released; `do104h2` (RTX 3060, New York, $0.083/h) two of four, kept. Windows in order O-A 22.44, C-B (no output), C-B 19.57, C-A 25.30, O-B 20.71 MB/s; a later C-B window (warm) 27.33 MB/s.
- 2026-11-25: **The pre-check was wrong.** The "slow" objects were 192 B, 163,840 B and other pieces under 1 MB (449 of the bundle's 863 pieces are under 1 MB; median piece 160 KB), each downloaded in full in about 1.1–1.5 s. Every object of 17 MB or more ran at 2.8–15.6 MB/s on every host. The same artifact explains report 169's raw check (192 B, 1 KB and 138 KB objects next to a 50 MB one at 12.2 MB/s) and report 172's host selection; both reports now carry a correction.
- 2026-11-25: All created resources deleted by their recorded names (CDN endpoint, 98 objects, bucket, temporary key; the bucket URL answers 404); hosts deleted by their recorded IDs. About $0.12 of GPU time plus cents of Spaces transfer.

## Conclusion

The CDN is not the fix, because the route from US hosts to the bucket was never the problem: with plain parallel downloads a host pulls these pieces at 20–25 MB/s, about 16.5 GB in 12–14 minutes. The slow cold loads of reports 169 and 172 (about 1 MB/s, measured from the coordinator's byte counter) must therefore come from the contributor's own download path: per-piece request overhead (half the pieces are under 1 MB, at about 1.5 s each to the first byte, about 11 minutes for those alone one at a time), the descriptor call per piece, hashing, or Node's fetch. That is the question to answer next, on one host, with a per-piece timeline of the contributor's loads.

## Limits

- One host for the windows; one cold CDN window instead of two.
- Windows of objects about 1 GB each finished early, so the rates are over 40–52 s.

## Reproduce

`runs/tools/m8venv/bin/python runs/experiments/DO-104/tools/cdn_bucket.py key|create|delete STATE.json`; on the host `python3 windows.py BASE_URL A.txt OUT.json` per window.
