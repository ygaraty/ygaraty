# Re-requesting crawling pieces did not speed a cold load from Spaces: new connections crawled too

| | |
|---|---|
| **Conclusion** | Inconclusive: two of three floor windows completed; 0.85–0.92 vs 0.95 MB/s, far below 3× (measured, RTX 4090 US) |
| **Status** | executed |
| **ID** | DO-103 (report 172) |
| **Category** | browser |
| **Where** | remote (one Vast.ai RTX 4090 24 GB in California running both `ygaraty launch` and a native contributor; pieces from the production bucket, read anonymously) |
| **Priority** | med |
| **Date** | Added 2026-11-24; finished 2026-11-24 |
| **Path** | `downloadPiece` rate floor (`?minRate=1`: below 1 MB/s over 10 s after 5 s, re-request on a new connection, at most 3 times per piece) on top of report 169's arm P; public `qwen3.8-27b` 1-worker bundle; 10-minute windows, P F F P P F |
| **Result** | Bytes received per 10-minute window (the coordinator's byte counter): P 0.95, 1.00, 0.92 MB/s (median 0.95); F 0.85 and 0.92 MB/s, with 12 and 13 re-requests; the third F load stopped after 228 s (0.3 MB/s; cause not captured, the driver crashed on its exit). Fewer than three F windows, so by the frozen gate **Inconclusive**; the two completed F windows were at 0.93× P, against the 3× needed. Peak memory 0.97–1.0 GiB in both arms |
| **Evidence** | measured-remote: `runs/experiments/DO-103/vast-do103e-20261124/` (`launch.log` byte counter, `loads/` per-load logs and `loads.json`), pre-checks in `runs/experiments/DO-103/precheck/`; local check `omarchy-20261124-local/` |

## Question

Report 169 (DO-62) found that from a US host most connections to the Spaces bucket crawl (bytes to a few hundred KB/s) while still returning data, and a few run at about 12 MB/s; the no-progress timeout never fires, so loads ran at about 1 MB/s whether pieces were fetched one at a time or in parallel. If a piece below a rate floor is re-requested on a new connection, does the load approach the fast connections' rate?

## Prediction

TBD. If fast and crawling flows are assigned per connection, re-requesting moves most bytes onto fast flows: several MB/s instead of about 1.

## Gate (frozen 2026-11-24, before renting)

- Host: one rented GPU in the US whose route to the bucket shows crawling flows before the run: of four parallel 20 s `curl`s of different pieces, at least two below 1 MB/s (report 169's check). Otherwise that host is released and another tried.
- Arms on that host, a native contributor joining `ygaraty launch` of the public `qwen3.8-27b` 1-worker preset (anonymous reads): **P** (default: three speculative lanes plus demand, 30 s no-progress timeout) and **F** (P with `?minRate=1`: a request below 1 MB/s over 10 s, after 5 s, is re-requested, at most 3 times per piece). Order P F F P P F.
- Each load runs a fixed 10-minute window from its first weight-file line; its rate is GB loaded / window (`runs/experiments/DO-62/tools/loads.py --window 600`). Peak resident memory per load.
- **Works** if F's median rate ≥ 3 × P's median rate, no F load fails, and F's peak memory ≤ P's + 1 GiB. **Rejected** if F's median < 3 × P's. **Inconclusive** if fewer than 3 loads per arm complete their window.

## Protocol

Code and offline tests first (mocked slow and fast responses, retry cap, cancellation). Then a rented GPU whose route to fra1 shows crawling flows (check with a few parallel `curl`s before the run, as report 169). Public bundle, read anonymously (no credentials, no writes). Budget and teardown deadline from the user first.

Related: [169 (DO-62)](169-spaces-parallel-pieces-crawling-flows.md), [113](113-kolibri-single-a100-contributor.md).

## Log

- 2026-11-24: Added at the user's request, from report 169.
- 2026-11-24: User: "run both. budget of 5 approved". Implemented the floor in `spaces_download.mjs` (`minRate`, `rateWindowMs` 10 s, `graceMs` 5 s, `maxSlow` 3; a request cut by the floor does not count as a failure) and `?minRate=MB/s` on the contributor link (logs `weights_min_rate` and each `weights_slow_retry`). Tests `tests/webgpu/browser/spaces_download.test.mjs` (3 pass; browser suite 132 pass). Local check (`omarchy-20261124-local`, public `qwen3.5-0.8b`): F loaded 321 pieces in 89.6 s with no slow retries. Gate frozen above, before renting; about $2 of the $5.
- 2026-11-24: **Host selection by the frozen pre-check.** `do103a` (California, 192.234.50.153): one of four flows below 1 MB/s (0.17, 1.8, 1.9, 5.7 MB/s), released. Another offer there was taken before renting. `do103c` (New Jersey, Quadro RTX 8000): four ranges at 7.6–9.0 MB/s, released (that check used four ranges of one piece rather than four pieces). `do103e` (California, 192.234.50.153, RTX 4090 24 GB, $0.483/h): four pieces, two at 20 KB in 20 s (33 KB/s), two at 7.1 and 12.1 MB/s: passed.
- 2026-11-24: **Loads** 20:34–21:32 UTC (`tools/loads.py --window 600 --min-rate 1`). The window measure in `loads.py` counts completed files only, and this bundle's first files are large (0.33 GB after 350 s), so every window showed the same 0.46 GB of completed files; the rates above come instead from the coordinator's byte counter (`launch.log`), segmented by each load's join time. Load 6 (F) ended at 228 s and the driver crashed on its exit (`ProcessLookupError` from `killpg`; fixed, and the exit code is now recorded). All loads stopped at their windows; none completed the bundle. Host deleted by its recorded ID after 1.06 h; with the released hosts about **$0.59**.

## Conclusion

The floor did what it was built to do (12–13 re-requests per load, no failures from it in the two complete windows), but the re-requested pieces arrived no faster: whatever makes most flows from this host to the bucket crawl is not escaped by opening a new connection. Over a 10-minute window both arms moved about 0.9–1.0 MB/s. The design question moves from the client to the delivery path: a mirror or CDN in front of the bucket, or a different region, rather than smarter retries.

## Limits

- Two of three F windows; one host, one route.
- Windows, not full loads (a full load would take about 5 hours at these rates).
- Rates from the coordinator's display counter (two-decimal GB, about 10 s per frame), aligned to join times within a few seconds.

## Reproduce

On a rented GPU with this tree in `/workspace/src`: `runs/experiments/DO-62/tools/host.sh setup`, `host.sh launch`, `LOADS_ARGS="--window 600 --min-rate 1" host.sh loads "P F F P P F"`; rates from `launch.log`'s `GB of 16.5 GB` frames by join time.
