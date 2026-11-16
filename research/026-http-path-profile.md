# Where does the HTTP path's time go?

| | |
|---|---|
| **Date** | 2026-10-25 |
| **Path** | Webapp HTTP server, production `Job` claim/result code, in-memory or PostgreSQL block store; 4 headless workers: Node running the browser SDK, or `ygaraty http-worker` (NumPy, same protocol) |
| **Result** | On a Ryzen 9 PRO 6950H, the HTTP path started with workers sleeping 30% of each update and, with fast workers, a job lock held 77–99% of the time. Fixes measured one at a time made MLP updates 1.2–1.7× faster in memory and 1.7–4.2× faster on PostgreSQL, and small-task LSTM updates 2.1–3.5× faster, with identical losses. The lock is now held 1–15% of the time. What limits each setup now is listed under "Current bottleneck". |
| **Evidence** | `runs/http-profile/` (`profile.json` per run), not synced. PostgreSQL 18.6 on the same machine. |

## Question

Experiment 23 profiled the ZeroMQ coordinator. The path Ygaraty ships is the HTTP one: the webapp server, browser contributors running `sdk.mjs`, and a block tensor store. Where does an update's time go there, and do the ZeroMQ fixes carry over?

## Setup

- **Workload:** the sized MLP of experiments 21 and 23 (batch 64, 1024 → hidden → 512, SGD, learning rate 0.01). Hidden 4096 for the comparison of fixes, and 32768 for experiment 23's workload. Tiles 128, 256 and 512.
- **Execution:** the real `ThreadingHTTPServer` and handler, and a `Job` subclass that differs from production only in how it builds the plan: it uses the MLP tape instead of a webapp recipe. Claims, results, commits, leases and the in-memory `MemoryStore` are production code. The SGD update runs as worker tasks, as in production. Workers are 4 Node processes running `BrowserWorker` from `sdk.mjs` unchanged (`http_profile_worker.mjs`), with float64 CPU kernels.
- **Measurement:** server stages are exclusive times, from wrappers around the job's and store's methods and around the job lock, on every request thread. Threads overlap, so server stages add up to more than the wall time. The profile also records how long the lock is held. Workers record every request, computation and sleep, and spans are matched to each update's time window. The first update is a warm-up; then 3 updates per run, 3 runs per tile.
- **Machine:** AMD Ryzen 9 PRO 6950H (8 cores), Linux, Python 3.14.7, NumPy 2.5.4, Node 26.8.2.

## First profile

Seconds per update (hidden 4096), median of 9 updates. Worker columns are per worker.

| Tile | Wall | Kernel | Network (claim + result) | JS encode and decode | Sleeping (no ready task) | Job lock held |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 3.66 | 0.47 (13%) | 2.36 | 0.20 | 0.54 | 71% |
| 256 | 2.24 | 0.56 (25%) | 0.79 | 0.21 | 0.67 | 45% |
| 512 | 2.15 | 0.91 (42%) | 0.45 | 0.13 | 0.67 | 29% |

- **Workers slept about 30% of each update.** When no task was ready, for example while a join waited for its last parts, the server answered "waiting" and the SDK slept a fixed 300 ms.
- **The job lock was busy.** It was held 71% of the time at tile 128, and request threads waited 2.9 s per update for it. Claim replies were encoded and written to the socket while holding it.
- **SHA-256 on every block read and write.** It cost 0.6–1 s of server time per update, and 2.7 s on experiment 23's workload.
- **Copies on both sides.** The server joined each reply into one buffer with `tobytes()`. The SDK decoded and encoded one float at a time through a `DataView`, then copied the inputs again with `Float64Array.from`.
- **Cleanup that grows with the number of blocks.** At tile 128, deletes cost 0.41 s, because `MemoryStore` scans all blocks on every delete, and adds up all block sizes on every write.

## Fixes, one at a time

Each fix was profiled, checked with `ygaraty parity check --executor http` (6/6 runs passed, none worse than the baseline: three LSTM cases with Node workers on the real server), and run through the test suite. Losses were identical in every run.

| Step | Change | Tile 128 | Tile 256 | Tile 512 |
|---|---|---:|---:|---:|
| Before | | 3.66 (13%) | 2.24 (25%) | 2.15 (42%) |
| 1 | Claims are held open until a task is ready | 3.27 (15%) | 1.72 (36%) | 1.86 (59%) |
| 2 | Claim replies are written outside the job lock | 3.21 | 1.67 | 1.82 |
| 3 | In-memory blocks are hashed on their first read only | 3.16 | 1.66 | 1.80 |
| 4 | Zero-copy binary frames | 3.01 (16%) | 1.43 (43%) | 1.68 (63%) |

Seconds per update, with the kernel share in brackets.

**1. Held-open claims.** An empty claim is held open for up to 10 s, without the job lock, until a commit or a retry may have made a task ready (`Job.claim(worker, wait=…)`; the HTTP route passes `CLAIM_WAIT_SECONDS`). Sleeping went to zero. `claim(worker)` without a wait behaves as before.

**2. Reply outside the lock.** The lock is now held 22–69% of the time instead of 29–82%, but the gain is 2–3%: the request threads then compete for Python's GIL.

**3. Hash once.** A block's key is never rewritten. `MemoryStore`'s bytes never leave the process, so each block is checked once. Postgres and Redis bytes cross a process boundary on every read and are still checked every time, so production keeps its guarantee and doesn't get this gain. Read hashing fell by 40%, but most blocks are read only about twice, so the effect on wall time is small.

**4. Zero-copy frames.** The server pads the JSON header with spaces so the floats start on an 8-byte boundary, and writes the arrays to the socket without joining them. The SDK reads inputs as `Float64Array` views of the reply and writes results with a single copy. Big-endian hosts and unaligned frames fall back to the old per-value path. JavaScript encoding and decoding fell 50–65%. At tile 128, socket writes rose (1.37 → 1.79 s of server time), because a reply is now several writes.

## Experiment 23's workload

Hidden 32768, tile 512, 2 runs of 2 measured updates.

| | Seconds per update | Kernel | Bytes sent per update |
|---|---:|---:|---:|
| ZeroMQ, NumPy workers (experiment 23, after its fixes) | 1.16 | 17% | 1.5 GB |
| HTTP, Node workers, before | 13.2 | 56% | 2.4 GB |
| HTTP, Node workers, after | 11.4 | 68% | 2.4 GB |

The losses match experiment 23's to every printed digit. JavaScript float64 kernels run at about 0.65 GFLOP/s per worker, so arithmetic dominates. HTTP sends 50% more bytes because the SGD update runs as worker tasks, so every weight and gradient travels both ways; the ZeroMQ coordinator updates weights itself.

## NumPy workers on the same server

`ygaraty http-worker` speaks the SDK's protocol (held-open claims, padded binary frames) and runs the allowlisted NumPy kernels. Its kernels are as fast as experiment 23's workers, so what remains is the HTTP path's own overhead. It passes `ygaraty parity check` against PyTorch on the three LSTM cases (executor `http-python`).

Hidden 4096, after the four fixes, median of 9 updates. ZeroMQ is experiment 23's coordinator with caching, on the same machine and workload.

| Tile | ZeroMQ, NumPy workers | HTTP, NumPy workers | HTTP, Node workers |
|---:|---:|---:|---:|
| 128 | 0.44 s | 1.80 s (lock held 77%) | 2.99 s (kernel 16%) |
| 256 | 0.23 s | 0.57 s (lock held 80%) | 1.43 s (kernel 43%) |
| 512 | 0.21 s | 0.41 s (lock held 87%) | 1.68 s (kernel 63%) |

On experiment 23's workload (hidden 32768, tile 512), an update takes 1.16 s on ZeroMQ, 3.55 s on HTTP with NumPy workers (lock held 91%), and 11.4 s on HTTP with Node workers.

- **With fast workers, the job lock is the limit.** Request threads wait on it for 0.8–2.9 s of thread time per update at hidden 4096, and 8 s on the bigger workload. Inside it are tensor reads with their first-read hashing, writes with hashing, deletes with the store's full scans, and commits. Everything else waits, including socket writes and HTTP parsing, which also compete for the GIL.
- **Per-task cost dominates small tasks.** The small LSTM parity case (about 13,000 tasks per update) takes 98 s over HTTP with NumPy workers, against 8 s over ZeroMQ.
- **JavaScript kernels are 20–40× slower than NumPy's** for the same tiles: 0.49–1.07 s against 0.03 s per worker per update.

## The server: only scheduling under the job lock

With NumPy workers, the job lock was held 77–91% of each update. The lock covered everything a claim or a result did: reading and hashing input blocks, writing and hashing result blocks, deleting consumed values (a full scan of the store each time), and every PostgreSQL call. Changes, measured one at a time:

1. **Data work outside the lock.** A claim leases a task under the lock, then reads its inputs after releasing it. That's safe: blocks never change, and an input is deleted only after every task that reads it has committed. A result is checked, verified (when sampled) and written outside the lock, under its attempt's own scope; the lock is taken again only to commit. If the lease was lost meanwhile, the commit is refused and the attempt's blocks are dropped. `MemoryStore` became safe across threads with a short internal lock, and deletes touch only the deleted version's blocks (reference counts).
2. **A PostgreSQL connection pool.** After step 1, PostgreSQL's single shared connection held 82–98% of each update: the queue had moved there. Each request thread now borrows its own connection. The store's SQL already takes row locks in a stable order, for workers in separate processes, so concurrent calls are safe.
3. **Database calls out of the lock.** Recording a lease, publishing assembly and transpose manifests, revoking leases, deleting consumed values and cleaning up expired attempts no longer run under the job lock. Commits queue the ones that scheduling doesn't need, and the same request runs them right after releasing the lock.
4. **Fewer transactions.** On PostgreSQL, a result's blocks and manifest go in one transaction (one per block before). Each request's revokes and deletes go in one `store.release()` call, which is one transaction with set-based SQL. A task now takes about 3 transactions, against 4 plus one per block written or value deleted before.

**In-memory store, NumPy workers,** seconds per update (job lock held):

| Hidden, tile | After fixes 1–4 | After the server changes | ZeroMQ path |
|---|---:|---:|---:|
| 4096, 128 | 1.80 (77%) | 1.33 (13%) | 0.44 |
| 4096, 256 | 0.57 (80%) | 0.44 (12%) | 0.23 |
| 4096, 512 | 0.41 (87%) | 0.28 (7%) | 0.21 |
| 32768, 512 | 3.55 (91%) | 2.04 (9%) | 1.16 |

With Node workers, tile 128 went from 2.99 to 2.50 s. At larger tiles JavaScript kernels dominate and Node barely changed: 11.35 → 11.09 s on hidden 32768.

**PostgreSQL, hidden 4096,** seconds per update, 2 runs of 2 measured updates each. "Before" is commit `da2ba14`, measured in a separate worktree with the same profiler.

| Workers, tile | Before | 1. Data outside the lock | 2. Connection pool | 3. Database calls outside the lock | 4. Fewer transactions |
|---|---:|---:|---:|---:|---:|
| NumPy, 256 | 10.5 | 14.1 | 5.38 | 3.27 | **2.53** |
| NumPy, 512 | 3.91 | 3.44 | 2.14 | 1.55 | **1.39** |
| Node, 256 | 11.7 | 9.45 | 9.84 | 4.44 | **3.52** |
| Node, 512 | 4.39 | 4.39 | 3.33 | 2.80 | **2.54** |

The job lock went from about 99% held to 1–3%. Step 1 alone did not help PostgreSQL, because the shared connection became the queue; it paid off once step 2 removed that.

**Small tasks gained the most.** The LSTM parity cases have thousands of tiny tasks per update. End to end (in-memory store): `lstm-edge-2layer` went from 134 to 38 s with Node workers and from 98 to 32 s with NumPy workers; `lstm-edge-1layer` from 46 to 21 s and from 37 to 18 s.

`ygaraty parity check --executor http http-python` passed 12/12, none worse than the baseline, after each step. The test suite passes with PostgreSQL (`POSTGRES_TEST_URL`, `HTTP_POSTGRES_TEST_URL`, `HTTP_PRODUCTION_TEST_URL` and `YGARATY_TEST_DATASET_URL` set). New tests run several threads against both stores at once, and check that `release()` keeps blocks other versions share.

## Current bottleneck

It depends on the workers and the store:

| Setup | What limits an update now | Evidence (hidden 4096 unless noted) |
|---|---|---|
| Node or browser workers, tiles of 256 and up | **JavaScript float64 kernels** | 43–63% of each update in memory; 68% on hidden 32768 |
| NumPy workers, in-memory store | **The HTTP server's per-request work in one Python process:** socket writes, HTTP parsing, block copies and first-read hashing, all competing for the GIL | Tile 128: socket writes 1.7 s and parsing 0.6 s of thread time in a 1.33 s update. Still 1.3–3× slower than ZeroMQ. |
| Any workers, PostgreSQL | **The database work itself**, now running in parallel | At tile 256, per update: result writes 3.3 s (190 MB stored and compressed), releases 2.4 s, lease records 1.7 s, reads 1.1 s of thread time. Still 5–9× slower than in memory. |
| Small tasks (the LSTM of experiment 24) | **The number of tasks:** each costs at least 2 requests, about 3 transactions and a few ms | An update with tens of thousands of tasks takes tens of seconds whatever the server does per task |

## Afterwards

The production translation job (`DatabaseTranslationJob`, experiment 24) gained a two-tier store on 2026-10-26: transient values in memory, weights and moments copied to PostgreSQL at the step end. On experiment 22's recipe that took an update from 106.5 s to 30.3 s; see experiment 24. The MLP measurements above are unchanged by it.

## Limits

Everything ran on one machine, so there is no real network, and the server, workers and PostgreSQL share CPU cores. PostgreSQL ran with default settings. Redis was not measured. The fix comparison is at hidden 4096; the bigger workload has only 4 measured updates per configuration.

## Running it

```bash
ygaraty experiment 26
```

It needs Node.js (18 or later) and takes about 3 minutes. It profiles Node and NumPy workers one after another (`--store-url postgresql:///<db>` profiles a PostgreSQL store), and writes `profile.json` to the run folder: every update's server stages, lock time, worker stages and loss. One configuration: `ygaraty experiment http_profile --hidden 4096 --tiles 256 --repeats 1 --workers-kind python`.
