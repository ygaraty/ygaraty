# Does gzip on the wire pay for the Transformer on home links? Yes on the weak workers, 4% faster; no on the strong ones

| | |
|---|---|
| **Date** | 2026-10-29 |
| **Path** | Experiment 42's best setup on DigitalOcean (nyc3, one private network; lon1 no longer sells c-2): the 30.3M Transformer, owned layers and 8 vocabulary slices, float32 wire with bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches, capacity placement with one layer per strong worker (`never`); 16 workers per job, 8 weak (c-2, 1 thread, links shaped to 100/20 Mb/s) and 8 strong (c-8, 4 threads, 500/100 Mb/s), 50 ms round trip. Between jobs only the code version and which workers run `--gzip-frames` change |
| **Result** | Gzip on the 8 weak workers only: 6.01, 6.02 and 5.97 s per update against 6.24, 6.23 and 6.33 s with the new code and no frame gzip (4.3% faster; 32.2 against 33.6 years by the nanochat KPI). On every worker: 6.06, 6.19, 6.12 s (2.4%, within the repeats' spread). On the strong workers only: 6.18 and 6.23 s (no gain). Experiment 42's commit reproduces its 6.18 s (6.27 and 6.18). Losses are identical to the last bit in all 13 jobs; no retries or rejected results. Strong workers lose time to compressing (their relay time rises 4%), weak ones save 9% of theirs; the bytes uploaded to the server fall 7% with the weak workers gzipping and 18% with all |
| **Evidence** | `runs/pods-data/exp49-jobs/` (per-job reports, updates, relay and coordinator logs, `spans-summary.json`, network counters; local, not in git) and the machines' logs and spans in `runs/pods-data/exp49/`; local gate in `runs/gzip-blocks-{unsorted,sorted}/` |

## Question

Experiment 42's best update is 6.18 s, bound by bytes through the relay and the workers' uplinks. Frames and relay blocks already travel as float32 or bf16 (experiments 31 and 34); gzip is off for them (`--gzip-frames`, `storage/relay.py`, `transports/http/compression.py`). Does gzip on top of that wire format shorten the update on 100/20 Mb/s links, or does the CPU it costs on a 1-thread weak worker and the small saving on float data (the code's comment says 4 to 8% for floats) make it a loss?

## Local gate (done 2026-10-29)

`ygaraty experiment gzip_blocks` runs the 1.06M-parameter tiny recipe (4 workers, owned layers, float32 wire, bf16 activations, 6 updates) against a local relay and, for every block a client PUTs, also compresses a copy at gzip level 1 and records sizes and times. The run is unchanged. Untrained weights, one machine:

| Blocks | Batches | Share of bytes | Smaller by | Compress | Seconds saved per MiB at 20 Mb/s |
|---|---|---:|---:|---:|---:|
| bf16 (activations and their gradients) | as drawn | 111.6 MiB | 68.5% | 11.7 ms/MiB | 0.287 |
| bf16 | sorted, 3,072 tokens | 48.5 MiB | 32.6% | 24.2 ms/MiB | 0.137 |
| float32 | sorted | 3.3 MiB | 14.4% | 24.9 ms/MiB | 0.061 |
| float64 | sorted | 24.1 MiB | 3.9% | 31.3 ms/MiB | 0.016 |
| unchecked (parameters and the like) | sorted | 12.1 MiB | 69.0% | 8.9 ms/MiB | 0.290 |

The gate passes on the bf16 blocks (at least 3% smaller, and compressing costs a fifth of what the saved bytes take on a 20 Mb/s uplink). The 4 to 8% figure in the code is for random float mantissas; bf16 activations are far more compressible, the drawn batches most (their padding is zeros), and the sorted ones, which experiment 42 used, still lose a third. Not established: untrained activations may compress differently from trained ones; this model is 128 wide against the 30.3M one's 512; compression time was measured on this workstation, and a 1-thread c-2 Droplet is slower (the weak-only job is where it shows). So the prediction below was wrong for bf16 by a wide margin, and the cloud jobs now test whether the saving survives the real model.

## Prediction made before the gate (kept for the record; the gate contradicted it for bf16, the cloud run showed it right about the strong workers)

- On a 20 Mb/s uplink a 4 to 8% saving is at best a few percent of the transfer time; bf16 blocks may compress differently from float32 ones, which is the point of measuring them separately.
- gzip at the fastest level costs CPU on the worker that sends. A weak worker has one BLAS thread, so it may lose more than it saves; a strong worker on 500/100 Mb/s probably gains nothing.

## Needs first

- `scripts/remote.sh` does not pass `--gzip-frames` to the workers it starts (`worker` mode and `serve34`'s printed lines); add an environment variable for it, like `YGARATY_CACHE_MIB` for the factorization worker. The flag itself exists since `5f0493e` (`transports/http/worker.py`, and `RelayClient(gzip=...)` for relay blocks).
- The gzip code (`transports/http/compression.py`, `storage/relay.py`, `transports/http/worker.py`, `webapp/server.py`, `relay/relay.go`) is commit `5f0493e`, already on `main`, which is what the machines clone. The Go relay must be rebuilt from it, since it now decodes and encodes gzip.
- That work also gzips JSON bodies whenever a worker asks (they all do), not only with `--gzip-frames`. So "gzip off" is no longer experiment 42's setup: its 6.18 s was measured with no compression at all. The baseline therefore has to be run, not assumed (see the jobs).
- A local gate before spending on machines: take the blocks one real update writes (capture from a loopback profile at the small recipe), and measure, for float32 and bf16 blocks separately, the gzip ratio and the milliseconds to compress and decompress one block on this workstation and on a c-2 Droplet. If the ratio is below 3% for both, stop: the cloud run cannot pay.
- The repository is private: `pods ship RUN` before `scripts/remote.sh bootstrap` (see `scripts/remote.sh`'s header).

## Plan

- One set of machines as in experiment 42: a g-8vcpu-32gb server, 10 c-2 and 8 c-8 Droplets (`--workers 18 --max-hourly 2`), bootstrap, then jobs of 22 updates each in alternating order, steady mean of updates 2 to 21:

| Job | Compression |
|---|---|
| experiment 42 | the commit experiment 42 ran (`b839f09`), nothing compressed: reproduces its 6.18 s, and tells whether JSON gzip alone moved anything |
| JSON only | the new code, `--gzip-frames` off everywhere (JSON bodies still gzipped) |
| all | `--gzip-frames` on every worker |
| weak only | `--gzip-frames` on the 8 weak workers (low CPU, slow links); the strong ones off |
| strong only | `--gzip-frames` on the 8 strong workers; the weak ones off |

- Each job twice if time allows, since experiment 47's repeats moved by up to 10%. Losses must be identical to the baseline's: gzip is lossless, so a difference is a bug, not a result.
- Report per-job bytes through the relay, worker send time and CPU time spent compressing (from the spans), not only seconds per update.
- About 100 minutes and $5, as experiment 42 (bootstrap is most of it). Delete only the run's own machines (`pods down RUN`).

## Gates

| Gate | Passes when |
|---|---|
| Local ratio and cost | at least 3% smaller for one block type and compression milliseconds below the time its saved bytes take at 20 Mb/s |
| Same losses | identical to the baseline at every update |
| Cloud result | a job beats the JSON-only job by more than the spread between its two repeats |

If no job wins, record that: it prices the 4 to 8% claim on real links and closes the question for this wire format.

## Result (2026-10-29)

One set of 17 machines (a g-8vcpu-32gb server, 8 c-2 and 8 c-8 Droplets in nyc3, $2.88 an hour, up about 65 minutes), 13 jobs of 22 updates in alternating order, steady means of updates 2 to 21 (per-update spread 0.2 to 0.4 s, so about 0.07 s of error on a job's mean). The experiment 42 jobs ran that commit (`b839f09`) on all 17 machines, the others `ff47670`.

| Job | Gzip | s/update (runs) | Mean | tokens/s | nanochat |
|---|---|---|---:|---:|---:|
| experiment 42's commit | none | 6.27, 6.18 | 6.23 | 416, 422 | 33.4 years |
| new code, JSON gzip only | JSON bodies | 6.24, 6.23, 6.33 | 6.27 | 418, 419, 412 | 33.6 years |
| every worker | + frames and blocks | 6.06, 6.19, 6.12 | 6.12 | 431, 421, 426 | 32.8 years |
| **weak workers only** | + frames and blocks | **6.01, 6.02, 5.97** | **6.00** | 434, 433, 437 | **32.2 years** |
| strong workers only | + frames and blocks | 6.18, 6.23 | 6.21 | 422, 419 | 33.3 years |

Gates: the local ratio gate passed (above); losses are identical in every job; the weak-only job beats the JSON-only job by 0.27 s, against spreads of 0.05 and 0.10 s between their repeats. The all-workers job beats it by 0.15 s against a 0.13 s spread: marginal. JSON gzip alone changed nothing (6.23 to 6.27 s), and the experiment 42 commit and the new code agree.

Where the time went, from the workers' spans (seconds over the 22 updates, mean per worker):

| Time in relay transfers | Weak workers | Strong workers |
|---|---:|---:|
| no frame gzip (4 jobs, 6 runs) | 62.8 to 64.3 | 47.6 to 48.0 |
| gzip on the weak workers | 56.8 to 57.0 | 48.3 |
| gzip on the strong workers | 62.5 to 62.8 | 49.7 to 50.0 |
| gzip on all | 56.1 to 57.1 | 49.7 to 50.0 |

A weak worker's 20 Mb/s uplink is the limit, and gzip takes 9% off its relay time (and 4% off its claim time, 133 to 128 s). A strong worker is not link-bound at 100 Mb/s up with 4 threads, so the compression only adds 4% to its relay time. Compute and post times do not move. That is why "all" sits between the two. Bytes arriving at the server (network counters on the server for one run each; the whole job, setup included): 1,866 MiB with no frame gzip, 1,740 (-7%) with the weak workers gzipping, 1,526 (-18%) with all; bytes it sent stayed at 3.1 GiB in all three, so the downloads were not compressed.

## What this does not show

- The gate's ratios came from the tiny model's untrained blocks, and the cloud jobs trained the real 30.3M model for only 22 updates from initialization, so a trained model's activations may compress differently.
- Downloads did not shrink (3.1 GiB sent in each of the three jobs measured), though workers asked for gzip; why was not investigated. They are the larger direction on the weak workers' 100 Mb/s links, so this is where more may be left.
  - **Review (2026-10-29): by design, not a bug, and the lever is larger than 10 to 20%.** Workers do send `Accept-Encoding: gzip` on every claim, but the coordinator gzips only text bodies: `compression.encode_body` calls `_gzip(..., frames=False)`, so a claim reply, which is a binary frame carrying the pushed relay blocks (this job runs `--pushed-inputs`), goes out raw whatever the worker asked (`ygaraty/transports/http/compression.py`, documented there as a coordinator-CPU choice made before the gate above measured bf16). So the weak workers' downloads, which are those pushed blocks, could not shrink. Two measurement gaps hide this: the coordinator's `sent_bytes` is the wire count, but the relay's `ServedBytes` adds `len(e.data)` before `writeBlocks` compresses (`relay/relay.go`), so relay GETs that were gzipped (a `--gzip-frames` worker asks) show no saving in its status either. What it could be worth: the pushed blocks are bf16 activations, which the gate above compressed to 68.5% as drawn, so gzipping the frames for workers that opt in (a request header set by `--gzip-frames`; browsers must not be opted in by their default `Accept-Encoding`) would take about 30% off the weak workers' downloads, at 11.7 ms per MiB of coordinator CPU at level 1: about 37 s of CPU over a job's 3.1 GiB. Still to do: the header and the frame path in `encode_body`, a wire-bytes counter on the relay, and the run.
- One region, one set of 17 machines, emulated links with no jitter or loss; each job twice or three times.
- The bytes are from the server's network interface for one run per job, not per worker; per-link bytes were not recorded.

## Reproduce

```bash
ygaraty experiment pods up exp49 --provider digitalocean --location nyc3 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 140 --max-hourly 2
ygaraty experiment pods up exp49 --workers 16 --worker-size c-8 --max-hourly 2          # w8 to w15, the strong workers
ygaraty experiment pods ship exp49                                                    # scripts/remote.sh and the deploy key
# server: bash /root/remote.sh bootstrap   workers: bash /root/remote.sh worker-setup   (about 3 minutes)
# per job, on the server (checkout b839f09 or main on all 17 first):
#   YGARATY_PLACEMENT=capacity YGARATY_PLACEMENT_PAIRS=never YGARATY_BATCH=64 YGARATY_LENGTH_WINDOW=64 YGARATY_BATCH_TOKENS=3072 YGARATY_WORKERS=16 bash /root/remote.sh serve34 22
# weak workers:   YGARATY_SKIP_SETUP=1 [YGARATY_GZIP_FRAMES=1] bash /root/remote.sh worker <coordinator> <relay> <job> <token> wN
# strong workers: YGARATY_SKIP_SETUP=1 [YGARATY_GZIP_FRAMES=1] YGARATY_UP_MBPS=100 YGARATY_DOWN_MBPS=500 YGARATY_WORKER_THREADS=4 bash /root/remote.sh worker ...
ygaraty experiment pods down exp49
ygaraty experiment gzip_blocks --output runs/gzip-blocks-sorted --steps 6 --length-window 16 --batch-tokens 3072    # the local gate
```

`runs/pods-data/exp49-jobs/drive49.py` is the driver used (10 jobs; `EXTRA=1` runs the three with network counters); about $3.
