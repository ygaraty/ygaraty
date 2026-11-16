# Capacity placement on a mixed pool over DigitalOcean: 25% faster with 6 fast uplinks among 16

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | Experiment 37's best setup (the 30.3M Transformer, owned layers and 8 vocabulary slices, bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches) on DigitalOcean (lon1, one private network) with 16 workers of two kinds: weak (c-2, 2 dedicated vCPUs, 1 BLAS thread, links shaped to 100/20 Mb/s) and strong (c-8, 8 dedicated vCPUs, 4 BLAS threads, 500/100 Mb/s); 50 ms round trip for all |
| **Result** | With the same 2 strong workers among 16, capacity placement (commit 1027ff2) takes the update from 9.7 s (first-come, where the strong workers ended up owning nothing) to 8.8 s, and with 6 strong workers to 7.3 s: 9% and 25% faster, 52 → 47 → 39 years by the nanochat KPI. Losses are identical update by update. The simulator (experiment 38) predicted 7.7 and 5.8 s: the gains are real but smaller, the simulator 14 to 21% fast as usual on home links. Placement gave strong workers consecutive pairs of layers even when there were enough strong workers for one layer each, so 2 of the 6 owned no layer. |
| **Evidence** | Logs in `runs/experiments/039-pods/do-mixed-*` (not synced) |

## Question

Experiment 38 priced capacity placement on a mixed pool at 17% (2 fast uplinks among 16) to 38% (6). Placement had been shown correct only on the tiny model, where the update is all latency (6% there). Does it pay at 30.3M scale on real links?

## Setup

`scripts/remote.sh serve34` with `YGARATY_PLACEMENT` (new, commit fa13fb9) and workers started with per-worker settings: weak workers `YGARATY_UP_MBPS=20 YGARATY_DOWN_MBPS=100`, strong ones `YGARATY_UP_MBPS=100 YGARATY_DOWN_MBPS=500 YGARATY_WORKER_THREADS=4` (also new). Each worker's hello reports its cores, GFLOP/s and memory; the coordinator measured the uplinks at 19 to 21 Mb/s (weak) and 103 to 108 Mb/s (strong), and GFLOP/s at 56 to 73 (weak) and 109 to 223 (strong). The strong workers started a few seconds after the weak ones, so first-come had no reason to favor them. Each run trained 22 updates; the table counts updates 2 to 21.

## Results

| Pool | Placement | s/update | Real tokens/s | nanochat | Simulated (38) |
|---|---|---:|---:|---:|---:|
| 16 weak (experiment 37, another day) | first-come | 10.1 | 260 | 54 years | 9.3 s |
| 14 weak + 2 strong | first-come | 9.7 | 269 | 52 years | 9.3 s |
| 14 weak + 2 strong | **capacity** | **8.8** | 298 | **47 years** | 7.7 s |
| 10 weak + 6 strong | **capacity** | **7.3** | 360 | **39 years** | 5.8 s |

Who owned the layers:

| Run | Encoder 0, 1 | Encoder 2 | Decoder 0, 1 | Decoder 2 |
|---|---|---|---|---|
| first-come, 2 strong | weak, weak | weak | weak, weak | weak |
| capacity, 2 strong | strong (one worker) | weak | strong (one worker) | weak |
| capacity, 6 strong | strong (one worker) | strong | strong (one worker) | strong |

The update that publishes the owners' state took 49 and 48 s with 2 strong workers and 24.6 s with 6. The first update took 72 to 73 s with capacity placement against 41 s first-come: placement waits for every worker's hello.

## What it taught

- **Capacity placement pays at 30.3M scale.** 9% with 2 fast uplinks, 25% with 6, without changing a number of the training. On a volunteer pool with a share of fibre-connected machines it is a free gain.
- **The placement rule leaves gain on the table.** It gives a strong worker two consecutive layers, so two pipeline stages share one uplink, the pattern that slowed experiment 36's 8 workers; with 6 strong workers it used only 4 for layers. A rule that spreads layers one per strong worker while there are enough of them is the next thing to price in the simulator.
- **The simulator stays fast by the usual margin** (14 to 21% here), and ranked the three settings correctly.

## Limits

One run per setting, 20 steady updates each; the strong workers combine a faster uplink, a faster downlink and more compute, so this does not separate which matters; emulated links (no jitter or loss).

## Reproduce

```bash
ygaraty experiment pods up exp39d --provider digitalocean --location lon1 --server g-8vcpu-32gb --workers 14 --worker-size c-2 --minutes 120
ygaraty experiment pods up exp39d --workers 20 --worker-size c-8          # adds w14 to w19, the strong workers
# scripts/remote.sh bootstrap on the server, worker-setup on each worker; then per run, on the server:
#   YGARATY_PLACEMENT=capacity YGARATY_BATCH=64 YGARATY_LENGTH_WINDOW=64 YGARATY_BATCH_TOKENS=3072 YGARATY_WORKERS=16 ./scripts/remote.sh serve34 22
# weak workers:   ./scripts/remote.sh worker <coordinator> <relay> <job> <token> wN
# strong workers: YGARATY_UP_MBPS=100 YGARATY_DOWN_MBPS=500 YGARATY_WORKER_THREADS=4 ./scripts/remote.sh worker ...
ygaraty experiment pods down exp39d
```
