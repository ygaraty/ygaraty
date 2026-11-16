# The sharded factorization over HTTP: 1, 2 and 4 worker cores on loopback and on home links

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | Workstation (below), then DigitalOcean (section "On DigitalOcean"). HTTP, production claim/result/commit code, NumPy `http-worker` processes (one BLAS thread, one pinned core each), in-memory store, no relay; AMD Ryzen 9 PRO 6950H workstation, 16 logical CPUs |
| **Result** | 8,192 x 8,192, rank 64, 1,024 tiles: a steady update takes 0.9 / 0.8 / 0.5 s on loopback with 1 / 2 / 4 cores, and 51 / 31 / 18.6 s on 100/20 Mb/s links at 50 ms. Loopback is limited by coordinator and byte handling (4 cores: 1.8x), home links by bytes per link (4 cores: 2.7x). Losses equal a dense NumPy run to 1e-16 in all six cells. On DigitalOcean (separate machines, Go latency proxy, twice each): home links 53.2 / 29-32 / 19.6-20.0 s per update for 1 / 2 / 4 workers, so the home-link scaling reproduces; on the plain VPC the update is slower than on loopback and 4 workers are no faster than 2 within five updates |
| **Evidence** | `runs/factorization-http/report.json`, `run.log`, `digitalocean-results.json`, `runs/pods-data/fact47/` (local, not in git) |

## Question

What does the 2D-sharded factorization of experiment 46's plan cost per update when workers are separate processes on one machine, with 1, 2 and 4 cores, once on loopback and once with each worker behind its own emulated home link?

## Setup

- Problem: X is 8,192 x 8,192 (512 MiB, seeded low-rank plus noise), rank 64, tiles 1,024 x 1,024 (64 tiles, 8 U and 8 V shards, 158 tasks per update), SGD with regularization 0.001 and learning rate 500 (the gradient is divided by the matrix size; 5,000 diverges at 2,048 per side). Five updates per cell; the first is a warm-up, because X's tiles travel once and workers then cache them (1 GiB each).
- `ShardedFactorizationJob` (`ygaraty/coordination/factorization_job.py`) puts every tile and shard in the coordinator's bounded in-memory store as its own immutable version and installs the updated shards as the next update's leaves. Nothing computes a whole U or V.
- Workers: `ygaraty.transports.http.worker`, 4 tasks per claim, one BLAS thread, `taskset` to cores 0..n-1; the coordinator and the shaper are confined to the other cores.
- Home link (`ygaraty/experiments/home_link.py`, a Python stand-in for `relay/cmd/latency-proxy`, which needs Go): one proxy per worker, 100 Mb/s down, 20 Mb/s up, 50 ms round trip, shared by that worker's connections. Checked: 4 MiB up and 8 MiB down in 2.4 s (2.40 expected).
- Tolerance fixed beforehand: losses against dense NumPy SGD on the same seeded problem, 1e-9 relative.

## Results

Seconds per update (updates 1 to 5), then the means of updates 2 to 5:

| Link | Cores | Updates | Steady mean | Speedup | Sent / received |
|---|---:|---|---:|---:|---:|
| loopback | 1 | 1.6 0.9 0.9 0.9 0.9 | 0.9 | 1.0 | 993 / 480 MiB |
| loopback | 2 | 1.3 0.8 0.8 0.7 0.8 | 0.8 | 1.1 | 1,390 / 480 MiB |
| loopback | 4 | 0.8 0.6 0.5 0.5 0.5 | 0.5 | 1.8 | 2,101 / 480 MiB |
| home | 1 | 94.3 50.9 50.9 50.9 51.0 | 50.9 | 1.0 | 993 / 480 MiB |
| home | 2 | 47.7 36.8 29.5 29.9 29.5 | 31.4 | 1.6 | 1,543 / 480 MiB |
| home | 4 | 24.6 22.1 18.9 17.1 16.3 | 18.6 | 2.7 | 2,089 / 480 MiB |

All six cells ran with no retries and a loss falling from 5.741 to 0.5184 in five updates, the same to 1e-16 as the dense reference in every cell.

- **Loopback.** A tile's three matrix products are small next to moving 8 MiB of X and about 1 MiB of result per task through one Python coordinator; 4 cores make the update 1.8 times faster, 2 cores barely 1.1. The coordinator's sent bytes grow with cores (993 to 2,101 MiB over five updates) probably because each worker caches only the tiles it saw and tasks are not routed to the worker that holds their X (not measured).
- **Home link.** A worker's uplink carries its tiles' results (about 1 MiB each, 64 per update, so 64 MiB at 20 Mb/s is 26 s of one link) and its downlink the factor shards it does not hold; with a cold X cache the first update moves 512 MiB through the downlinks (94 s on one, 25 s on four). Time per update falls with the number of links, 51 to 31 to 17-19 s: bytes per link, not cores, set the time, which is what experiment 31 found for the Transformer.
- The update-to-update trend with 4 home links (22, 19, 17, 16 s) is consistent with the X cache filling as tasks land on workers that already hold their tile (the cache counters were not read); steady state was not reached in five updates.

## On DigitalOcean

One `g-8vcpu-32gb` coordinator and four `c-2` Droplets (1 vCPU worth of BLAS thread each, `http-worker --max-tasks 4 --cache-mib 1024`), lon1, one VPC (private addresses), 32 minutes of machine time, about $0.35. Twelve fresh jobs of five updates, the same 8,192 problem: three worker counts, plain VPC network or the Go latency proxy at 100/20 Mb/s and 50 ms (`YGARATY_SHAPE=1`), two repetitions in opposite orders (plain then home, home then plain). Every job completed with no retries and its losses equal dense NumPy to 1.1e-16.

Seconds per update; the two repetitions are separated by `/`, "steady" is the mean of updates 2 to 5 for each:

| Link | Workers | First update | Steady | Last two updates |
|---|---:|---|---|---|
| plain | 1 | 4.5 / 4.1 | 1.63 / 1.61 | 1.54 / 1.60 |
| plain | 2 | 2.8 / 3.0 | 1.25 / 1.34 | 1.07 / 1.03 |
| plain | 4 | 2.6 / 2.8 | 1.64 / 1.51 | 1.24 / 1.23 |
| home | 1 | 97.8 / 97.9 | 53.2 / 53.3 | 53.2 / 53.2 |
| home | 2 | 49.2 / 49.2 | 32.4 / 29.1 | 27.6 / 29.3 |
| home | 4 | 25.1 / 25.3 | 20.0 / 19.6 | 18.8 / 18.0 |

- **Home links reproduce the workstation.** One worker takes 53.2 s whichever machine it runs on (the link, not the CPU, is the bottleneck), two take 27 to 31 s and four 17 to 20 s, against 51 / 31 / 18.6 s on the workstation behind the Python shaper. The first update, which carries X once, is 98 / 49 / 25 s against 94 / 48 / 25 s. Two workers had one slow update in one repetition (45.8 s among 27 to 28 s) and ranged 27 to 31 s in the other; there is no explanation for it in this data.
- **Plain network.** One worker takes 1.6 s per update here against 0.9 s on the workstation's loopback (a c-2 core is slower than a Ryzen 6950H core, and a real network adds to every task). Two workers reach 1.0 to 1.1 s by the last updates; four are no better than two (1.2 s) and were still falling, so the five-update steady means (1.5 to 1.6 s) are not a converged estimate. Nothing here separates coordinator-bound from cache-fill effects.
- The coordinator sent 993 MiB over the five updates to one worker, 1,238 to 1,542 MiB to two and 2,000 to 2,228 MiB to four (workstation: 993 / 1,390 to 1,543 / 2,089 to 2,101): more workers fetch more tiles, consistent with tasks not being routed to the worker that holds their X (the cache counters were not read).

So the workstation's home-link scaling is not an artifact of sharing one machine or of the Python shaper, but loopback scaling past two workers is not established either way.

## Limits

- One run per cell, one seed, one machine. The six cells ran in sequence, loopback first; there is no repeat to give a spread.
- The Python shaper is not the Go proxy of experiments 31 to 42, and it models neither jitter nor loss. Do not compare its seconds with theirs.
- X lives in the coordinator's memory (512 MiB). This says nothing about a matrix larger than memory: that is experiment 46, which needs a streaming X source.
- On the workstation the workers share one machine's memory bandwidth and caches; the DigitalOcean section runs them on separate machines.
- The DigitalOcean runs repeat each cell once (two jobs), five updates each; the plain-network 4-worker cell had not settled.
- Not tuned: tasks per claim, tile size, cache size, X-aware task routing.

## Reproduce

Locally (needs 6 free CPUs; about 15 minutes, 3 GiB):

```bash
ygaraty experiment factorization_http --updates 5 --check            # --cores 1 2 4 --links loopback home are the defaults
```

On DigitalOcean with the Python worker (done 2026-10-28, about 30 minutes and $0.35; `YGARATY_BRANCH` defaults to main):

```bash
ygaraty experiment pods up fact47 --provider digitalocean --location lon1 --server g-8vcpu-32gb --workers 4 --worker-size c-2 --minutes 90 --max-hourly 1
# on every machine: curl -fsSL .../scripts/remote.sh | bash -s worker-setup (the server too), then for each k in 1 2 4, shaped and plain, twice:
# server:  ./scripts/remote.sh serve-factorization 5                 # read job and token from the run folder's job.json
# w0..wk-1: YGARATY_SHAPE=1 ./scripts/remote.sh factorization-worker SERVER_PRIVATE_IP:8000 JOB TOKEN wN     (omit YGARATY_SHAPE for the VPC's own network)
ygaraty experiment pods down fact47                    # fetches runs/ first
```

Each k is a fresh job, so the workers' caches start cold. A driver script ran the twelve jobs in sequence over `pods ssh`, passing job and token to the workers without printing them.
