# A block relay: taking the coordinator out of the data path

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | HTTP path in production mode (experiment 24's job), PostgreSQL store, NumPy `http-worker` processes claiming 8 tasks each, tile 512, plus a Go block relay (`relay/`) on the same machine |
| **Result** | With workers fetching inputs from the relay and uploading results to it, the coordinator's traffic fell from 12.8 GB out and 5.3 GB in per update to 18 MB and 3 MB, and its tensor reads and socket writes from 16.8 s to 1.8 s of thread time. The update did not get faster on this machine: 14.9 s against 12.8 s at 4 workers, 14.6 s against 12.7 s at 8. Neither setup gains from 8 workers over 4. Without the relay the coordinator is the limit; with it, the plan's critical path and the workers' own Python-side byte handling are. Losses identical throughout; 0 relay misses. |
| **Evidence** | `runs/experiments/028/omarchy-20261026-*/` (profiles and logs), not synced |

## Question

Experiment 27 found the coordinator's byte path, not chain latency, bounding an update at 4 workers: 10 s of tensor reads and 6.7 s of socket writes of server thread time per 12.5 s update, and bulk tasks waiting 200 to 600 ms in the queue for it. The relay in `relay/` is a second server that holds blocks by content hash, so the coordinator keeps scheduling, leases, commits and validation and stops moving bytes. How much of the coordinator's work does it remove, and what does an update cost then?

## What was built

- **The relay** (`relay/`, Go, standard library only): `PUT /blocks/{key}` stores bytes whose SHA-256 is the key, a repeat adds a reference, `?f64=1` also requires finite float64 values; `GET /blocks?keys=…` returns several blocks in one Ygaraty binary frame; `DELETE /blocks` drops one reference per key and frees a block at zero; `GET /status`. About 330 lines, 10 GB/s on loopback for 2 MiB blocks.
- **Coordinator** (`YGARATY_RELAY_URL`, `YGARATY_RELAY_TOKEN`): the in-memory scratch tier pushes every block it writes to the relay under the SHA-256 the store already records, and drops a reference for every block it frees, so identical bytes shared by two versions (zero tiles) survive the first release. A claim reply names each input's blocks (hash, physical shape, region, orientation) instead of carrying its bytes; `BlockTensorStore.locate` selects them exactly as `read` would. Inputs whose blocks live only in the durable store (weights before the first update) still travel inline.
- **Worker** (`ygaraty http-worker`): fetches a reply's relayed blocks in one request and assembles each region itself (`storage/relay.py:assemble_region`, the same slicing as the store's read), keeps relayed tiles in its tile cache under the same keys, and stores each result in the relay as tiles, sending the coordinator only their hashes. The coordinator records the result as remote blocks and reads the bytes only when it needs them: the 1% sampled validation, the loss, persisting weights and moments, checkpoints.
- Not done: the browser SDK does not know the relay, so browser contributors still go through the coordinator. [Experiment 50](050-browser-workers-on-the-relay.md) plans that path (2026-10-29).

## Setup

One run per setting, 3 updates each (the first a warm-up), Ryzen 9 PRO 6950H, nothing else running, the dataset's first batch (43 source and 56 target steps). Relay budget 8 GiB; it ended every run empty (every block freed), with 0 misses.

## Result

Seconds per update, and the coordinator's bytes per update.

| Workers | Relay | Steps per task | Seconds per update | Coordinator sent | Coordinator received |
|---:|---|---:|---:|---:|---:|
| 4 | no | 1 | 12.77 (12.85 and 13.40 in earlier runs today) | 12,791 MB | 5,268 MB |
| 4 | coordinator pushes results, workers fetch | 1 | 16.14 | 18 MB | 5,268 MB |
| 4 | workers fetch and upload | 1 | 14.90 | 18 MB | 3 MB |
| 4 | workers fetch and upload | 4 | 13.54 | 15 MB | 2 MB |
| 4 | no | 4 | 12.53 (experiment 27) | 12,320 MB | 5,268 MB |
| 8 | no | 1 | 12.67 | 13,478 MB | 5,268 MB |
| 8 | workers fetch and upload | 1 | 14.59 | 18 MB | 3 MB |
| 8 | workers fetch and upload | 4 | 13.46 | 15 MB | 2 MB |

Coordinator thread time per update, 4 workers, one step per task:

| Stage | No relay | Relay |
|---|---:|---:|
| Tensor reads | 9.86 s | 1.04 s |
| Socket writes | 6.93 s | 0.71 s |
| Tensor writes (store) | 4.20 s | 0 (results are hashes) |
| HTTP parsing | 1.84 s | 0.43 s |
| Persist to durable store | 2.59 s | 2.44 s |
| Relay deletes | | 0.70 s |
| Claims held open (workers waiting) | 7.04 s | 14.64 s |

Per worker per update, 4 workers, one step per task. *Corrected 2026-10-26:* the first version of this table counted each relay fetch twice, inside the claim's decoding and again as relay traffic, because the worker fetched before closing the claim's span. The relay column below is a re-measurement with the attribution fixed (14.1 s per update, batched result uploads); its stages now add up to the wall time.

| | No relay | Relay |
|---|---:|---:|
| Kernels | 2.82 s | 2.92 s |
| Claim requests (mostly waiting for a ready task) | 6.92 s | 4.08 s |
| Result requests | 2.22 s | 1.09 s |
| Relay fetches and uploads | | 4.98 s |
| Region assembly | | 0.89 s |
| Decoding and encoding, including result hashing | 0.58 s | 0.10 s |

With 8 workers and the relay, each worker waits 6.6 s per update for a ready task (44.9 s of held-open claims) and computes for 1.8 s; its relay and decoding figures from that run carry the double count described above. The relay served 13.5 GB and took 5.3 GB per update.

## Reading

- **The coordinator's data path is gone.** Its per-update bytes fell by 99.9%, and its byte-moving stages from about 23 s of thread time to under 3 s. What remains on it is scheduling, leases, the 1% recomputation and the step end. That is the property a growing pool needs, and the first experiment is the one that measured it.
- **It did not make this machine faster**, for three reasons that the profile separates. The bytes still move, now through two extra requests per task (fetch, upload): 5 s of relay traffic per worker per update. Assembling regions from blocks costs 0.9 s and hashing results is a small part of 0.1 s; an attempt to cut those (results as one block, hashing in the relay) made the relay path slower, since whole blocks are larger than the regions consumers read. Second, every chain hop now includes a relay fetch, so the recurrence costs more per step; four steps per task recovers 1.4 s of that. Third, there is no more parallel work to give: at 8 workers each one idles 6.6 s per update because nothing is ready, which is the plan's critical path, the floor experiment 27 described.
- **Neither setup scales from 4 to 8 workers**, for different reasons. Without the relay the coordinator saturates (tensor reads 21 s, socket writes 13 s of thread time at 8 workers). With it, the dependency structure does. Removing the coordinator from the data path exposes the next limit; it does not remove it.
- **The first attempt was worse than nothing.** With the coordinator pushing results to the relay itself (the simplest integration), an update took 16.1 s: 5.3 GB of synchronous pushes on the result path, and 70 retried tasks per update from an ordering bug between relayed and cached tiles in one reply. Workers uploading directly fixed both.

## What this changes in the research map

Bytes through the coordinator (area D) are solved on one machine for Python workers. The next limits are the critical path of the LSTM plan (area E, a floor of roughly 6 s per update at any worker count) and the two extra requests per task the relay adds. Over a real network the relay's extra round trips per hop cost more and chunked chains matter more; both wait for the latency-injection run (area I).

## Limits

One run per setting on one batch. Everything on one machine, so coordinator, relay, PostgreSQL and workers share 8 cores, and "8 workers" means contention, not a bigger pool. No browsers: the SDK has no relay path yet. The relay token in a claim reply is the same for reads and writes; a public pool needs a read-only token for contributors.

## Running it

```bash
cd relay && go build -o ../relay-server ./cmd/relay && cd ..
./relay-server -addr 127.0.0.1:8093 -token SECRET -max-mib 8192 &
YGARATY_RELAY_URL=http://127.0.0.1:8093 YGARATY_RELAY_TOKEN=SECRET ygaraty experiment 28
```

Needs the experiment-24 setup (`YGARATY_DATASET_URL`, `YGARATY_STORE_URL`) and Go 1.22 or later. About 9 minutes on a Ryzen 9 PRO 6950H.
