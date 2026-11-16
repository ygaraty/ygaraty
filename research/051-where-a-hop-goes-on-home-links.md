# Where does a hop go on home links? In bytes, not round trips; sending small results inline saves at most 4%

| | |
|---|---|
| **Date** | 2026-10-29 |
| **Path** | Local, on the workstation (no cloud): the 1.06M-parameter tiny recipe (`1m-seed2026-v2k`, 128 wide, 2 layers), 8 NumPy workers (owned layers, 4 vocabulary slices, float32 wire, bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches), 10 updates, the latency proxy at 100 Mb/s down, 20 Mb/s up and 50 ms round trip. No simulator involved |
| **Result** | An update is 0.98 s on loopback and 4.0 to 4.4 s on emulated home links, with identical losses (9.58 to 7.04). The critical chain is 16 hops of about 300 ms: the relay upload of the result 104 ms (a 50 ms round trip plus about 0.4 ms per KB), the claim reply reaching the worker 86 ms, waiting to be handed a task 31 ms, the result post and commit 30 ms, the kernel 17 ms. Moving bytes is about 60% of a hop; the kernels are under 6%. Posting small results inside the result request, to save the relay upload round trip, was tried: at most 3 to 5% faster (3.8 to 3.9 s against 4.0), worse with larger thresholds (6.1 s at 128 KiB, 7 s with everything inline), and one of five runs hung. Reverted |
| **Evidence** | `runs/profile50/` (profile logs, `profile.json`, the capture in `cap/`; local, not in git); `ygaraty/experiments/hop_trace.py` |

## Question

Experiment 42 reached 6.18 s per update on home links with the 30.3M model, and the cost model says an update costs hops times round trips times the round trip (experiment 29). The profiler states "17 worker hops x 3 round trips = 51 round trips", but that line is a formula in `http_profile.py`, not a measurement. What is a hop made of on the tiny model, and which part is cheapest to remove?

## Update on loopback and on home links

| | Loopback | Home links |
|---|---:|---:|
| Seconds per update (median of updates 2 to 10) | 0.98 | 4.41 (3.96 to 4.06 in three later runs) |
| Tasks per update | 364 | 364 |
| Kernel time per worker per update | 0.14 s | 0.17 s |
| Claim network, per worker | 0.81 s | 4.19 s |
| Result network, per worker | 0.10 s | 0.89 s |
| Relay, per worker | 0.02 s | 1.31 s |

Tasks in flight on home links: 5 to 16 for 57% of the update, 2 to 4 for 35%, one for 5%. Fifty-one round trips at 50 ms are 2.55 s, which looked like 58% of the update; the trace below shows that count was not what the time is made of.

## The trace of a hop

With `YGARATY_CAPTURE_DIR` set, every process writes timestamped events on the same monotonic clock. `hop_trace.py` joins them per task (ready, assigned, sent, claim received, kernel, upload, post, commit) and walks the critical chain back from the update's last result. Updates 2 to 9; update 10 (7.6 s) holds the checkpoint and is left out. The chain is 16 hops (19 and 20 in two updates), 4.8 to 4.9 s, close to the uncaptured wall time.

| Phase | ms per hop | Share |
|---|---:|---:|
| Result upload to the relay | 104 | 35% |
| Claim reply reaching the worker (coordinator done to worker has it) | 86 | 29% |
| Ready until a worker is handed it | 31 | 10% |
| Result post until the coordinator commits it | 30 | 10% |
| Kernel | 17 | 6% |
| Kernel end to upload start | 11 | 4% |
| Input fetch and assembly | 9 | 3% |
| Assignment, and the wait before the post | 12 | 4% |

By op on the chain (medians; upload ms and KB, reply ms and pushed KB): `tf_owned_grad` 97 ms for 106 KB, reply 71 ms for 301 KB; `tf_owned_layer` 94 ms for 100 KB, reply 66 ms for 179 KB; `add_many` 96 ms for 100 KB; `tf_vocab_stats` 57 ms for 6 KB. Upload time is a round trip plus bytes over the 20 Mb/s uplink (50 ms + 100 KB at 2.5 MB/s = 90 ms; 6 KB gives 57 ms). The reply is half a round trip plus the pushed inputs over the 100 Mb/s downlink; the measured 66 to 86 ms is above that sum (25 + 14 to 24 ms), and the gap is not explained.

## Test: results of up to N KiB posted inside the result request

Removing the relay upload for small results saves one round trip per hop, which the formula priced at 17 x 50 ms = 0.85 s (19%). A worker flag posted results of at most N KiB as a float64 frame in the result request (rounded to the wire dtype first, so values equal what the relay would have returned; the coordinator's sampled check took the tolerance from the task's wire). Losses were identical to the last bit with everything inline. Seconds per update, median over updates 2 to 10, one run per row unless noted:

| Inline up to | Seconds per update |
|---|---|
| off (baseline) | 4.06, 3.96, 3.98 |
| 16 KiB | 3.90 |
| 32 KiB | 3.88 |
| 64 KiB | 3.78, 3.87, 3.83, 4.08 |
| 128 KiB | 6.05 |
| 256 KiB | 7.06 |
| everything (1 MiB) | 7.11, 6.74 |

Up to 64 KiB it is 3 to 5% faster, inside the spread between repeats. Above that the large tiles go through one post connection to the coordinator instead of two upload connections to the relay, and the update is 50 to 75% slower. One of five runs at 64 KiB hung before the first update; four later runs did not. The change was reverted and is not in the repository.

## Limits

- One machine, the tiny model, one captured run for the trace; the 30.3M model has the same chain with about 25 times the bytes per tile, so its share of bytes will be larger, not smaller.
- Capture adds overhead, so the chain total (4.8 s) is above the uncaptured 4.0 to 4.4 s.
- The proxy has no jitter or loss; the workers share the workstation with other work, so differences under about 0.15 s are noise.
- Why the claim reply takes 15 to 40 ms more than the link predicts is not explained (proxy buffering or TCP start are candidates).
- The 3-round-trips figure printed by `http_profile` is a formula and should not be read as a measurement.

## Reproduce

Needs PostgreSQL database `ygaraty_tiny`, the `1m-seed2026-v2k` dataset, and a built `relay-server`.

```bash
./relay-server -addr 127.0.0.1:PORT -token TOKEN -max-mib 4096 &
YGARATY_CAPTURE_DIR=runs/profile51/cap ygaraty experiment http_profile --recipe transformer --learning-rate 0.0005 \
  --workers-kind python --tiles 512 --max-tasks 16 --repeats 1 --relay-url http://127.0.0.1:PORT --relay-token TOKEN \
  --version 1m-seed2026-v2k --embedding 128 --hidden 384 --layers 2 --heads 4 --band 16 --max-task-mib 512 --workers 8 \
  --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 4 --pushed-inputs --eager-claims --length-window 16 \
  --batch-tokens 3072 --dataset-url sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite \
  --store-url postgresql:///ygaraty_tiny --steps 10 --rtt-ms 50 --down-mbps 100 --up-mbps 20 --output runs/profile51/home
ygaraty experiment hop_trace runs/profile51/cap
```

Drop `YGARATY_CAPTURE_DIR` and the shaping flags for the loopback row. Capture needs the `TimedLock.acquire`/`release` methods this experiment added to `http_profile.py`.

## Next

The bytes on a hop are the lever: about 100 KB up and 130 to 300 KB down per hop. Halving them saves roughly 0.5 to 0.7 s of 4.8 s. Candidates: bf16 for more tensors, smaller pushed inputs (only the rows a task reads), and explaining the reply gap.
