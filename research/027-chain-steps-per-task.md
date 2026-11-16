# Several LSTM steps per task: does cutting the chain's hops shorten an update?

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | HTTP path in production mode (experiment 24's job), PostgreSQL store, 4 NumPy `http-worker` processes claiming 8 tasks each, tile 512, task budget 128 MiB; experiment 26's profiler |
| **Result** | No. With 4 consecutive steps per task, each recurrence chain went from 43–56 worker hops to 11–14, and an update took 12.53 s instead of 12.85 s (2.5% less), with identical losses. The chains were never the limit on one machine: the update is bound by bytes through the coordinator (12.3–12.8 GB sent per update; 10 s of tensor reads and 6.7 s of socket writes of server thread time). Fewer hops should pay where a hop costs a real network round trip; that is not measured here. |
| **Evidence** | `runs/experiments/027/omarchy-20261026-121552/` (`chain{1,2,4}/profile.json` and logs), not synced |

## Question

The [research map](../../docs/research/areas.md) ranked the LSTM recurrence chain first among the bottlenecks: 222 sequential worker hops per update, each a claim, a transfer, a kernel and a result, with workers idling in between. The cheapest way to test that ranking is to run several consecutive steps in one task and see how much of the update the saved hops buy.

## Change

`split(tape, tile, fuse_bytes, chain_steps=k)` lowers each LSTM layer direction into tasks of up to `k` consecutive steps (`lstm_chain` forward, `lstm_chain_grad` backward), in processing order. A chunk reads its state from the row of the last step the previous chunk processed; the backward chunk also reads the forward row before its last processed step, for that step's `c_prev`. The arithmetic is the per-step arithmetic in the same order, so the numbers are identical (`tests/test_lstm_translation.py::ChainStepTests` checks every tape value and gradient for chain steps 1, 2, 3 and 8 against the untiled evaluation). `k` is capped by the task's output (8 MiB, `MAX_OUTPUT_BYTES`) and the task budget: at hidden 512 and batch 64 that is 4 forward steps (7.3 MB of output) and 5 backward steps. The job reads it from `YGARATY_LSTM_CHAIN_STEPS` (default 1, today's plan); the profiler sweeps `--chain-steps`. Only the NumPy worker has the two kernels; the browser SDK would need them before a browser pool could use this.

## Setup

One run per setting, 3 updates each (the first a warm-up), on the Ryzen 9 PRO 6950H, nothing else running. The batch is the dataset's first (43 source steps, 56 target steps). Losses 9.7492, 8.5862, 12.7353 in all three runs.

## Result

| Chain steps per task | Tasks per update | Tasks per chain (encoder, decoder) | Seconds per update | Forward chains end | Backward chains run | ms per chain task (forward) | MB sent |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 4,358 | 43, 56 | 12.85 | 2.8 s | 7.6–11.8 s | 13–20 | 12,793 |
| 2 | 4,054 | 22, 28 | 12.90 | 2.7 s | 7.5–11.6 s | 24–38 | 12,679 |
| 4 | 3,908 | 11, 14 | 12.53 | 2.6 s | 7.2–11.0 s | 45–62 | 12,320 |

Wall time by task family (each moment shared by the tasks in flight), chain 1 → chain 4:

| Family | Wall s | Compute s (4 workers) | Mean queue wait |
|---|---:|---:|---:|
| Matmuls (projections, weight gradients) | 3.06 → 3.13 | 5.8 → 5.9 | 212 → 227 ms |
| LSTM forward steps | 1.85 → 1.60 | 2.2 → 2.2 | 1 ms |
| LSTM backward steps | 1.48 → 1.21 | 1.5 → 1.9 | 3 → 7 ms |
| Output softmax and loss | 1.27 → 1.25 | 0.9 | 597 → 583 ms |
| Bias gradients, elementwise, optimizer, attention, sums, embedding gradient | 4.6 → 4.7 | 0.9 | 18–266 ms |

Server thread time per update, chain 1 → chain 4: tensor reads 10.15 → 9.75 s, socket writes 6.74 → 6.71 s, claims held open 6.77 → 6.00 s, tensor writes 4.34 → 4.28 s, HTTP parsing 1.9 s. Job lock held 7–8%. Worker kernel time 2.85 → 2.99 s per worker per update.

## Reading

- **A hop is cheap here.** A one-step task takes 13–20 ms end to end and a four-step task 45–62 ms, so each step costs 11–15 ms of bytes and arithmetic and the saved round trip is 5–8 ms. Over about 200 hops that is at most 1.6 s of chain-path time, and the chains overlap with bulk phases that keep the server busy anyway.
- **Bytes bound the update, not the chain.** The bulk families (matmuls, output softmax, bias gradients) wait 200–600 ms in the queue per task because the server is moving 12.5 GB per update through Python's HTTP; chunking the chains changed none of that. The step tasks themselves never wait (1–7 ms).
- **The ranking in the research map was wrong for this setup.** On one machine with 4 workers, data movement through the coordinator comes first; chain latency comes into play only when a hop includes a real network round trip. At 50 ms per round trip, 200 hops would cost 10 s and 50 hops 2.5 s per update, which is why the change is kept behind a setting rather than dropped. That claim needs an experiment with injected latency (research map, area I).

## Limits

One run per setting on one batch; the 0.3 s difference between chain 1 and chain 4 is within what two runs of the same setting differ by (experiment 24's updates ranged 8.2–18.1 s). No real network, no browsers. Steps per task are capped by the 8 MiB result size; a whole chain in one task would need a different result protocol.

## Running it

```bash
ygaraty experiment 27
```

Needs the experiment-24 setup (`YGARATY_DATASET_URL`, `YGARATY_STORE_URL`), about 6 minutes on a Ryzen 9 PRO 6950H.
