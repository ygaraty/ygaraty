# The experiment-22 recipe, trained by Ygaraty

| | |
|---|---|
| **Date** | 2026-10-26 (200 updates, in two runs joined by a checkpoint resume) |
| **Path** | The HTTP path in production mode: webapp server, `DatabaseTranslationJob`, PostgreSQL store. NumPy workers (`ygaraty http-worker`) first, then Node SDK workers, then browsers. |
| **Result** | 200 production updates on OPUS-100 with 4 NumPy workers at a median 13.6–14.5 s each, loss 9.75 → 6.3 (mean 6.23 over updates 101–200, against experiment 22's 6.30 over the same updates in PyTorch). Six full-size updates match float64 PyTorch tensor for tensor (1,002 checks, ≤3.4e-13). Runs are reproducible bit for bit, and a run interrupted twice by memory pressure from outside it was completed by resuming from its checkpoint. |
| **Evidence** | — |

## Question

Does Ygaraty reproduce experiment 22's learning curve on real data, and how long does it take per update? That turns experiment 22's compute budget into a measured time to a useful translator.

This runs on the HTTP path because that is the path Ygaraty ships. The ZeroMQ path stays as a reference for experiments 1–23, but its timings don't carry over: on the same workload, experiment 26 found HTTP 2–4× slower with the same NumPy workers, for different reasons.

## Prerequisites, done on 2026-10-25

- **Subword tokenizer.** A SQL dataset version can carry a SentencePiece model (`ygaraty dataset import --tokenizer`, or the `tokenizer` command). Training uses its subword IDs; pairs longer than `max_tokens` are skipped as in experiment 22. Checkpoints and snapshots record the tokenizer, and snapshots include `tokenizer.model`. Experiment 22's 908,831 pairs and its 16,000-piece tokenizer import in about 10 s.
- **Model sizes.** `YGARATY_LSTM_BATCH_SIZE` goes up to 256. At embedding 256 and hidden 512, tile 512 needs `YGARATY_MAX_TASK_MIB=64`.
- **Fewer, larger tasks.** The LSTM is lowered as one batched, time-major graph: each layer direction is one `lstm` node, planned as one fused task per time step (plus tiled recurrent products); attention and the vocabulary cross-entropy are tiled by rows; views cost no task. A batch-64 update is 10,425 tasks instead of about 1.6 million. Workers claim up to 64 tasks per request and return results together (`max_tasks`; `--max-tasks` on `ygaraty http-worker`). Parity against PyTorch passes on every executor (48/48 runs); the small LSTM parity cases run 2–3× faster over HTTP.

## First timing

One update of the full recipe (batch 64, tile 512, 4 `http-worker` processes claiming 8 tasks each, PostgreSQL store, SQLite dataset) on the Ryzen 9 PRO 6950H: **112 s**, loss 9.75 (about ln 16,000, as expected before training), no retries. The float64 arithmetic alone would take 1–2 s on four NumPy workers, so about 98% is overhead: moving the 460 MB of weights and Adam state through the store every update, the chain of about 360 dependent LSTM steps each costing round trips and transactions, and sampled validation. **Profiled** (experiment 26's profiler now runs this job: `--recipe lstm`): 120 s per update, kernels 3.3 s per worker. The server sent 25.5 GB and received 8.9 GB for one update; its thread time went to tensor reads (150 s), writes (81 s), releases (50 s), block hashing (44 s) and lease records (37 s). The plan re-read weights constantly: each of about 360 recurrent steps fetched the 8 MB `W_hh` tile again, and the HTTP path had no worker-side cache.

**Worker tile cache** (2026-10-26): workers keep received tiles under `(job, version, region)` keys and report additions and evictions; the server omits tiles a worker holds and skips their store reads. Per update: 25,037 hits, 9.6 GB not sent (25.5 → 15.9 GB), **106.5 s** instead of 120. The remaining traffic is activations and gradients, each written by one task and read by the next through PostgreSQL, plus about 85 s of lease and release transactions per update. The cache cannot help with those; keeping transient values out of the database can.

**Two-tier store and the step end** (2026-10-26). Values that live within one update (activations, gradients, partial products) now stay in server memory; only the new weights and Adam moments are copied to PostgreSQL at each step end, and checkpoints are exported from memory. The step end's export and copy run after the job lock is released, so workers keep claiming meanwhile; the new weights also stay in memory for the next update's reads. Single-block reads return views instead of copies. Two planner changes cut the task count: one task per LSTM backward step instead of two reading the same inputs, and matmuls whose inner dimension fits in two tiles run as one task instead of partial products plus a sum. Measured one at a time, same 4 NumPy workers:

| Change | Seconds per update | Job lock held |
|---|---:|---:|
| Worker tile cache (above) | 106.5 | 13% |
| Two-tier store | 35.9 | 45% |
| Step end off the lock, weights kept in memory, view reads, both fusions | 30.3 | 5% |

| Step end on its own thread (the row above still blocked result requests behind the export, through a lock) | 17.7 | 10% |
| Recurrent product inside the step task (one hop per step; chain 399 → 222 hops); matmuls fused over K within a byte budget | 13.75 | 8% |
| Regions of assembled values read from the part that holds them (LSTM outputs assembled per band of rows); matmul and placement blocks as wide as the task budget allows | **13.0** | 8% |

Losses were identical across all of these runs. Planning turned out to cost 0.09 s per update; the 11 s attributed to it was the checkpoint export.

**Where 17.7 s go** (the profiler's task timeline: `ygaraty experiment http_profile --recipe lstm`). Forward LSTM chains run from 0.8 to 4.4 s: encoder layer 1 in both directions (0.9 s), layer 2 (1.0 s), decoder layers 1 and 2 (1.3 and 0.9 s), at 17–24 ms per step with two worker hops each (the recurrent product, then the cell). Attention, the 16,000-wide output projection, softmax and cross-entropy and their backward take 4.4–9.6 s. Backward LSTM chains run 9.6–15.7 s at 15–52 ms per step, sharing the four workers with the weight-gradient matmuls. The optimizer and the tail take the last 2 s. Step tasks never wait for a worker (0–5 ms of queue time); bulk matmuls wait 460 ms on average, because the bulk phases are bound by bytes: 15.9 GB leave the server and 7.4 GB return per update through Python's HTTP at both ends, while each worker computes for 3.2 s. The dependency chain is 399 worker hops long (228 if the recurrent products were folded into the step tasks).

**After those changes** (13.0 s): forward chains run 0.2–2.8 s, backward chains 7.4–11.7 s, at 14–39 ms per step with one hop each; the output phase takes 2.8–7.4 s. Reading regions from the producing part was meant to let one layer's steps start behind the previous layer's, but the bidirectional encoder prevents most of it: layer 2's first rows need the reverse direction of layer 1 to finish its last step, and the decoder's second layer takes its initial state from the encoder's second layer. Per update the server now sends 12.8 GB: matmuls 4.0 GB, backward steps 1.4 GB, optimizer 1.1 GB, attention 1.0 GB, output softmax 0.9 GB, bias gradients 0.8 GB, partial sums 0.8 GB. What remains per update: workers wait for ready tasks along the recurrence chains (48 s of held-open claims summed over 4 workers), 12.8 GB still cross HTTP (activations between steps), the checkpoint export (11 s, now off the critical path but on a request thread, and 460 MB of disk per update), and the copy of new weights to PostgreSQL (3.3 s).

**Learning rate.** The SQL job hard-coded Adam's step at 0.01; at this size the first update took the loss from 9.75 to 82.9. It is now a job setting (`learning_rate`, default `YGARATY_LSTM_LEARNING_RATE`), used by the plan and the oracle alike. At experiment 22's 0.001 the first update goes 9.75 → 8.59.

## Verification at full size (2026-10-26)

`ygaraty experiment distributed_translation --mode verification` runs the same job in verification mode: PyTorch's initialization, and every logit, loss, gradient, weight and Adam moment of every update compared with the float64 oracle. Six updates at experiment 22's size on the OPUS-100 batches, 4 NumPy workers, 9,294 tasks per update:

| Update | Loss | Checks | Largest difference from PyTorch |
|---:|---:|---:|---:|
| 1 | 9.7477 | 167 | 3.4e-13 (weights) |
| 2 | 8.5603 | 167 | 3.4e-13 |
| 3 | 11.9961 | 167 | 3.4e-13 |
| 4 | 8.4582 | 167 | 3.4e-13 |
| 5 | 8.6442 | 167 | 3.4e-13 |
| 6 | 8.7567 | 167 | 3.4e-13 |

All 1,002 checks passed at rtol 2e-7, atol 2e-9; logits agree to 7e-14 and losses to 1e-14. The spike at update 3 is PyTorch's loss too for those weights and that batch (12.7 in the production run, which starts from the block-seeded initialization): Adam's first steps on a new batch, not a defect.

## First run (2026-10-26, stopped at 100 of 200 updates)

`ygaraty experiment 24` on the Ryzen 9 PRO 6950H, 4 `http-worker` processes, PostgreSQL, learning rate 0.001, checkpoint every 20 (`runs/experiments/024/omarchy-20261026-031516/`, not synced). The run was stopped by the tooling at update 100 when the machine ran low on memory: `/tmp` is a RAM-backed tmpfs here, and the day's profiler runs had left 9 GB of checkpoints in it. Nothing in the run itself failed: no retries, no rejected results.

| | |
|---|---|
| Seconds per update | median 13.6, mean 13.3 (8.2–18.1); 22 minutes for 100 updates |
| Loss, mean over updates 1–10 / 11–30 / 31–60 / 61–100 | 8.92 / 7.49 / 6.99 / 6.75 |
| Loss at updates 1, 10, 50, 100 | 9.75, 7.72, 7.38, 6.63 |

Experiment 22's PyTorch run reported a mean training loss of 7.58 over its first 100 updates (float32, its own shuffled and length-sorted batches); Ygaraty's mean over the same count is 7.19 on the dataset's first 6,500 pairs in order. Same scale and same trend, which is what the plan asks for; a closer comparison needs the same batches, which the verification mode provides tensor by tensor.

## Second run (2026-10-26, stopped at 119 of 200), and resuming

A second launch from scratch (`runs/experiments/024/omarchy-20261026-113752/`) reached update 119 in 29 minutes, median 14.4 s per update (8.0–26.2), loss 6.48 at update 119 (mean 6.51 over 101–119), again with no retries or rejected results. It was stopped the same way as the first, this time by a 20 GB process unrelated to the run. **Its 100 first losses equal the first run's bit for bit**: block-seeded initialization, the dataset in order and float64 kernels make a run reproducible, whichever workers execute which tasks in whatever order.

Rather than start a third run from zero, the job can now **resume from a checkpoint**: `DatabaseTranslationJob(..., resume=<checkpoint-NNNNNN.zip>)`, `ygaraty experiment 24 --resume <zip>`. The checkpoint already held everything needed: weights, Adam moments, the dataset cursor (`next_after_id`, epoch) and the number of completed updates. The resumed job imports the blocks into the store (hashes checked), continues as update `completed_updates+1` with Adam's bias correction at that step, and refuses a checkpoint whose model, training configuration or tokenizer differ. A test resumes a small job from its first checkpoint and gets the same losses, weights and moments as the uninterrupted job, bit for bit. The path is a constructor argument, not a job field, so HTTP clients cannot point the server at arbitrary files. The coordinator still does not resume itself after a crash (experiment 25's question); this is the manual version.

## Updates 81–200, resumed from checkpoint 80 (2026-10-26)

`ygaraty experiment 24 --resume .../checkpoint-000080.zip` on a recreated database (`runs/experiments/024/omarchy-20261026-122446/`). Its first losses equal the interrupted run's updates 81 and 82 to the last digit, so the resumed training is the interrupted training. 120 updates in 29 minutes; no retries, no rejected results. With the 17 minutes of updates 1–80, the 200 updates took 46 minutes of training time, as the first timing predicted.

| | |
|---|---|
| Seconds per update, updates 81–200 | median 14.5, mean 14.5 (7.6–23.3) |
| Loss, mean over updates 1–10 / 11–30 / 31–60 / 61–100 / 101–150 / 151–200 | 8.92 / 7.48 / 6.99 / 6.75 / 6.38 / 6.08 |
| Loss at updates 1, 50, 100, 150, 200 | 9.75, 7.38, 6.63, 6.34, 6.30 (lowest 5.59 at 190) |

Experiment 22's PyTorch run reports a token-weighted training loss of 7.58 over updates 1–100 and 6.30 over 101–200; Ygaraty's per-update means over the same ranges are 7.19 and 6.23, on the dataset's first 12,800 pairs in order instead of experiment 22's shuffled, length-sorted batches. Same curve, within what different batches account for. The success criterion is met for the first 200 updates; experiment 22 needed about 11,000 to reach its best chrF.

**The last 30 updates slowed to 15–23 s**, and the step-end thread fell behind: when update 200 completed, the weight copies to PostgreSQL of the last 90 updates were still queued, draining at one every 20 s while workers were claiming and in 3.5 minutes once they stopped; the final checkpoint waited for them. The database had bloated to 15 GB by then (21 GB after the queue drained), and the queued copies kept their versions' blocks in memory: the server reached 11 GB and the machine began to swap, which slowed everything else. That is the churn problem above seen from the other side: copying weights at checkpoint boundaries only would remove the bloat and the queue, and the 3 s copy per update with them.

**Disk.** The run directory keeps the last two checkpoints (443 MB each). The PostgreSQL database, which holds only the 460 MB of current weights and moments, grew to 10–14 GB in both runs: every update writes and deletes 460 MB, and vacuum reclaims the dead rows only after a lag, so the file reaches a plateau of about 25 updates' worth. A VACUUM loop beside the run did not help; a vacuum of the grown table takes minutes. Copying weights to PostgreSQL only at checkpoint boundaries (every 20 updates) would cut that churn 20× and save the 3.3 s copy per update; not done yet.

## Plan

1. Import the same OPUS-100 sample with `ygaraty dataset import --tokenizer`. Done here: 908,831 pairs with experiment 22's tokenizer.
2. `ygaraty experiment 24`: 200 updates in production mode with 4 `ygaraty http-worker` processes, a checkpoint every 20. It writes `updates.jsonl` (loss, seconds, example ids per update) and `report.json`. Compare the loss curve with experiment 22's first updates; Ygaraty runs float64 and experiment 22 float32, and the batches differ (experiment 22 shuffles and sorts by length), so compare the trend, not the values.
3. Seconds per update come from the same run; where they go, from `ygaraty experiment http_profile --recipe lstm` (experiment 26's profiler).
4. Repeat with Node SDK workers (`--workers-kind node`), then with browsers on a local network, to see what JavaScript kernels and a real network add.

## Success criteria

Loss trajectory within run-to-run noise of experiment 22 over the same updates. A measured seconds-per-update gives a projected time to reach experiment 22's "useful" level, reported together with the hardware used.

## Hardware

A GPU doesn't help this experiment yet.

- **The coordinator is the limit, not arithmetic.** With NumPy workers, kernels are under 7% of an HTTP update (experiment 26); faster kernels would not shorten it.
- **Ygaraty's parity is float64.** Consumer GPUs run float64 at 1/32 to 1/64 of their float32 speed. Datacenter GPUs (A100, H100) are fast in float64 but cost the most.
- **Browser GPUs go through WebGPU, which is float32.** Experiments 14–15 measured what that does to Adam updates; the compensated arithmetic of experiment 15 is the route for browser GPUs.

A GPU does help elsewhere:

- **The PyTorch reference** (experiment 22 with more data, `--train-size 1000000 --epochs 10`): about 17 hours on this Ryzen's CPU, likely 1–2 hours on a recent GPU (an estimate, not measured).
- **Later, WebGPU contributors**, once the coordinator keeps up with fast workers. That is the step where browser GPUs pay off.
