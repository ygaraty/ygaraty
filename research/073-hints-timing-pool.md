# N-gram hints make the pool's updates 58% slower, which cancels their learning gain

| | |
|---|---|
| **Conclusion** | **Failed gate:** hints make each 30.3M update 1.58× slower, against the 1.15× allowed (measured, 3 runs per arm, DigitalOcean). |
| **Status** | executed |
| **ID** | DO-5a, the timing stage of [DO-5](DO-5.md) (experiment 73) |
| **Category** | training |
| **Where** | remote: DigitalOcean sfo2 |
| **Date** | 2026-11-03 |
| **Path** | Experiment 61's pool and setup: server g-8vcpu-32gb, 8 weak c-2 workers (gzip frames), 8 strong c-8 workers, handoffs, capacity placement. 30.3M Transformer, 64 pairs and 3,072 tokens per update, 22 updates. Arm A without hints, arm B with experiment 55's source-filtered hints, ABBAAB |
| **Result** | Mean steady seconds per update (updates 2–21): A 4.75, 4.87, 4.89 (mean 4.84); B 7.68, 7.67, 7.59 (mean 7.65), **1.58×**. The gate allowed 1.15×. With experiment 55's 63–68% of the pairs, hints would reach the plain arm's loss in 1.00–1.07× its wall time: no gain. All 6 jobs completed with 0 retries and 0 rejected, with losses identical within each arm. Arm A reproduces experiment 61 within 4.2%. The hinted plan has 915 tasks per update instead of 708. Cost about $1.90 (41 minutes of a $2.875/h pool) |
| **Evidence** | [`evidence/training/do5a-hints-timing-pool-2026-11-03.json`](evidence/training/do5a-hints-timing-pool-2026-11-03.json) (per-update seconds and losses of all six jobs); driver `drive_do5a.py`, `analyze_do5a.py`, `setup.sh`, rehearsal and per-job logs in `runs/pods-data/do5a/` (local, not in git). Code dc2f86f on every machine |

## Question

[Experiment 55](055-ngram-hints-translation.md) found that source-filtered n-gram hints reach the plain model's held-out loss with 63–68% of the pairs, at +6% compute per update on the PyTorch reference. The distributed model computes the same numbers, so its learning per pair should repeat 55's. What was unknown is the cost per update on the real pool:
- 4 more critical-path hops (20 → 24, from the shape-only plan);
- more tasks;
- the coordinator's per-update hint lookup.

**On experiment 61's pool, how much longer is an update with hints?**

## Prediction and gate (frozen in [DO-5](DO-5.md) before the pool existed)

The prediction was a few percent more seconds per update, for about 25–30% less time to the same loss.

- **Correctness:** every job completes all 22 updates with 0 retries and 0 rejected, and losses are identical across repeats within each arm.
- **Pool check:** the plain arm's mean steady seconds per update is within 10% of experiment 61's 4.64 s.
- **Decision:** if the hinted arm is at most 1.15× the plain arm, run the learning curves. If not, fold the hint adds into the layer and embedding kernels first.

## Protocol

- **Pool:** nyc3 no longer sells experiment 61's dedicated sizes, so the pool ran in sfo2 with the same sizes: server g-8vcpu-32gb, w0–w7 c-2 (gzip frames), w8–w15 c-8 (500/100 Mb/s, 4 threads). The pods watchdog had a 90-minute limit.
- **Setup:** `bootstrap` plus `hint-index` on the server, `worker-setup` on the workers, 7.5 minutes in all.
- **Rehearsal:** small model, 6 updates per arm.
- **Main matrix:** ABBAAB, 22 updates, with the coordinator settings of experiment 61 and `YGARATY_HINTS=none` or `source`.

## Log

- 2026-11-03: Before provisioning, a local run (DO-9's) found that the hint setting's `self.hints` overwrote `Job.hints()`. Every translation job had failed at its first cached claim since 5e6f1b2; fixed in cec07b1.
- 2026-11-03: First provisioning attempt refused: `POST /v2/vpcs` returned HTTP 403, because the token lacked write scope. Nothing was created. With a full-scope token, the pool `do5a` came up in sfo2 at about 16:53.
- 2026-11-03: Rehearsal passed: both arms complete with 0 retries, and their losses part after update 1. Main matrix 17:08–17:33 (watch.py `finished`). The pool was deleted at 17:34, leaving 0 machines.

## Results

| Job | Mean s per update (2–21) | sd | Retries | Rejected |
|---|---|---|---|---|
| 0-A | 4.749 | 0.33 | 0 | 0 |
| 0-B | 7.678 | 0.57 | 0 | 0 |
| 1-B | 7.670 | 0.51 | 0 | 0 |
| 1-A | 4.869 | 0.34 | 0 | 0 |
| 2-A | 4.892 | 0.35 | 0 | 0 |
| 2-B | 7.593 | 0.47 | 0 | 0 |

- **Correctness: passed.** All jobs completed 22 updates. Losses are identical across repeats within each arm, with a maximum difference of 0.
- **Pool check: passed.** A's mean is 4.84 s, +4.2% against experiment 61's 4.64 s, so sfo2 behaves like nyc3.
- **Decision: failed.** B/A is 1.581, against the 1.15 allowed; the arms' ranges are far apart (slowest A 4.89 s, fastest B 7.59 s).
- **In KPI terms:** useful compute falls from 49.7 to 30.8 GFLOP/s, so nanochat time rises from 25.5 to 41 years. Learning per pair isn't counted in that figure. Counting experiment 55's 63–68% of the pairs, time to loss is 0.63–0.68 × 1.58 = **1.00–1.07× the plain arm's**.

## What it taught

- **The hint path costs a lot more on the pool than in compute.** Experiment 55 measured +6% compute. Here the plan grows from 708 to 915 tasks per update (+29%), and the critical path grows from 20 to 24 hops in the shape-only plan. On a chain of hops bound by latency, more tasks and more hops cost more than their FLOPs, about 2.8 s per update.
- **Any change to the model's graph is paid in hops on this pool.** A lever on learning per pair has to be measured with its distributed cost, not its FLOPs; the same applies to Muon (M-2).
- **The fix is the one DO-5 named:** fold the hint adds into the layer and embedding kernels. The hint vector would then travel with the inputs those tasks already take, rather than as its own tasks and hops. If that brings the update to within 1.15×, the learning curves become worth running.

## Limits

- **No breakdown of where the 2.8 s go.** There were no captures on the pool, so there's no split between the extra hops, the extra tasks' claims and uploads, and the coordinator's lookup. A local captured run on the 1.06M model with hints would give one for free.
- **One region and one pool,** 3 runs per arm, 22 updates.
- **Learning per pair is not measured here.** The 1.00–1.07× uses experiment 55's reference-model result.

## Reproduce

```bash
ygaraty experiment pods up do5a --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 90 --max-hourly 1
ygaraty experiment pods up do5a --workers 16 --worker-size c-8 --max-hourly 2.1
ygaraty experiment pods ship do5a
bash runs/pods-data/do5a/setup.sh            # bootstrap + hint-index on the server, worker-setup on the workers
SMALL=1 python runs/pods-data/do5a/drive_do5a.py runs/pods-data/do5a/rehearsal 6 0A,0B
python runs/pods-data/do5a/drive_do5a.py runs/pods-data/do5a/main 22      # ABBAAB, about 25 minutes
python runs/pods-data/do5a/analyze_do5a.py runs/pods-data/do5a/main
ygaraty experiment pods down do5a
```

`drive_do5a.py` is experiment 61's `drive_r1.py` with handoffs on in both arms and `YGARATY_HINTS=none` or `source` on the coordinator.
