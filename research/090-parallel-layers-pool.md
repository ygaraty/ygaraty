# Parallel layers 1–2 shorten the 30.3M update on the pool by 5.8%, short of 10%

| | |
|---|---|
| **Conclusion** | **Failed gate:** parallel layers 1–2 shorten the 30.3M update 5.8% (4.71 → 4.43 s), 10% required (measured, DigitalOcean pool, 3 runs each). |
| **Status** | executed |
| **ID** | DO-26 |
| **Date** | 2026-11-05 |
| **Path** | Experiment 61's pool in DigitalOcean sfo2 (its sizes are no longer offered in nyc3): g-8vcpu-32gb coordinator and relay, 8 c-2 weak workers (gzip frames) and 8 c-8 strong ones (4 threads, 100/500 Mb/s), emulated 100/20 Mb/s and 50 ms links, handoffs on, capacity placement, the 30.3M translator (3+3 layers, width 512). Arms: D dense (experiment 61's arm B) and P `YGARATY_PARALLEL_LAYERS=enc:1-2,dec:1-2`, interleaved 0D 0P 1P 1D 2D 2P, 22 updates each |
| **Result** | **Correctness passes:** the graph tests against PyTorch pass at df0fe30; losses bit-identical across each arm's three jobs; 0 retries and 0 rejected in all six. **Time fails:** steady mean (updates 2–21) D 4.697, 4.721, 4.701 s; P 4.372, 4.432, 4.492 s: **−5.8%** (4.707 → 4.432 s) against the 10% gate. Every P run beats every D run (slowest P 4.49 s, fastest D 4.70 s). Planned chain 16 hops against 20; all six layers on the 8-core workers in every run. The 4 removed hops save 0.28 s, about 70 ms each, not L5's 230 ms. Cost about $2.20 of the shared pool |
| **Evidence** | Measured: [evidence/training/do26-parallel-layers-pool-2026-11-05.json](evidence/training/do26-parallel-layers-pool-2026-11-05.json) (per-update seconds and losses of every job, owner maps, gate); raw logs in `runs/pods-data/paper/do26/` (not in git). Code df0fe30; driver `runs/pods-data/paper/drive_paper.py`, analysis `analyze_paper.py` (hashes in the evidence) |

## Question

Experiment 76 showed in the reference that running layers 2–3 of each side in parallel costs only 0.025 of held-out loss, while removing 4 of 12 hops. Experiment 78 built parallel layers into the distributed graph:
- **Learning:** at 30.3M, parallel layers 1–2 learned *better* than dense (held-out 5.593 against 5.717).
- **Timing:** on the workstation's tiny model the update was 9.5% shorter, just under the 10% gate. The chain fell from 19 to 15 hops, but the merged layer waits for both parallel layers, which raised the median hop from 192 to 232 ms.

On the 30.3M model and the mixed DigitalOcean pool, a hop is a larger share of a longer update (4.64 s, experiment 61), and the pool has more workers to hold the parallel layers side by side.

**On experiment 61's pool, do parallel layers 1–2 on both sides shorten the 30.3M update by at least 10%, with losses matching the parallel model's PyTorch reference?**

## Prediction and Gate (as frozen)

Prediction:

- **Time per update:** 12–18% shorter (4.64 → about 3.9–4.1 s). The 4 removed hops are worth about 0.9 s at L5's 230 ms, minus the fan-in wait.
- **The fan-in wait** is larger when the two parallel layers sit on workers of different speed. Capacity placement should put both on strong workers; check the placement in the chain trace.
- **Time to loss:** about 0.7× dense, combining this with experiment 78's lower loss at equal pairs (an estimate, not measured here).

Gate:

Frozen 2026-11-05 with the user's yes, before any pool run.
- **Primary:** mean seconds per update of the parallel arm P (updates 2–21 of 22) at least **10%** below dense D, over 3 interleaved runs each (0D 0P 1P 1D 2D 2P), as in experiment 61.
- **Correctness (required):** `tests.test_parallel_layers_graph` and `tests.test_layer_variants` pass at the run's commit (the graph against the PyTorch variant: forward, loss, every gradient, three owned Adam updates); P's 22 losses bit-identical across its three jobs, and D's across its three; 0 retries and 0 rejected results. The earlier proposal, losses within 5×10⁻⁴ of a PyTorch replay of the run, was replaced: no tool replays a pool job's batches into PyTorch at 30.3M.
- **Report also:** hops on the planned critical chain (16 against 20); which workers hold the parallel layers (the owner map); the held-out loss is not scored here. The fan-in wait is not measured: the pool's hosts share no clock for a cross-host capture.

## Protocol

- **Pool:** `pods up paper` in sfo2 with the watchdog at 300 minutes, then experiment 61's 16 workers (`pods up` twice), setup with `remote.sh bootstrap` and `worker-setup`; the pool was then kept for M-2 stage 3.
- **Rehearsal:** the small model cannot take parallel layers 1–2 (2 layers a side, so the group would hold each stack's last layer), so P and D were rehearsed on the full model for 3 updates: both complete, 0 retries, layers on w8–w15.
- **Runs:** `drive_paper.py OUT 22 0D,0P,1P,1D,2D,2P`, each a fresh coordinator (`remote.sh serve34`) with experiment 61's settings; the coordinator's env adds the arm's settings.
- **Analysis:** `analyze_paper.py`: steady mean of updates 2–21 per job, mean per arm, bit-identity within each arm. It reproduces experiment 61's −22.5% from that experiment's data.
- **Not measured:** the fan-in wait at the merged layer. A capture needs every process on one clock, and the pool's hosts share none, so the report gives the planned hops and the owner map instead.

## Results

| Job | Arm | Mean s/update (2–21) | SD | Retries | Rejected |
|---|---|---|---|---|---|
| 0D | D | 4.697 | 0.212 | 0 | 0 |
| 0P | P | 4.372 | 0.249 | 0 | 0 |
| 1P | P | 4.432 | 0.358 | 0 | 0 |
| 1D | D | 4.721 | 0.344 | 0 | 0 |
| 2D | D | 4.701 | 0.239 | 0 | 0 |
| 2P | P | 4.492 | 0.250 | 0 | 0 |

**Gate:** D 4.707 s, P 4.432 s, **−5.8%** (required −10%): **fail**. Correctness: losses bit-identical within D and within P, the graph tests pass, 0 retries and 0 rejected: pass.

**Placement:** capacity placement put all six layers on 8-core workers in every job (for example 0P: decoder layers on w8, w9, w10; encoder on w11, w12, w14).

**Against experiment 61:** this pool's dense arm ran at 4.71 s per update, against experiment 61's 4.64 s in nyc3.

## What it taught

- **Four fewer hops are worth 0.28 s here, about 70 ms each.** L5's 230 ms per hop assumed each removed hop came wholly off the critical chain. On the workstation (experiment 78) the merged layer's wait for both parallel layers raised the median hop from 192 to 232 ms; the same fan-in wait is the likely cause here, but without a capture the pool cannot show it.
- **Scale did not help.** The full model on the mixed pool saves a smaller share (5.8%) than the tiny model on the workstation (9.5%), though its updates are longer: the hops removed are a smaller part of a 4.7 s update.
- **Learning is untested here.** Experiment 78's better held-out loss at equal pairs (5.59 against 5.72, one seed) would still make parallel layers worth having for time to loss. That combination is an estimate, not a measurement.
- **The small rehearsal model can't run parallel layers 1–2.** It has 2 layers a side; full-model rehearsals of 3 updates did the job.

## Limits

- One pool, one region (sfo2), three runs per arm of 22 updates; held-out loss not scored.
- The fan-in wait and the chain's hops were not measured on the pool, only planned.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_parallel_layers_graph tests.test_layer_variants
P=".venv/bin/python -m ygaraty.experiments.pods"
$P up paper --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 120 --max-hourly 2
$P up paper --workers 16 --worker-size c-8 --max-hourly 2
$P ship paper && bash runs/pods-data/paper/setup.sh
.venv/bin/python runs/pods-data/paper/drive_paper.py runs/pods-data/paper/do26 22 0D,0P,1P,1D,2D,2P
.venv/bin/python runs/pods-data/paper/analyze_paper.py runs/pods-data/paper/do26
$P down paper && $P ls
```

## Log

- 2026-11-05: Added as the full-scale follow-up of experiments 76 and 78.
- 2026-11-05: Prepared. `tests.test_parallel_layers_graph` and `tests.test_layer_variants` pass at the working tree. Driver `runs/pods-data/paper/drive_paper.py` (arms D and P, handoffs on, tags 0D,0P,1P,1D,2D,2P) and `analyze_paper.py` (steady mean of updates 2–21, bit-identity within arms); the analysis reproduces experiment 61's −22.5% from its data. No captures on the pool: its hosts share no clock, so hops come from the plan (16 parallel against 20 dense, `test_owned_plan_is_four_hops_shorter`) and the owner map from the coordinator log. DigitalOcean offers the pool's sizes only in sfo2 now, not nyc3 ($2.875/h); D is rerun interleaved, so the comparison stays within one pool.
- 2026-11-05: Budget approved by the user: one pool for DO-26 then M-2 stage 3, **$15**, deleted by the watchdog at **300 minutes** (at most $14.40 at $2.875/h); expected about $12. DigitalOcean sfo2, experiment 61's sizes.
- 2026-11-05 21:13: Started at df0fe30. Pool `paper`: DigitalOcean sfo2, g-8vcpu-32gb coordinator, 8 c-2 (w0–w7) and 8 c-8 (w8–w15), watchdog deletes at 300 minutes. Driver `runs/pods-data/paper/drive_paper.py runs/pods-data/paper/do26 22 0D,0P,1P,1D,2D,2P`.
- 2026-11-05 21:38: Rehearsal passed: the small model cannot take parallel layers 1–2 (2 layers a side, so the group would hold the last layer), so P was rehearsed on the full model for 3 updates, with D after it: both completed, 0 retries, 0 rejected, layers on w8–w15. Update 2: P 6.21 s, D 4.81 s (one sample). Main matrix started.
- 2026-11-05 21:59: Main matrix finished, all 6 jobs complete (22 updates, 0 retries, 0 rejected). Steady means D 4.697, 4.721, 4.701 s; P 4.372, 4.432, 4.492 s: **P −5.8%** (4.707 → 4.432 s), gate −10%: **fail**. Losses bit-identical within each arm. Pool kept for M-2 stage 3.
