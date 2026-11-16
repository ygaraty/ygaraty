# On the pool, Muon reaches Adam's final held-out loss in 46% of Adam's training time

| | |
|---|---|
| **Conclusion** | **Works:** on the pool, Muon reaches Adam's 32,768-pair held-out loss in 46% of Adam's training time (measured, DigitalOcean, two seeds). |
| **Status** | executed (stage 1 as [experiment 74](074-owned-muon-step.md); stage 2 covered by its bound; stage 3 here) |
| **ID** | M-2 (stage 3) — Modeling Experiment |
| **Category** | training |
| **Where** | workstation (code), then DigitalOcean |
| **Date** | Added 2026-11-03; stage 3 run 2026-11-05 |
| **Path** | Experiment 61's pool in DigitalOcean sfo2 (shared with [experiment 90](090-parallel-layers-pool.md)): g-8vcpu-32gb coordinator and relay, 8 c-2 weak workers (gzip frames) and 8 c-8 strong ones (4 threads, 100/500 Mb/s), emulated 100/20 Mb/s and 50 ms links, handoffs on, capacity placement, the 30.3M translator. Arms: A, Adam everywhere; M, Muon 0.01 (momentum 0.95, 5 Newton–Schulz steps) on the six owned layers' matrices, Adam on norms, biases, embeddings and vocabulary (`YGARATY_OPTIMIZER=muon`). Seeds 2026 and 2027 (`YGARATY_INIT_SEED`), the same batches in every run, 512 updates of 64 pairs, held-out every 32 updates; order A26 M26 M27 A27 |
| **Result** | **Gate passes:** Muon reaches Adam's final held-out loss (5.292 and 5.304 after 512 updates) at update 230 and 229, after 1,235 and 1,236 s of training, against Adam's 2,692 and 2,707 s: **ratios 0.459 and 0.457**, limit 0.70 on both seeds. Muon's own final held-out loss is 4.762 and 4.742 (−0.53). An ordinary update takes 5.07 s against Adam's 4.92 s (+3.2%, experiment 74 bounded it at +6%); all four runs complete, 0 retries, 0 rejected, every layer on the 8-core workers. Not made the default (see What it taught). Cost about $9.60 of the shared pool's $11.80 |
| **Evidence** | Measured: [evidence/training/m2-muon-pool-2026-11-05.json](evidence/training/m2-muon-pool-2026-11-05.json) (per-update seconds and losses, held-out curves, owner maps, gate); raw logs in `runs/pods-data/paper/m2/` (not in git). Code df0fe30 plus `checkpoint_heldout.py` from c1e1fdd copied to the server (hash in the evidence); `tests/test_transformer_translation.py` `test_owned_muon_matches_the_reference_muon`; driver `runs/pods-data/paper/drive_paper.py`, analysis `analyze_paper.py` |

## Question

[Experiment 67](067-muon-owned-layers.md) found that Muon at 0.01 reaches Adam's 32,768-pair loss with 40–49% of the pairs. It failed only on CPU time per update: Newton–Schulz added 0.46 s to an update of about 0.59 s on the workstation, 1.77×. Experiments [68](068-cheaper-muon.md) and [69](069-muon-fewer-steps-higher-rate.md) found no cheaper variant that keeps the gain.

On the pool an update takes 4.64 s ([experiment 61](061-handoffs-30m-digitalocean.md)), set mostly by the chain of hops. Each owner steps only its own layer, and the owners step at the same time. So the cost that matters is **one layer's** Newton–Schulz, on one owner, in float64. That could be a small share of 4.64 s, or a large one, since a weak worker has few cores and owners compute in float64.

**On experiment 61's pool, does Muon reach Adam's held-out loss in clearly less wall time, once its owner step is counted?**

## Prediction

One layer's five matrices (512 by up to 2,048) take 0.1–0.5 s for 5 Newton–Schulz steps in float64 on 2 cores. That adds 2–11% to an update, so Muon reaches Adam's loss in about 45–55% of its wall time. The biggest risk: the step sits on the critical path, because the next forward pass needs the new weights.

## Gate

Frozen 2026-11-03, before stage 1's code.

- **Correctness:** one owned Muon step (`tf_owned_step` with `optimizer="muon"`) gives the same weights as the reference's Muon on that layer's matrices, with Adam on its norms and biases: relative error at most 1e-10 in float64, for an encoder layer and a decoder layer.
- **Stage 1 result:** the median time of one owned layer's whole step, Muon (5 Newton–Schulz steps, and 2) against Adam. It covers an encoder and a decoder layer of the 30.3M model (d 512, hidden 2,048), in float64 NumPy, with 1, 2 and 4 BLAS threads pinned to as many cores (`taskset`).
- **Go for stage 3:** the predicted Muon update is at most 1.3× Adam's 4.64 s (experiment 61). The prediction is 4.64 s + Δ, where Δ is the larger of the encoder and decoder layers' extra step time at 4 threads, as experiment 61's strong owners ran with `YGARATY_WORKER_THREADS=4`. That bounds Δ at 1.39 s. The bound treats the step as fully on the critical path: encoder layer 0's step follows the update's last backward task, and the next update's first forward waits for it. If Δ passes at 4 threads but not at 2, stage 3 needs layers placed only on 4-thread owners.
- **Lever (stage 3, measured):** Muon reaches Adam's held-out loss at a fixed pair count in at most 70% of Adam's wall time, on two seeds.

## Protocol

- **Stage 1 (workstation):** add Muon to `tf_owned_step` (momentum, Nesterov, the quintic Newton–Schulz of `ygaraty/reference/muon.py`, the same scaling). Embedding and vocabulary shards stay on Adam. Correctness: one owned Muon step equals the reference's Muon step on the same layer, within float64 tolerance. Then time one layer's step under `taskset` at the core counts of experiment 61's workers.
- **Stage 2 (offline):** add the measured step time to the owners' `tf_owned_step` in experiment 61's captured plan, with the simulator, or as the bound on the critical path if the step can't overlap.
- **Stage 3 (remote, as run):** experiment 61's pool and setup, Adam against Muon 0.01, order A26 M26 M27 A27, each a fresh coordinator (`remote.sh serve34`) with `YGARATY_HANDOFF=1 YGARATY_HELDOUT=1 YGARATY_CHECKPOINT_EVERY=32 YGARATY_INIT_SEED=<seed> YGARATY_OPTIMIZER=<adam|muon>`. Muon ran the full 512 updates too, so no target was needed in advance. `checkpoint_heldout watch` scored each checkpoint on the server (1,024 held-out pairs) while training went on.
- **Analysis (`analyze_paper.py`):** training time is the sum of per-update seconds (checkpoint updates included, held-out scoring not). Muon's time to Adam's final held-out loss is interpolated linearly between the two checkpoints that straddle it, and divided by Adam's total.
- Owners' memory and core counts (earlier TBD): Muon fit, and capacity placement put all six layers on the c-8 workers in every run.

## Log

- 2026-11-03: Added, after experiment 70. Raised by experiment 67's Limits: on home links, an update's cost could swamp Muon's CPU overhead.
- 2026-11-03: Stage 1 gate frozen (above). Started at 1c5d451, on the workstation (Ryzen 9 PRO 6950H), quiet (load 0.01).
- 2026-11-03: Stage 1 finished as [experiment 74](074-owned-muon-step.md). **Works:** at 4 threads the decoder layer's Muon step takes 351 ms against Adam's 68 ms. That bounds the update at 4.92 s, 1.06× Adam's (gate 1.3×), so stage 3 is a go. Stage 2 is covered by that bound, which treats the step as wholly on the critical path. Still to do for stage 3: emit `optimizer="muon"` from the planner (an owned-layer setting such as `YGARATY_OPTIMIZER=muon`) and keep layers on 4-thread owners. It also needs approval and a budget.
- 2026-11-05: Stage 3 prepared (uncommitted until approved). `YGARATY_OPTIMIZER=muon` (with `YGARATY_MUON_LR`, 0.01) makes the planner emit `optimizer="muon"` on every owned layer's `tf_owned_step`; embeddings, vocabulary shards, norms and biases stay on Adam; a production job without owned layers refuses it. `YGARATY_INIT_SEED` (2026) seeds the block-seeded initialization, for the second seed. `tests/test_transformer_translation.py` `test_owned_muon_matches_the_reference_muon`: two planned owned updates equal the PyTorch reference stepped by `reference/muon.py` plus Adam, to 1e-9 (it fails when the setting is off). Loopback smoke, tiny model, home links, 4 updates: Adam's losses are unchanged by the code, Muon's depart from update 2, seed 2027's from update 1, 0 retries (`runs/experiments/M-2/omarchy-20261105-201233-smoke/`). Placement: experiment 61's capacity placement already put all six layers on the 8-core workers (w8–w15), so no placement change is needed; the run checks the owner map. Driver and analysis: `runs/pods-data/paper/drive_paper.py` (arms A26, M26, M27, A27, each 512 updates with held-out every 32) and `analyze_paper.py` (training seconds to Adam's final held-out, linear between checkpoints; checked on synthetic data). Muon runs the full 512 updates, so the order A M M A needs no target in advance.
- 2026-11-05: Found while preparing: on unshaped loopback the tiny job stalls before its last update (also at 564aad9, before the pull); on emulated home links it finishes. Not on the pool's path (shaped links), but a pre-existing bug.
- 2026-11-05: DigitalOcean now offers g-8vcpu-32gb, c-2 and c-8 only in sfo2 and blr1 (API listing), not nyc3. The pool would be experiment 61's in sfo2: $0.375 + 8 × $0.0625 + 8 × $0.25 = $2.875/h.
- 2026-11-05: Budget approved by the user: one pool for DO-26 then M-2 stage 3, **$15**, deleted by the watchdog at **300 minutes** (at most $14.40 at $2.875/h); expected about $12. DigitalOcean sfo2, experiment 61's sizes.
- 2026-11-05: The loopback stall fixed. Cause: a coalescing poster sent more than 64 results in one post (an owner finishing the last update's many publish tasks at once); the coordinator refuses such a post with HTTP 400, the worker stopped, and its 97 leased tasks were never released. On shaped links results trickle out, so posts stayed small; checkpoint updates on the pool (M-2 publishes every 32 updates) were exposed too. Both posters now split at 64 (`MAX_RESULTS_PER_POST`), and the worker's error names the coordinator's reason. `tests/test_worker_post_limit.py`; 8 of 8 loopback runs finished with the same losses (before: 4 stalls in 18).
- 2026-11-05 21:13: Stage 3 started on pool `paper` (with DO-26, which runs first), at df0fe30.
- 2026-11-05 22:10: Rehearsal (4 updates, full model): A26 and M26 complete, 0 retries, 0 rejected; same update-1 loss (19.1933, same init), Muon's training loss then jumps (16.97 at update 2 against Adam's 9.70), as experiment 67's reference Muon starts higher (train 9.30 against 7.86 over its first 34 updates). Held-out scoring wrote nothing: the job now puts checkpoints in RUN/http-lstm-<job>/, where `checkpoint_heldout watch` did not look. Fixed in c1e1fdd (test added) and copied to the server by hand (sha256 b3a61674…0c1e); rescoring the rehearsal gave 16.73 at update 2 and 13.96 at update 4, about 20 s a checkpoint. Main runs started: A26, M26, M27, A27, 512 updates each, held-out every 32.
- 2026-11-06 01:18: All four runs complete (512 updates, 0 retries, 0 rejected). Gate: ratios 0.459 (seed 2026) and 0.457 (seed 2027), limit 0.70: **pass**. Pool deleted at 01:19 with `pods down paper`; `pods ls` shows 0 machines. Pool up 4 h 06 min at $2.875/h, about $11.80, of which DO-26 about $2.20.

## Results

| Run | Optimizer | Seed | Training s | Ordinary update s | Checkpoint update s | Held-out at 32 / 256 / 512 | Retries | Rejected |
|---|---|---|---|---|---|---|---|---|
| A26 | Adam | 2026 | 2,692 | 4.918 | 15.2 | 6.846 / 5.837 / 5.292 | 0 | 0 |
| M26 | Muon | 2026 | 2,761 | 5.055 | 15.3 | 6.769 / 5.223 / 4.762 | 0 | 0 |
| M27 | Muon | 2027 | 2,777 | 5.092 | 15.2 | 6.809 / 5.215 / 4.742 | 0 | 0 |
| A27 | Adam | 2027 | 2,707 | 4.915 | 16.2 | 6.843 / 5.876 / 5.304 | 0 | 0 |

"Ordinary update" is the mean of updates 2–512 that are not checkpoint updates (every 32nd, which publish the owners' weights).

**Gate:**

| Seed | Adam's final held-out | Adam's training s | Muon reaches it at | Muon's training s | Ratio |
|---|---|---|---|---|---|
| 2026 | 5.2919 | 2,691.8 | update 229.8 | 1,234.7 | **0.459** |
| 2027 | 5.3037 | 2,707.1 | update 228.6 | 1,236.3 | **0.457** |

Limit 0.70 on both seeds: **pass**.

**Cost per update:** Muon's ordinary update is 0.15 s (3.2%) longer than Adam's, half of experiment 74's 0.28 s bound, which assumed the whole step on the critical path.

**Pairs:** Muon needs 45% of Adam's pairs to reach Adam's final loss, within experiment 67's 40–49% from the PyTorch reference.

## What it taught

- **Muon's gain carries over to the pool almost whole.** Pairs to loss (0.45) times update cost (1.03) gives 0.46, as the run measured. On the pool, the chain of hops sets an update's time, so a heavier owner step barely shows. On the workstation's CPU the same Muon reached Adam's loss in 72–87% of its time (experiment 67).
- **The bound in experiment 74 was pessimistic by half.** Only part of the owner step sits on the critical path.
- **Not made the default.** The gate passes, but switching the optimizer changes the arithmetic of every training run. Earlier Adam results then stop being comparable, and Muon has been tested only at 32,768 pairs on this model. It stays a switch, `YGARATY_OPTIMIZER=muon`.
- **Bugs found while preparing:**
  - posters sent more than the coordinator's 64 results in one post, which stalled loopback jobs (fixed in df0fe30);
  - `checkpoint_heldout watch` missed checkpoints in the job's subfolder (fixed in c1e1fdd).

## Limits

- One pool and region (sfo2), one run per arm and seed, one learning rate (0.01), and 32,768 pairs. Seeds change the initialization only: every run trained on the same batches in the same order.
- Held-out loss is measured on 1,024 pairs, every 32 updates, and interpolated between checkpoints. Its values (Adam 5.29) are lower than experiment 67's reference (5.72), which used a different held-out set and measure. Compare within this experiment only.
- Training time excludes held-out scoring. The scorer ran on the coordinator's machine during training, in every arm alike.
- The order A M M A guards against a drift across the 3-hour window, but does not measure it.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_transformer_translation -k muon
P=".venv/bin/python -m ygaraty.experiments.pods"
$P up paper --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 300 --max-hourly 2
$P up paper --workers 16 --worker-size c-8 --max-hourly 2
$P ship paper && bash runs/pods-data/paper/setup.sh
.venv/bin/python runs/pods-data/paper/drive_paper.py runs/pods-data/paper/m2 512 A26,M26,M27,A27
.venv/bin/python runs/pods-data/paper/analyze_paper.py runs/pods-data/paper/m2
$P down paper && $P ls
```
