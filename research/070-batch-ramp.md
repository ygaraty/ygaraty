# A batch ramp saves about 5% of wall time, not 25–45%: no bigger batch reaches the 3,072-token loss by 131,072 pairs

| | |
|---|---|
| **Conclusion** | **Rejected:** no bigger-batch arm reaches 3,072 tokens' loss; the ramp saves about 5% (measured learning, simulated timing). |
| **Status** | executed |
| **ID** | DO-10 → experiment 70 |
| **Category** | training |
| **Where** | workstation and simulator; stage 3 on DigitalOcean was not run |
| **Date** | Added 2026-11-01; run 2026-11-02 and 2026-11-03 |
| **Path** | Local stages only, by the user's instruction. Stage 1 on the Ryzen 9 PRO 6950H workstation: learning curves in the PyTorch reference of `batch_learning` (30.3M translator, 131,072 pairs, seeds 2026 and 2027). Stage 2: seconds per update from the simulator on experiment 61's pool, anchored on its measured 4.64 s. Arms: fixed 3,072, 6,144 and 12,288 tokens, and a ramp with and without square-root rate scaling |
| **Result** | **No-go by the frozen local rule.** F6, F12, R and R0 all miss F3's 131,072-pair loss on both seeds. Held-out at 131,072 pairs, two-seed mean: F3 4.427, R 4.589, R0 4.677, F6 4.735, F12 5.179. Without the rule's limit on pairs, R reaches its own final loss in 95% of F3's simulated time (90%, 100%). Free, about 3.5 hours; the $3 stage 3 was not started |
| **Evidence** | [`public/research/evidence/training/do10-batch-ramp-2026-11-03.json`](evidence/training/do10-batch-ramp-2026-11-03.json) (curves, chrF, simulated batches, rule and post hoc numbers, hashes); raw runs, `stage2.py` and `analyze.py` in `runs/experiments/DO-10/omarchy-20261102-220938/` (local, not in git); `tests/test_batch_schedule.py`. Code at faf4355 plus this experiment's changes |

## Question

An update on home links costs about 4.6 s and is set mostly by its chain of hops (paper, Section 6), not by its work. A bigger batch should therefore cost less than proportionally more time per update. But experiment 35 found batch 128 learning less per pair than batch 64 **early** in training.

The critical-batch-size literature (McCandlish et al. 2018) says the useful batch grows as the loss falls, which is why large training runs ramp the batch.

**Does a batch that ramps from 3,072 to 12,288 tokens reach a fixed held-out loss in less wall time than a fixed 3,072-token batch, with time per update measured on the pool?**

## Prediction and gate, as frozen

**Plan's lever gate:** R (or the best arm) reaches F3's 131,072-pair loss in at least 20% less wall time, averaged over both seeds, with the stage-3 timings outside their repeats' spreads. Only stage 3 could decide this, and it wasn't run.

**Local go/no-go for stage 3,** frozen 2026-11-02 before the simulator ran and before any learning run:

- **Wall time per arm and seed:** pairs to that seed's F3 loss at 131,072 pairs (linear between evaluations), converted to updates at each batch size (linear between evaluations), times that size's seconds per update. An arm that doesn't reach the loss by 131,072 pairs fails on that seed.
- **Seconds per update:** experiment 61's measured 4.64 s at 3,072 tokens, times stage 2's simulated ratio to 3,072 tokens at each size.
- **Best arm:** the lowest two-seed mean wall time among F6, F12, R and R0.
- **Go:** the best arm's two-seed mean is at most 80% of F3's. DO-10 stays open and stage 3 waits for the user's approval and budget. **No-go:** finish as Rejected, labeled measured learning with simulated timing, with no money spent.
- **Sensitivity, reported and not part of the rule:** the same wall times with 12,288 tokens at 1.6× and 2.2×, because L5 has not earned trust on handoffs.

Predictions, against the results:

| Prediction | Result |
|---|---|
| 12,288-token updates take 1.6–2.2× as long as 3,072-token ones | **Wrong on the simulator, not measured:** 2.68× |
| F12 is about 0.2 worse than F3 at 32,768 pairs | **Wrong:** 0.77 worse |
| By 131,072 pairs that gap is under half | **Wrong:** still 0.75, so it barely closes |
| R reaches F3's final loss in 25–45% less wall time | **Wrong:** R doesn't reach it within 131,072 pairs on either seed |
| R is the best of the five arms in wall time | **Right among the bigger batches, on the post hoc measure:** about 5% ahead of F3 at its own final loss, with no gain on seed 2027 |

## Protocol

**Stage 1.** Experiment 41's settings: width 512, 3+3 layers, 8 heads, batch 64, length window 64, Adam 5e-4, 131,072 pairs (14% of an epoch), held-out every 8,192 pairs on 1,024 pairs, with `--hint-index runs/experiments/055-hints/index --eval-dedup --decode`. Runs were strictly sequential under the watchdog, for each seed in the order F3, F6, F12, R, R0.

| Arm | Tokens per batch |
|---|---|
| F3 | 3,072 fixed |
| F6 | 6,144 fixed |
| F12 | 12,288 fixed |
| R | 3,072 → 6,144 at 32,768 pairs → 12,288 at 65,536 pairs, square-root rate scaling |
| R0 | as R, no rate scaling |

The schedule steps fall on window boundaries (64 × 64 = 4,096 pairs), and each window is cut with one budget. Rates change at the first update after a step. F3 had to reproduce experiment 67's Adam runs at 32,768 pairs exactly, or the stage would be rerun.

**Stage 2.** Experiment 42's 8-strong `never` case with `HandoffSimulation`, L5's constants and the bundled cost model; median of batches 1, 5, 9, 13 and 17 at each size.

**Stage 3,** the plan's interleaved DigitalOcean jobs at 6,144 and 12,288 tokens, was not run: the local rule said no-go, and the user had limited this to the local stages.

The local skill's default of the tiny model doesn't apply: this is one PyTorch process, like experiments 41 and 67, and the plan defines the arms on the 30.3M model.

## Log

- 2026-11-01: Added (first committed as a numbered plan file; renumbered to a track ID, since numbers mark finished experiments). Local DO-6 reassigned to DO-10 to avoid a different upstream experiment; the original pre-merge file is in the local merge backup.
- 2026-11-02: Local stages only, by the user's instruction: no remote machines. Built `--batch-tokens-schedule` and `--lr-scaling` in `batch_learning` and `tests/test_batch_schedule.py`. Output in `runs/experiments/DO-10/omarchy-20261102-220938/`. `scripts/l5_validate.py` is at `papers/uplink-bound/scripts/l5_validate.py`.
- 2026-11-02: Stage 2, simulated, before any learning run (`stage2.py`, `stage2.json`). `upload_hash_ms_per_mib` was dropped because `HandoffSimulation` now rejects nonzero upload-stage costs, so this isn't exactly L5's setting.
- 2026-11-02: Stage 1 started at faf4355+dirty (the schedule in `batch_learning.py`, its test, this file and the index), seeds 2026 and 2027, with `drive.sh`. Watchdog: `--stall 300 --busy-stall 1200 --max-runtime 3600 --progress 'pairs [0-9]+'`. A 12,288-pair rehearsal of the ramp with decoding passed: 34, 22 and 12 updates per 4,096-pair window, so the budget steps as planned.
- 2026-11-03: All 10 runs finished, exit 0, no failed or interrupted runs. F3 reproduced experiment 67 exactly. `analyze.py` applied the frozen rule: no-go. The post hoc comparison below was added after seeing that result. No remote machine was created.

## Results

### The frozen rule

| Condition | Required | Outcome |
|---|---|---|
| F3 reproduces experiment 67 at 32,768 pairs | 5.62590381, 5.80734676 | **Exact on both seeds** |
| Best arm's wall time against F3's (F3: 6,366 s on both seeds) | ≤ 80% | **No arm has one:** F6, F12, R and R0 all miss the loss on both seeds |

The sensitivity cases at 1.6× and 2.2× change nothing, since the arms fail on pairs and not on time.

### Learning

| Arm | Held-out at 32,768, mean | Held-out at 65,536, mean | At 131,072, seed 2026 | At 131,072, seed 2027 | Mean | Against F3 | chrF, mean | Updates |
|---|---|---|---|---|---|---|---|---|
| F3 | 5.717 | 5.118 | 4.352 | 4.501 | **4.427** | n/a | 16.73 | 1,372 |
| F6 | 6.101 | 5.431 | 4.769 | 4.701 | 4.735 | +0.309 | 12.86 | 720 |
| F12 | 6.485 | 5.922 | 5.201 | 5.157 | 5.179 | +0.753 | 8.14 | 382 |
| R | 5.717 | 5.192 | 4.444 | 4.733 | **4.589** | +0.162 | 14.05 | 705 |
| R0 | 5.717 | 5.212 | 4.573 | 4.782 | 4.677 | +0.251 | 13.81 | 705 |

### Seconds per update, simulated

| Tokens | Simulated | Ratio to 3,072 | Anchored on 4.64 s |
|---|---|---|---|
| 3,072 | 4.54 (experiment 61 measured 4.64) | 1 | 4.64 |
| 6,144 | 7.19 | 1.58× | 7.35 |
| 12,288 | 12.15 | 2.68× | 12.41 |

One simulated batch at each larger size came out far shorter (3.02 s and 5.16 s); the median leaves them out.

### Without the limit on pairs (post hoc, not in the rule)

The rule stops at 131,072 pairs, the pairs F3 needs by definition. An arm that learns less per pair therefore fails however fast it is, so it can't show a win in time. This comparison was added after the no-go to check that the verdict doesn't rest on that alone: each arm's final loss against F3 at the same wall time, and its time to its own final loss as a share of F3's.

| Arm | 12,288 at 2.68× (stage 2) | At 2.2× | At 1.6× |
|---|---|---|---|
| R | **95%** (90%, 100%), −0.05 loss | 87%, −0.13 | **78%**, −0.25 |
| R0 | 105%, +0.04 | 96%, −0.04 | 86%, −0.16 |
| F6 | 112%, +0.11 | 112%, +0.11 | 112%, +0.11 |
| F12 | 163%, +0.45 | 134%, +0.25 | 97%, −0.03 |

R's end is at 5,251 s on stage 2's timing, 82% of F3's 6,366 s, with F3 at 4.636 by then against R's 4.589. The two seeds disagree: −0.10 on 2026, 0.00 on 2027.

Only the 1.6× column would clear 20%, and it's generous: it keeps 6,144 tokens at 1.58×, so doubling the tokens again would cost almost nothing. The simulator says 2.68×.

## What it taught

- **The verdict holds with or without the cap.** The cap makes the frozen rule's failure partly automatic, but on stage 2's timing the uncapped ramp gains 5%, not 20%.
- **Bigger batches cost far more early than predicted.** F12 is 0.77 worse at 32,768 pairs, not 0.2. The gap hardly closes by 131,072 pairs, so the late-training catch-up the ramp relies on doesn't appear in this range.
- **Bigger batches do learn more per update:** at 382 updates F12 is at 5.179 and F3 at about 5.614. At 2.68× the time per update, that doesn't pay.
- **The ramp's first step costs little, the second is where it shows:** R is 0.074 behind F3 at 65,536 pairs and 0.162 by 131,072, while also doing half the updates.
- **Square-root scaling helps a little:** R beats R0 by 0.088 on the mean, 0.13 on seed 2026 and 0.05 on seed 2027, within this schedule. The rate rule wasn't tuned.
- **chrF follows the loss:** 16.73 for F3, 14.05 for R, 8.14 for F12.

## Limits

- **Timing is simulated, not measured.** L5 failed its gate on handoffs, and [experiment 71](071-handoff-timing-holdouts.md) found the corrected simulator still overstates the handoff gain with region reads, by 4.3 points on experiment 60. Stage 2 matches experiment 61 within 2.2% only at 3,072 tokens, and has no measurement at larger sizes.
- **The plan's lever gate wasn't tested,** and stage 3 could still move the numbers: it's the only way to learn the real ratio at 12,288 tokens.
- **Stage 2 isn't exactly L5's setting:** `upload_hash_ms_per_mib` was dropped for the handoff engine.
- **14% of an epoch.** A ramp may do better late in training, past this range. This doesn't test that, and it doesn't test later ramp points or other rates, which the plan called guesses.
- **Two seeds,** and the seeds disagree on R's gain.
- **The comparison without the cap was chosen after seeing the result,** so it's context and not a second gate. Its loss target is each arm's own final loss, not F3's.
- **Not checked:** owner memory and task budgets at 12,288 tokens, which the plan wanted before stage 3.
- `ruff` isn't installed in this environment, so the lint and format checks on the new code were not run.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_batch_schedule
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
OUT=runs/experiments/DO-10/$(hostname)-$(date +%Y%m%d-%H%M%S)
BL=".venv/bin/python -m ygaraty.experiments.batch_learning"
COMMON="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --pairs 131072 --eval-every 8192 --hint-index runs/experiments/055-hints/index --eval-dedup --decode --optimizer adam --learning-rate 0.0005"
RAMP="--batch-tokens 3072 --batch-tokens-schedule 0:3072,32768:6144,65536:12288"
for seed in 2026 2027; do
  for t in 3072 6144 12288; do $BL $COMMON --seed $seed --batch-tokens $t --output $OUT/F$t-$seed/result; done
  $BL $COMMON --seed $seed $RAMP --lr-scaling sqrt --output $OUT/R-$seed/result
  $BL $COMMON --seed $seed $RAMP --lr-scaling none --output $OUT/R0-$seed/result
done
```

Stage 2 is `stage2.py` and the gate is `analyze.py`, both in the raw run directory, which isn't in git. The stage 2 settings are in Protocol and the Log. The original runs used the watchdog limits in the Log. The hint index comes from experiment 55.
