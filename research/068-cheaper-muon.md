# Cheaper Muon doesn't help: lower rates and fewer Newton–Schulz steps don't beat 80% of Adam's time

| | |
|---|---|
| **Conclusion** | **Failed gate:** cheaper Muon doesn't beat experiment 67's recipe; best 80% of Adam's time, limit 70% (measured, two seeds). |
| **Status** | executed |
| **ID** | WS-4 → experiment 68 |
| **Category** | training |
| **Where** | workstation |
| **Date** | Added and run 2026-11-02 |
| **Path** | The PyTorch reference of `batch_learning` on the 30.3M translator, as [experiment 67](067-muon-owned-layers.md); no coordinator or network. Arms: Muon 0.0025 and 0.005 at 5 steps, then the best rate (0.01) at 1, 2 and 3 steps; Adam and Muon 0.01 at 5 steps reused from experiment 67 after an exact anchor rerun |
| **Result** | **Tuning these two settings is not the route.** The best arm is still Muon 0.01 at 5 steps: 87% and 72% of Adam's time, mean 80% (required at most 70%, no seed above 80%), chrF 15.09 against 5.81. 3 steps comes close at 89% and 74% but ends at a worse loss, 5.176 against 5.033. Free; about 2 hours 10 minutes |
| **Evidence** | [`public/research/evidence/training/ws4-cheaper-muon-2026-11-02.json`](evidence/training/ws4-cheaper-muon-2026-11-02.json) (curves, chrF, timings, settings, hashes); raw runs in `runs/experiments/WS-4/omarchy-20261102-010750/` (local, not in git). Code at ec390dd, with only this file and the index dirty |

## Question

[Experiment 67](067-muon-owned-layers.md) failed its limit on time per update: Muon took 1.77× as long as Adam. Yet Muon 0.01 cut held-out loss by 0.68 and reached Adam's final loss in 72–87% of the training time. Two things were left open: 0.01 was the lowest rate tried and the best, and nearly all the added cost is the 5 Newton–Schulz steps.

**Do rates below 0.01 and fewer Newton–Schulz steps shorten the time to Adam's held-out loss, on the same seeds and batches?**

## Prediction and gate, as frozen

Frozen 2026-11-02, before any run.

- **Measure:** time to arm A's loss at 32,768 pairs, per seed: pairs needed (linear between evaluations) ÷ 32,768 × that step count's update-time ratio against Adam. An arm that does not reach the loss by 32,768 pairs fails on that seed.
- **Pass:** the best arm's two-seed mean is **at most 70%** of Adam's time, no seed above 80%, and its mean chrF is not lower than Adam's. Experiment 67's Muon 0.01 at 5 steps (72–87%, mean 79%) is the bar to beat.
- **Best arm:** the lowest two-seed mean time among all arms, chosen by that measure alone.
- **Time ratio:** the time check, ratio of median `update_seconds`.
- **Pass means:** a cheaper recipe worth carrying to the owners' stage 2. **Fail means:** experiment 67's recipe stays the best known, and tuning these two settings is not the route.
- **No rescue:** if the lowest rate or fewest steps wins, the grid is not extended here. That would be a new experiment.

Predictions, against the results:

| Prediction | Result |
|---|---|
| Newton–Schulz cost falls about in proportion to its steps; 2 steps add about 0.2 s per update | **Right:** 0.19 s at 2 steps, 0.47 s at 5 |
| Fewer steps lose some of the learning gain | **Right, but more than a little:** 2 steps keeps 0.29 of the 0.68 gain, and 1 step keeps none |
| The best rate may sit near 0.005 | **Right on final loss only:** 5.002 against 5.033, within noise. On the gate's measure 0.01 is faster |

## Protocol

As experiment 67 for everything not listed (30.3M, batches, 8 threads, 1,024 held-out pairs, chrF on those pairs). Arm A and Muon 0.01 at 5 steps are reused from experiment 67 because `batch_learning.py` is unchanged since then; one anchor rerun of Muon 0.01 on seed 2026 had to reproduce 5.0204 and chrF 14.34, or everything would be rerun.

| Stage | Arms | Seeds | Order |
|---|---|---|---|
| Anchor | Muon 0.01, 5 steps | 2026 | first |
| Time check | Adam, 1, 2, 3 and 5 steps; 8,192 pairs, 128-pair evaluation, 3 runs per arm | 2026 | A 1 2 3 5, then 5 3 2 1 A, then 2 A 5 1 3 |
| Rates | Muon 0.0025 and 0.005, 5 steps | 2026, 2027 | 2026: 0.0025 0.005; 2027: 0.005 0.0025 |
| Steps | The best rate at 5 steps, with 1, 2 and 3 steps | 2026, 2027 | 2026: 1 2 3; 2027: 3 2 1 |

The best rate at 5 steps is the lowest two-seed mean of pairs to A's final loss, among 0.0025, 0.005 and 0.01. Nothing ran at the same time, and the steps stage started only after the rates were read.

## Log

- 2026-11-02: Added, from experiment 67's result.
- 2026-11-02: Gate frozen and started at ec390dd (only this file and the index dirty), seeds 2026 and 2027, in `runs/experiments/WS-4/omarchy-20261102-010750/` (`drive.sh` has the exact commands). Watchdog: `--stall 300 --busy-stall 1200 --max-runtime 2400 --progress 'pairs [0-9]+'`.
- 2026-11-02: Anchor reproduced exactly (5.02038903, chrF 14.33673). Time check and rate arms finished, all exit 0. Update-time ratios: 1.160, 1.317, 1.469 and 1.779 at 1, 2, 3 and 5 steps. Pairs to A's final loss, two-seed mean: 0.0025 20,250, 0.005 16,588, 0.01 14,662, so the best rate by the frozen rule is 0.01 (0.005 has the lower final loss, 5.002 against 5.033). Steps stage started at 0.01 with `drive-steps.sh`.
- 2026-11-02: Steps stage finished, all exit 0. No failed or interrupted runs.

## Results

### Gate

| Condition | Required | Best arm: Muon 0.01, 5 steps | Outcome |
|---|---|---|---|
| Two-seed mean time to A's loss | ≤ 70% | **79.6%** | **Fail** |
| Slowest seed | ≤ 80% | **87.2%** | **Fail** |
| chrF, mean | not below Adam's 5.81 | 15.09 | Pass |

### Learning at 32,768 pairs

| Arm | Held-out, mean | Against A | chrF, mean | Pairs to A's loss | Update ratio | Time to A's loss | Mean time |
|---|---|---|---|---|---|---|---|
| A: Adam 5e-4 | 5.717 | n/a | 5.81 | n/a | 1 | n/a | 100% |
| **Muon 0.01, 5 steps** | 5.033 | −0.684 | 15.09 | 49%, 40% | 1.779 | 87%, 72% | **80%** |
| Muon 0.005, 5 steps | 5.002 | −0.714 | 16.40 | 54%, 47% | 1.779 | 97%, 84% | 90% |
| Muon 0.0025, 5 steps | 5.134 | −0.583 | 15.31 | 66%, 58% | 1.779 | 117%, 103% | 110% |
| Muon 0.01, 3 steps | 5.176 | −0.540 | 13.02 | 61%, 50% | 1.469 | 89%, 74% | 81% |
| Muon 0.01, 2 steps | 5.429 | −0.288 | 9.93 | 83%, 68% | 1.317 | 109%, 90% | 99% |
| Muon 0.01, 1 step | 5.713 | −0.003 | 5.17 | not reached, 93% | 1.160 | not reached, 108% | fails |

Time-to-loss percentages use seed 2026 first, then 2027. The anchor's whole held-out curve and its chrF equal experiment 67's run bit for bit.

### Time per update

Median `update_seconds` over 76 updates, three runs per arm, within 1% inside each arm:

| Adam | 1 step | 2 steps | 3 steps | 5 steps |
|---|---|---|---|---|
| 45.67 s | 53.00 s | 60.14 s | 67.08 s | 81.26 s |
| 1 | 1.160× | 1.317× | 1.469× | 1.779× |

The learning runs agree: 197 s for Adam, then 229, 260, 291 and 352 s (1.164×, 1.321×, 1.476×, 1.786×). The time check ran at rate 0.01, and the rate does not change the time per update.

## What it taught

- **The bar held:** no arm beat 5 steps at 0.01, and none came within reach of 70%. 3 steps ties it within noise on time (81% against 80%), so it is cheaper per update but no faster overall.
- **Each step removes as much learning as it saves in time:** pairs needed grow as fast as the update cost falls, so the time to loss stays flat from 5 to 3 steps and gets worse below that.
- **Below 0.01 is slower:** the rate gives the fastest early progress at 0.01, while 0.005 only catches up by the end. At that point the two are within the noise of two seeds.
- **The gate's measure favors higher rates than final loss does:** 0.005 has the best loss and chrF, and 0.01 reaches Adam's loss sooner. A recipe for a longer run could reasonably use a lower rate than one judged on this measure.
- **Fewer steps also mean a smaller update, so this grid does not isolate the steps:** Newton–Schulz starts from a matrix normalized to norm 1, and few steps don't lift it to full size. A check on Gaussian matrices, done after the gate was read, gives a Frobenius norm of about 3.4, 11, 21–25 and 21–22 at 1, 2, 3 and 5 steps. So 1 and 2 steps at rate 0.01 may behave partly like a much lower rate. Real momentum isn't Gaussian, so these ratios show the direction and not the real size. Raising the rate to match was not tried, because the frozen rule fixes the rate before the steps stage and forbids extending the grid.
  [Experiment 69](069-muon-fewer-steps-higher-rate.md) then tested this: real momentum shrinks more than these figures, and raising the rate to match makes the loss worse.

## Limits

- 3.6% of an epoch, on this CPU in float32. This doesn't measure the owners' cost in the distributed job, and it doesn't show the late-training effect.
- Two seeds can only separate differences of about 0.05 in loss. The ranking of 0.005 and 0.01 on final loss, and of 3 and 5 steps on time, is inside that.
- Step counts were tested at one rate. A pairing of fewer steps with a higher rate is untested (see above).
- Experiment 67's Muon 0.02 at 5 steps was not an arm here. With this run's ratio it would be 89% and 68%, mean 78%: slightly under the best arm's mean, within noise, and still failing the gate on both conditions.
- Time to loss interpolates linearly between evaluations 4,096 pairs apart.
- The desktop stayed on during the runs, though the time-check runs agree within 1% inside each arm.
- `ruff` is not installed in this environment; this experiment changed no code.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
OUT=runs/experiments/WS-4/$(hostname)-$(date +%Y%m%d-%H%M%S)
BL="python -m ygaraty.experiments.batch_learning"
MODEL="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072"
LEARN="$MODEL --pairs 32768 --eval-every 4096 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
SPEED="$MODEL --pairs 8192 --eval-every 100000 --eval-pairs 128 --seed 2026"
ADAM="--optimizer adam --learning-rate 0.0005"
muon() { echo "--optimizer muon --muon-lr $1 --muon-ns-steps $2 --adam-lr 0.0005"; }

# Anchor
$BL $LEARN --seed 2026 $(muon 0.01 5) --output $OUT/anchor-M0.01-5-2026

# Time check
i=0
for arm in A 1 2 3 5  5 3 2 1 A  2 A 5 1 3; do
  i=$((i + 1)); n=$(printf %02d $i)
  if [ $arm = A ]; then $BL $SPEED $ADAM --output $OUT/speed-$n-A
  else $BL $SPEED $(muon 0.01 $arm) --output $OUT/speed-$n-N$arm; fi
done

# Rates at 5 steps
for lr in 0.0025 0.005; do $BL $LEARN --seed 2026 $(muon $lr 5) --output $OUT/M$lr-5-2026; done
for lr in 0.005 0.0025; do $BL $LEARN --seed 2027 $(muon $lr 5) --output $OUT/M$lr-5-2027; done

# Steps at the best rate
for ns in 1 2 3; do $BL $LEARN --seed 2026 $(muon 0.01 $ns) --output $OUT/M0.01-$ns-2026; done
for ns in 3 2 1; do $BL $LEARN --seed 2027 $(muon 0.01 $ns) --output $OUT/M0.01-$ns-2027; done
```

Adam and Muon 0.01 at 5 steps come from experiment 67's Reproduce. The hint index comes from experiment 55. The original runs used the watchdog limits in the Log and wrote each result under `<run>/result`.
