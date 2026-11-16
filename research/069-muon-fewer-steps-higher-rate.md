# Raising Muon's rate at 1–2 Newton–Schulz steps makes it worse: update size wasn't the problem

| | |
|---|---|
| **Conclusion** | **Failed gate:** raising Muon's rate at 1–2 steps hurts; best 99% of Adam's time, limit 70% (measured, two seeds). |
| **Status** | executed |
| **ID** | WS-5 → experiment 69 |
| **Category** | training |
| **Where** | workstation |
| **Date** | Added and run 2026-11-02 |
| **Path** | The PyTorch reference of `batch_learning` on the 30.3M translator, as [experiment 68](068-cheaper-muon.md); no coordinator or network. Update size measured on real momentum, then Muon at 1 step (rates 0.08, 0.04) and 2 steps (rates 0.034, 0.017), with rates set by a frozen rule |
| **Result** | **Tuning rate and steps ends here on this model.** The best new arm, 2 steps at 0.017, takes 112% and 87% of Adam's time, mean 99% (required at most 70%, no seed above 80%), chrF 8.88 against Adam's 5.81. At both step counts the final loss worsens steadily as the rate rises from 0.01. Muon 0.01 at 5 steps (87%, 72%, mean 80%) stays the best known. Free; about 1 hour 35 minutes |
| **Evidence** | [`public/research/evidence/training/ws5-muon-fewer-steps-higher-rate-2026-11-02.json`](evidence/training/ws5-muon-fewer-steps-higher-rate-2026-11-02.json) (curves, chrF, timings, update sizes, settings, hashes); raw runs in `runs/experiments/WS-5/omarchy-20261102-104516/` (local, not in git). Code at 6cc6eb6, with only this file and the index dirty |

## Question

[Experiment 68](068-cheaper-muon.md) tested 1, 2 and 3 Newton–Schulz steps at rate 0.01 only, and none beat 5 steps. But few steps also shrink the update: on Gaussian matrices its Frobenius norm is about 3.4, 11 and 21–25 at 1, 2 and 3 steps, against 21–22 at 5. So 1 and 2 steps at 0.01 may have behaved partly like a much lower rate.

**Does Muon at 1 or 2 steps with a rate raised to restore the update size reach Adam's held-out loss sooner than experiment 68's best arm, on the same seeds and batches?**

## Prediction and gate, as frozen

Frozen 2026-11-02, before the update-size measurement and before any learning run.

- **Measure:** experiment 68's. Time to arm A's loss at 32,768 pairs, per seed: pairs needed (linear between evaluations) ÷ 32,768 × the step count's update-time ratio against Adam (1.160 at 1 step, 1.317 at 2, reused from experiment 68's time check). An arm that does not reach the loss by 32,768 pairs fails on that seed.
- **Pass:** the best new arm's two-seed mean is **at most 70%** of Adam's time, no seed above 80%, and its mean chrF is not below Adam's 5.81. Muon 0.01 at 5 steps (87%, 72%, mean 80%) is the bar to beat.
- **Best arm:** the lowest two-seed mean time among the new arms, chosen by that measure alone.
- **Pass means:** a cheaper recipe worth carrying to the owners' stage 2. **Fail means:** experiment 67's recipe stays the best known, and no more tuning of rate and steps on this model.
- **No rescue:** if an edge of the grid wins, the grid is not extended here. That would be a new experiment.

Predictions, against the results:

| Prediction | Result |
|---|---|
| A raised rate does better than 0.01 at the same step count | **Wrong:** every raised rate ends at a worse loss |
| Mean time of 75–95% | **Wrong, too hopeful:** 99% at best, and three of the four arms miss Adam's loss on at least one seed |
| Most likely a fail | **Right** |

## Protocol

As experiment 68 for everything not listed (30.3M, batches, 8 threads, 1,024 held-out pairs, chrF on those pairs). Arm A, Muon 0.01 at 5 steps and the time-check ratios are reused from experiments 67 and 68 because `batch_learning.py` and `muon.py` are unchanged since then (hashes 1ac803ca and 961a7cdf).

| Stage | Arms | Seeds | Order |
|---|---|---|---|
| Update size | Muon 0.01 at 5 steps, 32,768 pairs, no decoding. Every 10th update, for each of the 30 matrices, record the Frobenius norm of `orthogonalize` at 1, 2, 3 and 5 steps on the real momentum | 2026 | first |
| Anchor | Muon 0.01, 5 steps; must reproduce 5.0204 and chrF 14.34, or everything is rerun | 2026 | second |
| Learning | At 1 and 2 steps: the matched rate m and m/2 | 2026, 2027 | 2026: 1m, 1m/2, 2m, 2m/2; 2027: reversed |

**Rates, fixed by rule:** m at k steps = 0.01 × the median over all samples of (norm at 5 steps ÷ norm at k steps), rounded to two significant figures. m gives an update about the size of Muon 0.01 at 5 steps, and m/2 about the size of Muon 0.005 at 5 steps, the two best rates in experiment 68.

**Stop rule:** a step count with m below 0.0125 is not run, because its update is already within 25% of the size experiment 68 tested. If both step counts stop, the experiment finishes without learning runs. Neither stopped.

Nothing ran at the same time, and the learning runs started only after the rates were computed and logged.

## Log

- 2026-11-02: Added, from experiment 68's result.
- 2026-11-02: Gate frozen and started at 6cc6eb6 (only this file and the index dirty), seeds 2026 and 2027, in `runs/experiments/WS-5/omarchy-20261102-104516/`. Watchdog: `--stall 300 --busy-stall 1200 --max-runtime 2400 --progress 'pairs [0-9]+'`. A 1,024-pair rehearsal of the size measurement passed.
- 2026-11-02: Update size measured on real momentum (1,020 samples, 30 matrices, 34 updates): norm at 5 steps ÷ norm at k steps has median 8.00 at 1 step, 3.42 at 2 and 1.84 at 3, and shrinks over the run (1 step: 12.2 before update 50, 7.0 after update 170). Rates by the frozen rule: 1 step m 0.08, m/2 0.04; 2 steps m 0.034, m/2 0.017. Neither stops. Anchor reproduced exactly (5.02038903, chrF 14.33673). Learning runs started with `drive-learn.sh`.
- 2026-11-02: Learning runs finished, all exit 0. No failed or interrupted runs.

## Results

### Gate

| Condition | Required | Best new arm: 2 steps, rate 0.017 | Outcome |
|---|---|---|---|
| Two-seed mean time to A's loss | ≤ 70% | **99.5%** | **Fail** |
| Slowest seed | ≤ 80% | **112.2%** | **Fail** |
| chrF, mean | not below Adam's 5.81 | 8.88 | Pass |

### Learning at 32,768 pairs

Rows marked 0.01 come from experiments 67 and 68, for comparison. Time-to-loss percentages use seed 2026 first, then 2027.

| Steps | Rate | Held-out, mean | Against A | chrF, mean | Pairs to A's loss | Time to A's loss | Mean time |
|---|---|---|---|---|---|---|---|
| 5 | 0.01 | 5.033 | −0.684 | 15.09 | 49%, 40% | 87%, 72% | **80%** |
| 2 | 0.01 | 5.429 | −0.288 | 9.93 | 83%, 68% | 109%, 90% | 99% |
| **2** | **0.017** | 5.498 | −0.218 | 8.88 | 85%, 66% | 112%, 87% | **99%** |
| 2 | 0.034 | 5.659 | −0.058 | 6.48 | not reached, 75% | not reached, 99% | fails |
| 1 | 0.01 | 5.713 | −0.003 | 5.17 | not reached, 93% | not reached, 108% | fails |
| 1 | 0.04 | 5.863 | +0.147 | 5.50 | not reached on either seed | not reached | fails |
| 1 | 0.08 | 6.048 | +0.331 | 4.95 | not reached on either seed | not reached | fails |

Adam's mean is 5.717 with chrF 5.81. The anchor's whole held-out curve and its chrF equal experiment 67's run bit for bit. Update times agree with the reused ratios: 229 s at 1 step and 260–263 s at 2, against 197 s for Adam.

### Update size on real momentum

Median of norm at 5 steps ÷ norm at k steps:

| Steps | Whole run | 10th–90th percentile | Before update 50 | Updates 50–169 | From update 170 |
|---|---|---|---|---|---|
| 1 | 8.00 | 6.25–11.79 | 12.17 | 8.24 | 7.05 |
| 2 | 3.42 | 2.52–5.50 | 5.77 | 3.84 | 2.93 |
| 3 | 1.84 | 1.30–3.08 | 3.20 | 2.04 | 1.62 |

## What it taught

- **Update size wasn't the problem:** the size gap is real, and larger than the Gaussian check in experiment 68 suggested. But closing it hurts. At 1 step, loss rises 5.713 → 5.863 → 6.048 as the rate goes 0.01 → 0.04 → 0.08, and at 2 steps 5.429 → 5.498 → 5.659 as it goes 0.01 → 0.017 → 0.034.
- **So the quality of the orthogonalization is what matters at these step counts.** Muon's learning gain needs the singular values evened out, not only a larger update. A likely reason, not measured here: with few steps the largest singular directions still dominate, and a higher rate amplifies those most.
- **The early curves show the same thing:** higher rates at few steps match or lead at 4,096 pairs (1 step at 0.08 is at 6.86, against 7.15 at 0.01), then flatten and fall behind.
- **A Gaussian check was a poor guide:** real momentum shrinks about 1.3× more than Gaussian matrices at 1 step and 1.8× more at 2. Experiment 68's 3-step arm has an update 1.84× smaller than 5 steps, not about the same size. That measurement replaces the Gaussian figures, and it was worth taking before choosing rates.

## Limits

- 3.6% of an epoch, on this CPU in float32. This doesn't measure the owners' cost in the distributed job, and it doesn't show the late-training effect.
- Two seeds can only separate differences of about 0.05 in loss. The trends with rate here are much larger than that, and hold on both seeds.
- 3 steps at a raised rate (about 0.018) was not in the grid. Experiment 68's 3 steps at 0.01 ties the bar at 81%, so this is the one cheaper point these runs don't speak to directly, and the trend at 1 and 2 steps gives no reason to expect it to reach 70%. The frozen outcome is no more tuning on this model.
- Update size was measured on Muon 0.01 at 5 steps, seed 2026 only. Momentum itself doesn't depend on the step count, but it does depend on the weights, so the ratios could differ along other arms' trajectories.
- A single rate can't match size per matrix: the ratios spread by about 2× across samples.
- The desktop stayed on during the runs.
- `ruff` is not installed in this environment; this experiment changed no code in the repository.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
OUT=runs/experiments/WS-5/$(hostname)-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
BL="python -m ygaraty.experiments.batch_learning"
MODEL="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072"
LEARN="$MODEL --pairs 32768 --eval-every 4096 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
muon() { echo "--optimizer muon --muon-lr $1 --muon-ns-steps $2 --adam-lr 0.0005"; }

# Update size
cat > $OUT/measure_size.py <<'EOF'
import json, sys
import ygaraty.reference.muon as muon
from ygaraty.experiments import batch_learning
out = sys.argv.pop(1)
original, calls, log = muon.orthogonalize, 0, open(out, "w")
def recording(matrix, steps=5):
    global calls
    update, index = calls // 30, calls % 30
    calls += 1
    if update % 10 == 0:
        norms = {k: float(original(matrix, k).norm()) for k in (1, 2, 3, 5)}
        log.write(json.dumps(dict(update=update, matrix=index, shape=list(matrix.shape), norms=norms)) + "\n")
    return original(matrix, steps)
muon.orthogonalize = recording
batch_learning.main()
EOF
PYTHONPATH=$PWD python $OUT/measure_size.py $OUT/norms.jsonl $MODEL --pairs 32768 --eval-every 100000 --eval-pairs 128 --seed 2026 $(muon 0.01 5) --output $OUT/size

# Anchor
$BL $LEARN --seed 2026 $(muon 0.01 5) --output $OUT/anchor-M0.01-5-2026

# Learning, with the rates the rule gave
for arm in "0.08 1" "0.04 1" "0.034 2" "0.017 2"; do set -- $arm; $BL $LEARN --seed 2026 $(muon $1 $2) --output $OUT/M$1-$2-2026; done
for arm in "0.017 2" "0.034 2" "0.04 1" "0.08 1"; do set -- $arm; $BL $LEARN --seed 2027 $(muon $1 $2) --output $OUT/M$1-$2-2027; done
```

Adam and Muon 0.01 at 5 steps come from experiment 67's Reproduce, and the update-time ratios from experiment 68's. The hint index comes from experiment 55. The original runs used the watchdog limits in the Log and wrote each result under `<run>/result`.
