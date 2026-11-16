# Muon learns far more per pair, but its optimizer step makes CPU updates 1.77× slower

| | |
|---|---|
| **Conclusion** | **Failed gate:** Muon cuts held-out loss 0.68 but makes CPU updates 1.77× slower, limit 1.15× (measured, two seeds). |
| **Status** | executed |
| **ID** | WS-3 → experiment 67 |
| **Category** | training |
| **Where** | workstation |
| **Date** | Added 2026-11-01; run 2026-11-01 and 2026-11-02 |
| **Path** | Stage 1 only, on the Ryzen 9 PRO 6950H workstation: the PyTorch reference of `batch_learning` (30.3M translator, float32, 8 threads), no coordinator or network. Arms: Adam 5e-4; Muon at 0.01, 0.02 and 0.04 with Adam 5e-4 on the rest; the best Muon arm with target n-gram hints. Stage 2 was not built because stage 1 failed its gate |
| **Result** | **Not a lever by the frozen rule.** Held-out loss at 32,768 pairs, mean of seeds 2026 and 2027: Adam 5.717, Muon 0.01 **5.033 (−0.68**, required −0.23), chrF 5.81 → 15.09, but the update takes 1.769× as long (limit 1.15×). Muon 0.01 still reaches Adam's final loss in 72–87% of its training time on this CPU, and with hints 4.771. Free; about 2 hours |
| **Evidence** | [`public/research/evidence/training/ws3-muon-owned-layers-2026-11-02.json`](evidence/training/ws3-muon-owned-layers-2026-11-02.json) (curves, chrF, timings, settings, hashes); raw runs in `runs/experiments/WS-3/omarchy-20261101-225923/` (local, not in git); `tests/test_muon.py`. Code at e51accd plus this experiment's changes |

## Question

The NanoGPT speedrun (github.com/KellerJordan/modded-nanogpt) got its third-largest single drop from the **Muon** optimizer: 31.4 → 24.9 minutes, and 22.3 after further Muon changes.

Muon keeps one momentum buffer per 2-D weight matrix. It orthogonalizes the momentum with a few Newton–Schulz iterations and uses that as the update. Embeddings, norms and biases stay on Adam.

Muon needs the **whole matrix** in one place. Ygaraty's owners already hold whole layers (`tf_owned_step` in `ygaraty/runtime/transformer_kernels.py`), so Muon would add no communication. It would also halve the owners' optimizer state: one buffer instead of Adam's two.

**Does Muon on the Transformer layers lower the 30.3M translator's held-out loss at 32,768 pairs by at least as much as the n-gram hints' bar in experiment 55, with the same seeds and batches?**

## Prediction and gate, as frozen

- **Counts as a lever:** averaged over both seeds, the best Muon arm's held-out loss at 32,768 pairs is at least **0.23 below** arm A's (the size of experiment 35's batch effect, the bar experiment 55 used). Its chrF must not be lower, and its optimizer step must add at most 15% to training time per step.
- **Time to loss:** pairs needed to reach arm A's 32,768-pair loss, times the measured per-step cost ratio.
- **Time gate's measure:** a training-time check at 8,192 pairs with a 128-pair evaluation, rotated A M2 M2 A A M2. The ratio of median `update_seconds` must be at most 1.15. Update time excludes evaluations and decoding.
- **Best Muon arm:** the lowest two-seed mean held-out loss among M1–M3.
- **Baseline:** arm A is rerun here and is the comparison; it should reproduce experiment 55's arm N.
- **Rest of the parameters:** the plan's "AdamW" means the baseline's Adam with no weight decay (betas 0.9, 0.98, eps 1e-8), so the optimizer on the layers' matrices is the one thing that changes.
- **chrF:** greedy on the same 1,024 pairs, compared on two-seed means. Every arm runs `--decode --eval-dedup --hint-index` like experiment 55.

Predictions, against the results:

| Prediction | Result |
|---|---|
| Muon at 0.02 is 0.15–0.35 below Adam | **Wrong, too cautious:** 0.48 below |
| Newton–Schulz adds 3–8% per step on CPU | **Wrong:** 77%. The arithmetic added before the runs (about 2.6e11 FLOPs per update) had warned of this |
| MH is better than either alone, but less than the sum of the two gains | **Right:** −0.95, against −0.68 (Muon) and −0.40 (experiment 55's target hints), which sum to −1.08 |

## Protocol

Experiment 41's settings: width 512, 3+3 layers, 8 heads, batch 64, length window 64, 3,072-token batches, 32,768 pairs, held-out every 4,096 pairs on 1,024 pairs. Each run has 333 updates. Runs were strictly sequential under the watchdog: time check first, then seed 2026 as A M1 M2 M3 and seed 2027 as M3 M2 M1 A, then MH on both seeds.

Muon applies to the 30 matrices inside the encoder and decoder layers (attention in and out projections, feed-forward weights): momentum 0.95 with Nesterov, 5 Newton–Schulz steps with the quintic coefficients 3.4445, −4.7750, 2.0315, scaled by max(1, rows/cols)^0.5. Everything else stays on Adam 5e-4.

The local skill's default of the tiny model does not apply here: this is one PyTorch process, like experiments 41 and 55, and the gate is defined on the 30.3M model.

## Log

- 2026-11-01: Added (first committed as a numbered plan file; renumbered to a track ID, since numbers mark finished experiments). Local WS-2 reassigned to WS-3 to avoid a different upstream experiment.
- 2026-11-01: Built `ygaraty/reference/muon.py`, the `batch_learning` flags and `tests/test_muon.py`; design frozen. Started at e51accd+dirty, seeds 2026 and 2027. Run with `python -m ygaraty.experiments.batch_learning` because `.venv/bin/ygaraty` has a stale interpreter path. Watchdog: `--stall 300 --busy-stall 1200 --max-runtime 2400 --progress 'pairs [0-9]+'`. A 512-pair rehearsal with Muon, hints and decoding passed.
- 2026-11-02: Time check and 8 learning runs finished, all exit 0. Best Muon arm 0.01. MH finished, exit 0. No failed or interrupted runs.

## Results

### Gate

| Condition | Required | Muon 0.01 | Outcome |
|---|---|---|---|
| Held-out loss against Adam | ≤ −0.23 | **−0.684** (−0.606, −0.763) | Pass |
| chrF | not lower | 15.09 against 5.81 | Pass |
| Update time against Adam | ≤ 1.15× | **1.769×** | **Fail** |

### Learning at 32,768 pairs

| Arm | Held-out 2026 | Held-out 2027 | Mean | Against A | chrF, mean | Pairs to A's final loss | Time to A's final loss |
|---|---|---|---|---|---|---|---|
| A: Adam 5e-4 | 5.626 | 5.807 | 5.717 | n/a | 5.81 | n/a | n/a |
| M1: Muon 0.01 | 5.020 | 5.045 | **5.033** | **−0.684** | 15.09 | 49%, 40% | 87%, 72% |
| M2: Muon 0.02 | 5.216 | 5.259 | 5.238 | −0.479 | 11.57 | 50%, 38% | 88%, 68% |
| M3: Muon 0.04 | 5.493 | 5.507 | 5.500 | −0.217 | 8.93 | 67%, 45% | 118%, 79% |
| MH: Muon 0.01 + target hints | 4.751 | 4.791 | **4.771** | **−0.946** | 15.57 | 41%, 36% | about 75%, 65% |

Time to loss uses the time check's 1.769×. MH uses the learning runs' own ratio of about 1.81×, and leaves out the index build (135 s, experiment 55). MH's held-out loss with hints switched off is 5.066 and 5.110, 0.05 to 0.07 worse than Muon alone.

Arm A reproduces experiment 55's arm N exactly on both seeds (5.62590381 and 5.80734676, same chrF).

### Time per update

| Run | 1 A | 2 M2 | 3 M2 | 4 A | 5 A | 6 M2 |
|---|---|---|---|---|---|---|
| 76 updates, seconds | 46.54 | 80.97 | 80.83 | 45.46 | 45.74 | 80.93 |

Median 45.74 s against 80.93 s: **1.769×**. The learning runs agree, at about 0.59 s and 1.055 s per update (1.786× across M1–M3). The optimizer step goes from 0.04 s to 0.50 s; everything else is unchanged at about 0.55 s.

## What it taught

- **The learning gain is large and consistent:** every Muon rate beats Adam on both seeds, chrF rises with it, and the gain is about three times the bar. It is also larger than experiment 55's hints (−0.40), and the two mostly add.
- **The cost is arithmetic:** Newton–Schulz on matrices of 512 by up to 2,048 is real work on a CPU, about 0.46 s of added time per update here. The plan's 3–8% was wrong by an order of magnitude.
- **Failing the gate does not mean slower training:** on this CPU the overhead is smaller than the learning gain, so Muon 0.01 still reaches Adam's final loss sooner on both seeds. The frozen rule limits time per update, so the verdict stands. A different rule would be a new experiment.
- **The lowest rate won:** loss falls steadily as the rate drops from 0.04 to 0.01, and 0.01 is the edge of the grid, so the best rate may be lower.

## Limits

- 3.6% of an epoch and early training. Muon's published gains are on language models trained to a fixed target, and this does not show the late-training effect.
- Two seeds can only separate differences of about 0.05 in loss.
- The reference runs float32 on 8 CPU threads. Owners use float64 on their own cores, so this does not measure the distributed cost: it could be larger per owner, and it could matter less against an update of 4.64 s on home links.
- The time check ran at rate 0.02 only; the learning runs show the same time per update at all three rates.
- The desktop stayed on during the runs (browser, terminal, screensaver), though the time-check runs agree within 2.4% inside each arm.
- Learning rates were not tuned beyond three points, and 0.01 is at the edge of the grid.
- `ruff` is not installed in this environment, so the lint and format checks on the new code were not run.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_muon
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
OUT=runs/experiments/WS-3/$(hostname)-$(date +%Y%m%d-%H%M%S)
BL="python -m ygaraty.experiments.batch_learning"
MODEL="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072"
LEARN="$MODEL --pairs 32768 --eval-every 4096 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
SPEED="$MODEL --pairs 8192 --eval-every 100000 --eval-pairs 128 --seed 2026"
ADAM="--optimizer adam --learning-rate 0.0005"
muon() { echo "--optimizer muon --muon-lr $1 --adam-lr 0.0005"; }

# Time gate, in the order A M2 M2 A A M2
$BL $SPEED $ADAM --output $OUT/speed-1-A
$BL $SPEED $(muon 0.02) --output $OUT/speed-2-M2
$BL $SPEED $(muon 0.02) --output $OUT/speed-3-M2
$BL $SPEED $ADAM --output $OUT/speed-4-A
$BL $SPEED $ADAM --output $OUT/speed-5-A
$BL $SPEED $(muon 0.02) --output $OUT/speed-6-M2

# Learning
$BL $LEARN --seed 2026 $ADAM --output $OUT/A-2026
for lr in 0.01 0.02 0.04; do $BL $LEARN --seed 2026 $(muon $lr) --output $OUT/M$lr-2026; done
for lr in 0.04 0.02 0.01; do $BL $LEARN --seed 2027 $(muon $lr) --output $OUT/M$lr-2027; done
$BL $LEARN --seed 2027 $ADAM --output $OUT/A-2027

# Best rate with hints
for seed in 2026 2027; do $BL $LEARN --seed $seed $(muon 0.01) --hints target --output $OUT/MH-$seed; done
```

The hint index comes from experiment 55. The original runs used the watchdog limits in the Log and wrote each result under `<run>/result`.
