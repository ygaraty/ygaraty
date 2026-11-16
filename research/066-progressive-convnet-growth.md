# Growing the ConvNet from the head stalls at 33%; channel-copy layers are 3× slower even at full depth

| | |
|---|---|
| **Conclusion** | **Failed gate:** progressive growth stalls at 33% dev on 3/3 seeds; full-random reaches 60% in 226 s (measured, workstation). |
| **Status** | executed |
| **ID** | M-1 B → experiment 66 (M-1 A stays open in [M-1.md](M-1.md)) |
| **Category** | training |
| **Where** | workstation |
| **Date** | Added, started and finished 2026-11-01 |
| **Path** | DiLoCo CIFAR-10 `SmallConvNet` (94,538 parameters) on one host: 4 threaded replicas, H=50, batch 64, inner Adam 0.001, outer Nesterov 0.7/0.9, float64, 8 threads. 40,000 training, 5,000 development (growth and target decisions), 5,000 holdout (final only). Arms: `progressive` (head on fixed features, then three convolutions inserted toward the input), `full-random`, `full-copy` (final architecture with the insertion's channel-copy initialization), `head-only`. Seeds 2026–2028, 70 rounds |
| **Result** | **Not worth it.** No progressive run reached the 60% development target. All three ended `stage_stalled` at stage 1 after 24–26 rounds and about 39 s: stage 1 was still improving at its 15-round cap, and the frozen rule grows only after a plateau. Their holdout accuracy is 31.1–31.9%, 41 pp under full-random's 71.9–73.6%. Full-random reaches 60% in 205/226/241 s (median 226 s). Separately from the growth rule, the insertion's initialization is a handicap: full-copy, the same final architecture started from channel-copy convolutions, needs 627–628 s (2 of 3 seeds; the third peaks at 59.9%) and ends at 59.0–60.7% holdout. Growth is cheap early, about 1.5 s per round at stage 1 against 10.6 s for the full model, and at 20 s progressive leads 26–29% against 21–24%. By 39 s full-random has caught up on two of three seeds. Every insertion preserved logits exactly (error 0.0, tolerance 1e-10). Cost: free, about 80 minutes of workstation time |
| **Evidence** | [`public/research/evidence/training/m1b-progressive-convnet-growth-2026-11-01.json`](evidence/training/m1b-progressive-convnet-growth-2026-11-01.json) (per-arm summary, development curves, transitions, settings, data/split/source hashes). Raw: `runs/experiments/M-1/campaign-20261101-181903/` and verification `runs/experiments/M-1/workstation-20261101-175648/` (local, not in git). Code at `15db456`; the growth files are unchanged at `0afa40b` |

## Question

[M-1](M-1.md) idea B: can a model learn before it grows? Train the output head first, then insert hidden blocks toward the input only after the current stage has learned something and plateaued. Each insertion is initialized to preserve the previous predictor. Does this reach a fixed held-out quality sooner than training the same final architecture from scratch?

## Gate (frozen before running)

- Target: 60% development accuracy, sustained for two consecutive evaluations.
- Pass: every progressive seed reaches the target, the median time to target is at least 10% below full-random's, and holdout accuracy is no more than 0.5 pp below full-random's for every seed.
- Growth: at least 1% relative development-loss gain over stage entry, then 3 evaluations without a new best above 0.1%. Stage 0 must also beat the class prior by 1%. Pre-final stages are capped at 15 rounds, and a stage that misses the gate is recorded as stalled; growth is never forced.
- Budget: 70 rounds or 28 minutes per arm.

## Results

| Arm | Reached 60% | Time to 60% (s) | Holdout accuracy | Dev accuracy at 20 s | Dev accuracy at 39 s |
|---|---|---|---|---|---|
| progressive | 0/3 (stalled at stage 1) | — | 31.9, 31.1, 31.2% | 26.3, 27.6, 28.6% | 33.5, 33.2, 32.7% |
| full-random | 3/3 | 205, 226, 241 | 73.6, 72.9, 71.9% | 21.5, 24.5, 24.3% | 29.6, 32.5, 33.4% |
| full-copy | 2/3 | 627, 628, — | 59.8, 60.7, 59.0% | 19.6, 20.6, 22.7% | 25.6, 25.7, 27.6% |
| head-only | 0/3 | — | 20.1, 21.9, 21.2% | 21.3, 22.0, 22.4% | same (ends at 15 s) |

Seeds are listed in order 2026, 2027, 2028. Gate: 0/3 progressive seeds reached the target, and the holdout gaps are −41.7, −41.8 and −40.7 pp. **FAIL.**

Stage 0 passed its gate on every seed (development loss 2.13 against the prior's 2.30). It grew after 11 rounds and about 3 s. In stage 1, development loss fell on almost every round (for seed 2026, 2.12 → 1.84) until the 15-round cap.

## What it taught

- **The plateau rule can't grow a stage that is still learning.** With a 15-round cap and stage 1 improving about 1% per round, the plateau condition never triggers, so the run ends at the cap. This is the protocol working as frozen ("never force growth"). It doesn't show that growth itself is bad.
- **The function-preserving initialization costs more than the growth rule.** Full-copy isolates it: the same architecture and budget, started from channel-copy convolutions, takes 2.8× as long as random initialization to reach 60% and loses 13 pp of holdout accuracy. Copying RGB channels preserves logits but starts every layer with only 3 distinct features. Fixing the growth rule alone would put progressive on full-copy's track, far from a 10% speedup.
- **Small early models are cheap and learn quickly per second.** For the first 20 s, progressive is ahead. Without a better insertion, that lead doesn't survive.
- **Head-only** plateaus at about 22%, so stage 0's features (three pooled colours) are a weak starting point, as M-1 anticipated.

## Limits

- One model (a 94,538-parameter ConvNet), one dataset, 3 seeds.
- Single-host threaded timing (8 threads), not distributed time. Payload bytes are estimates.
- The 60% target and the thresholds are the proposals frozen in M-1, not tuned.
- The result rejects this growth rule with this insertion. It does not reject growing toward the input in general: a cap-triggered growth rule and a non-degenerate function-preserving insertion are untested.
- The campaign loop stopped once, before `full-copy-2028` launched, because the watchdog script was deleted mid-run. That arm and `head-only-2028` were rerun with identical flags and code. No arm was interrupted.
- Saved weights are not resumable checkpoints.

## Reproduce

Data: `ygaraty.experiments.cifar_prepare --archive cifar-10-binary.tar.gz --output runs/datasets/cifar10/cifar-pilot.npz --seed 2026` from the official archive (MD5 `c32a1d4ab5d03f1284b67883e8d87530`). This gives SHA-256 `b979811a7c37af84808aa97dac599ef91ded3463df9a16ac11ee8be5ff08440e`, and the split SHA-256 is `773c2eea…`.

Verification, then each arm (mode in `progressive full-random full-copy head-only`, seed in `2026 2027 2028`), under the experiments skill's `watch.py` (stall 300 s, busy stall 1200 s, max 1800 s):

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_growth_policy.py'
.venv/bin/python -m unittest discover -s tests -p 'test_convnet_growth.py'
.venv/bin/python -m unittest discover -s tests -p 'test_diloco_cifar.py'
.venv/bin/python -m ygaraty.experiments.diloco_cifar_growth \
  --data runs/datasets/cifar10/cifar-pilot.npz --output runs/experiments/M-1/<campaign>/<mode>-<seed>/result \
  --mode <mode> --seed <seed> \
  --replicas 4 --local-steps 50 --rounds 70 --batch 64 \
  --execution threaded --threads 8 --dtype float64 \
  --learning-rate 0.001 --outer-learning-rate 0.7 --outer-momentum 0.9 \
  --development-count 5000 --split-seed 6101 \
  --growth-min-gain 0.01 --growth-plateau-gain 0.001 \
  --growth-patience 3 --growth-stage-rounds 15 --growth-max-stage 3 \
  --target-accuracy 0.60 --target-patience 2 --max-seconds 1680
```
