# Owners one update ahead lose 0.17 of held-out loss even with a look-ahead correction

| | |
|---|---|
| **Conclusion** | **Failed gate:** gradients one update stale, corrected, lose 0.17 of held-out loss, against 0.1 allowed (measured learning screen, two seeds). |
| **Status** | executed (learning screen); the distributed stage is not built |
| **ID** | WS-6 (experiment 77) |
| **Category** | training |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-03 |
| **Path** | Learning screen in the PyTorch reference (`batch_learning`) of the 30.3M translator, experiment 67's Adam recipe, 32,768 pairs, seeds 2026 and 2027. Every encoder and decoder layer's gradient is computed on that layer's weights from k updates earlier (`--stale-layers k`), with or without a Nesterov-style extrapolation (`--delay-correction nesterov`). Arms: synchronous, k = 1 and 2, each with and without correction |
| **Result** | Mean held-out loss at 32,768 pairs against synchronous 5.717: k = 1 +0.386, **k = 1 corrected +0.166** (gate ≤0.1), k = 2 +0.565, k = 2 corrected +0.389. The correction recovers 57% of k = 1's loss and 31% of k = 2's, not enough; the distributed version is not worth building as specified. Free; about 1 hour |
| **Evidence** | [`evidence/training/ws6-stale-layer-weights-2026-11-03.json`](evidence/training/ws6-stale-layer-weights-2026-11-03.json) (curves, chrF, gate summary); raw runs, `queue.sh` and `evidence.py` in `runs/experiments/queue-20261103-185345/` (local, not in git). Code at 524853b: `StaleLayers` in `ygaraty/experiments/layer_variants.py`, `batch_learning --stale-layers/--delay-correction`; `tests/test_layer_variants.py` |

## Question

The update is bound by its chain of hops, and owners wait 3–5.5 s per update with nothing to do (L1). In asynchronous pipeline parallelism, each stage applies its update as soon as its own gradients are in, so consecutive updates overlap, at the cost of stale weights. Ajanthan et al. (ICML 2025, arXiv:2505.01099) correct for staleness with a modified Nesterov look-ahead.

**With owners allowed to run k updates ahead of the commit (k = 1, 2) and a delay correction on their optimizer, does the translator reach a fixed held-out loss in less wall time than the synchronous update?** This first stage only screens learning: how much the staleness costs per pair.

## Prediction and gate (frozen before any run, learning screen only)

**Prediction:** without correction, a small loss at k = 1 and a larger one at k = 2; with the correction, within 0.05 of synchronous at k = 1.

**Gate:** the screen passes, so the distributed version is worth building, if the corrected k = 1 arm's mean held-out loss over seeds 2026 and 2027 at 32,768 pairs is within **0.1** of the synchronous arm's. Otherwise stop here. The primary wall-time gate (at least 20% shorter time to the synchronous loss) belongs to the distributed stage.

**Staleness model:** every encoder and decoder layer's gradient is computed on that layer's weights from k updates earlier; the embedding, positions and output bias stay current, and the optimizer applies the gradient to the current weights. **Correction:** our reading of a Nesterov-style look-ahead, the stale weights extrapolated along their own last update, w(t−k) + k·(w(t−k) − w(t−k−1)). It is not claimed to be Ajanthan et al.'s algorithm.

## Protocol

- `StaleLayers` keeps the last k + 2 snapshots of the layer weights. Before the forward pass it loads the stale (or extrapolated) weights; after backward it restores the current ones, keeping the gradients; after the step it records a new snapshot. A test checks the gradient against one computed directly on the old weights, bit for bit, and that the current weights are restored before every step.
- The four arms ran in the shared 22-run queue described in [experiment 76](076-parallel-layers.md). Its synchronous baseline reproduces experiment 67's Adam arm (5.717).

## Log

- 2026-11-01: Added, from a literature review of distribution overhead.
- 2026-11-03: Learning-screen gate frozen with the user's yes.
- 2026-11-03: Queue `finished`, 22 of 22 runs exited 0. WS-6's arms ran 21:03–22:12.

## Results

| Arm | Seed 2026 | Seed 2027 | Mean | Against sync | chrF mean |
|---|---|---|---|---|---|
| Synchronous | 5.626 | 5.807 | 5.717 | — | 5.80 |
| k = 1 | 6.011 | 6.195 | 6.103 | +0.386 | 2.13 |
| **k = 1, corrected** | 5.846 | 5.920 | 5.883 | **+0.166** | 4.78 |
| k = 2 | 6.219 | 6.345 | 6.282 | +0.565 | 3.34 |
| k = 2, corrected | 6.059 | 6.152 | 6.106 | +0.389 | 2.63 |

**Screen: fails** (+0.166 against ≤0.1). The corrected k = 1 arm trails at every evaluation point, by 0.08–0.10 at 4,096 pairs and 0.11–0.22 at 32,768 pairs (per seed). Uncorrected stale arms also train unevenly: k = 2's loss rises between some evaluations. Three seed-2027 stale runs decode to degenerate output (chrF 0.09).

## What it taught

- **Staleness costs much more per pair than predicted.** One update of staleness costs 0.39 uncorrected, more than three times the 0.116 that the last 4,096 synchronous pairs buy.
- **The extrapolation helps, but not enough.** It recovers 57% of k = 1's cost and 31% of k = 2's. It is a simple reading of the look-ahead; Ajanthan et al.'s exact algorithm, or a lower learning rate for stale stages, might do better, but that is a new experiment.
- **Worth building only if a stale arm's loss per pair can approach synchronous.** At +0.17, a 30–50% shorter update would need the stale arm to make up 0.17 of loss in its saved time. At the synchronous curve's final slope that takes about 18% more pairs (more as the curve flattens), so the trade might break even. It could not be the clear gain the paper needs.

## Limits

- **Every layer is stale by the same k.** In a real pipeline, staleness grows with a stage's distance from the output; the last decoder layer would be almost current. This screen is pessimistic for that reason.
- **Adam at experiment 67's rate.** Asynchronous methods usually lower the learning rate or warm up; neither was tried.
- **Two seeds, one budget.**
- **Wall time not measured;** the distributed stage (coordinator `YGARATY_ASYNC_AHEAD`, owner step) was not built.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_layer_variants
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
COMMON="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072 --pairs 32768 --eval-every 4096 --optimizer adam --learning-rate 0.0005 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
for seed in 2026 2027; do
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --output runs/experiments/WS-6/sync-$seed
  for k in 1 2; do
    .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --stale-layers $k --output runs/experiments/WS-6/k$k-$seed
    .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --stale-layers $k --delay-correction nesterov --output runs/experiments/WS-6/k${k}n-$seed
  done
done
```
