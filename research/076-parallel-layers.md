# Parallel layers 2–3 on both sides remove 4 of 12 hops for 0.025 of held-out loss

| | |
|---|---|
| **Conclusion** | **Works:** parallel layers 2–3 on both sides remove 4 of 12 hops, costing 0.025 loss (measured learning, two seeds). |
| **Status** | executed |
| **ID** | M-7 (experiment 76) |
| **Category** | training |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-03 |
| **Path** | PyTorch reference (`batch_learning`) of the 30.3M translator (512/2048, 8 heads, 3+3 layers), experiment 67's Adam recipe, 32,768 pairs, seeds 2026 and 2027, opus100-en-pt. Arms: dense D, decoder layers 2–3 parallel (Pd), both sides (Ped), constant layer drop 0.2 (LD0.2), progressive layer drop to 0.25 (PLD) |
| **Result** | Mean held-out loss at 32,768 pairs against dense 5.717: Pd +0.033 removing 2 hops, **Ped +0.025 removing 4 hops**, both candidates (gate ≤0.05 and ≥1 hop). LD0.2 +0.178 (1.45 hops) and PLD +0.089 (0.97 hops) fail. Pool timing not measured: at L5's 230 ms per hop, Ped's 4 hops would be about 0.9 s of experiment 61's 4.64 s update, against about 3% more pairs to make up its loss (estimate). Free; about 1 hour of the 3-hour queue |
| **Evidence** | [`evidence/training/m7-parallel-layers-layer-drop-2026-11-03.json`](evidence/training/m7-parallel-layers-layer-drop-2026-11-03.json) (curves, chrF, hops per run, gate summary); raw runs, `queue.sh` and `evidence.py` in `runs/experiments/queue-20261103-185345/` (local, not in git). Code at 524853b: `ygaraty/experiments/layer_variants.py`, `batch_learning --parallel-layers/--layer-drop/--layer-drop-schedule`; `tests/test_layer_variants.py` |

## Question

Each owned layer adds a forward hop and a backward hop to the update's chain, and each hop costs about 230 ms (L5), whatever its size. Two model changes remove hops without breaking float64 parity, as long as the reference makes the same change with the same seed:

- **Parallel layers.** Layers 2 and 3 of a side both read layer 1's output, and their residual updates are added: x₃ = x₁ + f₂(x₁) + f₃(x₁). "Transformer Layers as Painters" (2024) finds that the middle layers of pretrained models tolerate running in parallel; here the model is trained that way from the start.
- **Layer dropping.** Each update skips some layers, with a seeded mask shared by the reference and the pool (LayerDrop, Fan et al. 2019; Progressive Layer Dropping, Zhang & He 2020). A skipped layer's tasks could be pruned from the update's graph (experiment 54), so the update gets shorter.

**At 32,768 pairs, which of these keeps held-out loss within 0.05 of the dense model, and how many hops does it remove per update?**

## Prediction and gate (frozen before any run)

**Prediction:** Pd costs 0.02–0.05, Ped 0.05–0.1, removing 2 and 4 hops. LD0.2 costs 0.03–0.08 and removes about 1.6 hops. The progressive schedule costs less than the constant rate.

**Gate:** an arm counts as a candidate if its mean held-out loss over seeds 2026 and 2027 at 32,768 pairs is within **0.05** of dense D, while removing at least **1 hop per update** on average. Hops are counted from the arm's layer structure and the masks actually drawn (`layer_hops`): 2 per live group of layers per side, 12 for the dense 3+3 model, not yet from the distributed task graph. **Deviation, decided before running:** no expected-depth rescaling for dropped layers; a skipped pre-norm layer passes its input through, and evaluation uses full depth.

## Protocol

- `VariantTranslator` runs the encoder and decoder layer by layer. With nothing parallel and nothing dropped it reproduces `TransformerTranslator` exactly (tested in train and eval modes). Parallel layers compute x + Σ(fᵢ(x) − x). The drop mask comes from `default_rng([seed, 707, update])`, never drops the first layer per side, and applies only in training.
- One queue of 22 runs on one commit, run one after another under watch.py. D is the shared baseline (also M-5's R and WS-6's synchronous arm). It reproduces experiment 67's Adam arm exactly: 5.717.
- Common flags: `--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072 --pairs 32768 --eval-every 4096 --optimizer adam --learning-rate 0.0005 --hint-index runs/experiments/055-hints/index --eval-dedup --decode`.

## Log

- 2026-11-03: Added, from a review of modeling ideas that shorten the chain of hops. Gate frozen with the user's yes.
- 2026-11-03: Queue started at aadfa58+dirty; the new code was committed unchanged as 524853b while it ran. watch.py with a 600 s stall, a 1,800 s busy stall and a 6-hour maximum. A tiny-model rehearsal of every mode passed first.
- 2026-11-03: Queue `finished`, 22 of 22 runs exited 0 (18:53–22:12). The M-7 arms ran 20:03–21:03.

## Results

Held-out loss at 32,768 pairs (1,024 validation pairs) and hops per update:

| Arm | Seed 2026 | Seed 2027 | Mean | Against D | Hops removed | Gate |
|---|---|---|---|---|---|---|
| D (dense) | 5.626 | 5.807 | 5.717 | — | 0 | — |
| Pd (dec 2–3) | 5.648 | 5.851 | 5.749 | +0.033 | 2 | **candidate** |
| **Ped (enc and dec 2–3)** | 5.630 | 5.853 | 5.742 | **+0.025** | **4** | **candidate** |
| LD0.2 | 5.863 | 5.926 | 5.894 | +0.178 | 1.45 | fails |
| PLD (0 → 0.25) | 5.718 | 5.894 | 5.806 | +0.089 | 0.97 | fails |

Greedy chrF on the deduplicated split was D 5.80, Pd 5.59, Ped 6.72, LD0.2 3.88 and PLD 4.54. At this loss the decoder still loops on repeated words in every arm, so chrF ranks the arms only loosely.

**Break-even (estimate, not measured).** Near 32,768 pairs, D's loss falls about 0.116 per 4,096 pairs. Ped's 0.025 is therefore about 880 pairs, or about 3% more pairs to the same loss. Its 4 hops at L5's 230 ms are about 0.9 s, roughly 20% of experiment 61's 4.64 s update, if the removed hops come off the critical path in full.

## What it taught

- **Parallel layers cost less than predicted, and parallelizing both sides costs no more than one.** Ped's +0.025 is below the predicted 0.05–0.1 and no worse than Pd's +0.033. Per seed, Ped is +0.004 and +0.046 and Pd +0.022 and +0.044. The difference between them is inside the noise.
- **Layer dropping does not pay at 3 layers per side.** Dropping a third of the droppable layers costs 0.18 at equal pairs. The progressive schedule halves that cost but also drops less, removing under 1 hop. With so few layers, each one carries too much to skip.
- **The shared baseline reproduced experiment 67's Adam arm to the digit**, so this queue's arms compare with experiments 67–70.

## Limits

- **Two seeds.** The baseline's seeds differ by 0.18, more than every gate margin. The comparison is paired, with the same seed and data order per arm, but the candidates' margins are small.
- **Hops are counted from the model's structure, not timed.** Parallel layers still do the same compute: two layer tasks run side by side on two owners, or one owner runs both. Whether the pool's update gets about 20% shorter depends on how the planner places them.
- **One budget (32,768 pairs).** A gap can grow or shrink with longer training.
- **No expected-depth rescaling for layer drop,** as frozen. LayerDrop's rescaling might reduce LD0.2's cost.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_layer_variants
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
COMMON="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072 --pairs 32768 --eval-every 4096 --optimizer adam --learning-rate 0.0005 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
for seed in 2026 2027; do
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --output runs/experiments/M-7/D-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --parallel-layers dec:2-3 --output runs/experiments/M-7/Pd-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --parallel-layers enc:2-3,dec:2-3 --output runs/experiments/M-7/Ped-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --layer-drop 0.2 --output runs/experiments/M-7/LD02-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --layer-drop 0.25 --layer-drop-schedule progressive --output runs/experiments/M-7/PLD-$seed
done
```

About 6 minutes of training per run on the workstation's 16 threads, plus about 3 minutes of greedy decoding.
