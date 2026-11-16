# Wider models learn no faster here: width 512 reaches the same loss 1.5 to 2.2 times sooner

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | The PyTorch reference (`ygaraty experiment batch_learning`) on this workstation, from a job's initial weights (commit 2c120a3), on the batches a job serves: sorted 3,072-token batches, the first 32,768 usable pairs of OPUS-100 en-pt, 16,000-piece vocabulary; 3 + 3 layers at widths 512, 768 and 1,024 |
| **Result** | Experiment 38 priced a wider model as the largest lever on the nanochat KPI (49 → 33 years at width 768, 25 at 1,024), since arithmetic grows with the width squared and bytes with the width. The learning check says otherwise this early in training. After 32,768 pairs held-out loss is 5.626 at width 512 (30.3M parameters), 5.676 at 768 (62M) and 5.784 at 1,024 (105M): wider models learn slightly less per pair, and each update is slower (9.3, 12.7, 16.3 s simulated on 16 home-link workers). To a held-out 5.8, width 512 takes 41 minutes, 768 takes 60 and 1,024 takes 89. Halving the learning rate for the wider models makes them worse (5.820, 5.825). The remote run of a wider model is not worth doing now. |
| **Evidence** | Curves in `runs/experiments/041-width/` (not synced) |

## Question

A wider model does more useful FLOPs per byte, so the KPI improves at once. It only trains faster if it reaches a given held-out loss in few enough updates to pay for its slower ones. Does it, before spending a remote run?

## Results

| Width · feed-forward · heads | Parameters | Learning rate | Held-out at 16,384 pairs | at 32,768 | Simulated s/update | Time to 5.8 |
|---|---:|---:|---:|---:|---:|---:|
| **512 · 2,048 · 8** | 30.3M | 5e-4 | **6.126** | **5.626** | 9.3 | **40.8 min** |
| 768 · 3,072 · 12 | 62.0M | 5e-4 | 6.150 | 5.676 | 12.7 | 59.8 min |
| 768 · 3,072 · 12 | 62.0M | 2.5e-4 | 6.310 | 5.820 | 12.7 | not reached |
| 1,024 · 4,096 · 16 | 104.6M | 5e-4 | 6.292 | 5.784 | 16.3 | 89.0 min |
| 1,024 · 4,096 · 16 | 104.6M | 2.5e-4 | 6.353 | 5.825 | 16.3 | not reached |

Each run is 333 updates of the same sorted batches (89% of tokens real). Times are updates × experiment 38's simulated seconds per update for 16 workers on home links (the simulator runs 7 to 26% fast, the same for every width to first order).

## What it taught

- **The FLOP KPI can be gamed by model size as well as by batch size.** A wider model raises useful FLOP/s without training faster. As with batches (experiment 37), every lever has to be judged by the time to the same held-out loss.
- **This dataset is too small for wider models.** One epoch of the training split is about 18.6M real tokens (908,831 pairs of about 20 tokens). By the Chinchilla rule of about 20 tokens per parameter, a compute-optimal model for one epoch has about 1M parameters: the tiny model's size. The 30.3M model is already about 30 times larger. Wider models can pay only with much more data: a bigger corpus, not a bigger model, is what would let the system spend more FLOPs usefully.
- **A lower learning rate does not rescue the wider models.** A higher one was not tried.

## Limits

32,768 pairs, 3.6% of an epoch: the very start of training, where smaller models are known to lead. One seed. Learning rates 5e-4 and 2.5e-4 only. Simulated, not measured, seconds per update for the wider models. The Chinchilla rule was measured on decoder-only language models, not this encoder-decoder translator; it is a guide to the order of magnitude.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
for w in "512 2048 8" "768 3072 12" "1024 4096 16"; do set -- $w
  ygaraty experiment batch_learning --embedding $1 --hidden $2 --heads $3 --layers 3 --version 1m-seed2026 --batch 64 \
    --length-window 64 --batch-tokens 3072 --learning-rate 0.0005 --pairs 32768 --eval-every 4096 --output runs/experiments/041-width/w$1
done      # about 5, 8 and 14 minutes
```
