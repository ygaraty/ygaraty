# Batch 128 against 64 at equal pairs: twice the pairs per update, little more progress per update

| | |
|---|---|
| **Date** | 2026-10-27 |
| **Path** | Experiment 34's design (owned layers and vocabulary, bf16 activations, inputs pushed with claims) on loopback: 8 NumPy workers, the relay, PostgreSQL; the 30.3M Transformer on OPUS-100 en-pt |
| **Result** | At equal pairs seen (6,400 in all), batch 128 trails batch 64 by 0.23 in loss over the last 1,280 pairs at a learning rate of 5e-4, and by 0.24 at 1e-3: its loss after 50 updates is batch 64's after about 50 updates. This early in training, doubling the batch buys almost nothing per update, and on home links, where an update of 128 pairs takes about 1.46 times as long, it trains slower. Along the way, a worker memory bug was fixed: 8 workers peaked at 18.3 GB, and at twice the batch now peak at 5.6 GB. |
| **Evidence** | Logs in the session's scratch directory (`batch/*`), not synced |

## Question

Experiment 33's simulator puts a batch of 128 pairs at 42.9 s per update on the home links against 29.3 s for 64: 1.37 times the pairs per second, since the 25 hops' round trips are shared by twice the pairs. It would help reach the goal of experiments 31 to 34 only if a pair in a batch of 128 teaches the model about as much as one in a batch of 64. Does it?

## Setup

The training pairs come from a sequential cursor, so update k of batch 128 trains on exactly the pairs of updates 2k−1 and 2k of batch 64. Each update's loss is computed on its own pairs before its step, pairs the model has not seen: an estimate of the held-out loss. Update k of batch 128 is compared with update 2k−1 of batch 64, both after training on the same 128(k−1) pairs. Batch 64 ran 100 updates and batch 128 ran 50 (6,400 pairs, 0.6% of the training split), Adam, no dropout, one run each. The arithmetic is deterministic: batch 64's first 50 updates reproduced experiment 34's bf16 curve to every printed digit.

## Results

Mean loss per window of batch-128 updates, against batch 64 at the same pairs:

| Batch-128 updates (pairs seen) | Batch 64, lr 5e-4 | Batch 128, lr 5e-4 | Batch 128, lr 1e-3 |
|---|---:|---:|---:|
| 2–10 (128–1,152) | 7.651 | 8.207 (+0.56) | 8.078 (+0.43) |
| 11–20 (1,280–2,432) | 7.056 | 7.218 (+0.16) | 7.302 (+0.25) |
| 21–30 (2,560–3,712) | 6.937 | 7.101 (+0.16) | 7.114 (+0.18) |
| 31–40 (3,840–4,992) | 6.819 | 7.086 (+0.27) | 7.116 (+0.30) |
| 41–50 (5,120–6,272) | 6.672 | 6.932 (+0.26) | 6.946 (+0.27) |

Batch 128 closes most of its early gap by 1,280 pairs and then holds 0.2 to 0.3 behind. Over its last ten updates its mean loss (6.93) is batch 64's over its updates 41 to 59 (6.94): about equal progress per update, not per pair. Doubling the learning rate to 1e-3 changes the curve by 0.005 on average. On loopback, an update of 128 took 21.2 s against 11.9 for 64 (10.6 against 11.9 s per 64 pairs).

## What it taught

- **This early, progress is per update, not per pair.** How much a bigger batch helps depends on how noisy the gradient is next to its signal, and early in training the signal dominates, so averaging over more pairs adds little. That changes as the loss falls (the critical batch size of large-batch training studies grows during training). Batch size is a lever for later in a run, raised in steps, not for the start.
- **For the goal, the levers stay per update.** 16 workers took the measured update to 21.8 s (experiment 34, over pods) without changing what the model learns; fewer bytes per hop (8-bit activations) is next.

## A worker memory bug, found on the way

The first attempt stopped at update 15 with 672 MB of the machine's 28 GB available: the 8 workers held 16 to 22 GB, up to 5.2 GB each. An embedding task reads 512 one-row regions of one relay block; the worker decoded the whole 1.8 MB block for each region and returned a view of the row, so each 4 KB row kept its own decoded copy alive, about 1 GB per embedding task. `assemble_region` now decodes each block once per fetch and copies a region smaller than its block; values are unchanged and a test checks both. The runs above also set `MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2` for every process, so the allocator returns freed NumPy arrays to the system (with the old code, the workers' peak over 4 updates went from 22.2 GB to 15.6 GB); the worker does not set this itself yet.

| | Peak, 8 workers | Largest worker | Lowest available |
|---|---:|---:|---:|
| Batch 64, old code | 18.3 GB | 5.2 GB | 3.4 GB |
| Batch 128, fixed (lr 5e-4; lr 1e-3 the same) | 5.6 GB | 1.6 GB | 15.1 GB |

## Limits

6,400 pairs, the first 0.6% of the data; one run per setting, two learning rates, no warmup; the losses are measured on each update's own fresh pairs, not on a held-out set. It says nothing about batch size later in training, where it matters more.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite YGARATY_STORE_URL=postgresql:///ygaraty_exp24
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2
./relay-server -addr 127.0.0.1:8113 -token q -max-mib 8192 &
P="ygaraty experiment http_profile --recipe transformer --learning-rate 0.0005 --workers-kind python --tiles 512 --max-tasks 16 --repeats 1 \
  --relay-url http://127.0.0.1:8113 --relay-token q --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 --layer-task-mib 512 \
  --output-task-mib 1024 --max-task-mib 512 --workers 8 --checkpoint-every 1000 --float32-wire --activation-wire bf16 --owned-layers \
  --owned-vocab 8 --pushed-inputs"
$P --batch 64 --steps 100 --output runs/exp35/b64      # about 21 minutes
$P --batch 128 --steps 50 --output runs/exp35/b128     # about 18 minutes; add --learning-rate 0.001 for the third column
```
