# Batches by length, and nanochat training time as the KPI: 1.4 times faster to the same loss

| | |
|---|---|
| **Date** | 2026-10-27 |
| **Path** | Experiment 36's design (16 NumPy workers, owned layers and vocabulary, bf16 activations, pushed inputs, eager claims) priced by the update simulator (experiment 33) on home links (100 Mb/s down, 20 Mb/s up, 50 ms); learning measured with the PyTorch reference on this machine's CPU, on the batches the job serves |
| **Result** | A new main KPI: how long nanochat's training (about 4e19 FLOPs) would take at the useful rate a run measures. Over RunPod, experiment 36 stands at about 190 years with batches of 64 and 130 with 128. 79% of the tokens the workers compute are padding, because a batch of 64 pairs is padded to its longest sentence. Batches sorted by length within windows of 64 batches bring that to 12%, and batches filled to a token budget let one update carry four times the real tokens at today's cost. In simulation, that is 143 years down to 34 to 78. But learning does not follow the FLOPs. At equal pairs, every sorted batch learns less per pair than corpus order (held-out loss 5.37 against 5.58 to 6.24 after 32,768 pairs). Converted to time, the best setting, a budget of 3,072 padded tokens, reaches corpus order's loss 1.44 times sooner, not the 3 times the FLOPs promise. Measured over DigitalOcean with 16 workers on emulated home links: 14.4 s per update in corpus order, 10.1 s with 3,072-token batches (2.9 times the real tokens per second; 54 years by FLOPs, about 108 adjusted for learning), 14.7 s with 6,144; the simulator ran 7 to 10% low on all three. |
| **Evidence** | Learning curves in `runs/experiments/037-learning/` (not synced); simulator runs in this write-up's commands |

## Question

The goal so far has been seconds per update. Seconds per update hide two things: how many tokens an update trains on, and how much of what the workers compute is padding. The user set a new main KPI: how long would nanochat's training take on this system? And which change moves it most?

## The KPI

nanochat's speedrun trains a GPT-2-grade model with about 4e19 FLOPs, in about 1.65 to 2 hours on one 8xH100 node. A run's useful rate is 6 FLOPs per parameter per real token, counting the parameters each token passes through:

- a source token: the encoder, 9.46M;
- a target token: the decoder and the output layer, 12.61M + 8.19M.

Padding counts for nothing. The KPI is 4e19 divided by that rate. It assumes the swarm keeps its measured rate on a model that size. A wider model does more arithmetic per byte it sends, so for a link-bound swarm the assumption is pessimistic. `ygaraty experiment kpi RUN...` computes it from a run's `updates.jsonl`. Jobs now record each batch's real and padded tokens, and `distributed_translation`, `http_profile` and the simulator print the KPI.

The earlier runs, recounted from their example IDs (steady updates; the first and the publishing update left out):

| Run (home links) | Workers | s/update | Real tokens/s | Useful GFLOP/s | nanochat |
|---|---:|---:|---:|---:|---:|
| Experiment 31's design | 8 | 277 | 4.7 | 0.43 | ~2,960 years |
| Experiment 34, RunPod | 8 | 33.3 | 39 | 3.56 | 356 years |
| Experiment 34, RunPod | 16 | 21.9 | 59 | 5.35 | 237 years |
| Experiment 36, RunPod, batch 64 | 16 | 17.6 | 74 | 6.71 | 189 years |
| Experiment 36, RunPod, batch 128 | 16 | 23.7 | 106 | 9.58 | 132 years |

(Means over the steady updates; experiment 36 quotes medians, 17.5 and 25.0 s. The pods train their own tokenizer, so recounts with this machine's can differ by a few tokens.)

## What was built

- **Batches sorted by length** (`YGARATY_LENGTH_WINDOW=N`, `--length-window`). The job reads N batches' worth of pairs from the cursor. It sorts them by their longer side, then their total, cuts them into batches, and serves those in an order seeded by the window's start, so updates don't run short to long. The cursor stays at the window's start until its last batch completes. Each batch's source records its place in the window, so a resume re-reads the window and serves the batch after the one the checkpoint completed (tested).
- **Batches by tokens** (`YGARATY_BATCH_TOKENS=T`, with a window). Sorted pairs fill a batch up to T padded tokens, so every update weighs about as many tokens. Each batch splits into the bands `batch_size/band` gives (8), so the plan keeps today's shape whatever the pair count.
- **The learning check** (`ygaraty experiment batch_learning`). It trains the PyTorch reference Transformer on the job's batches with the job's recipe: production's block-seeded initializer, Adam (5e-4, betas .9/.98), label smoothing 0.1. Every 4,096 pairs it measures the token cross-entropy on 1,024 held-out validation pairs. The reference and the job's tape give the same loss on the same weights and batch (19.2715). The job's own first-update loss, about 18.9, came from weights drawn differently from the same distribution. Cause, found on 28 October: the job seeds each block by the tape's name for the parameter (`encoder_layers_0_...`), `production_init` by the dotted name; with the tape's name it gives 18.8733, the job's loss. Every run here shares one initialization, so the comparisons stand.

## Results

**Padding.** Over the first 40 batches of the cursor, a batch of 64 pairs holds 1,308 real tokens and 6,128 padded ones (21% real). Sorting within windows (share of tokens that are real):

| Batch | Corpus order | Window 4 | Window 16 | Window 64 |
|---:|---:|---:|---:|---:|
| 64 | 26% | 56% | 80% | 88% |
| 256 | 22% | 54% | 79% | 88% |

**Throughput, simulated** (16 workers, 12 to 16 batches each; the simulator runs 12% low against experiment 36's emulated run, and 20 to 27% low against its RunPod mean):

| Batches | s/update | Real tokens/s | Real share | nanochat |
|---|---:|---:|---:|---:|
| 64, corpus order | 12.9 | 98 | 21% | 143 years |
| 128, corpus order | 20.2 | 129 | 20% | 108 years |
| 256, corpus order | 32.8 | 171 | 19% | 82 years |
| 64, sorted | 6.9 | 181 | 88% | 78 years |
| 128, sorted | 9.8 | 289 | 88% | 49 years |
| 256, sorted | 14.7 | 416 | 88% | 34 years |
| 1,536 tokens, sorted | 7.0 | 194 | 90% | 72 years |
| 3,072 tokens, sorted | 9.2 | 290 | 88% | 48 years |
| 6,144 tokens, sorted | 13.6 | 386 | 86% | 36 years |
| 12,288 tokens, sorted | 20.5 | 418 | 79% | 34 years |

**Learning, at equal pairs** (held-out loss; every run from the same initial weights and the same 32,768 pairs):

| Pairs seen | 64, corpus | 64, sorted | 128, sorted | 256, sorted | 1,536 tok | 3,072 tok | 6,144 tok |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 6.653 | 6.813 | 6.845 | 7.138 | 6.760 | 6.820 | 7.050 |
| 16,384 | 5.847 | 6.070 | 6.315 | 6.630 | 5.960 | 6.104 | 6.480 |
| 32,768 | **5.368** | 5.661 | 5.919 | 6.242 | 5.579 | 5.659 | 6.050 |
| Updates | 512 | 512 | 256 | 128 | 661 | 333 | 175 |

**Learning, at equal time.** Seconds to reach 5.66, corpus order's loss after about 21,900 pairs, at the simulated seconds per update:

| Batches | Seconds to 5.66 | Against corpus order |
|---|---:|---:|
| 64, corpus order | 4,410 | 1.00 |
| 1,536 tokens, sorted | ~4,050 | 1.09 |
| 64, sorted | 3,530 | 1.25 |
| **3,072 tokens, sorted** | **3,060** | **1.44** |
| 128 and 256 sorted, 6,144 tokens | not reached by 32,768 pairs | at best equal: 6,144 tokens ties corpus order at its 2,380 s |

## Over RunPod: not measured

The run was set up but never trained.

- **US:** every data center refused new CPU pods ("no longer any instances available").
- **Sweden:** pods never answered SSH.
- **Romania:** the 16-vCPU server forwarded only its SSH port. Its coordinator and relay ports timed out from outside, though two probe pods on other Romanian machines forwarded theirs. Its replacement had no public address after 10 minutes.
- **Czechia:** the 16 workers there were set up, but pods cannot reach one another inside a data center, even across machines.

All pods were deleted; the attempt cost about $1.2.

## Over DigitalOcean: measured

RunPod could not provide a working server, so the runs moved to DigitalOcean (`pods.py --provider digitalocean`,
commit 689f138). One server (g-8vcpu-32gb: 8 dedicated vCPUs, 32 GB) and 16 workers (c-2: 2 dedicated vCPUs,
4 GB each) in lon1, on one private network. Each worker reaches the coordinator and the relay through its own
latency proxy: a measured 0.2 to 0.3 ms plus 49.7 to 49.8 ms added makes 50 ms, at 100 Mb/s down and 20 Mb/s up.
Everything else is experiment 36's design. Each run trained 22 updates; the table counts updates 2 to 21 (the
first loads the owners' state, the last publishes it, 32 to 46 s).

| Batches | s/update | Real tokens/s | Real share | nanochat (FLOPs) | Simulated |
|---|---:|---:|---:|---:|---:|
| 64 pairs, corpus order | 14.4 | 89 | 21% | 158 years | 12.9 s |
| **3,072 tokens, sorted** | **10.1** | **260** | 88% | **54 years** | 9.2 s |
| 6,144 tokens, sorted | 14.7 | 342 | 86% | 41 years | 13.6 s |

- **Corpus order runs faster here than over RunPod** (14.4 s against 17.6), on dedicated vCPUs and a
  sub-millisecond private network under the same emulated links.
- **The simulator ran 7 to 10% low on all three**, closer than its 12 to 27% before.
- **Adjusted for learning**: at these measured times, the 3,072-token batches reach the learning check's
  held-out loss of 5.66 in 333 × 10.1 = 3,363 s, against corpus order's 342 × 14.4 = 4,925 s: 1.46 times
  sooner, so about 158 / 1.46 ≈ 108 years. The 6,144-token batches only tie corpus order at equal time.
- Cost: 29 minutes of 17 Droplets, about $0.66, plus $0.02 for two one-Droplet probes. Logs in
  `runs/experiments/037-pods/do-16w-*` (not synced).

## What it taught

- **A FLOP KPI needs a learning check beside it.** By FLOPs, sorted batches of 256 look four times better than corpus order (34 against 143 years). By time to the same loss, they are no better. Every sorted batch here learns less per pair, and the bigger the batch the less, as experiment 35 found for corpus order. Throughput gains count only where loss against time improves.
- **Sorting costs learning even at equal tokens per update.** A budget of 1,536 tokens trains as many real tokens per update as corpus order, and over more updates (661 against 512), yet ends 0.21 behind. Batches of one length give the gradient a narrower view than mixed ones. Early in training that costs more than the better token weighting gains.
- **The best setting so far: sorted batches of 3,072 padded tokens.** 9.2 s per update simulated, 1.44 times faster than today to the same loss. Adjusted for learning, that is experiment 36's measured 189 years divided by 1.44: about 130 years.

## Limits

- Throughput is simulated, not measured. The simulator runs low by 12 to 27% here, though its rankings have held so far.
- Learning covers 32,768 pairs, 3% of the split, the very start of training, where bigger batches help least. One seed, one learning rate.
- The PyTorch reference trains in float32 on this machine's CPU. The job trains in float64 with bf16 activations on the wire, so the curves agree in shape, not to the bit.

## Reproduce

Over DigitalOcean (a token in `YGARATY_DIGITALOCEAN_DEV`):

```bash
ygaraty experiment pods up exp37d --provider digitalocean --location lon1 --server g-8vcpu-32gb \
  --workers 16 --worker-size c-2 --minutes 90 --max-hourly 1.5
# then scripts/remote.sh bootstrap on the server, worker-setup on each worker, and per run on the server:
#   YGARATY_WORKERS=16 YGARATY_BATCH=64 [YGARATY_LENGTH_WINDOW=64 YGARATY_BATCH_TOKENS=3072] ./scripts/remote.sh serve34 22
# with YGARATY_PUBLIC_HOST and the ports from `pods addr exp37d server 8000` / `8101` (private addresses)
ygaraty experiment pods down exp37d
```

Locally:

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
ygaraty experiment kpi runs/experiments/034-pods/runpod-16w-eager-b64-20261027/server      # the KPI of a run
S="--workers 16 --rtt-ms 50 --down-mbps 100 --up-mbps 20 --owned-layers --owned-vocab 8 --float32-wire --activation-wire bf16 \
   --band 8 --push-inputs --eager-claims --tile 512 --layer-task-mib 1024 --output-task-mib 1024 --max-task-mib 1024"
ygaraty experiment simulate run $S --batch 64 --batches 16                                    # corpus order
ygaraty experiment simulate run $S --batch 64 --length-window 64 --batch-tokens 3072 --batches 12
ygaraty experiment batch_learning --batch 64 --pairs 32768 --output runs/experiments/037-learning/b64-corpus   # about 13 minutes
ygaraty experiment batch_learning --batch 64 --length-window 64 --batch-tokens 3072 --pairs 32768 \
  --output runs/experiments/037-learning/t3072-sorted                                         # about 5 minutes
```
