# N-gram hints: the translator reaches the same held-out loss with about 35% fewer pairs

| | |
|---|---|
| **Date** | 2026-10-30 (plan, built and run the same day; training-time check 2026-10-30) |
| **Path** | Local, on the workstation (the Ryzen 9 PRO 6950H of experiments 41 and 51). The PyTorch reference of `ygaraty experiment batch_learning` with a hint input added; no coordinator, workers, relay or network. Not on the simulation-only machine: its `AGENTS.md` forbids training jobs |
| **Result** | **A lever, by the frozen rule.** Held-out loss at 32,768 pairs, mean of seeds 2026 and 2027: no hints 5.717, target hints 5.320 (−0.40), source-filtered hints **5.291 (−0.43**, −0.42 and −0.43 per seed), against the required 0.23; chrF 5.8 → 6.9. The hinted runs reach arm N's final loss after 63 to 68% of the pairs. The validation split has no training duplicates (0% of targets, 2.7% of 8-grams), so the deduplicated and full splits are the same 1,024 pairs and the gain is not memorization. With hints off, the hinted model equals arm N (5.72–5.73). Hints add 6% to training time per update; the 16-shard index equals the single build exactly |
| **Evidence** | `runs/experiments/055-hints/`: `audit.json`, `index/index.json`, `{none,target,source}-{2026,2027}/{summary.json,curve.jsonl,decode.jsonl}`, `run.log`; the training-time check in `speed/` (local, not in git); `tests/test_ngram_hints.py` |

## Question

Every change in experiments 31 to 42 made an update cheaper. None made an update teach the model more, and experiments 35, 37 and 41 found that the cheaper settings (bigger batches, sorted batches, wider models) teach less per pair. The paper's "decades" result is useful FLOPs times learning per pair, so only the second factor is left to move.

[N-gram hinted LM](https://brunborg.com/n-gram-hinted-lm) adds, at each position, the embedding of the token that followed the longest matching n-gram in the training corpus to the residual stream, scaled by a learned trust that depends on the match length, its count and its purity. On FineWeb it reached the speedrun's target loss with far fewer tokens. **Does the same idea lower the 30.3M translator's held-out loss after 32,768 pairs by more than batch 64 beats batch 128 (0.23, experiment 35), on a held-out split that has had its duplicates removed?**

## Why this matters for a volunteer pool

- **It changes learning per pair, the factor the network work cannot reach.** The update itself, its bytes and its parity check stay as they are.
- **Building the hints is the factorization's shape, not the Transformer's.** Counting n-grams over corpus shards is stateless and wide, and the shards merge by summing counts: CPU and memory work for weak or idle workers, with no chain.
- **It costs almost nothing on the wire.** Hints for training tokens depend only on the corpus, so they are computed once and stored with the dataset: about 3 bytes per token (a 2-byte hint id and a 1-byte bucket), against 1,024 bytes per token for a bf16 activation at width 512. At run time the hint embeddings are rows of the tied table, which the vocabulary-slice owners already serve per band.

None of this is tested here. This experiment decides only whether the learning gain exists; if it does not, nothing else is worth building.

## What exists

- `ygaraty/experiments/batch_learning.py`: the PyTorch reference used by experiments 35, 37 and 41, trained on the batches a job would serve, with held-out loss every `--eval-every` pairs on `--eval-pairs` validation pairs, `--save` to keep the model.
- Dataset `opus100-en-pt`, version `1m-seed2026`: about 908,831 usable training pairs, about 18.6M real tokens per epoch (experiment 41), 16,000-piece SentencePiece vocabulary shared by both languages, sequences cut at 64 pieces.
- Experiment 41's width-512 arm as the baseline: held-out 6.126 at 16,384 pairs and 5.626 at 32,768, sorted 3,072-token batches, learning rate 5e-4, about 5 minutes.

## Design

### The hints

For each target position t (the decoder input is the target shifted right; the hint predicts target token t):

- **Arm N, no hints.** Experiment 41's setup, re-run in this harness so the seeds and code match.
- **Arm T, target n-gram hints.** Look up the previous n target tokens for n = 8, 5, 3, longest first. The first n that matches gives the hint: the most frequent next token after that n-gram in the training targets, its count bucket (1, 2–3, 4–7, 8–31, 32–255, ≥256) and its purity bucket (the share of that next token among all continuations: <0.5, 0.5–0.9, ≥0.9). No match: no hint (a learned "none" embedding). Orders 3/5/8 instead of the page's 6/10/20 because the sentences are at most 64 pieces.
- **Arm S, source-filtered hints.** As T, but take the top 8 continuations of the matched n-gram and pick the one with the highest count × max over source tokens s of Dice(s, candidate), the Dice coefficient computed from sentence-level co-occurrence in the training pairs. This is the cheapest way to make the hint depend on the English sentence, which a translation needs and a language model does not.

**Leave-one-out for training hints.** Each training pair's own n-grams are subtracted from the counts before its hints are looked up; otherwise every training token's longest n-gram matches itself and the model learns to copy the hint. Validation hints use the full counts. (The page reports the same over-reliance problem and works around it with a smaller corpus during training; leave-one-out is the standard alternative.)

### Where the hint enters the model

As the page does, in three places, each with its own learned scalar: added to the decoder's input embedding; added to the residual stream after decoder layer 1 (the middle of three); added to D2's output before the tied output layer. The hint vector is `trust_scale[order, count, purity] * E[hint_id] + trust_embedding[order, count, purity]`, E the tied embedding table. That is 3 × 6 × 3 = 54 trust pairs plus a "none" entry: about 28K new parameters on top of 30.3M. The encoder is unchanged.

### Duplicates first (stage 0)

OPUS-100 is known to contain repeated short segments, and the page's largest gains come from passages that recur in training. Before any training:

- the share of validation pairs whose exact target sentence appears in the training targets, and the share whose exact pair appears;
- the share of validation target 8-grams that occur in the training targets;
- a **deduplicated held-out split**: the validation pairs whose target sentence does not appear in training, at least 1,024 of them.

Every loss below is reported on both the full and the deduplicated held-out split.

### Building the index (stage 1), and its volunteer-shaped gate

The training targets' n-grams of orders 3, 5 and 8 are hashed to `uint64` and counted with `numpy.unique` (no Python dictionaries; about 9M target tokens, about 27M n-grams before deduplication), with the next token kept per n-gram. The same index is also built as 16 shards of the training split, merged by summing counts. **Gate: the merged index equals the single build exactly** (same keys, counts and argmax continuations). This is the property a pool would rely on, and it costs nothing to check here. Record build time, peak memory and index size per shard and merged.

## Frozen before the runs

- **Training:** experiment 41's width-512 settings: 3 + 3 layers, embedding 512, feed-forward 2,048, 8 heads, batch 64, length window 64, 3,072-token batches, learning rate 5e-4, 32,768 pairs, held-out every 4,096 pairs on 1,024 pairs.
- **Seeds:** 2026 and 2027 for each arm: 3 arms × 2 seeds = 6 runs. Same initial weights and batch order within a seed across arms.
- **Measures, at every evaluation point:**
  - held-out loss with hints (full and deduplicated split);
  - held-out loss of the hinted model **with hints switched off** (the "none" entry everywhere), because the page reports that the backbone gets worse at equal steps;
  - at 32,768 pairs only: greedy-decoded chrF and BLEU on the 1,024 deduplicated pairs, hints computed online from the generated prefix (and from the source for arm S).
- **Counts as a lever** if, averaged over the two seeds, the best hinted arm's deduplicated held-out loss at 32,768 pairs is at least **0.23 below** arm N's, its chrF is not lower, and its gain on the deduplicated split is at least **half** its gain on the full split. A gain on the full split only is reported as memorization of repeated segments, not as a lever.
- **Time to loss:** with the hinted run's curve, the pairs needed to reach arm N's 32,768-pair deduplicated loss. Index build time is reported next to it, not hidden.

## Predictions, made before any run

Guesses, recorded so the result can contradict them:

- Stage 0: 10–25% of validation targets appear verbatim in training (short, formulaic segments).
- Arm T: 0.3–0.6 lower held-out loss at 32,768 pairs on the full split, 0.1–0.25 on the deduplicated split, so it may miss the gate.
- Arm S: 0.05–0.1 better than arm T on the deduplicated split, because it can use the source.
- The hinted model without hints is 0.05–0.15 worse than arm N.
- The 16-shard merge equals the single build exactly; the index fits in under 2 GB.

## Stages

0. **Duplicate audit.** Data only, minutes.
1. **Index build and shard-merge gate.** Minutes.
2. **The six learning runs.** About 5 minutes each by experiment 41's timing, plus hint lookups; under an hour.
3. **Decoding** of the final checkpoints, chrF and BLEU. Minutes.
4. **Write-up** against the frozen rule and the predictions.

## What has to be built

- `ygaraty/experiments/ngram_hints.py`: the hashed index (orders 3/5/8, counts, argmax continuation, purity), leave-one-out lookup for training pairs, full lookup for validation, Dice table for arm S, shard build and merge, the duplicate audit. Unit tests for the merge gate and for leave-one-out on a toy corpus.
- `batch_learning.py`: a `--hints none|target|source` flag; hint ids and buckets carried with each batch; the three injection points and the trust tables in the reference model; a hints-off evaluation; `--decode` for chrF and BLEU (`sacrebleu`, already a common dependency; add if absent).

The Ygaraty tape and kernels are not touched. If the lever holds, the distributed version needs one new owned operation on the slice owners (gather the hint rows, scale, add), and the dataset needs the precomputed hint columns.

## Limits, stated in advance

- 32,768 pairs is 3.6% of an epoch. Hints may help most early (the model has not yet learned frequent phrases) and least later; this does not show the late-training effect.
- Two seeds per arm can only separate differences of about 0.05 in loss (experiment 41's runs differ by about that).
- One dataset with known duplication; OPUS-100 is not a clean benchmark. The deduplicated split removes exact target matches, not paraphrases.
- At inference the model depends on the index. The cost of serving it (in a pool: a distributed lookup) is not measured.
- The comparison is per pair. If hints are kept, the distributed update also needs its seconds per update re-measured; the expected change is near zero (3 bytes per token), but that is a prediction.

## Cost

Free: workstation time only, about 1 to 2 hours in all, no cloud.

## Results

### Stage 0: duplicates

| Split | Pairs | Target in training | Pair in training | Target 8-grams in training |
|---|---|---|---|---|
| Validation, all | 1,966 | 0% | 0% | 2.6% |
| Validation, usable (≤ 63 pieces) | 1,918 | 0% | 0% | 2.7% |
| Held-out (first 1,024 usable) | 1,024 | 0% | 0% | 2.7% |

OPUS-100's validation split already excludes its training sentences, so the deduplicated held-out split is the same 1,024 pairs as the full one, and every "deduplicated" number below equals its full-split twin. The duplicate gate is met trivially; the hints help on sentences the index has never seen whole.

### Stage 1: the index and the shard-merge gate

- **Coverage:** 885,606 usable training pairs and 12.5M target tokens.

  | Order | N-grams | Continuations |
  |---|---|---|
  | 3 | 4.80M | 7.60M |
  | 5 | 7.94M | 8.39M |
  | 8 | 6.52M | 6.55M |

- **Size:** 1.02 GB.
- **Single build:** 135 s, 4.0 GB peak.
- **16 shards:** 6 to 8 s each (103 s in all), 0.54 GB peak each.
- **Merge:** 27 s, 2.85 GB peak.
- **Gate:** the merged index equals the single build exactly (keys, counts and argmax continuations). The whole stage peaked at 6.7 GB of memory because it holds both builds for the comparison.

### Stage 2: learning

Held-out loss (1,024 pairs) by pairs seen:

| Arm, seed | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 | 28,672 | 32,768 | Hints off at 32,768 |
|---|---|---|---|---|---|---|---|---|---|
| N, 2026 | 6.833 | 6.610 | 6.302 | 6.126 | 5.991 | 5.835 | 5.741 | **5.626** | — |
| N, 2027 | 6.857 | 6.635 | 6.450 | 6.278 | 6.127 | 6.000 | 5.923 | **5.807** | — |
| T, 2026 | 6.784 | 6.481 | 6.127 | 5.894 | 5.709 | 5.509 | 5.377 | **5.234** | 5.652 |
| T, 2027 | 6.812 | 6.532 | 6.275 | 6.049 | 5.835 | 5.656 | 5.540 | **5.406** | 5.805 |
| S, 2026 | 6.781 | 6.468 | 6.112 | 5.872 | 5.685 | 5.482 | 5.351 | **5.209** | 5.645 |
| S, 2027 | 6.808 | 6.521 | 6.259 | 6.024 | 5.810 | 5.630 | 5.508 | **5.373** | 5.804 |

- **Arm N reproduces experiment 41:** seed 2026 gives 6.126 at 16,384 pairs and 5.626 at 32,768, exactly.
- **The gap grows at every point:** 0.05 at 4,096 pairs, 0.43 at 32,768 (arm S against arm N, same seed). It has not leveled off at the end of the run.
- **Time to loss** (pairs needed to reach the same seed's arm-N loss at 32,768, linear between points):

  | Arm | Seed 2026 | Seed 2027 |
  |---|---|---|
  | T | 22,200 (68%) | 21,100 (64%) |
  | S | 21,700 (66%) | 20,500 (63%) |

### Stage 3: decoding

Greedy decoding on the 1,024 deduplicated pairs; hints for the generated prefix looked up online.

| Arm | chrF, 2026 | chrF, 2027 | Mean chrF | Mean BLEU |
|---|---|---|---|---|
| N | 6.29 | 5.32 | 5.81 | 0.22 |
| T | 6.82 | 5.59 | 6.21 | 0.40 |
| S | 7.28 | 6.59 | 6.93 | 0.56 |

All of these are near the floor. At 32,768 pairs (3.6% of an epoch) the greedy output often repeats one word to the length limit. Decoding meets the gate's "chrF not lower" condition. It says little more than that.

### Training time

The six runs took 347 s (arm N) and 504 to 508 s (arms T and S) of training. Most of that difference is evaluation, not hints:
- The hinted runs evaluate 1,024 pairs four times per point (with and without hints, on both splits) where arm N evaluates twice. Each evaluation takes about 9 s.
- Hints for every pair are looked up once before training (under 1 s for 34,816 pairs).

A separate check measured training alone: 8,192 pairs, seed 2026, a 128-pair evaluation only at the start and end.

| Arm | Seconds for 76 updates |
|---|---|
| N (twice) | 48.6, 48.6 |
| T | 51.8 (+6.6%) |
| S | 51.6 (+6.2%) |

So the hinted model needs about 65% of the pairs at about 1.06 times the cost per pair: **about 30% less training time to arm N's 32,768-pair loss** on this reference. The index build (stage 1) is extra and one-time.

## Against the frozen rule

| Condition | Required | Measured | |
|---|---|---|---|
| Best hinted arm's deduplicated loss below arm N at 32,768 pairs | ≥ 0.23 | 0.426 (arm S) | passes |
| Its chrF not lower | ≥ 5.81 | 6.93 | passes |
| Deduplicated gain at least half the full-split gain | ≥ 0.5 × 0.426 | 0.426 (same split) | passes, trivially (stage 0) |

**Counts as a lever.**

## Against the predictions

| Prediction | Outcome |
|---|---|
| 10–25% of validation targets appear in training | Wrong: 0% (2.7% of 8-grams) |
| Arm T: 0.3–0.6 lower on the full split | Right: 0.40 |
| Arm T: 0.1–0.25 lower on the deduplicated split, may miss the gate | Wrong: 0.40, passes (the split has no duplicates to remove) |
| Arm S 0.05–0.1 better than arm T on the deduplicated split | Smaller: 0.03 on loss; 0.7 chrF |
| Hinted model with hints off 0.05–0.15 worse than arm N | Wrong: equal (5.729 and 5.724 against 5.717) |
| 16-shard merge equals the single build exactly | Right |
| Index fits in under 2 GB | Right for the index (1.02 GB); the build peaked at 6.7 GB while holding both builds for the gate |

## What it means

- **This is the first change since experiment 31 that makes a pair teach more,** not an update cost less. About a third fewer pairs reach the same loss, at 6% more compute per pair.
- **The source filter is worth keeping, but most of the gain comes from target n-grams alone.** Arm S beats arm T in both seeds on loss (by 0.03) and chrF (by 0.4 to 1.0).
- **Nothing in the hinted model depends on the hints being present.** With hints off it equals the unhinted model at the same pair count. Turning hints on adds the gain. The backbone loses nothing.
- **The pool-shaped part holds.** The index builds as 16 independent shards of 6 to 8 s and under 0.6 GB each, and merges exactly.

## Not shown, and next

- **Training beyond 3.6% of an epoch.** The gap was still widening at 32,768 pairs. A run to 131,072 or more pairs would show whether it closes, as the page found late in training.
- **The distributed update.** The prediction is that its seconds per update are unchanged: 3 bytes per token, plus a gather-scale-add on the slice owners. It needs the hint columns stored with the dataset and the owned operation; neither is built.
- **Seconds per update in the pool.** This is the PyTorch reference only. The 6% per update is CPU PyTorch on one machine.
- **The evaluation cost of these runs.** About a third of each hinted run went to repeated evaluations. Skipping the deduplicated split when it equals the full one, and evaluating with hints off only at the end, would remove it without changing a result.

## Reproduce

Stages 0 to 3 (about 70 minutes on the workstation):

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
python -m ygaraty.experiments.ngram_hints audit --version 1m-seed2026 --output runs/experiments/055-hints/audit.json
python -m ygaraty.experiments.ngram_hints build --version 1m-seed2026 --orders 3,5,8 --shards 16 --output runs/experiments/055-hints/index
for hints in none target source; do for seed in 2026 2027; do
  ygaraty experiment batch_learning --embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 \
    --length-window 64 --batch-tokens 3072 --learning-rate 0.0005 --pairs 32768 --eval-every 4096 --seed $seed \
    --hints $hints --hint-index runs/experiments/055-hints/index --eval-dedup --decode \
    --output runs/experiments/055-hints/$hints-$seed
done; done
```

The training-time check (about 4 minutes; a 128-pair evaluation, so training dominates):

```bash
for hints in none target source none; do
  ygaraty experiment batch_learning --embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 \
    --length-window 64 --batch-tokens 3072 --learning-rate 0.0005 --pairs 8192 --eval-every 100000 --eval-pairs 128 \
    --seed 2026 --hints $hints --hint-index runs/experiments/055-hints/index --output runs/experiments/055-hints/speed/$hints
done
```
