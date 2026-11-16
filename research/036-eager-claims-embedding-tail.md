# Eager claims and the embedding gradient at its owners: 17.6 s per update with 16 workers

| | |
|---|---|
| **Date** | 2026-10-27 |
| **Path** | Experiment 34's design (owned layers and vocabulary, bf16 activations, inputs pushed with claims) on emulated home links: one machine, 8 or 16 NumPy workers each behind a latency proxy (100 Mb/s down, 20 Mb/s up, 50 ms); the update simulator (experiment 33) to find and price the changes |
| **Result** | Two changes. Each vocabulary slice's owner now scatters its rows' embedding gradient itself, from the first layers' band results, which takes four tasks off the end of the update (3.3 s to 1.1 s). And workers can claim eagerly: they stream each result as soon as it is computed and claim again at once, their results in flight no longer holding back their next claim. With 16 workers the update takes **17.6 s** (21.3 s with the first change alone; 21.8 s measured over RunPod before either), inside the goal of 15 to 20 s, and **17.5 s over RunPod** (21.8 s there before); with 8 workers 25.3 s (30.2 and 32.3), held back by layers owned in adjacent pairs. The losses are the same to every printed digit. Batch 128 over RunPod takes 25.0 s, 1.4 times the pairs per second. |
| **Evidence** | Profiles in the session's scratch directory (`batch/home*`), not synced |

## Question

The simulator's profile of the 16-worker update, which matched the RunPod measurement (22.0 s against 21.8), put 3.3 s after the last layer's backward pass, in the embedding's gradient, and showed the first encoder layer's last band finishing 2.5 s after its inputs were ready, where the other layers took about 1 s. What causes each, and what do fixes buy?

## What the profile showed

- **The embedding's gradient took a detour.** After encoder 0's last band: `place_block` (0.35 s) copied the band's rows, `scatter_rows` (1.17 s) downloaded the whole 3,840-row gradient and scattered it into the batch's 743 gathered table rows, `add_many` (0.63 s) and `add` (0.75 s) summed it with the decoder side's, and only then did each slice owner take its rows and step. Three of those tasks uploaded a 1 MB float32 tensor over a 20 Mb/s uplink.
- **The first encoder hop was the worker protocol, not tile alignment (the first guess).** Encoder 0's owner claims all 8 bands at once. Its background poster coalesced whatever had finished into one relay upload and one post, so band 1 committed only once bands 2 to 4 were uploaded too (1.85, then 2.74 s for four bands at once). And a worker claims again only after its batch's last result has gone up with that claim, while the coordinator leases no more tasks to a worker whose results are still in flight: encoder 1's band 1, ready at 2.04 s, was claimed at 2.27, and its bands 5 to 7 waited about 0.9 s. The bursts formed again at every layer.
- **Priced before building** (simulated, 3 seeds): streamed posts alone change nothing at 16 workers (22.2 s); eager claims take 16 workers from 22.2 to 18.1 s and 8 workers from 29.3 to 19.7; with the embedding fix as well, 15.2 and 16.3 s.

## What was built

- **The slice owners scatter their own rows** (`tape.backward`, `tf_vocab_embed_grad`). For the owned vocabulary's gathered table, an embedding's gradient is no longer scattered into the table. Its row bands, the first layers' input gradients read as regions of their packed results, go to every slice's `vocab_embed_grad` with each row's place among that slice's gathered rows. The owner scatters each side in row order (`embedding_backward`'s arithmetic, with the scale factor) and adds the two sides in the order the tape would have. The position table's gradient is one `scatter_rows` over the bands of both sides. The plan has 382 tasks instead of 401. The owned-vocabulary test (two Adam updates of the owned plan against the plan that ships the table, within 1e-9) passes; the arithmetic differs from before only in that the intermediate sums are no longer rounded to float32 on the wire.
- **Eager claims** (`http-worker --eager-claims`). A `StreamPoster` uploads each result as soon as it is computed, on two connections taking results in order, and posts it as soon as its own upload is done; posts that pile up go in one request. No result is carried with the next claim, and the worker claims again at once. For a claim marked `eager`, the coordinator does not count the worker's tasks still on their way back against the claim's limit (`Job.claim`). A test with the real relay checks that eager workers reach the same losses as workers fetching or receiving their inputs, on the float64 and float32 wires. `scripts/remote.sh` workers use it.
- **The simulator** gains `--stream-posts` and `--eager-claims`, and four validation points.

## Results

Seconds per steady update (the second of three), the same batch:

| Workers | Before (experiment 34, inputs pushed) | Embedding gradient at its owners | + eager claims | Simulated, both |
|---:|---:|---:|---:|---:|
| 8 | 32.3 | 30.2 | 25.3 | 16.1 to 18.3 |
| 16 | 21.8 (RunPod) | 21.3 | **17.6** | 15.2 |

The losses are 18.8728, 9.7308 and 8.8745 in every run, as in experiment 34. Updates that publish the owners' state: 64.2 and 73.0 s with 8 workers, 36.8 and 34.1 s with 16. The 16 workers peaked at 3.9 GB of memory in all.

Where the 16-worker update goes, from the runs' task logs (last band of each phase):

| Phase | Embedding gradient at its owners | + eager claims |
|---|---:|---:|
| Embedding and 6 layers forward | 10.8 s | 7.8 s |
| Output layer: statistics, gradients | 3.8 s | 3.3 s |
| 6 layers backward | 5.4 s | 5.2 s |
| Embedding gradient and optimizer steps | 1.3 s | 1.1 s |

## Over RunPod: 17.5 s with 16 workers, and batch 128

The same code (commit 834de18) on RunPod CPU pods, as in experiment 34: a server in California (16 vCPUs), 16 workers of 2 vCPUs in Kansas and Washington, each behind its latency proxy (a real round trip of 41.6 ms in Kansas, 18.6 ms in Washington, made up to 50 ms; 100/20 Mb/s), workers with `--pushed-inputs --eager-claims`. Each worker owned at most one piece (6 layers, 8 slices, 2 none). Ten updates per run, no retries, no sampled result rejected.

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch 64, seconds | 56.1 | 18.8 | 16.6 | 17.2 | 17.8 | 16.7 | 18.7 | 19.5 | 15.6 | 36.3 |
| Batch 64, loss | 18.850 | 9.732 | 8.799 | 8.698 | 8.438 | 8.155 | 7.789 | 7.793 | 7.800 | 7.450 |
| Batch 128, seconds | 68.1 | 25.3 | 25.3 | 26.9 | 27.5 | 22.1 | 20.8 | 17.1 | 24.7 | 107.1 |
| Batch 128, loss | 18.792 | 9.654 | 9.006 | 8.462 | 8.291 | 7.977 | 7.834 | 7.563 | 7.584 | 7.523 |

- **Batch 64: 17.5 s per steady update** (median; mean 17.6, 15.6 to 19.5), against 21.8 s on RunPod before these changes and 17.6 s emulated on one machine: the emulation held again. The update that publishes the owners' state took 36.3 s (39.1 before).
- **Batch 128: 25.0 s** (median; mean 23.7, 17.1 to 27.5), 12.5 s per 64 pairs: 1.4 times the pairs per second of batch 64, as the simulator said (1.37). Experiment 35 found that this early in training a pair in a batch of 128 teaches less than one in a batch of 64, so the gain in pairs does not yet turn into faster training. The publishing update took 107.1 s, three times batch 64's, with no retry; one sample, not explained (the state published is the same size).
- Steady updates vary by about 25% within each run, more than on one machine; batches pad to their longest sentence, which may be part of it (the run log does not record lengths).

The run cost about $1.10 of pods, including 10 minutes of idle pods while the orchestration waited on the wrong file. Logs in `runs/experiments/034-pods/runpod-16w-eager-b64-20261027/` and `...-b128-20261027/`, not synced.

## What it taught

- **The worker protocol set a third of the forward pass.** Results posted in bursts and claims waiting on uploads made every layer queue its bands again; streaming results and claiming at once lets bands pass through the owners at their uplinks' pace. The backward pass gains less (5.4 to 5.2 s): there each owner also waits for the decoder layers' memory gradients and steps its optimizer.
- **Owner placement decides the 8-worker case.** In both 8-worker runs the coordinator gave adjacent layers to one worker (encoders 0 and 1, encoder 2 and decoder 0, decoders 1 and 2): in the first update, the first worker to claim takes two owners' load tasks in a row. Adjacent layers on one uplink cannot overlap their bands. The simulator's claims spread the layers by chance, which is why it puts 8 workers at 16 to 18 s. A placement rule (no second layer for a worker while others have none, never adjacent layers) is the next change.
- **The simulator was within 5% for the embedding change at 16 workers (20.3 against 21.3 s) and 14% low with eager claims (15.2 against 17.6)**, the same optimism as on home links before; at 8 workers it is far low because of owner placement, which it does not reproduce.
- **What remains at 16 workers** is the backward pass (5.2 s) and the output layer (3.3 s): bytes per hop, where 8-bit activations are the lever left.

## Limits

One run per setting, 3 updates each on one machine (16 workers and their proxies on 16 cores) and 10 over RunPod. NumPy workers only; the browser SDK has none of this. An eager worker can hold more leases than one claim's worth, bounded by its results in flight.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite YGARATY_STORE_URL=postgresql:///ygaraty_exp24
./relay-server -addr 127.0.0.1:8113 -token q -max-mib 8192 &
ygaraty experiment http_profile --recipe transformer --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 \
  --workers 16 --workers-kind python --tiles 512 --max-tasks 16 --max-task-mib 512 --layer-task-mib 512 --output-task-mib 1024 \
  --steps 3 --repeats 1 --learning-rate 0.0005 --relay-url http://127.0.0.1:8113 --relay-token q \
  --rtt-ms 50 --down-mbps 100 --up-mbps 20 --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 8 \
  --pushed-inputs --eager-claims        # about 3 minutes; without --eager-claims for the middle column; --workers 8 for the first row
ygaraty experiment simulate validate --seeds 3          # includes this experiment's four points
```
