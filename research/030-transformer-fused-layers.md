# A Transformer with fused layer tasks: the model built for the round-trip rule

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | HTTP path in production mode (the SQL translation job with `YGARATY_ARCHITECTURE=transformer`), PostgreSQL store, the Go relay, NumPy `http-worker` processes; experiment 26's profiler |
| **Result** | A 30.3M-parameter pre-norm Transformer translator (3+3 layers, width 512, feed-forward 2048, 8 heads) trains on OPUS-100 through the same job as the LSTM, as one fused task per layer and band of examples. Its critical path is 28 worker hops against the LSTM's 52 to 120. On one machine, 4 NumPy workers take 10.5 s per update (the LSTM's best, with a model half as large again) and 8 take 10.3 s; the update is bound by the layer tasks' arithmetic and the shared CPU. At 50 ms per round trip: 17.9 s, against the LSTM's 20.7 s. A 50-update training run reached update 41 before its workers died (cause not captured; the runner now keeps their stderr): median 12.2 s per update, loss 18.9 → 7.5 at update 10 → 6.7 at update 40, where the LSTM's run was after 100. Parity against a PyTorch reference holds at small size for logits, loss and every gradient (rtol 1e-9), and two verified updates pass through the job's oracle. |
| **Evidence** | `runs/experiments/030/omarchy-*/` (profiles per step, the training run), not synced |

## Question

Experiment 29 found the cost of an update over a network to be round trips along the critical path, and the LSTM's path grows with sequence length. A Transformer's does not, if a band of examples can run a whole layer as one task. How short does the path get, what does the update cost, and where does the time go then?

## What was built

- **A batched Transformer tape** (`ygaraty/models/transformer_translation.py`), example-major rows, nn.Transformer parameter names, tied scaled embeddings, learned positions, label smoothing 0.1, Adam with beta2 0.98. Shape-only specs seed it without PyTorch, as the LSTM's.
- **Fused layer tasks** (`ygaraty/runtime/transformer_kernels.py`): `tf_layer` runs one pre-norm layer (self-attention, cross-attention for the decoder, feed-forward, residuals) over a band of examples; `tf_layer_grad` recomputes that forward and returns one packed matrix: the input's gradient, the memory's, and every parameter's gradient. The attention probabilities and the feed-forward activations never leave the worker. The tape's backward rule (`Tape._tf_layer_backward`) sums the packed rows over the bands, then unpacks each parameter once. Bands are independent through the whole stack, so the plan is a few pipelines of layers, and the next layer's band starts as soon as its own rows exist.
- **The planner**: region readers resolve through row concatenations and through assemblies of regions, so a band reads its predecessor's output without a copy; unread copies are pruned; layer norms tile by rows; the embedding gradient and bias sums take budget-sized chunks; `admit_plan` holds fused layers to `YGARATY_LAYER_TASK_MIB` (their packed results exceed the 8 MiB request cap, so they need the relay).
- **Fewer, larger tasks elsewhere**: Adam as one `adam_step` task per row band with its three outputs assembled by region (115 tasks instead of 747 for this model); the fused output layer of experiment 29; relay uploads cut into the row bands their readers want (`upload_rows`), one round of parallel puts per result.
- **A background poster in the NumPy worker**: each result but a batch's last is posted on its own connection as soon as it is computed, coalescing whatever finished while a post was in flight, so a layer's output is not held back by the bulk tasks claimed with it. The last result goes with the next claim.
- **A PyTorch reference in the repository** (`ygaraty/reference/transformer.py`, nn.Transformer layers), which also replaced the sibling checkout the old Transformer tests needed: the suite has no missing-module errors any more.

## Setup

OPUS-100 EN-PT (experiment 22's tokenizer), batch 64, the dataset's first batches (43 source and 56 target steps), tile 512, task budget 512 MiB, layer budget 512 MiB, output-layer budget 1 GiB, relay on the same machine with 8 GiB, 16 tasks per claim. Three updates per profile (the first a warm-up), one run per setting, Ryzen 9 PRO 6950H with nothing else running. Losses 18.87, 9.73, 8.87 in every run of every step below.

## Result

Seconds per update, with the critical path and the tasks per update:

| Step | Tasks | Hops | 4 workers, 16 per band | 8 workers, 8 per band | 4 workers, 50 ms |
|---|---:|---:|---:|---:|---:|
| Fused layers, per-tile Adam, square uploads, 8 tasks per claim | 2,992 | 31 | 13.8 | 15.2* | 29.4 |
| + banded uploads, 512 MiB budget, 16 tasks per claim | 2,716 | 29 | 14.2 | 13.8 | 24.1 |
| + fused Adam, sums before unpacking, budget-sized reductions | 1,369 | 28 | 16.5 | 11.4 | 22.9 |
| + the background poster, coalescing | 1,369 | 28 | **10.5** | **10.3** | **17.9** |
| LSTM, its best setup (experiment 29) | 3,070–3,258 | 52–120 | 10.5 | | 20.7 |

\* 16 per band: 4 bands for 8 workers.

Where the 10.5 s go at 4 workers: each worker computes for 5.2 s (the layer tasks 3.0 s forward and 9.8 s backward summed over workers, the output layer 2.3 s, its gradient matmuls 2.0 s), waits 3.6 s for a ready task, spends 1.5 s posting results and 1.3 s on relay fetches. The coordinator sends 7 MB and receives 0.8 MB per update; the relay serves 6.5 GB and stores 3.1 GB. With 8 workers each computes for 3.3 s and waits 4.9 s; the coordinator's lock is contended for 5.4 s of request-thread time, and the 8 workers, the relay (which hashes 3.8 GB per update), PostgreSQL and the coordinator share 16 hardware threads.

The first loss of 18.9 is the untrained model predicting a copy of its input: with tied, scaled embeddings and a pre-norm residual stream, the current token's logit starts near 21. Adam undoes it within three updates.

## What each step taught

- **Fewer tasks were not faster by themselves.** Fusing Adam and the reductions halved the task count and made 4 workers slower (14.2 to 16.5 s): the bulk tasks got bigger, and a worker posted a layer's result only after the 15 bulk tasks claimed with it, so each backward layer took 2.1 s of wall for 0.6 s of arithmetic. The background poster fixed that (16.5 to 10.5 s). Batching results per claim saves round trips; holding a critical result for a batch costs more.
- **The layer phases are compute, not latency.** With b bands and at least b workers, a layer phase is one band task long, so the whole layer arithmetic (12.8 worker-seconds here) is on the critical path divided by b. Smaller bands buy parallelism only with more workers; on this machine 8 workers gained little because they share the CPU with the relay's hashing and the coordinator.
- **Over a network the Transformer still pays 3 round trips a hop through the relay**, 84 per update, and 17.9 s at 50 ms is 7.4 s over loopback. The poster's first version posted each result on its own and waited for a batch's posts before the next batch: 32.1 s at 50 ms. Coalescing finished results into one request while a post is in flight gave the 17.9 s, so a slow link costs a bounded number of round trips per batch.

## The training run

`ygaraty experiment distributed_translation --architecture transformer` with the settings above, 8 workers, 8 per band, a checkpoint every 25 updates, through the relay. It reached update 41 of 50 in 10 minutes before its workers exited; the runner held their stderr in a pipe, so the reason was lost, and it now writes each worker's stderr to a file. 105 tasks were retried over the run (leases of the dying workers), none rejected.

| | |
|---|---|
| Seconds per update, updates 1–38 | median 12.2, mean 14.8 (8.7–104; the slow one includes a 720 MB checkpoint export) |
| Loss at updates 1, 10, 20, 30, 40 | 18.87, 7.52, 7.29, 7.22, 6.65 |
| Mean loss over updates 31–41 | 6.91 |
| The LSTM's first run (experiment 24), loss at 10, 50, 100 | 7.72, 7.38, 6.63 |

Same data, same batch size, a different model and learning rate, so this says the Transformer learns at least as fast per update at this size; quality per compute is experiment 22's question, not measured here.

## What more workers and relays would give

Everything below is a projection from this experiment's measured parts, not a measurement.

An update costs about: the layer arithmetic divided by the bands that run in parallel (12.8 worker-seconds at this size), plus the output layer and its gradient (about 4 worker-seconds over 72 tasks), plus the optimizer and reductions (about 1 worker-second), plus 28 hops times the round trips a hop costs, plus the bytes through the relays divided by their bandwidth. The last term is what a second relay changes.

| Workers on their own machines | Bands | Arithmetic floor | Round trips at 50 ms | Relay bytes per update |
|---:|---:|---:|---:|---:|
| 8 | 8 | 1.6 + 0.6 + 0.2 s | 84 × 50 ms = 4.2 s | about 12 GB |
| 16 | 16 | 0.8 + 0.4 + 0.2 s | 4.2 s | about 16 GB |
| 64 | 64 | 0.2 + 0.3 + 0.2 s | 4.2 s | about 30 GB |

The relay term dominates on real links. One relay on a 1 Gb/s server moves 12 GB in about 100 s; two relays sharing the keys by hash halve that, and a 10 Gb/s link cuts it to 10 s. So with 8 to 16 remote workers on a 50 ms network, two 1 Gb/s relays give roughly 55 s per update, bandwidth-bound; two 10 Gb/s relays roughly 10 s; on a LAN the floor is the 3 to 5 s of arithmetic and round trips. Cutting bytes per update, which grow with the number of bands because every band's weights and gradients cross a relay, matters more than a second relay once the link is slow. Sharding the relay by key is a small change to the client (a list of relays, the key's first digits choosing one) and is not built yet; benchmarking it on one machine would show the relay's CPU relief, not the bandwidth effect.

## Limits

One run per setting on one batch; the coordinator, relay, store and workers share one machine, so 8 workers are contention, not a pool. Parity is checked end to end at small size (tests) and by the job's oracle for two tiny updates; the full-size run is checked only by the 1% sampled recomputation. The browser SDK has none of the fused kernels, so browsers cannot run this recipe yet. The quality a Transformer of this size reaches per update is not measured (experiment 22's counterpart is missing).

## Running it

```bash
cd relay && go build -o ../relay-server ./cmd/relay && cd ..
./relay-server -addr 127.0.0.1:8101 -token T -max-mib 8192 &
ygaraty experiment http_profile --recipe transformer --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 \
  --workers 8 --workers-kind python --tiles 512 --max-tasks 16 --max-task-mib 512 --layer-task-mib 512 --output-task-mib 1024 \
  --steps 3 --repeats 1 --learning-rate 0.0005 --relay-url http://127.0.0.1:8101 --relay-token T
# a training run: ygaraty experiment distributed_translation --architecture transformer ... with YGARATY_RELAY_URL set
```
