# A simulator of the update: the real plan and scheduler on modeled links

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | `ygaraty experiment simulate` (`ygaraty/experiments/simulate.py`): the job builds its task graph for one update as it does in production, with an in-memory store standing in for PostgreSQL and the relay; `Scheduler.claim` leases tasks on simulated time with the job's owner routing; workers, links, the relay and the coordinator are a discrete-event model; kernel seconds come from a cost model fitted to the compute times workers report |
| **Result** | A simulated update takes about a second. Against five measured runs the model is within 7% on loopback and 2% at a 50 ms round trip, and 14 to 26% low on the emulated home links (predicting 80 s where 90 to 96 s were measured, 134 for 178, 205 for 277), with the designs in the right order and the right ratios. Getting there took three facts about the coordinator that the model first had wrong, each worth 20 to 50% of the prediction. A sweep of 18 settings in two minutes shows the owned design's floor on 20 Mb/s uplinks (65 to 85 s whatever the number of workers) and what a faster uplink buys (45 to 49 s at 50 Mb/s). Four changes priced as rewrites of the plan before being built: folding the final norms and fusing the embedding steps save 12 s of the 78 (not the 28 the profiler's family shares suggested); bf16 activations take the update to 35 s; two owners per layer with the gradient shared take it to 43 s at 32 workers; both together to 26.5 s, four owners per layer with bf16 to 22 s, about 25 to 30 s after the model's optimism, with the last layer's gradient exchange as the tail. |
| **Evidence** | `ygaraty/simulation/cost_model.json` (fitted from two loopback profiles); the validation table below is `simulate validate --seeds 5` |

## Question

Every remaining idea for the update on home links (experiments 31 and 32) is a protocol or scheduling change, and a measured run of one costs 6 to 15 minutes of a machine. Can a model that reuses the real planner and scheduler, and fits the rest to what the profiler already measures, predict seconds per update well enough to rank ideas?

## What was built

- **The plan from the job itself.** `build_plan` constructs `DatabaseTranslationJob` with the real dataset and model settings, an in-memory store and no relay, then re-plans the steady-state update (owners loaded; optionally a publish update). It takes about a second for the 30M model.
- **The scheduler on simulated time.** `Scheduler.claim(now=...)` and `Job.choose` (owner routing) are the production code; commits go through `result_reference`.
- **Modeled:** each worker's downstream and upstream link as the latency proxy models them (large transfers queue, small messages slip between chunks), the round trip, the coordinator as a single server (a service time per request and a lock pass per task claimed), the relay as a shared link, the NumPy worker's protocol (claims of up to 16 tasks, one relay fetch per claim for the inputs the tile cache does not hold, the background poster that coalesces results while a post is in flight, the last result carried with the next claim, claims held open for 10 s, the 300 ms sleep after an empty one), the owners' state, requests jittered by up to 5 ms and kernel times varied by 10%, so seeds sample which worker claims first.
- **The cost model.** `http_profile` now writes `task_times.json` (per result: op, work, bytes, compute ms, worker, lease and commit times, layer, band). `simulate fit` fits seconds = a + b × work per op by least squares (`compiler/cost.py` defines the work: flops for matmuls and layer tasks, elements otherwise); ops whose work does not vary get a median. Fitted from two loopback profiles of experiment 30's and 32's designs (4,592 tasks).
- **Instruments.** `simulate run` prints the per-family in-flight and queue-wait table the profiler prints, the owner assignment and the update's critical chain (each task's ready, lease and commit times); `SIMULATE_TRACE=w3` traces a worker's claims, fetches and posts; `simulate timeline` prints the same from a measured run's task log, which is how the model was debugged.

## Result

`simulate validate --seeds 5`, the median over seeds against the measured steady-state update:

| Run | Measured s | Simulated s | Error |
|---|---:|---:|---:|
| Loopback, 8 workers, today's design | 10.3 | 11.0 | +7% |
| 50 ms, 4 workers, today's design, band 16 | 17.9 | 17.5 | −2% |
| Home link, today's design | 277 | 205 | −26% |
| Home link, owned layers | 178 | 134 | −25% |
| Home link, owned layers and vocabulary | 93 | 80 | −14% |
| (publish update of the last) | 111 to 123 | 109 | |

The measured ratios between the three home-link designs are 1 : 0.64 : 0.34; the simulated 1 : 0.66 : 0.39.

**What it took.** The first version over-predicted every run by 50 to 80%. Three corrections, in order of effect:

1. **Claims are held open.** The coordinator holds an empty claim for 10 s and retries at every commit (`CLAIM_WAIT_SECONDS`); the model had workers sleeping 300 ms per empty claim. This also means a worker whose poster still has results in flight waits in its claim until they commit, since the claim's load limit counts them.
2. **A shared link is reserved when the request arrives, not when it is sent.** The model reserved the relay's link for an upload at the time the upload started, so every fetch sent before it arrived queued behind it. A simulator bug, found from a 0.8 MB fetch taking 5 s.
3. **Concurrent claims interleave.** `Job.claim` takes the job lock once per task, so when several workers claim at once the ready tasks are dealt round-robin, and a worker rarely gets 16 of them. The measured run shows posts of 1.55 results on average; the model, claiming atomically, gave one worker 15 one-megabyte tasks and its uplink 6 s of work while the others waited, and put 27 s of embedding work on the critical chain where the real run has 4.5 s.

After these, the remaining error is a uniform 15 to 30% in the in-flight time of every family on home links, which no single mechanism explains; CPU contention of eight workers, proxies, the relay and PostgreSQL on one machine is the likely part the model leaves out.

**A sweep** (`simulate sweep`, owned layers and vocabulary, 50 ms, 100 Mb/s down, median of 3 seeds, seconds per update):

| Workers | Band | 20 Mb/s up | 50 Mb/s up |
|---:|---:|---:|---:|
| 8 | 8 | 84 | 49 |
| 8 | 4 | 88 | 48 |
| 8 | 2 | 80 | 57 |
| 16 | 8 | 67 | 45 |
| 16 | 4 | 66 | 44 |
| 32 | 8 | 68 | 45 |
| 32 | 2 | 67 | 49 |

Projections, not measurements, and 15 to 25% optimistic by the table above. They say the owned design is bound by each owner's uplink and the round trips of its chain, not by the number of workers: 16 workers beat 8 only because fewer workers own two layers, and 32 add nothing. A faster uplink is the one setting that moves it.

## What-ifs: the next changes priced before being built

The simulator takes rewrites of the plan, applied to the task graph the job built and simulated with the same costs. Two of them have since been built (experiment 34) and are the planner's: the final layer norms folded into the last layers' tasks and the embedding's lookup, scaling and position add as one task per tile; here they were `--fold-norms` (bypass the norm tasks, drop their weight-gradient and Adam tasks) and `--fuse-embedding` (one task charged the sum of the parts). The others remain: `--replicas k` gives a layer k owners holding its state, band b to owner b mod k, each owner sharing its packed gradient through the relay (17 MB for a decoder layer at float32) and its step reading the others', with a routing rule that puts one replica of a layer per worker; `--activation-bytes 2` sends everything tasks produce as 2-byte values (leaves keep the wire's 4), the bytes of bf16 activations without any claim about their numerics.

Seconds per update on the home links (50 ms, 100/20 Mb/s), owned layers and vocabulary, medians over 5 seeds (3 for four replicas):

| Plan | Tasks | 8 workers | 16 | 32 |
|---|---:|---:|---:|---:|
| As built | 505 | 78 | | |
| Final norms folded | 465 | 75 | | |
| Embedding steps fused | 451 | 70 | | |
| Both | 411 | 66 | 62 | 62 |
| Both, two owners per layer | 429 | 59 | 52 | 43 |
| Both, four owners per layer | 453 | | 58 | 36 |
| Both, bf16 activations | 411 | 37 | 35 | 36 |
| Both, two owners per layer and bf16 activations | 429 | 34 | 31 | 27 |
| Both, four owners per layer and bf16 activations (2 seeds) | 453 | | | **22** |

What the two embedding changes are worth is 12 s, not the 28 s the profiler's wall shares suggested: those shares charge a family for every moment it is in flight, including the slow periods it does not cause, while the simulator's critical chain charges only what others wait for (the measured timeline agrees: the embedding phase is 4.5 s, not 16). The two levers that matter are the bytes each owner's uplink carries per layer pass: halving them (bf16) halves the update, spreading them over owners helps only with enough workers to give each layer its own uplinks (32 workers for two owners per layer, since a worker also owns a vocabulary slice), and both together compound.

The critical chain of the best case (32 workers, two owners per layer, bf16, seed 1: 26.9 s): the six forward passes take 7.4 s (about 1.2 s each: four round trips and four bands of 0.45 MB through one uplink), the output layer 3.2 s, the six backward passes 11.7 s, and the last layer's gradient share and step 3.9 s at the tail. Four owners per layer with bf16 reach 22 s at 32 workers. With the model's 15% optimism on this link, 25 to 30 s. Under 20 s needs the gradient exchange in bf16 too or overlapped with the next update, and fewer round trips per hop (a hop is four today: the claim's reply, the fetch, the upload, the post).

Projections when written: the first two rows' changes and bf16 activations were built afterwards (experiment 34: 39.7 s measured against the 32 s simulated for the built plan, the model 19% optimistic as before; bf16 changed no loss by more than 5e-4 over 50 updates). Replication is not built; summing replicated gradients is not checked, and each replica would hold the whole layer's state (about 100 MB for a decoder layer).

## What it is for, and not

Ranking changes that differ by 2× (the next steps of experiment 32: fewer small tasks around the embeddings, results posted by priority, claims limited by bytes, a layer's bands over several owners) and answering scaling questions (workers, links, bands, a bigger model) in seconds, before one real run confirms the choice. Not for differences under 20%, for correctness (the lease expiry of experiment 31 is not modeled, nor retries), for browsers (no measured kernel times), or for the loopback regime at many workers, where the CPU is the limit.

## Running it

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
ygaraty experiment simulate validate --seeds 5
ygaraty experiment simulate run --workers 8 --rtt-ms 50 --down-mbps 100 --up-mbps 20 --float32-wire --owned-layers --owned-vocab 8
ygaraty experiment simulate sweep --owned-layers --owned-vocab 8 --float32-wire --rtt-ms 50 --down-mbps 100 \
  --sweep-workers 8,16,32 --sweep-up-mbps 20,50 --seeds 3
# what-ifs, any combination (4 replicas at 32 workers take minutes per seed):
ygaraty experiment simulate sweep --owned-layers --owned-vocab 8 --float32-wire --rtt-ms 50 --down-mbps 100 --up-mbps 20 \
  --fold-norms --fuse-embedding --replicas 2 --activation-bytes 2 --sweep-workers 8,16,32 --seeds 5
# refit after a kernel change: run http_profile (it writes task_times.json), then
ygaraty experiment simulate fit --from runs/profile-a runs/profile-b
```
