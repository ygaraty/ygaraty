# bf16 activations on the wire, the embedding fused and the final norms folded: the simulator's first two predictions, built

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | Experiment 32's setup: the 30.3M Transformer with owned layers and the vocabulary-split output layer, 8 NumPy workers on emulated 100/20 Mb/s links at 50 ms, float32 wire; loopback for the loss curves |
| **Result** | Activations and their gradients travel as bf16 (`YGARATY_ACTIVATION_WIRE=bf16`; parameters, their gradients, optimizer state, losses and statistics keep the wire's float32), each embedding tile is one task, and each stack's final layer norm is applied by its last layer's tasks, whose owner holds its weights. On the home links the update takes **39.7 s** (42.0 on the first update, 82.9 on the checkpoint update that publishes the owners' state), against 90.5 and 95.6 s before. Experiment 33's simulator had predicted 32 s for this plan, 19% optimistic, as it was for the earlier designs. Over 50 updates on loopback, bf16 activations change no loss by more than 5e-4 (mean 1e-4 on a mean loss of 7.5). The critical path is 25 worker hops (32 before); relay traffic 0.62 GB out and 0.37 GB in per update as the relay counts it, less as bytes. |
| **Evidence** | Profiles in the session's scratch directory (`exp34/*`), not synced; `scripts/remote.sh run34` reproduces the sequence on a pod |

## Question

Experiment 33 priced four changes without building them and said the two cheap ones, folding the final norms and fusing the embedding steps, were worth 12 s of 78, and bf16 activations about half the update. Are they, once built, and does bf16 through the backward change what the model learns?

## The model and how an update is distributed

As it stands after this experiment and its follow-ups; experiments 30 to 32 built most of it.

**The model** (experiment 30's recipe, `YGARATY_ARCHITECTURE=transformer`): a pre-norm encoder-decoder Transformer translating English to Portuguese, trained on OPUS-100 (1M pairs, a 16,000-piece SentencePiece vocabulary), batches of 64 sentence pairs, Adam at a learning rate of 5e-4, no dropout.

| Part | Shape | Parameters |
|---|---|---:|
| Embedding, shared with the output layer | 16,000 × 512 | 8.19M |
| Encoder | 3 layers: self-attention (8 heads), feed-forward 512 → 2048 → 512, two norms | 9.46M (3.15M each) |
| Decoder | 3 layers: as the encoder's, plus cross-attention and a third norm | 12.61M (4.20M each) |
| Positions (learned), output bias, final norms | | 0.05M |
| **Total** | | **30.3M** |

**The servers.** One server runs the coordinator (`ygaraty.webapp`), the relay (a Go store for the tensors tasks exchange) and PostgreSQL (the durable store: data, checkpoints). Workers are volunteers that pull: they claim tasks from the coordinator over HTTP, compute them, upload results to the relay and post them back. The coordinator compiles each update into a task graph from the model's tape (the forward pass and its automatic differentiation) and schedules it; a worker never sees more than its tasks.

**How the update is split:**

- **Bands of examples.** The 64 pairs are split into 8 bands of 8, and each Transformer layer runs as one fused task per band (forward, and backward with its parameter gradients). Bands are independent, so they move through the layers as in a pipeline: band 0 can be in the second layer while band 5 is in the first.
- **Owned layers** (experiment 31). Each of the 6 layers has an owner, a worker that keeps the layer's weights and Adam moments for the whole run. Every band task of that layer is routed to its owner, which also sums the bands' gradients and steps the optimizer. Weights never travel; only activations and their gradients do, from owner to owner through the relay. Owners publish their state to the server on checkpoint updates.
- **The output layer split by vocabulary** (experiment 32). The shared embedding and output layer is owned in 8 slices of 2,000 rows. The softmax over the whole vocabulary is computed across slices: each owner sends 4 numbers per row (its slice's maximum logit, sum of exponentials, the target's logit when the target is in the slice, and the logits' sum) and a partial result per band, instead of 29 MB of gradient.
- **What travels** (this experiment). Activations and their gradients as bf16; parameters, their gradients, optimizer state, losses and statistics as float32. The embedding is one task per tile; each stack's final norm is applied by its last layer's owner; the encoder output's gradient is summed band by band; claim replies carry the relay blocks their tasks read.
- **Synchronous and exact.** An update is one full optimizer step over the batch, numerically that of training on one machine (checked against a reference); there are no local steps (not DiLoCo) and no stale weights (not asynchronous).

With 8 workers there are 14 owned pieces (6 layers and 8 vocabulary slices), at most 2 per worker. The longest chain of an update is 25 worker hops: the embedding, 6 layers forward, the output layer's statistics and gradient, 6 layers backward, then the embedding's gradient and the slices' optimizer steps. On home links a hop costs mostly its owner's uplink: a layer's 8 bands of bf16 activations are about 4 MB, 1.6 s at 20 Mb/s.

The closest relative is SWARM-style pipeline and model parallelism with stateful stages; the differences are a central coordinator that schedules every task, work routed to owners rather than stages chosen by workers, and exact synchronous updates. A layer is one task on one owner, so it must fit in a worker (a decoder layer with its two Adam moments is 50 MB). The output layer, too large to move around, is the exception: it is split by vocabulary over several owners. The tiled design of experiments 24 to 29, which spreads one layer's arithmetic over many workers, is what the same planner produces for the primitive-operation models; switching a single oversized Transformer layer to it is not built.

## What was built

- **A bf16 block dtype** (`ygaraty/storage/tensors.py`): the top 16 bits of float32, rounded to nearest even, a quarter of float64's bytes. The store, the relay client, the worker and the durable tier encode and decode it beside float64 and float32; the relay checks uploads with `?bf16=1`; sampled validation allows the rounding (rtol 5e-3).
- **The planner decides what narrows** (`compiler/plan.py: narrow_activations`). The tape records which values are parameter-like (parameters, and the vocabulary gathers that stand for an owner's rows) and which nodes compute their gradients, including the sums, views and transposes that unpack a fused layer's packed gradients. Every other result is an activation or an activation gradient and gets `attrs['wire'] = 'bf16'`; a fused layer's backward result is narrowed band by band (the input gradients, not the packed parameter gradients). Losses, statistics, optimizer tasks, owners' state and scatters stay at the wire's width. A test checks the marks on the shipped and the owned plans.
- **The embedding as one op** (`tape.embed`): `table[ids] * sqrt(d) + positions[position_ids]` is one tape node, one `embed_rows` task per tile, and its backward scales inside the scatter. The arithmetic is what the three ops computed.
- **The final norms folded** (`tape.transformer_layer(..., norm=...)`): the stack's last layer takes the norm's weight and bias as two more parameters and applies the norm to its output; its backward returns their gradients in the packed result; owners load, step and publish them with the layer. The kernel's norm is the same code the separate task ran, so the forward is identical per row; the norm weights' gradients are summed per band and then over bands, a different order from before.
- **The simulator** builds plans with these in, so its `--fold-norms` and `--fuse-embedding` what-ifs are gone and `--activation-wire bf16` runs the planner's marks; its validation table carries the planner version each measured run used.
- **`scripts/remote.sh`**: bootstrap a RunPod CPU pod from the internet (repository, Go, Python, PostgreSQL, OPUS-100 from Hugging Face, the tokenizer) and run this experiment's sequence there, with nothing copied from a workstation.

## Result

The home links (8 workers, 100/20 Mb/s, 50 ms), seconds per update:

| Design | Steady update | Checkpoint update | Relay fetched / stored | Tasks | Hops |
|---|---:|---:|---:|---:|---:|
| Owned layers and vocabulary (experiment 32) | 90.5, 95.6 | 123, 111 | 0.78 GB / 0.46 GB | 812 | 32 |
| **+ bf16 activations, fused embedding, folded norms** | **39.7** | 82.9 | 0.62 GB / 0.37 GB* | 724 | 25 |
| Simulated for the built plan (experiment 33) | 32.2 | | 0.30 GB / 0.14 GB | | |

\* The coordinator's relay counters assumed the wire's dtype; they count bf16 blocks at float32 size, which the fix in this change corrects for later runs.

Per worker and update: 2.3 s of compute, 38 s in claims held open, 15 s posting results, 10.5 s fetching, 9 s in empty claims. By family (median of updates 2 and 3, so the publish weighs in): owners' load and publish 28 s, layers backward 9.8 s and forward 9.4 s, placements 4.1 s, the vocabulary gradient 3.3 s, everything else under 2 s. The checkpoint update is now twice the steady one: each owner uploads its layer's weights and both moments (50 MB for a decoder layer) over 20 Mb/s.

The loss curves, 50 updates on loopback, the same batches:

| Update | 1 | 2 | 3 | 10 | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float32 wire | 18.8733 | 9.7308 | 8.8744 | 7.5196 | 7.2874 | 7.2233 | 6.6459 | 7.1795 |
| bf16 activations | 18.8728 | 9.7307 | 8.8745 | 7.5196 | 7.2875 | 7.2231 | 6.6459 | 7.1796 |

The largest difference over the 50 updates is 5e-4, on the first update; the mean is 1e-4. The small model's three losses through HTTP and the relay are within 1.2e-4 of the float64 reference, with no sampled result rejected.

## What it taught

- **The simulator ranked right and priced within its stated error**: 32 s predicted, 39.7 measured, 19% low, as it was for the three earlier designs (14 to 26% low). Its what-if of 37 s for the same three changes, made before any of them existed, was 7% off the measurement.
- **bf16 activations cost nothing in learning at this size and budget**, which the rounding analysis suggested (three significant digits in values whose gradients are summed in float32) but only the curve shows.
- **What remains is the owners' round trips and the checkpoint update.** A steady update is 25 hops, each four round trips and a few hundred milliseconds of uplink; the checkpoint doubles the update, and experiment 33's next-step list stands: replicated owners (needs 16 or more workers), fewer round trips per hop, and the owners' state published in bf16 or in the background.

## Follow-up: gradients summed band by band

Profiling the 39.7 s update band by band showed every encoder backward pass starting only once the slowest decoder band had finished, up to 16 s after its own band was ready. The encoder output's gradient is the sum of the three decoder layers' memory gradients, and the tape placed those 24 band-shaped pieces into 512-row tiles; with 480-row bands, every tile read two bands, so the waits chained down to the slowest one. The tape now sums such pieces band by band (one `add_many` per band, then their concatenation) whenever several readers contribute the same full-width bands. The per-element order of the sums is the placement's, so gradients are bitwise unchanged: checked against the previous tape for the shipped plan (67 gradients) and the owned plan (210 published weights and moments after the step), and a test checks that an encoder band's gradient depends only on its own decoder bands.

On the home links the same batch takes **34.4 s** (39.7 and 38.8 before; the simulator had predicted a 2.9 s gain, the measured one is 5.3 s), the checkpoint update 62.1 s (82.9 before), with the same losses to every printed digit. Encoder bands now wait 0.6 to 1.2 s for their own decoder bands (mean 0.9, before 5.5), and the encoder backward ends 3.6 s after the decoder's instead of 7.9.

## Follow-up: claim replies carry their inputs

A hop cost three round trips: the result's upload and post, the claim reply, and the next worker's relay fetch. With `http-worker --pushed-inputs` a claim asks for its inputs too: the coordinator fetches the relay blocks the claimed tasks read, in one request to the relay beside it, and appends them, raw and in their stored dtype, to the binary claim reply (`Job.push_relayed`). The worker assembles its inputs from them and computes without a fetch of its own. Bytes on the worker's downlink are unchanged; the coordinator's link carries them instead of the relay's (266 MB per update here, against 5 MB before), and its relay fetches cost 0.37 s of request-thread time per update. A test with the real relay checks that pushed and fetched inputs give the same losses on the float64 and float32 wires, that blocks were pushed, and that workers made fewer relay requests.

On the home links the same batch takes **32.3 s** (34.4 before; the simulator had predicted 2.4 s), with the same losses; workers spend 7.9 s per update on relay requests instead of 12.1, and 3.5 s more in claims, whose replies are larger. The checkpoint update took 71.6 s, against 62.1 s in the previous run and 82.9 s in the first; with one sample each, checkpoint updates vary by about ten seconds and this one says nothing either way. (The coordinator's "fetched there" counter summed whole blocks per input since this experiment's first version; it now counts each input region's bytes.)

## Limits

One home-link run on one machine, one batch; the loss curves are 50 updates at one size and learning rate, with no translation-quality evaluation. The owner that applies a stack's final norm also carries its weights in every publish. The browser SDK has none of this.

## Running it

```bash
# experiment 31's servers and proxies, then:
ygaraty experiment http_profile --recipe transformer --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 \
  --workers 8 --workers-kind python --tiles 512 --max-tasks 16 --max-task-mib 512 --layer-task-mib 512 --output-task-mib 1024 \
  --steps 3 --repeats 1 --learning-rate 0.0005 --relay-url http://127.0.0.1:8101 --relay-token T \
  --rtt-ms 50 --down-mbps 100 --up-mbps 20 --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 8   # about 4 minutes
# the loss curves: the same on loopback with --steps 50 --checkpoint-every 1000, with and without --activation-wire bf16
# on a pod: ./scripts/remote.sh bootstrap && ./scripts/remote.sh run34
```

### Over pods: one server, 8 or 16 workers with shaped links (run 2026-10-27)

Closer to real volunteers than one machine: each worker is a pod of its own, and the latency proxy on that pod gives it a home connection (100 Mb/s down, 20 Mb/s up, a 50 ms round trip of which the real path's share is measured first and only the rest added). The server pod runs the coordinator, the relay, PostgreSQL and the dataset on its datacenter link. The proxy is used instead of `tc netem` because pods are containers without `NET_ADMIN`; it cannot add a home line's jitter or loss.

```bash
# server pod: 16 vCPUs, 32 GB, TCP ports 8000 and 8101 exposed
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/scripts/remote.sh | bash -s bootstrap
cd /workspace/ygaraty && ./scripts/remote.sh serve34 10      # prints one worker line per pod, then each update's seconds and loss
# each of 8 worker pods (2 to 4 vCPUs, 8 GB): paste its line, for example
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/scripts/remote.sh | bash -s worker HOST:PORT HOST:PORT JOB TOKEN w0
# a mixed pool: export YGARATY_DOWN_MBPS=50 YGARATY_UP_MBPS=10 YGARATY_RTT_MS=80 on some pods before pasting
```

`serve34` uses the distributed runner's external-worker mode (`--external-workers --listen 0.0.0.0:8000 --lease-seconds 120`), a fresh relay token per run, and RunPod's `RUNPOD_PUBLIC_IP` and `RUNPOD_TCP_PORT_<port>` for the addresses it prints. Each worker writes its spans and a time summary under `runs/experiments/034-pods/`. A dry run on one machine (the small model, two workers behind their proxies) gave the losses of this experiment's small check exactly.

**Smoke test on RunPod (2026-10-27).** `ygaraty experiment pods` creates the pods from a workstation with a RunPod API key, reaches them over SSH with a key of their own, and deletes them; each pod also deletes itself at a deadline, with the key RunPod gives the pod (which the GraphQL API accepts and the REST API refuses). One server (8 vCPUs, 16 GB, California) and two workers (2 vCPUs, 4 GB, Kansas; a real round trip of 41.5 ms, to which the proxy added 8.5) ran the small model for 3 updates: 11.0 s for the second update and 17.4 s for the third, which publishes the owners' state (losses 10.2976, 10.2546, 10.1303; the first update's 556 s include the workers' arrival). Bootstrapping the server took 13 minutes, a worker 2; the whole test cost about $0.15. Logs in `runs/experiments/034-pods/runpod-smoke-20261027/`, not synced.

**The full run over pods (2026-10-27).** The 30.3M model as in the home-link profile above, with every follow-up in (commit 8d7c92d): one server (16 vCPUs, 32 GB, California) and eight workers (2 vCPUs, 4 GB each, Kansas, two shared public addresses), each worker's real round trip 41.5 to 41.7 ms to which its proxy added the rest of 50 ms, with 100/20 Mb/s links. Ten updates, no retries, no sampled result rejected:

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seconds | 81.7 | 35.0 | 31.6 | 32.6 | 36.0 | 32.4 | 34.8 | 33.9 | 29.8 | 69.4 |
| Loss | 18.863 | 9.759 | 8.854 | 8.699 | 8.435 | 8.143 | 7.728 | 7.785 | 7.729 | 7.484 |

The eight steady updates take **33.3 s** (median and mean; 29.8 to 36.0), against 32.3 s for one update on one machine with emulated links: over real links from Kansas to California, the emulation held to 3%. The first update (81.7 s, 42.0 on one machine) includes the owners' loads and the workers joining over about 25 s; the last publishes the owners' state (69.4 s, 62 to 83 on one machine). The losses are the pod's own (its tokenizer is trained there) and fall as the machine's did. Compute was uneven: three workers computed for 58 to 84 s over the run, four for about 20 s and one for 0.4 s, which suggests the six layers were owned two to a worker (the simulator puts 16 workers, one owned piece each, 7 s ahead of 8). The pods' cores are slower than the workstation's (37 s of compute per worker over ten updates, against 23), still a tenth of the update. The run took 28 minutes of pods and about $0.45, the bootstrap 16 minutes of them. Logs in `runs/experiments/034-pods/runpod-8w-20261027/`, not synced.

**Sixteen workers (2026-10-27).** The same run with 16 workers (7 in Kansas, 9 in Washington, 18.6 ms from the server, each proxy making up 50 ms), commit 261aa8a:

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seconds | 100.1 | 22.9 | 21.8 | 21.6 | 23.6 | 21.6 | 18.6 | 23.2 | 21.7 | 39.1 |
| Loss | 18.791 | 9.727 | 8.849 | 8.704 | 8.402 | 8.191 | 7.707 | 7.753 | 7.760 | 7.452 |

The steady update takes **21.8 s** (median; mean 21.9, 18.6 to 23.6), against 33.3 s with 8 workers; the simulator had predicted 22.0 s for 16 workers and 29.3 for 8, so it was 12% low at 8 workers and right at 16. The publishing update takes 39.1 s instead of 69.4. The workers' computed tasks show why: with 8 workers the six layers had been owned two to a worker (w0, w4, w6), four workers held two vocabulary slices each and one owned nothing; with 16, each layer and each slice had an owner of its own and two workers owned nothing. The owners' uplinks no longer carry two layers' activations, which is the doubling up the simulator charged. The run took about 27 minutes of 17 pods, about $0.65. Logs in `runs/experiments/034-pods/runpod-16w-20261027/`, not synced.

**Workers must sit in another data center than the server.** RunPod's pods in one data center share its public address and cannot reach each other through it, nor through their private addresses; RunPod's Global Networking (`POD_ID.runpod.internal`) is offered for GPU pods only. Over the internet from a nearby data center is also what a volunteer's connection looks like. `pods up` places workers in Washington, Texas or Kansas by default, never in the server's data center; left to RunPod, they had landed in Sweden, 150 ms away. This is likely why the first attempt by hand (2026-10-26) had workers that never connected.
