# Owned layers on emulated home links: what a volunteer's connection does to an update

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | HTTP path in production mode, the Transformer recipe of experiment 30 (30.3M parameters, 3+3 layers, width 512, batch 64, bands of 8 examples), PostgreSQL store, the Go relay, 8 NumPy `http-worker` processes, each behind its own latency proxy with a 100 Mb/s downstream and a 20 Mb/s upstream link at 50 ms round trip (`relay/cmd/latency-proxy`, `--down-mbps`/`--up-mbps`); relay blocks as float32 (`YGARATY_FLOAT32_WIRE`) |
| **Result** | On those links today's design takes **277 s per update**, moving 3.9 GB out of the relay and 1.9 GB into it per update. With **owned layers** (each Transformer layer's weights and Adam moments stay on one worker between updates, only activations cross the network) the update takes **178 s**, with 2.3 GB and 0.93 GB. Losses are identical (18.873, 9.731, 8.874). Both are far from the goal of 15 to 20 s; the shared 8.2M-parameter embedding and output layer, still shipped as bytes, takes most of what remains (experiment 32). The float32 wire halves every transfer at one rounding per link; a small model's losses agree with the float64 wire to float32 precision, with no sampled result rejected. |
| **Evidence** | Profiles in the session's scratch directory (`exp32/home32*`), not synced |

## Question

Experiment 30's Transformer costs 10.5 s per update on one machine and about 18 s at a 50 ms round trip with unlimited bandwidth. A volunteer's home connection is the target: about 100 Mb/s down, 20 Mb/s up, 50 ms. What does the update cost there, and how much of it is the weights and gradients that every band task moves through the relay? Owned layers remove exactly those bytes.

## What was built

- **Asymmetric access links in the latency proxy.** `-down-mbps` and `-up-mbps` give each proxy process one downstream and one upstream link that all of its connections share, chunk by chunk, with the round trip on top. `http_profile --down-mbps --up-mbps` starts one proxy per worker, forwarding both the coordinator and the relay, so a worker's two servers share its home link as they would. Checked: 4 MiB up in 1.70 s at 20 Mb/s (1.68 expected), 8 MiB down in 0.67 s at 100 Mb/s.
- **A float32 wire** (`YGARATY_FLOAT32_WIRE=1`, `--float32-wire`). The scratch store and the workers write relay blocks as float32; each block records its type, readers get float64, arithmetic and the owners' state stay float64, and weights go back to float64 in the durable store. The relay checks uploads with `?f32=1` as it checks float64 with `?f64=1`. Sampled validation allows the one rounding (rtol 2.5e-7). Without the setting nothing changes.
- **Owned layers** (`YGARATY_OWNED_LAYERS=1`, `--owned-layers`). The tape marks the layers as owned and records, per layer, its parameters and its band tasks; the planner emits `tf_owned_layer` and `tf_owned_grad` tasks that take only the activations, one `tf_owned_step` per layer after every band's gradient, a `tf_owned_load` on the job's first update and `tf_owned_publish` tasks on checkpoint updates, whose results replace the stored weights and moments. The worker keeps the state in an `OwnedLayers` object: the gradient task packs the parameter gradients and keeps them; the step sums them in band order and applies Adam with the unowned plan's arithmetic, so both plans give the same numbers (test: two updates bitwise equal). The coordinator routes an owned task to the worker that first took one of that layer's tasks, up to ceil(layers/workers) layers per worker, puts a worker's own tasks first in its claims, never samples them (it cannot recompute them), and fails the job if an owner is not seen for three lease periods. Unpublished parameters keep their stored copy, so a checkpoint is always the last publish.

## Setup

OPUS-100 EN-PT, batch 64, the dataset's first batches, tile 512, budgets of 512 MiB (tasks and layers) and 1 GiB (output layer), 16 tasks per claim, 3 updates per run (the first a warm-up, the last a checkpoint update, which publishes the owners' state). One run per setting on a Ryzen 9 PRO 6950H with the workers, the proxies, the relay, PostgreSQL and the coordinator sharing it. Learning rate 5e-4.

Two earlier attempts with the float64 wire failed after four to five minutes with `Task ... exceeded 5 attempts` and `lease expired` retries: the profiler set its 600 s lease after the first update's scheduler had been built, so the first update ran on the production default of 30 s, and a claim of 16 output-layer tasks takes longer than that to upload over 20 Mb/s. The profiler now sets both. The production lease is still 30 s, which real volunteers on such links would hit; see Limits.

## Result

Seconds per update, update 2 and the checkpoint update 3:

| Design | Update 2 | Update 3 (publish) | Relay fetched / stored per update | Tasks per update | Critical path |
|---|---:|---:|---:|---:|---:|
| Today's design (experiment 30) | 277 | 295 | 3.9 GB / 1.9 GB | 1,393 | 28 hops |
| Owned layers | **178** | 188 | 2.3 GB / 0.93 GB | 1,000 | 28 hops |

Workers compute for 2.2 s per update in both; the rest is the network. Per worker and update, today's design spends 161 s in claims (held open while nothing is ready or its own results are in flight), 73 s posting results, 90 s on relay fetches and uploads, 29 s in empty claims; the owned design 108, 45, 39 and 31 s.

Wall time by task family in the owned run (seconds per update, each moment shared by the tasks in flight):

| Family | Seconds | Why |
|---|---:|---|
| Adam step of the shared embedding/output layer | 34.6 | 16,000 × 512 rows of weights and two moments, read and written through the relay |
| Output softmax and loss (`ce_fused`) | 29.8 | each band's 29 MB logit gradient uploaded over 20 Mb/s |
| Output-layer gradient matmuls | 14.6 | read those gradients back |
| Owner load and publish | 14.1 | the checkpoint update only |
| Transformer layers, forward and backward | 27.3 | activations only now; backward bands wait 1.2 s on average for an owner busy uploading other work |
| Partial sums, layer norms, accumulations, embedding and bias gradients | 49.3 | small tasks, each a round trip and a 1 MB upload |

The same run on loopback (float32 wire, 8 workers) takes 10.8 to 11.5 s per update for today's design, as in experiment 30; owned layers cost a little there (11.3 to 13.3 s with the owned vocabulary of experiment 32), since routing a layer's bands to one worker serializes them.

## What it taught

- **Bytes, not hops, set the time on a home link.** 84 round trips per update at 50 ms are 4.2 s; 5.8 GB through eight 20 Mb/s uplinks and 100 Mb/s downlinks are over 200 s. Experiment 30's projection of about 130 s for this design assumed the bytes spread evenly over the links; they do not, because each result waits in one worker's uplink queue behind whatever that worker uploads.
- **Ownership removes the layers' bytes but not the output layer's**, which is the largest tensor in the model (8.2M of 30.3M parameters) and is read and written by every band. That is experiment 32.
- **An owner's uplink is on the critical path.** All 8 bands of a layer's activations leave through its owner's 20 Mb/s link, and its bulk uploads delay its own layer's next band.

## Limits

One run per setting on one batch; everything shares one machine, so the 2 s of compute per update are not a volunteer's CPU. The production lease of 30 s is too short for a 16-task claim over a 20 Mb/s uplink; claims should be limited by bytes or leases renewed before volunteers run this. An owner that leaves loses its layer's state since the last publish: the job fails and resumes from the checkpoint; nothing re-replicates the state. Browsers cannot own layers yet (the SDK has no owned kernels).

## Running it

```bash
cd relay && go build -o ../relay-server ./cmd/relay && go build -o ../latency-proxy ./cmd/latency-proxy && cd ..
./relay-server -addr 127.0.0.1:8101 -token T -max-mib 8192 &
ygaraty experiment http_profile --recipe transformer --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 \
  --workers 8 --workers-kind python --tiles 512 --max-tasks 16 --max-task-mib 512 --layer-task-mib 512 --output-task-mib 1024 \
  --steps 3 --repeats 1 --learning-rate 0.0005 --relay-url http://127.0.0.1:8101 --relay-token T \
  --rtt-ms 50 --down-mbps 100 --up-mbps 20 --float32-wire            # today's design, about 15 minutes
# add --owned-layers for the owned run (about 10 minutes)
```
