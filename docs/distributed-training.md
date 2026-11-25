# DiLoCo training: implementation and results

Start an experimental DiLoCo session from the command line. Each contributor
trains a full replica of a small residual ConvNet using Ygaraty's Tape,
compiler and WebGPU engine. Choose Adam (default) or Muon with auxiliary Adam;
the coordinator combines model deltas using outer Nesterov momentum. No PyTorch
is needed for Ygaraty training; PyTorch is used only for independent comparisons.

This guide describes experimental training on `main`, including the GPU-resident
executor and coordinator optimizations. Install from a current source checkout;
historical benchmark results identify their own tested snapshots. See [measured results](#measured-results) and
[implementation reference](https://github.com/ygaraty/ygaraty/blob/main/ygaraty/training/README.md) for scope and code locations.

[Start a session](#start-a-browser-session) · [How rounds work](#how-a-diloco-round-works) ·
[GPU execution](#gpu-residency-and-coordination) · [Muon](#experimental-muon-local-optimizer) ·
[Results](#measured-results) · [Recovery](#resume-and-replace-a-contributor)

(start-a-browser-session)=
## Start a browser session

From this source checkout, install `python -m pip install -e '.[server]'`.
Then run:

```sh
ygaraty train serve --dataset fashion-mnist --num-workers 2 \
  --rounds 10 --local-steps 5 --batch-size 8 \
  --run-dir runs/fashion-demo --share --open-browser
```

The command downloads and verifies the public dataset, prints a contributor
link, and waits for both contributors. Open the link in Chrome/Chromium and
click **Start contributing**. Share it with another contributor, or open it
in another tab yourself. Keep the tabs open. A shared link grants access to
join this training session; share it with people you trust.

`--share` generates HTTPS through the same Cloudflare quick tunnel used by
inference launch. It needs `cloudflared` installed; no HTTPS proxy configuration
is required. Omit `--share` for a local `http://127.0.0.1:8766` session.
WebGPU requires HTTPS or localhost. Plain HTTP on a remote LAN address will
not provide WebGPU.

Use `--dataset mnist` for handwritten digits, and choose `--num-workers 1`,
`2`, `3` or `4`. Worker count is mandatory and stays fixed throughout a run.
Every worker holds a **complete replica**, including optimizer state; adding
workers does not divide the model's memory footprint. This experiment uses
HTTP for whole local rounds, not the inference worker's WebRTC transport.

## What runs

The default model has 3×3 convolutions with 8/16/16 channels, ReLU, two max
pools, a residual connection around the final convolution, global average
pooling and a ten-class head. It is a compact residual ConvNet research recipe,
not a reproduction of ConvNeXt or a state-of-the-art benchmark.

Tape differentiates the model and the compiler lowers forward, backward and
optimizer updates into portable tasks. Supported rounds execute these tasks on
WebGPU, retaining weights, activations, gradients and optimizer moments across
local steps. A missing GPU fails explicitly. `--precision float64` selects the
JavaScript CPU reference path for debugging and CI.

The dataset has disjoint, deterministic worker shards. Each local step draws
a fresh batch; each shard reshuffles at epoch boundaries. There is no stochastic
augmentation or dropout. Held-out test loss and accuracy are measured before
training and after every committed round by default. Use `--eval-interval N`
to evaluate every N rounds; the final round is always evaluated. Evaluation uses the first
256 test examples; choose `--eval-examples 10000` for the complete test split.
Small subsets and short runs are functional checks, not convergence evidence.

(how-a-diloco-round-works)=
## How a DiLoCo round works

1. The coordinator assigns the same committed global weights to every logical
   worker, together with that worker's optimizer state and deterministic batches.
2. Each worker runs `--local-steps H` forward/backward/optimizer updates on its
   own data shard. Workers compute concurrently and keep separate inner moments.
3. Each returns a **model delta**: starting global weights minus its final local
   weights. This is not a per-minibatch gradient all-reduce.
4. After every assigned worker finishes, the coordinator averages the deltas in
   deterministic worker order and applies the outer update. Defaults are
   `--outer-learning-rate 0.7` and `--momentum 0.9` (Nesterov).
5. The coordinator atomically saves the new global weights, outer momentum and
   every worker's optimizer state. Only then can workers start the next round.
   Inner moments survive synchronization; weights rebase to the new global model.

With `K` workers, `R` rounds, `H` local steps and batch size `B`, a run processes
`K × R × H × B` training samples. Increasing `K` at a fixed sample budget changes
sharding and the number of outer updates, so learning curves need not match.
Each worker must fit the **whole model and its training state**. Inference's
layer partitioning does not make this training path model-parallel.

(gpu-residency-and-coordination)=
## GPU residency and coordination

Weights and optimizer moments upload once per local round; batches upload as
needed. Supported forward/backward and optimizer intermediates stay on GPU,
with reusable scratch buffers and one final readback for losses, weights and
moments. Residency spans the local steps, not global synchronization. Outer
updates, evaluation and durable checkpoints still run on the coordinator.

Convolutions use FP32; other supported operations use compensated FP32 pairs.
Unsupported or oversized plans use the portable per-task path before execution
starts. GPU execution errors fail the round. See [execution limits and controls](native-training.md#gpu-resident-local-training).

The coordinator sends protocol JSON directly, caches immutable assignment
responses for retries, and wakes waiting requests when work is ready. Waits are
bounded to 20 seconds and release the session lock. The cache is cleared on
commit or abort and retains at most 16 MiB per worker. Larger responses work
without byte caching. Authentication is rechecked after waking; checkpoint
publication remains durable. DiLoCo currently uses coordinator HTTP with JSON
tensors, not the separate binary peer-tree transport.

Browser contributors use `ygaraty train serve`. The public test harness also
runs **Deno/wgpu contributors without Chromium**; see [native DiLoCo experiments](native-training.md#native-diloco-and-muon-experiments).
`ygaraty contribute` in the inference quickstart is not a training contributor
command. The small `ygaraty train --workers ...` example is a sequential local
CPU demonstration, not a distributed GPU run.

(experimental-muon-local-optimizer)=
## Experimental Muon local optimizer

Use `--optimizer muon` to apply Muon to convolution weight matrices while
keeping Adam for biases and the classification head. The outer DiLoCo update
remains Nesterov; each contributor still holds a complete model replica.

```sh
ygaraty train serve --dataset fashion-mnist --num-workers 1 \
  --optimizer muon --muon-learning-rate 0.02 --learning-rate 0.001 \
  --muon-momentum 0.95 --muon-steps 5 \
  --rounds 120 --local-steps 8 --batch-size 16 \
  --run-dir runs/fashion-muon --open-browser
```

`--learning-rate` controls auxiliary Adam, and `--muon-learning-rate` controls
Muon. In the Python recipe, the existing `optimizer` options dictionary applies
to the Adam group; Muon's settings are separate. Both use zero weight decay
in this experiment. The default optimizer remains `adam`. Use a different run directory for each configuration; resume
requires the original optimizer and settings.

The compiler expresses momentum, normalization and the Newton–Schulz matrix
iterations as Tape operations after the model's backward pass. The resident
executor keeps those operations on GPU, including momentum and normalization.
Our [Muon implementation](https://github.com/ygaraty/ygaraty/blob/main/ygaraty/compiler/muon.py) uses compensated arithmetic
rather than the upstream MuLoCo-1 reference's BF16 Newton–Schulz iteration;
its momentum and scaling also differ. This is not a bitwise reproduction of
upstream MuLoCo-1. No PyTorch runtime is involved. Muon uses one momentum tensor
per convolution matrix; auxiliary Adam retains first and second moments. Both are
saved in transactional checkpoints and restored on contributor replacement.

To reproduce the bounded, sequential one/three-worker comparison:

```sh
PYTHONPATH=. python tools/muon_experiment.py --seeds 2026 2027 \
  --samples 15360 --output runs/muon-comparison
```

This uses real Chromium WebGPU contributors on the same physical GPU. It writes
its protocol before starting, retains individual reports, and compares first
observed times to loss 1.5 and 1.0. An unreached target stays `null`; it is not
extrapolated. Every run uses the same sample budget and evaluation frequency.
Optimizer arms at a given worker count share initialization and sample streams;
changing worker count changes data shards and the number of outer updates.
The fixed learning rates are a pilot, not a tuned optimizer comparison.
Full resident rounds report task, submission, readback and transfer counters.
They omit `optimizer_seconds`: GPU timestamps are not collected, so optimizer-only
time cannot be separated reliably. Older per-task reports contain that field.
Summed worker times overlap for multiple contributors and are not wall-clock time.

(measured-results)=
## Measured results

### Physical GPUs: GPU residency, before the latest coordinator fixes

Three seeds, 15,360 Fashion-MNIST samples per trial; four RTX 3060 GPUs on one
host. ConvNet channels 8/16/16, batch 16, eight local steps, 120/30 rounds for
one/four workers. Evaluation uses 1,024 images every 1,536 training samples.

| Implementation | GPUs | Median time | Final loss | Accuracy | Time to loss ≤1.5 |
|---|---:|---:|---:|---:|---:|
| Ygaraty DiLoCo + Muon | 1 | 239.57 s | 1.2200 | 60.06% | 47.93 s |
| Ygaraty DiLoCo + Muon | 4 | 171.27 s | 0.7710 | 73.34% | 53.54 s |
| PyTorch MuLoCo-1 | 1 | 15.05 s | 1.1393 | 59.38% | 3.53 s |
| PyTorch, Ygaraty optimizer equations | 1 | 14.52 s | 1.3184 | 60.45% | 3.18 s |

Four Ygaraty workers finished the sample budget **1.40× sooner**, with lower
final loss, but took **12% longer to reach loss 1.5** than one Ygaraty worker.
They did not beat single-GPU MuLoCo-1 in runtime or time to that target.
Loss 1.0 was reached in 3/3 four-worker Ygaraty trials, 1/3 one-worker Ygaraty
trials and 2/3 upstream trials; these are first observed crossings, not sustained
thresholds. Optimizer differences and changed outer-update counts prevent an
optimizer-superiority claim.

Ygaraty timing includes contributor startup, training and evaluation; PyTorch
reports setup separately. Dataset loading and initial Ygaraty evaluation are
excluded. [All trials, timing definitions and reproduction commands](https://github.com/ygaraty/ygaraty/blob/main/research/resident-training-nvidia.md).

### Latest coordinator fixes: local validation only

After that remote run, profiling led to direct JSON responses, immutable response
caching and event-driven waits. Three-seed medians on **one Apple M1 GPU**, with
3,072 samples per trial:

| Contributor processes | Before | After | Speedup |
|---|---:|---:|---:|
| 1 | 11.21 s | 6.88 s | 1.63× |
| 4 | 9.34 s | 3.93 s | 2.38× |

All recorded local-step losses, evaluation losses and accuracies matched exactly.
The four processes share one GPU: this is an execution-path comparison, not
four-GPU scaling. **The updated coordinator has not been benchmarked on NVIDIA.**
Do not apply these multipliers to the remote times above.
[Profile, correctness checks and exact baseline reconstruction](https://github.com/ygaraty/ygaraty/blob/main/research/coordinator-overhead-local.md).

### What remains open

- WAN throughput, mixed-GPU scheduling and straggler handling for this resident
  ConvNet implementation. Fixed synchronous rounds wait for every worker.
- Larger models, longer training and tuned time-to-loss comparisons. Short
  Fashion-MNIST runs do not establish large-model convergence.
- Plan serialization, tensor payload size and local executor efficiency. GPU
  residency removes intermediate readbacks, not all host or network overhead.
- Model partitioning for training: the current DiLoCo path uses full replicas.

Protocol checks validate shapes, finite values and session identity; they do not
prove that a trusted contributor computed honest updates. Older research reports
may test other models or transports; their results do not validate every current
path. [Browse the public DiLoCo experiments](https://github.com/ygaraty/ygaraty/blob/main/research/README.md#diloco-training).

(resume-and-replace-a-contributor)=
## Resume and replace a contributor

Repeat the **same command and configuration**, adding `--resume`:

```sh
ygaraty train serve --dataset fashion-mnist --num-workers 2 \
  --rounds 10 --local-steps 5 --batch-size 8 \
  --run-dir runs/fashion-demo --share --open-browser --resume
```

Use the new link after restarting. The checkpoint includes global weights,
outer momentum, each logical worker's Muon/Adam moments and step count, metrics,
and retry receipts. Dataset SHA-256, implementation SHA-256, seed and config
are checked before resuming. The worker's step count determines its exact data
cursor and shuffle epoch. A replacement browser inherits that logical slot's
committed optimizer and sample stream.

`--round-timeout 600` bounds a round. `--lease-seconds 45` detects a disconnected
contributor. Either failure aborts the entire tentative round; all participants
restore the last committed generation. A timed-out pending slot can be filled
by another browser. After `--max-retries 3`, the session stops training and asks
for a restart. It never averages a partial membership silently. Transient
network failures retry the same result without running another optimizer step.

The server prints committed-round metrics and saves an atomic
`checkpoint.json` under the run directory. Stop it with Ctrl-C. Interrupted
rounds are replayed after resume; completed rounds remain committed. Run one
coordinator per run directory. Do not edit a checkpoint or change the installed
implementation while a session is running.

## Reproduce a browser check

With Chrome/Chromium installed:

```sh
PYTHONPATH=. python tools/training_e2e.py --num-workers 2 \
  --dataset fashion-mnist --precision webgpu --output runs/fashion-check.json
```

Set `YGARATY_CHROME` if the executable is not found automatically. This bounded
check launches isolated Chromium contributors, completes two rounds, compares
weights against the Python reference, records actual CPU/GPU task counts and
adapter details, and closes its browsers and server. Use one worker to compare
the same path without synchronization. The `fixture` dataset is explicitly
synthetic; `--precision float64` exercises a real browser using CPU arithmetic.

The optional GitHub **Training browser E2E** workflow runs one- and two-worker
checks weekly and on manual dispatch, using browser CPU execution on hosted
runners. Real GPU checks can be run locally or on provisioned GPU hosts.

This is trusted-contributor open research. Protocol validation does not prove
that a participant computed honest gradients. Multi-machine throughput,
heterogeneous GPU stability and time to a target accuracy remain separate
experiments; see the [public research list](https://github.com/ygaraty/ygaraty/blob/main/research/README.md#diloco-training).
Dataset sources and published checksums:
[MNIST resources](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py)
and [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

## Compare one and three local contributors

For a bounded learning experiment instead of a short numerical check, use
`--training-run`. This skips the full Python replay; numerical parity should be
checked separately with the short default run. For equal sample counts:

```sh
PYTHONPATH=. python tools/training_e2e.py --training-run \
  --dataset fashion-mnist --precision webgpu --num-workers 1 --rounds 120 \
  --local-steps 16 --batch-size 16 --channels 8 16 16 \
  --eval-examples 1024 --eval-interval 6 --timeout 900 --output runs/one.json

PYTHONPATH=. python tools/training_e2e.py --training-run \
  --dataset fashion-mnist --precision webgpu --num-workers 3 --rounds 40 \
  --local-steps 16 --batch-size 16 --channels 8 16 16 \
  --eval-examples 1024 --eval-interval 2 --timeout 900 --output runs/three.json

python tools/compare_training.py runs/one.json runs/three.json \
  --target-loss 1.5 --output runs/comparison.json
```

Run these sequentially. Each arm processes 30,720 examples and evaluates after
every 1,536 examples. The reports include observed time to a common loss,
final loss/accuracy, sample counts and HTTP body bytes. The comparator rejects
curves missing the final committed evaluation. Add `--plot runs/curves.png`
if Matplotlib is installed.

All contributors launched by this helper share your machine's GPU. This measures
same-host concurrency and sample efficiency, not three-GPU scaling. Equal
sample counts also mean fewer global updates with three workers; identical
accuracy is not guaranteed. A single seed and half an epoch are a pilot, not a
convergence benchmark. Initial setup/data loading and initial evaluation are
outside the reported timer; browser startup, training, periodic/final evaluation
and final checkpoint completion are included.

## Experimental tree gradient reduction

To isolate batching from DiLoCo's local optimizer updates, the experimental
runner computes gradients on browser workers and combines them in a pairwise,
sample-weighted tree before **one shared Adam update**. The tree currently runs
on the coordinator CPU; its HTTP transport is a star, not a peer-to-peer tree.

First check the GPU gradients against the full-batch CPU reference:

```sh
PYTHONPATH=. python tools/tree_gradient_experiment.py --num-workers 3 \
  --steps 5 --verify-steps 5 --eval-every 5 --eval-examples 48 \
  --output runs/tree-probe.json
```

Then run these sequentially to compare identical global batch sequences:

```sh
PYTHONPATH=. python tools/tree_gradient_experiment.py --num-workers 1 \
  --output runs/tree-one.json
PYTHONPATH=. python tools/tree_gradient_experiment.py --num-workers 3 \
  --output runs/tree-three.json
python tools/compare_gradient.py runs/tree-one.json runs/tree-three.json \
  --output runs/tree-comparison.json
```

Defaults are Fashion-MNIST, global batch 48 (48 on one worker or 16 on each of
three), 640 shared Adam steps, seed 2026, and a 15-minute deadline per run. This
uses the same 30,720 example IDs in the same order in both configurations.
Evaluation uses 1,024 held-out examples every 32 updates. The runner saves JSON
metrics and `.npz` gradient/weight snapshots. The comparator checks source,
dataset and sample-order hashes before comparing. Add `--plot runs/tree.png`
with Matplotlib installed for the learning curves.

No local Adam or DiLoCo outer update runs at a leaf. The shared optimizer uses
Ygaraty's existing compiler and local runtime. This bounded experimental runner
aborts an incomplete step; it is not the resumable `train serve` service. All
browser workers share the host GPU, so this does not establish physical-GPU
scaling or communication savings from a peer-to-peer tree.

## Native LAN experiment

Use [native training contributors](native-training.md) for Deno/wgpu workers,
binary tensor transport and direct peer-tree reduction with 1, 2, 4 or 10 workers.
This synchronous shared-Adam experiment is separate from DiLoCo rounds.
