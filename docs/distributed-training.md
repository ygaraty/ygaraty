# Train with browser contributors

Start an experimental DiLoCo session from the command line. Each contributor
trains a full replica of a small residual ConvNet using Ygaraty's Tape,
compiler, browser kernels and Adam optimizer. The coordinator combines their
updates using outer Nesterov momentum. No PyTorch is needed.

From a checkout of this branch, install `python -m pip install '.[server]'`.
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

Tape differentiates the model and the compiler lowers the forward, backward
and Adam updates into portable tasks. Convolution and matrix multiplication
run on WebGPU; supported elementwise and optimizer tasks currently run in
JavaScript. A missing GPU fails explicitly. `--precision float64` selects the
JavaScript CPU reference path for debugging and CI.

The dataset has disjoint, deterministic worker shards. Each local step draws
a fresh batch; each shard reshuffles at epoch boundaries. There is no stochastic
augmentation or dropout. Held-out test loss and accuracy are measured before
training and after every committed round by default. Use `--eval-interval N`
to evaluate every N rounds; the final round is always evaluated. Evaluation uses the first
256 test examples; choose `--eval-examples 10000` for the complete test split.
Small subsets and short runs are functional checks, not convergence evidence.

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
to the Adam group; Muon's settings are separate. Both use zero weight decay in this experiment. The default optimizer
remains `adam`. Use a different run directory for each configuration; resume
requires the original optimizer and settings.

The compiler expresses momentum, normalization and the Newton–Schulz matrix
iterations as Tape operations after the model's backward pass. Matrix products
run through the WebGPU SDK; scalar/elementwise work uses the existing CPU
runtime. The implementation follows the [Muon reference algorithm](https://github.com/KellerJordan/Muon),
using float64 CPU and compensated WebGPU matrix arithmetic instead of its BF16
iteration. No PyTorch runtime is involved. Muon uses one momentum tensor per
convolution matrix; auxiliary Adam retains first and second moments. Both are
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
`optimizer_seconds` measures SDK execution of optimizer tasks, including their
GPU transfers, but excludes plan compilation and protocol overhead. Summed
worker times overlap for multiple contributors and are not wall-clock time.

## Resume and replace a contributor

Repeat the **same command and configuration**, adding `--resume`:

```sh
ygaraty train serve --dataset fashion-mnist --num-workers 2 \
  --rounds 10 --local-steps 5 --batch-size 8 \
  --run-dir runs/fashion-demo --share --open-browser --resume
```

Use the new link after restarting. The checkpoint includes global weights,
outer momentum, each logical worker's Adam moments and step count, metrics,
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
experiments; see the [public research list](../research/README.md#diloco-training).
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
