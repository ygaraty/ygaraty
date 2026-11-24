# Native training contributors on a LAN

Experimental synchronous ConvNet training with **1, 2, 4 or 10 native GPU
workers**, our Tape/compiler and the public WebGPU kernels. Deno uses wgpu
directly: no Chromium, browser profiles, WebRTC or TURN is needed.

Each worker holds the whole model and trains a shard of the same global batch.
Gradients move directly up a binary peer tree. The root runs one shared Adam
update and broadcasts weights back down the tree. This is a separate experiment
from [DiLoCo's local training rounds](distributed-training.md).

## Prepare the machines

Install the same public Ygaraty revision and Deno on every worker. Deno **2.9.5**
is locally tested. On Linux, the GPU driver must expose Vulkan/WebGPU; installing
CUDA alone does not establish that. Workers reject unavailable and software
fallback adapters. A [RunPod RTX 4090 preflight](../research/runpod-native-preflight.md)
passed on Ubuntu 24.04 with Deno 2.9.5. A
[one-versus-four physical RTX 3060 comparison](../research/vast-native-one-vs-four.md)
also passed numerical checks: learning quality stayed close, but four workers
were slower over the tested internet links. Low-latency LAN scaling remains
unmeasured; see also the [local pilot](../research/native-peer-tree.md).

On a minimal Ubuntu GPU container, install `libvulkan1`, `vulkan-tools` and
`libegl1` alongside the injected NVIDIA graphics libraries. Our initial container
could see the GPU in `nvidia-smi` but selected software Vulkan until the EGL
loader was installed. Confirm the adapter name and hardware status before a run.

Use one worker per physical GPU initially. A machine with several GPUs needs
explicit device isolation: launching several workers does not automatically
assign them different GPUs. Record actual GPU identities before interpreting
scaling results.

Workers must reach each other directly on their private addresses. Allow the
chosen TCP port between cluster machines, and allow the coordinator to reach
every worker. The example uses HTTP on a trusted private LAN/VPC. For connections
across the public internet, use a private encrypted network. Configurations
contain a shared session secret; transfer them privately and never commit them.

## Create a session

On the coordinator, supply one reachable URL per worker. `--num-workers` is
mandatory; supported values are **1, 2, 4 and 10**.

```bash
ygaraty train lan init --num-workers 2 \
  --hosts http://192.168.1.10:8780 http://192.168.1.11:8780 \
  --directory lan-session
```

This writes `coordinator.json` and `worker-0.json`, `worker-1.json`, etc. with
private file permissions. Copy each worker's file to its matching machine.
The source fingerprint must match on every machine. Generate a new session after
changing the runtime source. The coordinator needs only Python and NumPy; workers
add Deno and a working GPU driver.

## Start contributors from the command line

On the first worker:

```bash
ygaraty train lan worker --config lan-session/worker-0.json \
  --bind 192.168.1.10 --port 8780
```

On the second worker:

```bash
ygaraty train lan worker --config lan-session/worker-1.json \
  --bind 192.168.1.11 --port 8780
```

Each command prints its rank and detected GPU once ready. The default bind
address is loopback; set the private interface for a multi-machine run. Stop
contributors with Ctrl-C after the experiment. Worker processes remain available
for inspection after completion, but each new experiment needs fresh processes
and a new session.

## Run training

```bash
ygaraty train lan run --config lan-session/coordinator.json \
  --num-workers 2 --dataset fashion-mnist \
  --global-batch 40 --steps 100 --eval-interval 20 \
  --eval-examples 1024 --verify-steps 3 \
  --output runs/lan-2/report.json
```

Dataset downloads are checksum-verified. The driver writes a JSON report and
final weights in an NPZ file. Reports contain worker adapter/runtime details,
per-step loss, evaluation loss/accuracy, elapsed time, tensor-body byte counts,
sample sequence and source hashes. `--verify-steps` compares initial updates
against the full-batch Python reference, with a maximum absolute weight-error
gate of `2e-5`. Verification and evaluation time are included in reported wall
time, so use the same settings for comparisons.

The default residual ConvNet uses channels `8 16 16`; change them with
`--channels`. Global batch size must be between the worker count and 64. It need
not divide evenly: the reduction weights each shard by its sample count.

For a 1/2/4/10 comparison, repeat session creation and training with each count
and its matching host list. Keep the global batch, steps, model, dataset, seed,
precision and evaluation settings fixed. Use at least three repetitions and
alternate measurement order. Compare time to the same observed evaluation loss
as well as throughput; three smoke-test updates do not show convergence.

## Local protocol pilot

From a public source checkout:

```bash
PYTHONPATH=. python tools/native_tree_pilot.py \
  --workers 1 2 4 10 --dataset fashion-mnist \
  --output-dir runs/native-pilot
```

This launches isolated local Deno processes, checks three updates against the
reference, then stops them and verifies their ports have closed. It uses one
physical GPU and **does not measure multi-GPU scaling**. Add `--cpu-test` to test
the protocol without a GPU; CPU reports are explicitly marked `float64`.

## Current limits

- Tensors use versioned binary little-endian float64 frames, with bounded sizes,
  shape/finite checks and payload checksums. GPU kernels compute in float32;
  transport does not quantize their results further. Tensor data is not JSON or
  base64. Input plans and batches still come from the coordinator each step.
- This is a host-memory HTTP collective, not GPU-direct/RDMA. Reduction and Adam
  run on worker CPUs; convolution/matrix kernels run on the GPU. The root sends
  final weights to the coordinator for evaluation, never leaf gradients.
- Membership is fixed. A missing peer, rejected tensor or partial broadcast
  aborts the run. Duplicate step requests reuse the same result. There is no
  resume, elastic membership or rollback after a partial broadcast: restart all
  workers with a new session. Failed reports are retained.
- The included Fashion-MNIST model is small. Ten GPUs may be slower because
  communication dominates; a successful run is not a speedup claim.

For cloud experiments, use same-region/private-network machines, check quota and
availability, measure actual latency/bandwidth, and arrange automatic resource
destruction independently of the training process. A VPC is not proof of a
particular LAN bandwidth or physical placement. Stopping training does not stop
cloud billing.

## Native DiLoCo and Muon experiments

The source-checkout harness also runs the transactional DiLoCo protocol through
Deno, using the same Tape/compiler and training client as browser contributors.
This path uses JSON tensors over coordinator HTTP; it does not use the binary
peer tree described above.

On a Linux host with four NVIDIA GPUs, Vulkan tools and Mesa's device-select
layer, first verify that each process can see exactly one distinct physical GPU:

```bash
python tools/probe_training_gpus.py --expected 4 --output runs/gpu-private
```

Keep that directory private: its probe report includes GPU identifiers. The
probe writes `selectors-1.json` and `selectors-4.json` for the harness. A failed
probe must be fixed before interpreting worker counts as physical GPU counts.

Check the one-worker numerical result against the Python reference:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. python tools/training_e2e.py \
  --runtime deno --num-workers 1 \
  --gpu-selectors runs/gpu-private/selectors-1.json \
  --precision webgpu --optimizer muon --channels 8 16 16 \
  --batch-size 16 --local-steps 2 --rounds 2 --timeout 300 \
  --output runs/muon-parity-1.json
```

Repeat with `--num-workers 4`, `selectors-4.json`, and a separate output file.
Both must pass the maximum absolute weight-error gate of `5e-5`. Without a
selector file, Deno chooses its default GPU; several processes may share it.

For measured training, add `--training-run --dataset fashion-mnist`. Match
`num_workers × rounds × local_steps × batch_size` between arms and evaluate at
matching aggregate sample counts. Keep optimizer settings and seed fixed within
each pair, alternate run order across seeds, and retain failed trials. The
harness stops its contributor processes and coordinator after each run; it does
not destroy cloud rentals.

### GPU-resident local training

Supported ConvNet rounds now execute the Tape/compiler graph on GPU across all
local steps. Weights and optimizer moments upload once per round; only batch
constants upload for each step. Forward/backward intermediates stay on-device,
and scratch buffers are reused after their last reader.

One final readback retrieves losses, weights and optimizer state. The trainer
publishes them only after the complete round succeeds. Cancellation, device loss
or a nonfinite intermediate leaves the previous checkpoint unchanged. Global
rebasing, outer DiLoCo updates and held-out evaluation retain their existing path.

Convolutions retain FP32 arithmetic. Dense, elementwise, loss and optimizer
operations use compensated FP32 pairs. The executor accepts bounded portable
plans: at most 20,000 tasks, 64 MiB of input uploads and 64 MiB of uniform metadata
per round, with a 256 MiB tensor-buffer arena. Unsupported operations or oversized
plans take the existing per-step path before GPU execution starts; execution
errors fail the round rather than silently switching arithmetic.

```sh
PYTHONPATH=. python tools/check_resident_training.py --output /tmp/resident-check.json
PYTHONPATH=. python tools/benchmark_resident_training.py \
  --num-workers 1 --samples 768 --output /tmp/resident-comparison
```

The Deno benchmark control `--training-execution tasks` keeps the previous
per-step path, including the earlier resident Muon optimization. The default is
`--training-execution resident`. Optimizer-only timings are omitted for full
resident rounds because GPU timestamps are not collected.

### Muon-only execution

The per-step fallback keeps each supported Newton–Schulz block on the GPU,
with one submission and one readback per convolution matrix. The compiler keeps
portable Tape operations; compensated arithmetic and optimizer equations are
preserved. Momentum, normalization and final weight updates still run on the host.

This path supports whole matrices that fit a single compiler tile, with a 64 MiB
intermediate allocation bound. Tiled or unsupported graphs use the existing task
executor; GPU execution errors still fail the operation. Float32 requests retain
the existing execution path. Buffers are scoped to a block, not cached across steps.

Run `PYTHONPATH=. python tools/benchmark_muon.py --output /tmp/muon.json` for local
numerical checks and paired timings. See the
[local results](../research/muon-resident-local.md) and
[NVIDIA results](../research/muon-resident-nvidia.md) for the earlier Muon-only measurements and reproduction commands.
The [full-round comparison](../research/resident-training-local.md) measures the
additional benefit of keeping forward/backward and state on GPU.

The [full-round NVIDIA comparison](../research/resident-training-nvidia.md)
checks one/four physical RTX 3060 GPUs against upstream MuLoCo-1 across three seeds.

### Coordinator profiling

Training assignment requests now wait up to 20 seconds for work and wake when
contributors join or a round commits or aborts. Authentication is rechecked after
waking. Assignment JSON is cached only for the current round, up to 16 MiB per
worker, and cleared on commit or rollback. Checkpoint durability is unchanged.

Use `tools/profile_training_coordinator.py` for phase timers or its separate
`--preparation-only` CPU profile. Use `tools/benchmark_training_coordinator.py`
with a baseline checkout for unprofiled comparisons. The
[local report and reproduction commands](../research/coordinator-overhead-local.md)
include source hashes, retry/recovery checks and all three-seed measurements.
