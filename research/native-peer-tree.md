# Native wgpu training with a direct peer tree

**Status:** local hardware-GPU numerical and transport smoke test passed for
1, 2, 4 and 10 worker processes. These all shared **one Apple M1 GPU on one
machine**. This does not establish physical multi-GPU scaling, Linux/NVIDIA
compatibility, convergence or a performance advantage.

## Question and implementation

Can our public Tape/compiler ConvNet train through native wgpu workers, binary
tensor frames and a real peer-to-peer reduction network without Chromium?

Yes, in the measured local configuration. `ygaraty train lan` compiles gradient
plans from the existing Tape forward/backward graph. Each Deno worker executes
its own batch shard with our WebGPU kernels. Ranks form a binary heap; each
parent directly requests its children's reduced gradients. Parents combine
sample-weighted means in deterministic rank order, including their own shard.
Only the root executes the compiled Adam update. Updated weights travel down
the same tree, with acknowledgements from every descendant before completion.

The coordinator sends control, plans and input batches and receives root weights
for evaluation. It does **not** receive or relay leaf gradients. This differs
from the earlier [coordinator-side tree experiment](tree-gradient-reduction.md).
Transport uses bounded, checksummed binary f64 tensor frames. Reduction and Adam
operate on host memory; this is not GPU-direct/RDMA. It is synchronous data
parallelism, separate from DiLoCo's local-update/outer-optimizer algorithm.

## Protocol

- Public source on `feature/public-diloco-training`, base
  `5441492d12651021ca345715d12d5e536fc3c410`, with uncommitted changes.
- Final runtime SHA-256:
  `14f4648635ae5642fff90a60c122cef95f7a51dc51620207e6d714eed43d200e`.
- Apple M1, 16 GiB unified memory; Deno 2.9.5 using native wgpu/Metal.
  Every worker reported `isFallbackAdapter: false`.
- Fashion-MNIST, checksum-verified source archives; residual ConvNet with
  channels 8/16/16, global batch 40, seed 2026, three shared Adam updates.
- Same global sample sequence at every worker count, 120 training examples per
  run; evaluate the same 16 held-out examples after each update.
- Every update compared against the unsplit Python reference; maximum absolute
  weight error must be at most `2e-5`. Same precision and optimizer settings.
- One sequential trial at each size, order 1/2/4/10; 300-second experiment limit.
  Compilation, transfers, verification and evaluation are included in wall time;
  worker startup and initial evaluation are excluded. Other test/build processes
  could run concurrently. Timings are diagnostic, not a controlled benchmark.

## Default-model results

| Worker processes | Wall time | Largest weight error vs CPU | Final evaluation loss |
|---:|---:|---:|---:|
| 1 | 4.85 s | 7.41e-7 | 2.292817880 |
| 2 | 6.87 s | 2.94e-7 | 2.292817881 |
| 4 | 4.19 s | 1.39e-6 | 2.292817882 |
| 10 | 5.35 s | 4.46e-7 | 2.292817880 |

All four met the numerical gate. Three updates and 16 evaluation samples are
far too few for a learning-quality conclusion. The smaller 2/4/4-channel smoke
matrix also passed, with maximum error below `6e-9`. An initial CPU-only matrix
validated the same protocol across separate processes and real loopback sockets.

Automated tests cover Python/TypeScript binary interchange, malformed/corrupt
frames, unequal shard weighting, direct parent-child routes, duplicate concurrent
step requests, wrong session/version, conflicting preparation and a disconnected
peer. A failed collective aborts the session. Partial weight broadcasts require
fresh processes and a new session; there is no recovery claim.

## Reproduction and evidence

See the [LAN user guide](../docs/native-training.md) for multi-machine commands.
The local default-model check is:

```bash
PYTHONPATH=. python tools/native_tree_pilot.py \
  --workers 1 2 4 10 --dataset fashion-mnist \
  --global-batch 40 --channels 8 16 16 \
  --output-dir runs/native-default-model
```

[Protocol and source identity](results/native-peer-tree/protocol.json),
[summary](results/native-peer-tree/summary.json), and
[individual reports, weights and cleanup records](results/native-peer-tree/)
preserve both earlier checks and the final runtime checks. Setup failures are
recorded separately; no failed training run was silently excluded.

All final worker processes stopped, and all listening ports closed. No paid
resources were created; external cost was **$0**. Session configurations,
credentials and host inventories are not included in the public evidence.

## Next physical-GPU experiment

Use one GPU per host on the same private network; verify GPU identity, driver,
hardware adapter and peer connectivity first. Compare 1/2/4/10 with fixed global
batch, model, sample sequence, training steps and evaluation settings. Run three
repetitions in alternating order. Measure latency/bandwidth and time to the same
observed evaluation loss. Retain failed runs and report storage/network cost.
The small model may be communication-bound and need not get faster with ten GPUs.
