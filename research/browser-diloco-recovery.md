# Browser DiLoCo: public service and recovery checks

Date: 2026-11-19. Tested implementation: a local snapshot before the DiLoCo merge
into `main`. These are historical measurements, not a retest of current main.
This checks the new public training path, including its TypeScript-generated
browser modules. It does not establish convergence or
multi-machine speedup.

## Question

Can one or two real Chromium WebGPU contributors train a Tape-compiled residual
ConvNet over HTTP, retain their Adam state across rounds, and produce the same
model as the public Python reference?

## Method

Run `tools/training_e2e.py` with `--dataset fashion-mnist --precision webgpu`
and `--num-workers 1` / `2`. Both runs used the same Mac GPU, reported by WebGPU
as Apple / Metal 3, with `isFallbackAdapter: false`. Each contributor used an
isolated Chromium profile. Convolutions and matrix multiplication ran on GPU;
other supported tasks ran in JavaScript. Each worker reported 42 GPU tasks and
168 CPU tasks per round.

The small check model uses channels 2/4/4, two local steps per round, batch
size 2, two rounds, and seed 2026. It uses the verified full Fashion-MNIST
training files; deterministic disjoint shards select the small number of
examples consumed in this check. Held-out evaluation uses only the first eight
test examples. A separate Python run uses identical initialization, shards and
optimizer settings. The harness requires maximum absolute final-weight error
at most `5e-5` for WebGPU.

Implementation SHA-256:
`d195c78a5bea605737ec98baa2ff58873ab71fb2e63f412c3988850274060a13`.
The aggregate hashes sorted package-relative filenames and contents of the
installed `.py` and `.mjs` files. Dataset SHA-256 is in the raw reports.

## Observed results

| Contributors | Completed rounds | Maximum absolute weight difference | Result |
| --- | --- | --- | --- |
| 1 | 2 | 2.47e-9 | Passed |
| 2 | 2 | 1.42e-9 | Passed |

Raw reports: [one worker](results/browser-diloco/fashion-1-worker.json),
[two workers](results/browser-diloco/fashion-2-workers.json).

The test suite separately checks numerical gradients with finite differences,
HTTP JavaScript/Python parity with a deliberately lost result acknowledgement,
idempotent and conflicting retries, stale results, round deadlines, expired
leases, replacement workers, aborted-round rollback, durable restart and
checkpoint publication failure. These fault tests use synthetic data and do
not simulate a WAN.

## Limits and next experiment

These very short runs are correctness tests, not accuracy benchmarks. The
one- and two-worker harnesses overlapped on the same machine, so their recorded
wall times must not be interpreted as scaling results. No paid remote compute
was used.

Next: place the coordinator and contributors on separate hosts, freeze the
implementation and dataset hashes, run a single-worker baseline and two-worker
DiLoCo under equal sample budgets, measure bytes transferred and time to a
specified held-out accuracy, and interrupt a contributor mid-round. Repeat
across heterogeneous GPUs before making reliability or scaling claims.

See [the user guide](../docs/distributed-training.md) for launching and resuming
sessions. Real multi-machine experiments still require hosts or an explicit
compute budget.
