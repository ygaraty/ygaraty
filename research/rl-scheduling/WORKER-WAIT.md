# Reducing worker noncompute time: offline protocol sweep

2026-10-30, base `31e48c7`. Four unchanged pruned local graphs, critical-path
scheduling, three network scenarios, eight protocol configurations and three seeds
(noiseless 0; noisy 1800/1801) produced **288 offline rollouts**. No numerical
workloads, workers, relays, training jobs or network emulators ran.

## Finding

Communication and claim settings can reduce modeled noncompute time substantially
relative to these corpora's conservative baseline: one task per claim, separate
input fetching, no eager claims. These are **not incremental gains over an already
optimized production setup**. Transformer experiments 51/54 already use pushed
inputs and eager claims. Transformer plans were not tested here.

Mean deterministic makespan gains across the four graphs (equal weighting):

| Change from baseline | LAN-4 | Home-4 | Heterogeneous-8 |
|---|---:|---:|---:|
| Pushed inputs only | 2.92% | 23.76% | 20.96% |
| Claim batches of 4 | 9.36% | 29.63% | 22.32% |
| Batches of 4 + streamed posting | 8.68% | 33.61% | 28.29% |
| Batches of 4 + eager claims | 2.38% | 38.22% | 60.64% |
| Pushed inputs + batches of 2 + eager | −2.51% | 49.60% | 67.44% |
| Pushed inputs + batches of 4 + eager | 4.99% | 50.05% | 68.36% |
| Pushed inputs + batches of 8 + eager | 8.61% | 49.36% | 66.47% |

Averages hide regressions: pushed inputs + batches of 4 + eager ranges from
−9.81% to +22.79% on LAN. Larger batches are not universally better; they reserve
ready work for a worker and can harm balance. These settings are candidates for
per-scenario tuning, not a new global default. The two noisy seeds are only a
sensitivity check; different protocol paths consume randomness differently.

Example: MLP-340, Home-4, noiseless, baseline versus pushed inputs/batch 4/eager:

- Makespan: **13.56 → 4.45 seconds**, a modeled 67.20% improvement.
- Compute utilization: **7.37% → 22.47%**.
- Noncompute worker-seconds: **50.24 → 13.79** (sum across four workers).
- Mean compute time per worker stays 1.00 s. Mean fetch time drops 3.64 → 0.81 s;
  uploading in the compute/claim loop drops 4.22 → 0 s, while posting takes place
  asynchronously instead. Upload work did not disappear.

The likely lever is avoiding serialized claim/fetch/upload waits and overlapping
posting with useful work. This is model attribution, not a measured diagnosis of
the current deployment or the new keep-alive implementation.

## What “waiting” means here

`noncompute_worker_s = worker_count * makespan - sum(worker_compute_s)`.
This includes communication, startup, dependency stalls, queueing and end-of-job
imbalance. It is **not a direct measurement of pure dependency idle time**.
Compute utilization is the complementary fraction over the same full simulation
horizon. Optimize makespan first; maximizing utilization alone can reward useless
work or slower execution on a different worker.

The simulator also reports claims, empty claims, sleeping, fetch, upload-in-claim,
posting and stream-upload-queue intervals. Posting overlaps computation. These
stage totals must not be summed as an exclusive wall-time partition; unfinished
wait intervals at termination may not appear in stage totals. Per-worker compute
times, all stage diagnostics and the modeled critical chain are saved per rollout.

`collect(..., diagnostics=True)` exposes these diagnostics without changing its
trajectory or default return format. A focused test checks trajectory identity,
known compute work and the capacity-time accounting identity.

## Limits and next step

This sweep reuses existing local cost/network models, float64 transport, fixed
worker counts and cache settings. It does not calibrate current HTTP connection
reuse, negotiated chunk sizes, uploader CPU or total worker memory. Increased
claim/eager queues need memory/backpressure validation before deployment; no
worker-memory-cap guarantee is made. Fused-factorization plans remain blocked
because their required-output contracts are missing.

For an already pushed/eager runtime, the next useful experiment is **adaptive
claim depth**: small batches when the ready frontier is narrow or workers differ,
larger batches when independent work and RTT justify it, with bounded queued
bytes and prompt critical-path result publication. Compare against the actual
configured baseline, not this study's conservative control. Do not replace the
scheduler or train an RL policy before checking whether simple batching rules
capture the improvement.

## Reproduction and artifacts

```bash
.venv/bin/python -m ygaraty.simulation.worker_wait \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-worker-wait-run
```

This metadata-only path uses no database/dataset connection. It requires existing
local corpus/cost files and hash-matching legacy MLP sources for root recovery.
The entry point blocks socket connections and numerical kernels. No source plan
is modified; the same pruned graph hash is used in every configuration.

Results are in workspace-parent `calibration-audit/rl-scheduling/worker-wait-v1/`
and the adjacent ZIP. Plans and raw diagnostic rollouts are compressed. The
bundle includes costs, source, report, verification results and checksums. Three
focused dataset tests passed; no numerical workload suite was run.

A copy of the compressed evidence bundle is tracked at
[`artifacts/worker-wait-v1.zip`](artifacts/worker-wait-v1.zip) for inspection on
another machine. The results retain their original `31e48c7` simulator baseline;
later transport-model changes may change rerun predictions.

The subsequent [adaptive-claim study](ADAPTIVE-CLAIMS.md) compares bounded
adaptive depth with tuned fixed depths; the first heuristic is not a universal win.
