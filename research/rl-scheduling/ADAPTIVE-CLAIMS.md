# Adaptive claim depth with bounded outstanding payloads

2026-10-30. **525 offline simulations completed.** This first adaptive heuristic
is not a universal improvement over tuned fixed batches. It helps some
heterogeneous-worker cases but regresses on LAN and has a severe deterministic
factorization outlier. Keep fixed-depth tuning as the default recommendation.
No workloads, policy training, workers, services or network emulators ran.

## Apples-to-apples comparison

The same four pruned local factorization/MLP graphs run with critical-path task
priority and fixed depths 1, 2, 4, 8 versus adaptive depth 1–8. All use pushed
inputs, streamed posting and eager claims, including fixed depth 1. This differs
from the engine's default `max_tasks=1`, which disables eager mode; the study
subclass keeps the same protocol enabled and separately limits claim depth.
No production worker, scheduler or simulator defaults were changed.

Both adaptive and fixed arms share outstanding-payload caps of 16 or 64 MiB per
worker. Scenarios are LAN-4, home-4 and heterogeneous-8. Seed 0 is deterministic
and selects the fastest fixed depth separately for each graph/scenario/cap. That
selected depth is evaluated against adaptive on noisy seeds 1900–1903. This
separates fixed-depth selection from evaluation seeds, but does not hold out
new graphs or estimate a statistically precise performance difference.

The report also compares adaptive with the fastest fixed depth **on each test
seed in hindsight**. That is an optimistic reference, not an implementable
selection rule. It matters because fixed-depth performance can also be unstable.

## Results

Mean evaluation-seed gain over the tuning-selected fixed depth, 64 MiB cap:

| Workload | LAN-4 | Home-4 | Heterogeneous-8 |
|---|---:|---:|---:|
| MLP-76 | −3.81% | +3.16% | +0.53% |
| MLP-77 | +1.31% | +2.58% | +5.22% |
| MLP-340 | −16.48% | −6.48% | +1.35% |
| Tiled factorization | −7.06% | −4.37% | +22.99% |
| Equal-weight mean | **−6.51%** | **−1.28%** | **+7.52%** |

Against the per-test-seed hindsight-best fixed depth, those scenario means become
−6.55%, −2.76% and **+0.24%**. Do not interpret +7.52% as a robust heterogeneous
speedup: factorization's +22.99% against its chosen depth 4 becomes **−5.40%**
against the hindsight-best fixed depths. Four noisy seeds are only a sensitivity
check; protocol/assignment changes do not share task-aligned random draws.

The 16 MiB MLP results are similar. MLP-340/Home drops to −6.62% versus −6.48% at
64 MiB. No cap was relaxed to obtain a speedup.

### The important outlier

Noiseless heterogeneous factorization finishes in 3.412 s at fixed depth 4 but
10.214 s under adaptive depth: **199.39% slower**. The critical-chain broadcast
`v20` is assigned to `w0` (100 Mb/s uplink) in the fixed run, and `w4` (10 Mb/s)
in the adaptive run. Its lease-to-commit interval is 0.738 versus 7.299 s. It
produces an 8 MiB tensor. This is strong model evidence that changing claim timing
can put a network-critical task on the wrong worker; the policy is compute/frontier
aware but does not use output bandwidth when assigning tasks.

The same class of slow result appears in some noisy fixed-depth runs. It is not
just an adaptive-depth problem, and it is not explained by increased utilization.
Do not deploy this heuristic as a general replacement for the best fixed setting.

## Adaptive rule (fixed before evaluation)

At the start of a claim, compute:

1. A fair share of currently ready compute tasks, weighted by configured worker
   speed. Metadata tasks do not count toward this frontier size.
2. A latency-amortization depth: ceil((worker RTT + coordinator service estimate)
   / median predicted ready-kernel time on that worker), with a minimum of one.
3. Take the smaller of those values and eight. Reduce it to one when the worker
   has at least half its payload allowance reserved, or eight outstanding tasks.

This uses current readiness, known worker profiles and existing local cost
estimates. It does not inspect future random timings or completion events.
Memory filtering preserves critical-path priority among tasks that fit. Metadata
operations remain coordinator fast paths. The actual batch may be smaller than
the requested depth when the frontier or memory limit prevents further claims.
All actual depths 1–8 occurred across the adaptive rollouts.

## What the memory bound guarantees

Before accepting a compute task, reserve:

- Full decoded backing tensors for its inputs (8 bytes/element), counting repeated
  inputs separately and assuming no cache sharing.
- Decoded output plus an output-sized serialization buffer.
- Modeled wire output bytes and a 4 KiB descriptor allowance.

Keep the entire reservation from claim until result **commit**, including queued
computation, computing tasks, upload queues, active uploads and pending result
notification. Merely starting an upload does not free credit. There is no
oversized-task bypass. If a task cannot fit by itself, the case is blocked before
simulation rather than allowed to deadlock or exceed the cap.

The bound is on modeled outstanding tensor-payload reservations, **not process
RSS**. It excludes persistent caches (the corpus settings reserve 512 MiB for
cache), global model/store memory, kernel scratch, Python/native allocator
metadata beyond the descriptor allowance and post-commit transport cleanup.
It does not implement a production worker's actual memory allocator. Runtime
admission would need to account for those components separately.

Observed maximum reservations:

| Cap | Maximum across all completed runs |
|---|---:|
| 16 MiB | 15.27 MiB |
| 64 MiB | 63.97 MiB |

Factorization's `v20` needs **25,169,928 bytes** under this accounting, so its
three 16 MiB scenario cases are blocked for every policy. Fused-factorization
plans are separately blocked by missing output contracts. MLPs run at both caps;
factorization runs at 64 MiB: 21 graph/scenario/cap combinations × five policies ×
five seeds = 525 rollouts.

## Verification and artifacts

Four focused tests cover adaptive responses to frontier/latency/backlog, memory
admission during uploads, exact reservation release, fixed-depth-one protocol
parity and rejection of individually oversized tasks. Independent verification
replays **196,500 reservation events**, checks every intermediate balance and
reported peak, verifies all reservations close, confirms all graph tasks commit,
and recomputes the 21 comparison summaries. All checks pass.

All arms use identical pruned graph hashes; no graph rewrite or numerical kernel
is involved. Utilization and noncompute time are reported as diagnostics, not as
substitutes for makespan. Noncompute time includes communication, startup and
imbalance, not only dependency waiting. Stage timers can overlap.

Reproduce with existing local metadata corpora and hash-matching legacy MLP
sources (needed for output contracts):

```bash
.venv/bin/python -m ygaraty.simulation.adaptive_claim_study \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-adaptive-claims-run
```

The entry point blocks sockets and numerical workload kernels. It uses no
database/dataset URL. Existing plans are checked unchanged and cost hashes are
verified. The output path must be new. Source/cost records, compressed plans,
compressed rollout diagnostics and reservation ledgers are written locally and
bundled as ZIP.

Successful artifacts are workspace-parent
`calibration-audit/rl-scheduling/adaptive-claims-v2/` and `adaptive-claims-v2.zip`.
The initial v1 stopped at the correctly rejected oversized task before producing
a comparison; it is marked incomplete and superseded. The independent verifier
in v2 is standard-library Python and requires no workload execution:

```bash
python3 /absolute/path/to/adaptive-claims-v2/verify.py
```

## Next lever

Before more depth tuning, test a **network-aware admission/placement guard for
large critical-path outputs**, combined with bounded queues. Use the already
pushed/eager, memory-bounded fixed batches as controls. Preserve the full negative
results here rather than choosing only the seeds on which adaptive batching wins.
Transformer and owned-state runtime gains remain untested by this study.

The follow-up [network-placement experiment](NETWORK-PLACEMENT.md) tests that guard
with identical bounded fixed-batch controls and reports both wins and regressions.
