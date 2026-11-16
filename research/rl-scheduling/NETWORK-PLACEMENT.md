# Network-aware admission for large high-rank outputs

2026-10-30. **840 offline simulations** compare bounded fixed batches with the
same batches plus a network-aware admission guard. The guard fixes two slow-uplink
placements in factorization, but does not establish a general advantage over the
best fixed batch sizes. It is an experimental rule, not a production change.

## Results

Fixed depths 1, 2, 4 and 8 are tested both with and without the guard. All arms use
pushed inputs, eager claims, streamed uploads, critical-path task priority and
the same 16/64 MiB outstanding-payload reservation limits. Fixed depth 1 keeps
eager mode enabled too. Graphs and required outputs remain unchanged.

Seed 0 selects a fixed depth separately for each arm and each graph/scenario/cap.
Fresh noisy seeds 2000–2003 evaluate those choices. A second comparison uses the
best unguarded fixed depth on each test seed in hindsight (an optimistic reference,
not a deployable choice). A same-depth ablation keeps the baseline's tuned depth
and changes only the placement guard.

At 64 MiB, selected-arm mean gains over four test seeds:

| Workload/scenario | Fixed → guarded depth | Gain vs tuning-selected fixed | Gain vs hindsight-best fixed | Guard only at baseline depth |
|---|---:|---:|---:|---:|
| Factorization, LAN | 8 → 8 | +0.76% | +0.76% | +0.76% |
| Factorization, home | 4 → 2 | +1.78% | −0.12% | 0% |
| Factorization, heterogeneous | 4 → 4 | **+27.83%** | **−12.35%** | +27.83% |
| Each of three MLPs, every scenario | Same selected depth | 0% | 0% or negative | 0% |

The MLP entry means the selected guarded and unguarded configurations have identical
makespans. Some other MLP settings generated deferrals, but they do not establish
an improvement in the selected comparison. Their hindsight references can still
be faster because a different fixed depth wins on a test seed.

Across all four graphs, equal-weight means versus the tuning-selected fixed depth
are +0.19% LAN, +0.45% home and +6.96% heterogeneous. Against hindsight-best fixed,
these are +0.13%, −1.21% and −3.43%. Do not headline +6.96% as proof the guard beats
the best fixed batch. Home's factorization gain primarily reflects selecting a
different batch size, not a same-depth placement improvement.

### The targeted slow-uplink failure

At fixed depth 4 on heterogeneous factorization:

| Test seed | Unguarded makespan | Guarded makespan | `v20` worker, before → after |
|---|---:|---:|---|
| 2000 | 3.424 s | 3.424 s | w0 → w0 |
| 2001 | 9.896 s | 3.454 s | w7 → w0 |
| 2002 | 10.214 s | 5.492 s | w5 → w0 |
| 2003 | 3.417 s | 3.417 s | w1 → w1 |

`v20` is the 8 MiB broadcast result identified in the adaptive-claim audit. Workers
w4–w7 have 10 Mb/s uplinks; w0–w3 have 100 Mb/s. The guard delays its claim by about
9–10 ms in seeds 2001/2002 to assign it to w0. This is direct evidence of the intended
model behavior. However, the guarded 5.492 s run is still **54.77% slower** than
that seed's best fixed batch. Avoiding one bad placement does not optimize the
remaining schedule.

The deterministic seed already assigns `v20` to a fast worker at depth 4, so that
case is unchanged. Four noisy seeds are a small sensitivity check, not a confidence
interval or held-out workload validation. Different schedules consume random draws
differently; matching seed numbers are not task-aligned common random numbers.

## Placement rule, frozen before the sweep

Only consider compute tasks with:

- Modeled wire output at least **256 KiB**.
- Compute-based remaining-path rank at least half the maximum rank among ready
  compute tasks. This is a high-rank proxy, not an oracle network critical path.
- A task reservation that fits the claiming worker's remaining payload allowance.

For each present worker with sufficient memory, estimate time from its outstanding
reservations and the candidate task: kernel estimate divided by worker speed,
input bytes divided by downlink rate, output bytes divided by uplink rate, and two
RTTs per task. All pending reservations are charged their full estimated service;
this conservative approximation does not know their exact remaining work.

Decline the local claim only if another eligible worker's estimate is at least
20% and 10 ms better, and the task has been ready for less than **250 ms**. Other
fitting tasks remain claimable in critical-path order. Recheck at intervals of at
most 50 ms. After 250 ms of ready age the guard stops vetoing the claim, even if
its preferred worker has not requested work. This bounds the guard's deferral
window, not the job's total dependency/queue wait.

The rule uses current reservations, declared bandwidth/RTT/speed and local cost
estimates. It never reads future event completion times, simulated link free times
or realized random kernel durations. It is both queue- and network-aware; any gain
on homogeneous links may come from queue balance. It does not model cache-aware
input placement, exact upload progress or shared-network congestion in its score.
The simulator still models its normal transfer/queue behavior during execution.

## Memory and unchanged computation

The same reservation scheme as [ADAPTIVE-CLAIMS.md](ADAPTIVE-CLAIMS.md) applies to
all arms: full decoded input backing tensors, decoded output and an output-sized
serialization buffer, modeled wire output, and 4 KiB of descriptor allowance.
Reservations start at claim and remain through result commit, including upload
queues and active uploads. A preferred worker without room is never used as a
reason to defer. Individually oversized tasks are blocked, with no bypass.

Maximum observed reservations are **15.27 MiB under 16 MiB** and **63.97 MiB under
64 MiB**. These bound modeled outstanding payloads, not process RSS: caches, kernel
scratch, global model/store memory and post-commit transport cleanup are separate.
The corpus cache setting remains 512 MiB. This is not a production allocator.

Factorization's `v20` needs 25,169,928 reservation bytes and therefore blocks all
16 MiB factorization cases. MLPs fit both caps. Fused-factorization graphs are
blocked by missing output contracts. There are 21 admitted graph/scenario/cap
combinations × two placement arms × four fixed depths × five seeds = 840 runs.

Pruning precedes this experiment with the same declared output contracts used in
the earlier audits. No computation is reordered within a task and no values are
removed by placement. All graph tasks commit. No numerical workload or equivalence
test was run; this is a discrete-event schedule comparison on metadata only.
Transformer and owned-state plans are not covered.

## Verification

- Nine focused offline tests pass: five placement tests plus four bounded-claim
  tests. They cover fast-uplink preference, memory/availability exclusion,
  bounded deferral and timer-driven fallback when a preferred worker never claims,
  unchanged small-task behavior, completion, reservation release, and exact control
  parity when the guard is disabled.
- All 84 deterministic unguarded fixed controls exactly match the previous
  adaptive-claim study, confirming the control behavior was retained.
- Independent artifact verification replays **314,400 reservation events** and
  verifies every balance and peak against its cap, with zero leaked reservations.
- All 109 placement deferrals satisfy size, ready-age, predicted-improvement and
  preferred-worker memory checks. All 840 run keys are unique and the 21 comparison
  summaries are independently recomputed.

## Reproduce and inspect

```bash
.venv/bin/python -m ygaraty.simulation.network_placement_study \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-network-placement-run
```

Use existing local corpus and cost files, including hash-matching legacy MLP
sources for output-contract recovery. The runner blocks socket connections and
numerical kernels, verifies source hashes, preserves input files and requires a
new output directory. This metadata-only path has no database/dataset connection.

The completed bundle is workspace-parent
`calibration-audit/rl-scheduling/network-placement-v1.zip`. It contains compressed
plans, raw diagnostics and reservation/placement events, cost tables, source/tests,
provenance/checksums, the report and independent verifier. Verify without running
workloads:

```bash
python3 /absolute/path/to/network-placement-v1/verify.py
```

## Decision

The targeted network-aware guard is worth further validation for large critical
outputs on heterogeneous links, but **keep fixed-depth tuning as the control**.
Do not change runtime defaults based on these four seeds. A useful next experiment
would distinguish queued compute from already-uploaded results in the service
estimate and compare across more independent scenarios; numerical or real-worker
validation belongs on the authorized execution machine.
