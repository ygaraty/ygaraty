# Bottleneck reporting and capacity-aware placement

2026-11-01. This is another step toward scheduling sovereign computing workloads
from explicit evidence. All workloads here remain simulated; no training,
contributor, relay, network emulator or database service was run.

## Reporting

The event simulator accepts diagnostics=True; the CLI exposes --diagnostics.
JSON contains a diagnostics entry for each completed update:

- Per-worker actual modeled upload/download payloads divided by configured link
  rates, and assigned compute seconds.
- Shared relay payload demand and coordinator busy time clipped to the update.
- An idealized capacity/compute-path bound, its gap to update completion, and an
  explicit flag if the bound exceeds the simulator's time.
- A compute-weighted longest dependency path and the dependency chain leading to
  the last completion, with task IDs, workers and ready/lease/compute/commit times.
- Chain-stage totals: ready queue, pre-compute delay, compute, and post-compute
  delay. These non-overlapping intervals along this chain sum to update time.

These are post-hoc diagnostics. They do not read future timings to make scheduling
decisions. Reporting is opt-in and tests verify that it does not change simulated
timings. Resources with configured rate zero are unlimited, not disconnected.

The observed dependency chain is **not** a full critical path through a
resource-contention graph. A dependency may wait behind unrelated work; its
post-compute interval includes upload queues, transfers and server waits.
Likewise, payload demand is an idealized capacity bound, not an exact max-flow
solution. It omits request headers, RTT, serialization and memory. The existing
small-transfer approximation can violate strict physical link capacity; the
report flags such violations rather than clamping them away.

Usage with an existing local dataset:

    .venv/bin/python -m ygaraty.experiments.simulate run --dataset-url sqlite:////absolute/path/to/existing.sqlite --diagnostics --output-json /absolute/local/result.json

Use the matching model/ownership/batching flags for that dataset. For a captured
plan, set diagnostics to true in the simulation settings of a local reference
suite; replay needs no dataset. All default scheduler behavior remains unchanged.

## Controlled comparison

The study reuses four existing, dead-task-pruned shape-only plans: three MLP
graphs (65, 65 and 266 surviving tasks) and tiled factorization (582 tasks).
Workload names preserve their original pre-pruning identifiers. Plans are never
rewritten by this study. Cost tables are the archived local tables, with their
coverage warnings retained.

All arms use fixed claim depth 4, pushed inputs, eager claims and streamed uploads.
Six arms separate effects:

1. Current FIFO event scheduler, without a new reservation cap.
2. FIFO with a 64 MiB cap on outstanding tensor reservations.
3. The same cap with compute-based remaining-path priority.
4. Priority plus capacity-aware large-output admission at 64 KiB.
5. The same guard at 256 KiB.
6. The same guard at 1 MiB.

The existing guard compares estimated compute, upload, download and RTT using
known worker profiles and outstanding reservations. It defers qualifying tasks
for at most 250 ms when another available worker is predicted materially faster.
It does not inspect future event times or realized compute durations. This is
task-level placement on unowned graphs, not permanent Transformer-owner placement.

Three modeled pools cover LAN, homogeneous home links and heterogeneous workers.
Two seeds (3100–3101) choose a candidate among priority alone and the three guards;
eight fresh seeds (4100–4107) evaluate that frozen choice. FIFO remains an explicit
baseline, not a candidate forced to lose. There are 720 total simulations.
The selected candidate can still be worse than FIFO: no default is promoted.

The paired gains are means of per-seed percentage reductions in seconds, not
percentage reductions calculated from rounded means. Same seed numbers are not
task-aligned common random numbers because different schedules consume random
draws differently. These are simulator-seed holdouts, not held-out real machines.

## Reproduce entirely offline

Use the already-local artifact directory and scenario manifest:

    .venv/bin/python -m ygaraty.simulation.bottleneck_study --archive /absolute/local/network-placement-v1 --scenarios /absolute/local/mlp-validation-v1/manifest.json --output /absolute/local/new-study

The runner refuses existing output directories, blocks socket connections and
workload-kernel execution, and records plan/cost/scenario/source hashes. A source
change during execution aborts the report. Detailed rollouts retain bottleneck
and chain diagnostics, coverage, reservation peaks and large-output assignments.

Results from the fixed-source run and remaining calibration limits follow below.

## Calibration and optimization limits

- These four graphs and the network guard were already used in earlier local
  studies. New seeds test scheduling variation, not unseen workloads.
- One cold update is simulated. Steady caches, startup/publication differences,
  owner migration and long-run learning quality are not evaluated.
- Per-op fallback costs, unknown fitted work ranges, approximate GFLOPS scaling,
  uncalibrated upload stages and the 50/53/54 transport/cache gaps remain.
- Graph cuts identify necessary bandwidth demand but do not account for every
  dependency, connection limit, browser preflight, network loss or relay service.
- Exact resource critical paths would require upload/link/coordinator scheduling
  edges in addition to dependency edges. The present report explicitly keeps
  that distinction.
- A 64 MiB outstanding-payload reservation cap is not measured process memory.
  The uncapped FIFO comparison includes the cap's effects; the bounded FIFO and
  priority-only baselines isolate them.
- Guard evaluation and graph analysis have zero modeled coordinator decision
  cost in this study. Real scheduler CPU overhead still needs measurement.
- Candidate selection uses only two tuning seeds and can overfit. Regressions
  on the eight evaluation seeds are retained; no production default is changed.
- Experiment 37 is untouched and validation-only. This study provides no new
  real-system accuracy or numerical-correctness evidence.

Validation: 52 focused offline tests passed, covering observation invariance,
per-update reset, path stage accounting, slow-uplink identification, metadata,
candidate execution, tuning/test separation, existing reference replay, bounded
admission and network-placement regressions.

## Fixed-source results

All values below are simulated. Positive gain means fewer seconds; negative gain means a regression.

| Workload | Pool | Candidate | FIFO s | Candidate s | Gain vs FIFO | Placement-only gain | Worst seed vs FIFO |
|---|---|---|---:|---:|---:|---:|---:|
| mlp_340 | heterogeneous_8 | critical_priority | 2.335 | 2.264 | +2.93% | +0.00% | -6.83% |
| mlp_340 | home_4 | critical_priority | 4.678 | 4.605 | +1.56% | +0.00% | -0.39% |
| mlp_340 | lan_4 | critical_priority | 1.513 | 1.471 | +2.75% | +0.00% | +1.88% |
| mlp_76 | heterogeneous_8 | critical_priority | 1.205 | 1.218 | -1.10% | +0.00% | -5.65% |
| mlp_76 | home_4 | critical_priority | 1.858 | 1.862 | -0.20% | +0.00% | -1.49% |
| mlp_76 | lan_4 | critical_priority | 0.431 | 0.434 | -0.52% | +0.00% | -3.27% |
| mlp_77 | heterogeneous_8 | capacity_64k | 1.415 | 0.980 | +30.77% | +14.18% | +29.63% |
| mlp_77 | home_4 | capacity_64k | 3.367 | 3.257 | +3.27% | +4.52% | +1.18% |
| mlp_77 | lan_4 | capacity_64k | 0.486 | 0.478 | +1.55% | +0.33% | -2.58% |
| tiled_1024 | heterogeneous_8 | critical_priority | 3.480 | 5.990 | -70.61% | +0.00% | -193.44% |
| tiled_1024 | home_4 | capacity_64k | 11.447 | 9.825 | +14.12% | +12.69% | +5.35% |
| tiled_1024 | lan_4 | capacity_256k | 2.185 | 1.843 | +15.64% | +2.22% | +12.57% |

Placement-only compares the selected candidate with the same capped critical-priority scheduler.
The uncapped/capped FIFO baseline difference is also preserved per seed in report.json.
The study found zero idealized capacity-bound violations across 720 simulations.

The strongest selected placement-only gain is heterogeneous mlp_77: +14.18%,
with +30.77% overall versus FIFO. Home factorization gains +12.69% from placement
and +14.12% overall. The small mlp_76 regresses slightly in all three pools.

**The selected heterogeneous factorization candidate fails.** Two tuning seeds
chose priority alone; it averages 5.990 s versus FIFO at 3.480 s, a mean paired
70.61% regression. Its worst evaluation seed is 193.44% slower. Do not reinterpret
the best evaluation-seed guard as a successfully selected policy.

At seed 4103, FIFO finishes in 3.483 s; priority alone takes 10.220 s.
The largest resource demand moves from w5 uplink at 2.333 s to w4 uplink at
8.992 s (11,239,424 payload bytes over 10 Mb/s). The dependency-chain
post-compute total grows from 2.267 to 8.914 s. This is the slow-uplink failure
the new reporting was intended to reveal.

A diagnostic guard arm on that same seed takes 4.212 s and places the 8 MiB
v20 result on fast worker w2. It improves on priority alone but remains slower
than FIFO. Its largest resource demand is then coordinator busy time (1.911 s).
This is an illustrative intervention, not a replacement chosen on held-out results.

Local artifacts: calibration-audit/bottleneck-placement-2026-11-01-final/report.json
and rollouts.jsonl.gz in the workspace parent. The earlier exploratory directory
is explicitly marked SUPERSEDED because reporting sources changed during that run.
The fixed-source final run is the cited evidence.
