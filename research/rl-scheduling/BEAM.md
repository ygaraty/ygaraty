# Local metadata audit and bounded offline beam search

Completed 2026-10-30. This experiment plans against the local discrete-event
simulator and frozen residual GNN. It does not train a policy, execute numerical
workloads, contact services, retrieve archived files or change the live scheduler.

## What the local files contain

The inventory scans remaining JSON files whose names contain `plan` and all
JSONL files under `runs/`. It found **241 task-bearing plan files** (232 legacy,
nine versioned), all with per-task shapes, and **139 JSONL traces containing
49,560 readable records**. No invalid JSONL lines or plan parse errors were
encountered in that inventory. This is an audit of remaining local metadata,
not a claim that every compiled artifact in every format was discovered.

Plan tasks expose kernels, inputs/dependencies, shapes and attributes. Legacy
formats can include values and run-state fields; versioned snapshots carry graph
checksums/provenance. Shapes and input regions support byte/cost estimates but
do not establish an actual host location or bandwidth at decision time.

Across JSONL files, top-level field coverage includes:

| Field | Files containing it | Records containing it |
|---|---:|---:|
| time | 38 | 39,159 |
| task | 41 | 42,114 |
| worker | 39 | 42,055 |
| event | 39 | 44,939 |
| op | 2 | 148 |
| seconds | 91 | 665 |
| fetched_bytes / written_bytes / inputs | 2 each | 148 each |
| memory | 7 | 2,934 |
| bandwidth / rtt_ms / cached / cache / links | 0 | 0 |

These are field-presence counts, not semantic calibration or complete-event
coverage. `seconds` in a metrics record is not necessarily kernel duration.
Nested fields were not exhaustively classified. We did not establish a complete,
synchronized history of placements, cache contents, transfer contention, worker
capacities and actual timings sufficient to replay a measured run. Prior cleanup
also removed large artifacts; the audit does not silently restore or download them.

The search therefore uses the existing **six validated local graph snapshots**
(three factorization, three reconstructed MLP), local cost tables and explicit
scenario assumptions. The audit is provenance/context, not evidence that the
simulator is calibrated to every inventoried trace. Source/snapshot hashes are
recorded. It is a model-based plan search, not retrospective replay of a real run.

## Search and validity

The CLI is `ygaraty.simulation.rl_beam`. We searched 18 cases: six graphs ×
`lan_4`, `home_4`, and `heterogeneous_8`, without timing noise or request jitter.
The beam has width 3 and depth 3, examines at most four evenly spaced decision
points per node and two alternative ready tasks per point. Critical-path-ranked
candidates and the GNN suggestion provide alternatives. Prefix decisions are
replayed from the start; each suffix is completed with critical path and ranked
by its resulting simulated makespan. Later interventions must follow earlier
forced-prefix decisions. Beam pruning is elitist; the initial critical-path and
GNN schedules remain eligible incumbents. No global optimality is claimed.

The experiment used **884 candidate/baseline simulator rollouts**, excluding
winner replay checks and noisy evaluation. It took **180.87 measured host seconds**
for the original audit/search/evaluation run. Full suffix rollout scoring uses
model-based future knowledge and substantial computation; comparison against a
single heuristic/GNN rollout is **not search-budget matched**, and search cost
is not included in simulated makespan.

At every assignment the selected task must be ready and all its dependencies
committed (or local graph leaves). The scheduler enforces leases/task accounting;
each task must execute exactly once and the graph must complete. Modeled worker
requests, compute rates, queues, links and caches determine placement and timing.
The search chooses task order, **not an arbitrary worker or transfer route**.
Owned/replicated graphs, owner attributes, cache-affinity overrides and multi-update
plans are rejected. Memory-budget eligibility, capabilities, failure recovery,
numerical correctness, and deployment feasibility are not validated by this scope.

Each winning schedule is replayed with its full task order and must reproduce
both worker assignments/lease times and makespan. A static priority permutation
must also reproduce it under the original deterministic conditions. With noisy
timing, the permutation chooses the earliest-ranked *currently legal* task;
worker assignments may change. This is a transferable priority policy, not a
promise that the original event/assignment trace will remain feasible verbatim.

## Results

Equal-case mean improvements for the deterministic searched settings:

| Family | Beam vs critical path | Beam vs frozen GNN policy |
|---|---:|---:|
| Factorization | +1.3329% | +0.5005% |
| MLP | +0.9841% | +0.6943% |

Example: fused-4096/home-4 improved from 30.1075 seconds for both baselines to
29.3796 simulated seconds. Several other cases simply retained a baseline.
Retaining the incumbents guarantees non-regression **only on the search model**;
these results are not a held-out test of search performance.

Fixed priority lists were then evaluated on fresh noisy seeds 1400–1404 without
further search. The initial three-policy comparison has 270 rollouts. A post-result
diagnostic added frozen critical-path and frozen GNN priority lists, using the
already computed deterministic schedules and no further tuning, for **450 unique
rollouts** total. Ratios below are equal-case mean makespan relative to dynamic
critical path; lower is better.

| Noisy evaluation policy | Factorization | MLP |
|---|---:|---:|
| Dynamic critical path | 1.000000 | 1.000000 |
| Dynamic residual GNN | 0.974152 | 0.999859 |
| Beam-derived fixed priority | 0.875297 | 0.993939 |
| Frozen critical-path schedule priority | 0.901075 | 1.000584 |
| Frozen GNN schedule priority | **0.874495** | 1.000372 |

The apparent 12.47% factorization gain over dynamic critical path is **not solely
an incremental search benefit**: a frozen GNN schedule already does slightly
better. Relative to frozen GNN priority, beam priority is 0.1214% worse on
factorization and 0.6433% better on MLP (mean paired case ratios). Relative to
frozen critical-path priority, beam improves 2.6007% and 0.6645%, respectively.

This identifies fixed global task ordering as a useful next investigation,
without proving a particular cache/network/tie-breaking mechanism or a general
real-system speedup. Five numeric seeds are a small diagnostic and are not
 task-aligned common random numbers. Graphs and scenarios were already inspected;
there is no unseen-graph generalization claim. The additional controls were not
used to alter the chosen beam schedules.

## Artifacts and reproduction

```bash
.venv/bin/python -m ygaraty.simulation.rl_beam \
  --runs /absolute/path/to/ygaraty/runs \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --checkpoint /absolute/path/to/gnn-residual-v1/selected.pt \
  --output /absolute/path/to/new-beam-directory
```

Workspace-parent artifacts: `calibration-audit/rl-scheduling/beam-v1/`:

- `audit.json.gz`: file inventory, checksums, plan/trace field coverage.
- `schedules.jsonl.gz`: baseline/winning orders, assignments, branch history,
  replay outcomes and per-case deterministic settings.
- `transfer.jsonl.gz`: original 270 noisy comparison rollouts.
- `transfer_with_controls.jsonl.gz`: all 450 noisy/control rollouts.
- `config.json`, `report.json`, `verification.json`, source snapshots and checksums.

All bulky records are compressed on write, with a ZIP bundle for local reuse.
All **23 focused RL tests pass**. A fresh process replayed all 54 CP/GNN/winner
schedules exactly, checked baseline parity with the established collectors on all
18 cases, verified the complete control grid and artifact/source checksums, and
recomputed the transfer aggregate. The frozen GNN checkpoint was unchanged.
