# Offline scheduling trajectory pilot

`ygaraty rl-dataset --plan /absolute/plan.json --cost-model /absolute/cost.json --output /absolute/new-directory --seeds 8 --workers 2`

This collector replays local checksummed graph snapshots using simulated time. It neither opens a dataset nor runs kernels, services, or training. A dataset URL is unnecessary for graph replay. If constructing plans through the dataset-backed planner, an explicit local absolute SQLite URL is still required.

Version 1 explores **unowned task ordering for the worker requesting work**. It does not choose workers, ownership, placement, or intentional waiting. Owned/replicated plans are rejected. Policies are FIFO, shortest modeled kernel duration (queue-order ties), and uniform random over ready tasks. One episode is one update, with one task per claim. Random behavior has an independent RNG from simulation noise. Cache affinity, if present in a snapshot, is replaced by the selected behavior policy.

Outputs contain a plan copy, cost table, source hashes, settings, episode summaries, cost coverage and JSONL transitions. Each transition records pre-action observations, ready/legal actions, the selected task, its probability under the named behavior policy, the next observation, elapsed simulated time and terminal status. The probability is conditional on the named policy, not a pooled-policy mixture. The reward is zero except at termination, when it is negative full episode makespan; undiscounted returns therefore equal negative makespan. Initial startup time is included in that terminal cost.

Observations include time, requesting worker, ordered ready tasks, committed/running tasks, queue lengths and aggregate cached bytes. Static graph and estimated kernel costs are in the plan/manifest. This is a partially observed trajectory: cache identities, pending network events and RNG state are omitted. Do not treat observations as complete Markov simulator checkpoints. Future sampled durations are never included as policy features.

## What the pilot establishes

A synthetic branching DAG can exercise collection, action legality, dependency completion, reproducibility and reward accounting. It is not a representative training corpus, calibrated performance evidence, or an RL-trained scheduler. Repeated seeds of one graph do not create independent workloads. FIFO and shortest may coincide on equal-cost fixtures.

Keep every policy/seed/configuration of a graph family in one dataset split; the graph checksum is the minimum grouping key, not sufficient protection against near-duplicate graph families. The pilot should remain a pipeline fixture, separate from future evaluation workloads.

## Next dataset increments

1. Collect varied local unowned graph snapshots and realistic worker/link scenarios; preserve provenance and cost-model versions. Add a critical-path heuristic and cache-aware baseline before judging learned policies.
2. Design a side-effect-free legal action mask for owned tasks before extending collection to placement. Current owner selection can mutate ownership, so probing candidates with that callback is unsafe.
3. Add policy-visible cache/input-transfer features and observed worker capacity, keeping sampled future costs hidden. Explicitly distinguish measurements from estimates.
4. Retain separately captured real scheduler decisions, legal candidates and outcomes for external validation. Aggregate timing reports and kernel samples alone cannot reconstruct decision transitions or alternate-action outcomes.
5. Evaluate policies on held-out graph families and hardware/network scenarios, across seeds and cost uncertainty. Existing placement-ranking mismatches must be resolved before claiming real scheduling improvements.

Local scheduling-policy fitting is permitted by the workspace machine policy when scheduled workloads remain simulated and inputs already exist locally. Live Ygaraty workload execution and validation still require a separately authorized host. See [the policy-training protocol and first result](TRAINING.md).

## Multi-workload collection

`ygaraty rl-campaign /absolute/campaign.json --output /absolute/new-directory`

The campaign JSON supplies `schema_version: 1`, a local `cost_model`, `workloads` (each with `name`, local `plan`, provenance-based `family`, and `split`), `scenarios` (each with `name` and simulator `settings`), `defaults`, `policies`, and `seeds`. Paths resolve relative to the configuration. The output must be new. Invalid plans, duplicate graph hashes, family overlap across splits, and invalid settings fail preflight. A failed run has no completion manifest; it is not a complete dataset.

Five behavior policies are available: `fifo`, `shortest`, `random`, `critical_path` (largest remaining estimated compute path), and `cache` (most decoded input bytes already cached on the requesting worker). Ties use ready-queue order. Critical-path estimates exclude transfer costs; neither heuristic is an oracle. Metadata operations have zero compute cost to match the engine. Policies replace any snapshot cache-affinity preference. Global kernel estimates are reference-speed values; current-worker speed is a constant multiplier across candidates.

Collection writes one gzip JSONL file per episode, copies all selected plan snapshots and the cost model, and publishes `manifest.json` only after all episodes pass action/dependency, completion, transition-chain, probability and reward checks. The manifest includes episode checksums, graph/family/split IDs, scenario settings, cost coverage, source hashes, runtime versions and counts of decisions with multiple available actions. Load an episode together with its graph and scenario metadata. Scenario profiles describe hypothetical environment parameters, including future arrivals; do not expose future arrival schedules as deployment-time policy observations.

One episode is a cold-cache single update. Changing `max_tasks` changes claim batching and can be specified per scenario. Seed pairing across behavior policies is not a task-aligned common-random-number experiment: policies can change the order in which simulated noise is consumed. Comparisons need repeated seeds and separate held-out workloads, not a single apparent win.

The first local campaign uses existing 1024×1024 tiled, 1024×1024 fused and 4096×4096 fused factorization snapshots. Duplicate snapshots and cache-policy variants are not counted as independent workloads. All belong to `dense_sgd_factorization` and stay in `train`; this does not provide independent validation or test coverage. These are shape-only compiler graphs, not captured real scheduler decisions. No text dataset or SQLite access is needed for replay.

Seven hypothetical scenarios cover loopback with one worker; a four-worker LAN; home links with four and sixteen workers; sixteen workers with a constrained shared relay/uplink and cache; eight workers with mixed compute/link speeds; and eight workers with four delayed arrivals. Each graph/scenario runs all five policies for three seeds. All costs and outcomes remain modeled. The bundled kernel model is not calibrated for fused factorization; per-episode coverage warnings must accompany downstream use.

## Current evidence status (2026-10-29)

[LOCAL-1](LOCAL-1.md) (2026-10-30) tested the critical-path residual GNN on an unseen Transformer family with its inference cost charged: no gain even when inference is free (−0.07%), a loss once charged (−0.33%, 95% interval −0.42% to −0.24%); critical path stays.

The subsequent [GNN comparison](GNN.md) is now available: 882 final simulated rollouts across factorization and held-out MLP graphs. It improves over model-v1 on factorization but remains slightly worse than FIFO in both family aggregates. Neither policy is integrated into the live scheduler or establishes a general scheduling gain. See [tool maturity](../../../docs/status/tools.md) for the project-wide assessment.

The collection pilot above describes the original factorization corpus. A later local policy-training run completed: its selected epoch-4 policy was about 4.0% worse than FIFO and 13.2% worse than the descriptive per-case best nonrandom heuristic in simulated makespan. These are new-seed results on seen graphs/scenarios, not real-system gains. The subsequent legacy importer and MLP validation corpus are described in [TRAINING.md](TRAINING.md); they were not used to select model-v1. Preserve the original paper and corpus as a dated collection-stage snapshot.

## Bottleneck and capacity-aware placement diagnostics

[BOTTLENECKS.md](BOTTLENECKS.md) documents opt-in per-update resource demand,
compute/dependency paths, and a fixed-depth offline comparison of FIFO,
critical-path priority and capacity-aware large-output placement. It separates
placement-only gains from priority and memory-cap effects, retains regressions,
and does not promote a production scheduler.
