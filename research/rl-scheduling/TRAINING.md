# First local scheduling-policy experiment

The workspace machine policy now permits scheduling-policy weight updates from
local simulated environments and existing data. Workload kernels and live
Ygaraty services remain prohibited. The CLI blocks socket connections and the
workload kernel executor while fitting and evaluating.

Run from the checkout with the existing environment:

```bash
ygaraty rl-train \
  /absolute/path/to/factorization-v1 \
  --output /absolute/path/to/new-model-directory --epochs 4
```

The output directory must not already exist. No inputs or packages are fetched.
The CLI uses one CPU thread, verifies the corpus completion manifest, graph and
cost hashes, and compressed episode checksums, and retains local checkpoints.

## Declared experiment

The first model is a 449-parameter, memoryless candidate scorer: 12 inputs,
32 tanh units, one score. At each claim it scores only the scheduler's ready
queue, so invalid actions cannot be selected. Features are reference estimated
kernel/downstream costs, decoded input/output sizes, dependency/successor counts,
metadata status, ready-queue position and size, committed/leased fractions,
and requester cache-byte totals. Features contain no task-ID embedding, future
worker arrivals, sampled durations, or cache identities. A single legal task
requires no model inference. Input/output byte features are decoded shape
estimates, not a full transport-cost prediction.

This is deliberately a small memoryless baseline despite partial observations.
It is not the history-based architecture proposed in the paper.

1. **Offline warm start:** verify the 315 saved episodes and sample at most
   12 choice decisions per episode. Four passes of reward-weighted imitation
   favor lower-makespan trajectories relative to the training FIFO mean for
   that graph/scenario. Weights are clipped exponential relative returns;
   this is an initialization heuristic, not an unbiased off-policy gradient.
2. **On-policy RL:** four passes through 21 graph/scenario cases, one stochastic
   rollout per case per pass (84 episodes), update the policy with REINFORCE.
   The loss uses the sum of chosen-action log probabilities, a preceding
   exponential-moving makespan baseline, per-case fixed normalization, and an
   entropy bonus of 0.0001 times the sum of decision entropies. Gradients are
   clipped at norm 1. Policy and environment generators use separate numeric
   seed ranges. Only the scheduling policy is numerically trained.
3. **Checkpoint selection:** greedy simulator rollouts on seeds 100 and 101;
   select among warm start (epoch 0) and the four RL checkpoints. The score is
   equal-case mean makespan divided by that case's training FIFO mean.
4. **Final evaluation:** freeze selection, then use seeds 200, 201 and 202 for
   all 21 cases and compare with all five behavior policies. These are fresh
   seeds on seen graphs/scenarios, not an independent workload-family test set.
   If epoch 0 wins, report the selected policy as the warm-start model even
   though RL fitting also occurred; `last_rl.pt` retains the final RL model.

The optimization objective is relative makespan with equal case weighting,
plus entropy regularization, rather than the paper's unweighted mean seconds.
Evaluation is greedy while training samples actions; no guarantee relates the
two. Full trajectory outcomes are saved in `report.json`. Inference overhead is
not charged to simulated time; measured total host wall time is separate.

## Outputs and interpretation

`warmstart.pt`, `selected.pt` and `last_rl.pt` store PyTorch state dictionaries;
load only trusted local files with `weights_only=True`. Recreate the `Policy`
class and use the recorded feature definition. `sources/` preserves the trainer,
engine, cost model code, collector and scheduler sources for the run; the report
records dataset/source hashes, seed sets, versions, learning curves and case
comparisons. Keep the referenced corpus (snapshots and cost table) with a model.

A good result would justify larger experiments, not a claim about real workers.
Three seeds per case, a single factorization family, uncalibrated costs and
omitted inference overhead limit conclusions. The best-heuristic comparison is
a descriptive per-case best-of-four baseline excluding random; it is not a
single deployable oracle and was not used to tune the learned checkpoint.

## First run: model-v1 (2026-10-29)

The run completed on one local CPU thread in 48.64 seconds of measured host
wall time. It warm-started from sampled choices in the 315 factorization
trajectories, then completed all 84 on-policy episodes. Validation chose epoch
4, so the selected checkpoint includes REINFORCE updates. Reloading that
checkpoint in a fresh Python process reproduced a saved greedy test rollout
exactly. All ten focused collector/import/training tests pass.

On the final three seeds across 21 seen graph/scenario cases, the mean of
case-level makespan ratios was **1.03985 versus FIFO** and **1.13203 versus the
best non-random heuristic per case**. These are simulated outcomes: the first
policy is about 4.0% worse than FIFO and 13.2% worse than that descriptive
best-of-heuristics comparator. Do not call this a scheduling improvement.
No hyperparameters were changed in response to final evaluation results.

Local checkpoints, source snapshots, complete episode comparisons and reload
verification are under the workspace parent's
`calibration-audit/rl-scheduling/model-v1/`. The full report separates simulated
completion times from measured host wall time. The paper's earlier “no policy
has been trained” status describes its pre-training snapshot; this document
records the subsequent exploratory result.

## Broader local artifacts

The `runs/` inventory contains 316 plan-like JSON artifacts: nine versioned
snapshots, 268 legacy `tasks`/`values` plans and 39 task-list-only plans. There
are 33 distinct canonical task-list hashes after removing legacy execution
status fields. This deduplication count is **not** a count of independent graph
families or fully replayable graphs: leaf shapes, ownership, grouping and cache
semantics still need validation.

The importer `ygaraty rl-import` accepts only ungrouped,
unowned legacy plans with explicit leaf shapes. It strips execution-status
fields, preserves graph operations, and records source checksums. It does not
infer missing shapes from sliced input regions or fabricate optimizer tasks.
Original cache policy is unavailable, so imported plans explicitly use no
reusable input caching; in that setting the cache heuristic ties FIFO.

Three imported MLP forward/backward graphs (76, 77 and 340 tasks) are reserved
as one **validation** family. All seven scenarios, five policies and three
seeds are collected separately under `mlp-validation-v1`. They were not used
for fitting or selecting model-v1. Their graph completion reward need not
represent a complete optimizer update. Together with the factorization corpus,
this provides six graph snapshots across two families, still without a third
independent test family. Preserve all MLP variants together in validation.

Committing `runs/` is unnecessary for local collection. For sharing, curate
versioned shape-only graphs, explicit reconstruction assumptions, local cost
models, provenance and family assignments. The entire directory is about
6.6 GB and largely consists of binary arrays/tensors; committing repeated
artifacts does not make the dataset more diverse. No Git commit or upload was
performed as part of this experiment.
