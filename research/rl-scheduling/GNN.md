# First graph-policy comparison

This experiment trains only a scheduling-policy neural network on local metadata
and simulated time. The CLI disables socket connections and numerical workload
kernel execution. No dataset download, service or remote job is used.

```bash
ygaraty rl-gnn \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --mlp-checkpoint /absolute/path/to/model-v1/selected.pt \
  --output /absolute/path/to/new-gnn-directory
```

## Architecture

The model operates on task nodes with directed dependency edges; leaf-input
shapes contribute through task features rather than separate leaf nodes.
There are no learned task-ID embeddings. Node inputs are the seven static
features already used by the first MLP: transformed kernel cost, downstream
critical-path estimate, decoded output/input sizes, dependency/successor counts,
and metadata indicator.

A linear projection produces 16-dimensional node states. Three residual tanh
message-passing rounds concatenate each node state with mean predecessor and
mean successor states. A mean-pooled graph vector supplies graph-level context.
The legal-task scorer concatenates the node embedding, pooled graph vector,
and the original 12 static/dynamic candidate features, then applies a
32-unit tanh MLP and scalar output. Dependency messages influence parameters
through backpropagation; this is not a fixed graph-feature preprocessor.

Graph embeddings are computed once per rollout and recomputed after each
optimizer step. During RL they remain connected to the episode's gradient
graph. Current ready order, progress and requester cached-byte totals enter the
actor at each decision, but do not propagate through message-passing layers.
This is a **static-graph, memoryless** GNN baseline. It does not recover hidden
cache identities, pending network events or history. Ready-order input and
tie-breaking intentionally preserve the existing scheduler interface; only
the graph encoder is equivariant to a reordering of serialized task nodes.

## Frozen protocol

- Training: the existing 315 factorization episodes; at most 12 choice decisions
  sampled per episode, four reward-weighted imitation passes, then 84 on-policy
  REINFORCE episodes (four passes through 21 graph/scenario cases).
- Objective: per-case makespan normalized by the fixed training FIFO mean,
  an EMA control-variate baseline, summed chosen-action log probabilities,
  and summed entropy bonus coefficient 0.0001. This is the same objective and
  rollout budget as the first MLP, not a matched parameter-count experiment.
- Initialization seed 2027; offline sample seed 2026; policy/environment training
  seeds use separate ranges. No old final seeds are reused.
- GNN checkpoint selection: factorization rollouts at seeds 300 and 301 only,
  comparing the warm-start checkpoint and four RL checkpoints.
- Final comparison: seeds 400, 401 and 402, all seven scenarios, both factorization
  and MLP forward/backward graphs. The original 449-parameter MLP is frozen;
  its earlier checkpoint-selection protocol is not retrospectively changed.
- Baselines: FIFO, shortest estimated kernel time, uniform random, static
  critical-path priority and cached-input-byte priority. All receive identical
  graph/scenario/seed settings and retain the collector's existing semantics.
- The MLP workload family is never used to fit or select the GNN. It retains its
  `validation` dataset label; there is still no independent third test family.
  This family is a legacy forward/backward reconstruction with no reusable
  caching and no synthesized optimizer tasks, not a measured optimizer update.

Final greedy rollouts differ from the stochastic training policy. Single
initializations, different network capacities, and different model-selection
seed sets mean observed differences cannot be attributed solely to graph
message passing. A paired capacity/seed ablation would be a separate experiment.
The random baseline retains the existing collector's shared numeric seed
convention; a seed match is not task-aligned common random numbers.

## Metrics and artifacts

`config.json` is written before fitting. The report records all selection,
training and final rows, checkpoint and corpus hashes, source snapshots,
parameter counts, measured host wall time and checkpoint-reload verification.
The selected checkpoint can be epoch 0 (warm start) if RL updates do not improve
selection score; `last_rl.pt` is retained separately.

Case means first average the three final seeds. Summary ratios then weight each
graph/scenario case equally. A ratio to FIFO below 1 is better. Cross-family
summaries are kept separate. The per-case best non-random heuristic is a
descriptive best-of-four comparison, not a deployed oracle. Differences are
modeled predictions; three seeds do not establish confidence or tail behavior.

Policy inference time is measured on this host, including initial graph
embedding and feature encoding, and divided by the number of multi-action
choices for reporting. It is not added to simulated makespan and is not a
measurement of production coordinator overhead. Neural policies skip inference
when only one task is legal. Heuristic inference overhead is not benchmarked.

## Completed result: gnn-v1 (2026-10-29)

The 3,953-parameter GNN completed four warm-start passes and 84 on-policy RL
episodes. Factorization-only validation selected epoch 2 (after 42 RL episodes).
The original 449-parameter MLP checkpoint remained unchanged. The complete run,
including final baseline evaluation, took 146.78 seconds of measured host wall
time on one CPU thread.

Final comparison contains **882 unique rollouts**: seven policies, 42
workload/scenario cases and three fresh seeds each. All values below are
mean case-level **simulated** makespan ratios to FIFO; lower is better.

| Policy | Factorization, seen graphs | MLP graphs, held-out family |
|---|---:|---:|
| GNN | 1.0031 | 1.0141 |
| Frozen MLP policy | 1.0364 | 1.0016 |
| FIFO | 1.0000 | 1.0000 |
| Shortest estimated task | 1.0307 | 1.0186 |
| Critical path | 1.0005 | 0.9876 |
| Cache-aware | 1.0093 | 1.0000 |
| Random | 1.0390 | 1.0326 |

On factorization, the mean of per-case gains versus the MLP policy is +3.32%,
with 18/21 case wins. On the held-out MLP workload family it is -1.24%, with
2/21 case wins. These averages of paired case ratios differ from taking a ratio
of the table's aggregate means. FIFO remains marginally better in the
factorization aggregate, while critical path is best on the held-out family.
The GNN is 8.49% worse than the per-case best non-random heuristic on
factorization, and 3.65% worse on the held-out family; this comparator can
choose a different heuristic per case.

This trial supports neither a general scheduling win nor a deployment
recommendation. It shows an improvement over the initial MLP on the training
family, and a regression on a different family. No hyperparameters were changed
after final results were inspected. The GNN has greater capacity and a different
initialization; this experiment does not isolate message passing causally.

Measured policy inference/encoding time, including amortized graph embedding,
was about 0.283 ms per choice for the GNN versus 0.217 ms for the MLP on
factorization, and 0.220 versus 0.103 ms on MLP workload graphs. These local
instrumented observations are not production overhead measurements and are
not included in the simulated completion times above.

All 13 focused `test_rl*.py` tests pass. The GNN encoder responds to changed
edges, is equivariant to serialized node permutations, and receives nonzero
message-layer gradients. The comparison runner matches the original MLP
runner. Saved GNN weights reloaded in a fresh process reproduce an existing
final rollout in each family exactly. The final row grid, summary recomputation
and unchanged MLP checkpoint hash also pass verification.

Artifacts are stored in the workspace parent's
`calibration-audit/rl-scheduling/gnn-v1/`: `selected.pt`, `last_rl.pt`,
`warmstart.pt`, frozen `config.json`, detailed `report.json`,
`comparison.csv`, `verification.json` and `sources/`. The MLP workload corpus
remains reserved as validation, with no independent third test family.

## Continued training: gnn-v2 (2026-10-29)

The selected `gnn-v1` weights were continued for **252 additional simulated
RL episodes** (12 passes over 21 factorization/scenario cases). Adam and the
per-case moving baselines were restarted because the previous pilot saved
weights only. No additional imitation passes were performed; the 315 local
training episodes were checksum-checked and supplied fixed FIFO normalization.
The architecture, learning rate 0.0002, entropy coefficient 0.0001 and objective
were unchanged. No archived `runs/` payloads or remote inputs were needed.

```bash
ygaraty rl-gnn \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --mlp-checkpoint /absolute/path/to/model-v1/selected.pt \
  --resume-checkpoint /absolute/path/to/gnn-v1/selected.pt \
  --rl-epochs 12 \
  --output /absolute/path/to/new-gnn-directory
```

This continuation protocol uses fixed selection seeds 500–504 and final seeds
600–609, distinct from the first pilot. Training uses environment seeds
30000 + epoch * 100 + case index, policy seeds 400000 + epoch * 100 + case
index, and shuffle seed 2029. These ranges are fixed for this experiment, not
a fresh-seed allocator for subsequent invocations. Configuration was written
before fitting. Both the frozen previous GNN and frozen MLP were reevaluated
on the same final graph/scenario/seed grid as the continued GNN and heuristics.

Selection used factorization only and included the unchanged initial weights
as epoch 0. Epoch 12 was selected: its score was 1.0005 versus 1.0147 initially,
relative to the original training-corpus FIFO normalization. This score is not
the final ratio below, which uses fresh, paired FIFO rollouts. The MLP workload
family was excluded from fitting and selection, but its previous results had
already been inspected; fresh seeds do not make it a new independent test family.

The final evaluation contains **3,360 unique rollouts**: eight policies, 42
cases, ten seeds. Equal-case mean ratios of simulated makespan to FIFO are:

| Policy | Factorization | MLP workload family |
|---|---:|---:|
| Continued GNN | 1.0156 | 1.0162 |
| Previous GNN, frozen | 1.0256 | 1.0142 |
| MLP policy, frozen | 1.0430 | 1.0029 |
| FIFO | 1.0000 | 1.0000 |
| Shortest estimated task | 1.0390 | 1.0187 |
| Critical path | 0.9827 | 0.9877 |
| Cache-aware | 0.9853 | 1.0000 |
| Random | 1.0382 | 1.0274 |

Mean paired gains against the previous GNN are **+0.92%** on factorization
(12/21 case wins) and **-0.21%** on MLP graphs (9/21 wins). Against the frozen
MLP policy they are +2.61% and -1.32%, respectively. These are averages of
per-case ratios, not ratios of aggregate table values. Critical path has the
best aggregate on both families. More RL updates did not produce a general
scheduling improvement or a reason to replace the heuristics. No further
hyperparameter tuning was performed after these final results were observed.

The complete experiment took **228.90 seconds** of measured host time on one
CPU thread. Inference/encoding time per choice was about 0.121 ms for the GNN
and 0.112 ms for the MLP on factorization, and 0.090/0.078 ms on MLP graphs.
These are local host observations, excluded from simulated makespan, not
production coordinator measurements. Ten seeds and one continued model lineage
do not establish training-seed robustness, tail behavior, or unseen-family
performance.

Artifacts live in the workspace parent's
`calibration-audit/rl-scheduling/gnn-v2/`: selected/initial/last weights,
`training_state.pt` (last model, Adam state and moving baselines), frozen config,
full report, comparison CSV, source snapshots, and verification output.
The CLI's `--resume-checkpoint` takes weights only; loading the optimizer state
from `training_state.pt` is not yet implemented by that option. Verification
checks the full result grid, summary recomputation, changed trained weights,
unchanged baseline checkpoints, and exact fresh-process reload outcomes in
both families. The 13 focused RL tests pass. All execution used local inputs
with socket connections and scheduled workload kernels disabled.

## Actual runtime scheduler baseline (2026-10-29)

Code inspection corrects the distinction between the strongest experimental
heuristic and the actual runtime default:

- `runtime/scheduler.py`: ready tasks initially enter a deque in task order;
  `claim(..., choose=None)` uses `popleft()`. Newly unblocked tasks normally
  join the back, and retries join the back. `consumers_first` defaults to false.
- `coordination/jobs.py:Job.claim_now`: ordinary unowned jobs pass no selector.
  Optional memory-budget enforcement selects the first ready task that fits;
  owned jobs instead use `coordination/routing.py:OwnerRouting` for first/capacity
  placement and ownership eligibility. These are not critical-path policies.
- `submission/jobs.py:CompiledJob.make_step` constructs the same default scheduler
  and inherits the claim path. This is ready-queue FIFO for eligible unowned tasks.
- `coordination/factorization_job.py:ShardedFactorizationJob.make_step` sets
  `consumers_first=True`: newly unlocked consumers go to the front. This is
  intended to consume/free intermediate results sooner. It is not critical-path
  ranking, and several children unlocked together follow the scheduler's actual
  repeated-appendleft ordering.

The new `ygaraty.simulation.rl_runtime_baseline` CLI executes the real
`Scheduler.claim` default branch inside the offline simulator, with no selector
callback. It also measures the consumers-first flag as a separately labeled
runtime variant. The existing five heuristic collectors and three frozen neural
checkpoints are reevaluated on the same 42 cases and seeds 600–609. No retraining
or checkpoint selection occurs. Inputs, socket guards and workload-execution
guards remain local/offline. The CLI accepts `--training-corpus`,
`--heldout-corpus`, `--gnn-checkpoint`, `--previous-gnn-checkpoint`,
`--mlp-checkpoint`, and a fresh `--output` directory.

All **420 default-runtime rollouts match the old FIFO baseline exactly in task
assignment order and simulated makespan**. The complete grid has 4,200 unique
rollouts. All 3,360 previously evaluated policy/seed results reproduce exactly.
Equal-case mean simulated makespan ratios to Ygaraty's default are:

| Policy | Factorization | MLP workload family |
|---|---:|---:|
| Ygaraty default runtime (FIFO) | 1.0000 | 1.0000 |
| Ygaraty consumers-first variant | 1.0355 | 1.0408 |
| Continued GNN | 1.0156 | 1.0162 |
| Previous GNN | 1.0256 | 1.0142 |
| MLP policy | 1.0430 | 1.0029 |
| FIFO collector | 1.0000 | 1.0000 |
| Shortest estimated task | 1.0390 | 1.0187 |
| Critical path | 0.9827 | 0.9877 |
| Cache-aware | 0.9853 | 1.0000 |
| Random | 1.0382 | 1.0274 |

The continued GNN is about 1.56%/1.62% slower than the default on these two
families. Critical path is about 1.73%/1.23% faster, but remains an experimental
heuristic, not the runtime default. Consumers-first is slower on these graphs;
this does **not** demonstrate that it is worse for its intended sharded workload
or memory objective. The corpus contains tiled/fused dense-factorization and
legacy MLP graphs, not the sharded-factorization production job. No peak-result
memory benefit is measured here. Ownership, worker memory-budget eligibility,
lease failures, coordinator-side metadata fast paths, production overhead, and
multi-update execution are outside this single-update task-order comparison.
The native/default runner rejects owned, replicated, or affinity-overridden plans
rather than mislabeling them as ordinary FIFO.

The run took 110.50 seconds measured on the host. Artifacts are in
`calibration-audit/rl-scheduling/runtime-baseline-v1/` in the workspace parent:
`config.json`, `report.json`, `comparison.csv`, `verification.json`, and source
snapshots including the inspected job implementations. All 15 focused RL tests
pass, including runtime FIFO parity, consumers-first ordering, and unsupported
scope rejection. The complete result grid, previous rollout equality, aggregate
recomputation and source snapshot hashes were independently checked. No live
scheduler behavior was changed.

## Critical-path residual GNN (2026-10-30)

`ygaraty.simulation.rl_residual` adds a bounded learned correction to critical-path
ordering. It initializes the graph encoder and actor hidden layer from `gnn-v2`,
zeros the final actor layer, and trains all policy parameters. For each ready set,
let `r` be the existing log-transformed downstream-cost feature. The score is
`4 * (r - max(r)) / max(max(r) - min(r), 1e-8) + 0.5 * tanh(gnn_score)`.
The prior preserves critical-path ordering and queue-order ties at initialization;
its double-precision rank features avoid rounding them to float32. The bounded
correction primarily changes decisions between tasks with similar priorities.
Static graph embeddings remain cached per rollout. This is a hybrid heuristic
plus GNN, not evidence that graph message passing alone beats critical path.

Training uses self-critical REINFORCE: a same-seed greedy rollout is the
control-variate baseline for each stochastic rollout, and the advantage is
`(greedy_makespan - sampled_makespan) / critical_path_makespan`. The greedy and
critical-path baselines receive no gradients and do not depend on the sampled
actions. Adam uses learning rate 0.001, gradient clipping at 1, and the existing
summed decision entropy bonus coefficient 0.0001. Adam is restarted. Equal-case
passes use only the factorization graphs, without additional imitation or
workload execution. Numeric seed matching is not task-aligned common random
numbers.

The fixed protocol completed 24 passes (**504 RL episodes**), evaluated each
checkpoint on selection seeds 700–704, and retained the zero-residual starting
policy as an eligible checkpoint. Epoch 7 was selected (selection ratio 0.983911
to critical path); later training updates did not improve selection. Final seeds
800–819 were used only after freezing selection. They are distinct from all
prior experiments. The family split was retained, but these graph families were
already inspected earlier; this is not a new independent family test.

```bash
ygaraty rl-residual \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --gnn-checkpoint /absolute/path/to/gnn-v2/selected.pt \
  --mlp-checkpoint /absolute/path/to/model-v1/selected.pt \
  --output /absolute/path/to/new-residual-directory
```

The final grid contains **7,560 unique rollouts**: nine policies, 42 cases,
20 seeds. Equal-case mean simulated makespan ratios to **critical path** are:

| Policy | Factorization | MLP workload family |
|---|---:|---:|
| Residual GNN | **0.9900** | 1.0002 |
| Critical path | 1.0000 | **1.0000** |
| Zero-residual prior ablation | 1.0000 | 1.0000 |
| Previous GNN (gnn-v2) | 1.0453 | 1.0288 |
| MLP policy | 1.0703 | 1.0157 |
| FIFO / default ordering | 1.0264 | 1.0126 |
| Shortest estimated task | 1.0759 | 1.0324 |
| Cache-aware | 1.0148 | 1.0126 |
| Random | 1.0661 | 1.0401 |

The learned correction improves factorization mean case-level makespan by
**0.9985%**, with 12/21 case wins. On MLP graphs it is **0.0193% slower**,
with 6/21 wins. A 5,000-replicate seed-block bootstrap (seed 2032, resampling
20 seed columns jointly across cases, recomputing ratios of per-case means)
gives percentile 95% gain intervals of **[0.3824%, 1.5258%]** and
**[-0.0509%, 0.0140%]**, respectively. Those intervals describe seed variation
conditional on these fixed graphs and the selected checkpoint; they do not
capture graph-family or training-initialization uncertainty. This is a small
factorization win, not a general win over critical path or a deployment claim.
No further tuning used the final results.

The zero-residual ablation matches critical path exactly in all 840 final
rollouts, in addition to the initial 105 selection rollouts. Verification checks
all 7,560 result keys, aggregate recomputation, distinct seed partitions, changed
message-layer weights, unchanged original GNN/MLP checkpoints, source hashes,
and exact fresh-process checkpoint reload outcomes in both families. All 17
focused RL tests pass. The experiment took **630.16 seconds** of measured host
time on one CPU thread. All makespans are simulated and exclude policy inference
cost; no actual model workload was trained or service contacted.

Artifacts are in the workspace parent's
`calibration-audit/rl-scheduling/gnn-residual-v1/`, including `selected.pt`,
`initial.pt`, `last_rl.pt`, `last_optimizer.pt`, config, report, comparison CSV,
source snapshots, and verification/uncertainty results. **Load `selected.pt` with
`ResidualGNN`, not `GNNPolicy`**: their weight shapes are compatible but the
critical-path prior and bounded scoring behavior differ. The selected policy
has not replaced the runtime scheduler.

## Counterfactual data and ablations

The next offline experiment collected 155 paired critical-path decisions and ran
1,092 ablation rollouts. The gain persists with identical workers, zero transfer
delay and no caching, and is concentrated in the tiled graph; fused graphs show
essentially no gain. See [the dataset schema, replay protocol, results and limits](COUNTERFACTUAL.md).

Counterfactual preference fine-tuning has now been evaluated: the unchanged
residual policy won checkpoint selection, so it remains the current model.
The final trained candidate retained a factorization advantage over critical
path but did not improve the current GNN. See [training results](COUNTERFACTUAL.md#training-on-the-pairs-2026-10-30).

A subsequent [metadata audit and offline beam search](BEAM.md) found modest
improvements over deterministic CP/GNN schedules. Noisy transfer controls show
that much of the factorization benefit also comes from simply freezing a GNN
schedule into a global priority list; it should not all be credited to beam search.
