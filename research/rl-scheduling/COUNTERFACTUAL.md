# Counterfactual dataset and gain ablations

Completed offline on 2026-10-30 using the frozen residual GNN from
`gnn-residual-v1/selected.pt`. No model weights were updated and no live workload,
service, network connection, dataset download, or archive restoration was used.
The existing shape-only graphs and cost tables were sufficient.

## Dataset

The workspace parent's `calibration-audit/rl-scheduling/counterfactual-v1/`
contains **155 paired training examples** in `counterfactuals.jsonl.gz`, alongside
compressed graph snapshots, the local cost model, a manifest, source snapshots,
and verification results. There are 22 alternative wins, 22 critical-path wins,
and 111 ties (absolute makespan tolerance 1e-12 seconds). The many ties are real
labels; they should not be relabeled as successful alternatives.

The collector visits critical-path trajectories for the three factorization
training graphs and seven scenarios. It selects at most four decision points per
case, preferring evenly spaced states where the GNN disagrees with critical path,
then filling from other choice states. Up to two alternatives come from the GNN
proposal and the next critical-ranked candidates. This is intentionally a small,
disagreement-enriched dataset, not a representative sample of arbitrary states.
The MLP workload family contributes **no training examples**.

At each selected state, the collector replays the same critical-path action prefix
from the beginning, then chooses either critical path's action or an alternative.
Both branches finish with critical path. The original branch reproduces the entire
original action sequence and makespan. A fingerprint checks ready/waiting tasks,
committed/running tasks, requester, clock, cache contents and LRU order, link/server
availability, pending event identifiers/times, queues, parked workers and RNG state
at the branch. Random UUID lease tokens are excluded because they authenticate
results but do not affect simulated timing. This is deterministic prefix replay,
not cloning Python generators or a claim to serialize a complete simulator state.

**Noise and request jitter are zero for every counterfactual.** Consequently,
future execution does not depend on schedule-dependent assignment of random timing
draws. This avoids the existing same-numeric-seed pairing problem. It does not
solve stochastic state branching; that remains future work.

Each example includes legal candidates and their feature vectors, the policy
observation, prefix actions, worker, branch-state digest, critical and alternative
actions, continuation policy, both makespans, and `gain_s`/`gain_fraction`.
Positive gain favors the alternative. These are continuation-specific action
comparisons, not optimal Q-values or measured real-system labels. Observations
exclude hidden future events; the fingerprint is audit metadata, not a model input.

For training, use pairwise ranking or regress the signed gain. Keep ties explicit,
and avoid treating their floating-point sign as evidence. Alternative rows from
one state share `state_group`; graph/family-level splits are required for any
future validation, rather than random row splits. The current manifest marks
all pairs as training data. It does not claim an independent validation dataset.

One example: on `tiled_1024/lan_4`, choosing `v1/block/0/0` rather than
`prediction/block/640/128` at decision 42 saves **0.14156 simulated seconds**
(**5.47%** of that deterministic episode), with an identical prefix and critical-
path continuation. That single-branch gain is not the overall policy gain.

## Ablations

**1,092 complete rollouts** compare the frozen residual GNN and critical path:
42 graph/scenario cases × (eight deterministic factorial combinations + five
fresh noisy seeds) × two policies. No checkpoint selection or tuning is performed.

The factorial experiment disables noise/jitter in every cell, including its full-
conditions control, and toggles all combinations of:

- Identical workers: remove worker profiles, including compute/link heterogeneity
  and late arrivals; retain each scenario's default links, reference compute speed,
  and worker count. This does not conserve aggregate compute capacity.
- No transfer delay: unlimited worker/relay bandwidth and zero network RTT/jitter.
  Coordinator service and per-task claim service remain. Bytes are still counted.
- No caching: set worker LRU capacity to zero; keep graph reusable-input semantics.
  Metadata-only relay fetches/control requests may still exist even with unlimited
  links. This is not a simulator with every non-compute operation removed.

Positive values below mean lower mean case-level makespan than critical path.

| Conditions | Factorization gain | MLP-family gain |
|---|---:|---:|
| Full conditions, new noisy seeds 1000–1004 | +1.4395% | -0.0156% |
| Full conditions, noise/jitter disabled | +0.6934% | +0.0907% |
| No cache | +0.9032% | +0.0907% |
| No transfer delay | +1.5713% | -0.1209% |
| No transfer delay or cache | +2.1570% | -0.1209% |
| Identical workers | +0.7779% | +0.0336% |
| Identical workers, no cache | +0.6813% | +0.0336% |
| Identical workers, no transfer delay | +1.3611% | -0.1995% |
| Identical workers, no transfer delay or cache | +1.3597% | -0.1995% |

All rows other than the first are deterministic. The first row is a new five-seed
diagnostic, not a replacement for the earlier 20-seed final evaluation or its
uncertainty interval. Individual ablations and their interactions change absolute
makespans and sometimes worker assignments; these percentages cannot be added as
a causal decomposition of the previous 1% improvement.

### What explains the gain?

The factorization gain is concentrated in **tiled_1024**:

| Graph | New noisy comparison | Deterministic full | Identical workers, no transfer/cache |
|---|---:|---:|---:|
| tiled_1024 | +4.3441% | +2.0802% | +4.0792% |
| fused_1024 | 0.0000% | 0.0000% | 0.0000% |
| fused_4096 | -0.0255% | 0.0000% | 0.0000% |

The gain survives without worker heterogeneity, caching, or network transfer
latency, and without timing randomness. These effects are therefore **not
necessary** for the observed tiled-graph improvement. The evidence supports a
ready-task ordering/dependency-timing benefit under finite worker and coordinator
resources. It does not isolate pure dependency parallelism from batching, metadata
fast paths, or coordinator service. No peak-memory objective was measured. The
MLP-family changes are small and mixed, and the fused graphs offer little gain;
this remains a narrow result rather than a general scheduling advantage.

## Reproduction and checks

```bash
ygaraty rl-counterfactual \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --checkpoint /absolute/path/to/gnn-residual-v1/selected.pt \
  --output /absolute/path/to/new-counterfactual-directory
```

All 19 focused RL tests pass. A fresh process reproduced all **310 branch
rollouts**, their observations and feature vectors. Verification also checks
legal actions, unique pair/result keys, signed labels, source/data checksums,
aggregate recomputation, and six graph-level parity checks against the existing
critical-path and GNN runners. The original experiment took **56.60 seconds**
of measured host time, excluding subsequent verification. The GNN checkpoint
remained unchanged. Data and ablation rows are gzip-compressed on write; a ZIP
bundle is provided for convenient local reuse.

## Training on the pairs (2026-10-30)

`ygaraty.simulation.rl_pairwise` fine-tuned the frozen residual GNN on all 155
training-family pairs. This is supervised counterfactual preference learning,
not an additional on-policy RL experiment. The bounded critical-path prior and
network architecture were unchanged. No MLP-family labels entered fitting or
checkpoint selection.

The loss is binary cross-entropy on the alternative-minus-critical task scores.
An alternative win has target 1, a critical-path win target 0, and a tie target
0.5. Non-tie weights are `clip(abs(gain_fraction)/0.01, 0.1, 5)`; ties have weight
0.1. Each state's rows share equal total state weight, so states with two
alternatives are not counted twice. A KL penalty with coefficient 0.02 anchors
the full ready-set distribution to the frozen original policy. All GNN weights
are trainable; Adam uses learning rate 0.0002 and gradient clipping at 1.

A fixed 60 full-batch epochs were evaluated every 10 epochs, using factorization
selection seeds **1100–1104**. The unchanged starting policy was an eligible
checkpoint. The selection score is mean rollout makespan normalized by paired
critical-path makespan. The training objective fell from **0.15425 to 0.14870**,
and message-layer weights changed, but no trained checkpoint improved selection:
epoch 0 scored 0.995524, while the best trained candidate scored 1.006515 at epoch
10. Epoch 0 was retained. This demonstrates why decreasing preference loss is not
sufficient evidence of better scheduling performance.

The final comparison uses **20 fresh seeds, 1200–1219**, on both families.
The selected/current/critical-path evaluation has 2,520 rollouts. A separately
labeled diagnostic evaluates the already saved final trained checkpoint on the
same final grid, adding 840 rollouts for **3,360 total**. This diagnostic was
announced after selection, before its final evaluation, and was not used to
change the selected checkpoint or tune hyperparameters.

| Policy | Factorization ratio to critical path | MLP-family ratio |
|---|---:|---:|
| Current residual GNN | **0.988562** | 1.000479 |
| Final pairwise-trained checkpoint (epoch 60) | 0.990005 | 1.000206 |
| Selected checkpoint (epoch 0; unchanged) | **0.988562** | 1.000479 |
| Critical path | 1.000000 | **1.000000** |

The final trained model still beats critical path by about 1.00% on factorization,
but does not improve the current model's roughly 1.14% gain on these new seeds.
Its MLP-family ratio is slightly closer to critical path, but there is no evidence
here for replacing the current policy. The unchanged selected/current checkpoints
produce identical makespans in all 840 final cases/seeds.

The limited 44 non-tied examples, disagreement-enriched states, deterministic
labels, and critical-path continuation are plausible limitations: the learned
policy's subsequent decisions and noisy evaluation can differ from the label
rollouts. This experiment does not isolate which limitation caused the regression.
The existing families were already inspected, and this remains one training
lineage. No new claim of unseen-family generalization or statistical significance
is made from the small differences. We did not retune after seeing final results.

```bash
ygaraty rl-pairwise \
  --training-corpus /absolute/path/to/factorization-v1 \
  --heldout-corpus /absolute/path/to/mlp-validation-v1 \
  --pair-dataset /absolute/path/to/counterfactual-v1 \
  --checkpoint /absolute/path/to/gnn-residual-v1/selected.pt \
  --output /absolute/path/to/new-pairwise-directory
```

Artifacts are in the workspace parent's
`calibration-audit/rl-scheduling/gnn-counterfactual-v1/`. `selected.pt` retains the
current policy; `last_trained.pt` is the final trained diagnostic model. Both must
be loaded with `ResidualGNN`. Detailed selection and final rollout files are gzip-
compressed. `comparison_with_trained.jsonl.gz` and `verification.json` include the
additional diagnostic; `report.json` records the original selection/evaluation
run. A ZIP bundle contains checkpoints, configs, sources, reports and checks.

All **21 focused RL tests pass**, including preference-gradient direction and
neutral ties. Verification checks the complete result grid, summaries, checkpoint
immutability, trained message-layer updates, compressed artifact/source checksums,
selection choice, seed separation and exact fresh-process reloads in both families.
The original training/selection/final run took **164.66 seconds** of measured host
time, excluding the subsequent final-trained diagnostic and verification. All
work was offline; no numerical workload kernels, services or external inputs were
used. No runtime scheduler change was made.
