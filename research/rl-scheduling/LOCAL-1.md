# LOCAL-1: the residual GNN on an unseen workload family, with inference cost charged

Run 2026-10-30 on the Omarchy workstation (AMD Ryzen 9 PRO 6950H, Python 3.14.7, PyTorch 2.14.1 CPU, one thread), following the LOCAL-1 plan (appended below). Only the scheduling-policy network trained; every workload ran in simulated time, with socket connections and the workload kernel executor disabled.

**Result: rejected.** On the held-out Transformer family the retrained residual GNN does not beat critical path even when inference is free, and it loses once its measured decision time is charged:

| Charge on the simulated coordinator | Mean improvement over critical path (5 seeds × 21 cases × 20 seeds) | 95% paired bootstrap interval |
|---|---:|---:|
| None | −0.07% | −0.22% to +0.04% |
| Measured on this host | −0.33% | −0.42% to −0.24% |
| Coordinator 4× slower | −1.57% | −1.66% to −1.48% |

The frozen rule needed a positive lower bound under the measured charge and no scenario regressing more than 2%. The lower bound is −0.41%; the worst scenario regresses 1.29% (`lan_4`). Critical path stays the scheduler. The 1% factorization gain from the [residual GNN](GNN.md) does not transfer to a different model family.

## What ran

| Split | Family | Graphs (tasks) | Use |
|---|---|---|---|
| Train | Dense SGD factorization | `tiled_1024` (662), `fused_1024` (98), `fused_4096` (358) | fitting |
| Validation | Legacy MLP forward/backward | `mlp_76`, `mlp_77`, `mlp_340` | checkpoint selection |
| Test | Transformer translator | `transformer_tiny` (899), `transformer_tiny_b8` (963), `transformer_30m` (1,344) | final evaluation only |

Each graph runs under the corpus's seven scenarios (`loopback_1`, `lan_4`, `home_4`, `home_16`, `congested_16`, `heterogeneous_8`, `late_join_8`). The protocol, seeds, arms and acceptance rule were written to [`local-1/protocol.json`](local-1/protocol.json) (SHA-256 `82693114…`) before any fitting, and nothing was changed after. Arms:

- **Critical path**, the baseline, and **FIFO**, descriptive only.
- **Zero-residual ablation.** The residual head is zeroed. With no charge it reproduced critical path in all 420 rollouts of each split, 1,260 in total, as asserted.
- **Five retrained residual GNNs**, the primary arm. Initialization seeds 3000–3004 give each a fresh random graph encoder with a zeroed residual head, so epoch 0 is critical path. All five share the `rl-residual` objective, learning rate and entropy, and the same budget of 24 epochs × 21 training cases. Each selects its checkpoint on the validation family against paired critical path, with epoch 0 as a candidate.
- **The `rl-residual` recipe re-run on this host** (`residual_rerun`), the secondary arm. See the deviations below.

Final evaluation used environment seeds 1000–1019 on every family: 34,020 rollouts.

**Charging inference.** Decision latency was measured on this host with one thread, 50 warm-ups and the median of 21 repeats. The samples were 1,397 decisions drawn from critical-path rollouts of all nine graphs, and a straight line in ready-queue length was fitted to them:

| Policy | Fixed cost | Per ready task | Median decision |
|---|---:|---:|---:|
| Residual GNN | 74.6 µs | 1.19 µs | 133 µs |
| Critical path | 0.23 µs | 0.06 µs | 3.1 µs |

Graph precomputation takes 0.19–1.04 ms for the GNN embedding and 0.11–2.85 ms for the critical-path ranks. A new opt-in hook (`Simulation.decision_s`) charges these costs to the simulated coordinator. A claim holds the coordinator for its decision while other workers keep computing. The precomputation holds the coordinator before the first claim, charged in full in each one-update episode. That is the upper end: across reused updates it amortizes toward zero, but it is under 3 ms here and is not what decides the result. FIFO is never charged.

## Results

Mean over cases of each arm's mean makespan divided by critical path's, on the test family. Lower is better, and wins count cases out of 21 where the arm beats critical path.

| Arm | No charge | Wins | Measured charge | Wins | Coordinator 4× slower |
|---|---:|---:|---:|---:|---:|
| Critical path | 1.0000 | — | 1.0000 | — | 1.0000 |
| Residual, seed 0 | 1.0010 | 5 | 1.0037 | 7 | 1.0160 |
| Residual, seed 1 | 1.0003 | 8 | 1.0028 | 9 | 1.0155 |
| Residual, seed 2 | 1.0009 | 9 | 1.0029 | 6 | 1.0152 |
| Residual, seed 3 | 1.0007 | 10 | 1.0033 | 8 | 1.0164 |
| Residual, seed 4 | 1.0007 | 5 | 1.0037 | 6 | 1.0156 |
| Residual, recipe re-run | 1.0029 | 5 | 1.0052 | 7 | 1.0174 |
| Zero-residual ablation | 1.0000 | 0 | 1.0029 | 8 | 1.0158 |
| FIFO | 1.0118 | 3 | 1.0079 | 7 | 1.0073 |

The zero-residual row isolates the cost of running the network: it makes critical path's decisions but pays the GNN's latency, and that alone costs 0.29%. Under the measured charge, the scenarios with one or few workers suffer most: `lan_4` +1.29% and `loopback_1` +1.22%. That is where claims are sequential and every microsecond of coordinator time shows. With the coordinator 4× slower those two rise to +5.8% and +4.9%. On the congested and heterogeneous pools the charged residual is slightly ahead (−0.28% and −0.41%), but not by enough to offset the rest. FIFO trails critical path by up to 21% (`transformer_tiny_b8` on `home_16`), and by 10–14% in three other cases. So ordering matters on Transformer graphs, but the learned residual adds nothing to critical path's ordering.

The other families, as descriptive results only. Training-family numbers are not evidence of generalization:

| Family, charge | Retrained residual: mean improvement over critical path | Per seed |
|---|---:|---|
| Validation (MLP), none | −0.003% (−0.015% to +0.007%) | −0.02 to +0.01% |
| Validation (MLP), measured | −0.20% (−0.32% to −0.08%) | −0.21 to −0.16% |
| Train (factorization), none | +0.49% (−0.27% to +1.19%) | −0.04, +0.09, +0.46, +0.68, +1.27% |
| Train (factorization), measured | −0.45% (−1.51% to +0.52%) | −1.06 to +0.24% |

Even on its own training family, the gain depends on the initialization seed: from −0.04% to +1.27%, with an interval that includes zero. The earlier single-lineage +1% was within that spread. The selected epochs were 11, 7, 10, 8 and 1. Their validation scores were 0.9998–1.0000, so selection on the MLP family had almost nothing to choose between ([records](local-1)).

## Deviations from the plan, and why

- **Inputs rebuilt here, not copied.** The other workspace's `calibration-audit/` artifacts are not on this machine.
  - The factorization corpus was rebuilt from the recorded settings. Its graph hashes (`48c1b2ce…`, `f2d9fe46…`, `6fa3f32d…`) and cost digest (`8c1a1ee8…`) match, and so do its 315 episodes and 117,390 transitions.
  - model-v1 retrained to exactly the published numbers: epoch 4, 1.03985 against FIFO and 1.13203 against the best heuristic.
  - The MLP graphs were re-imported from the 2026-10-29 `runs/` backup in Spaces. Every copy of each size gives one graph: `4f3ef99c…` (76 tasks), `af2bd85a…` (77), `e796d945…` (340). Their full hashes are in the protocol.
  - gnn-v1 then reproduced the published table exactly, including the MLP-family column, so these are the graphs the earlier work used.
- **The frozen residual checkpoint could not be recreated.** gnn-v2 starts from the same weights and scores the same 1.0147 at epoch 0. Its 12 policy-gradient epochs then diverge from the published run: it selects epoch 7 instead of 12 and scores 1.0311 against FIFO instead of 1.0156. The code and seeds are unchanged, so floating-point differences between the Mac's ARM build and this x86 host are the likely cause.
  - The residual trained on it (`residual_rerun`, checkpoint `33ef5c5a…`) scores 0.9944 on factorization and 1.0004 on MLP, against the published 0.9900 and 1.0002. That is the same qualitative result from the same recipe, not the same weights.
  - It stands in for the "frozen current residual GNN" arm. The primary evidence, the five independent seeds, does not depend on it.
- **Initialization.** "Five initialization seeds" was implemented as five fresh random encoders. The published residual instead starts from gnn-v2's encoder, and that lineage is the re-run arm.
- **Selection on the validation family.** The published residual selected on factorization, its own training family. Here the plan's family separation applies: fit on factorization, select on MLP, evaluate on Transformer.
- **The third family.** Every importable legacy plan in the backup is another MLP. The LSTM and Adam language-model plans there have no leaf shapes, so the importer refuses them.
  - The Transformer graphs were therefore built by the repository's own planner (`ygaraty simulate run --plan-output`). They use the 1M recipe at bands 16 and 8, and the 30.3M sizes of experiment 30, on the local OPUS-100 database. All are unowned and ungrouped.
  - They are a different model family from both earlier ones: fused attention layers, embeddings, scatter and Adam. The cost table was fitted on Transformer profiles, so these ops have measured costs, whereas factorization falls back for some.
  - The family was packaged before training, and none of its outcomes were looked at until the final evaluation.

## Limitations

- **Simulated, not measured.** Every makespan is simulated. Only the decision latencies and precomputation times are measurements, taken on one host. A server-class coordinator would be faster, and a small cloud instance slower. The 4× row brackets the slower case.
- **One test family.** The conclusion holds for Transformer translator graphs under these seven scenarios. Within that family, the three graphs share most of their structure.
- **The cost is a fitted line.** Charged cost uses the latency fit, not per-call timings. The GNN's cost grows only 1.19 µs per ready task, so the fit is dominated by its 75 µs fixed cost.
- **The policy's inputs are limited.** It still sees partial observations: no cache identities, no worker or network features. A policy that saw those could behave differently. That is LOCAL-2's question, and this run gives LOCAL-2 no repeatable failure pattern to start from. The residual's small wins and losses scatter across cases and seeds.

## Reproduce

Run these from the repository root. All inputs are local: the OPUS-100 SQLite database, the `runs/` backup's MLP plans, and the bundled cost model. The full sequence took about 25 minutes here.

```bash
ygaraty simulate-factorization --rows 1024 --columns 1024 --rank 32 --tile 128 --implementation reference --plan-output runs/local-1/plans/tiled_1024.json
ygaraty simulate-factorization --rows 1024 --columns 1024 --rank 32 --implementation fused --block-rows 64 --plan-output runs/local-1/plans/fused_1024.json
ygaraty simulate-factorization --rows 4096 --columns 4096 --rank 32 --implementation fused --block-rows 64 --plan-output runs/local-1/plans/fused_4096.json
ygaraty rl-import --family mlp --output runs/local-1/plans/mlp_76.json  <backup>/runs/optimized-comparison/baseline/step-0000/plan.json   # 77, 340: large-benchmark-20261021/tile-128 and tile-64
ygaraty simulate run --dataset-url sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite --dataset opus100-en-pt \
  --dataset-version 1m-seed2026-v2k --embedding 128 --hidden 384 --layers 2 --heads 4 --band 16 --tile 128 --batch 64 --workers 4 \
  --plan-output runs/local-1/plans/transformer_tiny.json      # band 8 for _b8; 30m: version 1m-seed2026, 512/2048/3/8, band 8
cp ygaraty/simulation/cost_model.json runs/local-1/
cp public/research/rl-scheduling/local-1/*-campaign.json runs/local-1/
ygaraty rl-campaign --output runs/local-1/factorization-v1 runs/local-1/factorization-campaign.json   # and mlp-, transformer-
ygaraty rl-train runs/local-1/factorization-v1 --output runs/local-1/model-v1 --epochs 4
# gnn-v1, gnn-v2 (--resume-checkpoint gnn-v1/selected.pt --rl-epochs 12), rl-residual: the commands in GNN.md
python -m ygaraty.simulation.rl_heldout freeze --study runs/local-1/study --train runs/local-1/factorization-v1 \
  --validation runs/local-1/mlp-validation-v1 --test runs/local-1/transformer-test-v1 --rerun-checkpoint runs/local-1/gnn-residual-v1/selected.pt
for k in 0 1 2 3 4; do python -m ygaraty.simulation.rl_heldout train --study runs/local-1/study --seed $k & done; wait
python -m ygaraty.simulation.rl_heldout latency --study runs/local-1/study
python -m ygaraty.simulation.rl_heldout evaluate --study runs/local-1/study --processes 14
python -m ygaraty.simulation.rl_heldout summarize --study runs/local-1/study
```

The [`local-1/`](local-1) folder holds the frozen protocol, the three campaign configs, every seed's selection curve, the latency measurements, the summary, and all 34,020 final rollouts (`final_rows.json.gz`). Checkpoints and corpora stay in `runs/local-1/`, which is not committed.

## Plan as frozen on 2026-10-30 (LOCAL-1)

“Local” means training and evaluating the scheduling-policy network against simulated Ygaraty workloads. Policy weight updates are allowed. Actual factorization, Transformer, ConvNet and DiLoCo training, workload kernels, live coordinators/workers, browsers, relays, network emulators and external stores are not allowed on this machine.

Use existing local data only. Keep socket/workload-executor guards active; do not download missing inputs. If a dataset-backed simulator is used, provide an explicit existing local SQLite URL. Save policy checkpoints, trajectories, source snapshots and reports locally, with no synchronization.

**Priority: first.** The residual policy improved simulated factorization makespan by about 1%, with no established MLP-family improvement. Its current evidence excludes inference overhead and independent workload-family uncertainty.

References: [GNN protocol/results](GNN.md), [counterfactual study](COUNTERFACTUAL.md), [original policy](TRAINING.md).

### Prerequisites

- Verify existing corpus manifests, graph/cost hashes, checkpoint types and feature definitions. Load the selected residual checkpoint with `ResidualGNN`, not the shape-compatible older policy class.
- Identify a third independent workload family from existing local metadata. Validate dependency/leaf shapes and output/ownership/cache semantics before importing. Keep all variants of one family in the same partition.
- If no trustworthy third family exists, report that blocker. A new size or seed of a previously evaluated graph is not an independent family.
- Freeze a new protocol and seed manifest. Previously examined MLP and factorization results are development evidence, not fresh final-test data.

### Experiment arms and controls

1. Critical-path heuristic.
2. Frozen current residual GNN.
3. Zero-residual ablation, which must reproduce the heuristic.
4. Residual GNN trained from at least five initialization seeds, using the same rollout budget and training corpus for each.

Use family-separated validation for checkpoint selection and an untouched family for final evaluation. Evaluate at least 20 environment seeds per final case. Do not select a checkpoint or alter hyperparameters using the final family. Preserve FIFO as a descriptive reference, not the main acceptance baseline.

### Charge inference cost explicitly

Measure actual local policy and heuristic decision latency at representative ready-queue sizes, with fixed thread settings, warmup and repeated trials. Include graph-embedding construction and report its amortization across graph reuse.

Inject measured scheduling cost into the simulator's coordinator resource rather than merely adding all worker decisions to final makespan; scheduling and execution may overlap. Publish both zero-cost and charged-cost results, and sensitivity to a slower coordinator. Label injected overhead as a modeling assumption based on measured local timings.

### Proposed acceptance rule, to freeze before final evaluation

Require a positive lower 95% confidence bound for mean improvement over critical path on the held-out family after charging overhead, and no predeclared scenario's mean regression above 2%. Report uncertainty across both training initializations and environment seeds. With only one held-out family, restrict conclusions to that family.

If the gain disappears after overhead, retain critical path and report the negative result. Do not expand model size merely to improve training-family results.

**Deliverable:** frozen split/seed manifest, checkpoint-selection records, per-family/per-case comparison, latency measurements, charged-cost simulations and complete negative results. Separate measured policy-training runtime from simulated workload completion times.
