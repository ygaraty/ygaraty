# Trigger the offline 81-expression experiment

From the repository root, run:

```bash
public/research/rl-scheduling/run-cse-81.sh
```

Or choose a new output directory:

```bash
public/research/rl-scheduling/run-cse-81.sh --output /absolute/path/to/new-cse-run
```

The trigger uses `.venv/bin/python`; set `PYTHON=/path/to/python` to use another
existing environment with the repository's simulation dependencies. It does not
install anything, fetch data, sync results, start workers, or run numerical kernels.
It refuses to overwrite an output directory.

`cse-81.json` pins the local tiled-1024 graph checksum, cost-table checksum,
662 original tasks, and the expected 81 exact expression redundancies. Its corpus
path is resolved relative to the repository root, normally the workspace parent's
`calibration-audit/rl-scheduling/factorization-v1`. The existing local corpus is a
prerequisite, not a tracked dataset or something the script downloads. Missing
inputs or changed graph/cost checksums cause an error; edit the configuration only
when intentionally preparing a different experiment.

The experiment:

1. Checks the original graph and cost model, and repeats exact-expression analysis.
2. Allows only the expected `mul`/`broadcast` duplicates; retains a representative
   and redirects ordered input references, preserving repeated operand multiplicity.
3. Produces a separate 581-task **hypothetical** graph and a map from every original
   task ID to a surviving task ID. No dead-task pruning or arithmetic reassociation.
4. Checks that every original task's expression reconstructs from its representative
   after alias substitution, validates the new graph, and confirms source unchanged.
5. Simulates original and hypothetical graphs with FIFO/critical path under LAN,
   home-network, and heterogeneous-worker conditions. Each has one deterministic
   run and five noisy seeds: 72 rollouts total.
6. Writes compressed plans/rollouts, the alias certificate, pinned config, local
   cost model, runner source and report; bundles outputs into a ZIP alongside them.

The first local smoke run predicted deterministic critical-path improvements of
9.23% (LAN), 12.94% (home), and 9.73% (heterogeneous). These are model predictions,
not measured runtime improvements. Same numeric seeds do not align timing noise
by task after graph rewriting; the deterministic comparison is the cleaner
controlled comparison.

**Not a deployable optimized plan.** The snapshot lacks an explicit full output
contract. All original logical task values are preserved symbolically through the
alias map, but the live runtime does not consume this experiment's alias
certificate. Correct deployment needs output-name preservation, immutable inputs,
compatible precision/backend semantics and independently checked rewrite
preconditions. The run itself computes no tensors and modifies no
production/compiler path; the numerical equivalence of the rewrite is checked
separately, below.

## Numerical validation (2026-10-30)

`tests/test_cse_experiment.py::CSENumericalTests` executes the original 662-task
graph and the 581-task hypothetical graph with the real NumPy kernels
(`ygaraty.runtime.local.execute_plan`) on seeded random leaves and compares every
original value with its survivor under the alias map. All 662 values are
**bitwise identical** to their survivors (seed 0 in the test; seeds 0, 1 and 2
in the first run, 1,986 comparisons, worst absolute difference 0.0). The leaves
and the cross-update value `X` are the same in both graphs. The 81 removed
expressions are 64 `mul` tiles of the step-size broadcast by the residual
(`v22` duplicating `v21`), 8 `mul` tiles each for the regularization term on `U`
and on `V` (`v17`/`v16`, `v13`/`v12`) and one duplicated `broadcast` of the
regularization scalar (`v15`/`v11`); their consumers are `assemble`, `add` and
`mul`. `mul` and `broadcast` are deterministic, read-only kernels, so equality is
exact, not approximate.

The test also rebuilds the pinned graph: `ygaraty.simulation.factorization.build_plan()`
with its defaults reproduces `graph_sha256` `48c1b2ce…` and the bundled
`ygaraty/simulation/cost_model.json` reproduces `cost_table_sha256` `8c1a1ee8…`,
so a machine without the `calibration-audit` corpus can recreate it: write the
plan with `ygaraty simulate-factorization --plan-output …`, copy the cost table,
take the scenario settings from
`papers/RL-task-distribution/data/corpus-summary.json`, and point a copy of
`cse-81.json` at that directory. Done this way on 2026-10-30, the run reproduced
the deterministic critical-path gains quoted above exactly (9.23%, 12.94%,
9.73%).

What this does not show: the live runtime still does not consume the alias
certificate, so a deployed rewrite also needs the consumer-side alias resolution
and output-name preservation described above.

## Where the duplicates come from, and where else they occur (2026-10-30)

All 81 are products of the autodiff rule for `mul` (`ygaraty/compiler/tape.py`,
`acc(a, mul(g, b)); acc(b, mul(g, a))`): when both operands are the same value,
as in `residual * residual`, `U * U` and `V * V`, the two gradient terms are the
same expression, computed twice and then added. The duplicated `broadcast` is the
`sum` gradient of the two 1,024 × 32 penalties. Emitting the product once and
adding it to itself, which is what the CSE'd graph does, is bitwise identical
(doubling is exact), so the clean fix is in the tape's `mul` backward (or a
`square` primitive), not a runtime pass. Changing it would change the pinned
graph hash of this corpus, so it is left as a proposal here.

The same audit (`plan_audit.analyze` on the compiled task list) on the other
plans this repository runs, compiled as their jobs compile them:

| Plan | Tasks | Exact duplicates |
|---|---:|---:|
| Sharded 2D factorization over HTTP (experiments 46, 47; `build_sharded_plan`, 1,024 and 8,192) | 158 | 0 |
| Sized MLP of `http_profile` (64 × 1024 → 512 → 512, tiles 128 and 256) | 172 / 70 | 0 |
| `examples.mlp`, `large_mlp`, `memory_mlp` | 40 / 50 / 332 | 0 |
| Tiny Transformer translator (1M recipe, batch 64, band 16, tile 512, Adam) | 733 | 0 |
| 30.3M Transformer translator (experiment 30 sizes, batch 64, band 8) | 2,626 | 0 |
| LSTM translator (experiment 24 sizes, batch 64) | 2,241 | 4 (`transpose`) |
| Compiler reference factorization `tiled_1024` (this experiment) | 662 | 81 |

Only the compiler's reference factorization plan carries a material count, and
that plan is used by the simulator and the RL corpus, not by any deployed job
(the deployed factorization uses the hand-built sharded plan above). A measured
A/B of the rewrite on the real coordinator is therefore not pursued: the pass
has no deployed customer, and the 81 tasks disappear at their source with the
autodiff fix.
