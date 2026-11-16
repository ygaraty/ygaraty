# Local validation results: simple passes on the tiled factorization and the legacy MLPs

Executed 2026-10-30 on the workstation (AMD Ryzen 9 PRO 6950H, Linux, Python 3.14,
NumPy from `.venv`, OpenBLAS, one BLAS thread), following
[LOCAL-VALIDATION.md](LOCAL-VALIDATION.md). Repository at `c3b0b30`. Bundle:
`runs/simple-passes-local/` and `runs/simple-passes-local.zip` (local, not in git):
`harness.py`, `results.json` (environment, graph hashes, root and alias contracts,
fixture recipes and input hashes, every per-root comparison, raw paired timings,
memory samples), `source-plan.json`, and the simulator's own run under `simulation/`.

## Scope

| Workload | Status | Why |
|---|---|---|
| Tiled factorization (`tiled_1024`) | **numerically passed; prune and combined improve local time and memory** | below |
| MLP-76, MLP-77, MLP-340 | **numerically passed; prune and combined improve local time** | [MLP section](#mlp-76-mlp-77-and-mlp-340) below; unblocked once the legacy plans were recovered from the `runs/` backup |
| Fused-1024, Fused-4096 | blocked by design | no output contract, as in the study |

The `simple-passes-v2.zip` archive was not available, so the tiled factorization
candidates were regenerated here from the pinned graph: `simplify()` rebuilt the
graph (`48c1b2ce…`) and its output contract from the repository, and
`plan_simplify --corpus` on a locally rebuilt corpus (see CSE-81.md, "Numerical
validation") reproduced the study's modeled gains exactly (combined 5.66%
deterministic, 6.09% noisy; prune 80 tasks, assembly 6 aliases, transpose 1,
combined 578 tasks). Variant graph hashes are in `results.json`.

## Output contract

Roots as recovered by `output_contract`: `loss`, `updated/U`, `updated/V`, and the
leaves `X`, `U`, `V`, `v9` (a 1 × 1 constant). The compared outputs are the three
task-valued roots, each resolved through the candidate's alias map; the leaves are
checked unchanged after every execution. No other value is consumed by the shape-only
factory, so this contract is complete for that caller; a job that also read
diagnostics would have to add them before pruning.

## Numerical result: passed, exact

Five fixture classes × seeds 0 to 9 × four candidates = 200 comparisons of the three
roots (600 arrays). Fixtures: standard normal float64 leaves; `U`, `V` zero with `X`
normal; values scaled by 1e-3; values scaled by 1e3; and Fortran-order
(noncontiguous) copies of the normal leaves. Each execution got its own copies; the
inputs' SHA-256 was the same before and after every run.

Every root matched **byte for byte** (`tobytes()` of the contiguous array, so signed
zeros included), with equal shape and dtype and all values finite. Mismatch count 0,
maximum absolute and relative error 0.0 in all 200 comparisons. No tolerance was
needed or used.

The test `tests/test_plan_simplify.py::SimplifyNumericalTests` repeats the seed-0
normal fixture for all four variants in about a second.

## What the passes remove

- **Prune (80 tasks):** 64 `scale` tiles and their `assemble` that form `v24`, the
  gradient with respect to `X`. `X` is not trainable, so the autodiff computes a
  1,024 × 1,024 gradient nobody reads. The other 15 are whole-matrix `assemble`
  tasks of named intermediates (`prediction`, `residual`, `data_loss`, `v1`, …) that
  the tiled compiler emits for every named value and nothing consumes.
- **Assembly (6 aliases):** single-block assemblies of 1 × 1 values (`loss`,
  `data_loss`, `v7`, `v8`, `v10`, `v19`) replaced by their only input.
- **Transpose (1 alias):** `v25`, the transpose of the transpose of `V`, aliased to `V`.
- **Combined:** 578 tasks, 7 aliases, 77 pruned.

## Local performance: prune and combined pass the proposed rule; the metadata passes do not

Sequential `execute_plan` on one core, same fixture copies for every variant in a
trial, 5 warm-ups, 30 trials with the five variants in shuffled order (seed 2026),
timing execution plus alias resolution only. 95% bootstrap confidence interval of
the median paired gain, 10,000 resamples.

| Variant | Median | Median paired gain | 95% CI | Rule (≥ 2%, lower bound > 0) |
|---|---:|---:|---|---|
| original | 31.9 ms | — | — | — |
| prune | 20.5 ms | **35.5%** | [34.8, 36.7] | passes |
| assembly | 31.0 ms | 1.9% | [−0.4, 3.6] | fails: inconclusive |
| transpose | 31.5 ms | −0.5% | [−1.1, 3.0] | fails: neutral |
| combined | 20.6 ms | **35.5%** | [34.8, 35.8] | passes |

An earlier run of the same harness, minutes before, gave prune 29.5% [27.8, 32.6]
and combined 31.2% [30.2, 32.0] with the same qualitative picture; other sessions'
load on the workstation moves the absolute numbers. The local gain is far above
the simulator's 5 to 6% because `execute_plan` has no per-task overhead other than
the kernel: the 64 pruned `scale` tiles and the 16 assembles are a third of the
sequential arithmetic and copying. The simulator's figure includes coordinator,
claim and transfer time that pruning does not remove. Neither number is a
distributed speedup.

**Peak memory**, fresh process per variant, resident set sampled every 0.2 ms during
20 executions, above the RSS at the start of execution: original 35.7 MiB, assembly
36.3, transpose 36.3, **prune 28.4, combined 28.4** (20% less). The dropped
1,024 × 1,024 gradient is 8 MiB. Rewrite time: `simplify()` on the 662-task graph
takes under a second and is paid once per plan.

## Decision

- **Prune:** numerically passed, local performance improved. Candidate for
  integration, scoped to this graph, float64, the sequential executor and the
  recovered contract. The cheapest form is not a pass at all: stop the autodiff from
  emitting gradients of non-parameter inputs (`X`), which is what 65 of the 80 tasks
  are.
- **Assembly, transpose:** numerically passed, performance inconclusive or neutral.
  Keep experimental.
- **Combined:** passed and improved, but its gain is prune's.
- **MLP:** see the MLP section's decision below.
- **Fused:** blocked by design, as above.

Validating a deployed effect still needs the real coordinator and workers with
alias-aware consumers; `execute_plan` measures local kernel cost only.

## MLP-76, MLP-77 and MLP-340

Executed 2026-10-30 on the same workstation (Python 3.14.7, NumPy 2.5.4, scipy-openblas,
one BLAS thread, 28 GiB RAM), repository at `3821125` with uncommitted changes from other
sessions that do not touch the files below. File SHA-256 before testing: `plan_simplify.py`
`99b38b36…`, `SIMPLE-PASSES.md` `dce87968…`, `LOCAL-VALIDATION.md` `af2b30fa…`. Bundle:
`runs/simple-passes-local-mlp/` and `runs/simple-passes-local-mlp.zip` (local, not in git):
`harness.py`, one `results.json` per workload and run (about 270 KB each), the regenerated
candidates and simulator report under `simulation/`, and the logs.

### Provenance

The legacy MLP plans came from the 2026-10-29 `runs/` backup in Spaces
(`runs/spaces/backups/runs-20261029/`), as described in [LOCAL-1.md](LOCAL-1.md), "Deviations":
`optimized-comparison/baseline` (76 tasks) and `large-benchmark-20261021/tile-128-repeat-1`
and `tile-64-repeat-0` (77 and 340). Every copy of each size in the backup imports to one
graph: `4f3ef99c…`, `af2bd85a…`, `e796d945…`. These are the graphs the other workspace used:
gnn-v1 retrained on corpora built from them reproduced the published MLP-family column
exactly. The snapshots record their legacy source file and its SHA-256, and
`output_contract` re-checks that hash before reading the contract.

The harness also compared each snapshot with its legacy source: the task lists are equal
apart from the stripped `attempts` and `committed` fields, the leaf shapes are equal, and
every value in the source is float64. The legacy files hold shapes and dtypes only, not
array contents, so the fixtures below are constructed, not recovered.

`simple-passes-v2.zip` is still not here. As for the tiled factorization, the candidates
were regenerated: `plan_simplify --corpus runs/local-1/factorization-v1 --corpus
runs/local-1/mlp-validation-v1 --output runs/simple-passes-local-mlp/simulation`. They match
[SIMPLE-PASSES.md](SIMPLE-PASSES.md) wherever it records a figure: task counts, pruned and
aliased counts, 720 rollouts and every modeled gain to two decimals (below). The study does
not record candidate graph hashes, so those could not be compared; they are listed here and
in `results.json`. The regenerated tiled_1024 envelopes are byte-identical in content to the
earlier `runs/simple-passes-local/simulation/` ones. The harness reads the regenerated
envelopes, checks each against a fresh `simplify()` and its certificate roots against the
contract, and loads only `plan` through `load_plan`, which verifies the graph hash.

| Workload | Variant | Tasks | Pruned | Aliases | Graph SHA-256 | Modeled gain, det. / noisy |
|---|---|---:|---:|---:|---|---:|
| MLP-76 | prune, combined | 65 | 11 | 0 | `8a27d171ed27…` | 7.12% / 6.85% |
| MLP-76 | assembly, transpose | 76 | 0 | 0 | unchanged (`4f3ef99c…`) | 0 / 0 |
| MLP-77 | prune | 65 | 12 | 0 | `f3eff174359f…` | 9.52% / 9.39% |
| MLP-77 | assembly | 76 | 0 | 1 | `590cb48e399e…` | 0 / 0 |
| MLP-77 | transpose | 77 | 0 | 0 | unchanged (`af2bd85a…`) | 0 / 0 |
| MLP-77 | combined | 64 | 12 | 1 | `e5186edfddf8…` | 9.52% / 9.39% |
| MLP-340 | prune, combined | 266 | 74 | 0 | `4c2f438d470b…` | 17.74% / 17.85% |
| MLP-340 | assembly, transpose | 340 | 0 | 0 | unchanged (`e796d945…`) | 0 / 0 |

### Output contract and what the passes remove

Roots from the hash-verified legacy source: `loss`, the parameter gradients `v13`, `v9`,
`v7`, `v3` (W1, b1, W2, b2) and the parameters `W1`, `b1`, `W2`, `b2`. The compared outputs
are the five task-valued roots; the parameters are leaves and are checked unchanged after
every execution. This does not cover an optimizer update: the legacy plans contain none.

- **Prune** removes, in all three graphs, the gradient with respect to the input `x`:
  `v10` (the transpose of `W1`) and the tiles of `v11 = dscores_hidden · W1ᵀ` with their
  `add_many` and `assemble` tasks (10, 11 and 73 tasks). `x` is data, not a parameter, so
  nothing reads it. This is the same waste as the tiled factorization's `X` gradient, and
  the same one experiment 54 now stops at plan time (`input_gradients=False`); the legacy
  plans predate that.
- **Assembly** on MLP-77: `output_product`, a single-block assembly of its only tile,
  aliased to `output_product/block/0/0`. No other graph has one.
- **Transpose:** no double transposes in any MLP graph.

For variants whose graph is unchanged (marked above), the timing comparison is an A/A
control: the same graph measured twice under the same protocol. It shows the noise floor.

### Fixtures

Float64 leaves from `numpy.random.default_rng(seed)`, drawn in sorted leaf-name order.
`x` (6 × 2 for MLP-76, 128 × 256 for the others) and the parameters are standard normal.
`v1`, the 1 × 1 upstream gradient of the loss that `ce_backward` multiplies by, is the
constant 1.0 in every fixture. The labels are the integer targets stored in the
`cross_entropy`/`ce_backward` attributes of the legacy plans, used unchanged and all valid
class indices (2 classes for MLP-76, 64 for the others). Fixture classes: normal; zeros
(parameters zero, `x` normal, so every ReLU input is exactly 0); small (`x` and parameters ×
1e-3); large (× 1e3); and Fortran-order copies of the normal leaves. Seeds 0 to 9. The
benchmark uses the normal fixture with seed 12345.

### Numerical result: passed, exact

Five fixture classes × 10 seeds × four candidates = 200 comparisons per workload, five
roots each (3,000 arrays in total). Every root matched **byte for byte** with equal shape
and dtype and all values finite: mismatch count 0, maximum absolute and relative error
(denominator `|original|`, floored at 1e-300) 0.0. Input SHA-256 was unchanged after
every run. No tolerance was needed or used. The same check passed again in each of the two
benchmark repeats below (600 more comparisons per workload).

`tests/test_plan_simplify.py::SimplifyNumericalTests::test_legacy_mlp_roots_are_bytewise_identical`
repeats the seed-0 normal fixture for prune, assembly and combined on all three graphs and
pins the pruned and alias counts. It reads the snapshots from `runs/local-1/plans/` and
skips where they are absent.

### Local performance

Sequential `execute_plan` on one core, same protocol as above except **100 paired trials**
instead of 30, fixed before collecting because these graphs run in 1 to 25 ms. Five
warm-ups, five variants in shuffled order per trial (seed 2026), execution plus alias
resolution timed, 95% bootstrap interval of the median paired gain (10,000 resamples).

The workstation was not quiet. DiLoCo worker processes from other sessions kept the load
average at about 10 during the first run and 17 to 23 during the two repeats, on 16 logical
CPUs; it never fell below 3 in a 30-minute wait. Absolute times therefore drift by up to
5× between runs (MLP-340 original: 25 ms, 121 ms, 132 ms), and the interquartile ranges are
wide. The paired gains are more stable. Run 1 is the primary result; runs 2 and 3 repeat it.

| Workload | Variant | Run 1 median (IQR) | Run 1 gain [95% CI] | Run 2 gain [95% CI] | Run 3 gain [95% CI] |
|---|---|---:|---:|---:|---:|
| MLP-76 | original | 1.86 ms (1.60–2.12) | — | — | — |
| MLP-76 | prune | 1.68 ms (1.44–1.95) | **7.4%** [5.4, 13.7] | 9.8% [−2.9, 14.2] | 10.9% [9.2, 13.2] |
| MLP-76 | combined | 1.72 ms (1.49–1.94) | **7.9%** [3.5, 11.3] | 12.9% [7.5, 20.4] | 9.9% [8.1, 11.8] |
| MLP-76 | assembly (A/A) | 1.88 ms (1.62–2.15) | −0.4% [−3.4, 4.4] | −8.0% [−17.6, 2.9] | −3.0% [−6.0, −1.5] |
| MLP-76 | transpose (A/A) | 1.95 ms (1.68–2.14) | −4.3% [−9.5, 1.2] | 0.4% [−10.5, 7.0] | −3.3% [−7.3, −2.5] |
| MLP-77 | original | 19.1 ms (12.6–23.5) | — | — | — |
| MLP-77 | prune | 15.3 ms (8.5–18.5) | **19.7%** [15.6, 22.9] | 21.0% [17.4, 26.6] | 20.8% [13.1, 28.2] |
| MLP-77 | combined | 16.4 ms (9.1–19.3) | **15.3%** [12.4, 18.4] | 19.3% [14.0, 25.4] | 20.8% [16.0, 26.5] |
| MLP-77 | assembly | 19.6 ms (13.8–23.4) | −1.7% [−5.2, 4.5] | 2.8% [−3.1, 9.6] | 5.7% [1.3, 9.0] |
| MLP-77 | transpose (A/A) | 19.9 ms (11.4–24.0) | −3.6% [−6.7, 0.9] | −0.6% [−5.0, 8.1] | 2.0% [−6.1, 6.9] |
| MLP-340 | original | 25.4 ms (12.4–31.0) | — | — | — |
| MLP-340 | prune | 19.1 ms (10.5–22.9) | **22.1%** [16.3, 24.4] | 16.7% [12.4, 26.7] | 21.3% [17.1, 26.5] |
| MLP-340 | combined | 19.0 ms (9.3–23.1) | **21.6%** [18.1, 27.0] | 22.0% [13.0, 28.0] | 18.1% [12.2, 25.2] |
| MLP-340 | assembly (A/A) | 24.6 ms (11.3–28.3) | −0.9% [−4.7, 3.5] | −6.0% [−14.4, 3.2] | −1.1% [−4.8, 8.3] |
| MLP-340 | transpose (A/A) | 24.9 ms (11.1–28.6) | 1.3% [−5.4, 5.9] | 1.4% [−5.9, 7.8] | 0.7% [−5.1, 8.5] |

How to read it:

- **The A/A rows set the noise floor.** Identical graphs differ by up to 8% in median paired
  gain, and twice (MLP-76, run 3) the interval excludes zero at about −3%. Under this load
  the bootstrap interval understates the run-to-run spread, so a single run's "lower bound
  above zero" is weaker evidence than the handoff assumes.
- **Prune and combined on MLP-77 and MLP-340** gain 15 to 22% in every run, about twice
  the largest A/A offset (8%) and well above the typical one. The pruned `x` gradient, a
  128 × 512 by 512 × 256 product, is 16.8M of the graphs' 63M matmul multiply-adds (27%),
  plus its tile copies, which fits the size of the gain. Prune and combined differ only by MLP-77's one alias; their
  difference (19.7% against 15.3% in run 1, equal in run 3) is within the noise.
- **Prune and combined on MLP-76** gain 7 to 13% and pass the rule in five of six run ×
  variant cases (run 2 prune: lower bound −2.9%). The graph is 6 × 2 inputs; the time is
  per-task Python overhead in `execute_plan`, so removing 11 of 76 tasks saving about a
  tenth of the time is what one would expect. The margin over the A/A offsets is smaller
  than for the larger graphs.
- **Assembly on MLP-77** (the only real metadata candidate) is −1.7%, 2.8% and 5.7% across
  runs; only run 3 clears the rule, and that is within the A/A range. Inconclusive.
- **The simulator's figures** (7.1%, 9.5%, 17.7%) and these local ones measure different
  things, as for the tiled factorization; neither is a distributed speedup.

**Peak memory**, same method as above. MLP-77 and MLP-340: about 5.9 MiB and 5.7 MiB above
the start RSS for every variant, equal within 8 KiB. The dropped `x` gradient is 256 KiB
and does not show at this resolution, so no memory change was resolved, in either
direction. MLP-76 runs too briefly for the 0.2 ms sampler (one to three samples per
process); no memory result for it.

**Rewrite cost**, measured under the same load: `output_contract` plus `simplify()` with all
passes takes 35, 83 and 198 ms for MLP-76, -77 and -340. At run 1's median savings that is
recovered after about 215, 27 and 29 executions of the plan.

### Decision for the MLP graphs

- **Prune:** numerically passed on all three graphs (exact, 200 comparisons each). Local
  performance improved on MLP-77 and MLP-340 in every run (15 to 22%), and on MLP-76 in
  run 1 (7.4% [5.4, 13.7]) with one of the two repeats missing the lower bound. Candidate
  for integration, scoped to these graphs, float64, the sequential executor and the
  loss/gradient/parameter contract. As for the tiled factorization, the pruned work is the
  gradient of a non-parameter input, which `input_gradients=False` (experiment 54) already
  avoids at plan time for new plans.
- **Combined:** passed and improved; its gain is prune's.
- **Assembly** (MLP-77 only): numerically passed, performance inconclusive. Keep experimental.
- **Transpose, and assembly on MLP-76 and MLP-340:** no candidate rewrites; nothing to decide.
- No memory change was resolved, and the benchmarks ran on a loaded machine. A repeat on an
  idle machine would tighten the MLP-76 result.
