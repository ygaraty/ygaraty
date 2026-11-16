# Handoff: numerical and performance validation on another machine

## Scope and current evidence

Run this procedure on a machine authorized to execute numerical workloads. The
current simulation-only workspace must not run these numerical tests or benchmarks.
Use local files and an already provisioned Python environment; no services,
coordinators, workers, remote datasets, or training jobs are needed for this test.

The target is the hypothetical plans in `simple-passes-v2`: output-root pruning,
single-input assembly bypass, double-transpose cancellation, and their combination.
The production compiler and scheduler have not been changed. Structural checks and
720 discrete-event rollouts passed; **numerical equivalence and actual performance
of these new passes have not been established**.

The existing 81-expression CSE test covers a separate transformation. Its success
does not validate these new passes. CSE is not included in their timing comparison.

## Files to bring

Use the repository commit containing this guide (based on `52f9cbd`), including:

- `ygaraty/simulation/plan_simplify.py`
- `tests/test_plan_simplify.py`
- `public/research/rl-scheduling/SIMPLE-PASSES.md`
- this `LOCAL-VALIDATION.md`

Record the checked-out commit and hashes of these files before testing. If
transferring a modified checkout instead, include untracked files explicitly;
`git diff` alone omits them.

Bring the workspace-parent archive
`calibration-audit/rl-scheduling/simple-passes-v2.zip` (about 134 KB), preserving
its original copy. Use `v2`, not the incomplete `v1` run. Extract it locally.
It includes 30 compressed graph/certificate envelopes, the report, compressed
rollouts, and structural verification artifacts. It does **not** contain real
numerical input arrays or a numerical/performance runner for the new passes.

Use an existing environment with this repository's dependencies, including NumPy.
Record Python, NumPy, BLAS, CPU, OS, thread settings and available RAM. Do not infer
input data from shape metadata alone: provide local valid fixtures and record
how constants, labels, parameters and inputs were initialized.

Re-running the simulator is optional. It additionally needs the original local
corpora, cost tables and hash-matching legacy MLP source plans; see
[SIMPLE-PASSES.md](SIMPLE-PASSES.md). Absolute source paths may need relocation
while preserving the source content hash. Missing inputs must be reported rather
than fetched. Existing candidate envelopes can be tested without regenerating them.

## Required outputs: agree before executing

The certificate's `roots` is the current, conditional output contract:

| Graph family | Outputs retained by this study | Limitation |
|---|---|---|
| Tiled factorization | Loss, updated parameters, original leaves | Add any externally consumed gradients or diagnostics before pruning |
| MLP | Loss, parameter gradients, parameters | This does not validate an optimizer update or a multi-step training loop |
| Fused factorization | No recovered contract | Pruning is blocked; do not invent an output contract |

Confirm this contract against the real caller. If an additional output is needed,
regenerate candidates with that output as a root. Do not silently omit it from
comparison. An unknown contract is **blocked**, not a numerical pass.

Pruned intermediates intentionally disappear. Compare every required output under
its original name, resolving that name through the candidate's `aliases` map.
The 81-expression CSE test instead reconstructs every original intermediate;
that stronger expectation does not apply to dead-task elimination.

## Numerical test to implement and run on the execution machine

There is no ready-made numerical runner for `simple-passes-v2` yet. Build the
following small harness using the existing APIs, keeping its source with results:

1. Read each `<workload>-<variant>.json.gz` with `gzip` and `json`. Each file is
   an envelope with `plan` and `certificate`; write **only `plan`** to a temporary
   JSON file and pass it to `ygaraty.simulation.reference.load_plan(path)`.
   Verify its graph hash and pair it with the same workload's `original` envelope.
2. Use the same required roots for original and candidate. Resolve aliases
   transitively with cycle detection. Require every resolved root to exist.
   Check leaf names/shapes match and reject unsupported runtime semantics.
3. Topologically order tasks using task dependencies. The `ordered` helper in
   `tests/test_cse_experiment.py` is an example. Never assume serialization order
   is executable, and never change operand order or repeated operands.
4. Generate each fixture once and give each execution independent copies of the
   same leaf arrays. Preserve semantic constants and valid labels. Record seed,
   dtype and an input-content hash. Check input arrays remain unchanged afterward.
5. Call `ygaraty.runtime.local.execute_plan(tasks, inputs, keep)`. For the original,
   `keep` contains original required roots; for the candidate it contains resolved
   roots. The executor borrows inputs and releases intermediates not needed later.
   Reconstruct the candidate output dictionary under original names.
6. Compare **every** root: shape, dtype, finite-value status and element values.
   Save per-root mismatch counts, maximum absolute error and maximum relative
   error (define the relative-error denominator). Fail on missing outputs,
   unexpected nonfinite values or input mutation. Loss-only comparisons are insufficient.

Start with seeds 0–9 and valid zero, signed, small-magnitude and larger finite
fixtures where the operation domains allow them. Test every dtype used in the
intended deployment separately. Include contiguous and supported noncontiguous
inputs to expose layout/copy assumptions. First test the exact archived shapes;
smaller fixtures are useful diagnostics but do not replace them.

**Expected numerical result:** pruning changes no retained arithmetic, and the
metadata rules are intended to preserve values. Require exact element equality
first on the same deterministic backend and dtype, with matching shape and dtype.
If claiming *bitwise* identity, compare contiguous logical-element bytes as well;
`np.array_equal` alone does not distinguish signed zero and is not a bytewise test.

Layout changes can affect downstream kernel choices and floating-point rounding.
If exact equality fails, report the failure before considering approximate
acceptance. Any tolerance must be specified per dtype and output in advance of
an approximate-validation run, justified by the application's error budget, and
reported as approximate equivalence. Do not enlarge tolerances until a failing
case passes. Finite tests establish evidence for the tested cases, not a proof
for all inputs or a guarantee of an unchanged training trajectory.

Separately, the existing pinned CSE regression can be run on this other machine:

```bash
# From the repository root, using the provisioned environment:
.venv/bin/python -m unittest discover -s tests -p 'test_cse_experiment.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_plan_simplify.py' -v
```

The first command executes real kernels, including the 662→581 CSE comparison
with seed 0. The second checks structure only. Neither replaces the new harness.

## Performance test after numerical checks pass

Benchmark original, prune, assembly, transpose and combined separately on identical
fixtures. Do not combine CSE with these passes in the first experiment.

- Fix backend, thread counts, input dtype, process configuration and hardware power
  settings. Set BLAS thread limits before importing NumPy; record their actual values.
- Warm up each variant at least five times. Run at least 30 paired trials, alternating
  or randomizing original/candidate order with a recorded seed. Use identical inputs
  within each pair and avoid competing workloads.
- Time execution plus required alias/output reconstruction with a monotonic clock.
  Load graphs, initialize inputs, copy fixtures and validate results outside that
  interval for both variants. Keep only the agreed outputs in both runs, not all
  baseline intermediates. Validate outputs again outside timing.
- Report median elapsed time, dispersion, paired percentage gain
  `100 * (1 - candidate_seconds / original_seconds)`, and a 95% bootstrap confidence
  interval over paired gains. Report each workload/variant/dtype separately.
- Measure peak memory separately in fresh processes with identical measurement
  methods. Include NumPy/native allocations; Python-only allocation tracking is
  insufficient. Report rewrite/compilation time separately and the number of plan
  reuses needed to recover it. Do not claim end-to-end benefit from execution time alone.

Proposed acceptance rule, to fix before collecting results: numerical checks pass,
median paired gain is at least 2%, and the lower 95% confidence bound on median
paired gain exceeds zero. Otherwise label the result neutral, inconclusive or a
regression as appropriate. Report memory regressions even if timing improves.
The 2% threshold is a proposed practical floor, not a prediction from the simulator.

`execute_plan` is a sequential local kernel executor: it does not implement FIFO
or critical-path worker placement, transfers, network contention or the GNN.
These measurements establish local execution cost only. They cannot validate a
distributed scheduling speedup. Keep real local timings and simulated makespans
in separate tables; validating deployed performance requires a later authorized
experiment in that actual runtime, including compatible alias handling.

## Expected outcomes, not guarantees

| Workload | Original → combined tasks | Modeled combined makespan gain |
|---|---:|---:|
| Tiled factorization | 662 → 578 | 5.66% |
| MLP-76 | 76 → 65 | 7.12% |
| MLP-77 | 77 → 64 | 9.52% |
| MLP-340 | 340 → 266 | 17.74% |
| Fused-1024 / Fused-4096 | 98 → 98 / 358 → 358 | 0%; pruning blocked |

These are deterministic simulator averages across three scenarios and two
policies, not expected local benchmark percentages. Pruning is the strongest
candidate. Assembly-only was 0.43% worse on tiled factorization in simulation;
double-transpose-only showed no modeled improvement. Fewer tasks do not guarantee
a speedup. Do not add these gains to the separate CSE gains.

## Result bundle and decision

Save a compressed bundle containing the harness, source revision/file hashes,
original and candidate graph hashes, exact root/alias contracts, fixture recipes
and hashes, environment details, per-root numerical checks, raw paired timings,
confidence-interval method/seed, memory results and a short decision per variant:

- **Blocked:** missing input/contract, unsupported operation or alias semantics.
- **Numerically failed:** any required comparison fails; do not recommend the pass.
- **Numerically passed, performance inconclusive/regressed:** keep it experimental.
- **Numerically passed, local performance improved:** candidate for integration;
  scope the claim to the tested inputs, backend and local executor.

Compress large fixtures and logs; do not include credentials. Preserve the
unchanged original plans as controls. No production change is justified by
simulated timing alone or by passing the separate CSE test.
