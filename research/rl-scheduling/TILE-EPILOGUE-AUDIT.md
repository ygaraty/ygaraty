# Completed reduction → bias → ReLU before assembly

2026-10-30. Offline metadata transformation and discrete-event simulation, based
on `c8da641` plus the local grouping tools. **This is a promising modeled rewrite,
not a numerically validated runtime optimization.** No workloads, training jobs,
workers, services or network emulators were executed.

## Results against pruning alone

Three local MLP graphs contain one eligible hidden-layer pattern each. The
factorization graph has no activation pattern. Fused-factorization plans remain
blocked because their required-output contracts have not been recovered.

| Workload | Pruned → rewritten tasks | Packed tiles | Per-tile memory allowance | Deterministic gain, 0 ms wrapper | Deterministic gain, +2 ms/tile |
|---|---:|---:|---:|---:|---:|
| MLP-76 | 65 → 64 | 6 | 480 B | 9.13% | 8.88% |
| MLP-77 | 65 → 64 | 4 | 2,494,464 B | 15.51% | 15.31% |
| MLP-340 | 266 → 265 | 16 | 954,368 B | 9.58% | 9.39% |
| Tiled factorization | 582 → 582 | 0 | — | 0% | 0% |

The table uses the 8 MiB admission limit and averages per-case gains over three
scenarios and two policies. The 1 MiB limit **rejects MLP-77**; MLP-76 and MLP-340
fit and produce the same graph at both limits. The intermediate +0.5 ms/tile
setting also retains the positive mean gains (9.07%, 15.46%, 9.53%). All wrapper
settings additionally charge a packing-cost estimate; 0 ms does not mean free packing.

With +2 ms/tile, deterministic critical-path-only average gains are 8.83%, 15.37%
and 9.27% for MLP-76/77/340 respectively. All six deterministic scenario/policy
cases improve for each eligible workload, but MLP-340's LAN gain is only about
0.88%. Three noisy seeds give mean gains of 10.34%, 15.19% and 8.80%; MLP-340 has
an individual −0.68% regression. Identical seeds do not provide task-aligned noise
when the graph changes. These sensitivity results are not confidence intervals
or a promise of a runtime gain.

## What changes, and why both outputs survive

Original:

```text
ordered partial products → completed tile reductions → product assembly
  → whole bias addition → whole ReLU
                         └→ backward reads the preactivation
```

Hypothesis:

```text
each ordered tile reduction → tile bias addition → tile ReLU
                           └──────── pack [preactivation; activation]
packed top halves    → preactivation assembly under its original ID → backward
packed bottom halves → activation assembly under its original ID   → consumers
```

The original `hidden_product` assembly is removed only if its sole consumer is
the bias operation and it is not externally required. Both `v0` (preactivation)
and `hidden` (activation) remain available under their original names. All
backward tasks and their input references remain unchanged. Packing both values
prevents the invalid optimization of dropping the preactivation needed by ReLU's
backward operation.

Each packed task performs exactly the original `add_many` reduction, then bias
addition and ReLU. The reduction's input list is copied without sorting,
deduplication or reassociation. The existing kernel initializes an accumulator
with zeros and adds inputs in fixed list order; that behavior is part of the
hypothetical operation's contract. Bias and activation never operate on an
individual partial product. Repeated reduction operands remain repeated.

Bias is a row-vector leaf. A tile covering columns `[c0,c1)` reads the matching
bias slice `[0,1,c0,c1]`. Tiles must cover the full matrix exactly without overlap
or gaps. The packed shape is `[2*tile_rows, tile_columns]`, with preactivation in
the top half and activation in the bottom half. The two assemblies select those
regions into the original tile destinations.

The graph loses only one task overall: N reduction tasks become N packed tasks,
the product assembly disappears, and the two full-size arithmetic tasks become
two metadata assemblies. Worker arithmetic dispatches decrease by two. This
changes when elementwise work becomes ready and where intermediate values are
stored/transferred; task-count reduction alone does not explain the modeled gain.
The cost model charges per-tile elementwise calls rather than assuming free fusion.

## Admission and preservation checks

Pruning uses the existing recovered contracts: loss, parameter gradients and
parameters for the MLPs; loss, parameter updates and leaves for factorization.
Extra diagnostics must be declared before pruning. Required, reusable and
persistent reduction/product values block this rewrite; required preactivation
and activation values are supported because their original IDs remain.

The current implementation accepts only plain, unsliced `add_many` reductions
feeding an exact rectangular assembly, followed by empty-attribute `add_bias`
and `relu`. All partial inputs have the complete tile shape. Each reduction has
only the product assembly as a consumer. Unknown attributes, wire annotations,
owned/replicated/cache-affinity plans, overlapping rewrites, unexpected bias
shapes and generated-ID collisions are rejected. A failed admission leaves the
whole pattern unchanged, not partially transformed.

The simulated scenarios use float64 transport. The runner rejects float32-wire
or narrower activation-byte settings for this mode: removing a serialization
boundary can change rounding. The hypothesis does not establish equivalence for
bf16/float32-wire configurations or the owned Transformer runtime.

## Memory limits

For each packed task, admission counts full backing tensors for distinct partial
inputs and the bias, five tile-sized buffers (reduction, biased value, activation,
and the two-tile packed result), plus scratch equal to four times the reduction's
input-plus-output footprint. Tensor accounting uses eight bytes per element.
The packed output must also fit 8 MiB. Admission limits are 1 MiB and 8 MiB.

Matmul remains in the original producer tasks, so no BLAS workspace is added to
these groups. The amounts in the results table are modeled allowances, not
measured RSS. They exclude process overhead, caches, concurrently retained claim
inputs and baseline worker memory. They bound the proposed task's tensor
requirements under the stated model, not total worker memory. A real executor
must enforce its complete worker budget and preserve intermediate finite-value
checks as well as operation order.

## Cost assumptions and scope

`offline_tile_epilogue` is a synthetic simulator-only operation, not a registered
numerical kernel. Its `extra_work` charges the original reduction work, tiled bias
and ReLU work, packing, and a wrapper sensitivity of 0/0.5/2 ms per tile.
Original kernel intercepts are retained for every tiled call. Packing charges the
local generic `*` cost model at four tile elements per output element of one tile
(read two tiles and write two tiles); this is an uncalibrated packing estimate.
The wrapper sensitivity scales with the simulator's worker-speed model.

No runtime grouping protocol is assumed to implement this automatically. Existing
runtime grouping can return every intermediate; this hypothesis publishes only
the packed pair. Numerical execution elsewhere must verify the original and
rewritten loss, gradients and parameters, plus preactivation/activation and input
immutability, across representative inputs. Keeping reduction order is necessary
but timing simulation alone cannot prove floating-point equivalence.

## Experiment and verification

Four eligible graphs × seven variants (prune plus two memory limits × three
wrapper costs) × three scenarios (LAN-4, home-4, heterogeneous-8) × two policies
(FIFO, critical path) × four runs (noiseless seed 0; noisy 1700–1702) = **672 rollouts**.
Source plans and source cost hashes are checked. All raw timings and settings are
saved. No claim of fresh cost calibration is made.

Validation completed:

- 28 snapshots passed the standard dependency/shape/region validator.
- 24 rewrite certificates passed structural checks for exact tile partitions,
  unchanged reduction input lists, bias column slices, packed output regions,
  retained tensor names, unchanged outside tasks and memory limits.
- An artifact verifier independently checked all 672 unique rollout keys and
  recomputed all 48 gain summaries.
- Ten focused structural tests passed across grouping modes, including three
  new tests for repeated reduction operands, backward preservation, memory and
  output rejection, overlapping tiles, wire attributes and malformed bias shape.

These are structural checks; no numerical kernel test ran on this machine.

## Reproduce and inspect

From the repository root with the existing local corpora:

```bash
.venv/bin/python -m ygaraty.simulation.plan_grouping \
  --mode tile_epilogue \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-tile-epilogue-run
```

Use a new output directory. Hash-matching local legacy MLP source plans are needed
to recover their output contracts. No database or dataset URL is involved in this
metadata-only workflow. The runner blocks connections and numerical kernels.

Completed bundle: workspace-parent
`calibration-audit/rl-scheduling/tile-epilogue-v1.zip`. It includes compressed
plans/certificates and rollouts, reports, source/tests, cost tables, provenance,
checksums and verifier output. Verify from the repository root:

```bash
PYTHONPATH=. .venv/bin/python /absolute/path/to/tile-epilogue-v1/verify.py
```

The next step is a small opt-in numerical prototype on the authorized execution
machine, using the packed-pair contract above and the validation procedure in
[LOCAL-VALIDATION.md](LOCAL-VALIDATION.md), before any compiler/runtime integration.

For server testing and Transformer applicability checks, follow
[SERVER-VALIDATION.md](SERVER-VALIDATION.md). A copy of the compressed tile-epilogue
bundle is tracked under `artifacts/` so it is available after pulling this commit.
