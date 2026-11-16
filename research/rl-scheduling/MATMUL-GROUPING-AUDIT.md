# Matmul plus activation: offline audit

2026-10-30, based on `c8da641` plus the local grouping audit implementation.

**No eligible groups were found in the existing factorization/MLP corpora.**
Pruning plus matmul–activation grouping produces the same graph as pruning alone,
so the modeled gain is exactly zero, including every noisy rollout. This is a
negative result about these compiled graphs, not a general claim about epilogue
fusion or a numerical runtime benchmark.

## What was tested

The new `matmul_activation` mode of `ygaraty.simulation.plan_grouping` considers:

- `matmul → relu/sigmoid/tanh` (two operations).
- `matmul → add_bias → relu/sigmoid/tanh` (three operations).

It accepts only complete chains ending in an activation. It does not group a
matmul–bias prefix that never reaches an activation. There is no arithmetic
reassociation, partial-sum elimination, activation pushdown or tile-layout change.

| Workload | Tasks after pruning | Matmul tasks audited | Immediate consumers | Accepted groups | Modeled gain |
|---|---:|---:|---|---:|---:|
| Tiled factorization | 582 | 74 | 72 add; 2 assemble | 0 | 0% |
| MLP-76 | 65 | 30 | add_many | 0 | 0% |
| MLP-77 | 65 | 28 | add_many | 0 | 0% |
| MLP-340 | 266 | 176 | add_many | 0 | 0% |
| Fused-1024 / Fused-4096 | — | — | Missing required-output contract | Blocked | Not evaluated |

The 308 matmuls above are counted once per graph, not once per sweep setting.
Factorization has no activation to group. A representative MLP path is:

```text
partial matmul → add_many → assemble → add_bias → relu
```

For the first hidden-product tile, MLP-76's `add_many` has one task input,
MLP-77's has two and MLP-340's has four. The next whole-value assemblies have
6, 4 and 16 task dependencies respectively. Even the one-input reduction does
not make a direct chain: the assembly still joins multiple tiles. The audit
records every matmul's immediate consumers in each candidate certificate.

## Output and safety constraints

The audit uses the same hash-verified local plans and recovered output contracts
as [the pruning study](SIMPLE-PASSES.md). Factorization retains loss, updated
parameters and leaves; MLP retains loss, parameter gradients and parameters.
Extra caller diagnostics would need to be declared before pruning. Missing
contracts block both pruning/grouping comparison rather than guessing roots.

A hidden chain value must have exactly one consumer and must not be a required,
reusable or persistent value. Successors may depend on the preceding task and
leaves, but not another task; joins and fan-out stop grouping. The last output ID
survives. Only empty-attribute matmul, bias and activation operations are allowed;
wire annotations, ownership, replication, cache affinity, unknown attributes,
reductions and metadata operations are excluded. Original operand order, repeated
operands, regions and operation definitions are preserved symbolically.

Required outputs remain available in every candidate. The verifier expands each
group back into its original operations and compares the entire pruned graph.
Here all 24 candidate graphs also have exactly the same graph hash as their
pruning baseline because no group was accepted.

## Memory admission

The sweep tests 1, 8 and 32 MiB **per-group tensor-memory allowances**, with limits
of two or three operations. It charges full external source tensors (including
sources read through slices), all intermediate/output tensors retained until the
group finishes, and scratch equal to four times the largest node's total input
and output footprint. All tensor accounting uses eight bytes per element.

Matmul groups additionally reserve **1 MiB for BLAS scratch**. This is an explicit
model assumption, not a measured bound on a particular BLAS implementation.
A candidate exceeding the allowance is rejected; incomplete prefixes are discarded.
Positive structural fixtures demonstrate successful grouping under a larger budget
and rejection under 1 MiB. In the real corpus, graph structure blocks all groups
before memory becomes decisive.

These limits are not process-RSS limits: caches, simultaneously retained claim
inputs, Python/native allocator overhead and baseline worker memory are outside
this estimate. No production memory-cap claim is made. Before deploying any
future accepted group, determine a backend-specific scratch bound and account
for total worker memory. This study accepts no new corpus groups, so it introduces
no additional grouped tensor allocation in its modeled plans.

## Simulation and validation

Four eligible workloads × seven variants (prune plus 2 lengths × 3 budgets) ×
three scenarios (LAN-4, home-4, heterogeneous-8) × two policies (FIFO and critical
path) × four timing runs (deterministic seed 0; noisy seeds 1700–1702) gives
**672 rollouts**. All settings and per-case timings are saved. Identical graph
hashes mean these are repeated controls, not evidence from independent rewrites.

As in [the cheap-chain audit](GROUPING-AUDIT.md), hypothetical groups use a synthetic
`offline_group` operation with additive original kernel costs and zero additional
wrapper cost; no matmul acceleration is assumed. It models final-result-only
publication, which is not the existing runtime grouping protocol. No production
compiler, worker or scheduler is changed. With no groups accepted, these wrapper
assumptions have no effect on the measured simulation comparison.

Checks completed:

- 28 snapshots loaded through the normal graph/dependency validator.
- 24 candidate expansions verified against pruning baselines; roots and hashes preserved.
- 672 unique rollout keys and all 48 summary aggregates independently checked.
- Seven focused structural tests passed across the cheap and matmul modes. They
  cover successful direct/bias grouping, exact operation expansion and cost
  decomposition, protected outputs, memory rejection, incomplete chains, joins,
  fan-out, wire boundaries and repeated sliced operands.

No numerical workloads, training, services or network calls were executed.
Numerical equivalence is not established by these timing simulations. Successful
synthetic test fixtures are structural tests, not new performance benchmarks.

## Reproduction and artifacts

From the repository root, using existing local metadata corpora:

```bash
.venv/bin/python -m ygaraty.simulation.plan_grouping \
  --mode matmul_activation \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-matmul-grouping-run
```

The output directory must be new. Original MLP source files must be available
locally and match their recorded hashes for output-contract recovery. This path
uses no database and needs no dataset URL. Connections and workload kernels are
blocked while the study runs. Source plans are checked unchanged.

The completed local bundle is workspace-parent
`calibration-audit/rl-scheduling/matmul-grouping-v1.zip`. It contains compressed
plans/certificates, compressed raw timings/settings, reports, source and tests,
cost tables, provenance hashes, checksums and a structural verifier. To verify:

```bash
PYTHONPATH=. .venv/bin/python /absolute/path/to/matmul-grouping-v1/verify.py
```

## Next useful question

Audit **completed tile reduction → bias → activation** before whole-matrix
assembly. This requires a different compiler transformation, not simply broader
chain matching. It must retain partial-product reduction order, apply activation
only after the complete sum, preserve any preactivation values needed by backward
operations, and bound tile memory. Activation on each partial product would
change the result. First establish whether the assembly can be delayed without
breaking downstream consumers; only then evaluate that rewrite in simulations.

Transformer/owned plans remain outside this corpus comparison. No speedup or
numerical-equivalence claim is made for them.

The subsequent [tile-epilogue audit](TILE-EPILOGUE-AUDIT.md) implements a simulation-only
rewrite retaining both preactivation and activation after each completed reduction.
