# Bounded grouping after pruning: offline audit

2026-10-30, based on `c8da641`. This experiment executes no numerical workload,
training, services or network requests. It compares the same pruned graph with
and without conservative grouping; it does not change the production compiler.

## Result

**Cheap single-consumer chains offer almost no additional benefit on the tested
local graphs.** One scalar chain qualifies in factorization; none qualify in the
three MLP graphs. Increasing the group length or tensor-memory budget changes
nothing. These results do not rule out larger kernels, joins or another corpus.

| Workload | Pruned tasks | Pruned + grouped tasks | Groups | Mean deterministic gain over pruning |
|---|---:|---:|---:|---:|
| Tiled factorization | 582 | 581 | 1 | 0.31% |
| MLP-76 | 65 | 65 | 0 | 0% |
| MLP-77 | 65 | 65 | 0 | 0% |
| MLP-340 | 266 | 266 | 0 | 0% |
| Fused-1024 / Fused-4096 | — | — | — | Blocked: missing output contract |

The sole group preserves `v7/block/0/0 → v8/block/0/0`, retaining the final ID.
It eliminates one scalar intermediate (8 bytes at float64) and one schedulable
task. The conservative tensor-memory allowance is 128 bytes. The deterministic
gain ranges from 0 to 1.20% across scenario/policy cases. With three noisy seeds,
the mean is 0.40%, with individual gains from −2.01% to +5.30%. These small noisy
changes are not convincing performance evidence. Different task counts change
random-number consumption; identical seeds are not task-aligned noise controls.

## Safety and scope

The audit starts from local checksum-validated versioned plans and the output
contracts recovered by `plan_simplify.output_contract`. Pruning runs first with
those roots. Factorization retains loss, parameter updates and leaves; MLP retains
loss, parameter gradients and parameters. Any additional externally consumed
outputs must be added before pruning. Missing contracts block the whole comparison.

Grouping permits only `add`, `mul`, `scale`, `relu`, `relu_backward`, `sigmoid`,
`sigmoid_backward`, `tanh`, `tanh_backward` and `add_bias`, with reviewed attribute
keys. It does not include matmul, reductions, metadata operations, nested groups,
wire annotations, owned/stateful operations, replication or cache-affinity plans.
In particular, coordinator `assemble`/`transpose` fast paths must not become worker
kernels merely because grouping appears to reduce node count.

A chain can extend only when its current last value has exactly one consumer,
is not a required/reusable/persistent value, and the next task has exactly that
one task dependency. Additional leaf inputs are allowed. Fan-out and joins stop
the chain. The final output ID is unchanged. Original operations, operand order,
repeated operands, attributes and input regions remain intact inside the group.

Every candidate was expanded symbolically and compared with the entire pruned
baseline. All 16 grouped variants reconstruct exactly the original task definitions.
This is a structural equivalence check, not numerical execution or a universal
proof about backend precision and aliasing.

## Memory limits

The sweep admits groups of at most 2 or 4 operations under **1 MiB or 8 MiB**
tensor-memory allowances. Admission charges:

- Full external source tensors, including sources accessed through slices.
- Every internal output, conservatively retained until the group completes.
- Scratch allowance equal to four times the largest constituent operation's
  input-plus-output footprint, at eight bytes per element.

A proposed extension exceeding the budget is rejected. All accepted groups fit;
none of the corpus candidates were rejected for memory. Raising the budget cannot
help when the blockers are graph structure or unsupported operations.

This limits incremental group tensor requirements, **not process RSS**. Python,
allocator overhead, worker caches, concurrently retained claim inputs and the
baseline plan's memory use are outside this allowance. No worker-wide memory-cap
or measured peak-memory claim is made. Runtime integration still needs admission
against its actual worker budget. The narrow elementwise allowlist avoids unknown
BLAS/large-kernel scratch requirements.

## Simulator assumptions and runtime gap

A synthetic `offline_group` operation has zero base cost and an `extra_work` list
containing each original operation's work estimate. The existing simulator adds
all constituent kernel costs, including their intercepts; it assumes no arithmetic
speedup. Critical-path ranks use the same decomposed costs. Cost coverage and
source cost hashes are saved in the report; the group wrapper has no measured
calibration and its zero extra overhead is optimistic.

The hypothesis posts **only the final group result**. This is not equivalent to
turning on the existing runtime `group()` unchanged: `execute_task()` returns all
intermediates for auditing, and the ZMQ worker sends that result list. These
hypothetical snapshots are explicitly labeled `offline_group` and are not runnable
production plans. The HTTP translation path does not acquire grouping support
from this experiment. Any integration must define output publication semantics,
account for wrapper overhead, and undergo numerical validation elsewhere.

Wire precision comes from the original scenario settings. Groups with explicit
wire annotations are excluded. No float64 internal-byte count is presented as
measured wire traffic; the scalar opportunity is negligible in either format.

## Experiment and verification

Four eligible graphs × five variants (prune, then 2/4 operations × 1/8 MiB) ×
three scenarios (LAN-4, home-4, heterogeneous-8) × two policies (FIFO, critical
path) × four timing runs (noiseless seed 0 and noisy seeds 1700–1702) = **480
rollouts**. All four grouping settings yield identical candidate graphs within
each workload, so they are sensitivity settings, not independent replications.

All 20 snapshots load through the normal plan validator; all required roots
survive. An independent artifact verifier checks symbolic expansion, memory
certificates, the 480 unique rollout keys and all 32 summary aggregates. Four
focused structural tests pass, including protected outputs, memory rejection,
fan-out/joins, wire boundaries and repeated sliced operands. No numerical test
suite was run on this machine.

Reproduce from the repository root with existing local corpora:

```bash
.venv/bin/python -m ygaraty.simulation.plan_grouping \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-grouping-run
```

This metadata-only workflow needs no dataset URL or database. The original legacy
MLP files must exist and match their provenance hashes to recover their contracts.
The command blocks socket connections and numerical kernels, requires a fresh
output directory, preserves input files, compresses plan snapshots and rollout
records, and writes a ZIP archive.

Completed artifacts: workspace-parent
`calibration-audit/rl-scheduling/grouping-v1/` and `grouping-v1.zip`. The bundle
contains source, costs, plans, certificates, raw timings/settings, report,
verification script/result and checksums. From the repository root:

```bash
PYTHONPATH=. .venv/bin/python /absolute/path/to/grouping-v1/verify.py
```

## Next decision

Keep pruning as the demonstrated optimization. Do not prioritize deploying this
narrow grouping rule for the observed 0.31% modeled average. The MLP graphs are
mostly blocked by excluded kernels/metadata, while factorization has many joins.
The next distinct experiment is a **producer plus cheap epilogue** audit (for
example matmul followed by activation), with kernel-specific scratch limits and
an explicit check for loss of parallelism. That is outside this study's cheap-only
allowlist. Transformer benefits remain untested here: these runs use the local
factorization/MLP corpora, not experiment 54's owned Transformer runtime plan.

The follow-up [matmul–activation audit](MATMUL-GROUPING-AUDIT.md) checks direct
and bias epilogues: intervening reductions/assemblies block every local candidate.
