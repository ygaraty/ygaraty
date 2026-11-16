# Three simple plan improvements, beyond duplicate expressions

Pulled through `52f9cbd` on 2026-10-30. The updated CSE notes establish that the
81-expression opportunity is in the compiler reference factorization graph, not
the deployed sharded-factorization plan. The upstream numerical CSE tests were
not rerun here: this machine remains restricted to offline simulations. The following study makes no new numerical-equivalence claim.

## Passes and output contracts

`ygaraty.simulation.plan_simplify` builds separate hypothetical plans for:

1. **Output-root pruning:** remove tasks not needed by declared required outputs,
   retaining unknown/owned operations and their dependencies conservatively.
2. **Single-input assembly bypass:** replace an assembly only when its sole full
   input has exactly the output shape and the assembly block covers that whole
   shape. No region slicing or additional attributes are allowed.
3. **Double-transpose cancellation:** alias the second transpose to the original
   input only when both are unsliced single-input transposes with empty attributes
   and compatible original/intermediate/output shapes.
4. A combined variant, compared with the unchanged baseline and each pass alone.

The metadata passes preserve logical names through an explicit alias map. They
may change copying/layout/lifetime behavior; deployment needs a compatible alias
consumer and numeric/precision validation. The transpose-only pass retains the
first transpose if it is otherwise unused; the combined pruning pass can remove
it. No arithmetic reassociation, fusion, or CSE is performed.

For tiled factorization, the shape-only compiler factory is rebuilt and its graph
hash must match the local snapshot. Required roots are its loss, updated parameters
and original leaves. For MLP graphs, the original legacy source file must match
its recorded hash; roots are its loss, parameter gradients and parameters. These
are conditional contracts: if external diagnostics or other intermediate values
are required, they must be added as roots before pruning. This study does not
preserve every original intermediate result.

Fused-factorization output contracts were not recovered. Pruning there is marked
**blocked**, not guessed; the output graph remains unchanged. Immutable persistent
leaf inputs are preserved, while owned, replicated, cache-affinity and persistent
non-leaf inputs are rejected. Original source files are never overwritten.

## Offline comparison

Six graphs × five variants × three scenarios × two policies × four timing runs
(one deterministic and three noisy seeds 1600–1602) produced **720 rollouts**.
Scenarios are LAN-4, home-4 and heterogeneous-8; policies are FIFO and critical path.
The same local cost models were used throughout. Gains below average the per-case
ratios across all six scenario/policy combinations for each workload.

| Workload | Original → combined tasks | Deterministic gain | Noisy gain |
|---|---:|---:|---:|
| Tiled factorization | 662 → 578 | 5.66% | 6.09% |
| MLP-76 | 76 → 65 | 7.12% | 6.85% |
| MLP-77 | 77 → 64 | 9.52% | 9.39% |
| MLP-340 | 340 → 266 | 17.74% | 17.85% |
| Fused-1024 | 98 → 98 | 0.00% | 0.00% |
| Fused-4096 | 358 → 358 | 0.00% | 0.00% |

The useful lever is **pruning computation outside required outputs**, not repeated
expressions. Pruning alone removed 80 tasks from tiled factorization (5.06%
deterministic gain) and 11/12/74 tasks from the MLP snapshots (7.12%/9.52%/17.74%
gain). These findings concern the existing corpus graphs, not every deployed job.

Metadata-only cleanup was not reliably beneficial. Six assembly aliases on tiled
factorization made modeled completion time **0.43% worse** when tested alone;
one double-transpose alias had zero measured simulated gain. MLP-77 had one
assembly alias with zero timing gain. The other MLP graphs and both fused graphs
had no matching metadata opportunities. Fewer nodes do not necessarily mean a
better schedule: metadata operations are already coordinator fast paths, while
aliases can change ready order, cache identity, and transfer behavior. No peak
memory or actual coordinator overhead improvement was measured.

These are modeled makespan predictions, not real training speedups. Numeric seeds
are not task-aligned common random numbers when tasks are removed. The noiseless
results provide the cleaner controlled comparison; three noisy seeds are only a
small sensitivity check. The existing cost tables are not newly calibrated.

## Reproduction, validation and artifacts

```bash
.venv/bin/python -m ygaraty.simulation.plan_simplify \
  --corpus /absolute/path/to/factorization-v1 \
  --corpus /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-simple-passes-run
```

The entry point disables socket connections and numerical workload kernels. It
uses existing local plans/costs only, writes compressed candidate plans with alias
and pruning certificates, and produces compressed rollouts plus a ZIP bundle.
A fresh output directory is required.

Successful artifacts are in workspace-parent
`calibration-audit/rl-scheduling/simple-passes-v2/`. An earlier incomplete `v1`
attempt stopped at an overly conservative guard on persistent *leaf* metadata;
it is labeled incomplete and superseded. That guard was narrowed without changing
or dropping immutable leaf inputs.

Verification loads all 30 candidate snapshots through the normal graph validator,
checks required output references, confirms that conditional pruning leaves no
remaining unreachable candidates, checks all 720 unique rollout keys, and
independently recomputes aggregate results. All **28 focused structural/RL tests**
pass. No numerical workload tests were executed and no production compiler or
scheduler behavior was changed. The next practical candidate is an opt-in,
output-contract-aware pruning pass, followed by authorized numerical validation
on a suitable execution machine.

## Validation on another machine

See [LOCAL-VALIDATION.md](LOCAL-VALIDATION.md) for the handoff files, numerical
output contract, required harness, paired benchmark protocol and acceptance
criteria. The modeled gains above remain hypotheses until those checks pass.

The subsequent [bounded grouping audit](GROUPING-AUDIT.md) compares pruning with
pruning plus cheap single-consumer chains; it finds only one scalar opportunity.
