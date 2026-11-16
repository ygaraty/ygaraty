# Report-only dead-task and common-expression audit

Completed offline on 2026-10-30. No plan was rewritten, no numerical kernel was
executed, and no performance or numerical-equivalence claim is made.

The pass scanned task-bearing JSON files under local `runs/`, `factorization-v1`
and `mlp-validation-v1`. Unlike the earlier filename-based inventory, this also
finds `bundle.json` and hash-named corpus snapshots. It found **262 plans/bundles**,
representing **30 distinct serialized task/root structures**. Repeated snapshots
are deduplicated for candidate totals; these are not graph-isomorphism classes.
All source files were hashed and verified unchanged after the pass.

## Findings

| Finding | Result |
|---|---:|
| Files with usable declared output roots | 239 |
| Files blocked for dead-task analysis | 23 |
| Unique structures with dead-task candidates | 16 |
| Conditional dead-task candidates across those structures | 734 |
| Unique structures with exact common-expression candidates | 4 |
| Potential redundant expressions across those structures | 212 |
| Invalid input plans detected | 0 |

Dead-task and repeated-expression counts may overlap and must not be added.
Task counts are not predicted compute, transfer, memory or wall-time savings.

Representative findings:

| Local plan | Tasks | Conditional dead candidates | Exact repeated-expression redundancy |
|---|---:|---:|---:|
| `runs/stream-capacity-128m/bundle.json` | 1,630 | 266 | 0 |
| `runs/lstm-small-window/bundle.json` | 1,072 | 75 | 52 |
| Tiled factorization corpus snapshot | 662 | **Blocked: missing roots** | 81 |
| Legacy tile-64 MLP cached plan | 340 | 74 | 0 |
| `runs/redis-memory-verified/bundle.json` | 440 | 70 | 0 |

The 266 candidates in stream-capacity include 256 matmul tasks, eight add-many
tasks, a transpose and an assembly. The tiled factorization graph's 81 exact
redundancies comprise 80 mul tasks and one broadcast. For example,
`v21/block/128/384` and `v22/block/128/384` have identical ordered inputs, regions,
shape, operation and attributes. This is meaningful redundancy to inspect, but
merging requires aliases: consumers must still receive both logical references,
and reductions must retain input order and multiplicity.

## What the pass establishes

**Dead-task candidates:** collect declared `loss`, `gradients`, `updates`,
`outputs`, `parameters`, and `keep` references; retain ancestors of these roots.
Unknown operations and owned tasks are also retained as roots, along with their
ancestors, so the pass does not discard unreviewed effects. Candidates are tasks
outside this conservative live set. Graph checks cover unique task IDs, cycles,
missing references when leaves/initializers are declared, and versioned snapshot
checksums. Legacy external references can only be inferred; the report flags this.

This is reachability **conditional on the declared output contract**, not proof
that the contract includes every externally observable value. Diagnostics,
requested intermediates, optimizer state, validation behavior, failure semantics,
or other retained values may require additional roots. Absence of roots yields a
blocked result (`dead_candidates: null`), never a guess based on terminal nodes.
Malformed root fields and unresolved root references also block the conclusion.

The simulation snapshot serializer currently preserves tasks/leaves and runtime
metadata but omits `loss`/`updates` roots. Its graph is therefore useful for
scheduling but insufficient for dead-task elimination. Before rewriting these
plans, we should explicitly preserve required outputs in the plan format or
supply a checked external output contract.

**Common expressions:** a narrow allowlist of functional operations reviewed in
`runtime/kernels.py` is compared within each plan. Signatures retain exact ordered
input IDs and regions, shapes, all attributes, and additional task fields. Only
`id`, `attempts`, and `committed` are omitted. Unknown/owned operations are excluded.
There is no commutative operand sorting, arithmetic reassociation, recursive alias
substitution, or cross-plan/value-version merging. This conservative detector can
miss equivalent expressions; it should not invent algebraic equivalences.

Exact signatures identify syntactic candidates, not permission to merge. Safe
rewrites still require immutable input versions, compatible precision/layout and
backend behavior, preservation of output IDs/aliases, lifetime/ownership semantics,
and an explicit observable-output contract. Even identical floating-point kernels
need not promise bitwise equality across different execution backends. No replay
or numerical-equivalence proof was attempted on this simulation-only machine.

## Artifacts and checks

```bash
.venv/bin/python -m ygaraty.simulation.plan_audit \
  --root runs \
  --root /absolute/path/to/factorization-v1 \
  --root /absolute/path/to/mlp-validation-v1 \
  --output /absolute/path/to/new-audit-directory
```

Artifacts are in workspace-parent
`calibration-audit/rl-scheduling/plan-audit-v1/`: compressed per-plan findings,
summary, verification record and audit source. A ZIP bundle is also available.
Every candidate lists its original file and task IDs for review.

Three focused tests cover output-root reachability, preservation of unknown and
owned operations, exact-expression input order/attributes, aliases of declared
outputs, missing roots and cycle rejection. An independent pass rechecked
reachability and expression equality for all 30 unique structures, then verified
that **all 262 input hashes remained unchanged**. This completes a report-only
screening; no graph cleanup, numerical rerun or runtime deployment occurred.

## Runnable follow-up for the 81 factorization expressions

A separate [pinned offline what-if experiment](CSE-81.md) can now be triggered with
`public/research/rl-scheduling/run-cse-81.sh`. It builds an alias-carrying hypothetical
581-task graph, checks symbolic reconstruction and compares modeled timings.
This does not change the report-only audit or assert numerical equivalence.

A subsequent [three-pass simulation study](SIMPLE-PASSES.md) recovered explicit
output roots for selected graphs and found useful pruning opportunities on MLP
plans as well as the reference factorization graph. Metadata alias passes were
mostly neutral and sometimes slower; these remain separate hypothetical plans.
