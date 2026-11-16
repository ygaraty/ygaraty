# Does the MLP tile-epilogue rewrite still gain under pessimistic overhead?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-2 (number added on finish) |
| **Category** | simulation |
| **Where** | this machine |
| **Date** | Added 2026-10-30 |
| **Path** | Discrete-event simulation of pruning alone against pruning plus completed-reduction → bias → ReLU on hash-verified MLP graphs; sensitivity sweep |

## Question

The current MLP rewrite predicts roughly 9–15% improvement, but has no numerical/runtime validation.

References: [tile-epilogue audit](rl-scheduling/TILE-EPILOGUE-AUDIT.md), [server validation handoff](rl-scheduling/SERVER-VALIDATION.md).

## Gate

Metadata contracts and admission checks pass; candidate improves in the predeclared target scenarios under pessimistic overhead. Any regression remains visible. Passing this gate permits proposing server numerical tests, not deploying the rewrite.

## Protocol

Use existing local artifacts, shape-only graphs, local cost models and discrete-event simulation. Do not execute workload kernels, start services, emulate a live network, fetch datasets or contact external stores. Where a simulator needs a dataset, pass an explicit `--dataset-url sqlite:////absolute/path/to/dataset.sqlite`; resolve an existing local file before execution. Report missing inputs rather than fetching them. Prefer the direct simulator entry point over numbered runners that may synchronize results.

Record revision and dirty-source hashes, input/cost-model hashes, scenario settings, seeds and output contracts. Save outputs to a fresh local directory. Separate simulated time from measured host runtime. These plans do not authorize remote execution.

- Compare pruning alone with pruning plus completed-reduction → bias → ReLU on the same hash-verified graphs. Preserve both preactivation and activation, original reduction order/multiplicity and every declared output root.
- Sweep admissible per-task memory limits, packing/copy cost, wrapper overhead, worker compute imbalance, link imbalance and available parallelism. Keep infeasible variants explicitly rejected.
- Reuse the existing 1/8 MiB and 0/0.5/2 ms sensitivity points, then add a pessimistic overhead setting before examining results. Record graph changes and estimated lifetimes.
- Evaluate deterministic cases and at least 20 noisy seeds per selected scenario. Changing the graph changes event/noise alignment; identical seeds alone do not establish paired randomness.
- Report per-case results, not only pooled gains. Separate graph-generation/search cost from predicted execution time and estimate reuse needed to amortize it.

**Deliverable:** sensitivity matrix, infeasibility reasons, modeled break-even overhead and a pinned candidate bundle for the server handoff.

## Log

- 2026-10-30: Added.
- 2026-11-01: Moved to its own file.
