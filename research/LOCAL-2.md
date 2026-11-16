# Diagnose the remaining scheduling opportunity

| | |
|---|---|
| **Conclusion** | **Dropped:** LOCAL-1 found no repeatable failure pattern to diagnose. |
| **Status** | dropped |
| **ID** | LOCAL-2 |
| **Category** | rl |
| **Where** | this machine |
| **Date** | Added 2026-10-30; dropped 2026-10-30 |
| **Path** | Counterfactual decisions and fixed-priority controls on the simulated workloads of [LOCAL-1](rl-scheduling/LOCAL-1.md) |

## Question

Run only if LOCAL-1 identifies a repeatable failure pattern. Use counterfactual decisions and fixed-priority controls to distinguish task ordering from cache/load placement. Compare any new feature or training objective as one ablation at a time, on development data. Preserve an untouched final split for the next experiment; do not repeatedly tune against LOCAL-1's final results.

**Decision:** a small, robust improvement justifies server scheduler evaluation later. It does not authorize changing the production scheduler or running actual workloads locally.

## Log

- 2026-10-30: Added. Dropped the same day: [LOCAL-1](rl-scheduling/LOCAL-1.md) was rejected with no repeatable failure pattern.
- 2026-11-01: Moved to its own file.
