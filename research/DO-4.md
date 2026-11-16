# Does the pinned MLP tile-epilogue bundle pass numerical validation on a server?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | DO-4 (number added on finish) |
| **Category** | correctness |
| **Where** | remote |
| **Date** | Added 2026-10-30 |
| **Path** | The pinned MLP tile-epilogue handoff in [rl-scheduling/SERVER-VALIDATION.md](rl-scheduling/SERVER-VALIDATION.md), on a rented machine; needs OFF-2 first |

## Question

Does the tile-epilogue rewrite (completed reduction → bias → ReLU) compute the same numbers as pruning alone on the hash-verified graphs, and is it faster sequentially? Needs [OFF-2](OFF-2.md) first.

## Gate

Numerical acceptance against pruning alone precedes timing; sequential timing precedes any distributed claim. Missing hash-verified source plans are blockers.

## Protocol

If cloud time remains, execute the pinned MLP tile-epilogue numerical handoff in [rl-scheduling/SERVER-VALIDATION.md](rl-scheduling/SERVER-VALIDATION.md). Missing hash-verified source plans are blockers. Preserve ordered reductions, preactivation/backward consumers, all required roots and input immutability. Compare against pruning alone. Numerical acceptance precedes timing; sequential timing precedes any distributed claim.

## Log

- 2026-10-30: Added, as the optional validation in the DigitalOcean plan.
- 2026-11-01: Moved to its own file.
