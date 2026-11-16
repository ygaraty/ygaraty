# Does more frequent sync (H) recover 8-replica ConvNet learning efficiency?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | DO-3 (number added on finish) |
| **Category** | training |
| **Where** | remote |
| **Date** | Added 2026-10-30 |
| **Path** | One rented CPU host with matched total CPU: plain Adam and DiLoCo replicas 4/8 × H=5/20/50 on the CIFAR ConvNet; then three seeds and separate machines for promising settings |

## Question

[Experiment 44](044-diloco-convnet-cifar-one-host.md) left open whether more frequent synchronization recovers eight-replica learning efficiency.

## Gate

Predeclare the 60% validation target and the maximum example budget; evaluate at matching example counts. Single-host resource-matched screening does not establish distributed speedup.

## Protocol

Start on one rented CPU host, with matched total CPU resources, to screen learning efficiency: plain Adam and DiLoCo replicas 4/8 × H=5/20/50. Keep batch, split and initialization fixed; select a total example budget divisible by every complete-round size. Evaluate at matching example counts. Predeclare the 60% validation target and maximum budget.

Confirm promising settings with three seeds and a tuned single-machine baseline. Only then move selected settings to separate machines. Keep float64 wire as the control so synchronization frequency is not confounded with precision.

**Deliverable:** accuracy/loss versus examples, examples and time to target, synchronization count, compute and transfer cost. Mark unreachable targets explicitly. Single-host resource-matched screening does not establish distributed speedup.

## Log

- 2026-10-30: Added.
- 2026-11-01: Moved to its own file.
