# What is left of the pairs penalty with handoffs on?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-4 (number added on finish) |
| **Category** | transport |
| **Where** | this machine |
| **Date** | Added 2026-11-01 |
| **Path** | Analysis of [experiment 63](063-pairs-penalty-with-handoffs.md)'s captures (`runs/exp64/cap-{P2h,N2h}-cap`); no new runs |

## Question

With handoffs on, pairs still lose 3.9% of an update. On handed chain hops, the time from upload end to the next kernel is 67 ms with pairs against 38 ms with one layer each (+368 ms per update), mostly hidden by pairs' faster uploads. Where do those milliseconds go: the consumer's handoff wait, the relay read, decoding, or the consumer running something else first?

## Prediction

TBD.

## Gate

TBD. Proposed: assign at least 80% of the P2h − N2h difference after the upload to named spans (`worker.handoff_wait`, the relay read, decode, other kernels on the consumer).

## Protocol

Extend `chain_trace`'s split to handed hops, using the `worker.handoff_wait` spans and the relay read spans in the captures, by consumer type.

## Log

- 2026-11-01: Added.
