# Are the consumers of a pair holder's results busy when they claim?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-3 (number added on finish) |
| **Category** | transport |
| **Where** | this machine |
| **Date** | Added 2026-11-01 |
| **Path** | Analysis of the captures of experiments 62 and 64 (`runs/exp59b/cap-*-cap`, `runs/experiments/WS-1/omarchy-20261101-1451/runs/cap-*-cap`); no new runs |

## Question

Without handoffs, the larger part of the pairs penalty is the claim: +215 ms per update in [experiment 62](062-pair-holder-upload-slots.md), +226 ms in [experiment 64](064-concurrent-result-posts.md). About a quarter of chain hops with pairs had a consumer that sent its claim only after the result was posted, against 9–10% with one layer each. Is the claim penalty consumers that were busy (claim sent late), or slower replies to claims already waiting?

## Prediction

TBD. Most likely: busy consumers, because pairs put both layers' tasks on one worker.

## Gate

TBD. Proposed: split each chain hop's "result posted to claim reply" into (a) the time from post end to the claim request's start when the claim started later, and (b) the rest. Busy consumers are the cause if (a) accounts for at least 60% of the P2 − N2 claim difference in both experiments.

## Protocol

`chain_trace` already records each hop's claim start and end (`worker.claim_http`) and the post end. Add the split, and break it down by consumer (pair holder, weak layer owner, vocabulary owner) and by what the consumer was doing (its kernel or its claim loop's previous reply).

## Log

- 2026-11-01: Added.
