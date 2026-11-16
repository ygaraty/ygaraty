# With handoffs on, pairs of layers lose 3.9% instead of 10.3%

| | |
|---|---|
| **Conclusion** | **Found:** handoffs shrink the pairs penalty from 10.3% to 3.9% of an update (measured, workstation, 3 runs per arm). |
| **Status** | executed |
| **ID** | 63 (WS-2; proposed as experiment 64) |
| **Category** | transport |
| **Where** | workstation (this machine) |
| **Date** | Added, started and finished 2026-11-01; follow-up 2 of [experiment 62](062-pair-holder-upload-slots.md) |
| **Path** | Local, on the workstation, with experiment 62's setup plus non-blocking handoffs: tiny model, 8 NumPy workers on emulated 100/20 Mb/s, 50 ms links, 2 of them with 100 Mb/s uplinks. Arms: pairs on the 2 fast workers (P2h) against one layer each (N2h) |
| **Result** | **Gate 1 passes, but some penalty survives.** P2h takes 2.670 s per update and N2h 2.570 s. The difference is 0.099 s (+3.9%), against experiment 62's 0.371 s (+10.3%) without handoffs, so handoffs remove 73% of it. That is under the frozen limit of half, but over the 2% that would make the pairs rule moot. All 90 captured chain hops are handed off in both arms (gate 2). Losses are bit-identical to experiment 62's in all 8 runs. On the handed hops, pairs still cost time after the upload: 67 against 38 ms per hop, +368 ms per update over the chain. Pairs win most of that back on faster uploads (73 against 91 ms per hop, −228 ms), so the chain is only 54 ms longer. Cost: free, about 5 minutes of runs |
| **Evidence** | [`evidence/placement/pairs-penalty-with-handoffs-2026-11-01.json`](evidence/placement/pairs-penalty-with-handoffs-2026-11-01.json) (per-run seconds per update and losses, placements, the chain split). Raw: `runs/exp64/r{1,2,3}-{P2h,N2h}/`, captures `runs/exp64/cap-{P2h,N2h}-cap/` (local, not in git; the directory keeps the plan's working number) |

## Question

Experiment 62 found the pairs penalty after the upload, with handoffs off. Of P2's +506 ms per update over N2 on the chain, about 260 ms is the result post and 215 ms is the claim reply. A handed hop skips both. The consumer is handed the task before its inputs exist and reads them from the relay as they land (experiments 58 and 59). Handoffs are now the default on pods (experiment 61).

So does the placement rule "one layer per fast worker" still matter in the configuration production runs? If the penalty is mostly gone, the paper's pairs result (experiments 42, 53, and L1) holds only without handoffs, and the placement program's missing term matters less than L1 suggests.

## Design

Experiment 62's recipe and strong workers, with `--handoff` (non-blocking, as in experiment 59) in every run. Upload slots stay at 2.

| Arm | Placement |
|---|---|
| **P2h** | pairs (`always`) |
| **N2h** | one layer each (`never`) |

- 3 timed runs per arm (12 updates; steady = median of updates 2 to 10), alternating: P2h N2h, N2h P2h, P2h N2h.
- Then one captured run per arm for `chain_trace --slots`.
- Before each run, wait for a 1-minute load under 3 and no other training process.

## Frozen before the runs

**Correctness gate:** losses bit-identical update by update in every run, and equal to experiment 62's (handoffs change when a task runs, not what it computes).

**Predictions:**
1. **Handoffs shrink the penalty:** P2h − N2h is at most half of experiment 62's P2 − N2 (0.371 s, so at most 0.186 s), measured as the difference of the arms' medians.
2. **What is left is not the post-and-claim path:** in the P2h capture, at least 60% of chain hops are handed off.

**What each outcome means:**
- **1 holds and the penalty is under 2% of N2h:** the pairs rule no longer matters with handoffs on. The paper says the pairs penalty is a property of the coordinator path, and placement work moves elsewhere.
- **1 holds but the penalty is still 2% or more:** some of it survives handoffs. Its chain split names the next part to remove.
- **1 fails:** handoffs do not remove the pairs penalty. Experiment 62's split over the coordinator path does not explain the penalty that production runs see. The next step is the same split on handed hops: handoff wait and relay read.

## Results

Placement did what each arm asked:
- **P2h:** the strong workers w0 and w1 held E0+E1 and D0+D1.
- **N2h:** w0 and w1 held D0 and D1, E0 and E1 went to the weak workers w6 and w4, and the four slices went to four other weak workers.

**Correctness: passes.** All 6 timed runs give experiment 62's 12 losses, and both captures its first 8, to the last printed digit.

| Arm | Seconds per update (r1, r2, r3) | Median |
|---|---|---|
| P2h | 2.658, 2.705, 2.670 | **2.670** |
| N2h | 2.570, 2.558, 2.662 | **2.570** |
| Difference | | **0.099 s (+3.9%)**; experiment 62 without handoffs: 0.371 s (+10.3%) |

1. **Passes:** 0.099 s is under the frozen 0.186 s. The penalty is still 3.9% of N2h, so the outcome is the middle one: some of it survives handoffs. It is also a narrower result than experiment 62's. N2h's third run (2.662 s) is slower than P2h's first (2.658 s), whereas in experiment 62 every pairs run was slower than every one-each run, by 0.30 s.
2. **Passes:** all 90 chain hops are handed off in both captures.

### What is left

Medians of the 90 captured chain hops (all handed), and sums per update over the chain:

| | P2h | N2h | P2h − N2h, sum per update |
|---|---|---|---|
| Chain | 2,537 ms | 2,483 ms | +54 ms |
| Median hop | 164 ms | 148 ms | |
| Upload | 73 ms | 91 ms | −228 ms (pairs put more of the chain on the 100 Mb/s uplinks) |
| Upload end to next kernel | 67 ms | 38 ms | **+368 ms** |

- **The cost of pairs is still after the upload,** at about three quarters of experiment 62's size (+368 against +506 ms per update). On a handed hop, that is the consumer's handoff read and whatever it waits for before its kernel. `chain_trace` does not split that part yet.
- **Faster uploads hide most of it.** Pairs keep more of the chain on the two 100 Mb/s uplinks. The chain is only 54 ms longer, while the steady medians differ by 99 ms.

## What it means

- **The pairs rule still holds with handoffs on, but it matters far less:** 3.9% of an update instead of 10.3%. For the paper, the pairs penalty is mostly a cost of the coordinator path (post and claim), which handoffs remove. A smaller part survives on the handed path.
- **Handoffs give this pool a large speedup.** Against experiment 62's runs from the same afternoon, they take 3.600 → 2.570 s per update with one layer each (−28.6%), and 3.971 → 2.670 s with pairs (−32.8%). That is in line with experiments 58 and 59 (−23.3% on uniform links). These are separate campaigns, not interleaved, so the size is indicative.
- **[experiment 64](064-concurrent-result-posts.md) still matters, but only for runs without handoffs.** It tests whether the post queue causes experiment 62's penalty. With handoffs, the post is off the chain.
- **Next, if this is worth more:** split the handed hop's time after the upload (`worker.handoff_wait`, relay read, decode) in `chain_trace`, P2h against N2h. That is analysis of these captures, with no new runs.

## Limits

- Experiment 62's limits apply: a small model on one machine, emulated links and no jitter, and strong workers that only announce more GFLOP/s. The tiny model's uploads are round-trip-bound. On the 30.3M model, uploads are byte-bound, and the penalty could have a different source.
- **Three runs per arm,** and one N2h run overlaps the P2h range. The 3.9% is a difference of medians, not a separation.
- **One captured run per arm.** The chain split is from those captures.

## Reproduce

Experiment 62's command (its Reproduce section) with `--handoff`, and `--strong-uploaders 2`:

```bash
runs/exp64/drive.sh run NAME always|never 2 12 [cap]   # one run, about 40 s
runs/exp64/campaign.sh                                  # all 8, about 5 minutes
ygaraty experiment chain_trace --slots runs/exp64/cap-P2h-cap runs/exp64/cap-N2h-cap
```

The relay server was rebuilt at `e576e2a` before the runs. Experiment 62's runs used the build from `01:57` the same day, which differs only in reading gzipped handoff requests, which these workers do not send.

## Log

- 2026-11-01: Added; gate frozen above before any run. Relay server rebuilt from `e576e2a` (the gzip handoff-read fix) before the runs.
- 2026-11-01 14:46: Started at `39eacf7+dirty` (experiment 62's uncommitted tools; committed in `6cd154c`). Driver `runs/exp64/campaign.sh`, output in `runs/exp64/`. Started before the watchdog rule; each run took about 40 s.
- 2026-11-01 14:50: Finished, 8 of 8 runs exit 0. Gates computed by script from the profiles and logs (the summary in the evidence JSON).
