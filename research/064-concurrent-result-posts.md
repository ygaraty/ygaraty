# Removing the result-post queue recovers only 17% of the pairs penalty; the claim path is what is left

| | |
|---|---|
| **Conclusion** | **Failed gate:** four posts in flight remove the post queue but recover only 17% of the pairs penalty (measured, workstation). |
| **Status** | executed |
| **ID** | 64 (WS-1; proposed as experiment 63) |
| **Category** | transport |
| **Where** | workstation (this machine) |
| **Date** | Added, started and finished 2026-11-01; follow-up 1 of [experiment 62](062-pair-holder-upload-slots.md) |
| **Path** | Local, on the workstation, with experiment 62's setup, handoffs off: tiny model, 8 NumPy workers on emulated 100/20 Mb/s, 50 ms links, 2 with 100 Mb/s uplinks. Arms: pairs (P2), pairs with 4 upload slots and 4 posts in flight on the 2 fast workers (P4q), one layer each (N2) |
| **Result** | **Gate 2 passes, gate 3 fails.** The penalty reproduces: P2 3.910 s per update, N2 3.624 s (+7.9%; experiment 62 had +10.3%). P4q takes 3.860 s, recovering 17% of the gap against the frozen 30%. Its three runs (3.755–3.877 s) overlap P2's (3.865–4.030 s). The post queue did go: the chain's wait for a post in flight fell from 243 to 43 ms per update (ratio 0.18, gate ≤ 0.33). But the post itself shrank by only 110 ms, and the claim path grew by 50 ms. Losses are bit-identical to experiment 62's in all 13 runs, with 0 retries and 0 rejected. By the frozen reading, **the post queue is not the cause either: the claim path is next.** Of P2's +445 ms per update after the upload, +226 ms is claim replies. In 26% of P2's chain hops the consumer sent its claim only after the result was posted, against 9% in N2. Cost: free, 10 minutes of runs |
| **Evidence** | [`evidence/placement/concurrent-result-posts-2026-11-01.json`](evidence/placement/concurrent-result-posts-2026-11-01.json) (per-run seconds per update and losses, placements, the chain split). Raw: `runs/experiments/WS-1/omarchy-20261101-1451/` (watchdog `status.json`, `output.log`, and one directory per run; local, not in git). Tests: `tests/test_exp62_tools.py` |

## Question

In experiment 62, results almost never waited for an upload slot on the chain. But between upload end and the next kernel, P2 (pairs) spent 506 ms more per update than N2 (one layer each):
- **About 200 ms was results waiting for the worker's one result post in flight.** The worker posts one request at a time and puts whatever finished during it into the next request. Per update, that wait came to 312 ms in P2 and 108 ms in N2.
- **With four upload slots (P4), the queue moved there.** The pair holders' chain results waited a median of 50 ms for the post in flight, and P4 gained nothing.

**Is the single post in flight the cause?** Remove it on the pair holders by letting several posts be in flight at once, each on its own connection, with nothing else changed. As in experiment 62, this changes only the strong workers.

## What was built

- `http-worker --posters K`: K post threads in `StreamPoster`, each with its own HTTP connection (default 1, today's behaviour). `http_profile --strong-posters K` sets it on the strong workers only. Unit tests: K threads, each on its own connection, and `--posters` reaching only the strong workers. Built and tested before the runs, committed in `6cd154c`.
- Safety, checked before the runs:
  - The coordinator's result route runs on a threading server and commits under the job's lock (`coordination/jobs.py`, `Job.record_result`). Without handoffs, a task is claimed only after its inputs have committed, so a worker's results arriving out of order or in parallel change nothing it checks.
  - With handoffs, a consumer's result waits for its producer's commit (`coordination/handoff.py`, `handoff_wait`), so more posters are also safe there. They are not tested with handoffs here.

## Design

Experiment 62's recipe and strong workers, handoffs off.

| Arm | Placement | Upload slots on the strong workers | Posts in flight on the strong workers |
|---|---|---|---|
| **P2** | pairs (`always`) | 2 | 1 (experiment 62's P2) |
| **P4q** | pairs | 4 | 4 |
| **N2** | one layer each (`never`) | 2 | 1 |

P4q has 4 upload slots because experiment 62 showed that with only more posters, the upload slots would become the queue again: P4's off-chain slot queue was 7 ms against P2's 68 ms. P4 itself is not rerun. Experiment 62 measured it at 3.904 s, within P2's spread.

- 3 timed runs per arm (12 updates; steady = median of updates 2 to 10), order rotated: P2 P4q N2, P4q N2 P2, N2 P2 P4q.
- Then one captured run per arm for `chain_trace --slots`.
- Before each run, wait for a 1-minute load under 3 and no other training process.

One 3-update rehearsal of P4q came first, since several posts in flight had never run. About 10 minutes of runs.

## Frozen before the runs

**Correctness gate:** losses bit-identical update by update in every run, and equal to experiment 62's.

**Predictions:**
1. **The penalty reproduces:** P2 is at least 5% slower than N2, as in experiment 62 (+10.3%). If not, stop and report.
2. **The post queue goes:** in P4q's capture, the chain's wait for a post in flight is at most a third of P2's, summed per update. Experiment 62's P2 had 312 ms.
3. **Removing it removes its share of the penalty:** P4q recovers at least 30% of P2 − N2. If the post queue is the cause, it accounts for about 200 of the chain's 414 ms gap, so about 40% is predicted. The rest is in the claim path.

**What each outcome means:**
- **2 and 3 hold:** the post queue is measured as part of the cost of pairs. The paper's Section 6 states it, and more posters become a candidate default (on everyone, as a separate A/B).
- **2 holds but 3 fails:** the post queue is not the cause either. The claim path (experiment 62's +215 ms) is next.
- **2 fails:** the posters do not remove the wait. Find out why from the capture before drawing any conclusion.

## Results

**Correctness: passes.** All 9 timed runs, the 3 captures and the rehearsal give experiment 62's losses to the last printed digit. Every run had 0 retries and 0 rejected results. Placement had experiment 62's shape in each arm: pairs on the two strong workers in P2 and P4q, and in N2 the decoder layers on the strong workers and the encoder layers on two weak ones. Only which of two identical workers got which layer varied.

| Arm | Seconds per update (r1, r2, r3) | Median | Against N2 |
|---|---|---|---|
| P2 | 3.910, 3.865, 4.030 | **3.910** | +7.9% |
| P4q | 3.755, 3.877, 3.860 | **3.860** | +6.5% |
| N2 | 3.678, 3.624, 3.554 | **3.624** | |

1. **Passes.** P2 is 7.9% slower than N2. Experiment 62 measured 10.3% on the same setup two hours earlier, and every P2 run is again slower than every N2 run.
2. **Passes.** In P4q's capture, the chain's wait for a post in flight is 43 ms per update, against 243 in P2 (0.18 of it). N2 has 81.
3. **Fails.** P4q recovers 17% of P2 − N2 (0.050 of 0.286 s), against the 30% needed. Its runs overlap P2's.

### The chain, after the upload

Sums per update over the 90 captured chain hops (all through the coordinator), and medians per hop:

| | P2 | P4q | N2 | P2 − N2 | P4q − P2 |
|---|---|---|---|---|---|
| Chain | 3,956 ms | 3,835 ms | 3,633 ms | +323 | −121 |
| Upload | 1,198 (70 per hop) | 1,224 (74) | 1,283 (90) | −85 | +26 |
| Upload end to result posted | 1,154 (58) | 1,044 (60) | 955 (57) | +199 | **−110** |
| … of which waiting for a post in flight | 243 | 43 | 81 | +162 | **−200** |
| Result posted to claim reply | 1,289 (65) | 1,339 (70) | 1,063 (62) | **+226** | +50 |
| Claim reply to kernel | 111 (1) | 100 (1) | 91 (1) | +20 | −11 |
| Claim sent only after the result was posted | 26% of hops | 27% | 9% | | |

- **Removing the wait saved about half of it.** The post queue fell by 200 ms per update, but the post time fell by only 110 ms. Requests that used to wait now run alongside each other, on the same emulated uplink, and each takes longer.
- **The claim path did not move, or grew.** P4q's claim replies came 50 ms per update later than P2's. With results posted sooner, more of the hop is spent waiting for the consumer's claim.
- **The claim is the larger half of the penalty in both experiments:** +226 ms here, +215 ms in experiment 62. In P2 and P4q, about a quarter of chain hops had a consumer that sent its claim only after the result was posted, so it was not waiting for the result. That is three times as often as in N2. The consumers of a pair holder's results are busy more often, most simply because a pair holder runs both layers' tasks and claims between them.

## What it means

- **For the paper:** without handoffs, the cost of pairs is not upload slots (experiment 62) and not the single post in flight (this experiment). The claim path is what is left: consumers that claim late because they are busy. With handoffs on, the penalty shrinks to 3.9% ([experiment 63](063-pairs-penalty-with-handoffs.md)), so the claim path matters most for runs without handoffs.
- **More posters are not a default candidate.** They remove the queue they target, but they do not shorten the update measurably.
- **Next, if this is worth more:** split the claim time in these captures into the time before the consumer sent its claim (it was busy) and the reply itself. That is analysis, with no new runs.

## Log

- 2026-11-01: Added, with `--posters` built and unit-tested. Gate frozen above before any run. Queued to start when WS-2 finishes.
- 2026-11-01 14:51: Started at `6cd154c`, under watch.py (stall 300 s, busy stall 1200 s, max runtime 2400 s; progress = run exit lines). Driver `runs/experiments/WS-1/campaign.sh`, output in `runs/experiments/WS-1/omarchy-20261101-1451/`.
- 2026-11-01 14:52: Rehearsal (P4q, 3 updates) exit 0, losses equal to experiment 62's first 3.
- 2026-11-01 15:01: Finished. Watchdog state `finished`, 13 of 13 runs exit 0. Gates computed by script from the profiles, logs and captures (the summary in the evidence JSON).

## Limits

- Experiment 62's limits apply: a small model on one machine, emulated links, and strong workers that only announce more GFLOP/s.
- More posts in flight also mean more concurrent requests to the coordinator. On loopback, the coordinator is not the bottleneck (experiment 58), so this should matter little.

- **Three runs per arm,** and P4q's runs overlap P2's. The 17% is a difference of medians.
- **One captured run per arm.** The chain split is from those captures, and the post-in-flight wait varies between campaigns (P2: 312 ms per update in experiment 62, 243 here).

## Reproduce

```bash
runs/experiments/WS-1/drive.sh run OUT NAME always|never SLOTS POSTERS 12 [cap]   # one run, about 50 s
python3 <experiments skill>/scripts/watch.py run --dir OUT --stall 300 --max-runtime 2400 --progress 'exit [0-9]+|waiting|campaign finished' -- runs/experiments/WS-1/campaign.sh OUT/runs
ygaraty experiment chain_trace --slots OUT/runs/cap-P2-cap OUT/runs/cap-P4q-cap OUT/runs/cap-N2-cap
python -m unittest tests.test_exp62_tools
```

The driver is experiment 62's command with `--strong-uploaders SLOTS --strong-posters POSTERS`. The campaign waits for experiment 63's to finish and for the machine to be idle, rehearses P4q for 3 updates, then runs the 9 timed runs and 3 captures (about 10 minutes).
