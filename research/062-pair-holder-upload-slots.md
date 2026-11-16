# Pairs of layers lose 10%, but not to upload slots: the queue is after the upload

| | |
|---|---|
| **Conclusion** | **Failed gate:** upload slots don't cause the 10% pairs penalty; results wait for the single result post (measured, workstation). |
| **Status** | executed |
| **ID** | 62 (planned as 59-b) |
| **Category** | transport |
| **Date** | 2026-11-01. Planned as experiment 59 (pair holders' upload slots), numbered 62 when it ran, since 59 to 61 had been taken |
| **Path** | Local, on the workstation that ran experiments 51, 54 and 58. Experiment 58's 1.06M-parameter recipe with 8 NumPy workers on emulated links, 2 of them with 100 Mb/s uplinks. Not on the simulation-only machine: its `AGENTS.md` forbids training jobs, coordinators, workers and link emulators |
| **Result** | **Not the cause; the queue is elsewhere.** The tiny model shows the pairs penalty: P2 is **10.3% slower** than N2 (3.971 against 3.600 s per update; gate 1 passes). But results almost never wait for an upload slot on the chain: a median of **0.27 ms** on the pair holders' 56 chain hops (gate 2 fails). Doubling the slots recovers **18%** of the gap (3.904 s, inside P2's spread), with an unchanged on-chain wait (gate 3 fails). Losses are bit-identical in all 12 runs. The penalty sits after the upload: **+506 ms per update** on the chain between upload end and the next kernel. About 200 ms of that is results waiting for the worker's one result post in flight, and about 215 ms is claim replies arriving later. With four upload slots, the pair holders' chain results wait a median **50 ms** for that post: the queue moved downstream, it did not go away. Follow-ups: [experiment 64](064-concurrent-result-posts.md) and [experiment 63](063-pairs-penalty-with-handoffs.md) |
| **Evidence** | [`evidence/placement/pair-holder-upload-slots-2026-11-01.json`](evidence/placement/pair-holder-upload-slots-2026-11-01.json) (per-run seconds per update and losses, placements, the chain split). Raw: `runs/exp59b/r{1,2,3}-{P2,P4,N2}/`, captures `runs/exp59b/cap-{P2,P4,N2}-cap/` (local, not in git; the directory keeps the plan's working name). Tests: `tests/test_exp62_tools.py` |

## Question

The paper's L1 decomposition (`papers/uplink-bound/L1-BETA.md`) found that one layer per fast worker beats pairs (experiments 42, 53), but no upload-time model explains it:
- no uplink is busy for more than half an update;
- the placement program with honest upload times ranks every pool wrong.

The gap on the pairs cells (2.3–3.4 s) correlates with the time a pair holder's finished results wait for one of its two upload slots (about 3.1 s per update for a decoder-pair holder). A pair holder makes 25–34 round-trip-bound uploads per update instead of 14–20, so both slots fill and results wait on the chain.

**Is that queue the cause?** Measure it on the critical path, then remove it by giving the pair holders more upload slots, with nothing else changed.

## What was built

1. **Mixed links in `http_profile`:** `--strong N --strong-up-mbps M --strong-cores C --strong-gflops G --worker-gflops g`. The first N workers get their own latency proxy at M Mb/s up and announce C cores and G GFLOP/s. The others keep `--up-mbps` and announce g. Capacity placement (`YGARATY_PLACEMENT=capacity`) measures each worker's uplink from its probe, and saw about 99 Mb/s on the strong workers and 20 on the others.
2. **Upload slots per worker:** `http-worker --uploaders K` (passed to `StreamPoster`, default 2), and `http_profile --strong-uploaders K` for the strong workers only.
3. **Placement rule passthrough:** `YGARATY_PLACEMENT` and `YGARATY_PLACEMENT_PAIRS=always|never` reach the coordinator in `http_profile`. Each run's `profile.json` records who held what (`placement`) and the measured links (`capacities`).
4. **`chain_trace --slots`:**
   - For every hop on the critical chain: the producer, its slot wait (upload start minus kernel end, which includes encoding; of which the `worker.upload_queue` wait), and each worker's total slot wait per update.
   - Added after the gates failed: how the time from upload end to the next kernel divides for hops through the coordinator. The parts are the result post (of which waiting for the worker's one post in flight, `worker.post_queue`), the claim reply after the post, and reply to kernel.

Unit tests in `tests/test_exp62_tools.py`:
- `--uploaders` sets the number of upload threads;
- strong workers get their link, hello and slots, and the defaults leave every worker alike;
- `chain_trace` on experiment 58's capture reproduces its 223 ms median hop over 90 chain hops, with slot wait added, and the split after the upload adds up.

## Design

Recipe: experiment 58's command, with handoffs off (as in experiments 42 and 53), `--workers 8 --up-mbps 20 --down-mbps 100 --rtt-ms 50 --strong 2 --strong-up-mbps 100`. The model has 4 layers (E0, E1, D0, D1) and 4 vocabulary slices.

| Arm | Placement | Upload slots on the two strong workers |
|---|---|---|
| **P2** | pairs (`always`): strong workers hold E0+E1 and D0+D1 | 2 |
| **P4** | pairs (`always`) | 4 |
| **N2** | one layer each (`never`): strong workers hold two layers, the other two go to 20 Mb/s workers | 2 |

- 3 timed runs per arm (12 updates; steady = median of updates 2 to 10, as in experiment 58), order rotated: P2 P4 N2, P4 N2 P2, N2 P2 P4.
- Then one captured run per arm (8 updates) for `chain_trace`.
- Before each run, wait for a 1-minute load under 3 and no other training process.

## Frozen before the runs

**Correctness gate:** losses bit-identical update by update in every run (placement and slot count do not change the arithmetic).

**Predictions:**
1. **The tiny model shows the pairs penalty:** P2 at least 5% slower than N2. If it is under 2%, this setup cannot test the mechanism. Stop there and report it.
2. **The queue is on the chain:** in P2, chain hops leaving a pair holder carry a median slot wait of at least 30 ms. In N2, chain hops leaving any owner carry at most 10 ms.
3. **Removing the queue removes the penalty:** P4 recovers at least half of P2 − N2, with the pair holders' on-chain slot wait at most a third of P2's.

**What each outcome means for the paper:**
- **2 and 3 hold:** slot queueing is measured as the cost of pairs. Section 6 states it as measured, and the placement program gets a queueing term.
- **1 holds but 3 fails:** queueing is not the cause. Section 6 says the pairs penalty is unexplained.
- **1 fails:** no conclusion. The test moves to the 30.3M model on DigitalOcean (R1's pool, about $3).

## Results

Placement did what each arm asked:
- **P2:** the strong workers w0 and w1 held D0+D1 and E0+E1, and vocabulary slices went to four weak workers.
- **P4:** the same pairs, with w0 and w1 swapped.
- **N2:** w0 and w1 held D1 and D0, E0 and E1 went to the weak workers w3 and w5, and the slices went to four other weak workers.

**Correctness: passes.** All 9 timed runs give the same 12 losses to the last printed digit, and the captures agree.

| Arm | Seconds per update (r1, r2, r3) | Median | Against N2 |
|---|---|---|---|
| P2 | 3.971, 4.018, 3.926 | **3.971** | +10.3% |
| P4 | 3.904, 4.019, 3.887 | **3.904** | +8.4% |
| N2 | 3.626, 3.600, 3.575 | **3.600** | |

1. **Passes.** P2 is 10.3% slower than N2, and every P2 run is slower than every N2 run, by at least 0.30 s.
2. **Fails.** On the chain, results do not wait for upload slots. P2's pair holders' 56 chain hops have a median slot wait of 0.27 ms, the same as N2's owners (0.27 ms), against the 30 ms the mechanism needed. Off the chain, a slot queue does exist on the pair holders: 68 and 71 ms per update over all their results, against 1 to 28 ms on the other workers. That is two orders of magnitude below the 3.1 s per update L1 measured on the 30.3M model, where uploads are byte-bound rather than round-trip-bound.
3. **Fails.** P4 recovers 18% of P2 − N2, and its three runs overlap P2's (3.887–4.019 against 3.926–4.018). The extra slots did remove the off-chain queue (7 and 8 ms per update on the pair holders), and the pair holders' on-chain slot wait stayed at 0.28 ms (ratio 1.02).

By the frozen reading, **upload-slot queueing is not the cost of pairs**, at least on the tiny model.

### Where the 0.4 s goes

The chain has 15 hops in every arm, all through the coordinator (handoffs off). It is 3,996 ms in P2, 3,936 ms in P4 and 3,582 ms in N2. Medians of the 90 captured chain hops, and sums per update over the chain:

| | P2 | P4 | N2 | P2 − N2, sum per update |
|---|---|---|---|---|
| Kernel end to upload start | 0 ms | 0 ms | 0 ms | |
| Upload | 71 ms | 72 ms | 87 ms | −75 ms (pairs put more of the chain on the 100 Mb/s uplinks) |
| Upload end to result posted | 72 ms; 1,228 | 70 ms; 1,271 | 57 ms; 971 | **+257 ms** |
| … of which waiting for the post in flight | 0 ms; 312 | 2 ms; 370 | 0 ms; 108 | **+204 ms** |
| Result posted to claim reply | 66 ms; 1,263 | 71 ms; 1,228 | 63 ms; 1,048 | **+215 ms** |
| Claim reply to kernel | 1 ms; 109 | 1 ms; 99 | 1 ms; 75 | +34 ms |
| Claim sent only after the result was posted | 20% of hops | 23% | 10% | |

- **The penalty is after the upload, not before it.** The time from upload end to the next kernel sums to +506 ms per update in P2. That is all of the chain's 414 ms gap plus the 75 ms the faster uploads save. It is spread over every consumer: hops into the vocabulary workers, which hold no layer, are as slow as hops into the pair holders (167 against 169 ms after the upload in P2; 129 and 131 ms in N2).
- **Half of it is the result post.** A worker has one result-post thread, and it coalesces whatever finished while a post was in flight into the next request. A chain result that is uploaded during a post waits for that post's round trip (about 56 ms on these links). A pair holder posts about twice as often as a single owner, so this happens more on the chain.
- **With four upload slots, that wait becomes the queue.** In P4, the pair holders' 51 chain hops wait a median of **50 ms** for the post in flight, against 0 ms in P2: results leave the uploaders sooner and stack up at the single poster. That is why P4 gains nothing, and why gate 3's failure does not clear queueing on the result path in general. It clears the upload slots only.
- **The other half is the claim.** After the result is posted, the consumer's claim reply comes about 215 ms later per update in P2. Twice as many consumers in P2 sent their claim only after the result was posted (20% against 10%). Those workers were busy and not waiting for the result. Which part of the claim path the rest is spent in is not separated here.

## What it means

- **For the paper, the frozen reading applies:** "1 holds but 3 fails". Section 6 should not say the pairs penalty is queueing for upload slots, as `paper.md` and L1-BETA.md now suggest. On the tiny model, it is the per-worker result path after the upload: one result post in flight at a time, and later claims.
- **That is a hypothesis, not yet measured as the cause.** [experiment 64](064-concurrent-result-posts.md) tests it the way this experiment tested the slots: more result posts in flight on the pair holders, nothing else changed.
- **Production runs no longer have this path on most hops.** Handoffs are now the default (experiments 58, 59 and 61). A handed hop skips both the post and the claim, which is exactly where the penalty sits. [experiment 63](063-pairs-penalty-with-handoffs.md) asks whether pairs still lose with handoffs on.

## Limits

- **A small model on one machine,** with emulated links and no jitter or loss. Its uploads are round-trip-bound (71–87 ms), so the slot queue that L1 found on the 30.3M model (byte-bound uploads, 3.1 s per update) has no room to form here. This experiment cannot rule out slot queueing on the 30.3M model. It shows that the penalty exists without it.
- **The strong workers only announce more GFLOP/s.** Their real compute speed is the same as the others', so P2 against N2 also includes how placement spreads compute. P4 against P2 is the clean causal comparison. It changes only the slot count, and it shows no effect.
- **One captured run per arm** (6 updates of chain each). The hop split is from those captures; the timings are from the 9 timed runs.
- **Slot and post waits come from spans,** so they are wall-clock time on one machine's clock. Chain membership comes from `chain_trace`'s walk through the plan.

## Reproduce

Each run (about 50 s), with `NAME`, `PAIRS` (`always` or `never`), `SLOTS` and `STEPS`, and `YGARATY_CAPTURE_DIR` set for the captured runs:

```bash
YGARATY_PLACEMENT=capacity YGARATY_PLACEMENT_PAIRS=$PAIRS \
ygaraty experiment http_profile --recipe transformer --learning-rate 0.0005 --workers-kind python --tiles 512 --max-tasks 16 --repeats 1 \
  --relay-url http://127.0.0.1:$PORT --relay-token $TOKEN --version 1m-seed2026-v2k --embedding 128 --hidden 384 --layers 2 --heads 4 --band 16 --max-task-mib 512 \
  --workers 8 --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 4 --pushed-inputs --eager-claims --length-window 16 --batch-tokens 3072 \
  --dataset-url sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite --store-url postgresql:///ygaraty_tiny --steps $STEPS \
  --rtt-ms 50 --down-mbps 100 --up-mbps 20 --strong 2 --strong-up-mbps 100 --strong-cores 8 --strong-gflops 40 --worker-gflops 10 --strong-uploaders $SLOTS \
  --output runs/exp59b/$NAME
ygaraty experiment chain_trace --slots runs/exp59b/cap-P2-cap runs/exp59b/cap-P4-cap runs/exp59b/cap-N2-cap
python -m unittest tests.test_exp62_tools
```

The local driver is `runs/exp59b/drive.sh`. `runs/exp59b/campaign.sh` runs all 12 (about 10 minutes), and `runs/exp59b/analyze.py` evaluates the gates.
