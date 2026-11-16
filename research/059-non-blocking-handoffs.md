# Non-blocking handoffs: late handoffs were workers stuck on an earlier one; removing that gives another 6%

| | |
|---|---|
| **Date** | 2026-11-01 (follow-ups 1 and 2 of [experiment 58](058-coordinator-off-the-hop.md)) |
| **Path** | Experiment 58's setup: the tiny Transformer, 8 NumPy workers, emulated 100/20 Mb/s, 50 ms home links, 12 updates, workstation, no cloud |
| **Result** | Yes. Median seconds per update (updates 2 to 10, 3 interleaved runs per arm): no handoffs 3.933, experiment 58's handoffs 3.205, **non-blocking handoffs 3.018** (−5.8% against experiment 58's loop, −23.3% against none). The arms' run ranges do not overlap. Losses are bit-identical in all 9 runs, with 0 retries and 0 rejected. Late handoffs on the traced chain fell from 42 to 9 of 90, and the chain from 3.10 to 2.74 s |
| **Evidence** | `runs/exp59/r{1,2,3}-home-{h0,hb,hn}/profile.json`, capture `runs/exp59/cap-home-hn-cap/` and experiment 58's captures (local, not in git) |

## Question

In experiment 58's trace, 42 of the 90 handed hops on the critical chain were late: the consumer's relay read went out only after the producer's result had landed. Those hops cost 119 ms after the upload instead of 22. Was the coordinator handing tasks out late (follow-up 1), or was the worker unable to claim (follow-up 2)?

## Diagnosis

The question was what the consumer was doing between its producer's lease and its own claim. In experiment 58's capture (`cap-home-h1-cap`, updates 2 to 7), a late consumer started its claim about 157 ms after the producer was leased (median). In 41 late hops with a claim after the lease, the window averaged 213 ms:

| Consumer's main thread, mean ms per late hop | |
|---|---|
| **Waiting on an earlier handed task's inputs** | **155** |
| Its previous claim request | 35 |
| Kernels | 20 |
| Input assembly | 1 |

Once it claimed, it was handed the task at once. Experiment 58's worker ran a reply's tasks in order and blocked on a handed task's inputs, so a worker holding one handed task could not claim the next, even when that next task was on the chain. The coordinator was not the problem: `claim_handoff` takes 0.06 ms per call (median). So follow-up 1's fix is follow-up 2.

## What changed

With `--handoff`, the worker now claims on a thread of its own (`WorkerExecution.run_handoff`).
- The main thread waits for whichever comes first: a claim reply, or a handed task's relay read.
- It runs a reply's ready tasks at once, and a handed task as soon as its inputs arrive.
- Claims keep going out while handed tasks wait. A worker still holds at most 4 handed tasks (`Handoff.HANDOFF_MAX`).
- `YGARATY_HANDOFF_BLOCKING=1` keeps experiment 58's loop, as the control arm here.

Nothing changes on the coordinator or the relay.

## Design

Three arms, 3 runs each on home links. The order rotated across repeats (h0 hb hn, then hn hb h0, then hb hn h0), and each run started only on an idle machine (1-minute load under 3):
- **h0:** no handoffs;
- **hb:** experiment 58's blocking handoffs;
- **hn:** non-blocking handoffs.

One captured hn run (8 updates) followed, for the chain trace.

## Results

| Arm | Run 1 | Run 2 | Run 3 | **Median** | Tasks handed off per run |
|---|---|---|---|---|---|
| h0 | 3.914 | 3.933 | 3.944 | **3.933** | 0 |
| hb (experiment 58) | 3.205 | 3.190 | 3.249 | **3.205** (−18.5%) | 1,178–1,187 |
| hn | 3.000 | 3.018 | 3.069 | **3.018** (−23.3%) | 1,217–1,225 |

hb reproduces experiment 58 (3.18 s then). All nine runs give the same twelve losses, with 0 retries and 0 rejections.

`chain_trace` on the captured runs:

| | Experiment 58 (blocking) | Non-blocking |
|---|---|---|
| Chain (15 hops) | 3,103 ms | 2,744 ms |
| Median hop | 188 ms | 171 ms |
| Upload | 98 ms | 103 ms |
| Upload end to the consumer's kernel | 73 ms | 42 ms |
| Late handoffs on the chain | 42 of 90 | 9 of 90 |
| Reply after the upload, consumer waiting in time | 22 ms | 32 ms |

Six of the nine remaining late hops are consumers whose claim was already in flight, held open by the coordinator, when the producer was leased. The coordinator wakes held-open claims only on commits. Waking them on leases was tried in experiment 58, before this change, and gave nothing measurable; it now concerns about 10% of the chain's hops.

## What it means

- **The late handoffs were the worker's doing, and a non-blocking loop removes most of them.** The gain, −5.8% per update, is beyond the spread of either arm.
- **Handoffs now take 23% off the update** on the tiny model's home links, still with bit-identical losses.
- **What is left of a hop is almost only the upload.** The median is 171 ms, of which the upload is about 100 ms and the time from upload end to the next kernel 42 ms. The rest of the hop is the uplink, as the paper says.

## Limits

- Experiment 58's limits all apply: emulation, the tiny model, one machine, one capture per arm.
- The reply arrives 32 ms after the upload instead of 22 when the consumer waits in time. With more handed tasks in flight, the 100 Mb/s downlink carries more at once. Not separately measured.
- A worker can now hold up to 4 handed tasks while it computes others; that limit was not varied.

## Reproduce

As in [experiment 58](058-coordinator-off-the-hop.md), with `--handoff`. Add `YGARATY_HANDOFF_BLOCKING=1` for the blocking arm. The local driver is `runs/exp59/campaign.sh`, and `runs/exp59/precise.py` reads the results; the campaign takes about 7 minutes.
