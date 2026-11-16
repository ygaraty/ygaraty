# Dropping the tasks nobody reads: 24% fewer tasks, 4% less time per update on the tiny Transformer, on loopback and on home links alike

| | |
|---|---|
| **Date** | 2026-10-30 |
| **Path** | Workstation: experiment 51's recipe (the 1M Transformer, `1m-seed2026-v2k`, 8 NumPy workers, relay, float32 wire, bf16 activations, owned layers, 4 vocabulary shards, pushed inputs, eager claims, sorted 3,072-token batches, 12 updates), once on loopback and once with every worker behind an emulated 100/20 Mb/s link at 50 ms; `YGARATY_PRUNE_PLAN=0` against `1`, two repeats in alternating order |
| **Result** | Pruning removes 35 of the job's 143 tasks per update (24%) and 1.5 of the 10.5 MB of results stored on the relay per update (14%); losses are identical update by update in all eight runs. Time per update (mean of updates 2 to 12, two repeats each): loopback 0.37 → 0.35 s (3.7%), home links 3.99 → 3.83 s (3.9%; 3.99/3.84 and 3.98/3.83 in the two repeats). The prediction of 10 to 25% on home links was wrong: it priced the removed results at their float64 size (45 MB per update); on the wire they are 1.5 MB, bf16 and float32 blocks of small assembles, 0.19 MB per worker, under 0.1 s of a 20 Mb/s uplink. The gain is the same few percent on both links, so it is claims and kernel time, not bytes. |
| **Evidence** | `runs/exp54/r{1,2}-{loopback,home}-prune{0,1}.log` and `*/profile.json`, `drive54.sh`, `analyze54.py` (local, not in git). Code: `compiler.plan.prune_unused`, `Tape.backward(input_gradients=)`, `YGARATY_PRUNE_PLAN`; `tests/test_plan_prune.py` |

## Question

The dead-task audits of 2026-10-30 (CSE-81.md, LOCAL-VALIDATION-RESULTS.md) found that every compiled
plan carries tasks whose results nothing reads: the whole-matrix `assemble` the tiled planner emits for
each named intermediate, and in the MLP and factorization tapes the gradient of an untrained input.
Pruning them is exact (bytewise identical roots on the factorization, 200 comparisons) and a third
faster on a sequential executor, where arithmetic is all there is. On the distributed runtime the
question is different: these tasks sit at the end of their chains with no consumer, so they are never
on the critical path. What they cost is claims, kernel time on a worker and, above all, result bytes
through the relay on a 20 Mb/s uplink. Does removing them shorten the update?

## What changes

Two compiler additions, both off by default (pinned plan hashes in the RL corpus and the CSE
experiment depend on the unchanged graphs):

- `compiler.plan.prune_unused(tasks, keep)`: the tasks reaching a kept value (the update's
  parameter updates and loss), owned tasks and tasks of unreviewed operations are kept with
  everything they depend on; the rest go. `YGARATY_PRUNE_PLAN=1` applies it in the translation job
  and the profile job.
- `Tape.backward(loss, input_gradients=False)`: the reverse pass drops the gradient of leaf inputs
  that are not parameters. Not exercised here: the Transformer's inputs are token ids with no
  gradient.

On the tiny Transformer's unowned plan pruning removes 34 of 733 tasks (26 `assemble`, 8
`place_block`), 45 MB of float64 results per update (5% of all result bytes). The owned plan the job
runs differs slightly; the job's own count is reported.

## Prediction, made before any run

- Losses are identical update by update with and without pruning (values cannot change).
- Loopback: under 5% difference; the removed tasks are cheap copies and the update is bound by
  coordinator and byte handling (experiment 47, 51).
- Home links: 10 to 25% less time per update. 45 MB per update over 8 workers is 5.6 MB per worker,
  2.2 s of a 20 Mb/s uplink; experiment 51 put the update at 4.0 s with bytes about 60% of a hop.
  If the gain is under 5% on home links, the removed results were mostly not uploaded (consumed on
  the producing worker or never fetched), and the saving is kernel time only.

## Results (2026-10-30)

Eight 12-update runs, order r1: off, on; r2: on, off per link. Steady = mean of updates 2 to 12.

| Run | Link | Prune | Tasks/update | Steady s/update | Relay results MB/update |
|---|---|---|---:|---:|---:|
| r1 | loopback | off | 143 | 0.37 | 10.5 |
| r1 | loopback | on | 108 | 0.35 | 9.0 |
| r2 | loopback | off | 143 | 0.36 | 10.5 |
| r2 | loopback | on | 108 | 0.36 | 9.0 |
| r1 | home | off | 143 | 3.99 | 10.5 |
| r1 | home | on | 108 | 3.84 | 9.0 |
| r2 | home | off | 143 | 3.98 | 10.5 |
| r2 | home | on | 108 | 3.83 | 9.0 |

Losses: identical at every update across all eight runs. Means: loopback 0.37 → 0.35 s (3.7%), home 3.99 → 3.83 s (3.9%). The home-link difference of 0.15 s repeats exactly in both pairs but sits inside one update's spread (standard deviation about 1.0 s over 11 updates), so it is a consistent small effect, not a precisely measured one.

**Why the prediction missed.** The removed tasks are the planner's whole-matrix `assemble` and `place_block` of unread intermediates. Their float64 size (45 MB per update on the unowned plan) is what the prediction charged to the uplinks. In the job they travel as bf16 activations or float32 blocks, many are 1 × 1 or a few KB, and the relay shows only 1.5 MB per update less with pruning: 0.19 MB per worker, 0.08 s of a 20 Mb/s uplink. What pruning does save is 35 claims, 35 small kernels and 35 result posts per update; on this plan that is worth about 4% whatever the link.

**A bug found on the way.** The first attempt ran both arms at 143 tasks: the profile's translation job builds its `RuntimePolicy` from explicit arguments, so the environment flag never reached it. Fixed (`prune_plan=` passed through in `http_profile.lstm_job`); the task count per update is the check that the flag is live.

## What it taught

- **Dead tasks cost claims, not bytes, on the Transformer.** 24% of the tiny job's tasks per update are unread assembles, but they carry 14% of the result bytes and none of the critical path, so removing them is worth about 4% on loopback and on home links alike. The wire format already made them cheap.
- **Pruning is safe and exact** at the runtime: identical losses in all eight runs, and bytewise identical roots on the factorization (LOCAL-VALIDATION-RESULTS.md). It can be on by default once the pinned-hash consumers (RL corpus, CSE-81) are rebuilt or exempted; until then `YGARATY_PRUNE_PLAN=1`.
- **The large wins are elsewhere.** Where a plan computes the gradient of an untrained input (MLP, factorization) the dead work is real arithmetic, 12 to 20% of tasks and a third of sequential time; `Tape.backward(input_gradients=False)` removes it at the source. The Transformer has no such input.

## Limits

One machine, 8 workers, 12 updates per run, two repeats; home links emulated (no jitter or loss). The tiny model's update is latency-bound (0.35 s on loopback for 108 tasks), so per-task overheads dominate and a 4% task-count effect is about what the removed 24% of tasks can give off the critical path. On the 30.3M plan the unread assembles are 1.5% of tasks and 9% of float64 result bytes; the expected gain there is smaller still.

## Reproduce

```bash
# runs/exp54/drive54.sh (copy of the scratch driver): r{1,2}-{loopback,home}-prune{0,1}, 12 updates each
YGARATY_PRUNE_PLAN=1 ygaraty experiment http_profile --recipe transformer ... --rtt-ms 50 --down-mbps 100 --up-mbps 20   # see the driver for the flags
python -m unittest tests.test_plan_prune
```
