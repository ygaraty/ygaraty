# One layer per fast worker beats pairs on real links: 7 to 16% faster, where the simulator said pairs pay

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | Experiment 39's setup (the 30.3M Transformer, owned layers and 8 vocabulary slices, bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches, capacity placement) on DigitalOcean (lon1, one private network): 16 workers per job, weak (c-2, 1 BLAS thread, links shaped to 100/20 Mb/s) and strong (c-8, 4 threads, 500/100 Mb/s), 50 ms round trip for all; the pair rule chosen per job with `YGARATY_PLACEMENT_PAIRS` (commit b839f09) |
| **Result** | Experiment 39's rule gives a fast worker two consecutive layers whenever it can finish them within a typical worker's time for one. Giving each fast worker one layer instead is faster on every pool measured: 7.96 → 7.44 s per update with 4 strong workers among 16, 7.18 → 6.70 s with 6, 7.35 → 6.18 s with 8 (7, 7 and 16%; 43 → 40, 39 → 36 and 39 → 33 years by the nanochat KPI). Losses are identical. The simulator had priced pairs as better with 2 to 5 strong workers, a tie at 6 and spreading 8% better only at 7 or more, so the rule committed for this run (`short`) still pairs at 6 and fewer; one layer each (`never`) is now the default (commit 20f439e). Spans show why: a pair holder spends about twice as long uploading as a single-layer owner, so its stage sets the pace, and the simulator models uploads as about half as costly as they are. |
| **Evidence** | Server reports, worker logs and spans in `runs/pods-data/exp42d/` (not synced) |

## Question

Experiment 39 saw capacity placement leave 2 of 6 strong workers with no layer, because pairs of consecutive layers went to the 2 strongest. Is one layer per strong worker faster, and does the simulator know when?

## Simulated first

`ygaraty experiment price_changes pairs` on experiment 39's measured pools (strong: 8 cores, 190 GFLOP/s, 103/380 Mb/s; weak: 2 cores, 64 GFLOP/s, 20/94 Mb/s), means over 4 hash seeds × 3 simulator seeds × 8 batches (spread ±0.1 s):

| Strong of 16 | always (experiment 39's rule) | never (one layer each) | short (pairs unless fast workers outnumber layers) |
|---:|---:|---:|---:|
| 2 | 7.12 s | 7.44 s | 7.12 s |
| 4 | 6.20 s | 6.50 s | 6.20 s |
| 5 | 5.86 s | 6.18 s | 5.86 s |
| 6 | 5.82 s | 5.76 s | 5.82 s |
| 7 | 5.74 s | 5.34 s | 5.34 s |
| 8 | 5.73 s | 5.29 s | 5.29 s |

Past 8 strong workers the rules agree: the median worker is fast, so no pair passes the test. On this pricing the `short` rule went into commit b839f09 as the default.

## Measured

One set of machines (a g-8vcpu-32gb server, 10 c-2 and 8 c-8 Droplets), six jobs of 22 updates in alternating order; the 4-strong pool ran 2 of the c-8 Droplets as weak workers (weak links, 1 thread). Steady means of updates 2 to 21:

| Strong of 16 | Rule | s/update | Real tokens/s | nanochat | Simulated | Measured / simulated |
|---:|---|---:|---:|---:|---:|---:|
| 8 | always | 7.35 | 355 | 39 years | 5.73 | 1.28 |
| 8 | **short (one layer each here)** | **6.18** | **422** | **33 years** | 5.29 | 1.17 |
| 6 | always | 7.18 | 363 | 39 years | 5.82 | 1.23 |
| 6 | **never** | **6.70** | 389 | 36 years | 5.76 | 1.16 |
| 4 | always | 7.96 | 328 | 43 years | 6.20 | 1.28 |
| 4 | **never** | **7.44** | 351 | 40 years | 6.50 | 1.14 |

The 6-strong "always" job repeats experiment 39's best run (7.3 s): 7.18 s. With pairs, 8 strong workers are no faster than 6 (7.35 and 7.18 s); without them, they are 8% faster (6.70 → 6.18 s).

Who held what:

| Job | Strong workers | Weak workers |
|---|---|---|
| 8, always | E0+E1, D0+D1, E2, D2, V0 to V3 | V4 to V7 |
| 8, short | E0, E1, E2, D0, D1, D2, V0, V1 | V2 to V7 |
| 6, always | E0+E1, D0+D1, E2, D2, V0, V1 | V2 to V7 |
| 6, never | E0, E1, E2, D0, D1, D2 | V0 to V7 |
| 4, always | E0+E1, D0+D1, E2, D2 | V0 to V7 |
| 4, never | E0, D0, D1, D2 | E1, E2, V0 to V7 |

With 4 strong workers, one layer each puts two encoder layers on 20 Mb/s workers, and it is still faster than two pairs on 100 Mb/s ones.

## Why pairs cost more than simulated

Over each run's 22 updates, from the workers' spans:

| Job | Worker | Pieces | Uploading (relay + post) | Compute |
|---|---|---|---:|---:|
| 8, always | w11 | D0+D1 | 152 s | 38 s |
| 8, always | w12 | E0+E1 | 141 s | 30 s |
| 8, always | w15, w17 | D2; E2 | 82, 77 s | 20, 16 s |
| 4, always | w11 | D0+D1 | 153 s | 38 s |
| 4, always | w13 | E0+E1 | 141 s | 23 s |
| 4, always | w12, w10 | D2; E2 | 83, 81 s | 22, 15 s |

A pair holder uploads both layers' results through one uplink, so its uploads take twice as long, and its stage of the pipeline becomes the slowest; compute is a small part. The simulator doubles a pair holder's uploads too, but prices them at about 2.5 s an update (1.2 to 1.6 s for a single layer), where the measured runs average 6.4 to 7.0 s an update (3.5 to 3.8 s), the first and the publishing updates included. With uploads that cheap, the doubled stage did not set the pace in the simulation. Its error is larger with pairs (measured 23 to 28% slower than simulated) than without (14 to 17%). Upload time on these links is what the cost model most needs refitting on.

## The measured GFLOP/s were noise

The same c-8 Droplet reported very different speeds in its hello from job to job (w10: 211, 207, 114, 215, 117 and 217 GFLOP/s), but ran the same kernels at the same speed (w10's layer backward: 57 ms per task at a reported 117 GFLOP/s, 57 ms at 217; w12's: 91 ms at 111, 88 ms at 220). The hello's 0.3 s matmul probe does not measure these machines' speed reliably. Placement still put the layers on strong workers, since every strong worker passed the tests by a wide margin, but a ranking among strong workers by that number means little.

## What it taught

- **On real links, one layer per fast worker beats pairs** at 4, 6 and 8 fast workers among 16. A pair puts two stages' uploads on one uplink, and the update waits for it. `never` is now the default (commit 20f439e); 2 fast workers among 16 was left unmeasured here, where the simulator still prefers pairs (11.9 against 13.2 s in `placement_check`); experiment 53 measured it: a tie (8.84 against 8.60 s), because a 20 Mb/s worker owning a decoder layer paces both placements.
- **The simulator ranks placement rules wrongly when uploads decide them.** It underprices uploads by about half, so it misses when doubling one worker's uploads sets the pace. It ranked pools right (more strong workers, faster), but this choice it got backwards for 4 and 6 strong workers. A rule chosen by the simulator needs a real A/B, as here.
- **Capacity placement with one layer per fast worker is the fastest setup so far**: 6.18 s per update and 33 years with 8 fast uplinks among 16, against experiment 39's 7.3 s and 39 years with 6.

## Limits

One job per setting, 20 steady updates each; the four rules were not run in repeated, interleaved order, though the "always" jobs reproduce each other and experiment 39 (7.18 to 7.35 s against 7.3). 2 strong workers among 16 were not measured with one layer each (experiment 53 did: a tie). The strong workers combine faster links and more compute, as in experiment 39. Emulated links (no jitter or loss).

## Reproduce

```bash
ygaraty experiment price_changes pairs                  # the simulated table, about 20 s per seed
ygaraty experiment pods up exp42d --provider digitalocean --location lon1 --server g-8vcpu-32gb --workers 10 --worker-size c-2 --minutes 120
ygaraty experiment pods up exp42d --workers 18 --worker-size c-8 --max-hourly 2     # w10 to w17, the strong workers
# scripts/remote.sh bootstrap on the server, worker-setup on each worker; then per job, on the server:
#   YGARATY_PLACEMENT=capacity YGARATY_PLACEMENT_PAIRS=always|short|never YGARATY_BATCH=64 YGARATY_LENGTH_WINDOW=64 YGARATY_BATCH_TOKENS=3072 YGARATY_WORKERS=16 ./scripts/remote.sh serve34 22
# weak workers:   ./scripts/remote.sh worker <coordinator> <relay> <job> <token> wN
# strong workers: YGARATY_UP_MBPS=100 YGARATY_DOWN_MBPS=500 YGARATY_WORKER_THREADS=4 ./scripts/remote.sh worker ...
# wait until the previous job's coordinator has exited before starting the next (port 8000)
ygaraty experiment pods fetch exp42d && ygaraty experiment pods down exp42d
```

The machines cost $3.00 an hour; the run took about 100 minutes in all, about $5.
