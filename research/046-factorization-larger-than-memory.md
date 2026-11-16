# A factorization larger than any machine's memory

| | |
|---|---|
| **Date** | 2026-10-28 (timing ladder, workstation); 2026-10-29 (11-update training run, DigitalOcean) |
| **Path** | Sharded 2D factorization over HTTP on loopback (`ygaraty/experiments/factorization_http.py --generated-x`): NumPy workers, one pinned core each, under a `systemd-run` memory cap; workers generate their own X tiles from the seeded spec, so X is never stored or sent. Ladder: 12 workers on the workstation. Training run: 14 workers on one DigitalOcean c-16 (16 dedicated vCPUs, 32 GB, sfo2) |
| **Result** | 11 updates of a 122,880 x 122,880 factorization (112.5 GiB of float64 X) run correctly at 205 s per update on 14 cores, 0 retries, and the loss falls at every update (79.13 -> 0.503), but X is not fitted: the fall is the regularization term (78.64 at the start) shrinking U and V towards zero, and 0.503 is the loss of U V^T = 0 (mean X^2 / 2). The learning rate and regularization were wrong for training; corrected 2026-10-29, see "Correction" below. Peak RSS 1.58 GiB in the coordinator and 1.34 GiB in the largest worker (under 8 GiB and under 1/4 of X); the 14 workers together reach about 19 GiB because each fills its 1 GiB tile cache, so the run's cap had to be 28 GiB, not 8. 8,192 matches the dense reference exactly over 3 updates; `audit_locality` finds 0 violations at full size. The coordinator's peak does grow with the matrix (389 MiB at 30,720, 1.58 GiB at 122,880), so that criterion is not met as written |
| **Evidence** | Workstation ladder: `runs/factorization-http/generated-{8192,30720,30720b,61440,122880}/report.json`. DigitalOcean: `runs/pods-data/fact46/server/runs/factorization-http/do46/` (both reports, logs, `rss.tsv` per-process RSS every 30 s) and `runs/pods-data/fact46/server/run46.sh` (local, not in git) |

## Question

Can the sharded plan train a factorization whose X alone is 4 times this workstation's RAM (28 GiB), with every process staying under a hard memory cap far below X's size, and the same loss curve a small-scale reference gives?

## Timing ladder (done)

The streaming executor is not a separate executor: `build_sharded_plan(..., generated_x=True)` gives tile tasks only U_i and V_j and an `x_spec` attribute, and the kernel (`factorization_kernels._generated_x`, bounded and range-checked) builds the tile on the worker. No X leaf exists; the coordinator holds only the 2 x 60 factor shards. `audit_locality` also checks that a tile generates its own X block.

| Side | Tiles | Update (12 cores, plain loopback) | Sent / received | Coordinator peak RSS |
|---:|---:|---:|---:|---:|
| 8,192 | 16 | 2.1 s (4 cores) | 245 / 192 MiB for 3 updates | — |
| 30,720 | 225 | 30.0 s | 1,694 / 1,260 MiB for 2 | 389 MiB |
| 61,440 | 900 | 119.8 s | 5,650 / 4,562 MiB for 2 | — |
| 122,880 | 3,600 | 483 s (one update) | 9,963 / 8,643 MiB | 1,551 MiB |

- 0.133 s of wall time per tile at every size, against about 0.29 s of single-core work per tile (0.24 s generating X, 0.05 s in the kernel): 12 workers are about 2.2 times faster than one core, not 12 times. The coordinator, moving about 2 MiB of result per tile through Python, is the likely limit (as in experiment 47); this was not profiled.
- A first attempt at the full size was killed by the 8 GiB cap after five minutes: the scheduler's ready queue is first-in-first-out and all 3,600 tiles are ready at the start, so the reductions that free tile results queued behind every tile and the coordinator held about 7 GiB of results. `Scheduler.consumers_first` (opt-in, set by `ShardedFactorizationJob`) runs a task that has just become ready before waiting ones; the coordinator's peak is then 1.55 GiB. The cap stopped the run without touching the session.
- Not done here (the 11-update run below does the first): two or more updates at the full size (about 16 minutes), home links at this scale (7 GiB of results per update would take about 50 minutes per update on one 20 Mb/s link), and a streaming-memory check across a different tile size.

## Training run on DigitalOcean (2026-10-29)

One c-16 Droplet (sfo2, $0.50/h, about 1 h used) ran the plan's two checks back to back: 3 updates at 8,192 with `--check` (4 workers), then 11 updates at 122,880 (14 workers; 15 were refused, as the coordinator and shaper need 2 of the 16 CPUs). Same problem as the ladder: rank 64, blocks 2,048, learning rate 500, regularization 0.001, seed 2026, `--cache-mib 1024`, commit `f2b0fec`.

| Update | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Loss before | 79.126 | 20.155 | 5.415 | 1.731 | 0.810 | 0.579 | 0.522 | 0.507 | 0.504 | 0.503 | 0.503 |
| Seconds | 214.8 | 205.8 | 205.6 | 205.7 | 205.5 | 205.8 | 205.7 | 205.9 | 205.6 | 204.8 | 205.0 |

- **Dense check:** at 8,192 the losses (5.7411 -> 1.8050 -> 0.8245) equal the dense NumPy reference exactly, not only to 1e-9.
- **Time:** 0.057 s of wall time per tile with 14 cores, against 0.133 s with 12 on the workstation, so the update takes 205 s, not 483. The ladder's guess that the coordinator caps the rate near 0.13 s per tile does not hold on this machine; why it is 2.3 times faster was not profiled. Each update moves about 9.95 GiB of results out and 8.4 GiB in over loopback, as on the workstation. The share of time spent regenerating X was not measured separately (the ladder's single-core estimate is 0.24 of 0.29 s per tile).
- **Memory:** per process (the report's `peak_rss_mib`, confirmed by `rss.tsv`): coordinator 1,619 MiB, largest worker 1,368 MiB. Workers grow for about 20 minutes, then level at about 1.2 GiB: the 1 GiB tile cache filling (64,871 cache hits). The workstation ladder's 0.47 GiB per worker was taken after one update, before the cache had filled.
- **The cap:** the cap applies to the whole scope (coordinator plus every worker), not to each process. At 8 GiB the first full-size attempt was killed after 10 minutes, before its first update (coordinator at 1.38 GiB). The relaunch started at 12 GiB, and the running scope's cap was raised to 20 and then to 28 GiB (`systemctl set-property --runtime`) as the caches filled, peaking at about 19 GiB in all. No process came near 8 GiB. The workstation run fit under 8 GiB only because it stopped after one update with 12 workers. Keeping 14 workers under 8 GiB in all would need a smaller `--cache-mib` (untested).
- **Locality:** `audit_locality` on the full-size generated-X plan: 0 violations; the widest read is 8 MiB (an 8-way `add_many`), the tile task's 2 MiB; 5,696 tasks per update over 3,600 tiles.

Against the success criteria:

| Criterion | Result |
|---|---|
| Peak RSS of every process under 8 GiB and under 1/4 of X (28 GiB) | Met: 1.58 GiB at most |
| Peak RSS does not grow with the matrix across the ladder | Not met: the coordinator went from 389 MiB at 30,720 to 1.58 GiB at 122,880 (X 16 times larger). U and V together are only 120 MiB at full size, so the rest is results in flight; growth is far slower than X's, but it is growth |
| Loss falls over at least 10 updates at full size | Met as written, but empty: the fall is the regularization term decaying, not X being fitted (see Correction) |
| 8,192 matches the dense reference to 1e-9 | Met: equal |
| `audit_locality(plan)['violations'] == []` | Met |
| Time per update and the X-regeneration share | Time reported; share estimated from the ladder only |

## Correction (2026-10-29): the loss fell without X being fitted

Experiment 48's protein runs split the loss into its two terms. With learning rate 500 and regularization 0.001, as here, each update multiplies U and V by about (1 - 500 x 0.001) = 0.5 while the data gradient, divided by the matrix's N^2 entries, is far too small to grow them. So the loss falls from (regularization 78.64 + data 0.48) to the data loss of U V^T = 0, which is mean(X^2) / 2 = 0.503 for this X. Every number above stands as engineering (time, memory, bytes, agreement with the dense reference), but no update fitted X.

A learning rate that fits needs the scale of X: the largest stable step is about N^2 / lambda_max(X) (gradients are divided by N^2), and regularization 0.001 times such a rate is far above 1, so it must be near 0. On experiment 48's protein similarity at 8,192 (lambda_max 5,078), rate 0.75 N^2 / lambda_max with no regularization explains 94% of ||X||^2 in 50 updates; rates of 2 N^2 / lambda_max and more diverge. This synthetic X was not rerun with such settings.

## Needs first

- The sharded test file passes (done 2026-10-28: 18 tests; the size guard no longer builds a billion-entry list before refusing, which had exhausted RAM and swap and made systemd-oomd kill the terminal).
- ~~An executor that never holds X~~ (done: `generated_x`, see the ladder).

## Plan

- Problem: 122,880 x 122,880, rank 64, blocks 2,048 x 2,048 (60 x 60 = 3,600 tiles), SGD with the default regularization, the plan's default seed.
- Run every command under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`, so an overrun is killed instead of taking the session down. Record peak RSS per process.
- Check scale first on a ladder (8,192, 30,720, 61,440 on a side) before the full size; stop and report if peak RSS grows with X.
- Correctness: at 8,192 x 8,192 compare three updates with the dense NumPy reference to 1e-9 relative, as the sharded tests do at toy size.
- Then 2 and 4 HTTP workers with 4 GiB each, to show the cap holds per worker and what moving the partial gradients costs per update (tile returns about (b_r + b_c + 1) x rank elements).

## Success criteria (fixed before running)

- Peak RSS of every process stays under 8 GiB and below one quarter of X's size, and does not grow with the matrix across the ladder.
- Loss falls over at least 10 updates at full size; the 8,192 case matches the dense reference to 1e-9.
- No task or leaf larger than its tile; `audit_locality(plan)['violations'] == []`.
- Report time per update and the share spent regenerating X tiles. Do not claim a speedup.

## Limits

X is synthetic and regenerated from a seed rather than read from disk or the network, so this shows memory capacity of the plan, not I/O at that scale. One seed.

## Reproduce

On a 16-CPU machine with at least 24 GB (the DigitalOcean run used `ygaraty experiment pods up fact46 --provider digitalocean --location sfo2 --server c-16 --workers 0 --minutes 180`):

```bash
systemd-run --scope -p MemoryMax=8G -p MemorySwapMax=0 \
  ygaraty experiment factorization_http --rows 8192 --columns 8192 --rank 64 --block 2048 \
  --updates 3 --cores 4 --links loopback --generated-x --check --output runs/factorization-http/do46/generated-8192
systemd-run --scope -p MemoryMax=28G -p MemorySwapMax=0 \
  ygaraty experiment factorization_http --rows 122880 --columns 122880 --rank 64 --block 2048 \
  --updates 11 --cores 14 --links loopback --generated-x --timeout 9000 --output runs/factorization-http/do46/generated-122880
```

The GitHub repository is private, so the Droplet got the code as a git bundle over SSH (`git bundle create main.bundle main`, then `git clone main.bundle /workspace/ygaraty`); `scripts/remote.sh` cannot clone it anonymously.
