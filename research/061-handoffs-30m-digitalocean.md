# Handoffs on the 30.3M Transformer: 22.5% shorter updates on DigitalOcean home links, now the default

| | |
|---|---|
| **Date** | 2026-11-01; R1 of [NEXT_2026-10-31.md](../../papers/uplink-bound/NEXT_2026-10-31.md) |
| **Path** | Experiment 49's pool and best setup on DigitalOcean nyc3: the 30.3M Transformer, owned layers and 8 vocabulary slices, float32 wire with bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches, capacity placement (`never`); 16 workers per job, 8 weak (c-2, 1 thread, gzip frames, links shaped to 100/20 Mb/s) and 8 strong (c-8, 4 threads, 500/100 Mb/s), 50 ms round trip, a g-8vcpu-32gb coordinator. Two arms, three interleaved repeats (ABBAAB), 22 updates each: **A** experiment 49's best, **B** A plus handoffs (experiments 58 to 60: `YGARATY_HANDOFF=1` on the coordinator and the workers) |
| **Result** | **Worth it; now the default.** Handoffs give 4.66, 4.66 and 4.59 s per update against 6.00, 5.96 and 6.01 s: **22.5% less time per update** (5.99 → 4.64 s). The slowest B beats the fastest A by 1.3 s, while repeats within an arm differ by under 0.07 s. By the nanochat KPI, 39.8 → 51.3 useful GFLOP/s: **31.9 → 24.7 years**. All six jobs give the same 22 losses to the last bit, with 0 retries and 0 rejected results. The simulator's recorded prediction was −22 to −31% ([HANDOFF-SIM.md](../../papers/uplink-bound/HANDOFF-SIM.md)); the measurement sits at its low end and matches the tiny model's −23% (experiment 59). Cost about $2.20 |
| **Evidence** | [`public/research/evidence/handoff/r1-30m-handoffs-digitalocean-2026-11-01.json`](evidence/handoff/r1-30m-handoffs-digitalocean-2026-11-01.json) (per-update seconds and losses of all six jobs, report hashes); the driver `drive_r1.py`, `analyze_r1.py`, rehearsals and per-job logs in `runs/pods-data/r1/` (local, not in git). Code: `39e5509` and `e576e2a`; `go test ./...` in relay/, `tests/test_handoff.py` |

## Question

Handoffs took the coordinator off the hop on the tiny model (experiments 58 to 60: −23% on emulated home links, one machine). Does the gain hold on the 30.3M model, on separate machines with real per-worker links and a mixed weak/strong pool, the setup of the paper's headline number?

## Two bugs found on the way

Both would have voided the run, and both are fixed and tested:

- **The pods coordinator ignored handoffs.** `distributed_translation` built its runtime policy from command-line flags and never read `YGARATY_HANDOFF`, so arm B would have quietly run without handoffs. Fixed in `39e5509`.
- **Handoffs failed on every worker with `--gzip-frames`.** Such a worker gzips its JSON requests too, and the relay's `POST /handoff/get` read the body as plain JSON: every handed task on a weak worker failed and was retried, until one ran out of attempts (the first small-model rehearsal: 241 retries by update 5, then `Task v55/loss/1 exceeded 5 attempts`). Experiments 58 to 60 ran without gzip, so they never met it. The relay now inflates a gzipped request, as batched PUTs already did (`e576e2a`; `TestHandoffAcceptsGzippedRequest` fails without the fix). A rehearsal without gzip had confirmed the cause first.

The machines cannot pull after setup (the deploy key is shredded), so they ran `39e5509` with `relay/handoff.go` of `e576e2a` copied in and the relay rebuilt; the evidence records the file's hash. The workers' Python code is the same in both commits.

## Rehearsal (small model, 6 updates)

| Arm | Seconds per update (2 to 5) | Retries | Losses |
|---|---|---:|---|
| A | 2.8, 2.8, 2.6, 2.8 | 0 | — |
| B, before the relay fix | 19.8 to 61.7 | 241 | failed at update 6 |
| B, gzip off on the weak workers | 1.8, 2.0, 1.8, 1.8 | 0 | identical to A |
| B, after the fix | 1.8, 1.8, 1.8, 1.8 | 0 | identical to A |

## Result (30.3M, 22 updates, steady window updates 2 to 21)

| Repeat | A: experiment 49's best | B: + handoffs | Change |
|---|---:|---:|---:|
| 0 | 5.995 s (sd 0.30) | 4.663 s (sd 0.29) | −22.2% |
| 1 | 5.956 s (sd 0.32) | 4.662 s (sd 0.22) | −21.7% |
| 2 | 6.008 s (sd 0.30) | 4.592 s (sd 0.22) | −23.6% |
| **Mean** | **5.99 s** | **4.64 s** | **−22.5%** |

Arm A reproduces experiment 49's gzip-on-weak setting (5.97 to 6.02 s there), so the pool is the same as before. The gate in the plan (B faster beyond the repeats' spread) passes by a factor of about 20.

## What it taught

- **The coordinator hop was a fixed cost of every update at full scale too.** Handoffs remove the same share of the update on the 30.3M model (−22.5%) as on the tiny one (−23%), although the 30.3M model's updates are 1.5 times longer and its pool is mixed.
- **The simulator ranked it right and sized it at the low end.** Its −22 to −31% band contains the measurement; the 8-strong-pool estimate of about 4.3 s was optimistic by 0.3 s.
- **Default changed.** `distributed_translation` turns handoffs on unless `YGARATY_HANDOFF=0`, and `scripts/remote.sh` exports `YGARATY_HANDOFF=1` for the coordinator and the workers. The coordinator hands tasks only to workers that ask for handoffs and only with a relay, so other setups are unaffected. `RuntimePolicy.from_env` and `http_profile` keep it off by default, as experiments 58 to 60 measured it.
- **A combination nobody ran is a combination that does not work.** Handoffs and gzip frames were each tested alone; their first meeting failed every handed task.

## Limits

One pool, one link shape, three repeats, 22 updates; the losses show identical learning, but no run trained to a held-out target. The handed-unowned limit (1) and the four handed tasks a worker holds were not varied at this size.

## Reproduce

```bash
ygaraty experiment pods up r1 --provider digitalocean --location nyc3 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 180 --max-hourly 2
ygaraty experiment pods up r1 --workers 16 --worker-size c-8 --max-hourly 2        # w8 to w15, the strong workers
ygaraty experiment pods ship r1
# server: bash /root/remote.sh bootstrap   workers: bash /root/remote.sh worker-setup   (about 3 minutes; runs/pods-data/r1/setup.sh)
SMALL=1 python runs/pods-data/r1/drive_r1.py runs/pods-data/r1/rehearsal 6 0A,0B     # rehearsal, small model
python runs/pods-data/r1/drive_r1.py runs/pods-data/r1/main 22                        # ABBAAB, about 25 minutes
python runs/pods-data/r1/analyze_r1.py runs/pods-data/r1/main
ygaraty experiment pods down r1
```

With handoffs now the default in `scripts/remote.sh`, arm A needs `YGARATY_HANDOFF=0` on the coordinator and the workers.
