# With 2 fast workers among 16, pairs and one layer each tie: a 20 Mb/s worker owning a decoder layer paces both

| | |
|---|---|
| **Date** | 2026-10-30 |
| **Path** | Experiment 42's pool on DigitalOcean (lon1, one private network): a g-8vcpu-32gb server (coordinator, relay, PostgreSQL), 14 weak workers (c-2, 1 BLAS thread, links shaped to 100/20 Mb/s) and 2 strong workers (c-8, 4 threads, 500/100 Mb/s), 50 ms round trip for all; the 30.3M Transformer translator, owned layers and 8 vocabulary slices, bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches, capacity placement. Per job only `YGARATY_PLACEMENT_PAIRS` changes: `always` (a fast worker takes two consecutive layers) or `never` (one layer each, the default since experiment 42) |
| **Result** | One layer each is 2.7% faster: 8.60 s per update (8.62, 8.58) against 8.84 s with pairs (8.88, 8.79), steady means of updates 2 to 21 of four 22-update jobs run A B B A; by the nanochat KPI 46 against 47 years. Losses are identical update by update across the four jobs. The spans show why it is a tie: in every job the slowest uploader is the 20 Mb/s worker that owns decoder layer 2 (8.7 to 9.1 s of uploads per update, the update's length), while a pair holder's uploads take 6.5 to 6.9 s on its 100 Mb/s uplink and never set the pace. The prediction on record (a tie within 5%, both near 8 s, paced by a weak decoder-layer owner) holds; the paper's placement model with the measured per-piece overhead ranked the cell right (7.16 against 7.23 s) and ran 17% under measured, the bytes-only simulator had pairs ahead by 4 to 10% and was wrong again. `never` stays the default at every strong-worker count measured (2, 4, 6, 8). |
| **Evidence** | Per job `report.json` (owners, seconds, KPI), `updates.jsonl`, `coordinator.log` in `runs/pods-data/exp53/jobs/<n>-<rule>/`; every worker's spans and logs in `runs/pods-data/exp53/w*/`; the driver `runs/pods-data/exp53/drive53.py` (all local, not in git). Code at `e702701` |

## Question

Experiment 42 measured pairs against one layer each with 4, 6 and 8 strong workers among 16 and found one layer each 7 to 16% faster every time, so `never` became the default. It never ran the pool with 2 strong workers, the cell where every offline model still prefers pairs or calls it a tie:

| Model | pairs (`always`) | one layer each (`never`) | says |
|---|---:|---:|---|
| Update simulator, experiment 42's table (`price_changes pairs`) | 7.12 s | 7.44 s | pairs by 4% |
| Update simulator, `placement_check --pool two-strong` | 11.9 s | 13.2 s | pairs by 10% |
| Paper placement model, calibration A (bytes over uplink only; `papers/uplink-bound/scripts/placement_lp.py`) | 5.45 s | 6.96 s | pairs by 22% |
| Paper placement model, calibration B (measured per-piece overhead β = 2.93 s, fitted on the 8-strong `always` job) | 7.23 s | 7.16 s | one layer each by 1%: a tie |

Calibration B is the only model that ranked experiment 42's held-out 4- and 6-strong cells right. Here it says the two placements are paced by the same stage: a 20 Mb/s worker that owns a decoder layer (10.6 MB of results per update at 2.5 MB/s, plus β). With `always` the strong workers take E0+E1 and D0+D1 and E2, D2 go to weak workers (experiment 39's 2-strong placement, 8.8 s then); with `never` they take the two heaviest layers and four layers, at least one decoder layer among them, go to weak workers. Either way a weak decoder-layer owner is in the pipeline, and the question is whether the pair stage on a 100 Mb/s uplink (measured 6.9 s per update in experiment 42's `always` jobs, 152/22) is slower than it.

## Frozen before the runs

- Pool, model, batches and links as in the table above; `YGARATY_WORKERS=16`, 22 updates per job (the first warms up, the last publishes the owners' state).
- Steady seconds per update = mean of updates 2 to 21, as in experiment 42.
- Four jobs in the order `always`, `never`, `never`, `always` (A B B A): the user asked for two, and the models' tie prediction puts the difference inside one job's noise (experiment 42's repeats of the same cell differ by 0.2 s), so each rule is run twice and the comparison is on the pair of means. The first pair alone is also reported.
- Losses are expected identical update by update between rules (placement does not change the arithmetic); the write-up checks it.
- Who held what is read from each job's `coordinator.log` (placement lines) and reported as experiment 42 did.

## Predictions, made before any run

1. **A tie within noise.** Both rules land between 7.5 and 8.5 s per update and differ by under 5%, because a weak decoder-layer owner sets the pace in both. The paper's calibration B (7.16 against 7.23 s) says this; its absolute values run 10 to 16% under measured on experiment 42's cells, so the measured values should be about 8 s.
2. **If one layer each wins by more than 7%**, as it did with 4, 6 and 8 strong workers, the pair stage is slower than a weak decoder-layer owner's stage, and β fitted on single layers understates a pair's overhead.
3. **If pairs win by more than 5%**, the hop saved by co-locating two consecutive layers (the simulator's dependency bound) is real at this pool, the default rule needs a threshold at 2 strong workers, and experiment 39's 8.8 s stands as the pairs figure.
4. The bytes-only models (the simulator, calibration A) will again overstate pairs' advantage, whatever the outcome, because they price a pair holder's doubled uploads at about half their cost (experiment 42).

## Results (2026-10-30)

One pool (lon1), four jobs in the order run; steady = mean of updates 2 to 21 (standard deviation over those 20 updates in brackets). Hello reports: strong workers 236 to 270 GFLOP/s, 102 to 113 Mb/s up measured; weak workers 20 Mb/s.

| Job | Rule | Strong workers held | Layers on 20 Mb/s workers | Steady s/update | KPI |
|---|---|---|---|---:|---:|
| 0 | always | w14: E0+E1, w15: D0+D1 | E2 (w13), D2 (w4) | 8.88 (0.62) | 47 years |
| 1 | never | w14: D0, w15: D1 | E0 (w4), E1 (w8), E2 (w10), D2 (w1) | 8.62 (0.50) | 46 years |
| 2 | never | w14: D0, w15: D1 | E0 (w8), E1 (w9), E2 (w1), D2 (w4) | 8.58 (0.44) | 46 years |
| 3 | always | w14: D0+D1, w15: E0+E1 | E2 (w11), D2 (w1) | 8.79 (0.49) | 47 years |

Means: `always` 8.84 s, `never` 8.60 s, a difference of 0.24 s (2.7%); the first pair alone gives 8.88 against 8.62 (2.9%). Repeats of the same rule agree to 0.1 s. Losses are the same in all four jobs at every update (19.2462, 9.6896, 9.5588, 7.8355, ... 7.4019 at update 22); no retries, no rejected results. Experiment 39's 8.8 s for this pairs placement reproduces (8.88, 8.79).

Where each update's time went, from the workers' spans (seconds per update, over 22 updates; "upload" is the relay store plus the result post):

| Job | Worker, piece | Upload | of which relay | Compute |
|---|---|---:|---:|---:|
| 0 always | w4, D2 (20 Mb/s) | **9.13** | 7.62 | 1.23 |
| | w13, E2 (20 Mb/s) | 7.19 | 5.85 | 1.04 |
| | w15, D0+D1 (100 Mb/s) | 6.88 | 4.68 | 1.68 |
| | w14, E0+E1 (100 Mb/s) | 6.50 | 4.32 | 1.22 |
| 1 never | w1, D2 (20 Mb/s) | **9.02** | 7.59 | 1.27 |
| | w4, E0 (20 Mb/s) | 7.73 | 6.29 | 0.85 |
| | w8, E1; w10, E2 (20 Mb/s) | 7.31; 6.98 | 5.93; 5.62 | 0.99; 1.03 |
| 2 never | w4, D2 (20 Mb/s) | **8.71** | 7.32 | 1.21 |
| | w8, E0; w9, E1; w1, E2 | 7.69; 7.45; 6.98 | 6.23; 6.03; 5.68 | 0.97; 0.99; 0.90 |
| 3 always | w1, D2 (20 Mb/s) | **8.91** | 7.50 | 1.24 |
| | w11, E2 (20 Mb/s) | 7.31 | 5.96 | 1.01 |
| | w14, D0+D1; w15, E0+E1 (100 Mb/s) | 6.85; 6.47 | 4.68; 4.28 | 1.63; 1.14 |

Vocabulary-slice owners upload 4.6 to 5.1 s per update; unowned workers less. In every job the weak worker holding D2 uploads for as long as the update lasts (8.7 to 9.1 s against 8.6 to 8.9 s per update): its stage is the pipeline's pace. The pair holders' 6.5 to 6.9 s match experiment 42's pair stages (6.4 to 7.0 s), but here they sit under the D2 owner's stage, so pairing costs nothing and one layer each gains nothing. With 4 or more strong workers every decoder layer sits on a 100 Mb/s uplink and the pair stage becomes the slowest, which is why one layer each won there by 7 to 16%.

Why `never` is still 0.24 s ahead: with pairs, the two strong workers' uploads run 6.5 to 6.9 s, close behind the D2 owner, and the update waits for whichever finishes last on a given update; with one layer each nothing else comes close to the D2 owner. The 2 to 3% is consistent across both pairs of jobs but is within what pool drift could do; it does not change the rule.

## Against the models

| Model | pairs | one layer each | predicted winner | measured |
|---|---:|---:|---|---|
| Paper placement model, calibration B (β fitted on experiment 42) | 7.23 | 7.16 | one layer each by 1% | one layer each by 2.7%: ranking right, values 18% low |
| Update simulator, experiment 42's table | 7.12 | 7.44 | pairs by 4% | wrong |
| `placement_check --pool two-strong` | 11.9 | 13.2 | pairs by 10% | wrong (and 35 to 50% high) |
| Paper placement model, calibration A (bytes only) | 5.45 | 6.96 | pairs by 22% | wrong, values 23 to 38% low |

The models that count only bytes over uplink have pairs saving a hop and nothing else; the measured per-piece overhead (about 2.9 s per owned piece per update, calibration B) is what makes a 20 Mb/s decoder owner the pacer whatever the pairing rule. Calibration B's constant under-estimate (10 to 18% across experiments 42 and 53) is in the dependency chain and the server, which it prices at round trips plus bytes only.

## What it taught

- **With 2 strong workers among 16 the pairing rule does not matter**: 8.84 against 8.60 s. The weak worker owning decoder layer 2 paces the update either way. `never` remains the default without a threshold; at 2 strong workers it is harmless, at 4 to 8 it wins.
- **The pace is set by the slowest layer owner's uplink**, and a decoder layer uploads half again the bytes of an encoder layer (it also returns the gradient with respect to the encoder memory; measured relay time 7.3 to 7.6 s against 5.6 to 6.3 s). With fewer strong workers than decoder layers, one decoder layer sits on a home uplink, and no placement rule removes that stage. The next gain on such pools is not placement but the bytes a decoder-layer owner uploads, or a third strong worker.
- **A model needs the per-piece overhead to rank placements.** The paper's calibration B is the only offline model that has ranked all four measured cells right (4, 6, 8 and now 2 strong workers); the update simulator, which prices uploads at bytes over rate, has ranked three of four wrong.

## Cost

$1.75 an hour for 17 machines (14 × c-2 at $0.0625, 2 × c-8 at $0.25, g-8vcpu-32gb at $0.375) from 23:50 to 01:30 UTC: about 1 h 40 min, under $3. The four jobs took 22 minutes; the rest was bootstrap (the dataset's download, tokenizer and import, 13 minutes) and a first driver attempt whose launching ssh session hung on the server while the coordinator ran without workers.

## Limits

Four jobs on one pool and one evening, two per rule, so a 2 to 3% difference is suggestive and not more; the tie within 5% is the finding. The strong workers' uplinks were shaped to 100 Mb/s with 4 BLAS threads, as in experiments 39 and 42; whether one strong worker with a faster uplink would own two layers profitably was not asked. Emulated links (no jitter or loss).

## Reproduce

```bash
python -m ygaraty.experiments.pods up exp53 --provider digitalocean --location lon1 --server g-8vcpu-32gb --workers 14 --worker-size c-2 --minutes 150 --max-hourly 2
python -m ygaraty.experiments.pods up exp53 --workers 16 --worker-size c-8 --max-hourly 2     # w14, w15: the strong workers
python -m ygaraty.experiments.pods ship exp53
# scripts/remote.sh bootstrap on the server, worker-setup on each worker; then per job, on the server:
#   YGARATY_PLACEMENT=capacity YGARATY_PLACEMENT_PAIRS=always|never YGARATY_BATCH=64 YGARATY_LENGTH_WINDOW=64 YGARATY_BATCH_TOKENS=3072 YGARATY_WORKERS=16 ./scripts/remote.sh serve34 22
# weak workers w0..w13:    ./scripts/remote.sh worker <coordinator> <relay> <job> <token> wN
# strong workers w14, w15: YGARATY_UP_MBPS=100 YGARATY_DOWN_MBPS=500 YGARATY_WORKER_THREADS=4 ./scripts/remote.sh worker ...
# (the driver runs/pods-data/exp53/drive53.py does this for the four jobs, passing the token from job.json without printing it)
python -m ygaraty.experiments.pods fetch exp53 && python -m ygaraty.experiments.pods down exp53
```

The pool costs $1.75 an hour (14 × c-2 at $0.0625, 2 × c-8 at $0.25, the server at $0.375).
