# Corrected handoff timing gets experiment 59 right but still overstates region reads: 4.3 points off on experiment 60

| | |
|---|---|
| **Conclusion** | **Failed gate:** corrected handoff timing passes experiment 59 (1.2 points) but misses 60 by 4.3; 61 not evaluated (simulated). |
| **Status** | executed |
| **ID** | OFF-5 → experiment 71 |
| **Category** | simulation |
| **Where** | this machine (the workstation, which holds the inputs) |
| **Date** | Added 2026-11-01; run 2026-11-03 |
| **Path** | Offline replay of the measured runs' own batches in sequence: corrected non-blocking `HandoffSimulation` against ordinary `Simulation`, L5's hop constants, experiment 54's fitted kernels for the tiny runs and experiment 61's reported capacities for the 30.3M pool |
| **Result** | **Not accepted.** Change in seconds per update, simulated against measured: 59 h0→hn −24.4% against −23.3% (**1.2 points, passes**), 60 h0→r −28.4% against −24.1% (**4.3 points, fails**), 61 A→B −29.1% against −22.5% (6.6 points, **not evaluated**: gzip isn't modelled and the pods' tokenizer isn't here). The simulator predicts a 0.15 s gain from region reads that experiment 60 measured as none. Free, a few minutes of CPU |
| **Evidence** | Simulated. [`public/research/evidence/handoff/off5-handoff-timing-holdouts-2026-11-03.json`](evidence/handoff/off5-handoff-timing-holdouts-2026-11-03.json) (cases, per-seed values, anchor, batch checks, hashes); raw replay, plans and cost model in `runs/experiments/OFF-5/omarchy-20261103-103426/` (local, not in git); `tests/test_simulation_handoff.py`. Code at da12f31 plus this experiment's changes |

## Question

Do the corrected upload-landing, request-ordering and lease-clock semantics improve handoff timing predictions on held-out experiments 59–61 without refitting?

Related: the [handoff implementation](simulation-references/HANDOFF.md), the [L5 protocol](../../papers/uplink-bound/L5-PROTOCOL.md) and [L5's failed gate](../../papers/uplink-bound/L5-RESULTS.md), whose handoff test missed by 4.6 points on pooled 59 and 60. This is distinct from OFF-1's general transport calibration.

## Prediction

Correct event ordering may alter the predicted gain; direction and size were unknown. Historical L5 residuals are baselines, not results from corrected code.

## Gate, as frozen

Frozen before any corrected replay:
- Preserve L5's handoff-gain tolerance: the predicted change in update time must be within **3 percentage points** of the measured change for each eligible matched comparison: 59 h0/hn, 60 h0/r and 61 A/B. All three must pass for acceptance.
- Report the signed seconds-per-update error for each arm, and the change from historical L5 wherever populations and summaries match. Do not substitute pooled 59/60 medians for experiment-specific results.
- A missing input or unsupported protocol yields **not evaluated**, never a pass. Do not claim full L5 validation: its placement and aggregate-error gates are outside this experiment.

## Protocol

As frozen on 2026-11-01:
- Use frozen exp54 kernel costs for workstation cases, and the original bundled costs with recorded worker capacities for 61. Preserve L5's transport constants, and never fit on 59–61.
- Hash source, costs, plans, profiles and configuration, and retain historical outputs. Use identical plans and settings for the ordinary and handoff arms.
- For 59/60, report the median of updates 2–10 per run and the median across three repeats. For 61, the mean of updates 2–21 per job, then the mean across three repeats.
- Replay complete matching batch sequences, retaining warm-up and cache state.
- Exclude 59 hb (blocking) and 60 r4 (changed unowned limit) from acceptance.
- For 61, match gzip on weak workers, mixed links, reported capacities and ownership, and don't invent profiles.

**Replay settings on the workstation, frozen 2026-11-03 before any replay:**

- **Inputs:** the audit of 2026-11-01 ran on the simulation machine and found nothing. The workstation has `runs/exp54`, `runs/exp59`, `runs/exp60` and `runs/pods-data/r1/main/*`, and nothing was downloaded.
- **Prerequisites built first,** each opt-in, with defaults unchanged:
  - `HandoffSimulation` now charges `upload_hash_ms_per_mib` on handed-path uploads, as `Simulation.upload` does. L5's handoff engine never charged it: its `streamed_upload` skipped the stage, so L5's handoff arm ran without it.
  - `handoff_reads="whole"` reads every named result in full, as in 59 and 60's f arm. `region` stays the default (60's r).
  - `replay_sequence` runs consecutive batches in one simulation, with update k on batch k's plan and owners, caches and clocks carried over.
- **Constants:** L5's five hop constants (0.53, 0, 86.3, 29.8, 3.7) and `L2.PROTOCOL`, with kernel noise and jitter at their defaults.
- **Kernels:** for 59 and 60, a model fitted with `simulate fit` on experiment 54's eight runs, run twice to check it hashes the same. For 61, the bundled model scaled by each job's reported GFLOP/s. Links are 50 ms: 100/20 Mb/s for the 2-core workers and 500/100 Mb/s for the 8-core ones.
- **Batches:** a run's update k uses batch index k−1, checked against 61's recorded tokens.
- **Arms:**
  - 59: h0 against hn, with whole reads.
  - 60: h0 against r, with region reads; f (whole reads) as a diagnostic.
  - 61: A against B, with region reads.
  - Seeds 0, 1 and 2 stand in for the three repeats, summarized as the measurements are.
- **Not modelled:** gzip frames on 61's weak workers, so 61 is not evaluated and its numbers are a diagnostic. 59's whole reads are logical bytes, not exact blocks.
- **Anchor:** L5's pooled tiny no-handoff case is recomputed with the refitted costs.

## Log

- 2026-11-01: The user authorized adding and validating OFF-5, offline only. On the simulation machine, the audit found no exp59/60 raw profiles or captures, no exp61 job reports, no frozen L2 snapshots and no exp54 fitted cost model; no download was attempted. It recomputed experiment 61's six job means from the saved per-update observations, each within 0.00011 s of the evidence.
- 2026-11-01: Local OFF-3 reassigned to OFF-5 to avoid a different upstream experiment. The original pre-merge file is in the local merge backup.
- 2026-11-03: Started on the workstation, where the inputs are. Built the three prerequisites (handoff hashing, whole reads, batch sequences) with tests, and froze the replay settings above before any replay.
- 2026-11-03: Replay started at da12f31+dirty (the prerequisites, their tests, this file, the index and M-2), seeds 0, 1 and 2, with `replay.py` in `runs/experiments/OFF-5/omarchy-20261103-103426/`. Experiment 54's refitted cost model hashes the same on two fits (77a6e8d6…). Watchdog: `--stall 600 --busy-stall 2400 --max-runtime 10800`.
- 2026-11-03: The first start failed after the anchor: the driver passed `L2.PROTOCOL`'s `updates=2` into `replay_sequence`, which rejects it. Fixed in the driver and restarted; that output is kept in `run-failed-updates-arg/`.
- 2026-11-03: Finished, `finished`, exit 0. The 61 batch check failed on all 132 updates. The cause: the pods trained their own SentencePiece model (sha256 `fb73068e…`), not this workstation's (`24cc7eb7…`), on the same dataset (`f3b85415…`). That model isn't on this machine. Checked afterwards that every batch in the tiny captures matches its replayed plan exactly.

## Results

### The gate

| Comparison | Measured change | Simulated change | Points | Outcome |
|---|---|---|---|---|
| 59 h0 → hn (whole reads) | −23.26% (3.933 → 3.018 s) | −24.42% (3.701 → 2.797 s) | **−1.16** | **Pass** |
| 60 h0 → r (region reads) | −24.13% (3.948 → 2.995 s) | −28.38% (3.701 → 2.650 s) | **−4.26** | **Fail** |
| 61 A → B (region reads) | −22.50% (5.986 → 4.639 s) | −29.14% (6.522 → 4.621 s) | −6.64 | **Not evaluated:** gzip and batches don't match |

Not accepted: one of the two evaluated comparisons fails, and the third couldn't be evaluated.

### Signed errors per arm

| Arm | Measured | Simulated | Error |
|---|---|---|---|
| 59 h0 | 3.933 (3.914, 3.933, 3.944) | 3.701 (3.692, 3.701, 3.721) | −5.9% |
| 59 hn | 3.018 (3.000, 3.018, 3.069) | 2.797 (2.718, 2.797, 2.828) | −7.3% |
| 60 h0 | 3.948 (3.901, 3.948, 4.021) | 3.701 | −6.3% |
| 60 r | 2.995 (2.942, 2.995, 3.046) | 2.650 (2.650, 2.704, 2.650) | **−11.5%** |
| 60 f, diagnostic | 3.017 (3.001, 3.017, 3.121) | 2.797 | −7.3% |
| 61 A, diagnostic | 5.986 | 6.522 (6.558, 6.496, 6.511) | +8.9% |
| 61 B, diagnostic | 4.639 | 4.621 (4.581, 4.651, 4.632) | −0.4% |

59 h0 and 60 h0 run the same plain protocol on the same batches, so they share one simulated value.

**Against historical L5:** L5 pooled 59 and 60, so per the gate there's no matching population to compare per experiment. The anchor reproduces L5's pooled no-handoff value exactly (3.672943 s with the refitted costs). L5's pooled handoff value moves from 2.638 to 2.691 s (+2.0%) with the event-ordering fixes and the hashing stage.

## What it taught

- **The overstatement is in region reads.** The simulator gets whole reads right: −24.4% against −23.3% for 59, and −23.6% for 60's f arm. With region reads it predicts another 0.15 s per update (2.797 → 2.650 s). Experiment 60 measured no such gain (3.017 against 2.995 s, within the repeats). The simulator charges region reads only the bytes they save. The real relay does work per region and still waits for every name, so whatever region reads cost in practice isn't priced.
- **The event-ordering fixes moved the handoff predictions only a little** (+2% on L5's pooled case), and in the right direction.
- **61 is likely the same story,** with the same region reads and a 6.6-point overstatement. Two things block a matched test: the pods' tokenizer and gzip.
- **A frozen check caught a wrong input:** 61's replay would have used batches that differ from the runs'.
- **Bug found:** L5's `HandoffSimulation` silently dropped the 3.7 ms/MiB upload hashing that the plain arm paid, so L5's handoff gain had a small built-in bias. It's now charged on both arms.

## Limits

- **Simulated:** this tests the simulator against past measurements, and establishes neither real-system speed nor numerical correctness.
- **Tiny runs, one machine.** 59 and 60 are the 1M model on emulated home links on this workstation. The 30.3M comparison (61) wasn't evaluated.
- **The tiny batch mapping is checked for updates 1–8.** That's what the captures hold; updates 9–12 follow the same rule unchecked.
- **Whole reads are logical bytes,** not exact block overfetch, deduplication, framing or compression.
- **Not modelled:** gzip (61), relay long-poll service time, per-region relay work, failed attempts and lease recovery.
- **Seeds stand in for repeats:** the simulator's seeds sample owner assignment and kernel noise, not the run-to-run variation of a real machine.
- **The plain arm also runs short,** by 6% on the tiny runs (L5's known gap, the lease wait), so both arms' errors carry that.
- The full test suite has 5 failures in `test_placement`, `test_http_compression` and `test_http_lstm`. They fail the same way with this experiment's code changes stashed, so they predate it.
- `ruff` isn't installed in this environment, so the lint and format checks on the new code were not run.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_simulation_handoff
OUT=runs/experiments/OFF-5/$(hostname)-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT
.venv/bin/python -m ygaraty.experiments.simulate fit --from runs/exp54/r{1,2}-{home,loopback}-prune{0,1} --cost-model $OUT/cost-exp54-1.json
cp runs/experiments/OFF-5/omarchy-20261103-103426/replay.py $OUT/
.venv/bin/python $OUT/replay.py $OUT        # writes $OUT/results.json and prints the gate
```

`replay.py` lives in the raw run directory, which isn't in git. It needs the local `runs/exp59`, `runs/exp60`, `runs/pods-data/r1/main` and the OPUS-100 SQLite dataset. The evidence file records every hash: sources, cost model, plans, dataset and experiment 61's evidence.
