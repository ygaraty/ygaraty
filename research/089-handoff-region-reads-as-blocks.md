# Pricing region reads as the relay serves them cuts the simulator's handoff error to 3.1 points, just short of 3

| | |
|---|---|
| **Conclusion** | **Failed gate:** pricing region reads as whole relay blocks cuts experiment 60's error 4.3 → 3.1 points, 3 allowed (simulated, against captures). |
| **Status** | executed |
| **ID** | OFF-6 |
| **Date** | 2026-11-05 |
| **Path** | Offline, workstation: experiment 60's captures (`runs/exp60/cap-{h0,f,r}-cap`) for the bytes and chain of handed reads; one opt-in mechanism in `HandoffSimulation` (`handoff_reads="blocks"`); experiment 71's replay (`replay.py`, its cost model, plans and settings, seeds 0–2) |
| **Result** | **The mechanism is right but not enough.** A real region read brings every uploaded block its regions touch, whole and once: the `embed_rows` tasks read hundreds of one-row regions of the vocabulary gathers and get each gather's block (359 kB, where the simulator charged 262 kB of rows). Priced that way, the simulator's handed bytes match the capture (15.4 against 16.0 MB per update, task by task). Gate: **60 h0→r −27.20% simulated against −24.13% measured, 3.07 points (3 allowed): fail**; 59 h0→hn 1.16 points, still a pass. Experiment 61 (diagnostic) 6.6 → 3.6 points. The rest of the error is on the chain: both captures show 41 ms from a handed producer's upload to the next kernel, where the simulator gives 62 ms whole and 46 ms region. Free |
| **Evidence** | Simulated against measured captures: [evidence/handoff/off6-region-reads-as-blocks-2026-11-05.json](evidence/handoff/off6-region-reads-as-blocks-2026-11-05.json); scripts and outputs in `runs/experiments/OFF-6/omarchy-20261105-213747/` (`bytes.py`, `manifest.py`, `per_task.py`, `sim_chain.py`, `replay.py`, `results.json`). Code: `HandoffSimulation.read_blocks` and `block_read_bytes` (opt-in; the default stays `region`); `tests/test_simulation_handoff.py` `test_block_reads_cost_every_overlapped_band_whole` |

## Question

[Experiment 71](071-handoff-timing-holdouts.md) found that the corrected simulator gets whole-result handoff reads right (59, and 60's f arm), but credits region reads with 0.15 s per update that [experiment 60](060-handoff-reads-by-region.md) measured as none.

**Where does a handed region read spend its time in experiment 60's captures, and does one mechanism measured there bring 60 h0→r within 3 points without spoiling 59?**

## Prediction and Gate (as frozen)

Prediction: TBD at the time; candidates were the relay's per-region block lookup, the wait for every name, or the worker's assembly.

Gate, frozen 2026-11-05 with the user's yes, before fitting:
- **Fit only on experiment 60's captures**, following L5's rule: one mechanism with constants measured from the capture, opt-in, defaults unchanged.
- **Pass:** 60 h0→r within 3 points of the measured change, and 59 h0→hn still within 3 points, both with experiment 71's replay settings.
- **61 stays a diagnostic** unless the pods' tokenizer and gzip can be matched.

## Protocol

1. **Bytes.** Per handed task, the bytes the worker actually fetched (`worker.handoff_bytes`, median over updates 2–8 of each capture), against the simulator's handed payload (seed 0, the same updates), whole and region (`bytes.py`, `manifest.py`, `per_task.py`).
2. **Chain.** `chain_trace` on the three captures (updates 2–7), and the same walk on the simulator's per-task times (`sim_chain.py`): per hop, producer kernel end → upload landed → consumer kernel start.
3. **Mechanism.** `handoff_reads="blocks"`: a region read is priced as the relay serves it (`relay.recipe_needs`): it is followed through assemble and transpose to the uploading tasks, and every block it touches comes whole, each block once. A producer uploads one block per band of `upload_rows`, else per 512 × 512 tile: the block shapes in experiment 60's capture (376-row bands, 512-row tiles). Nothing is fitted.
4. **Replay.** Experiment 71's `replay.py` with a `blocks` arm; 60 h0→r is evaluated with it, 59 h0→hn with whole reads as before. 61 runs with `blocks` as a diagnostic.

## Results

**Bytes.** Region reads do save bytes in reality: handed reads fetched a median 19.60 MB per update whole (f) and 15.15 MB with regions (r). Summed over task medians:

| | Measured f | Measured r | Simulated whole | Simulated region | Simulated blocks |
|---|---|---|---|---|---|
| Handed bytes per update (MB) | 20.74 | 15.97 | 19.42 | 13.83 | **15.40** |

The largest differences are the `embed_rows` tasks (`v5/block/*`, `v24/block/*`): each reads hundreds of one-row regions of the vocabulary gathers, and the relay sends each gather's block whole, so region and whole reads cost the same (359 kB). The old pricing charged the rows (262 kB, and 6 kB for the small last block). With `blocks`, those match the capture.

**Chain** (medians, updates 2–7):

| | Hops | Chain | Median hop | Producer upload | Upload end → next kernel |
|---|---|---|---|---|---|
| Measured h0 | 15 | 3,964 ms | 239 ms | — | — |
| Measured f | 15 | 2,768 ms | 170 ms | 105 ms | **41 ms** |
| Measured r | 15 | 2,842 ms | 185 ms | 110 ms | **41 ms** |
| Simulated whole (seed 0) | 15 | 2,448 ms | 164 ms | 94 ms | 62 ms |
| Simulated region (seed 0) | 15 | 2,413 ms | 163 ms | 94 ms | 46 ms |

**Gate** (`results.json`):

| Case | Measured | Simulated | Points | |
|---|---|---|---|---|
| 59 h0→hn (whole) | −23.26% | −24.42% | 1.16 | pass |
| **60 h0→r (blocks)** | −24.13% | −27.20% | **3.07** | **fail (3 allowed)** |
| 60 h0→r, region as in experiment 71 (diagnostic) | −24.13% | −28.38% | 4.26 | — |
| 60 h0→f (diagnostic) | −23.57% | −24.42% | 0.84 | — |
| 61 A/B, blocks (diagnostic) | −22.50% | −26.10% | 3.60 | — (6.64 with region) |

Simulated seconds per update (median of updates 2–10, seeds 0–2): plain 3.701; whole 2.718, 2.797, 2.828; region 2.650, 2.704, 2.650; blocks 2.674, 2.760, 2.694.

## What it taught

- **Region reads are block reads.** The relay cuts a read at its producers' upload blocks, and dedupes blocks shared by several inputs; a one-row read of a 332-row gather brings all 332 rows. The simulator's limit note said this was not modelled; it is now an opt-in setting, and its bytes match the capture.
- **That is about a third of the error.** It takes 1.2 of experiment 60's 4.3 points away, and 3 of experiment 61's 6.6.
- **The rest is on the critical chain.** In both captures a handed hop on the chain waits 41 ms from its producer's upload to its kernel, whole or region: the critical consumer needs its producer's result whole either way. The simulator makes that wait 62 ms with whole reads and 46 ms with regions, so it still credits regions with time on the chain. Matching the 41 ms would need the chain's downlink and decode costs, a second mechanism that the frozen rule does not allow here.
- A 0.07-point miss is still a miss: the rule was frozen before fitting, and a rerun with a second mechanism is a new experiment.

## Limits

- One mechanism, tiny model, one capture per arm (8 updates); the simulator's chain walk uses seed 0 only.
- Some tasks are still off: `v61` fetches 19 kB in reality and 116 kB in the simulator, and the owner steps 0.3 against 0.5–0.8 kB.
- 61 stays a diagnostic: the pods' gzip frames are not modelled.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_simulation_handoff
O=runs/experiments/OFF-6/<host>-<timestamp>; mkdir -p $O
cp runs/experiments/OFF-5/omarchy-20261103-103426/cost-exp54-1.json $O/; cp -r runs/experiments/OFF-5/omarchy-20261103-103426/plans $O/
ln -s $PWD/runs/experiments/OFF-5/omarchy-20261103-103426/l2-snapshots $O/l2-snapshots
cp runs/experiments/OFF-6/omarchy-20261105-213747/*.py $O/
.venv/bin/python $O/per_task.py $O       # bytes per handed task, measured and simulated
.venv/bin/python $O/sim_chain.py $O      # the simulated chain
.venv/bin/python -m ygaraty.experiments.chain_trace runs/exp60/cap-h0-cap runs/exp60/cap-f-cap runs/exp60/cap-r-cap
.venv/bin/python $O/replay.py $O         # results.json and the gate
```

It needs experiment 60's captures and runs (`runs/exp59`, `runs/exp60`), `runs/pods-data/r1/main` and the OPUS-100 SQLite dataset.

## Log

- 2026-11-03: Added, after experiment 71.
- 2026-11-05: Gate frozen with the user's yes. Inputs checked on the workstation: `runs/exp60/cap-{h0,f,r}-cap`, `runs/exp59`, `runs/exp54`, `runs/pods-data/r1/main` and experiment 71's `runs/experiments/OFF-5/omarchy-20261103-103426/replay.py`.
- 2026-11-05 21:38: Started, in `runs/experiments/OFF-6/omarchy-20261105-213747/` (experiment 71's cost model and plans copied). Measured from experiment 60's captures (updates 2–8):
  - **Bytes are saved, as the simulator assumes.** Handed reads fetched a median 19.60 MB per update with whole reads (f) and 15.15 MB with region reads (r), from `worker.handoff_bytes` (`manifest.py`). The simulator's handed payload, seed 0: 18.3 against 13.4 MB (`bytes.py`, `bytes.json`). Block overfetch is small (real region reads 1.7 MB above the logical bytes), so it does not explain the error.
  - **The saved bytes are off the critical chain.** `chain_trace` on h0, f and r: 15 hops each; handed chain hops take a median 105 ms (f) and 110 ms (r) of producer upload, then **41 ms from upload end to the next kernel in both**. Region reads leave the chain's hops unchanged (median hop 170 against 185 ms, chain 2768 against 2842 ms), so the 4.5 MB saved per update lands on reads that wait anyway.
  - **Next:** the simulator's region credit (0.15 s per update in experiment 71) should therefore come from chain-critical handed reads whose transfer it shortens. Check the simulated chain's handed hops, whole against region; the candidate mechanism is that the chain's inputs are read whole in both arms (the critical consumer needs all of its producer's result).

- 2026-11-05 21:50: Mechanism added, opt-in: `HandoffSimulation(handoff_reads="blocks")` prices a region read as the relay serves it, every uploaded block its regions touch (bands of upload_rows, else 512 × 512 tiles, seen through assemble and transpose), each block once. Its handed bytes match the r capture: 15.4 MB per update against 16.0 measured, task by task (`per_task.py`). The `embed_rows` tasks read hundreds of one-row regions of the vocabulary gathers and get each gather's block whole (359 kB), where the old region pricing charged 262 kB of rows. Replay under watch.py (stall 300 s, maximum 1 h).
- 2026-11-05 21:44: Replay `finished` (watch.py). **Gate failed by 0.07:** 60 h0→r simulated −27.20% against −24.13% measured, 3.07 points (3 allowed; experiment 71's region pricing gave 4.26). 59 h0→hn unchanged at −24.42 against −23.26, 1.16 points, a pass. Diagnostics: 60 h0→f 0.84 points; 61 A/B −26.10 against −22.50, 3.60 points (6.64 with region pricing; gzip still not modelled). The remaining error is on the chain: the simulator's handed chain hops take 46 ms from upload to kernel with region reads against 62 ms whole, while both captures show 41 ms (`sim_chain.py`).
