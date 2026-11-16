# Policy study with L5's hop costs: frozen protocol

Written 1 November 2026, before the sweep ran. It covers four follow-ups in one sweep, because they share every run:

- **#3:** protect large outputs from slow uplinks;
- **#4:** count communication in critical-path priority;
- **#5:** sweep claim depth and placement together;
- **#6:** fall back conservatively to FIFO.

Code: `ygaraty/simulation/policy_study.py` (`PolicySimulation`), `ygaraty/simulation/policy_sweep.py`. Tests: `tests/test_policy_study.py`.

## What runs

- **Corpora:** `runs/local-1/factorization-v1`, `mlp-validation-v1` and `transformer-test-v1`, all seven scenarios each.
  - Factorization and MLP graphs are pruned by their output contracts, as in NETWORK-PLACEMENT.md. `fused_1024` and `fused_4096` have no contract and are left out, as there.
  - The Transformer graphs have no contract either, and run unpruned. Every task still commits.
- **Reservation caps:** 16 and 64 MiB, as published. The exception is the Transformer graphs: the reservation model charges them up to 258 MiB (tiny) and 31 GiB (30.3M) for one task, so they run with the cap effectively off (64 GiB).
- **Configurations:** 28 in all, each at depths 1, 2, 4 and 8:
  - `fifo` without a guard;
  - `compute` priority (today's critical path) and `comm` priority, each with no guard, guard `v1`, or the `transfer` guard.
- **Hop costs:**
  - **`l5`** (primary): L5's measured hop: `coordinator_ms` 0.53, `claim_unit_ms` 0, `reply_ms_per_mib` 86.3, `decode_ms_per_mib` 29.8, `upload_hash_ms_per_mib` 3.7.
  - **`published`:** the corpora's own settings, as the earlier studies ran. Used as a sensitivity check.
- **Seeds:**
  - tuning: 0 (deterministic) and 100–104 (noisy);
  - test: 2000–2009, untouched until evaluation.
- **Depth selection:** each configuration's depth is the one with the lowest mean makespan over the tuning seeds. Every reported gain is the mean over the test seeds.

The new rules are defined in the module docstring of `policy_study.py`:
- **`comm` priority** adds each producer's output transfer to the critical path. That transfer is priced at the pool's median link, plus one round trip, plus L5's per-byte reply and decode costs.
- **The `transfer` guard** qualifies a task by its upload time on the claiming worker (at least 50 ms) and its `comm` rank. It estimates each worker's finish as a pipeline: its compute queue, then the uplink queue of its pending outputs.

## Criteria

All are on the `l5` hop. A decision is **robust** only if it also holds on the `published` hop; otherwise it is reported as cost-dependent.

| # | Accepted if | Comparison |
|---|---|---|
| 3 | The `transfer` guard beats `v1` on heterogeneous factorization (gain over `compute/none`), no heterogeneous MLP case is worse than −1%, and no case is worse than −2% | `compute/transfer` and `compute/v1` against `compute/none`; also against the best fixed depth in hindsight |
| 4 | Mean gain over all cases above 0, no scenario's mean worse than −2%, and the worst case for `compute` against `fifo` improves | `comm/none` against `compute/none` |
| 5 | Descriptive. Reports the depth gain (tuned depth against depth 1), the placement gain at the same depth, and the joint choice, apart, under the same caps | `compute` priority |
| 6 | The conservative selector's worst test slowdown against FIFO is at most 2% in every case, its mean gain is above 0, and its worst case beats the seed-0 selector's | Selectors over all 112 configuration-and-depth pairs, against `fifo/none` at its tuned depth |

The two selectors in #6:
- **seed-0** takes the fastest configuration on seed 0 alone, the earlier practice.
- **Conservative** maximizes the mean tuning gain minus twice the worst tuning slowdown, and keeps FIFO unless that score is positive.

The 70.61% regression named in the follow-up list was not found in the repository. #4 therefore targets the worst case for `compute` against `fifo` measured in this sweep.

Every residual is published either way.
