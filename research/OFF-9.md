# With handoffs, should strong workers own contiguous layers again? An interval planner priced by the simulator

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-9 (number added on finish) |
| **Category** | simulation |
| **Where** | this machine |
| **Date** | Added 2026-11-05 |
| **Path** | Offline: corrected `HandoffSimulation` with L5's hop phases (experiment 71's replay setup), experiment 61's pool (8 c-2 and 8 c-8 workers, emulated 100/20 Mb/s, 50 ms) and experiments 42/53's mixed-uplink pools, the 30.3M translator's plan. No training, no network |

## Question

Today's placement rule gives each fast worker one layer (`pairs='never'`, experiment 42). It won over pairs of consecutive layers by 7–16% on real links, before handoffs. Experiments 62–64 traced that pairs penalty to the claim path, not to upload slots or the post queue. With handoffs on, the penalty fell from 10.3% to 3.9% (experiment 63, tiny model, workstation).

BloomBee (arXiv 2604.21072) places models with a dynamic program over contiguous layer intervals:
- one interval per node;
- cost = profiled compute + λ + bytes ÷ bandwidth per boundary;
- O(N·L²), under 1 ms.

Its cost model is close to the byte-only model that ranked all 4 of our measured pools wrong (Section 6, calibration A). Our event simulator, with each phase of a hop priced as measured (L5), ranked them right.

**With handoffs, does an interval planner priced by the simulator choose contiguous layers for strong workers on experiment 61's pool, and does its best plan beat one layer per fast worker in simulation? Does a BloomBee-style cost model rank the same plans the same way?**

## Prediction

- **Validation first:** the simulator reproduces experiment 63's pairs penalty with handoffs (3.9% measured) within 3 points. It reproduces experiment 61's handoff update time (4.64 s) within L5's error, but experiment 71 found it still credits region reads with about 0.15 s too much (OFF-6).
- **Experiment 61's pool** (equal uplinks, unequal cores): the planner picks contiguous pairs of decoder layers on c-8 workers. Each pair removes a hop (about 230 ms) at the price of one longer stage. Simulated gain over one layer each: 2–6%.
- **Experiments 42/53's pools** (uplinks differing 5×): one layer each stays best with 6 or 8 strong workers; with 2 strong workers, pairs and one each tie, as measured in experiment 53.
- **The BloomBee-style model** prefers pairs on every pool and overstates the gain 2–3×, as calibration C did in Section 6, because it misses the fixed non-byte time of a hop.

## Gate

TBD, to be frozen before the simulation. Proposed:
- **Validation (required before planner results count):** the simulated pairs penalty with handoffs on experiment 63's setup within **±3 points** of 3.9%, and experiment 61's arm B within 10% of 4.64 s.
- **Counts as worth a measured run:** the planner's best plan on experiment 61's pool at least **5%** shorter than one layer per fast worker, in simulation. If so, add a WS or DO follow-up to measure it.
- **Report also:**
  - the chosen plan per pool;
  - the top 5 plans and their simulated times;
  - Kendall's τ between the simulator's ranking and the BloomBee-style model's ranking over all enumerated plans;
  - the planner's run time.

## Protocol

1. **Planner** (`ygaraty/coordination/interval_plan.py`, offline only):
   - **Decision:** for each side (encoder, decoder), a split of its layers into contiguous intervals and an assignment of intervals and vocabulary slices to workers, at most one interval per worker.
   - **Search:** a dynamic program over (layers placed, workers used), with workers ordered by strength as in `place`.
   - **Exhaustive check:** at 3+3 layers, enumerate every split and assignment and check that the program finds the true minimum under its own cost model.
   - **Two cost models:**
     - **S:** each candidate priced by `HandoffSimulation` (corrected, L5 hop constants), 3 seeds of noise;
     - **B:** BloomBee-style, the sum of profiled compute per interval + per boundary λ (25 ms one-way) + bytes ÷ the worker's uplink, with the same kernel costs and bytes.

     The program searches under B, which is fast. S re-prices the top 20 plans under B plus `pairs='never'` and `pairs='always'`. At 3+3 layers, S also prices every plan.
2. **Validation:**
   - experiment 63's two arms (P2h against N2h) on its tiny-model setup;
   - experiment 61's arm B.
3. **Pools:**
   - experiment 61's;
   - experiment 42's three cells (4, 6, 8 strong of 16) and experiment 53's (2 strong), with handoffs on (counterfactual: these were measured without handoffs);
   - one scale check: a 12+12-layer model of the same width on experiment 61's pool, where the dynamic program matters and enumeration does not fit.
4. **Inputs:** experiment 71's replay inputs (cost model, plans, experiment 61's capacities). They were built on the workstation under `runs/experiments/OFF-5/`. If they are not on this machine, report it and keep the status open. Copying them is the user's step, not a download.

## Log

- 2026-11-05: Added, from comparing BloomBee's interval planner with our capacity placement. Best after OFF-6, whose region-read pricing changes handoff times; rerun the validation if OFF-6 changes the simulator.
