# Robustness with L5's calibrated costs: pairs penalty now holds in all four pools; pruning, the guard and adaptive depth still don't decide

2026-11-01, offline on the workstation. Script: `papers/uplink-bound/scripts/robustness_l5.py`. Data: `papers/uplink-bound/data/robustness-l5.csv`. It repeats [ROBUSTNESS.md](ROBUSTNESS.md) with the simulator recalibrated in L5 ([L5-RESULTS.md](../../../papers/uplink-bound/L5-RESULTS.md)), over the uncertainty of L5's measured constants instead of L2's wide box.

## Verdict

**With the calibrated hop, the simulator can now decide placement, and handoffs are robustly large. It still cannot decide pruning, the guard, or adaptive depth.**

| Decision | Before (L2's costs, 60 points) | Now (L5's costs, 81 points) | Measured |
|---|---|---|---|
| One layer each over pairs, 8 strong | robust, +4 to +12% | **robust**, +9 to +22% | +15.9% |
| 6 strong | 98% | **robust**, +6 to +18% | +6.7% |
| 4 strong | 53% (wrong at the reference point) | **never reversed**, −0.1 to +17% (94% same sign; the rest within 1% of zero) | +6.5% |
| 2 strong | 42% | **robust**, +2 to +13% | +2.7% |
| 512 over 128 tiles, home | robust, +2 to +9% | **robust**, +2 to +10% | +12.0% |
| Pruning, home | 17% | 88% around zero, −2.6 to +2.1% | +5.4% |
| Pruning, loopback | 52% | 35%, −5.9 to +6.9% | +5.4% |
| Handoffs, tiny home | not run | **robust**, +26 to +32% | +23.6% |
| Handoffs, 30.3M 8 strong | not run | **robust**, +25.5 to +34.5% | not measured (R1) |

- **Placement is settled by the model now, not only by measurement.** "One layer each" wins in all four pools at every point. The simulator overstates the penalty at 4 and 6 strong workers (about +12–15% against +7% measured), as L5 found.
- **Handoffs:** the tiny gain is overstated at every point of the grid. The lowest simulated tiny gain (+26.4%) is still above the measured +23.6%. So L5's handoff miss is not explained by any of its four constants within these ranges; it is a cost the model lacks (as L5 suspected, the relay read or the later lease wait). The 30.3M gain stays above +25% everywhere, so the corrected expectation for R1 is still the −20 to −26% of L5.
- **Pruning** is still not decidable by the simulator. At home it predicts about zero against +5.4% measured.

## Grid

All 81 points run L5's hop, with `upload_hash_ms_per_mib` 3.7 and `claim_unit_ms` 0 fixed:

| Constant | Values | Why |
|---|---|---|
| `coordinator_ms` | 0.3, 0.53, 1 | lock held per request, one capture |
| `reply_ms_per_mib` | 43, 86.3, 130 (×0.5, ×1, ×1.5) | measured on one latency proxy; real links unknown |
| `decode_ms_per_mib` | 15, 29.8, 45 | as above |
| Kernel scale | ×0.8, ×1, ×1.25 | L5's held-out residuals were −12% to +11% |

Kernels as in L5: fitted on experiment 54 for the tiny model; the bundled model scaled by reported GFLOP/s for 30.3M; each corpus's own `cost.json` for the scheduling studies. Robust means the same sign as at L5's point at all 81 points, with gains under 1% either way counted as zero, as before. The snapshots are the first two of each case's batches, as in ROBUSTNESS.md.

## The scheduling decisions

**The guard against the best fixed depth.**
- On the MLPs: zero at nearly every point (at most +0.7%).
- On factorization: zero within ±3% on LAN and home. On the heterogeneous pool it ranges from −9.0 to +17.8%, with the same sign at 43% of points.

So the guard still has no effect the simulator can establish. The follow-up policy study ([POLICY-STUDY-PROTOCOL.md](POLICY-STUDY-PROTOCOL.md)) evaluates a transfer-time guard on noisy seeds, which this study does not.

**Adaptive claim depth against the best fixed depth:**
- negative at L5's point in 20 of 21 cases;
- negative at every point in 10 cases (the MLPs on LAN and home, except `mlp_77` at home), by −1.7 to −19%;
- never robustly positive.

It stays rejected, now on firmer ground than before.

## Limits

- **One seed:** the deterministic tuning seed, as in ROBUSTNESS.md.
- **Two snapshots per case** rather than L5's full set, so values at L5's point differ slightly from L5-RESULTS.md. For example, the 8-strong pairs penalty is +13.4% here against +15.4% there.
- **What the grid spans:** L5's constants, not the mechanisms L5 left out (the lease wait). A missing mechanism moves results in ways this grid cannot show, as the handoff overstatement does.

## Reproduce

```bash
L2_SNAPSHOTS=/tmp/l2-snapshots python papers/uplink-bound/scripts/robustness_l5.py /tmp/cost-exp54.json            # about 25 minutes
L2_SNAPSHOTS=/tmp/l2-snapshots python papers/uplink-bound/scripts/robustness_l5.py /tmp/cost-exp54.json summary
```
