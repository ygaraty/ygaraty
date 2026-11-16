# Robustness to calibration uncertainty: two simulated decisions survive, the rest flip inside the measured ranges

2026-11-01, offline on the workstation. Script: `papers/uplink-bound/scripts/robustness.py`. Data: `papers/uplink-bound/data/robustness.csv` (one row per decision and grid point). It follows L2 ([L2-RESULTS.md](../../../papers/uplink-bound/L2-RESULTS.md)), which found the simulator's kernel and per-hop costs wrong by large factors.

## Verdict

Of the simulated decisions tested, only two keep their sign over the whole range of plausible costs:
- **One layer each over pairs, with 8 strong workers:** +4 to +12% (measured +16%).
- **512 tiles over 128 for the tiny model:** +2 to +9% (measured +12%).

Every other decision flips somewhere inside the ranges L1 and L2 measured:
- pairs against one each with 6, 4 and 2 strong workers;
- pruning;
- the network-aware guard;
- adaptive claim depth.

The simulator alone should not choose them. They need measurement, or a recalibrated simulator.

## Grid

The ranges are labelled sensitivity ranges, not fits. There are 60 points:

| Cost | Values | Why this range |
|---|---|---|
| Upload | `legacy` (one aggregate relay request per result, as the published studies ran); or one block per result with a serialized per-upload overhead of 0, 15, 30 or 50 ms | L1: 75–95 ms beyond the bytes per upload on 100 Mb/s links, with two slots overlapping |
| Coordinator service | 0.5, 2 (default), 4 ms per request | L2: the default is too slow on loopback |
| Kernel scale | ×0.125, ×0.5, ×1, ×2 on every cost coefficient | L2: the bundled kernels are about 8 times too slow for the tiny model here |

At every point, each decision is made again on the deterministic seed, the tuning seed of each published study:
- **Guard** ([NETWORK-PLACEMENT.md](NETWORK-PLACEMENT.md)): the best guarded depth against the best fixed depth, both chosen at that point.
- **Adaptive** ([ADAPTIVE-CLAIMS.md](ADAPTIVE-CLAIMS.md)): adaptive depth against the best fixed depth.
- **Uplink-bound decisions (L2):** pairs or one layer each (4 pools), pruning (2 links), 128 or 512 tiles. These are scored against what was measured.

The corpora are `runs/local-1/factorization-v1` and `mlp-validation-v1`. `fused_1024` and `fused_4096` exceed the reservation cap, as in the published study.

**Robust** means the decision has the same sign as at the published point (legacy upload, 2 ms, ×1) at all 60 points. A gain under 1% in either direction counts as zero. The criterion was set before the run, in the script's docstring.

## Results

**Uplink-bound decisions:** gain of the second arm, simulated, against measured.

| Decision | At the published point | Same sign | Range over the grid | Measured |
|---|---|---|---|---|
| 8 strong: one each over pairs | +6.4% | **100%** | +4.2 to +11.8% | +15.9% |
| 6 strong | +2.2% | 98% | −1.6 to +6.7% | +6.7% |
| 4 strong | −4.0% (wrong) | 53% | −5.0 to +4.4% | +6.5% |
| 2 strong | +2.0% | 42% | −4.3 to +4.0% | +2.7% |
| 512 over 128 tiles, home | +5.0% | **100%** | +2.2 to +9.2% | +12.0% |
| Pruning, home | +1.2% | 17% | −2.6 to +3.5% | +5.4% |
| Pruning, loopback | +2.8% | 52% | −6.6 to +8.6% | +5.4% |

Every simulated gain is smaller than the measured one, at every point.

**Guard against the best fixed depth.** At the published point it gains nothing in 20 of 21 cases; the targeted `v20` failure appears only on noisy seeds, which this study does not run. Inside the ranges it moves both ways:

| Case | Range over the grid |
|---|---|
| `mlp_340` on LAN | −11.0 to +5.0% |
| `tiled_1024` heterogeneous | −6.9 to +15.1% |
| `mlp_77` heterogeneous | 0 to +12.1% |

So the guard's effect depends on costs we do not know.

**Adaptive against the best fixed depth.** It flips in most cases (15–80% same sign). Its two large losses at the published point keep their sign in most of the grid:
- `tiled_1024` heterogeneous: −199% at the published point, 63% same sign;
- `mlp_340` on LAN: −16%, 80% same sign.

Adaptive depth stays rejected.

## What it means

- **Measure the fragile decisions instead of simulating them.** For the paper, "one layer each" at 2 and 4 strong workers rests on the measurements of experiments 42 and 53, not on any model.
- **The two robust decisions are also the two largest measured effects.** Small measured effects (2–7%) are inside what the cost uncertainty can produce.
- **The guard and adaptive-depth studies should be rerun only after recalibration** (L2's next fix: hop costs, per-machine kernels). Their published gains were measured at one point of a wide range.

## Limits

- **One seed:** the deterministic tuning seed only. The published noisy-seed evaluations (2000–2003, 1900–1903) were not repeated at every grid point.
- **The per-upload overhead is charged serially** on each worker's upload-preparation resource, in the simulator's block mode. That approximates round trips which, on a real worker, partly overlap across the two upload slots.
- **The grid is a box, not a distribution.** Its corners can combine costs that no real machine has.
