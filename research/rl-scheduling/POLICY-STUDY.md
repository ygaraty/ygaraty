# Policy study on L5's hop: communication-aware priority passes narrowly; both guards and both selectors fail their criteria

2026-11-01, offline on the workstation. Protocol: [POLICY-STUDY-PROTOCOL.md](POLICY-STUDY-PROTOCOL.md), committed before the sweep (`70acb3e`).
- **Code:** `ygaraty/simulation/policy_study.py`, `policy_sweep.py`.
- **Analysis:** `policy-study/analyze.py`, with its output in `policy-study/analysis.txt`.
- **Per-case results:** `policy-study/summary.json`.
- **Raw rollouts** (75,264 runs) are in `runs/policy-study/rollouts.csv.gz`, which is not committed.

## Verdict

**Nothing here should change a runtime default.** Only #4 meets its criteria, by a small margin and on the calibrated hop only.

| # | Decision | Result on L5's hop | On the published hop | Accepted? |
|---|---|---|---|---|
| 3 | Transfer-time guard | Beats v1 on mixed-pool factorization (+0.6% against −0.4%); mixed-pool MLP not worse (`mlp_77` +15%). But `transformer_tiny` on the mixed pool loses **−33.9%** | Worst case −2.8% (`fused_1024`; −0.7% without it) | **No** (a case below −2%) |
| 4 | Communication in critical-path priority | Mean **+0.7%**; worst scenario mean −0.2%; fixes the worst compute-vs-FIFO case by +5.0%. Worst single case −10.2% | Worst scenario mean −1.4%, and the target case gets worse (−0.9%) | **Passes, but cost-dependent** |
| 5 | Depth and placement apart | Depth +5.1% on average, placement at the same depth +1.0–1.4% (range −57 to +42%), the two together +8.2% | Depth +8.6%, placement +4.1–4.5%, together +14.1% | Descriptive |
| 6 | Conservative fallback to FIFO | Mean +5.1%. Cases with a test seed worse than −2%: **3 of 84**, against 20 for the seed-0 selector. But the worst seed is −122% | Mean +7.2%, 1 case below −2% (worst −3.0%, against −8.6% for seed-0) | **No** (worst slowdown above 2%) |

What the sweep shows beyond the criteria:

- **Choose depth first.** On average, depth is the larger lever, and the only one on LAN and loopback. Placement matters on the heterogeneous, congested and late-join pools, where it is large but erratic, swinging both ways.
- **The conservative selector removes most bad outcomes but not the worst one.** On L5's hop it cuts the cases with a seed slower than −2% from 20 to 3 and raises the mean gain from +3.4% to +5.1%. The remaining failure is `tiled_1024` on the mixed pool. Its makespan is bimodal on the noisy seeds (the `v20` placement of NETWORK-PLACEMENT.md), and no tuning seed shows the bad mode for the chosen configuration. No rule that chooses on tuning seeds can bound it.
- **Critical path itself is not safe against FIFO here.** At its tuned depth it averages only +0.8% over FIFO, and is worse than −2% on some seed in 29 of 84 cases. Its worst losses are on the unowned Transformer graphs, for example −34% on `transformer_30m`/`home_16`. LOCAL-1 found critical path ahead by up to 21%, but under its corpora's protocol of one task per claim with no pushed inputs. With eager claims, pushed inputs and tuned depth, the ordering advantage shrinks or reverses.
- **Guards help the Transformer graphs a lot at no reservation cap.** v1 gains +33–43% on `transformer_30m` on the congested, `home_16` and late-join pools, on both hops. `home_16` has identical links, so this is queue balancing, not uplink choice, as NETWORK-PLACEMENT.md warned. These graphs are unowned and ungrouped and their makespans are hundreds of seconds, so this does not transfer to the owned production plans.

## Per decision

**#3, guards against no guard** (`compute` priority, each at its tuned depth, mean over test seeds). Cases where either guard moves 2% or more, plus every mixed-pool case, are in `analysis.txt`. Selected:

| Case | v1 | Transfer-time |
|---|---|---|
| `tiled_1024`, mixed pool | −0.4% | +0.6% |
| `mlp_77`, mixed pool | 0 | **+15.0%** (+16.7% published) |
| `tiled_1024`, `home_16` | +3.5% | +11.2% |
| `transformer_tiny`, mixed pool | +5.1% | **−33.9%** (+18.5% published) |
| `transformer_tiny_b8`, mixed pool | +31.7% | +32.2% |
| `transformer_30m`, `home_16` | +43.4% | +43.8% |
| Mean over all 84 cases | +2.1% | +2.7% |

- **The targeted factorization case barely moves on L5's hop,** in either direction.
- **The transfer-time guard's mixed-pool MLP gain is real in the simulator,** and holds on both hops. It defers `mlp_77`'s large outputs to the fast uplinks; v1's 256 KiB rule never fires there.
- **Its `transformer_tiny` loss is a same-depth loss** (−22.5%). It made 1,888 deferrals on that graph's deterministic seed at depth 4; why its estimate misleads there is not yet traced.

v1 would also fail the "no case below −2%" rule: −10.7% on unpruned `fused_1024` on the mixed pool, and −5.0% on `transformer_tiny` on LAN.

**#4, `comm` against `compute` priority**, mean over workloads per scenario:

| Scenario | L5 hop | Published hop |
|---|---|---|
| `congested_16` | +0.7% | +0.8% |
| `heterogeneous_8` | +0.1% (−10.2 to +11.7%) | −1.4% (−37.1 to +10.3%) |
| `home_16` | +1.2% | +0.5% |
| `home_4` | +1.1% | +1.9% |
| `lan_4` | −0.2% | −0.5% |
| `late_join_8` | +1.8% | +1.1% |
| `loopback_1` | +0.5% | 0 |

The gain is small and its sign varies by case, with the largest swings on the mixed pool. It is a candidate for live measurement, not a default.

**#6, selectors against FIFO at its tuned depth:**

| Selector | L5: mean | L5: worst seed | L5: cases below −2% | Published: mean | Published: worst | Published: below −2% |
|---|---|---|---|---|---|---|
| Seed 0 only (earlier practice) | +3.4% | −81% | 20 | +7.2% | −8.6% | 11 |
| Conservative | **+5.1%** | −122% | **3** | **+7.2%** | **−3.0%** | **1** |
| Critical path, tuned depth | +0.8% | −57% | 29 | +0.5% | −70% | 29 |

The conservative selector kept FIFO in 28 of 84 cases on L5's hop, and in 34 on the published hop.

## Deviations from the protocol

- **`fused_1024` and `fused_4096` were run, unpruned.** The protocol said they would be left out. The sweep code instead runs any graph without an output contract unpruned, as it does the Transformer graphs. `fused_4096` exceeds both reservation caps, so only `fused_1024` contributes cases: 14 per hop.
  - **No verdict depends on them.** On L5's hop, the deciding cases of #3 and #6 are `transformer_tiny` and `tiled_1024`. #4's target case is `transformer_30m`.
  - **On the published hop they decide #3 there.** Its worst case on that hop is `fused_1024`/late join at −2.8%. Without `fused_1024`, the worst would be `mlp_77`/`home_4` at −0.7%, and the transfer-time guard would meet its criteria on that hop. The verdict stands, because it is taken on L5's hop. `fused_1024` also has five of #6's six worst conservative cases on the published hop.
- **The Transformer graphs run without a reservation cap,** as the protocol said. Every Transformer number is therefore outside the memory limits under which the factorization and MLP results hold.

## Limits

- **Simulated only.** It uses L5's hop, which overstates handoff gains and the pairs penalty at 4 and 6 strong workers ([ROBUSTNESS-L5.md](ROBUSTNESS-L5.md)). No guard, priority or selector here has been run on real workers.
- **Ten test seeds per case.** The worst-seed statistics rest on bimodal schedules like `tiled_1024`'s, so a bad mode with a 5% chance has a fair chance of not appearing at all.
- **The corpora are unowned plans.** The production Transformer plans are owned and grouped, and this study's classes refuse them, so none of these results speaks directly to the uplink-bound runs.

## Reproduce

```bash
python -m ygaraty.simulation.policy_sweep run --output runs/policy-study --processes 12      # about 35 minutes here
python public/research/rl-scheduling/policy-study/analyze.py runs/policy-study/summary.json
python -m unittest tests.test_policy_study
```
