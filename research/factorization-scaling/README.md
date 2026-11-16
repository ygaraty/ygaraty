# Small SGD matrix factorization: correctness and offline scaling

This study compares two dense rank-k factorization plans: a reference using
Ygaraty's matrix operations, automatic differentiation and tiled planner, and
a fused row-shard kernel with bounded gradient and loss reductions. Every
matrix entry is observed; zeros are values, not missing ratings.

Both use a full-matrix batch with the SGD update rule. They do not sample
minibatches or implement sparse observed-entry training.

For the prepared larger fused HTTP execution path, see the [remote test
procedure](REMOTE.md). It includes separate bounded tensor uploads, a two-host
correctness gate and a repeated 1/2/4/8-host benchmark. Remote execution remains
unvalidated until that procedure runs on an authorized external environment.

For X with m rows and n columns, learn U (m × k) and V (n × k):

```
L = ||U Vᵀ - X||² / (2mn) + λ (||U||² + ||V||²) / 2
```

The recipe is in `ygaraty/models/factorization.py`. `build_tape(m, n, k)` creates
only shape metadata for offline planning. Passing `values={'X': X, 'U': U,
'V': V}` builds the same differentiated graph from supplied finite arrays. The
caller appends SGD tasks with `compiler.with_optimizer`; the model does not
start a training loop, worker, coordinator, or network connection.

`build_fused_plan(m, n, k, block_rows=64)` also constructs only shapes and
tasks. Each row shard computes its residual, U gradient, partial V gradient,
and scalar loss locally, reading the same immutable U/V snapshot. The dense
residual stays inside the kernel. U updates remain row-sharded, while bounded
trees combine V gradients and losses. L2 regularization is applied once per
factor element. Shard size does not change with the worker count. The Python
and browser implementations share this contract; the browser fused kernel
currently uses its CPU backend.

## Three checks

1. **Numerical correctness:** a fixed 3 × 4 matrix, rank 2. Compare loss,
   both gradients, and one SGD update with independent NumPy equations;
   check gradients by finite differences. Cover regularization 0 and 0.07,
   tile sizes 1, 2 and 3, and fused/unfused matrix-product planning. Inputs
   must remain unchanged. Fused row blocks of 1, 2 and 5 also check an uneven
   final shard and agreement with the reference graph. These are tiny
   in-process unit checks, not a throughput benchmark or a convergence
   demonstration.
2. **Modeled loopback strong scaling:** keep a 1,024 × 1,024 matrix, rank 32,
   and partition size fixed while increasing independent worker slots from
   1 to 2, 4, 8 and 16. Zero RTT and unlimited access-link rates; shared
   relay and coordinator costs remain enabled.
3. **Modeled home-link strong scaling:** the same graph, worker counts,
   seeds and cost model, with 50 ms RTT, 100 Mb/s down and 20 Mb/s up per
   worker.

The timing experiments use no matrix payloads or external datasets. They
replay a shape-only graph, so no SQLite URL is required. They never execute
model kernels or train the factors. A slot represents an independent,
fixed-speed modeled worker, not an empirically measured CPU core.

## Reproduce

From the repository root, using the existing local environment:

```bash
python -m unittest discover -s tests -p 'test_factorization*.py' -v
mkdir -p runs/factorization-scaling
ygaraty simulate-factorization \
  --implementation reference \
  --rows 1024 --columns 1024 --rank 32 --tile 128 \
  --workers 1,2,4,8,16 --seeds 3 --updates 2 \
  --output runs/factorization-scaling/results.json \
  --plan-output runs/factorization-scaling/plan.json \
  > runs/factorization-scaling/report.txt
```

That command reproduces the historical second-update baseline below. For the
new comparison, run all six named graph, claim-size and cache variants with
the same dimensions, three updates and seeds 0, 1 and 2:

```bash
ygaraty experiment factorization_compare \
  --rows 1024 --columns 1024 --rank 32 --tile 128 --block-rows 64 \
  --workers 1,2,4,8,16 --seeds 3 --updates 3 \
  --output-dir runs/factorization-comparison
```

Run the fused plan alone with one task per claim, or reproduce the larger
fixed-workload study:

```bash
ygaraty simulate-factorization \
  --implementation fused --max-tasks 1 --block-rows 64 \
  --rows 1024 --columns 1024 --rank 32 \
  --workers 1,2,4,8,16 --seeds 3 --updates 3 \
  --output runs/factorization-comparison/fused-1024.json
ygaraty simulate-factorization \
  --implementation fused --max-tasks 1 --block-rows 64 \
  --rows 4096 --columns 4096 --rank 32 \
  --workers 1,2,4,8,16 --seeds 3 --updates 3 \
  --output runs/factorization-comparison/fused-4096.json
```

With matplotlib already available, render the saved predictions without
rerunning simulations:

```bash
python public/research/factorization-scaling/plot_comparison.py
```

This writes `comparison.png` and `comparison.svg` in the comparison directory.

The equivalent dispatcher is `ygaraty simulate-factorization`. An existing
local cost model can be selected with `--cost-model`. Outputs must not collide
with each other or the input cost model. Generated artifacts remain in `runs/`.

## Historical reference predictions: second update, 2026-10-28

The reference graph contains **662 tasks/update**, including loss evaluation
and SGD for both factors. Learning rate is 0.01 and λ is 0.001. The report takes the median
second-update time across seeds 0, 1 and 2; min/max ranges describe modeled
scheduling variation, not calibration confidence.

The default bundled cost model is unchanged. Simultaneous simulated claims now
resume in deterministic worker order; a subprocess regression test checks
that timing metrics are identical across Python hash seeds. Both scenarios retain a shared
10 Gb/s relay, 2 ms coordinator service per request, 0.5 ms claim-unit service,
5 ms request jitter, 10% kernel noise, up to 16 tasks per claim, and a 512 MiB
cache per worker. Float64 transport and ordinary pull/claim behavior are used.

| Modeled slots | Loopback s/update | Speedup | Efficiency | Home-link s/update | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.503 | 1.00× | 100.0% | 52.628 | 1.00× | 100.0% |
| 2 | 2.134 | 1.64× | 82.1% | 26.746 | 1.97× | 98.4% |
| 4 | 1.487 | 2.36× | 58.9% | 15.539 | 3.39× | 84.7% |
| 8 | 1.369 | 2.56× | 32.0% | 10.507 | 5.01× | 62.6% |
| 16 | 1.240 | 2.82× | 17.6% | 7.799 | 6.75× | 42.2% |

Speedup is median T(1) / median T(p); efficiency is speedup / p. The declared
near-linear threshold is **80% efficiency**, not an assertion in the tests.
Loopback meets it at 2 slots, and home links at 2 and 4. Neither scales
linearly to 16. Home links have a higher relative speedup because each added
worker contributes another access link; absolute timing remains slower.

## Why scaling stops, and what the numbers cannot establish

- The reference planner leaves generic `sum` and its backward `broadcast` as
  whole-matrix tasks. The largest loss reduction alone is modeled at about
  **0.562 seconds**. The compute-only graph has about 2.495 kernel-seconds of
  total work and a 0.586-second critical path: a work/span ratio of **4.26**.
  This ratio excludes communication and is not a bound on the network-inclusive
  speedup in the table.
- Coordinator work, claim granularity, network transfers and dependency waits
  further reduce scaling. Worker stage times overlap and must not be summed
  to reconstruct wall time.
- The reference plan's cache keys include the update number: its second
  update does **not** assume X stays cached across updates.
- Costs were fitted for earlier workloads, not this factorization. The bundled
  22.55 ms matmul intercept dominates many small rank-32 products; several
  elementwise operations and SGD use fallback costs. Calibration ranges,
  single-core throughput and BLAS thread count are not established for this
  workload. The JSON report includes all coverage warnings and source/graph
  checksums.
- CPU contention, shared memory bandwidth, BLAS oversubscription and host core
  limits are not simulated. These predictions cannot establish measured
  multicore scaling. This workspace remains limited to offline simulation
  and the tiny numerical checks described above.

## Fused comparison: third update, 2026-10-28

The following predictions come from
`runs/factorization-comparison/comparison.json`. They report the median third
update across the same three seeds, not the historical second update above.
Problem dimensions, objective, learning rate, network settings and cost
coefficients are unchanged. The fused 1,024 × 1,024 plan has **98 tasks/update**
and uploads **6.820 MB/update**, compared with **662 tasks** and **79.692 MB**
for the reference. MB is decimal relay traffic, not peak worker memory.

First compare the original **16 tasks per claim** setting for both plans.
The reference marks only X reusable within an update; the fused plan marks
X, U and V reusable. The fused variant also enables versioned immutable-X
caching across updates. Consequently, this compares the graph and its cache
policies together, not kernel fusion in isolation.

| Plan, 16 tasks/claim | Loopback 1 slot, s | Loopback 16 slots, s | Home 1 slot, s | Home 16 slots, s |
|---|---:|---:|---:|---:|
| Reference | 3.4935 | 1.3122 | 52.6353 | 7.6668 |
| Fused, persistent X | 1.9244 | 0.3649 | 5.9580 | 1.8816 |

At 16 slots, this is a modeled **3.60× reduction in loopback update time**
and **4.07× on home links** relative to the original reference configuration.
Those implementation comparisons are different from strong-scaling speedup,
which divides each variant's own one-slot time by its p-slot time.

One task per claim is a separate tuning choice. It distributes the short
fused graph more evenly on loopback but increases request overhead on home
links. These results retain the slower reference claim-size variant as an
ablation, without replacing the original baseline with it:

| Plan, 1 task/claim, 16 slots | Loopback s/update | Own-baseline speedup | Home s/update | Own-baseline speedup |
|---|---:|---:|---:|---:|
| Reference | 2.0722 | 4.34× | 11.1208 | 13.18× |
| Fused, no persistent X cache | 0.3122 | 8.48× | 2.4313 | 7.98× |
| Fused, persistent X | 0.3145 | 8.39× | 2.4161 | 7.75× |
| Fused, persistent X and modeled cache affinity | 0.3491 | 7.56× | 2.2550 | 8.30× |

The reference claim-size-1 home result looks nearly linear because its
one-slot time is 146.5376 seconds, versus 52.6353 seconds with claim size 16.
It is slower in absolute time at 16 slots too. A larger relative speedup
alone is therefore insufficient evidence of an improvement. Among these
settings, the fastest fused home result at 16 slots still uses claim size 16.

The no-persistent-cache and persistent-X rows isolate cross-update caching. Cache
reuse for U/V within an update remains enabled in both fused variants, so
comparing the reference and fused plans at claim size 1 still combines
graph changes with within-update factor caching. The report calls the
cross-update ablation `fused_batch1_no_persistent_cache`.

Cache affinity is an optional **simulator-only scheduling policy**, not a live
coordinator implementation. Here it reduces home-link time but worsens
loopback time. Existing live transport caching already supports immutable
input keys; a shape-only plan's synthetic X version must be replaced by a
version identifying the actual immutable dataset before binding real data.
U/V caches remain scoped to an update. Reports preserve cold and every-update
times: the third update does not imply that every worker has a warm X cache.

## Larger fixed workload: 4,096 × 4,096, rank 32

`runs/factorization-comparison/fused-4096.json` uses 64-row shards, one task
per claim, persistent X, and no cache affinity. Its **358 tasks/update** give
the scheduler more independent work. All rows use that same graph and report
the median third update across seeds 0, 1 and 2.

| Modeled slots | Loopback s/update | Speedup | Efficiency | Home-link s/update | Speedup | Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.4662 | 1.00× | 100.0% | 105.0354 | 1.00× | 100.0% |
| 2 | 8.7774 | 1.99× | 99.5% | 55.4984 | 1.89× | 94.6% |
| 4 | 4.5190 | 3.87× | 96.6% | 30.9311 | 3.40× | 84.9% |
| 8 | 2.4026 | 7.27× | 90.9% | 17.3605 | 6.05× | 75.6% |
| 16 | 1.4362 | 12.16× | 76.0% | 10.7075 | 9.81× | 61.3% |

Under the declared 80% threshold, loopback is near-linear through 8 slots,
and home links through 4. Neither reaches that threshold at 16 slots. This
is a separate, larger-workload strong-scaling study; its speedup is not a
comparison against the smaller matrix.

The reference graph at this size fails existing task admission: its
whole-matrix loss reduction exceeds the memory budget, and its backward
broadcast exceeds the 8 MiB task-output cap. The study does not bypass those
limits or report a reference-versus-fused timing ratio for this larger case.

## Cost and interpretation limits

Fused timing is decomposed into **three existing matmul costs plus pointwise
passes and a fallback dispatch intercept**. All constituent intercepts and
coefficients are retained; none were refitted. This decomposition is an
uncalibrated modeling assumption, **not a proven upper bound** on the real
kernel. In particular, the browser CPU implementation has no factorization
throughput calibration. Numerical parity establishes the update equations,
not speed, convergence, or hardware efficiency.

The JSON reports retain the cost table, coverage warnings, reusable-input
sets, cache policy, source hashes and graph hashes for reproducibility.

The fused graph removes the reference's whole-matrix residual transfers and
loss/broadcast bottleneck, but coordinator service, claim granularity,
gradient reductions, cache placement and bandwidth still limit scaling.
Metadata assembly is modeled without CPU/storage cost, worker stage times
overlap, and physical host contention remains unmodeled. Every timing and
speedup in this document is a modeled prediction; no live training or
service-based benchmark was run on this machine.
