# A protein-space factorization on WebGPU workers with a memory guarantee

| | |
|---|---|
| **Date** | 2026-10-29 (plan; stages A to D run the same day, D as an engineering test; bytes per tile cut twice the same evening, tile groups then a float32 wire; E not started) |
| **Path** | Sharded 2D factorization (experiments 46 and 47) with browser WebGPU workers: WGSL kernels for `factorization_tile`, `add_many` and `sgd`; per-worker advertised memory budgets enforced on the coordinator and again in the worker |
| **Result** | Stage A passed on hardware WebGPU (AMD Radeon 680M, RDNA 2, headless Chromium over Vulkan): the tile kernel with X generated on the GPU matches NumPy float64 to 1.4e-7 of each gradient block's scale and 3.5e-8 in the loss (gates 1e-5 and 1e-6), and its 64-bit noise hash is bit-exact on 576 entries. A 2,048 x 2,048 rank-64 tile takes about 19 ms, X's generation and the upload included. Stage B passed: headless browser tabs (1, 2 and 4, sharing that GPU) run 10 updates of 8,192 x 8,192 with every loss within 3.4e-8 of the dense float64 reference (gate 1e-5), all 160 tiles on the GPU, none rejected; losses are bit-identical across tab counts, and an update takes 1.14, 0.66 and 0.46 s. Stage C passed: memory budgets are counted by one formula in Python and JavaScript (equal on every task of the plan), the coordinator offers a tab only tasks within its budget, and the tab refuses the rest; 13 oversized, lying or malformed tasks were refused with no GPU buffer created, a tile at exactly its budget ran with GPU allocations equal to the bound, and two tabs with 16 MiB and 1 GiB budgets reached the same losses as stage B with every tile on the large tab. (B and C used the synthetic problem's learning rate 500 and regularization 0.001, under which the loss falls only because U and V decay; they show correct, identical computation, not fitting.) Stage D, an engineering test on real data, passed: 4 tabs factor the similarity of 100,000 Swiss-Prot proteins' ProtT5 embeddings (64 PCA components) for 30 updates, explaining 78.4% of ||X||^2 and matching exact float64 to 2.9e-8 at every update, 53.8 s per update; each tab moves about 1.5 GB each way per update. Tile groups (one task per 5 x 5 tiles, returning its shards' summed gradients) then cut that to 440 MB down and 358 MB up per tab per update (0.74 / 0.60 MB per tile against 2.61 / 2.47) with identical losses, at 46.8 s per update: this single GPU's 2,401 tiles x 19 ms. A float32 wire (frames carry float32 both ways, rounded once per link) halves that exactly, 222 / 179 MB per tab, and admits 11 x 11 groups: 105 to 174 / 82 to 104 MB per tab per update (0.22 / 0.15 MB per tile, 12 to 16 times less than stage D), losses within 1e-7 of exact float64, explained share unchanged to six digits, 45.9 s per update. 5 of 750 group results were retried on tab cache misses (every loss matched) |
| **Evidence** | `runs/experiments/048/stage-a/20261029-133230/report.json`, `runs/experiments/048/stage-b/20261029-135243/report.json`, `runs/experiments/048/stage-c/20261029-140950/report.json`, `runs/experiments/048/stage-c-training/20261029-140330/report.json`, `runs/experiments/048/stage-d/20261029-144754/report.json` (8,192 proteins), `runs/experiments/048/stage-d/20261029-144853/report.json` (100,000), `runs/experiments/048/stage-d/20261029-162016/report.json` (8,192 with 2 x 2 groups), `runs/experiments/048/stage-d/20261029-155546/report.json` (100,000 with 5 x 5 groups), `runs/experiments/048/stage-d-float32/20261029-183549/report.json` (8,192, float32 wire), `runs/experiments/048/stage-d-float32/20261029-183637/report.json` (100,000, 5 x 5, float32 wire), `runs/experiments/048/stage-d-float32/20261029-190414/report.json` (100,000, 11 x 11, float32 wire) (local, not in git); `tests/test_float32_wire.py`, `tests/test_memory_budget.py`, `tests/test_factorization_webgpu.py` |

## Question

Can volunteer browsers factor a dense similarity matrix over open protein embeddings (about 10^6 x 10^6, rank 64) with the loss falling over updates, while no worker ever allocates more than the memory it advertised?

## Why this problem

Open protein collections (UniRef50, the ESM Metagenomic Atlas, AlphaFold DB) hold hundreds of millions of entries, many of unknown function. A rank-64 factorization of an all-vs-all embedding similarity is a map of protein space. The matrix is dense and defined by a function, which is the case the sharded plan handles without storing X (experiment 46). Not checked: whether such a map tells a biologist anything that Foldseek or MMseqs2 clustering does not. Ask someone in the field before stage D.

## What "guarantee" means in a browser

A tab can be throttled or killed and WebGPU can report out-of-memory late, so the claim is layered, not absolute:

1. **Declared budget.** The worker advertises it at hello: a fraction (default a quarter) of what the adapter reports (`maxBufferSize`, `maxStorageBufferBindingSize`), under an absolute ceiling.
2. **Admission.** Each task's working set (inputs, outputs, scratch) is computed from shapes, as `admit_plan` and `max_task_bytes` do now; a task goes only to a worker whose budget covers it.
3. **Enforcement.** The worker recomputes the byte count from the task and refuses over-budget work before allocating. Allocation runs inside an error scope; an out-of-memory or `device.lost` fails the lease and the worker is not given that task again.

Tests prove the check, not the browser: they show that oversized or malformed tasks are refused without allocating.

## Stage A (2026-10-29): the tile kernel on WebGPU

`webapp/static/factorization_gpu.mjs` computes one `factorization_tile` in three compute passes: the residual E = U V^T - X with X generated on the GPU, then dU = E V / total with each row's sum of squares, then dV = E^T U / total. The reductions over a tile's width use compensated (Kahan) sums, guarded against fast math as the matmul kernel is. X is a low-rank product plus hashed noise (`x_values`): the latent rows of X's two factors come from NumPy's generator, which a GPU cannot reproduce, so they travel as inputs (latent x (rows + columns) values, 128 KiB for a 2,048 tile instead of 32 MiB of X); the splitmix64 noise is computed on the GPU in 32-bit words, as WGSL has no 64-bit integers.

The harness (`ygaraty experiment factorization_webgpu`) runs headless Chromium and compares with `runtime/factorization_kernels.py` in float64. Gates, fixed before the first run: every hash word equal; per case, the largest error in dU and in dV at most 1e-5 of that block's largest reference value; the loss within 1e-6 relative.

| Case | Tile, rank | dU error | dV error | Loss error | GPU time |
|---|---|---:|---:|---:|---:|
| Experiment 46's tile (offset in 122,880) | 2,048 x 2,048, 64 | 9.7e-8 | 9.7e-8 | 3.5e-8 | 16.8 ms |
| Last rows of a 10^6 matrix (index above 2^32) | 512 x 512, 64 | 1.4e-7 | 8.3e-8 | 3.5e-8 | 3.0 ms |
| Ragged, latent 1, no noise | 1,000 x 300, 1 | 1.3e-7 | 8.1e-8 | 1.4e-9 | 2.4 ms |
| Large tile | 4,096 x 4,096, 64 | 1.0e-7 | 9.2e-8 | 3.4e-8 | 53.5 ms |

- **Hash:** 576 of 576 words equal (64 entries x 3 matrix widths up to 2^32 - 1 x 3 seeds).
- **Seeds:** a first run used seed 2^62 - 1, which the CPU spec allows, and missed 192 words: a JSON number is a double in a browser, so the seed arrived rounded. Browser jobs must keep seeds below 2^53; the kernel now refuses larger ones. Stage B's job has to enforce this.
- **Time:** GPU time is the three passes and the readback, X's generation included; the upload adds 2 to 3 ms. Experiment 46 measured about 290 ms for a 2,048 tile on one CPU core (240 ms generating X), so this GPU is roughly 15 times one core. The harness's own NumPy column (218 ms) is not comparable: X is given there, and BLAS uses every core.
- **Getting the GPU:** headless Chromium on Linux falls back to SwiftShader (software) without `--enable-features=Vulkan --use-webgpu-adapter=default --ignore-gpu-blocklist --enable-gpu --use-gl=angle`; the harness uses these by default and refuses a software adapter unless told otherwise. On SwiftShader the same gates pass, 60 to 100 times slower.
- **Not done in stage A:** `add_many` and `sgd` stay on the browser's float64 CPU path, which already has them. They are elementwise on 2,048 x 64 shards, where a GPU dispatch costs more than the work. A second GPU vendor (Apple, on the M5) is still to run. The memory budget (work package 3) belongs to stage C.

## Stage B (2026-10-29): browser tabs train 8,192 x 8,192

`ygaraty experiment factorization_browser_training --tabs 1 2 4` starts the webapp coordinator with a `ShardedFactorizationJob` on loopback and opens each tab as its own headless Chromium process on the contributor page (`/work/<job>#token=...&autostart=1`), with stage A's GPU flags. What changed to make it work:

- **Latent rows as leaves.** `build_sharded_plan(..., latent_inputs=True)` adds leaves `XL/i` and `XR/j` (the tile's rows of X's latent factors, 8 columns each, never updated), and a tile reads them before U_i and V_j. The CPU kernel accepts the same four inputs and builds X with `x_from_latent`, the code `x_values` now uses, so its tiles equal the generated-X plan's bit for bit (tested). Tabs cache the latent rows like any other tile.
- **Routing.** `ComputeSDK.execute` sends a four-input tile with an `x_spec` to `factorization_gpu.mjs` (imported only then); a tab without WebGPU refuses it. Reductions, regularization and SGD stay on the tab's float64 CPU code.
- **The job** takes `precision='webgpu'`, without which the coordinator refuses GPU results, as the parity recipe does.
- **Autostart.** The contributor page starts by itself when its link carries `autostart=1`, so headless tabs need no click.

Problem: rank 64, blocks 2,048 (16 tiles), learning rate 500, regularization 0.001, seed 2026, 10 updates. Gates, fixed before the first run: the job completes with no rejected result; every tile result comes from the GPU; at every update the loss is within 1e-5 relative of the dense float64 reference (experiment 47's check).

| Tabs | Loss, update 1 -> 10 | Largest relative error | GPU tiles | Seconds per update (after the first) | Coordinator sent / received |
|---:|---|---:|---:|---:|---:|
| 1 | 5.741101 -> 0.498056 | 3.4e-8 | 160 / 160 | 1.14 | 673 / 671 MB |
| 2 | 5.741101 -> 0.498056 | 3.4e-8 | 160 / 160 | 0.66 | 758 / 671 MB |
| 4 | 5.741101 -> 0.498056 | 3.4e-8 | 160 / 160 | 0.46 | 893 / 671 MB |

- **Same answer at every tab count.** The losses are bit-identical across 1, 2 and 4 tabs: a tile's GPU result does not depend on which tab ran it, and the reductions add in a fixed order.
- **Time.** The four tabs share one GPU, yet 4 tabs are 2.5 times faster than 1: the GPU part of an update is about 0.3 s (16 tiles), and the rest (claims, the 2 MiB float64 result of each tile, the reductions in JavaScript) overlaps across tabs. For scale, the CPU ladder of experiment 46 took 2.1 s per update at this size with 4 Python workers on the workstation. Not profiled further.
- **Bytes.** About 67 MB each way per update: each tile returns a 2 MiB float64 result, and since tabs also run the reductions, those results go back out to them. More tabs send more because each tab's cache must fill.
- **Not shown:** that the tabs used the hardware adapter is inferred, not recorded: they ran with stage A's flags, which gave the Radeon, and SwiftShader is 60 to 100 times slower on these tiles, which would have made an update take far longer than 1.14 s. The contributor page should report its adapter (for stage C anyway). Loopback only; one GPU; no memory budget yet.

## Stage C (2026-10-29): memory budgets

**The count.** `runtime/budget.py` and `webapp/static/budget.mjs` give the bytes one task may make a tab allocate, GPU and JavaScript together: the inputs as received and the float64 result for every op; result-sized scratch for the elementwise ops; for a GPU tile, float32 copies of the inputs in JavaScript and on the GPU, the float32 residual, the GPU output, its readback and its copy, and 64 bytes of parameters. An op not in the table (only the factorization's six ops are) has no bound and is refused on a job that enforces budgets. A test runs both on every task of the 8,192 plan and of a ragged plan, and on seven malformed tasks: the numbers are equal, and both refuse every malformed one.

**The protocol.** A tab advertises `budget_bytes` with its claims: by default a quarter of the adapter's buffer limit, at most 1 GiB (1,073,741,823 bytes on the Radeon), or what its link sets (`#budget=BYTES`). It also reports its adapter, which the job records. A job with `enforce_budgets` offers a budgeted worker only ready tasks within its budget and stamps each with `memory_bytes`. The tab's `admitTask` recomputes the count before copying an input or creating a buffer, and refuses the task if its count differs from the coordinator's or exceeds its budget; it also refuses a reply longer than its tasks could fit before reading the body. Jobs that do not enforce budgets, and workers that advertise none (the Python workers), behave as before. The device is now opened with the adapter's buffer limits instead of WebGPU's defaults.

**Adversarial cases** (`ygaraty experiment factorization_memory`, the Radeon, every GPU buffer counted by wrapping `createBuffer`). Gates, fixed before the run: every case meant to be refused is refused with no GPU buffer created; every admitted case stays within its bound's GPU share; the exactly fitting tile matches NumPy to 1e-5.

| Case | Outcome | GPU buffers |
|---|---|---:|
| Tile at exactly its budget (26,763,840 bytes) | ran; GPU bytes 20,070,464 = the bound's GPU share; error 1.2e-7 | 8 |
| Same tile, budget one byte smaller | refused | 0 |
| Reduction under a 1 KiB budget | refused | 0 |
| Coordinator understates the size 100 times | refused: counts differ | 0 |
| 65,536 x 65,536 tile (17.5 GB) under 1 GiB | refused | 0 |
| Negative or fractional shape; missing input shape; three inputs; no x_spec | refused | 0 |
| `matmul` (no bound) | refused | 0 |
| No budget on a budgeted task | refused | 0 |
| Seed 2^60 | admitted by size, refused by the kernel's spec check before allocating | 0 |
| Task from a job that does not enforce budgets | ran, as before | 8 |

**Training with budgets.** Two tabs on the 8,192 problem of stage B with `--enforce-budgets --budgets 16777216 default`. Every task other than a tile needs at most 6.3 MB and every tile 26.8 MB, so 16 MiB fits everything but tiles. The 16 MiB tab completed 399 tasks, none on the GPU; the 1 GiB tab ran all 160 tiles (and 61 other tasks). The job finished with no rejection and the losses are bit-identical to stage B's. Both tabs reported `amd`, `rdna-2`, `isFallbackAdapter: false`, which settles stage B's open point that the tabs used the hardware adapter.

**What this does not show.** The JavaScript share of the bound is enforced by construction (admission before allocation) but not measured: a page cannot read its own heap without cross-origin isolation. The response body is held whole before admission; the Content-Length check bounds it, but a gzipped reply's length is its compressed size. An out-of-memory error from the GPU fails the task inside the kernel's error scope, but no case forced one. Losing the device mid-task (`device.lost`) leaves the tab without WebGPU, so it refuses later tiles; not exercised here.

## Stage D (2026-10-29): 100,000 real proteins, as an engineering test

Run as an engineering test, as decided: whether such a map tells a biologist anything Foldseek or MMseqs2 clustering does not is still unasked, and nothing here claims it does.

**Data.** UniProt's ProtT5 per-protein embeddings for Swiss-Prot (`knowledgebase/embeddings/uniprot_sprot/per-protein.h5`, release of 2026-09-03: 1,386,097,056 bytes, SHA-256 `404f807d7360d3ed1d400988307beaa4f609e10d0fbc161ee1cc965fc40810dc`; 575,736 proteins, 1,024 float16 values each). `ygaraty experiment protein_embeddings` takes a seeded sample, L2-normalizes and centres it, projects it on its top 64 principal components (68% of the variance for both samples) and scales it so that X = A A^T / 8 has off-diagonal entries of about unit spread. X is the similarity of the proteins in that 64-dimensional space, so it has rank 64 exactly, and a rank-64 factorization can fit it completely. The sample of 100,000 takes 16 s to prepare; h5py is the new optional extra `protein`.

**How X reaches the tiles.** The job takes the rows of A as the latent leaves (`latent_values`) with noise 0, so every tile, on a tab's GPU or a CPU worker, builds its block of X from its two shards of A exactly as for the synthetic problem.

**Fitting, not decaying.** A first run at 8,192 with the synthetic settings (rate 500, regularization 0.001) passed its gate and fitted nothing: the data loss stayed at mean(X^2) / 2 while the regularization term decayed. That is also what stages B and C, and experiment 46, measured (corrected there). Gradients are divided by N^2, so the step must scale as N^2 / lambda_max(X): on the dense 8,192 problem, rates of 5,000 and 10,000 fit, and 20,000 (1.5 N^2 / lambda_max) and above diverge. The harness now sets 0.75 N^2 / lambda_max and no regularization for real data (8,192: 9,912; 100,000: 121,535). The reference is exact float64 gradient descent computed in factor space (the loss and gradients of X = A A^T / 8 need only 64 x 64 products), equal to the dense one to 2.4e-16 at 8,192 and cheap at any size.

Gates, fixed before the first run: the job completes with no rejected result, every tile on the GPU, the loss falls at every update and stays within 1e-5 of the reference; bytes per tab recorded.

| Proteins | Tabs | Updates | Explained share of ||X||^2 | Largest error vs float64 | GPU tiles | Seconds per update | Per tab per update, down / up |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8,192 | 2 | 50 | 0.935 | 1.9e-9 | 800 | 0.55 | 29.5 / 25.2 MB |
| 100,000 | 4 | 30 | 0.784 | 2.9e-8 | 72,030 | 53.8 | 1,566 / 1,486 MB |

- The loss fell at every update in both runs and matches the reference throughout; the reference for 100,000 reaches 0.925 explained at 60 updates.
- 100,000 proteins is 2,401 tiles per update (49 shards a side). At 53.8 s per update the four tabs take 22 ms per tile of wall time, against stage B's 29 ms (16 tiles, 0.46 s): the run scales with the number of tiles, and the shared GPU (about 19 ms per tile, stage A) is close to the limit. The coordinator used about one core.
- Bytes: about 6.2 GB each way per update in all, 2.6 MB per tile, almost all of it the 2 MiB float64 tile result going up and coming back down to the tab that reduces it. On a 20 Mb/s uplink one tab's 1.49 GB would take about 10 minutes per update. This, not compute, is what stage E must solve.

## Bytes per tile (2026-10-29): tile groups

**Why this first.** Stage D's report, per tile and per update: 2.47 MB up, of which the tile's float64 result, (2,048 + 2,048 + 1) x 64 x 8 = 2,097,664 bytes, is 85%; 2.61 MB down, mostly that same result re-sent to the tab that reduces it, plus the shards. Summed over an update that is 2 N^2 rank x 8 / b bytes each way: it shrinks only with the tile side b, and b is capped by the GPU's memory (the residual is b^2 x 4 bytes) and by the 8 MiB result limit. Compute per tile grows as b^2 while its result grows as b, so a worker on a 20 Mb/s uplink spends about 0.8 s sending a tile it computed in 0.02 s: 40 times link-bound before any machine is remote. Nothing else in stage E matters until this ratio moves.

**The levers, and why groups.** Float32 results halve the bytes once and cost the dense check's exactness. Sending each tile result once instead of twice (reductions on the coordinator) halves the downloads but moves the reductions, and the design keeps the coordinator out of the arithmetic. A bigger tile is the real lever, bytes being 1/b, but it costs b^2 memory, so it stops at 4,096 here. A **tile group** gives the bigger tile's bytes at the small tile's memory: one task computes the p x q tiles of p row shards and q column shards one at a time and returns the summed gradient of each of its p + q shards, (p + q) bands instead of 2 p q, so bytes follow the group's perimeter while compute follows its area. 5 x 5 is one fifth of the bytes; the memory stays that of one tile plus the group's inputs. The reductions then read one band per group. The result's bands are summed in float64 in a fixed order, so a group is deterministic and a worker-independent function of its inputs, as a tile is.

**What changed.** `build_sharded_plan(group=(p, q))` emits `tiles/I/J` tasks with op `factorization_tiles`; the CPU kernel (`_tile_group`), the GPU kernel (`factorizationTilesGPU`, which loops `factorizationTileGPU`), the memory bounds in Python and JavaScript (the group's inputs and result plus one tile's GPU share), the locality audit, the cost model and the task-count guard know the op. A group's result may reach 12 MiB (one request body), above the usual 8 MiB cap, so at 2,048 x 64 bands p + q is at most 11. The harness takes `--group`, and now requires every tab to have joined and records why any result was retried; an earlier 8,192 run passed its gates with one of two tabs idle and misreported bytes per tab, which this would have caught.

**Measured** (same data, seeds, tabs, updates and GPU as stage D; losses bit-identical to the single-tile runs, within 2.9e-8 of the reference):

| Proteins, tabs | Tiles per update | Tile tasks per update | Down / up per tile | Down / up per tab per update | Seconds per update |
|---|---:|---:|---|---|---:|
| 8,192, 2, single tiles | 16 | 16 | 3.68 / 3.15 MB | 29.5 / 25.2 MB | 0.55 |
| 8,192, 2, 2 x 2 groups | 16 | 4 | 2.50 / 2.10 MB | 20.1 / 16.8 MB | 0.48 |
| 100,000, 4, single tiles | 2,401 | 2,401 | 2.61 / 2.47 MB | 1,566 / 1,486 MB | 53.8 |
| 100,000, 4, 5 x 5 groups | 2,401 | 100 | 0.74 / 0.60 MB | 440 / 358 MB | 46.8 |

- **Bytes:** 3.5 times less down and 4.1 times less up at 100,000, against one fifth for the results alone: the rest is the shards a group reads (20 of 1 MiB each, mostly cached) and the reductions' own traffic. A tab's 358 MB upload would take 2.4 minutes on a 20 Mb/s uplink, against 10 minutes before and 46 s of GPU time: still link-bound, by 3 times rather than 13.
- **Time:** 53.8 to 46.8 s, and 2,401 tiles x 19.4 ms (stage A's time per tile) is 46.6 s. The four tabs saturate this one GPU; on loopback the single-tile run's extra 7 s was framing and task overhead, which the groups removed. Bytes will only show in time on real links (stage E).
- **5 of 3,000 group results were rejected and retried** (so the run failed its "no rejection" gate while every loss matched). A rejection with a retry is a worker-reported error, and the only one a tab raises is a cache miss: a group reads 20 shards of 1 MiB, the tab's LRU cache holds 256 MiB, and with the previous update's U and V still cached the latent rows can be evicted before the coordinator learns it. The reasons were not recorded in that run (they are now), so this is the likely cause, not a shown one. A larger tab cache, or dropping the previous update's shards first, would remove it.
- **At 1,000,000 proteins** the plan builder refused single tiles outright (239,121 tile tasks against its 100,000-task guard); with 5 x 5 groups it builds: 9,604 tile tasks, 30,683 tasks in all, 291 MB of plan in Python per update. Groups make the million-protein plan expressible; whether one coordinator can drive 30,000 tasks per update is the task-store question (historical internal note; not part of this repository).

**Still on the table for stage E:** ~~float32 results (another half)~~ (done below), a larger cache per tab, and workers on real links. Before real links, the coordinator must leave the byte path: browser workers on the relay, planned as [experiment 50](050-browser-workers-on-the-relay.md). ~~The next lever beyond groups is the group's side: p + q is capped by the 12 MiB result; float32 results would allow 11 x 11.~~

## Bytes per tile, second cut (2026-10-29): a float32 wire

**Why next.** After groups, the next item on the scaling list was the coordinator: one Python process every byte passes through, which moved at most about 227 MB/s during the single-tile runs, so it caps an update at about 14 s at 100,000 proteins however many GPUs join, and at 1,000,000 with 5 x 5 groups it would move about 320 GB per update, 24 minutes, against 75 GPU-minutes of tiles. Halving every byte is the cheapest cut of both numbers before the relay takes the bytes off the coordinator altogether, and it shrinks what the relay will later carry too. The GPU tile already computes from float32 copies of its inputs, so sending them as float32 changes nothing in a tile; and a group's result cap (12 MiB in one request body) counts bytes, so float32 results admit a group of twice the side sum: 11 x 11 instead of 5 x 5 at 2,048 x 64, bytes per tile another 2.2 times down.

**What changed.** A binary frame may name its wire: `wire: "<f4"` in the header, float32 arrays after it (`transports/http/frames.py`); absent, float64 as before. A job whose config says `wire: "float32"` (`ShardedFactorizationJob(wire=...)`, `Job.wire_dtype`) puts it in every claim reply, and the server narrows each input once as it leaves (`encode_block`, which refuses a value beyond float32's range rather than send infinity); both workers (the browser SDK and `http-worker`) widen on arrival, compute in float64 as before, and send results in the same wire; a binary result in another wire is refused. The store stays float64, so each value is rounded once per link: U and V go out rounded, the SGD update's result comes back rounded. A sampled recomputation reads the inputs as the worker got them and allows the result's rounding (rtol 2.5e-7). The plan's result cap takes the wire's bytes per value (`wire_bytes`), so `group=11` builds on a float32 wire and is refused on a float64 one. The memory bound is unchanged (inputs are float64 once received). The harness takes `--wire float32`.

**What it costs.** The losses are no longer bit-identical to the float64 runs: U and V are float32-rounded once per update, so the gate is the tolerance (1e-5 of the exact reference), as declared.

**Measured** (same data, seeds, tabs, updates and GPU as before):

| Proteins, tabs, groups | Wire | Down / up per tile | Down / up per tab per update | Seconds per update | Largest error vs float64 | Explained share |
|---|---|---|---|---:|---:|---:|
| 8,192, 2, 2 x 2 | float64 | 2.50 / 2.10 MB | 20.1 / 16.8 MB | 0.48 | 1.9e-9 | 0.93448157 |
| 8,192, 2, 2 x 2 | float32 | 1.26 / 1.05 MB | 10.1 / 8.4 MB | 0.50 | 7.0e-8 | 0.93448157 |
| 100,000, 4, 5 x 5 | float64 | 0.74 / 0.60 MB | 440 / 358 MB | 46.8 | 2.9e-8 | 0.784 |
| 100,000, 4, 5 x 5 | float32 | 0.37 / 0.30 MB | 222 / 179 MB | 46.0 | 9.7e-8 | 0.7835 |
| 100,000, 4, 11 x 11 | float32 | 0.22 / 0.15 MB | 105 to 174 / 82 to 104 MB | 45.9 | 7.9e-8 | 0.7835 |

- **Bytes:** exactly half. The 5 x 5 run moved 26.7 GB down and 21.5 GB up in all against 53.3 and 43.0 GB on float64, ratios 1.996 and 2.000: on this plan every byte but the headers is a float. Per tab per update 222 / 179 MB, from 1,566 / 1,486 MB before groups: 7 and 8.3 times less in the two cuts together. A tab's 179 MB upload is 72 s on a 20 Mb/s uplink, against 46 s of GPU time: the first setting where a home uplink is within a factor of two of the compute.
- **Precision:** the losses differ from the float64-wire run by at most 8.6e-8 relative over 30 updates, and from the exact reference by 9.7e-8 (float64 wire: 2.9e-8); the explained share is 0.783476 on both wires. The rounding of U and V to float32 once per update does not accumulate visibly in 30 updates. Bit-identity across tab counts, which stages B and D had, is lost by construction.
- **Time:** 46.0 against 46.8 s per update, the GPU's 2,401 x 19 ms either way.
- **11 x 11 groups** (25 tile tasks per update, 121 tiles each, 11.5 MiB float32 results): 0.22 / 0.15 MB per tile, 3.4 and 4 times less than single tiles on this wire would give, 12 and 16 times less than stage D's 2.61 / 2.47. Per tab the bytes are now uneven (105 to 174 MB down, 82 to 104 MB up): 25 tasks over 4 tabs, and the tab that takes 7 of them also downloads their shards. 15.6 GB down and 10.8 GB up in all per 30 updates, from 53.3 / 43.0 GB with 5 x 5 groups on float64 and 188 / 178 GB with single tiles. Same 45.9 s per update; the largest error 7.9e-8.
- **Rejections: cache misses, 5 of 750 group results**, as the float64 and float32 5 x 5 runs also had (5 of 3,000 each) with every loss matching. The harness had kept only the job's last 300 events and lost the reasons; it now records every retry with its update, and the 11 x 11 run names them: a tab reports `cache miss: <key>` at updates 2, 4, 4, 15 and 29, on tile groups only. The coordinator sent the group's inputs as "cached" because its hints said the tab held them, and the tab's 256 MiB LRU cache had evicted one: at 100,000 proteins an update's U, V and latent shards are 110 MiB of float64 in the cache, and with the previous update's still held, the cache sits at its edge. (Chromium's crash handler also wrote lock-file errors in all four tabs at the same instant, about 10 minutes into each run, with no retry at that update: unrelated.) A retry costs one group's compute, 2.3 s at 11 x 11. The runs are marked FAIL by the "no rejection" gate alone; every other gate held. *Fixed the same night (experiment 50's SDK work):* the mechanism is within one claim reply, an earlier task's inputs evicting a later task's "cached" tile (the Python worker's `fetch_relayed` comments on the same race); the SDK's cache now pins the keys a reply says the tab holds while its tasks run, and its default grew to the Python worker's 512 MiB. The 5 x 5 float32 run repeated with that SDK: 0 rejected, losses bit-identical, PASS (`runs/experiments/048/stage-d-float32/20261029-201308/report.json`). That run also shows two slow updates (143 and 99 s against a 45.9 s median) at the 10-minute mark where every run's tab logs show Chromium's crash handler writing lock files; the cause of that stall was not identified.

## Work packages

1. WGSL kernels for `factorization_tile`, `add_many` and `sgd`, matching the NumPy kernels to float32 tolerance. Experiments 14 and 15 found precision problems, so reductions use compensated sums.
2. A browser job and tile source. Stages A to C use seeded synthetic tiles (`x_spec`), so no data moves; stage D shards real embeddings from open storage.
3. One byte-count function shared by coordinator and worker, with a test that both agree.
4. Budget-aware routing: tiles sized to the smallest budget supported (256 to 512 MiB), tasks offered only to workers whose budget fits.

## Stages and gates

| Stage | What | Passes when |
|---|---|---|
| A | Browser kernels against NumPy on one tile | maximum error within float32 tolerance |
| B | 8,192 x 8,192, loopback, 1, 2 and 4 browser tabs | loss matches the experiment 47 reference |
| C | Adversarial memory test: oversized and malformed tasks, tiny budgets | no over-budget allocation; clean rejections |
| D | Protein pilot: 100k to 1M proteins, embeddings sharded to workers | loss falls over many updates; bytes per worker per update measured (passed at 100k as an engineering test) |
| E | Volunteer-like links (home-link shaper), then real volunteers | time per update compared with the simulator |

Start with A and C: local, cheap, and they decide whether the rest is worth doing.

## Risks

- Real embeddings must be downloaded by each worker; that may dominate every other cost, and the synthetic runs cannot predict it.
- Browser limits differ; mobile and integrated GPUs set the floor on tile size.
- Experiment 46 has since trained 11 updates at the larger-than-memory scale (122,880 per side), but on CPU workers; its workers settle near 1.2 GiB each once their tile cache fills, so a browser budget must count any cache it keeps.
- The coordinator is the likely limit at many workers (experiments 46 and 47), so stage E may be coordinator-bound before it is link-bound.
- Each tile returned a 2 MiB float64 result, re-downloaded by the tab that reduces it; tile groups cut this to one fifth per 5 x 5 group, and the float32 wire with 11 x 11 groups to about a sixteenth (above). On a 20 Mb/s uplink a tab's 82 to 104 MB per update is 33 to 42 s against 46 s of GPU time: the first setting where the uplink is not the longer side, on paper.
