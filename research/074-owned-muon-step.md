# Muon's owner step adds 0.28 s to the pool's update, small enough to keep its gain

| | |
|---|---|
| **Conclusion** | **Works:** Muon's owner step adds 0.28 s per update, so the update takes 1.06× Adam's, against 1.3× allowed (measured step, bounded, workstation). |
| **Status** | executed (stage 1); stage 3 executed as [experiment 91](091-muon-pool-time-to-loss.md) |
| **ID** | M-2 stage 1 (experiment 74) |
| **Category** | training |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-03 |
| **Path** | Muon added to the owners' `tf_owned_step` (`optimizer="muon"`). Timed one owned 30.3M layer's whole step: encoder and decoder layers, 8 bands, float64 NumPy, at 1, 2 and 4 BLAS threads on separate physical cores, 2 rotated repeats of 15 steps each |
| **Result** | Median step at 4 threads: Adam 55 and 68 ms (encoder, decoder); Muon with 5 Newton–Schulz steps 261 and 351 ms; Muon with 2 steps 146 and 192 ms. The larger extra time is the decoder's 0.283 s, which bounds the update at 4.92 s, **1.06×** Adam's 4.64 s (gate ≤1.3×). The bound holds at every setting measured, at worst 1.16× (1 thread, 5 steps). With experiment 67's 40–49% of the pairs, Muon would reach Adam's loss in about 42–52% of its wall time on the pool. Free; about an hour |
| **Evidence** | [`evidence/training/m2-owned-muon-step-2026-11-03.json`](evidence/training/m2-owned-muon-step-2026-11-03.json) (medians per layer, arm and thread count; the gate per setting; the benchmark's hash); raw timings, `bench.py` and `drive.sh` in `runs/experiments/M-2/omarchy-20261103-175323/` (local, not in git). Code: `orthogonalize` and the Muon branch of `OwnedLayers.execute` in `ygaraty/runtime/transformer_kernels.py`; `tests/test_owned_muon.py` |

## Question

[Experiment 67](067-muon-owned-layers.md) found that Muon reaches Adam's held-out loss with 40–49% of the pairs. It failed on CPU time: in a single-machine update of about 0.59 s, Newton–Schulz added 0.46 s. On the pool, each owner steps only its own layer, the owners step at the same time, and an update takes 4.64 s ([experiment 61](061-handoffs-30m-digitalocean.md)).

**How long does one owned layer's Muon step take in the owners' float64 NumPy, and how much can it add to the pool's update?**

## Prediction and gate (frozen in [M-2](091-muon-pool-time-to-loss.md) before the code)

**Prediction:** 0.1–0.5 s for 5 Newton–Schulz steps on 2 cores, adding 2–11% to an update.

**Gate:**
- **Correctness:** one owned Muon step equals the reference's Muon on the layer's matrices, with Adam on its norms and biases, within relative error 1e-10 in float64.
- **Go for stage 3:** the predicted update is 4.64 s + Δ, at most 1.3× Adam's (Δ ≤ 1.39 s). Δ is the larger extra step time of the encoder and decoder layers at 4 threads, the setting experiment 61's strong owners used. The bound assumes the whole step lies on the critical path: encoder layer 0's step follows the update's last backward task, and the next forward waits for it.

## Protocol

- **Code:** `tf_owned_step` with `optimizer="muon"` follows `ygaraty/reference/muon.py`: momentum 0.95 with Nesterov, the quintic Newton–Schulz (3.4445, −4.7750, 2.0315) on the smaller side, scaled by max(1, rows/cols)^0.5. It applies to every matrix with both sides above 1 and keeps its buffer in the layer's first moment. Norms and biases keep the Adam step unchanged. The planner doesn't emit the setting yet; that is stage 3's work.
- **Correctness test:** `tests/test_owned_muon.py`, two updates on an encoder layer (5 steps) and a decoder layer (2 steps), against the torch reference.
- **Timing:** `bench.py` loads one layer with random weights and 8 bands of random packed gradients, then times 15 steps after 2 warm-ups, per arm (Adam, Muon with 5 steps, Muon with 2 steps). `drive.sh` runs it at 1, 2 and 4 threads (`OPENBLAS_NUM_THREADS`, `taskset -c 2`, `2,4`, `2,4,6,8`, one logical CPU per physical core), twice in rotated order. The workstation was idle (load 0.1).

## Log

- 2026-11-03: Gate frozen. Built the owned Muon step and its test.
- 2026-11-03: Correctness test. The matrices match the reference within 1e-10. One bias (Adam) differed by 8.8e-12 absolute (1.8e-10 relative), because torch's Adam groups its bias corrections differently from ours. The test checks Adam's parameters at atol 1e-11 instead; the Adam code is unchanged and already matches the shipped plan bit for bit.
- 2026-11-03: Three driver attempts failed before timing anything, on the import path, `taskset` and the attrs merge; none produced numbers. The fourth ran 17:53–17:56.

## Results

Median milliseconds of one whole `tf_owned_step` (30 timed steps per cell, two repeats pooled):

| Threads | Layer | Adam | Muon, 5 steps | Muon, 2 steps |
|---|---|---|---|---|
| 1 | encoder | 53.4 | 594.2 | 279.4 |
| 1 | decoder | 67.9 | 794.2 | 373.2 |
| 2 | encoder | 55.0 | 377.3 | 191.6 |
| 2 | decoder | 68.5 | 507.7 | 256.6 |
| 4 | encoder | 54.6 | 261.4 | 146.4 |
| 4 | decoder | 68.4 | 351.0 | 192.2 |

The repeats agree within about 1%.

| Setting | Δ (s) | Predicted update (s) | Against Adam | Gate (≤1.3×) |
|---|---|---|---|---|
| **4 threads, 5 steps (the gate)** | **0.283** | **4.92** | **1.06×** | **go** |
| 4 threads, 2 steps | 0.124 | 4.76 | 1.03× | go |
| 2 threads, 5 steps | 0.439 | 5.08 | 1.09× | go |
| 1 thread, 5 steps | 0.726 | 5.37 | 1.16× | go |

**Correctness: passed. Go for stage 3: yes.**

## What it taught

- **On the pool, Muon's cost is a small share of the update.** On a single machine, Newton–Schulz added 78% to the update (experiment 67). On the pool it adds at most 6% at 4 threads, because each owner orthogonalizes only its own layer's matrices, all owners do so at the same time, and the update itself is ten times longer.
- **The prediction was right:** 0.38–0.51 s for 5 steps at 2 threads, inside the predicted 0.1–0.5 s, and 3–16% added across all settings measured.
- **Unlike experiment 73's hints, Muon adds no tasks or hops,** only compute inside a task the owner already runs. That's why its cost on the pool is about its FLOPs.

## Limits

- **The bound is pessimistic.** Only encoder layer 0's step must finish before the next forward; the decoder steps may overlap the end of the backward chain. The real cost is probably lower.
- **Measured on the workstation's Ryzen cores, not DigitalOcean's.** A c-8 vCPU is a hyperthread, and the pool's cores may be 1.5–2× slower. Even at twice the time, the 4-thread Δ (0.57 s) gives 1.12×. Only 1 thread at twice the time (1.45 s, 1.31×) would miss the gate, so stage 3 should keep layers on the 4-thread strong owners, as capacity placement already does.
- **Learning per pair comes from experiment 67's PyTorch reference** (float32, single machine), not from the pool's float64 owners with bf16 activations. Stage 3 measures it.
- **Memory:** Muon keeps one buffer, so the owners' optimizer state shrinks; `v` is still allocated in this build.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_owned_muon
D=runs/experiments/M-2/<host>-<timestamp>    # copy bench.py and drive.sh from the original run directory
bash $D/drive.sh                             # 1, 2 and 4 threads, twice, about 3 minutes
```

`drive.sh` runs `env PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=N OMP_NUM_THREADS=N MKL_NUM_THREADS=N taskset -c CPUS .venv/bin/python $D/bench.py $D/tN-rR.json`, with one logical CPU per physical core.
