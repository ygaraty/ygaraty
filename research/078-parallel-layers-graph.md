# Parallel layers 1–2 shorten the distributed update 9.5%, just short of the gate, and learn better

| | |
|---|---|
| **Conclusion** | **Failed gate, narrowly:** parallel layers 1–2 make updates 9.5% shorter, against the 10% required (measured, workstation, 3 runs per arm). |
| **Status** | executed |
| **ID** | WS-8 (experiment 78) |
| **Category** | training |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-03 |
| **Path** | The distributed Transformer graph, given parallel layers 1–2 on both sides: layers 1 and 2 read the embeddings, layer 2 returns only its residual update, and layer 3's task adds x + r₂ itself (`YGARATY_PARALLEL_LAYERS=enc:1-2,dec:1-2`). Stage 0: learning in the PyTorch reference at 30.3M. Timing: tiny Transformer (128/384, 4 heads, 3+3 layers) on experiment 59's setup (8 Python workers, emulated 100/20 Mb/s, 50 ms links, handoffs, owned layers and 4 vocabulary shards), dense and parallel interleaved ABBAAB |
| **Result** | Median seconds per update over updates 2–10: dense 4.097, parallel 3.707 (**−9.5%**; gate ≤−10%). The arms' runs don't overlap: 3.68–3.74 s against 4.10–4.17 s. Traced chain: 19 → 15 hops, but the median hop grows from 192 to 232 ms, because the merged layer waits for its second input. Stage 0 passed with a better model than dense: held-out loss 5.593 against 5.717 (−0.123, 30.3M, two seeds). Parity: losses within 5e-16 of PyTorch over 3 updates. Free; about 40 minutes of runs |
| **Evidence** | [`evidence/training/ws8-parallel-layers-graph-2026-11-03.json`](evidence/training/ws8-parallel-layers-graph-2026-11-03.json) (stage 0 curves and chrF, per-update walls and losses of every timing run, chain-trace figures, driver hashes). Raw runs, captures, `stage0.sh`, `drive.sh`, `campaign.sh`, `analyze.py` and `chain_trace.txt` in `runs/experiments/WS-7/omarchy-20261103-230928/` (local, not in git). Code: `parts` and `residual` in `Tape.transformer_layer` and the layer kernels, `FusedLayerStack`'s groups, the owned-layer lowering, the embedding scatter, `YGARATY_PARALLEL_LAYERS` in `coordination/lstm.py`; `tests/test_parallel_layers_graph.py` |

## Question

In [experiment 76](076-parallel-layers.md)'s PyTorch reference, running layers 2 and 3 of both sides in parallel removed 4 of the 12 layer hops per update for only 0.025 of held-out loss. Each hop costs about 230 ms (L5), so the update might get about 20% shorter. That held only if the removed hops came off the critical path, and if the merged layer didn't add a fan-in wait of its own.

**With parallel layers in the distributed graph, matching the PyTorch reference, how much shorter is the update than the dense 3+3 model?**

## Prediction and gate (frozen before any run)

**Prediction:** 10–20% shorter updates. That is less than 4 × 230 ms, because the merged layer waits for the slower of its two owners and the backward pass fans out to two owners.

**Gate:**
- **Correctness:** losses and weights match the PyTorch variant within 1e-12 relative over 3 updates; dense losses stay bit-identical to today's graph.
- **Decision:** median seconds per update over updates 2–10, from 3 interleaved runs per arm (ABBAAB). The parallel arm must be at least **10% shorter** than dense.
- **Report also:** hops on the traced critical chain, and the fan-in wait at the merged layer.

**Wiring changed to layers 1–2, before any run (decided with the user).** Layers 2–3 are the last of each stack, so the final layer norm runs on their merged sum. That merge costs a transfer of its own unless it is fused into every consumer: the decoder layers' memory input and the vocabulary tasks. With layers 1–2 parallel, layer 3 keeps its fused norm and sums its parts itself. This saves the same 4 hops with a contained change. Its learning was not measured, so a stage 0 was added:
- **Stage 0 gate (frozen):** in the PyTorch reference, `--parallel-layers enc:1-2,dec:1-2` over seeds 2026 and 2027 at 32,768 pairs must have mean held-out loss within **0.05** of dense (experiment 76's D, 5.717).

## Protocol

- **Graph.**
  - `Tape.transformer_layer` takes `parts`: values added to the layer's input inside its task, placed after the memory among its inputs. It also takes `residual`: the task returns only y − x.
  - `FusedLayerStack` builds each group of parallel layers with `layer_variants.groups`. The group's first layer returns its full output y₁ = x + r₁, and the others return r. The next layer reads y₁ and the r's as its parts, so x + r₁ + r₂ is formed inside its own task.
  - **Backward:** the merged layer's input gradient feeds each part. A residual layer's gradient subtracts dy.
  - **Embeddings:** the two first-layer gradients of the embedding bands would need an `add_many` task of their own, one hop more. Instead, each vocabulary owner scatters both terms itself.
  - A view read whole by several concatenations is copied once (`whole`): the parallel group's input gradient feeds two values.
  - Groups that contain the last layer, or that are combined with hints, are rejected.
- **Correctness:** `tests/test_parallel_layers_graph.py` checks the graph against the PyTorch variant (forward, loss and every gradient). It runs three owned updates (layers and 2 vocabulary shards owned) against PyTorch with Adam, and two owned updates against the plan that ships weights. It also checks the planned critical path: 20 hops dense, 16 parallel.
- **Timing:** `drive.sh` runs experiment 58's `http_profile` command with `--layers 3`, with `YGARATY_PARALLEL_LAYERS` set or empty. `campaign.sh` runs dense, par, par, dense, dense, par at 12 updates each, waiting for an idle machine before each run, then one capture per arm for `chain_trace`. `analyze.py` computes the gate.

## Log

- 2026-11-03: Added as the follow-up of experiment 76 (M-7), as WS-7; renamed WS-8 at write-up, because another session had added a different WS-7 (shared-seed evolution strategies) at the same time. Gates frozen with the user's yes; wiring changed to layers 1–2.
- 2026-11-03: Stage 0 at 79d6577, under watch.py (stall 600 s, maximum 1 hour), finished: mean 5.593 against 5.717, a pass.
- 2026-11-03: Wiring built. The first plan's critical path was 17 hops, not 16: summing the embedding gradients added an `add_many` hop, now removed by scattering each term. The first loopback run failed with duplicate task IDs (`v129/copy`): the same input-gradient view was copied once per concatenation that read it. Fixed in `whole`, and the scheduler's error now names the repeated IDs. Loopback smoke runs of both arms completed with 0 retries. Dense losses equal those of the unmodified commit, run in a separate worktree.
- 2026-11-03: Timing campaign under watch.py (stall 300 s, maximum 1 hour) `finished`, 23:28–23:36; all 8 runs exited 0.

## Results

**Stage 0 (PyTorch reference, 30.3M, 32,768 pairs):**

| Arm | Seed 2026 | Seed 2027 | Mean | Against dense | chrF mean |
|---|---|---|---|---|---|
| Dense | 5.626 | 5.807 | 5.717 | — | 5.80 |
| Layers 1–2 parallel | 5.524 | 5.662 | **5.593** | **−0.123** | 7.50 |

The parallel arm is ahead at every evaluation, on both seeds. **Pass.**

**Correctness:**
- **Loss:** within 5.1e-16 of PyTorch over 3 owned updates.
- **Weights:** within 7.7e-10 of PyTorch. That misses the literal 1e-12, but **the dense graph misses it by as much (6.5e-10)**: PyTorch's Adam groups its bias corrections differently (as in experiment 74). The 1e-12 criterion for weights was unattainable as written, and the wiring adds nothing measurable to the gap.
- **Owned updates:** they match the plan that ships weights within 1e-9.
- **Dense:** losses identical to the unmodified code's. **Pass on losses**, with the weight criterion noted as mis-set.

**Timing (tiny model, emulated home links, handoffs):**

| Run | Arm | Median s per update (updates 2–10) |
|---|---|---|
| r1 | dense | 4.169 |
| r2 | parallel | 3.742 |
| r3 | parallel | 3.677 |
| r4 | dense | 4.096 |
| r5 | dense | 4.097 |
| r6 | parallel | 3.707 |
| **Median** | dense 4.097, parallel 3.707 | **−9.5% (gate ≤−10%): fail** |

Every run had 485 tasks per update, 0 retries and 0 rejected tasks, and both arms repeated the same losses in all 3 runs. Updates 2–7 of the captures, from `chain_trace`:

| | Dense | Parallel |
|---|---|---|
| Hops on the chain | 19 | 15 |
| Chain (ms) | 3,883 | 3,587 |
| Median hop (ms) | 192 | 232 |
| Upload end to the next kernel, handed hops (ms, median) | 52 | 92 |

## What it taught

- **The hops do come off the chain, but each remaining hop gets slower.** The chain loses 4 of 19 hops, yet the median hop grows by 40 ms. A merged layer can start only when both parts have arrived from two owners, so the wait after the first upload grows from 52 to 92 ms. At this model's 192 ms hops, the 4 hops are worth about 0.77 s on paper; they saved 0.39 s in practice.
- **Parallel layers 1–2 are a better model here, not a worse one.** At 30.3M and 32,768 pairs they beat dense by 0.12 of loss, and parallel 2–3 (experiment 76) by 0.15. A likely reason is that layer 3 gets a second, independent view of the embeddings early in training. This is one budget and two seeds.
- **The trade still favors parallel layers in time to loss.** The gate measured seconds per update, and it narrowly failed. Interpolating the curves, parallel layers reach dense's final loss at about 28,800 pairs (seed 2026) and 26,700 (seed 2027), about 15% fewer pairs. With 9.5% shorter updates, that is roughly 0.77× dense's wall time. This is an estimate that combines the 30.3M learning runs with the tiny model's timing.
- **Two plan bugs that only parallel wiring exposes** (a summed embedding gradient that costs a hop, and a doubled view copy) are fixed and tested.

## Limits

- **Tiny model, workstation, emulated links.** On experiment 61's pool, the 30.3M layers' larger uploads could change the fan-in wait either way.
- **One budget (32,768 pairs) and two seeds for learning,** with the timing from a different, smaller model.
- **The weight-parity criterion was set too strictly** for any graph compared with PyTorch's Adam (see Results).
- **Only parallel layers 1–2 were built.** Layers 2–3 need the final norm fused into consumers.
- `scripts/remote.sh` doesn't pass `YGARATY_PARALLEL_LAYERS` yet.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_parallel_layers_graph tests.test_layer_variants
# Stage 0 (about 20 minutes): experiment 76's COMMON flags plus
.venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed 2026 --parallel-layers enc:1-2,dec:1-2 --output runs/experiments/WS-8/P12-2026
# Timing (about 8 minutes): copy drive.sh, campaign.sh and analyze.py from the original run directory into D, then
bash $D/campaign.sh && .venv/bin/python $D/analyze.py $D
.venv/bin/python -m ygaraty.experiments.chain_trace $D/cap-dense-cap $D/cap-par-cap
```

`drive.sh` runs experiment 58's command (in its Reproduce section) with `--layers 3` and `--handoff`, plus `YGARATY_PARALLEL_LAYERS=enc:1-2,dec:1-2` for the parallel arm. It needs the relay binary, the `ygaraty_tiny` PostgreSQL store and the OPUS-100 dataset.
