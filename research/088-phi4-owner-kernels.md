# Four owner-kernel changes make a Phi-4 decode step 43–50% faster on an A100, with the same tokens

| | |
|---|---|
| **Conclusion** | **Worth it, now default:** four owner changes cut Phi-4's decode step 43–50% on an A100, same tokens (measured). |
| **Status** | executed |
| **ID** | DO-19 |
| **Date** | 2026-11-05 |
| **Path** | Phi-4 (experiment 80's revision) in bf16 with the float32 head, one owner on one Vast.ai A100 PCIe 80 GB in Oklahoma. The torch owner kernels with and without four switches (`$YGARATY_PHI4_FAST`, a026df3): `rope` (a rotary table instead of recomputing cos and sin in float64 at every layer), `mask` (no causal mask for a one-token decode step), `sdpa` (torch's fused attention), `head32` (one float32 copy of the head, one product). In-process benchmark, then the pipeline with persistent single-owner sessions |
| **Result** | **Passes the frozen gate.**<br>**Benchmark** (median decode step, base against all four): 43.1 → 24.2 ms at 128 tokens (−43.9%), 43.0 → 24.4 at 1,024 (−43.3%), 49.7 → 24.9 at 3,968 (−50.0%). The p95 fell from 46–51 ms to 25 ms.<br>**Tokens:** identical to the baseline in every arm and length, and every arm's tokens equal experiment 80's reference.<br>**Pipeline:** inter-token median 43.3 → 24.6 ms at 128 tokens and 50.2 → 25.3 ms at 3,968; first token 0.054 → 0.036 s and 0.94 → 0.69 s; 0 retries and 0 rejected.<br>**Now:** the four switches are on by default; `head32` skips its copy on a GPU without room. About $0.66 |
| **Evidence** | Measured: [evidence/transport/do19-owner-kernels-a100-2026-11-05.json](evidence/transport/do19-owner-kernels-a100-2026-11-05.json) (with profiler tables); raw logs in `runs/pods-data/do19/`. Code a026df3 on the machine; tests `tests/test_phi4_fast.py` |

## Question

[Experiment 86](086-phi4-split-one-a100.md) measured 47 ms per token for Phi-4 on one A100, reading its weights at about a third of the GPU's bandwidth. **Which owner-kernel changes reduce decode latency, with the same tokens?**

## Prediction

Growing-cache copies and repeated launches may explain part of the owner time; experiment 81 did not attribute their individual costs. (The growing cache was already replaced by block allocation in experiment 83's code; this run tests the remaining candidates.)

## Gate (frozen 2026-11-05, before any machine was made; the user's yes)

- **Machine:** one Vast.ai A100 80 GB (`any-80to81gb-sm80`, US and CA), Phi-4 bf16 with the float32 head, one owner; budget about $0.75, deleted at 60 min.
- **Switches** (`$YGARATY_PHI4_FAST`, a026df3): `rope`, `mask`, `sdpa`, `head32`. CUDA graphs are not attempted in this run and are reported as not done.
- **Benchmark** (`runs/experiments/DO-19/bench_remote.py`, in process): a decode step's owner time with all switches off (base), each alone, and all four. Interleaved over 3 rounds at prompts of 128, 1,024 and 3,968 tokens (seed 2026), 64 decode steps each; a profiler table of 8 decode steps for base and all.
- **Pipeline confirmation:** persistent single-owner jobs with all switches off (0) and on (F), 0 F F 0 at 128 tokens and F 0 0 F at 3,968, each 1 warmup plus 5 measured sessions of 128 new tokens.
- **Pass:**
  1. all four switches cut the benchmark's median decode-step time at least 20% at every prompt length, with no p95 regression;
  2. `rope` and `mask` give tokens identical to base everywhere;
  3. every arm's tokens on experiment 80's 3 prompts (16 tokens) equal experiment 80's reference up to its first top-2 margin below 0.5;
  4. every job completes with 0 rejected results.
- **Reported:** each switch's own effect, the pipeline's inter-token latency both ways, and the profiler tables.

## Log

- 2026-11-04: Added; renumbered from local DO-17 to DO-19. Needed DO-18 first (experiment 86).
- 2026-11-05: Switches implemented (a026df3) and tested on CPU (`tests/test_phi4_fast.py`: `rope` and `mask` bit-identical in float64, float32 and bf16; `sdpa` and `head32` within tolerance with the same tokens). The benchmark was dry-run on the tiny model. The user froze the gate.
- 17:58: A100 PCIe 80 GB in Oklahoma at $0.933/h; set up in 10 min (29.3 GB in 299 s).
- 18:09–18:12: Benchmark. 18:14–18:28: pipeline correctness and timing, all with 0 retries and 0 rejected.
- 18:28: Deleted; `pods ls` showed 0. Up about 30 min.
- After the run: the switches became the default (`FAST_DEFAULT`), with `YGARATY_PHI4_FAST=none` to switch back. `head32` keeps the chunked path when the GPU lacks room for its float32 copy plus 1 GiB.

## Results

### Benchmark (median and p95 of 192 decode steps per arm and length)

| Arm | 128 tokens | 1,024 tokens | 3,968 tokens |
|---|---|---|---|
| base | 43.13 / 47.47 ms | 43.01 / 46.43 ms | 49.67 / 51.44 ms |
| rope | 30.19 / 34.24 | 31.48 / 32.98 | 41.72 / 41.90 |
| mask | 40.52 / 42.89 | 40.67 / 42.29 | 48.67 / 50.09 |
| sdpa | 36.31 / 38.00 | 36.11 / 37.80 | 36.11 / 38.06 |
| head32 | 42.03 / 44.46 | 40.84 / 43.11 | 47.66 / 49.40 |
| **all four** | **24.22 / 24.56** | **24.37 / 24.68** | **24.85 / 25.03** |

Gate: −43.9%, −43.3% and −50.0% at the three lengths (bar: 20%), with p95 lower everywhere. Tokens were identical to base in all 15 arm-length pairs (rope and mask required, the rest observed). Every arm's tokens on experiment 80's prompts equal the reference.

### Profile (8 decode steps at 1,024 tokens)

| | base | all four |
|---|---|---|
| Kernel launches per step | 2,274 | 1,380 |
| Host (CPU) time per step | 47.7 ms | 28.0 ms |
| Matrix products (`aten::mm`) per step | 18.7 ms | 18.3 ms |
| Element-wise copies (`aten::copy_`) per step | 5.7 ms | not in the top rows |

### Pipeline (persistent, one owner; 2 jobs per arm, 10 measured sessions of 127 gaps)

| Prompt | Off: median / p95 | On: median / p95 | First token, off → on |
|---|---|---|---|
| 128 | 43.3 / 44.5 ms | 24.6 / 25.3 ms | 0.054 → 0.036 s |
| 3,968 | 50.2 / 50.9 ms | 25.3 / 25.8 ms | 0.938 → 0.692 s |

## What it taught

- **The rotary embedding was the biggest single cost:** 13 ms of a 43 ms step. It is about 10 small kernels per layer, recomputed in float64 at every layer, 40 times per token.
- **Fused attention matters most at long contexts:** −13.6 ms at 3,968 tokens, against −7 ms at 128. The separate score, mask and softmax kernels grow with the context.
- **The combination beats the sum of its parts, because the step was launch-bound.** Kernel launches fell 2,274 → 1,380 per step, and host time 47.7 → 28.0 ms. With fewer launches the GPU stops waiting for the CPU, so the step's time falls to about the matrix products' 18.3 ms plus about 6 ms.
- **The step is now close to memory-bound.** 29 GB in 18.3 ms of products is about 1.6 TB/s, 82% of the A100 PCIe's 1.94 TB/s. The 6 ms left is host overhead, which CUDA graphs could remove (not attempted here).
- **Every Phi-4 or TwIL setup on a GPU gets this,** because it is now the default: about 25 ms per token on one A100, against experiment 81's 44 ms on two H100s.

## Limits

- One GPU type and one host; the gain on smaller GPUs, where compute rather than launches may bound a step, was not measured.
- One owner. With several owners, each owner's step shrinks the same way, but the network share is unchanged (experiment 87).
- The benchmark times the owner's stage call, the pipeline its inter-token gaps; both on one machine.
- The 4-bit path (fused kernel, experiment 83) was not re-measured with these switches.
- Cost is estimated from the hourly price and Vast.ai's per-GB rate, not read from the invoice.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_phi4_fast
.venv/bin/python runs/pods-data/vast_drive.py do19 up        # 1 Vast.ai A100 80 GB, ~$0.93/h, deleted at 60 min
.venv/bin/python runs/pods-data/vast_drive.py do19 bench     # runs/experiments/DO-19/bench_remote.py on the machine
.venv/bin/python runs/pods-data/vast_drive.py do19 correctness
.venv/bin/python runs/pods-data/vast_drive.py do19 timing
.venv/bin/python runs/pods-data/vast_drive.py do19 down && python -m ygaraty.experiments.pods ls
```
