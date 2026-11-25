# Fewer dispatches and pipelined submission bring Inkling 2 Mini's A100 decode to 75 ms per token, with the same tokens

| | |
|---|---|
| **Conclusion** | Works: Inkling 2 decode 75 vs 115 ms per token on one A100 (−35%), 37% fewer dispatches, tokens identical (measured, Vast.ai) |
| **Status** | executed |
| **ID** | DO-98 (report 162) |
| **Category** | browser |
| **Where** | workstation (680M: correctness, dispatch count, encode time); remote (one A100 for the timing that decides) |
| **Priority** | high |
| **Date** | Added 2026-11-24; started 2026-11-24; finished 2026-11-24 |
| **Path** | `inkling2_gpu.mts` single-token path after report 159: loop controls, the per-visit rows kernels, Q-side products, residual copies, convolutions, and how a decode step is submitted |
| **Result** | Same A100 SXM4 40 GB, alternating runs: decode at 1k positions **75 ms per token** (80, 70) against 115 (120, 110) for report 159's code: ratio 0.65 (gate ≤ 0.8) and under the 80 ms bound; 110 against 150 at 8k. 64 greedy tokens identical to report 156's arm A on all 5 prompts; math check within 3.1e-5 of NumPy. Per step (pass profile, chat): 1,378 dispatches in 6 submissions against 2,183 in 1; host encoding 41.1 ms against 47.5, GPU 26.9 ms against 35.0. Against report 156's kernels (165 ms on the same GPU class), 2.2 times faster |
| **Evidence** | measured-remote: `runs/experiments/DO-98/vast-do98a-20261124/` (`gate.json` by `DO-96/tools/gate.py`); measured-local: `runs/experiments/DO-98/omarchy-20261124/` |

## Question

Report 159 (DO-96) made Inkling 2 Mini's A100 decode 110 ms per token (from 165), with the same tokens, but GPU time is only about 52 ms of it: 2,183 dispatches per step, all encoded on the host before the one submission runs. On the 680M, encoding a step takes about 48 ms (about 22 µs per dispatch; `decode_profile.mjs --mode pass`). Which of the remaining costs can go?

- **Fewer dispatches:** each looped visit's 4 loop controls take 6 dispatches each (two norms, down, gain, forget, combine); fuse the norms into one, and gain + forget + combine into one kernel (a 64-wide dot per output). The 4 rows kernels per visit (Q, K and R clips, V norm) into one. The Q-side products of the same input (Q, R, sink, gate; K and V too on sliding first-pass visits) through `matmulGroup`. Residual add + copy through a pool kernel that reads and writes one binding. K and V convolutions in one dispatch.
- **Pipelined submission:** submit a decode step every few visits without waiting, so the GPU runs while the host encodes the rest (WS-32's deferred `end()`).

## Prediction

About 1,450 dispatches per step (−33%). Encode time falls in proportion. With pipelining, encode overlaps GPU time, so the A100 step approaches max(encode, GPU) plus sync rather than their sum: about 60–80 ms per token at 1k positions.

## Gate (frozen 2026-11-24, before any change)

1. **Same tokens:** each change passes `DO-96/tools/step.sh` on the 680M (tiny F32 models, real chat prompt token by token with forced routing within 2e-3 of NumPy, 64 greedy tokens identical to report 156's arm A); the final code passes on effort0, chat, code and tool. On the A100: 64 greedy tokens identical to arm A on all 5 prompts, and the forced math check within 2e-3.
2. **Faster (remote, decides):** on one A100 SXM4 40 GB, decode at 1k positions at most 80 ms per token (half of report 156's same-machine 165 ms would be 82), and at most 0.8 × report 159's kernels (`INK_FAST` with only the DO-98 switches off) on the same machine; median of 2 runs per arm, alternating.
3. A change that costs tokens is dropped, whatever it gains.

## Protocol

Local, one change at a time: profile (`decode_profile.mjs`, detail and pass modes), then `step.sh`. Remote: report 159's kit and `a100.sh`, with the new switches; budget and teardown from the user before renting.

Related: report 159 (DO-96), report 156 (DO-95), WS-32 (deferred submission), WS-36 (grouped products).

## Log

- 2026-11-24: Added at the user's request ("go ahead with do96 improvements") after report 159's failed speed gate. Gate frozen above before any change. 680M pass-mode profile of report 159's code (`DO-96/omarchy-20261123-2226/s9-pass`): encode 47.6 ms, GPU 111.7 ms per step, 2,183 dispatches.
- 2026-11-24: Local part, one change at a time, each through `DO-96/tools/step.sh` (output `runs/experiments/DO-98/omarchy-20261124/`); switches `ctrl`, `rows`, `group`, `pool`, `pipeline` in `FAST_DECODE`:

  | Step | Change | Dispatches per step | 680M GPU ms | Chat forced, last logits | Greedy (64) |
  |---|---|---:|---:|---:|---|
  | — | Report 159's code | 2,183 | 114 | 1.8e-5 | identical |
  | c1 | Loop control in 2 dispatches: norms folded into `down` (CTRL_DOWN), gain + forget + combine (CTRL_MIX) | 1,799 | 112 | 2.2e-5 | identical |
  | c2 | Q, K, V, R rows kernels in one dispatch (exact) | 1,655 | 107 | 2.2e-5 (same bits as c1) | identical |
  | c3 | Same-input products grouped (`matmulGroup`: Q/R/sink/gate; K/V or K/dynK, V/dynV) (same sums as matvecC) | 1,475 | 111 | same bits | identical |
  | c4 | Residual add + copy inside the pool in one dispatch (exact) | 1,378 | 104 | same bits | identical |
  | c5 | Pipelined submission: a decode step submitted every 8 visits without waiting | 1,378 (6 submissions) | 114 | same bits | identical |

  Dispatches −37%. The 680M cannot show c5's effect or the per-dispatch saving: its streamed layers wait for uploads, and its GPU time is near its memory bandwidth. Pass-mode profile (`c6-pass`): encode 53.1 → 46.2 ms per step. A CPU profile of the host (`c7-cpuprof`) puts JavaScript dispatch encoding at about 0.1 s over 6 steps, so the A100's gap between GPU time and token time is native (driver and per-dispatch) cost, measurable only there. Kit for the remote run: `runs/experiments/DO-98/kit.tar.gz` (`tools/make_kit.sh`, `tools/a100.sh`: "old" = report 159's code, plus pass-mode profiles).
- 2026-11-24: Remote run approved: budget $2, one A100 SXM4 40 GB, watchdog 120 minutes; rented after DO-87's blocks finish so its GGUF download cannot share a host link with DO-87's timing.
- 2026-11-24: Run `do98a` (A100 SXM4 40 GB, US, $0.563/h): setup 00:58–01:43 UTC (the GGUF download took 43 min), DO-98 runs 01:43–01:59, then M-8's pruned-GGUF timing (same machine, own gate) until 02:07. Deleted by recorded ID at 02:09: about 70 minutes, about **$0.66**, shared with M-8 (within $2). Gate computed by `DO-96/tools/gate.py` on the fetched files: both items pass.

## Results

| Gate item | Result | Pass |
|---|---|---|
| 1. Same tokens | 64 greedy tokens identical to arm A on effort0, chat, code, tool, long; forced math check ≤ 3.1e-5 | Yes |
| 2. Decode at 1k ≤ 80 ms and ≤ 0.8 × report 159's code, same machine | 75 ms (80, 70) against 115 (120, 110): 0.65 | Yes |

Not gated:

| | Report 159's code | DO-98 |
|---|---:|---:|
| Decode at 8k, ms per token | 150 | 110 |
| Prefill 1k / 8k, s | 4.9–5.0 / 43.4 | 4.4–4.8 / 43.1 |
| Dispatches per decode step | 2,183 | 1,378 |
| Submissions per decode step | 1 | 6 |
| Host encoding per step (pass profile) | 47.5 ms | 41.1 ms |
| GPU time per step (pass profile) | 35.0 ms | 26.9 ms |

The timing runs gave the same tokens in both arms. Report 159's code measured 115 ms here against 110 in its own run, and 150 at 8k against 190: hosts differ, which is why each run compares against old code on the same machine.

The step is now host-bound: encoding takes about 30 µs per dispatch in Dawn's native layer, more than the GPU's 27 ms in all. Pipelining overlaps the two, so a step costs a little more than the encoding.

## Limits

- `greedy.mjs` reports seconds per token to two decimals: 10 ms resolution. The gate's 80 ms bound is met with one run at 80 and one at 70.
- One A100 host, two runs per arm at 1k and one at 8k. Pass-profile wall times include timestamp overhead.
- The 680M could not show c5's gain or the per-dispatch saving.

## Reproduce

Local: `runs/experiments/DO-96/tools/step.sh OUT` (`INK_FAST` switches `ctrl`, `rows`, `group`, `pool`, `pipeline`). Remote: `bash runs/experiments/DO-98/tools/make_kit.sh kit.tar.gz`, unpack as /workspace/kit on a GPU machine (ubuntu:22.04, NVIDIA_DRIVER_CAPABILITIES=all), `bash runs/experiments/DO-96/tools/a100.sh setup` and `... run` (the kit's a100.sh is DO-98's); then `python runs/experiments/DO-96/tools/gate.py RUN_DIR`. Code 12f33ea plus this working tree.
