# New one-token kernels make Inkling 2 Mini decode a third faster on an A100 with the same tokens; the rest is per-dispatch cost

| | |
|---|---|
| **Conclusion** | Failed gate: new decode kernels 110 vs 165 ms per token on one A100, −33%, not −50%; tokens identical (measured, Vast.ai) |
| **Status** | executed |
| **ID** | DO-96 (report 159) |
| **Category** | browser |
| **Where** | workstation (680M screen: correctness, GPU time per kernel); remote (one rented GPU of arm A's class for the timing that decides) |
| **Priority** | high |
| **Date** | Added 2026-11-22; started 2026-11-23; finished 2026-11-24 |
| **Path** | `inkling2_gpu.mts` single-token path: routed experts, dense `Q8_0` products, attention and the small per-visit kernels, on one stage holding the whole model (report 156's arm A setup) |
| **Result** | Same A100 SXM4 40 GB, alternating runs: decode at 1k positions 110 ms per token (110, 110) against 165 (160, 170) for report 156's kernels, ratio 0.67 against the 0.5 required; 190 against 220 at 8k; prefill unchanged (4.4–4.8 against 4.9–5.4 s at 1k, 43.1 against 43.4 at 8k). Tokens: 64 greedy tokens identical to report 156's arm A on all 5 prompts, math check within 3.1e-5 of NumPy (bound 2e-3). Per decode step, GPU time 94.6 → 52.2 ms but wall time 285 → 230 ms (profiled), with 2,583 → 2,183 dispatches: most of the step is per-dispatch cost. On the 680M, GPU time 251 → 114 ms. The kernels stay the default |
| **Evidence** | measured-remote: `runs/experiments/DO-96/vast-do96a-20261123/` (`gate.json` by `tools/gate.py`); measured-local: `runs/experiments/DO-96/omarchy-20261123-2226/` |

## Question

Report 156 decoded Inkling 2 Mini at 160 ms per token (1k positions) and 210 ms (8k) on one A100 40 GB, with simple kernels. A decode step reads about 2.4–3.9 GB of weights (48 visits: 8 experts each plus attention), so the bandwidth floor on an A100 is a few ms. Experiment 113 found Kolibri-1's single stage bound by the fixed cost per GPU call (about 0.15 ms each, about a thousand calls per token). Inkling issues more: about 40 dispatches per visit, 48 visits.

Which changes buy most, and how fast does it get with the same tokens?

- **Experts:** the one-token path decodes `IQ4_NL` byte by byte, a workgroup per (row, slot), and runs gate_up, SiTU and down as separate dispatches. Instead: word-wise decoding as in `PAIR`'s K-quant decoders, several rows per workgroup, and gate_up + SiTU fused.
- **Dense `Q8_0` products** go through the generic kernel; the coalesced and fast decoders (WS-46/47, `q8FastUpTo`) exist.
- **Fewer dispatches per visit:** fuse norm + projection, the clip/RMS rows kernels, the conv + shift pairs, and the residual copies; group Q/K/V/R/sink/gate products of the same input (`matmulGroup`).
- **One submission per decode step**, no per-visit flushes; and the resident decode loop (WS-32, `residentDecode`), which needs the per-visit states' rewind.

## Prediction

Each change is exact or within float32 reordering, so tokens are unchanged. Decode drops to about 30–50 ms per token on an A100: dispatch count and the expert kernels dominate today. Prefill gains less.

## Gate (frozen 2026-11-23, before the remote run)

One rented A100 40 GB (Vast.ai; user's budget $5, a watchdog deletes it after 150 minutes). Everything resident, prefill chunks of 512, the kit built from this tree. "Old" is report 156's kernels on the same machine (`INK_FAST=experts=0,conv=0,router=0,dense=0,chunks=0`, checked on the 680M to give report 156's 2,583 dispatches and kernels); "new" is the default.

1. **Same tokens (new):** on report 156's 5 prompts, 64 greedy tokens identical to its arm A answers (NumPy-checked), and the forced-routing math check's last-position logits within 2e-3 of NumPy.
2. **Faster:** new decode ms per token at 1k positions at most half of old on the same machine (median of 2 runs per arm, alternating). Report 156's 160 ms, the 8k timing, prefill times and the per-kernel GPU profile of a decode step are reported, not gated.
3. A change that costs tokens is dropped, whatever it gains.

Prediction recorded before the run: the A100 is limited per dispatch (about 4 GB of weights a step is about 3 ms of bandwidth there, against 160 ms measured), and the local changes remove only 15% of the dispatches, so item 2 is more likely to fail than pass.

## Protocol

1. Workstation: per-kernel GPU time of one decode step on the 680M (detail profile; `tools/bench.mjs` extended to decode). Add changes one at a time; check each with `tools/gpu_check.mjs` (real weights, streamed experts) and `tools/synth.py` (prefill/decode, chunks).
2. Freeze the gate, then one rented GPU (A100 40 GB, about $0.5/h) with report 156's arm A kit (`tools/armA.sh`, `make_kit.sh`) and the new modules. Budget and teardown deadline from the user first.
3. Answers checked on the workstation with `greedy_check.py`, as in report 156.

Related: report 156 (DO-95), report 113 (Kolibri-1 on one A100), WS-46/47 (fast `Q8_0`), WS-32 (resident decode).

## Log

- 2026-11-22: Added after report 156.
- 2026-11-23: Local part started at 12f33ea+dirty (DO-95's and WS-51's uncommitted tree), output in `runs/experiments/DO-96/omarchy-20261123-2226/`. Dawn `webgpu@0.6.2` from `runs/tools/dawn`. The 680M now fails prefill with 512-token chunks ("Not enough memory for command submission", device lost); all local runs use 32-token chunks, with layers 6–17's experts resident and the others streamed per visit. Tools in `runs/experiments/DO-96/tools/`: `decode_profile.mjs` (GPU time per kernel of single-token steps, "detail" timestamps), `check.sh` (tiny F32 models; real chat prompt token by token with forced routing against NumPy; 64 greedy tokens against arm A's NumPy-checked answers).
- 2026-11-23: Baseline profile (`p0-baseline`, chat prompt, 3 decode steps): 251 ms GPU time and 2,583 dispatches per step; expert `IQ4_NL` 104 ms (41%), generic `Q8_0` 79 ms (32%), router 23 ms (9%, one thread per token), `F32` products 14 ms, expert `Q8_0` 12 ms. Bytes read per token ≈ 4.0 GB (experts 1.85, attention 1.37, head 0.49, loop controls 0.23), a 60–70 ms floor on the 680M and about 3 ms on an A100: the 680M is kernel-bound, the A100's 160 ms is mostly per-dispatch cost.
- 2026-11-23: Changes added one at a time, each profiled (3 decode steps, chat) and checked by `tools/step.sh`: tiny F32 models (2 seeds, chunk 4 then token by token), the real chat prompt token by token with forced routing (last logits against NumPy, bound 2e-3), and 64 greedy tokens against arm A's NumPy-checked answer. Switches for A/B: `FAST_DECODE` / `setFastDecode` in `inkling2_gpu.mts`, set from the tools by `tools/flags.mjs` (`INK_FAST`, `GPU_OPTS`). Baseline check (`c0-baseline`): chat forced 1.6e-5, greedy identical.

  | Step | Change | GPU ms per step | Dispatches | Chat forced, last logits | Greedy (64) |
  |---|---|---:|---:|---:|---|
  | 0 | Baseline (report 156's kernels) | 251 | 2,583 | 1.6e-5 | identical |
  | 1 | Router: a workgroup per token, top-k by reduction (exact) | 240 (router 23 → 0.9) | 2,583 | 1.6e-5 (same bits) | identical |
  | 2 | One-token experts: 8 rows per workgroup, word-wise IQ4_NL / Q8_0, gate_up + SiTU fused | 167 (experts 124 → 62) | 2,535 | 1.4e-5 | identical |
  | 3 | Decode convolution and history shift in one dispatch (exact) | 170 | 2,343 | 1.4e-5 (same bits as 2) | identical |
  | 4 | Dense Q8_0 through the WS-47 decoders (`q8FastUpTo: 1`) | 156 (Q8_0 76 → 72) | 2,343 | 2.1e-5 | identical |
  | 5 | ... through the coalesced kernel (`matvecC` 16x4; 8x2, 32x2, 16x2w256 were slower) | 126 (Q8_0 → 38) | 2,343 | 1.8e-5 | identical |
  | 6 | Expert chunks up to the device's binding limit, at most 1 GiB (exact) | 114 | 2,183 | 1.8e-5 (same bits as 5) | identical |

  Evidence: `p0-baseline`, `c0-baseline`, `s1-route` … `s6-chunks`, sweep in `s5-sweep`. Steps 4–5 were set through `GPU_OPTS`; step 7 makes them the stage's default (`Inkling2Stage.load` sets `matvecC` 16x4 unless the deployment chose one, and `q8FastUpTo` ≥ 1).
- 2026-11-23: Step 7, the defaults with no flags (`s7-defaults`): 114 ms GPU time and 2,183 dispatches per decode step on the 680M (251 and 2,583 at baseline, so 2.2 times faster in GPU time and 15% fewer dispatches). Chat forced 1.8e-5 of NumPy; 64 greedy tokens identical to arm A on effort0, chat, code and tool. code and tool were checked against arm A's answers, which teacher-forced NumPy checked (report 156). The long prompt was not run here (its prefill streams 12 layers of experts per 32-token chunk on this GPU). The browser suite (127) and `tests/test_inkling2.py` (16) pass, and `npm run check` finds the generated modules current. Wall time per step stays about 3.4 s here, dominated by uploading the streamed experts of 12 layers, so local wall time says nothing about the A100.
- 2026-11-23: **Local part done.** What is left for the remote run: the gate (proposed above) still needs freezing, the user's budget and teardown deadline, and arm A's kit rebuilt from this tree. Not done locally, and still the main cost on an A100 if the per-dispatch overhead dominates as predicted: fewer dispatches per visit. 2,183 per step remain: norms 346, elementwise 348, dense products 356, F32 products 336 (router and loop controls through the generic kernel, 13.6 ms here), the conv 192 and the rows kernels 192.
- 2026-11-23: Remote run approved by the user ("you can run on A100"; budget $5). Gate frozen above. Switches `router`, `dense` and `chunks` added so that `INK_FAST=experts=0,conv=0,router=0,dense=0,chunks=0` is report 156's code (680M: 2,583 dispatches, its kernel names; `s8-switches`). Kit `runs/experiments/DO-96/a100-kit/kit.tar.gz` (4.8 MB, `tools/make_kit.sh`; run script `tools/a100.sh`), rehearsed unpacked on the 680M. Vast.ai run `do96a`: A100 SXM4 40 GB, US, $0.563/h, driver 595.91.07, 150-minute watchdog (at most $1.40). Setup: apt, Node and Vulkan in 30 s; GGUF in 32 parts checked by sha256.
- 2026-11-24: Run `do96a` (A100 SXM4 40 GB, US, $0.563/h): setup 22:47–23:16 UTC (the GGUF's 32 parts all passed on the first try, 28 min), runs 23:16–23:33, all exit 0. Machine deleted by its recorded ID at 23:36 (about 50 minutes, about $0.48 of the $5). Gate computed by `tools/gate.py`: item 1 passes, item 2 fails (ratio 0.67). A second A100 rented for DO-87 landed on the same Vast host (another GPU) from 23:29; the DO-96 timing runs had ended at 23:25.

## Results

| Gate item | Result | Pass |
|---|---|---|
| 1. Same tokens (new kernels) | 64 greedy tokens identical to arm A on effort0, chat, code, tool, long; forced math check 2.4e-5 to 3.1e-5 | Yes |
| 2. Decode at 1k ≤ 0.5 × old, same machine | 110 ms (110, 110) against 165 (160, 170): 0.67 | No |

Not gated:

| | Old (report 156's kernels) | New |
|---|---:|---:|
| Decode at 8k, ms per token | 220 | 190 |
| Prefill 1k, s | 5.4, 4.9 | 4.8, 4.4 |
| Prefill 8k, s | 43.4 | 43.1 |
| Decode step, GPU time (detail profile, chat) | 94.6 ms | 52.2 ms |
| Decode step, wall time (same profile) | 285 ms | 230 ms |
| Dispatches per step | 2,583 | 2,183 |

The timing runs gave the same tokens in both arms, and the 1k answer equals report 156's timing answer. On the A100 the new step's largest GPU costs are the generic `F32` products (10.0 ms, 336 dispatches: router and loop controls), the `Q8_0` products (9.2 ms) and the norms (8.9 ms, 346 dispatches); the expert kernels fell from 36.7 + 3.4 ms to a few ms. The wall time left above GPU time (about 180 ms over 52 ms) is the per-dispatch cost the prediction expected: removing 400 dispatches and 42 ms of GPU work saved 55 ms of wall time.

## Limits

- `greedy.mjs` reports seconds per token to two decimals, so the timings have 10 ms resolution; the ratio 0.67 is far enough from 0.5 for the verdict not to depend on it.
- One A100 host; two runs per arm at 1k and one at 8k.
- Detail profiling times each dispatch in its own pass, so its wall time exceeds a plain run's.
- The 680M runs streamed layers 0–5 and 18–23's experts per visit and used 32-token prefill chunks.

## Reproduce

Local: `runs/experiments/DO-96/tools/step.sh OUT` (profile and checks; `INK_FAST` / `GPU_OPTS` switch the changes through `tools/flags.mjs`). Remote: `bash runs/experiments/DO-96/tools/make_kit.sh kit.tar.gz`, copy it to a GPU machine (ubuntu:22.04, NVIDIA_DRIVER_CAPABILITIES=all) as /workspace/kit, then `bash runs/experiments/DO-96/tools/a100.sh setup` and `bash runs/experiments/DO-96/tools/a100.sh run`; back here, `python runs/experiments/DO-96/tools/gate.py RUN_DIR`. Code 12f33ea plus this working tree.
