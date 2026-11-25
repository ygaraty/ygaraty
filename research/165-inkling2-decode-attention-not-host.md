# At 1k positions Inkling 2 Mini's A100 decode waits on the GPU's attention kernel, not on host encoding: a cheaper host path saves 1%

| | |
|---|---|
| **Conclusion** | Failed gate: cheaper host path 68.3 vs 68.9 ms per token at 1k; attention, not encoding, dominates (measured, A100) |
| **Status** | executed |
| **ID** | DO-99 (report 165) |
| **Category** | browser |
| **Where** | workstation (680M and tiny model: host profile, correctness); remote (one Vast.ai A100 SXM4 40 GB: host and GPU profiles, timing) |
| **Priority** | med |
| **Date** | Added 2026-11-24; started 2026-11-24; finished 2026-11-24 |
| **Path** | `webgpu_kernels.mts` `fastHost` (default on): bind groups reused per dispatch slot by comparing bindings, uniforms written in place, repeated `setPipeline` skipped. Against `fastHost: false` (report 162's code), on the same A100 |
| **Result** | Item 2 **fails**: per-token time after the 1k prompt (run + read, median of 39 steady steps, alternating twice), 68.1 and 68.4 ms against 69.0 and 68.8: ratio 0.99 (bound ≤ 0.75). Item 1: identical tokens in every A100 run (41 at 1k, 41 on chat) and DO-96's check on the 680M (logits within 3.4e-5, 64 greedy chat tokens as arm A); the 5-prompt greedy check was not run on the A100. **Found:** at 1k positions the GPU takes 63 ms per step, and the attention kernel is 38 ms of it (48 dispatches); host encoding of a steady step is about 31 ms and runs under the GPU's time. Report 162's 41 ms "host-bound" reading came from 5-step averages at 27 positions, whose first step compiles the decode pipelines (130–540 ms) |
| **Evidence** | measured-remote: `runs/experiments/DO-99/vast-do99a-20261124/` (`steady2/` per-token at 1k, `steady/` chat, `k1/` GPU per kernel and pass at 1k, `prof/` host call timings and CPU profile); local: `omarchy-20261124-base`, `omarchy-20261124-fasthost`, `tiny/` |

## Question

Report 162 (DO-98) brought the A100 decode to 75 ms per token, but each step is now bound by the CPU: 41.1 ms of host encoding against 26.9 ms of GPU time for 1,378 dispatches, about 30 µs per dispatch in Dawn's native layer. Pipelined submission overlaps the two, so a step costs a bit more than its encoding. Which host costs per dispatch can go, so that the step approaches the GPU's 27 ms?

## Prediction

TBD. Bind-group creation and validation are the usual per-dispatch host costs in Dawn. If most dispatches reuse pre-built bind groups, encoding could drop toward 10–15 µs per dispatch (about 15–20 ms per step), making the step GPU-bound at about 30–40 ms per token.

## Gate (frozen 2026-11-24, before any A100 timing)

1. **Same tokens:** 64 greedy tokens identical to report 156's arm A on all 5 prompts; math check (forced routing, last-position logits) within 2e-3 of NumPy.
2. **Faster:** decode at 1k positions on the same A100, alternating runs as report 162 (t1k twice per arm), new at most 0.75× report 162's code (`fastHost: false` and any other DO-99 switches off). Reported: 8k, host encode and GPU time per step.

## Protocol

1. Workstation: host CPU profile of a decode step (`DO-96/tools/decode_profile.mjs --mode pass`, plus a Node `--cpu-prof`) split into bind-group creation, pass begin/end, set calls and dispatch. Change one cost at a time, checking tokens against NumPy after each (as DO-96).
2. One rented A100 for the timing (DO-98's kit). Budget and teardown deadline from the user first.

Related: [162 (DO-98)](162-inkling2-fewer-dispatches-a100.md), [159 (DO-96)](159-inkling2-decode-kernels-a100.md), [164 (DO-97)](164-inkling2-direct-links-three-contributors.md).

## Log

- 2026-11-24: Added at the user's request, from report 162's finding that the A100 step is host-bound.
- 2026-11-24: Prioritised by the user ("prioritise 99 now then 100"). Budget $2, machine deadline 2 h.
- 2026-11-24: **Local host profile** (`tools/host_profile.mjs`: each WebGPU call and `Gpu.dispatch`/`pipeline`/`uniform` timed on its prototype, medians because the averages carry garbage-collection pauses). Real model on the 680M (`omarchy-20261124-base`, residency 8–15, chunk 32): 1,378 dispatches per step; `Gpu.dispatch` median 10.3 µs (setPipeline 0.7, setBindGroup 1.7, dispatchWorkgroups 1.4, uniform 2.1, pipeline lookup 0.1; the rest is the bind-group key in JavaScript), plus 296 `createBindGroup` per step (median 17.7 µs). The streamed layers' weights land in new buffers each step, so their bind groups miss the cache; uniform offsets also move between steps. Tiny model fully resident (`tiny/`, 285 dispatches per step): `Gpu.dispatch` median 4.8 µs, mean 10 µs; 14 bind groups per step. So on this CPU the dispatch path costs about 5–10 µs, against report 162's 30 µs per dispatch on the A100 host: the A100 host itself must be profiled.
- 2026-11-24: A100 rented for the host profile: Vast.ai `do99a` (62348097), A100 SXM4 40 GB, California, $0.567/h, watchdog 120 min. Kit: DO-98's plus `tools/host_profile.mjs`, `tools/prof.sh` and the tiny GGUF (same profile as the workstation's, for comparison).
- 2026-11-24: **First change, `fastHost`** (`webgpu_kernels.mts`, default on; `fastHost: false` is the previous path): bind groups reused per dispatch slot of a submission by comparing bindings directly (no key string), uniforms written straight into the staging copy, repeated `setPipeline` skipped. Browser tests 128/128. Tiny model resident (`tiny/fast-*`, 100 steps, two runs per arm): identical 101 tokens; step 4.9 → 4.05 ms, dispatch median 4.3 → 3.4 µs. Real model on the 680M (`omarchy-20261124-fasthost`, DO-96's `check.sh`): synthetic models 9/9, chat forced decode within 3.4e-5 (27/27, 0 flips), 64 greedy tokens same as arm A.
- 2026-11-24: Gate frozen (above) as proposed, before any A100 timing.
- 2026-11-24: **A100 run** (`do99a`, Vast.ai 62348097, A100 SXM4 40 GB, California, $0.567/h; host AMD EPYC 7K62, 46-CPU quota, load about 8). The GGUF download stalled at half (0.5–2.6 MB/s from Hugging Face), so parts 16–19 were pushed from the workstation and the rest refetched with 8 ranges each; whole-file sha256 matched. Deleted by its recorded ID after 1.49 h (about **$0.84** of $2).
  - Host calls (`prof/real.*`, 10 steps): `Gpu.dispatch` median 11.4 µs (setPipeline 1.4, setBindGroup 2.2, dispatchWorkgroups 2.1, uniform 1.9), so about 16 ms per step in steady state. The large means (pipeline lookup 14 ms per step, bind groups) are the first decode step compiling `ink_expvec_*` and building every bind group.
  - Steady steps without wrappers (`steady/`, chat, 40 steps): step 1 130–540 ms, then the host part of a step (`run()`, which returns before the GPU finishes) median 31.2–33.7 ms either way.
  - Per token after the 1k prompt (`steady2/`, run + `readToken`, 40 steps, alternating false/true twice): medians 69.0, 68.4, 68.1, 68.8 ms (all-steps means 73.6–75.9, matching report 162's 75). `run()` alone is 67 ms of it: at 1k positions it waits for the GPU.
  - GPU at 1k (`k1/`, DO-99's copy of `decode_profile.mjs` with chunked prefill): pass profile GPU 62.8 ms per step, encode 50.2 (5-step average, step 1 included). Per kernel (detail mode, 76.3 ms with its overhead): `ink_attention` 38.2 ms (50%, 48 dispatches), `ink_mix` 6.0, `ink_ctrl_mix` 3.6, expert kernels about 3 each.
- 2026-11-24: **Gate** (frozen above): 1 partly checked (identical tokens across all A100 runs and arms; 680M forced check and 64 greedy chat tokens; not the 5-prompt greedy run on the A100). 2 FAIL: 68.25 vs 68.9 ms, ratio 0.991. The timing used `host_profile.mjs` (per token, steady steps) rather than `greedy.mjs`'s 33-minus-1 requests; both include `readToken`, and the all-steps means reproduce report 162's 75 ms.

## Conclusion

The question was wrong at the point that matters. At 27 positions the A100 step is about 31 ms of host encoding against 27 ms of GPU, so the host matters there. At 1k positions the GPU takes 63 ms, half of it in attention, and the host's 31 ms run underneath. The cheaper host path (`fastHost`) is correct and slightly faster (tiny model −17% per step on this workstation; 1–2 ms per step on the A100), and stays on by default, but it cannot move time per token at 1k.

The lever is the decode attention kernel: one workgroup per head walks every position, so 48 attention dispatches cost about 0.8 ms each at 1k (more at 8k, which explains report 162's 110 ms there). Splitting positions across workgroups (split-K, as flash-decoding) should bring it to a few ms. After that, host encoding (about 31 ms) would be the next limit.

## Limits

- One A100 host (shared EPYC 7K62); four timing runs of 40 steps each.
- Per-call timings with wrappers add about 0.1–0.3 µs per call; the timing runs used no wrappers.
- The 5-prompt greedy check was not repeated on the A100; the change only alters which bind group object is reused and how uniforms are written, and every run's tokens were identical between arms.

## Reproduce

Workstation: `node runs/experiments/DO-99/tools/host_profile.mjs MANIFEST_DIR PROMPT.tokens --steps N --chunk C --resident A-B [--opts '{"fastHost":false}'] [--plain]` (manifest from `tools/manifest.py MODEL.gguf x OUT`); `bash runs/experiments/DO-96/tools/check.sh OUT chat`.

A100: DO-98's kit (`runs/experiments/DO-98/tools/make_kit.sh`) plus DO-99's tools and the tiny GGUF; `a100.sh setup`, then `tools/prof.sh`, `tools/steady.sh`, `tools/steady2.sh`, and `tools/decode_profile.mjs ... effort0 t1k.tokens --chunk 512 --resident 0-23 --mode detail|pass`.
