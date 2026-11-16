# Paid contributors can run our engine without a browser: Node with Dawn is exact and 6% faster than Chrome; wgpu offers f16 but is 2× slower today

| | |
|---|---|
| **Conclusion** | Worth it (as a path): native runs exact; Dawn 6% faster, wgpu adds shader-f16 but 2× slower (measured, A100) |
| **Status** | executed |
| **ID** | DO-68 (experiment 128) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Our stage modules (`qwen35_gpu`, `kolibri_gpu`, `webgpu_kernels`, the plans and WGSL) unchanged under three WebGPU implementations: headless Chrome (today), **Node with Dawn** (`webgpu` npm 0.6.2, Google's Dawn bindings) and **Deno with wgpu** (built in). The `bench` page's code as a script. Workstation first (680M: runs, same tokens, speed), then one rented A100 (with DO-70's copy probe): Qwen3.5-9B, one conversation and batches (experiments 123, 125, 127) |
| **Result** | Qwen3.5-9B on an A100 SXM4 40 GB, ms per step at 64 positions (GPU ms): Chrome 36.6 (21.9), Node + Dawn 34.5 (21.9), Deno + wgpu 71.5 (26.0); tokens per second at 1 / 8 / 32 conversations: 28.3 / 84.8 / 107.3, 29.2 / 87.5 / 108.6, 16.5 / 46.9 / 58.1. Tokens identical across all three at 64 and 1,024 positions; every batch row bitwise equal to its lone conversation in every runtime. Copy probe: Chrome 727, Dawn 745, wgpu **1,336 GB/s**. Features: only wgpu offers `shader-f16` on this card; only Chrome offers `chromium-experimental-subgroup-matrix`. By the frozen gate (15% faster, or a feature Chrome lacks): no runtime is 15% faster (Dawn 5.6%), but wgpu's `shader-f16` qualifies, so **Worth it** as a path, not as a speedup. Workstation screen first (680M): Dawn exact, 4–6% slower; wgpu cannot hold the model in 3 GiB of VRAM. About $0.28. |
| **Evidence** | measured-remote and local; curated `public/research/evidence/browser/native-webgpu-a100-2026-11-13.json`; raw `runs/pods-data/do68/out/` (A100) and `runs/experiments/DO-68/omarchy-20261113-003543/` (screen); code 08dfae4 |

## Question

Our kernels are plain WGSL; the same shaders could run natively on Vulkan, Metal or D3D12 without Chrome. On rented GPUs, what does the browser cost per token (host overhead, Dawn's validation, process hops), and could rented contributors run without a browser?

Paid contributors are rented machines (user, 2026-11-13: "we also need to test with dawn or wgpu ... because it might be common to have paid contributors"): they need no browser, and a native process may lift Chrome's limits seen in experiment 123: no `shader-f16` on NVIDIA Linux Chrome, a 725 GB/s copy roof on an A100, about 12 ms per token outside the GPU, and `subgroup-matrix` (tensor cores) only as a Chromium experiment.

## Prediction

Little GPU-time difference; host time per token lower natively (no renderer and GPU-process hop), mostly hidden already by the resident loop (experiment 120).

## Gate (proposed; freeze before running)

1. **Runs:** each native arm loads Qwen3.5-9B and decodes; adapter features and limits recorded (`shader-f16`, `subgroups`, subgroup-matrix, max buffer and storage sizes).
2. **Correctness:** tokens identical to Chrome's for 32 steps, logits within 2e-3 of the largest (bitwise expected with the same driver and shaders, reported).
3. **Speed:** ms per token for one conversation, host ms outside the GPU, tokens per second at 8 and 32 conversations, copy GB/s. **Worth it** if a native arm is at least 15% faster per token than Chrome on the A100, or makes a feature Chrome lacks usable there (f16, tensor cores).
Workstation screen first (free; no A100 rental without the user's approval and budget).

**A100 run (frozen 2026-11-13, after the screen, before renting).** User: "Commit, rent, $3 cap, 90 min". One Vast.ai A100 (SXM4 preferred), Qwen3.5-9B Q4_K_M, arms C, N, D (Node 26.8.2 with `webgpu` 0.6.2 from `npm ci`; Deno 2.9.8 from its release), order C N D D N C, each the screen's two benches (one conversation at 64 and 1,024 positions with the copy probe; batches 1, 8, 32 with `--batch-check` and `--matvec-b 8`).
1. **Runs:** each arm completes both benches (a failing arm is reported with its error, not retried beyond one rerun).
2. **Correctness:** tokens identical to Chrome's at 64 and 1,024; batch rows equal their lone conversation.
3. **Decision:** **Worth it** if a native arm's step at 64 positions is at least 15% shorter than Chrome's (median of its two runs), or if it offers `shader-f16` or a subgroup-matrix feature that Chrome lacks on that adapter (a path for later kernels); **Works, no gain** if 1 and 2 pass without that; **Failed gate** otherwise. Copy GB/s per arm answers DO-70's Chrome-versus-native question (CUDA's own copy is not measured here).

**Workstation screen (frozen 2026-11-13, before its first run).** Radeon 680M, Qwen3.5-9B Q4_K_M, arms Chrome (C), Node with Dawn (N, `webgpu` 0.6.2), Deno with wgpu (D, Deno 2.9.8), order C N D D N C (D dropped before the first screen run, see Log: wgpu cannot hold 5.9 GiB in the 680M's 3 GiB of VRAM; order C N N C). Each run: `bench --contexts 64,1024 --matvec 2,4,8 --matvec-c 16,4` (one conversation, detail profile, copy probe) and `bench --batches 1,8,32 --batch-check --matvec 2,4,8 --matvec-c 16,4 --matvec-b 8`.
1. **Runs:** both benches complete in every native run; adapter features recorded.
2. **Correctness:** each native arm's decoded tokens (base-64, base-1024) identical to Chrome's; every batch row equal to its lone conversation.
3. **Speed (reported, no threshold on the 680M):** ms per step, GPU ms, encode ms, copy GB/s, tokens per second at 8 and 32.
Pass 1 and 2: propose the A100 run (budget asked). The speed gate above applies there.

## Protocol

A native harness: the bench page's driver (`qwen35_parity.mjs`'s `bench`) run as a Node or Deno script with `navigator.gpu` from Dawn or wgpu, reading the same manifest and weight files from disk instead of the harness's HTTP server. The contributor protocol (coordinator, WebRTC) is out of scope here; a native paid contributor is the follow-up if this works.

## Log

- 2026-11-12: Added at user request, from a list of outside ideas (ExecuTorch's WebGPU backend; fewer launches per token; decode-shaped kernels). The other ideas in that list were already done or planned: dequantization inside the matvec (always), fusion (experiment 117, WS-36), one submission per step and one read-back per hop (DO-61, experiment 120), a batch-1 matvec path (experiment 121), grouped MoE decode (WS-37). Claims about ExecuTorch (release date, ~1B largest validated LLM, 4-bit groupwise g64 weights, fp32 activations, timestamp profiling) are as relayed by the user, not yet checked.

- 2026-11-13: priority raised to high and scope extended at user request ("we also need to test with dawn or wgpu some changes because it might be common to have paid contributors"): both runtimes are on the workstation (Node 26.8.2 with `webgpu` 0.6.2 from npm; Deno at /usr/bin/deno). Related: DO-70 (the copy roof outside Chrome) runs on the same A100 rental.

- 2026-11-13 00:3x: native runner `ygaraty/webapp/native_runner.mjs` (sets `navigator.gpu` from Dawn under Node or uses Deno's wgpu, sends the page's relative requests to the harness server, imports the same page module) and `YGARATY_WEBGPU_RUNTIME=chrome|node|deno` in `browser_binary.run_page`; `webgpu` 0.6.2 pinned in package.json (its postinstall, a macOS quarantine-attribute removal, left unrun); `result.adapter.features` recorded. f16 probe identical in all three runtimes, each on the 680M (Dawn "radv: Mesa 26.2.5", wgpu "AMD Radeon Graphics (RADV REMBRANDT)").
- Rehearsal (`runs/experiments/DO-68/omarchy-20261113-003543/rehearsal-*`), batches 1 and 8 with `--batch-check`: **Node** exit 0, rows bitwise equal to the lone conversation, 9.6 and 28.3 tokens per second (Chrome about 10.3 and 31). **Deno** loads the model, then "not enough memory left": the 680M has 3 GiB of VRAM and 14 GiB of GTT (`mem_info_vram_total`, `mem_info_gtt_total`); wgpu allocates storage buffers from VRAM only, Dawn also from GTT, and the stage needs 5.9 GiB. No GPU reset in the kernel log. **Before the screen**, D is dropped from the workstation arms (it cannot hold the model here) and moves to the A100 run; the screen is C N N C with the same benches and gate items for N.

- 2026-11-13 00:48: workstation screen finished (watch.py `finished`, 10 min; `runs/experiments/DO-68/omarchy-20261113-003543/`, `summary.json`). Medians of two runs per arm:

  | | Chrome | Node + Dawn |
  |---|---|---|
  | ms per step at 64 / 1,024 positions | 96.8 / 98.1 | 102.5 / 101.3 |
  | GPU ms (64) | 88.3 | 94.0 |
  | encode + expand ms (64) | 1.7 + 1.1 | 3.7 + 2.6 |
  | copy GB/s | 62.3 | 62.3 |
  | tokens per second, 1 / 8 / 32 conversations | 10.3 / 30.9 / 32.2 | 9.3 / 28.1 / 30.1 |

  - Gate 1 passes for N: both benches complete in both runs. Features: Dawn in Node offers the same core set as Chrome (`shader-f16`, `subgroups`, `subgroup-size-control`, `timestamp-query`) without Chrome's `chromium-experimental-*` ones.
  - Gate 2 passes: base-64 and base-1024 tokens identical across all four runs; every batch row's tokens and last logits bitwise equal to its lone conversation in every run.
  - Speed (reported): on the 680M Node with Dawn is 4–6% slower per step than Chrome (GPU ms 6% higher, host encoding 2 ms more); no gain here, where the GPU time dominates.
  - Per the frozen rule, the A100 run (C, N, D; DO-70's copy probe) is now proposed; budget to be asked.

- 2026-11-13 00:5x CET: rented `do68` (Vast.ai A100 SXM4 40 GB, $0.589/h, 90-min watchdog); setup's launch over SSH timed out but setup ran (exit 0, WebGPU probe exact). `do68.sh` from 08dfae4. The Node install failed (no `xz` to unpack Node's archive), so N1-ctx and N1-batch failed at once; installed `xz-utils` and `libatomic1` while D1 ran (network and disk only), and reran N1 as N1r after C2. Order as run: C1, D1, D2, N2, C2, N1r. Deno 2.9.8 installed from its release zip.
- 2026-11-13 01:23 CET: results copied to `runs/pods-data/do68/out/`; `pods down do68`; about 0.47 h, **$0.28**. Another machine in the Vast.ai account (`ygaraty-panel-paid-…`, started by the web app's paid-contributor flow, "NOT made here") was left alone.

## Results

A100 SXM4 40 GB, medians of two runs per arm:

| | Chrome 157 | Node 26.8.2 + Dawn (`webgpu` 0.6.2) | Deno 2.9.8 + wgpu |
|---|---|---|---|
| ms per step at 64 / 1,024 positions | 36.6 / 37.8 | **34.5** / 36.4 | 71.5 / 74.1 |
| GPU ms at 64 | 21.9 | 21.9 | 26.0 |
| encode + expand ms at 64 | 5.5 + 2.0 | 7.1 + 2.6 | 6.4 + 2.6 |
| tokens per second, 1 / 8 / 32 conversations | 28.3 / 84.8 / 107.3 | 29.2 / 87.5 / 108.6 | 16.5 / 46.9 / 58.1 |
| copy probe GB/s | 727 | 745 | **1,336** |
| `shader-f16` | no | no | **yes** |
| subgroup-matrix | `chromium-experimental-subgroup-matrix` | no | no |

- Gate 1: every arm completes both benches (N1 after its install fix). Gate 2: base-64 and base-1024 tokens identical in all six runs; batch rows bitwise equal to their lone conversation at 1, 8 and 32 in every run.
- Gate 3: Node + Dawn's step is 5.6% shorter than Chrome's (under 15%); Deno + wgpu's is 95% longer. wgpu offers `shader-f16`, which Chrome lacks on this adapter: **Worth it** as frozen, for the feature path.

## What it taught

- **A paid contributor needs no browser.** The same JavaScript engine and WGSL run under Node with Dawn with Chrome's exact tokens and a slightly shorter step (the renderer and GPU-process hop cost about 2 ms per token here); Dawn's GPU time is identical, as expected from the same Dawn and driver.
- **The 725 GB/s roof of experiment 123 is Dawn's, not the browser's:** native Dawn copies at 745 GB/s, wgpu at 1,336 (86% of this card's 1.56 TB/s). Part of DO-70's question is answered (CUDA's copy not measured).
- **wgpu is the faster memory path but the slower engine today:** our model's GPU time is 19% longer under wgpu (26.0 against 21.9 ms) despite its faster copies, and about 37 ms per step goes outside the GPU (read-back or submission latency in Deno's binding), which halves throughput. Its `shader-f16` on NVIDIA opens f16 arithmetic (half the register and cache traffic for activations) that neither Dawn build allows here.
- Native installs need care on rented images: Node's archive needs `xz`, Node needs `libatomic1`; Deno's single binary needs only `unzip`.

## Limits

- One A100 (40 GB) and one host; the first Node run failed on its install and was rerun after C2, so the order was not fully interleaved.
- Deno's step overhead was not profiled (whether read-back, `mapAsync` polling or submission); wgpu was not tried from Rust directly.
- The copy probe is our WGSL kernel; CUDA's copy for the same card was not measured.

## Reproduce

```bash
# workstation screen (680M), code 08dfae4 or later:
bash runs/experiments/DO-68/screen.sh runs/experiments/DO-68/<host-ts>
# A100: contributor_fleet up RUN --gpus 1 --gpu "A100 SXM4" --max-hourly 1.6 --minutes 90 --min-ram 64 --disk 100
# then on the machine: apt-get install -y xz-utils libatomic1 unzip; bash do68.sh (runs/pods-data/do68/do68.sh)
# any harness command under another runtime: YGARATY_WEBGPU_RUNTIME=node|deno (YGARATY_NODE, YGARATY_DENO for the binaries)
```
