# Tested

Ygaraty's goal is to **compile any computation into a distribution plan**: one
program, split across the devices people contribute, with the plan saying who
computes what and what crosses the network. This page records how far that is
true today, from measured runs only.

- **Verified: inference.** Qwen3.5-family language models (and earlier Phi-4,
  TwIL-8B and Kolibri-1) run as WebGPU pipelines across browser and native
  contributors, with answers checked against reference implementations.
- **In progress: training.** The Tape compiler and DiLoCo rounds run locally
  and in tests (including a real Chromium WebGPU round), but no distributed
  training run of the current code has been timed. Training results live with
  the research reports, not here.

Every number below comes from a run, its log or its report. A dash (—) means
**not collected**, not zero. "Not collected yet" at the end lists the gaps.

## How to read the tables

- **TTFT**: time to first token, from sending the request to the first streamed
  text, including the prompt's prefill.
- **tok/s per chat**: streamed tokens after the first one, divided by the time
  they took. **Combined**: all chats' tokens divided by wall time. Reports that
  measured milliseconds per token are converted (1000 / ms).
- **Concurrency**: conversations decoding at the same time (`--slots`).
- **Context**: the conversation capacity the run was prepared for, and the
  position measured at when the report says so. Short prompts (under 100
  tokens) say little about speed at long positions.
- **Workers on one GPU** share it, so those speeds are lower than the same
  split on separate GPUs, and no network is involved.

## Current code (2026-11-16)

The public presets (`ygaraty launch --model ...`), Chrome or headless Chromium
contributors. Context 4,096 tokens and replies capped at 128–256 tokens in
every row; prompts were one short question (about 60 tokens with the template).

| Model | GPU | Workers, route | Concurrency | TTFT (s) | tok/s per chat | Combined tok/s | Check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | Radeon 680M laptop | 2 on one GPU, WebRTC | 1 | 0.08 | 54 | — | answers coherent; — reference |
| Qwen3.5-0.8B | Radeon 680M | 2, relay | 1 | 0.08 | 40 | — | same answers |
| Qwen3.5-0.8B | Radeon 680M | 2, WebRTC over LAN (`--lan`) | 1 | 0.08 | 52 | — | same answers |
| Qwen3.5-0.8B | Radeon 680M | 2, Cloudflare quick tunnel (relay) | 1 | 0.28 | 6.5 | — | same answers |
| Qwen3.5-0.8B | Radeon 680M | 2, WebRTC | 1 / 2 / 3 / 4 | 0.09–0.25 | 50 / 33–34 / 29–34 / 26–29 | 46 / 63 / 78 / 92 | — |
| S5-Nano | Radeon 680M | 2, WebRTC | 1 / 2 / 3 / 4 | 0.39–1.19 | 16 / 9–11 / 8–9 / 6.5–7.8 | 14 / 17 / 20 / 24 | — |
| Qwen3.5-9B | RTX 3090 (Vast.ai) | 2 on one GPU, WebRTC | 1 / 2 / 4 | 0.22–1.4 | 33 / 22–25 / 16–18 | 22 / 41 / 59 | — |
| Qwen3.8-27B | RTX 3090 (Vast.ai) | 1 | 1 / 4 | 0.43–2.27 | 20 / 7.5–9.4 | 17 / 29 | answers = 2- and 4-worker runs |
| Qwen3.8-27B | RTX 3090 | 2 on one GPU, WebRTC | 1 / 2 / 4 | 0.57–1.78 | 13 / 8.4–9.2 / 6.4–7.3 | 10 / 16 / 24 | matches llama.cpp, see below |
| Qwen3.8-27B | RTX 3090 | 4 on one GPU, WebRTC | 1 / 4 | 0.95–1.84 | 12 / 6.4–7.3 | 9.5 / 24 | answers = 1- and 2-worker runs |

- **Qwen3.8-27B correctness:** greedy answers equal llama.cpp's (b12432, same
  Q4_K_M file, thinking off) for the first 37–483 characters of four answers,
  one of them entirely, then differ where two next words are nearly tied. On
  the same GPU llama.cpp decodes at 39.5 tok/s alone, against 20 tok/s here
  with one worker.
- **Qwen3.5-0.8B correctness** (2026-11-15): greedy tokens equal the
  transformers reference on the same quantized weights.
- **Concurrency fix:** before this release Qwen3.5-0.8B crashed a contributor
  with three or more simultaneous chats (Qwen3.8-27B would have too); the rows
  above are after the fix.
- **localhost.run** tunnel: the model never became ready (pieces truncated,
  tunnel expired); it is not a supported share route.
- **Ready time** (contributors joining to chat): 0.8B 67–121 s, 27B 76 s (4
  workers) to 184 s (1 worker), 9B 396 s including a 7.4 GB download.

## Earlier verified runs (research reports, pre-release code)

Same kernels' lineage, run through the experiment harness or the hosted
panel; numbers are per report and were not rerun on the current code. Links
go to the report.

### Across real networks

| Model | GPUs and places | Workers, route | Context, measured at | Concurrency | TTFT (s) | tok/s per chat | Combined | Check | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | 3× RTX 3060 + Quadro RTX 4000; Poland, Spain, Sweden | 4, relay via Amsterdam | 8,320; 24 / 1,024 / 8,192-token prompts | 1 | 3.7 / 8.1 / 37.3 | 1.8 / 1.7 / 1.7 | — | first 48 tokens = transformers | [092](research/092-qwen35-webgpu-four-contributors.md) |
| Qwen3.5-9B | 4× RTX 3060, Poland | 4, relay | — | 1 / 8 | — | 1.7 / 1.2–1.6 | 1.7 / 16.2 | tokens = runs alone (12,098 tokens) | [093](research/093-qwen35-interleaved-conversations.md) |
| Qwen3.5-9B | 2× RTX 3060, Poland | 2, relay | 8,320; 24 / 8,192-token prompts | 1 | 1.1 / 29.2 | 4.1 / 3.9 | — | identical across runs | [095](research/095-qwen35-two-stages.md) |
| Qwen3.5-9B | 2× / 4× RTX 3060, Poland | 2 (one copy) / 2 copies of 2 | — | 8 / 32 / 16 | — | ≥ 0.6–1.8 | 15.6 / 19.6 / 29.1 | 360/360 conversations = alone | [096](research/096-qwen35-copies-on-one-coordinator.md) |
| Qwen3.5-9B | 2× RTX 3060, Poland, different machines | 2, relay → WebRTC | 24 / 8,192-token prompts | 1 | 0.9–1.2 / 24–31 | relay 4.7, WebRTC 12.0 | — | identical | [097](research/097-qwen35-webrtc-direct-links.md) |
| Qwen3.5-9B | 2× RTX 3060, Poland | 2, WebRTC | — | 1 / 8 | — | 9.6 / 4.6 | 9.6 / 36.2 | tokens identical | [110](research/110-batched-decode-rented.md) |
| S5-Nano | RTX 3060 + RTX 3060 Ti; Poland, Sweden | 2, relay / WebRTC / TURN | — | 1 | 0.95–1.4 (31 when a link failed to open) | 3.6 / 4.0 / 3.5 | — | texts identical; some links never opened | [098](research/098-user-sessions-webrtc-routes.md) |
| S5-Nano | 2× RTX 3060, Poland | 2, relay / WebRTC | — | 1 | 0.64–0.97 | 3.4 / 9.1 | — | identical after cancel and replace | [103](research/103-fast-sessions-rollout-rerun.md) |
| Kolibri-1 | A100 80 GB, Montana; production coordinator | 1, relay | 4,096 | 1 | 0.8–1.3 | 3.5 | — | coherent; — reference | [113](research/113-kolibri-single-a100-contributor.md) |

### One machine

| Model | GPU | Workers | Context, measured at | Concurrency | TTFT (s) | tok/s per chat | Combined | Check | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B | Radeon 680M | 4 on one GPU | 8,192-token prompt | 1 | 92 | 6.3 | — | = single-browser pipeline | [092](research/092-qwen35-webgpu-four-contributors.md) |
| Qwen3.5-9B | Radeon 680M | 1, chat path | 1,024 positions | 1 | 0.59 | 8.0 | — | — | [118](research/118-webgpu-gap-webllm-llamacpp-680m.md) |
| Qwen3.5-9B | Radeon 680M | 1, production contributor, resident decode | 64-token turns | 1 | — | 11.1 | — | tokens = non-resident | [129](research/129-resident-decode-qwen35.md) |
| Qwen3.5-9B | Radeon 680M | 2 on one GPU | — | 1 / 8 | — | 7.1 / 2.5 | 7.1 / 19.4 | tokens identical | [108](research/108-batched-decode-rows-per-thread.md) |
| Qwen3.5-9B | A100 80 GB | 1, pipeline harness | — | 1 | — | 24.4 | — | = NumPy reference (32 tokens) | [121](research/121-coalesced-matvec-kernel.md) |
| Qwen3.5-9B | A100 40 GB | 1, bench | 64 / 1,024 positions | 1 / 8 / 32 | — | 28 / — / — | 28 / 85 / 107 (Chrome); Node+Dawn 29 / 88 / 109; Deno+wgpu 17 / 47 / 58 | tokens identical across runtimes | [128](research/128-native-webgpu-a100.md) |
| TwIL-8B | Radeon 680M | 2, 3, 4 on one GPU | 4,096 | 1 | — | 6.6–7.1 | — | 32/32 tokens = reference | [101](research/101-twil-8b-dense-webgpu-family.md), [102](research/102-three-four-worker-plans.md) |
| Phi-4 | RTX 3090 | 2 on one GPU | 19-token prompt | 1 | — | 10.9 | — | 32/32 tokens = reference | [106](research/106-phi4-dense-webgpu-family.md) |
| Kolibri-1 | A100 80 GB | 1, pipeline harness, fused | 19-token prompt | 1 | — | 21 | — | 32/32 tokens = reference | [117](research/117-webgpu-fused-norms.md) |
| S5-Nano | Radeon 680M | 2 on one GPU, WebRTC | **16,384**, replies to 2,048 | 1 | 0.61 | 10.2 | — | — | [149](research/149-long-replies-full-context-local.md) |
| S5-Nano | Radeon 680M | 2, TURN with 1 / 123 / 243 ms RTT | 16,384 | 1 | 0.4 / 0.9 / 2.2 | 10.8 / 2.9 / 1.7 | — | — | [151](research/151-turn-rtt-decode-local.md) |

Kernel benchmarks (no served chat), for the ceiling: Qwen3.5-9B on an A100
reaches 13.65 ms per step (73 tok/s) at 64 positions with repacked weights
([142](research/142-k-repack-a100.md)); on the 680M 89–90 ms per step (11 tok/s)
([135](research/135-matvec-group-680m.md)). WebLLM on the same 680M: 8.9 tok/s
([118](research/118-webgpu-gap-webllm-llamacpp-680m.md)).

## Not collected yet

- **Qwen3.8-27B across machines:** every 27B run had all workers on one RTX 3090;
  no network between stages, no separate GPUs, no home links.
- **Current code over real networks:** today's runs are all on one machine;
  the cross-network numbers above are from pre-release code.
- **Long context on the current presets:** published bundles hold 4,096
  tokens; speed and TTFT were measured with short prompts only. Long-context
  results exist only for S5-Nano at 16,384 (research build) and Qwen3.5-9B at
  8,192.
- **More than 4 concurrent chats** on the current code (up to 32 in research
  benches).
- **Reference checks** for S5-Nano and Qwen3.5-9B on the current code, and
  TTFT under load for Qwen3.5-0.8B over LAN or Cloudflare with more than one chat.
- **Native contributors** (`--runtime deno`, `node`, `bun`) with the current
  presets: only research benches (A100) measured them.
- **Browsers other than Chrome/Chromium** (Firefox, Safari), Windows and macOS
  contributors, phones.
- **3-worker presets:** prepared and validated, never run.
- **Memory peaks:** GPU RAM per worker is the planner's estimate; peaks were
  not measured.
- **Training:** no timed distributed training run of the current code.
