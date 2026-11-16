# On the Radeon 680M our Qwen3.5-9B decode runs at 65% of native llama.cpp and WebLLM at 72%; per byte of weights the two browsers tie

| | |
|---|---|
| **Conclusion** | Found: Qwen3.5-9B decode on the 680M, ours 65% of native, WebLLM 72%; equal GB/s (measured) |
| **Status** | executed |
| **ID** | WS-30 (experiment 118) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Radeon 680M (RADV, shared LPDDR5), one engine at a time, the same model: **Qwen3.5-9B**. WebLLM 0.2.88 (`Qwen3.5-9B-q4f16_1-MLC`, library v0_2_87, served locally) in headless Chrome; ours (`qwen35_session local --chat --splits 0,32`, Q4_K_M GGUF, the chat path users get); llama.cpp b12318 Vulkan (`llama-bench`, the same Q4_K_M GGUF) |
| **Result** | Decode tokens per second, median: llama.cpp Vulkan 12.30 (short) and 12.19 (1,024 positions); ours 8.01 and 7.95 (124.8 and 125.9 ms per token), 65% of native at both depths; WebLLM 8.91 at short depth (2 runs), 72% of native; WebLLM at 1,024 positions did not complete (a GPU ring timeout during its prompt, twice). Effective bandwidth: llama.cpp 62.7 GB/s, ours 40.8, WebLLM 39.8 (its q4f16_1 weights are 4.47 GB per token against the Q4_K_M GGUF's 5.10): per byte the two browser engines tie, and both reach about two thirds of native. First token on the short prompt: ours 0.59 s, WebLLM 1.41 s. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/webgpu-gap-680m-2026-11-12.json`; raw `runs/ws30/` (llama/bench.json, ours/run1-3 reports and logs, webllm/run1-2 logs and Chrome logs); code 283965a |

## Question

WebLLM reports about 75% of native on an Apple M3 Max; nobody has measured it, us or native on our hardware. Which share of the 680M's bandwidth does each reach in decode, and so how big is our gap in a unit that compares different models?

## Prediction

llama.cpp Vulkan 50–70% of peak, WebLLM 35–55%, ours 15–25%.

## Gate (frozen 2026-11-12, before running)

Diagnostic: **Found**. The same model on the same GPU, decode only, greedy, 128 generated tokens after a warm-up, at two depths: short (prompt under 64 tokens) and about 1,024 positions. 3 runs per engine and depth; the median.

- Per engine: decode tokens per second (WebLLM: its runtime stats; ours: the coordinator's ms per token after the first token; llama.cpp: `llama-bench -p 0 -n 128 -d 0,1024 -r 3 -ngl 99`).
- **Share of native** = tokens per second ÷ llama.cpp's at the same depth: the headline number for WebLLM and for us.
- Effective bandwidth = weight bytes read per token (the GGUF's tensors without the token embedding; for WebLLM its parameter shards without the embedding) × tokens per second, also as a share of the device's measured copy bandwidth (WS-31's copy kernel; `dmidecode` needs root here, so the memory type is unverified: the Ryzen 9 PRO 6950H takes DDR5-4800, 76.8 GB/s, or LPDDR5-6400, 102.4 GB/s).
- Each engine's output sanity-checked (a coherent answer; tokens are not compared, the quantizations differ).

## Protocol

A WebLLM page served locally (its npm package and model files fetched once into `runs/`), driven by the same headless Chromium; `llama-bench -p 0 -n 128`; our `bench`. One engine at a time on the 680M (it drives the display).

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-12: gate frozen (above); started. llama.cpp b12318 (ubuntu-vulkan-x64 release), WebLLM 0.2.88 (npm) with `mlc-ai/Qwen3.5-9B-q4f16_1-MLC` and its library `v0_2_87/base/Qwen3.5-9B-q4f16_1_cs1k-webgpu.wasm`, fetched into `runs/ws30/`. WebLLM 0.2.88 ships Qwen3.5-9B, so the comparison uses one model throughout (the Qwen3-8B plan was dropped before any run).
- 13:3x: llama.cpp `llama-bench -p 0 -n 128 -d 0,1024 -r 3 -ngl 99`: 12.30 ± 0.00 and 12.19 ± 0.01 tokens per second.
- 13:40–13:46: ours, 3 sessions (`runs/ws30/ours.sh`, under watch.py, stall 600 s): warm-up, short (25 prompt tokens), reset, long (1,027 prompt tokens), 128 tokens each; all exit 0, answers coherent.
- 13:47: WebLLM run 1 (`runs/ws30/webllm_bench.py`, headless Chrome with our harness's flags): loaded in 14.5 s; warm-up 8.60, short 8.91 tokens per second; the long prompt ended in `VK_ERROR_DEVICE_LOST`: the kernel log shows an amdgpu gfx ring timeout from that Chrome and a ring reset (succeeded; the display recovered). Protocol amendment: run 2 with `prefill_chunk_size: 128` (prefill only; decode unaffected).
- 13:49: WebLLM run 2: short 8.91 again; the long prompt hit the same ring timeout. Not retried: the 680M drives the display, and a third reset was not worth the long-depth number (both other engines change by under 1% between depths).

## Results

| Engine | Weights per token | Short depth tok/s | 1,024 positions tok/s | Share of native | Effective GB/s |
|---|---|---|---|---|---|
| llama.cpp b12318, Vulkan | 5.10 GB (Q4_K_M) | 12.30 | 12.19 | 100% | 62.7 |
| WebLLM 0.2.88 | 4.47 GB (q4f16_1) | 8.91 | did not complete | 72% | 39.8 |
| Ours (chat path, one stage) | 5.10 GB (Q4_K_M) | 8.01 | 7.95 | 65% | 40.8 |

## What it taught

- On this bandwidth-starved iGPU the gap to WebLLM is the quantization, not the kernels: WebLLM moves 12% fewer bytes per token (q4f16_1 has no 6-bit tensors; Q4_K_M keeps some Q6_K) and is 11% faster; per byte of weights we are level.
- Both browser engines stop at about two thirds of native (40 against 63 GB/s). The last third is the kernels' share of peak bandwidth (WS-31, WS-34) and per-token overhead.
- The A100 gap is of a different kind: there the step is host-bound (experiment 117: 28 ms GPU of 47), which the 680M (125 ms per token, GPU-bound) hides. The plan's host-side experiments (WS-32, WS-33) matter for big GPUs; the kernel experiments for both.
- Our prefill is faster for short prompts (first token 0.59 against 1.41 s) and never hung the GPU: our prompt chunks are 64 tokens. WebLLM's Qwen3.5-9B prompt of about 1,000 tokens hung the 680M's gfx ring twice.

## Limits

- One device, one model; WebLLM's long-depth number is missing and its short depth has 2 runs, not 3.
- Different quantizations by design (each engine's standard 4-bit build), so tokens per second and bytes per token are both reported; share of native uses tokens per second.
- The 680M's peak bandwidth is not measured (no root for `dmidecode`; WS-31's copy kernel will measure it); llama.cpp's 62.7 GB/s is the practical roof here.
- Our number is the chat path through the local coordinator (DO-61's local loop, one stage); llama.cpp's is a pure decode loop.

## Reproduce

```bash
# llama.cpp
curl -LO https://github.com/ggml-org/llama.cpp/releases/download/b12318/llama-b12318-bin-ubuntu-vulkan-x64.tar.gz
LD_LIBRARY_PATH=. ./llama-bench -m runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf -p 0 -n 128 -d 0,1024 -r 3 -ngl 99 -o json
# ours (prompts in runs/ws30/{short,long}_prompt.txt)
bash runs/ws30/ours.sh
# WebLLM: npm pack @mlc-ai/web-llm@0.2.88 into runs/ws30/webllm-npm; python3 runs/ws30/fetch_mlc.py
python runs/ws30/webllm_bench.py runs/ws30/webllm/run.json   # avoid the long prompt on a display GPU
```
