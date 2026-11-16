# TwIL-8B runs on the new dense WebGPU family with reference parity and serves a user-panel chat

| | |
|---|---|
| **Conclusion** | **Works:** TwIL-8B's 32 greedy tokens match the reference, and it serves a panel chat (measured, workstation). |
| **Status** | executed |
| **ID** | WS-12 |
| **Date** | 2026-11-09 |
| **Path** | `webAI-Official/TwIL-8B` at 9563ba7, `TwIL-8B-Q4_K_M-balanced.gguf` (Granite 4.2 fine-tune; 40 layers, hidden 4096, 32/8 heads of 128, untied head; SHA-256 17bda82d…), pinned and prepared with the new `ygaraty model` commands, on the new `dense` model family: `dense_gguf.py` (loader; NumPy reference in the GGUF's layout) and `dense_gpu.mjs` (WebGPU stage on the shared kernels). Workstation Radeon 680M, Chrome WebGPU |
| **Result** | **All four gate lines met.** Kernels within 5.1e-7 of the largest output (gate 1e-5); stage (embedding and 4 layers) per-row error within 4.3e-7, minimum cosine 0.9999998 (gates 1e-3, 0.99999); two-stage pipeline at deployment precision (float16 cache and wire): 32/32 greedy tokens equal the reference's, logits within 2.5e-4 of the largest (gate 2e-3), median 140 ms per token with both stages on the one GPU; served user-panel session (two browser contributors, bundle from `ygaraty model prepare`): all requests answered on relay and direct routes with identical texts, the 6-token limit held, coherent answers. Planned per-stage memory matched the measured allocations within 0.2% |
| **Evidence** | Measured: [evidence/browser/twil-8b-dense-webgpu-2026-11-09.json](evidence/browser/twil-8b-dense-webgpu-2026-11-09.json) (kernel, stage and per-step pipeline errors, generated text, served answers, allocations); raw reports in `runs/experiments/WS-12/` and `runs/checks/twil-local-20261109-1613/` (not in git). Offline: `tests/test_dense_family.py` (tiny Granite GGUF written with llama.cpp's Q/K permutation: reference equals transformers' `GraniteDecoderLayer` within 1e-4) |

## Question

Does the onboarding recipe carry a model of a new architecture (Granite: dense, grouped-query attention,
head width 128, Granite's attention multiplier, untied head) from a Hugging Face repository to a served
user-panel session, with the browser stage matching the reference numerically?

## Prediction

Kernels within the device self-test tolerance (1e-5 of the largest output), stage outputs within float32
noise of the reference, greedy tokens identical except at near ties, and a coherent served chat.

## Gate

Frozen 2026-11-09 16:08 CET, before any GPU run. All on this workstation's Radeon 680M, Chrome WebGPU.

1. **Kernels** (`qwen35_webgpu kernels`, the dense cases: Q4_K and Q6_K at 1–64 tokens): every case's
   max |error| at most 1e-5 of the largest |output| (the contributors' admission tolerance).
2. **Stage** (`stage --first 0 --last 4 --chunks 9,1,1,1 --kv f32`, embedding plus 4 layers): every call's
   per-row max |error| / max |reference| at most 1e-3, and minimum cosine at least 0.99999.
3. **Pipeline** (`pipeline --splits 0,20,40 --generate 32 --kv f16 --wire f16`, the deployment precision):
   every generated token equals the reference's greedy token, except where the reference's top-two margin is
   below 0.05; logits max |error| at most 2e-3 of the largest |logit| at every step.
4. **Serving** (`scripts/user_session_check.py`, a two-worker TwIL bundle from `ygaraty model prepare`,
   arms relay and direct): every request answered, the 6-token request generates at most 6 tokens, request
   texts identical across the two arms, and the answers are coherent English (read by a person).

A failure of any line fails the gate. The bf16 safetensors are not compared numerically: the reference uses
the same Q4_K/Q6_K weights as the browser, so it measures the implementation, not the quantization.

## Protocol

The onboarding guide's steps (historical model-onboarding notes):

1. `ygaraty model inspect` pinned the commit and file hashes, read the GGUF header over HTTP range requests
   (1.9 s, no weight download), chose the `dense` family, passed its checks and derived the chat pieces from
   the template embedded in the GGUF (`<|im_start|>system\n<|im_end|>\n` to start;
   `<|im_start|>assistant\n<think></think>` to open the answer, thinking off).
2. `ygaraty model fetch` downloaded and verified the file; `plan` gave 4 / 3 / 2 GiB per worker for 2 / 3 / 4
   workers at a 4,096-token capacity.
3. The three GPU checks above, then `ygaraty model prepare --splits 0,20,40` and two local user-panel sessions
   (`scripts/user_session_check.py`, a coordinator on localhost, two headless Chromium contributors).

## Log

- 2026-11-09 16:08 CET: added; gate frozen. Already done offline: the GGUF's `attn_q`/`attn_k` rows match
  llama.cpp's permutation of the bf16 checkpoint (relative difference 0.08, the Q4_K error, against 1.40
  unpermuted; `attn_v` 0.02), confirming the adjacent-pair RoPE layout; tiny random Granite: NumPy reference
  equals transformers' `GraniteDecoderLayer` within 1e-4.
- 16:08: kernels passed. 16:09: stage passed. 16:10: pipeline passed.
- 16:11: first serving run failed at once ("Inference failed"): the bundle's chat format was stored as
  `self.chat`, which shadowed the session's `chat()` method. Renamed `chat_style`; a test now checks it.
- 16:14: serving passed on both routes.

## Results

| Check | Measured | Gate |
|---|---|---|
| Kernels (12 cases, Q4_K and Q6_K, 1–64 tokens) | worst 5.1e-7 of the largest output | ≤ 1e-5 |
| Stage (embedding + layers 0–3, chunks 9,1,1,1, f32 cache) | per-row ≤ 4.3e-7; cosine ≥ 0.9999998 | ≤ 1e-3; ≥ 0.99999 |
| Pipeline (0,20,40; f16 cache and wire; 32 tokens) | 32/32 tokens equal; logits ≤ 2.5e-4 (smallest margin 0.068) | all (near ties excepted); ≤ 2e-3 |
| Serving (relay and direct, three requests) | all answered; 32 / 6 / 32 tokens; identical texts; coherent | as stated |

Pipeline prompt "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is";
greedy continuation " Madrid." followed by the model's reasoning about the instruction (a raw prompt, not
chat). Served answers (both routes): "The sky appears blue because molecules and tiny particles in the
atmosphere scatter shorter (blue) wavelengths of sunlight more effectively than longer (red) wavelengths…";
"The sky is blue because atmospheric"; "1. Electrical wiring – copper is widely used in electrical systems…".
Requests took 5.1–5.6 s for 32 tokens with prefill, both stages sharing the 680M. Stage allocations 2.97 and
3.08 GB against the plan's 2.968 and 3.074 GB.

## What it taught

- **A new architecture fits the recipe as a family, not a fork.** The dense stage adds RoPE, the KV cache and
  grouped-query attention (about 300 lines) to the shared kernels; matrix products, norms, SwiGLU, gather and
  argmax are reused unchanged, and the pages choose the stage by `config.family`.
- **Checking a converter convention against the real weights is cheap and decisive.** One layer's dequantized
  rows against the bf16 checkpoint separated the permuted layout (0.08) from the unpermuted one (1.40) before
  any GPU run.
- **Chat formats can come from the checkpoint.** Rendering its own template with marker strings gives the
  ChatML pieces; Qwen bundles without them keep the old behaviour.
- **Bug found:** an attribute named like a method (`self.chat`) broke chat; renamed and tested.

## Limits

- One workstation GPU with both stages; no separate contributors' machines, no measured network latency.
- Only the two-worker split was prepared and served; the 3- and 4-worker plans in `twil_8b.json` are planned
  from memory estimates, not validated.
- Thinking is off (the template's `<think></think>`): the model card reports lower scores without reasoning;
  enabling it needs a larger generate budget per turn.
- Licence: webAI Non-Commercial License (the base Granite 4.2 is Apache 2.0).

## Reproduce

```bash
ygaraty model inspect webAI-Official/TwIL-8B --gguf TwIL-8B-Q4_K_M-balanced.gguf --out runs/models/twil-8b/model.json
ygaraty model fetch runs/models/twil-8b/model.json --dir runs/models/twil-8b
M=runs/models/twil-8b; G=$M/TwIL-8B-Q4_K_M-balanced.gguf
python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $G --config $M/config.json --out OUT/kernels
python -m ygaraty.experiments.qwen35_webgpu stage --gguf $G --config $M/config.json --tokenizer $M/tokenizer.json \
  --first 0 --last 4 --chunks 9,1,1,1 --kv f32 --out OUT/stage
python -m ygaraty.experiments.qwen35_webgpu pipeline --gguf $G --config $M/config.json --tokenizer $M/tokenizer.json \
  --splits 0,20,40 --generate 32 --kv f16 --wire f16 --out OUT/pipeline
ygaraty model prepare runs/models/twil-8b/model.json --dir $M --splits 0,20,40 --output RUN/models/twil-8b-2
# local coordinator and browsers as runs/experiments/DO-43/run_local.sh, with
# scripts/user_session_check.py --bundle twil-8b-2/bundle.json --model-id twil-8b --splits 0,20,40 --arms R,W
```
