# Phi-4 runs on the dense WebGPU family with reference parity and serves a panel chat

| | |
|---|---|
| **Conclusion** | **Works:** Phi-4's 32 greedy tokens match the reference, and it serves a panel chat (measured, Vast.ai RTX 3090). |
| **Status** | executed |
| **ID** | WS-14 |
| **Category** | browser |
| **Where** | remote: one Vast.ai GPU host. The WS ID was given when the run was planned for the workstation. |
| **Date** | Added 2026-11-09; started 2026-11-09; finished 2026-11-09 |
| **Path** | `microsoft/phi-4-gguf` at 6edc2ef, `phi-4-Q4_K.gguf` (Q4_K_M with Q4_K, Q5_K and Q6_K tensors; 40 layers, hidden 5120, 40/10 heads of 128, FFN 17920, vocab 100352, untied head; SHA-256 5652b9be…). Config and tokenizer come from `microsoft/phi-4` at 2db69c1. The `dense` family was extended for the `phi3` GGUF architecture: fused `attn_qkv` and `ffn_up` are served as row ranges, RoPE rotates halves, and the chat format has a `<\|im_sep\|>` separator. Code 3ba4ce0. One Vast.ai RTX 3090 24 GB (Indonesia), headless Chrome 157 WebGPU on Vulkan, with the coordinator and both contributors on that host. |
| **Result** | **All five gate lines passed.** (1) Kernels: within 6.0e-7 of the largest output for Q4_K, Q5_K and Q6_K at 1–64 tokens (gate 1e-5). (2) Stage, embedding plus 4 layers with the new half-rotation RoPE: per-row error within 9.3e-7, minimum cosine 0.9999998 (gates 1e-3 and 0.99999). (3) Two-stage pipeline at deployment precision (float16 cache and wire): 32/32 greedy tokens equal the reference's, with no reference margin below 0.068 and logits within 1.9e-4 of the largest (gate 2e-3); median 92 ms per token with both stages on the one GPU. (4) Chat format: the coordinator's token ids equal transformers' `apply_chat_template` in four cases. (5) Served user-panel session on a 2-worker bundle from `ygaraty model prepare`: every request answered on relay and direct routes, with identical texts, the 6-token limit held, and coherent answers. Cost about $0.08 of a $1 budget (18 minutes). |
| **Evidence** | Measured: [evidence/browser/phi4-dense-webgpu-2026-11-09.json](evidence/browser/phi4-dense-webgpu-2026-11-09.json) (kernel, stage and per-step pipeline errors, generated text, served answers, allocations, the converter-convention check, the chat check). Raw reports in `runs/experiments/WS-14/vast-20261109-2208/` and `runs/experiments/WS-14/chat-local/` (not in git); the gate was computed by `runs/experiments/WS-14/gate.py`. Offline: `tests/test_dense_family.py`, where a tiny Phi-4-shaped GGUF's reference layer equals transformers' `Phi3DecoderLayer` within 1e-4. |

## Question

Does the onboarding recipe that carried TwIL-8B ([101](101-twil-8b-dense-webgpu-family.md)) carry Phi-4 to a
served user-panel chat, by extending the `dense` family rather than adding a new one? Phi-4 is the first
checkpoint where llama.cpp's converter stores fused tensors and leaves the Q/K rows unpermuted. It is also the
first whose chat template is not plain ChatML. The Torch/CUDA owner path already serves Phi-4
([79](079-phi4-local-engine.md), [80](080-phi4-real-layers-cpu.md), [81](081-phi4-two-h100-pipeline.md)), so a
pass gives it a browser-contributor path.

Related: [DO-23](DO-23.md) asks whether WebGPU owners can run Phi-4 under DO-30's shared numerical contract
(our own checkpoint format, bf16/int4). This experiment used the GGUF dense family instead, so it doesn't
answer DO-23.

## Prediction

Every line passes, as in 101. The new code adds no arithmetic: the fused tensors become row-range views, and
RoPE changes only which pair of indices it rotates. Both were already checked offline against transformers.
The serving line was the most likely to need work, because Phi-4's chat format is new.

## Gate

Frozen 2026-11-09 22:02 CET (approved by the user), before any GPU run. All on one Vast.ai GPU host, headless
Chrome WebGPU (Vulkan), with the coordinator and both contributors on that host as in 101.

1. **Kernels** (`qwen35_webgpu kernels`, the dense cases: Q4_K, Q5_K and Q6_K at 1–64 tokens): every case's
   max |error| at most 1e-5 of the largest |output| (the contributors' admission tolerance).
2. **Stage** (`stage --first 0 --last 4 --chunks 9,1,1,1 --kv f32`, embedding plus 4 layers): every call's
   per-row max |error| / max |reference| at most 1e-3, and minimum cosine at least 0.99999. This line tests
   the new half-rotation RoPE kernel.
3. **Pipeline** (`pipeline --splits 0,20,40 --generate 32 --kv f16 --wire f16`, the deployment precision):
   every generated token equals the reference's greedy token, except where the reference's top-two margin is
   below 0.05; logits max |error| at most 2e-3 of the largest |logit| at every step.
4. **Chat format**: for one system and user message, and for a second user turn after an answer, the token
   ids the coordinator sends (`chat_prompt`, `chat_tokens`) equal the ids of transformers'
   `apply_chat_template` on `microsoft/phi-4`'s tokenizer at 2db69c1 with `add_generation_prompt=True`.
5. **Serving** (`scripts/user_session_check.py`, a two-worker Phi-4 bundle from `ygaraty model prepare`,
   arms relay and direct): every request answered, the 6-token request generates at most 6 tokens, request
   texts identical across the two arms, and the answers are coherent English (read by a person).

A failure of any line fails the gate. As in 101, the reference uses the same Q4_K_M weights as the browser,
so lines 1–3 measure the implementation, not the quantization.

## Protocol

The onboarding guide's steps (historical model-onboarding notes):

1. On the workstation: `inspect` (the pinned record is `deploy/models/phi-4.json`), `fetch` and `plan`, the
   converter-convention check against the bf16 checkpoint, and gate line 4, all on the CPU.
2. On one Vast.ai RTX 3090, with `NVIDIA_DRIVER_CAPABILITIES=all`, at 3ba4ce0 (pushed to main), the driver
   `runs/experiments/WS-14/ws14_remote.sh` (not in git; copied to the host) ran the following:
   - `remote.sh qwen35-contributor-setup` (the repository, Chrome, Vulkan and a WebGPU probe), the web
     requirements, and `ygaraty model fetch deploy/models/phi-4.json`.
   - The `kernels`, `stage` and `pipeline` checks (gate lines 1–3).
   - `ygaraty model prepare --workers 2,3,4`.
   - Two user-panel sessions on the 2-worker bundle (arms R, W; gate line 5), as in 101: a coordinator on
     localhost, a TURN host with nothing listening, and two headless Chrome contributors.

## Log

- 2026-11-09: Added, at 994a1e4+dirty (the `phi3` extension of the `dense` family is uncommitted).
- 2026-11-09: Offline checks passed. `tests/test_dense_family.py` gained a tiny Phi-4-shaped GGUF (fused
  Q8_0 `attn_qkv`, fused `ffn_up`, unpermuted rows). Its split tensors equal the source rows and are byte
  ranges of the fused tensors in a bundle. The reference layer equals transformers' `Phi3DecoderLayer`
  within 1e-4, and the `<|im_sep|>` chat format is derived from Phi-4's own template.
- 2026-11-09: `inspect` (metadata only): compatible, family `dense`, `rope_pairs: halves`. The chat starts
  empty and opens with `<|im_start|>assistant<|im_sep|>`. `plan` at capacity 4,096 gives 4.56 / 4.68 GiB per
  worker for 2 workers (`memory_gb` 6), 3.07–3.17 GiB for 3 (4), and 2.24–2.39 GiB for 4 (3). `fetch`
  downloaded and verified the files.
- 2026-11-09: Converter conventions on the real checkpoint (CPU, layer 0, against the bf16 weights at
  2db69c1). The split Q/K/V rows differ from `qkv_proj` by 0.037–0.044 (relative) unpermuted, against 1.39
  with Llama's permutation, so Phi-4's RoPE is over halves. Gate and up differ from `gate_up_proj` by 0.071
  in order, against 1.39–1.43 swapped. The reference's layer 0 equals transformers' `Phi3DecoderLayer` on
  the same dequantized weights within 4.8e-7 (outputs up to 4.06).
- 2026-11-09 22:02: gate frozen. The run moved from the workstation to Vast.ai at the user's request. Budget
  **$1**, teardown deadline **2.5 h** after `pods up` (the watchdog's `--minutes 150`), approved by the user.
- 2026-11-09 22:06: started at 3ba4ce0 (pushed to main). Pool `ws14`: one Vast.ai RTX 3090 24 GB
  (Indonesia, $0.236/h); the watchdog would delete it after 149 min. The driver logs and stamps each step on
  the host. It ran detached and was polled from the workstation every minute, instead of under `watch.py`.
- 2026-11-09 22:10: gate line 4 (chat format) **passed** on the workstation CPU. The coordinator's token ids
  equal transformers' `apply_chat_template` (microsoft/phi-4 at 2db69c1, `add_generation_prompt=True`) for
  system and user (`chat_prompt`), a session's first turn, a second turn after a finished answer, and a
  second turn after an answer cut at the token limit (`chat_tokens`).
- 2026-11-09 22:08–22:23 (host times converted to CET): setup took 7 min, kernels 23 s, stage 44 s and
  pipeline 3 min 51 s. `prepare` for 2/3/4 workers took 59 s, and the two served sessions 2 min 5 s. Every
  step exited 0.
- 2026-11-09 22:24: reports and bundle manifests copied to `runs/experiments/WS-14/vast-20261109-2208/`.
  `pods down ws14` deleted the machine, and `pods ls` showed 0 machines. It ran 18 minutes, about $0.08.

## Results

All measured on the RTX 3090 (`nvidia`/`ampere` adapter), Chrome 157, NVIDIA driver 595.99.02.

| Gate line | Measured | Gate | Pass |
|---|---|---|---|
| 1. Kernels | 12 cases (Q5_K ×5, Q4_K ×3, Q6_K ×4; 1–64 tokens): worst 5.97e-7 of the largest output | ≤ 1e-5 | yes |
| 2. Stage, layers 0–3, chunks 9,1,1,1, f32 cache | per-row error 9.26e-7 worst; cosine ≥ 0.9999998 | ≤ 1e-3; ≥ 0.99999 | yes |
| 3. Pipeline 0,20,40, f16 cache and wire | 32/32 tokens equal; smallest reference margin 0.068; logits within 1.93e-4 | tokens equal except near ties; ≤ 2e-3 | yes |
| 4. Chat token ids (CPU) | equal in 4 of 4 cases | equal | yes |
| 5. Serving, relay (R) and direct (W) | 3/3 requests answered per arm, identical texts, 6-token request gave 6 | as stated | yes |

The pipeline's prompt is "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain
is" (19 tokens). It continued: " Madrid. The capital of Germany is Berlin. The capital of the United Kingdom is
London. The capital of Canada is Ottawa. The capital of Australia is Canberra."

Pipeline timing, with both stages on the one GPU: weights loaded in 46 s, prefill of 19 tokens took 888 ms,
and decode took a median 92 ms per token. Allocation was 9.07 GB for both stages at the check's capacity.
On TwIL-8B in 101 the decode median was 140 ms on the workstation's Radeon 680M. The devices differ, so this
isn't a model comparison.

Served answers (both arms identical; 32/6/32 tokens, all cut at the limit):

- "The sky appears blue due to a phenomenon called Rayleigh scattering, where molecules and small particles in
  the Earth's atmosphere scatter sunlight in all directions. Blue light is"
- Follow-up limited to 6 tokens: "The sky is blue because shorter"
- "Copper is a versatile metal with a wide range of applications due to its excellent electrical and thermal
  conductivity, corrosion resistance, and malleability. Here are three"

Both sessions were ready in 64 s (R, which includes the first weight download into the browser profiles) and
26 s (W). Request times were 5.24 / 1.16 / 4.82 s on R and 3.57 / 1.02 / 3.22 s on W. The contributors share
one host, so these are not route measurements.

## What it taught

- Extending a family covered a new GGUF architecture without new arithmetic. The fused tensors became
  row-range views, which K-quants allow because their blocks never cross rows, and RoPE needed only a choice of
  index pair. Converter conventions really do differ between architectures, though: `phi3` keeps
  transformers' Q/K rows, while `llama` permutes them. The quick test is the one from the guide: compare
  dequantized rows against the bf16 weights both ways (0.04 against 1.39 here).
- Templates that are ChatML-shaped but not identical (Phi-4: `<|im_sep|>` after the role, nothing after
  `<|im_end|>`) need two pieces in the chat format, not a new protocol. Comparing token ids against
  `apply_chat_template` on the CPU checks this before any GPU time.
- The whole GPU part took 15 minutes of an RTX 3090 for about $0.08, so a rented GPU is a cheap substitute
  when the workstation's GPU is busy.

## Limits

- **One host.** One GPU served both stages and both contributors, so this measures neither links nor
  multi-machine latency, and W's "direct" links ran on one host.
- **Only the 2-worker bundle was served.** The 3- and 4-worker bundles were prepared, but not served or
  checked numerically.
- **Not deployed yet.** Installing on the production coordinator is a separate step: the handoff in
  historical deployment notes (not distributed here).
- **Implementation, not quantization.** The reference uses the same Q4_K_M weights, so this doesn't measure
  Q4_K_M's quality against bf16 Phi-4.
- **"Coherent English" was judged by Claude**, from the answers quoted above, not by the user.
- **One NVIDIA desktop GPU.** Other vendors' WebGPU stacks (AMD, Intel, Apple) weren't run for Phi-4; TwIL on
  AMD passed in 101.

## Reproduce

```bash
ygaraty model inspect microsoft/phi-4-gguf --gguf phi-4-Q4_K.gguf --revision 6edc2ef \
  --config-repo microsoft/phi-4 --config-revision 2db69c1 --out runs/models/phi-4-gguf/model.json
ygaraty model fetch deploy/models/phi-4.json --dir runs/models/phi-4-gguf
M=runs/models/phi-4-gguf; G=$M/phi-4-Q4_K.gguf
python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $G --config $M/config.json --out OUT/kernels
python -m ygaraty.experiments.qwen35_webgpu stage --gguf $G --config $M/config.json --tokenizer $M/tokenizer.json \
  --first 0 --last 4 --chunks 9,1,1,1 --kv f32 --out OUT/stage
python -m ygaraty.experiments.qwen35_webgpu pipeline --gguf $G --config $M/config.json --tokenizer $M/tokenizer.json \
  --splits 0,20,40 --generate 32 --kv f16 --wire f16 --out OUT/pipeline
ygaraty model prepare deploy/models/phi-4.json --dir $M --workers 2,3,4 --root OUT/models --bundle-root phi-4/v1
# a local coordinator and two browsers as runs/experiments/WS-14/ws14_remote.sh `serve`, with
# scripts/user_session_check.py --bundle phi-4/v1/2-workers/bundle.json --model-id phi-4 --splits 0,20,40 --memory-gb 6 --arms R,W
```

On Vast.ai: `YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up ws14
--provider vast --server RTX_3090 --workers 0 --minutes 150 --max-hourly 0.5`. Then `pods ship ws14`, copy the
driver to `/root/ws14.sh`, and run `setsid nohup bash /root/ws14.sh all` there. Finish with `pods down ws14`.
