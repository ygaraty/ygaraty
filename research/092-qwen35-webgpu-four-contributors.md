# Qwen3.5-9B Q4 runs on four rented WebGPU contributors: 580 ms per token, first token after 37 s at 8,192 tokens

| | |
|---|---|
| **Conclusion** | **Works:** Qwen3.5-9B Q4 ran on four physical WebGPU contributors, 580 ms per token, reference tokens kept (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-36 |
| **Category** | browser |
| **Where** | workstation (kernels, reference, dry runs on an AMD Radeon 680M), then Vast.ai and DigitalOcean |
| **Date** | Added and run 2026-11-06 |
| **Path** | Qwen3.5-9B Q4_K_M (unsloth GGUF, packed weights) split into four consecutive 8-layer stages (A with the embedding, D with the output projection), each in its own Chrome tab on its own GPU: RTX 3060 12 GB (Poland), RTX 3060 12 GB (Poland, same site), Quadro RTX 4000 8 GB (Spain), RTX 3060 12 GB (Sweden), Chrome 157 on Xvfb, ANGLE on Vulkan. Float32 arithmetic, float16 KV cache and wire, 64-token prefill chunks, capacity 8,320 positions; a coordinator on a DigitalOcean s-4vcpu-8gb in Amsterdam admits each browser after a self-test and relays every hidden state and token over https |
| **Result** | **Gate met:** three sessions completed on four separate physical WebGPU contributors, each stage on its own GPU, every device admitted by the self-test (worst kernel 5.3e-7), no device loss. 24-, 1,024- and 8,192-token prompts with 128 tokens each: first token 3.7, 8.1 and 37.3 s after go; 568, 577 and 580 ms per token (p95 635–698), about 300 ms of compute and 270 ms of relay. Tokens: the chat session's first 48 equal the transformers reference's; the 8,192-token session's 128 equal an AMD run's exactly. Application GPU memory 1.07–2.04 GB per contributor. Cost about $1.15 at listed prices, budget $10 |
| **Evidence** | Measured: [evidence/browser/qwen35-webgpu-contributors-2026-11-06.json](evidence/browser/qwen35-webgpu-contributors-2026-11-06.json) (per-session timings, tokens, per-stage self-test, loads, memory, compute); raw reports in `runs/experiments/DO-36/rented-20261106-session/{chat,k1,k8}/` and local runs `runs/experiments/DO-36/omarchy-20261106-*` (not in git). Code 611f007 (sessions), chat in 83ee843; tests `tests/test_qwen35_gguf.py`, `tests/test_qwen35_session.py` |

## Question

Can four friends' browser contributors, each supplying 4 GB of usable GPU
memory, jointly serve Qwen3.5-9B with correct prefill and cached decoding? Measure
time to first token and generation latency on their real links, including an
8,192-token prompt followed by up to 128 generated tokens.

Related: [DO-23](DO-23.md) and [DO-30](DO-30.md) define the shared native/WebGPU
numerical contract; [experiment 83](083-twil-int4-four-capped-gpus.md) tested
TwIL-8B on four native GPUs capped at 3.5 GiB and failed near 30k tokens;
[experiment 87](087-persistent-sessions-internet.md) measured persistent sessions
on a different, closer native-GPU pool. Neither establishes Qwen browser support.
Reuse their infrastructure and memory lessons, but validate Qwen's own operations.

## Prediction

- **Capacity:** the published [Q4_K_M text-model file](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/main)
  is approximately 5.68 GB, averaging 1.42 GB per contributor before state and
  scratch. The end stages also own the embedding and output projection, so equal
  layer counts do not imply equal memory. Keep weights packed during execution.
- **State:** the [model configuration](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)
  has 32 layers, with full attention every fourth layer. Eight layers per stage
  therefore gives two full-attention KV caches and six recurrent states. FP16 KV
  storage at 8,192 positions is 64 MiB per stage, calculated as
  `2 layers * 2 (K,V) * 4 heads * 256 dimensions * 2 bytes * 8192 positions`.
  Recurrent state, scratch, transfer buffers and browser overhead are additional.
- **Communication:** a 4,096-element hidden vector occupies 8 KiB on an FP16 wire
  or 16 KiB on an FP32 wire. An 8,192-token prompt sends about 64 MiB per stage
  boundary with FP16. Each decoded token traverses the full pipeline before the
  next can begin; capacity pooling does not promise a fourfold single-chat speedup.
- These are analytical estimates. Actual browser peaks, kernel throughput and
  link latency remain unmeasured; no tokens-per-second outcome is assumed.

## Gate

**Frozen 2026-11-06 by the user: the only gate is running with WebGPU
contributors** — one session of Qwen3.5-9B Q4_K_M completes on four separate
physical WebGPU contributors (item 1 below), each stage on its own
contributor. Items 2–4 are no longer pass/fail: memory, correctness against
the reference and latency are measured and reported, not gated.

The earlier proposal, kept for context:

1. **Actual browser execution:** all model stages, including embedding and final
   projection, execute on WebGPU across four separate physical contributors.
   No CPU/native service executes browser-assigned model work. Multiple tabs on
   one GPU or capped native CUDA workers do not satisfy this gate.
2. **Memory and completion:** every contributor stays within 4,000,000,000 bytes
   of live application-owned GPU allocations, including packed weights, state,
   scratch and staging buffers. Complete the declared prompt/decode lengths with
   no allocation failure, device loss or context truncation. Record browser,
   process and driver memory separately where observable; an application buffer
   counter alone does not establish total physical GPU usage below 4 GB.
3. **Correctness:** compare operations, blocks, final logits, chunked prefill and
   cached decoding with an independent reference using the identical quantized
   weights and declared boundary conversions. Proposed target: identical greedy
   tokens on three fixed prompts with 128 generated tokens each, subject to EOS.
   Numerical tolerances and any near-tie exception are TBD and must be frozen
   before results are seen. Report comparison with BF16 separately as a
   quantization effect, not the distributed-execution baseline.
4. **Latency:** report cold startup, warm time to first token, median/p95
   inter-token latency and tokens/s for every prompt length. The threshold for
   calling the experience useful is TBD; completion alone is a capacity result.

## Protocol

1. On authorized external test hosts, prepare pinned model/tokenizer artifacts,
   prequantized stage shards, reference fixtures and versioned browser kernels.
   Start with Q4_K_M if supported. Any alternate Q4 format requires its own
   explicit scales/rounding contract and matched reference; format support is not
   implied by the existence of a GGUF file. Omit vision and speculative/MTP paths.
2. Implement and validate Qwen's Gated DeltaNet, convolution/recurrent state,
   gated full attention, normalization, RoPE, dense FFNs, embedding and output
   projection through the shared numerical framework. Reuse DO-23/DO-30 work;
   their complete Phi-4 experiments are not a prerequisite for Qwen layer checks.
   Validate FP32 fixtures and the selected FP16/FP32 accumulation/state policy,
   then packed Q4 arithmetic, before attempting the four-browser gate.
3. Inventory four distinct devices, browsers, adapter features, shader-f16 and
   buffer/binding limits. Establish allocation headroom on each; target below
   the 4 GB cap to allow implementation overhead. Partition large tensors into
   supported buffers. Count weight-loading and upload peaks as well as steady
   inference memory; do not expand the complete shard to floating point on GPU.
4. Begin with A owning embedding and layers 1-8, B layers 9-16, C layers 17-24,
   and D layers 25-32 plus final projection. Rebalance contiguous ranges for
   measured stage memory and speed, then freeze the placement for timed runs.
   Each contributor loads only its shard and keeps weights, KV and recurrent
   state resident. Preallocate session capacity for at least 8,320 positions
   for the 8,192-prompt plus 128-output case. Use bounded prefill chunks.
5. Use persistent A -> B -> C -> D activation handoffs, with the next token ID
   fed back to A and text streamed to the client. Prefer direct peer channels
   where available; record actual direct/relay routes, one-way timing where
   measurable, RTT, upload bandwidth and readback/upload overhead. A coordinator
   handles admission and placement without requiring a fresh scheduling cycle
   for every layer. Its execution host must also be outside this simulation-only
   machine. Initial scope is one session with all four contributors available.
6. Pin prompt token IDs at lengths 128, 1,024 and 8,192; generate up to 128 tokens,
   recording EOS and actual lengths. Include three fixed coding/instruction
   fixtures for correctness. Use at least one warmup and five measured repeats
   per length, with separate cold-start measurements. Do not infer request-tail
   guarantees from this small sample. FP16 boundary conversion must be present
   in the reference if used; an FP32 diagnostic is a separately labeled arm.
7. Save commit, checkpoint/shard hashes, browser manifests, allocation traces,
   stage timings, transport bytes, reference logits/tokens, exact commands and
   raw results under `runs/experiments/DO-36/<host>-<timestamp>/`. Distinguish
   calculated predictions, numerical checks and measured remote performance.

As run: steps 1–4 as written; step 5 through the coordinator only (no direct
peer channels); step 6 one session per prompt length (24, 1,024 and 8,192
tokens, 128 generated each), not the five repeats and three coding fixtures,
since the gate frozen by the user needs only one completed session.

## Log

- 2026-11-06: Added on the user's explicit request for Qwen3.5-9B on four WebGPU
  contributors with 4 GB each. Status is open; no weights downloaded, services
  started or results claimed.
- 2026-11-06: Plan agreed with the user: rehearse on rented machines (four
  separate GPUs running Chromium with WebGPU, budget to be approved), then the
  real-link run on friends' devices. Milestone 1, free and local: one stage
  matching the reference on this workstation's GPU (AMD Radeon 680M, RDNA 2,
  Chromium 155, Vulkan; `shader-f16` and subgroups available).
- 2026-11-06: Inputs. `unsloth/Qwen3.5-9B-GGUF` Q4_K_M at revision 3885219,
  5,680,522,464 bytes, sha256 03b74727…b7e8 (matches the hub's LFS hash);
  config and tokenizer from `Qwen/Qwen3.5-9B` at c202236. The file mixes four
  block types: Q4_K, Q5_K (DeltaNet `attn_qkv`, `ssm_out`), Q6_K (`output`,
  14 of 32 `ffn_down`, 4 `attn_v`) and Q8_0 (`ssm_alpha`, `ssm_beta`), so the
  kernels decode all four.
- 2026-11-06: Built (uncommitted until now): `inference/qwen35_gguf.py`, which
  reads the GGUF and runs transformers' own decoder layers in float32 on the
  dequantized weights as the reference, after undoing llama.cpp's five
  conversions (`ssm_a = -exp(A_log)`, norms stored +1, conv squeezed,
  `dt_bias` renamed, value heads tiled; `tests/test_qwen35_gguf.py` checks
  the head order against the converter's own reorder). The WebGPU side,
  `webapp/static/qwen35_gpu.mjs`, keeps the file's bytes packed and runs in
  float32: one matrix-vector kernel per block type, RMSNorm (per head, gated),
  the causal convolution and its state, the gated delta rule (one workgroup
  per value head, tiled order), partial RoPE, a KV cache, blockwise
  online-softmax attention, the chunked embedding gather. Harness
  `experiments/qwen35_webgpu.py` (`kernels`, `stage`) over a binary-file
  page runner, `experiments/browser_binary.py`.
- 2026-11-06: Development checks (tolerances not yet frozen; not the gate).
  Kernels: each block type's matrix times random inputs within 1.8e-7 of the
  largest output (NumPy float64). Stage A (embedding, layers 0–7) against
  transformers: 19-token prompt, prefill then 3 cached steps, worst per-token
  error 3.4e-7 of the row's largest value (min cosine 0.9999999); 743-token
  prompt in 64-token chunks then 3 cached steps, worst 1.7e-6, DeltaNet states
  within 5.7e-6 after 743 updates. Application GPU allocations after loading
  stage A for 743 positions: 1,600 MiB (float32 KV); the browser's own memory
  is not measured yet. Timing on the 680M: prefill about 2.1 ms per token per
  layer, a cached step about 12 ms per layer (naive kernels; not optimized).
  Runs in `runs/experiments/DO-36/omarchy-20261106-*`.
- 2026-11-06: Found: one submission of 8 layers × 128 tokens (about 2.2 s)
  exceeded the amdgpu job timeout; the kernel reset the GPU ring (it
  recovered). The stage now submits one layer at a time; Windows' TDR (about
  2 s) makes this a requirement for contributors too. Prefill chunk size must
  keep every submission well below that on the slowest device.
- 2026-11-06: Item 1 of the next steps done: every stage, the output layer
  and greedy choice, and all four stages in one browser. A later stage takes
  hidden states in (`stage --first`); the last stage adds the output norm,
  the output projection in 7 chunks of at most 128 MiB, and a GPU argmax.
  Stage D alone (layers 24–31 plus output, on the reference's hidden input
  after layer 23): worst per-token error 1.1e-6, states within 5.4e-7.
- 2026-11-06: Pipeline check (`pipeline`, splits 0,8,16,24,32, hidden states
  read back to the CPU between stages as the wire would carry them): a
  24-token chat prompt ("Write a Python function that returns the n-th
  Fibonacci number", thinking off), prefill, then 24 greedy tokens chosen on
  the GPU. Reference: the prompt plus the generated tokens in one causal pass
  through transformers, one stage at a time, then the output layer at each
  generated position (teacher forced). **All 24 tokens equal the reference's
  greedy choice**; logits within 1.7e-5 (at most 7e-7 of the largest), the
  smallest top-two margin 0.38. Text: "Here’s a simple and efficient
  iterative implementation in Python: … def fibonacci(n: int) -> int:".
  Application GPU allocations per stage at 48 positions: A 1.59 GB, B 1.02,
  C 1.01, D 1.86 (the output projection 0.80); at 8,320 positions the float32
  KV adds about 0.13 GB per stage. One iGPU running all four: prefill of 24
  tokens 1.7 s, a token every 398 ms (median; p95 407; per stage 90, 90, 92,
  123 ms). Run `omarchy-20261106-115712-pipeline`.
- 2026-11-06: Item 2, precision policy (proposed; part of the gate to
  freeze). Float32 for all arithmetic, the residual stream, norms, the
  DeltaNet's convolution and recurrent states and the logits; float16 for
  the attention KV cache and for hidden states on the wire between stages.
  `shader-f16` is not required: the cache packs two halves per u32 word.
  Found: WGSL leaves `pack2x16float`'s rounding to the implementation, and
  on the 680M (RADV) it rounds toward zero (54% agreement with NumPy on the
  test set, 100% toward zero on normal values); `unpack2x16float` may flush
  subnormals. The kernels therefore convert in integer arithmetic, round to
  nearest even like NumPy and JavaScript's `Float16Array`, so every
  contributor stores the same bits (`f16` mode: 262,144 values including
  exact ties, subnormals, overflow and specials, all bit-identical; back
  conversion exact). With truncation, stage A's error against the
  float16-rounding reference was 2e-4; with the exact conversion 1.2e-5.
  The reference applies the same roundings (`qwen35_gguf.kv_cache`: keys and
  values as stored; hidden states at each boundary).
- 2026-11-06: Item 2, faster kernels. K-quant matrices are decoded a 32-bit
  word at a time (pairs of 32-element units sharing their quant bytes):
  decode uses a matrix-vector kernel (4 rows per 64-thread workgroup,
  single-token variant), prefill a tiled kernel (16 rows x 16 tokens, the
  inputs staged in 16 KiB of workgroup memory). Attention: a workgroup per
  (token, KV head, 512 keys), the 4 query heads sharing each K/V read, then
  a combine pass; cache laid out [head][position][256]. Submissions hold at
  most 64 token-layers (a decode step submits the whole stage). Kernel
  checks: all within 3.7e-7 of the largest output (`kernels`, 10 cases
  incl. unaligned Q6_K blocks and a row count not a multiple of 16);
  single-token weight streaming 21–37 GB/s (it was about 10). Stage A,
  743 tokens in 64-token chunks plus 3 steps: float32 cache 2.1e-6 (as
  before), float16 cache 1.2e-5 against its reference; states within
  6.6e-6. Runs `omarchy-20261106-13*`.
- 2026-11-06: Pipeline under the policy (FP16 cache and wire), same chat
  prompt, 48 greedy tokens: **48/48 equal the same-policy reference**;
  logits within 3.4e-3 (1.1e-4 of the largest; a hidden value flipping one
  float16 step at a boundary is up to 0.03 at |x| near 64, so the agreement
  is looser than the float32 run's 1.7e-5); smallest margin 0.24. The
  policy's own effect (the reference with and without it): logits differ
  by at most 0.012, the greedy token is the same at 48/48 steps. Largest
  |hidden| at a boundary 87 (float16's limit is 65,504). Decode on the one
  iGPU running all four stages: 149 ms per token median (p95 153; stages
  34, 33, 33, 50 ms), was 398. Run `omarchy-20261106-133849-pipeline-f16`.
- 2026-11-06: Item 3, real memory per contributor (`contributors`: each
  stage alone in its own headless Chromium, one after another, capacity
  8,320 positions; an 8,192-token prompt, the first tokens of experiment
  048's report, sha256 in the run's report.json, in 64-token chunks, then
  128 single-token steps; each stage's inputs are the previous stage's
  FP16 wire outputs). A sampler reads the browser's process tree every
  0.25 s: PSS per process type and the GPU memory amdgpu attributes to its
  DRM clients (fdinfo `drm-total-vram`/`-gtt`). Peaks, stage A / B / C / D:
  application allocations 1.77 / 1.17 / 1.15 / 2.04 GB; driver-attributed
  GPU memory (all in the GPU process) 1.84 / 1.24 / 1.23 / 2.11 GB, so the
  browser's own GPU overhead is 60–70 MB; system memory (PSS, all
  processes) 0.94 / 0.73 / 0.69 / 1.06 GB, mostly the renderer (0.41–0.73 GB,
  the page's JS and weight downloads) and the GPU process (0.12–0.17 GB).
  Every stage is within the 4 GB cap with at least 1.9 GB to spare. On
  this 680M the GPU memory is the 3 GB carve-out (VRAM counters), not GTT.
  Timing per contributor: load 3.5–6.1 s; prefill of 8,192 tokens
  75.6–77.8 s (a 64-token chunk 0.60 s at the start, 0.63 s at position
  8,128); a step at about 8,200 positions 35 ms (D: 54 ms). Stage D's
  greedy tokens continue the report's table as the text does. Run
  `omarchy-20261106-134836-contributors` (an earlier identical run,
  `…-134207-…`, had the process types mislabeled; same totals).
- 2026-11-06: Faster prefill. A register-tiled kernel (`gemm`) for 16 or
  more tokens: a 128-thread workgroup computes 64 rows x 64 tokens, each
  thread 4 rows x 8 tokens; per 32-column step the rows' decoded weights
  and the tokens' inputs are staged in 16 KiB of workgroup memory; each
  256-column block is summed separately before being added to the total
  (without that, the error of 12,288-column rows was 10x the other
  kernels'). 5–15 tokens keep the tiled kernel, 1–4 the matrix-vector one.
  At 64 tokens on the 680M: 1.4–1.9 TFLOPS (ffn_up 3.45 ms, ffn_down 4.56,
  attn_qkv 2.71), was 0.40 (2.6–4.5x); tile shapes 4x4, 8x4, 8x8 and 32-
  or 64-wide tiles were slower (`--gemm`, runs `…-1453*`–`…-1456*`).
  Kernel checks within 5.5e-7 of the largest output. Stage A (743 tokens):
  1.14e-5 against the reference, states within 6.7e-6 (as before); a
  64-token chunk of 8 layers 160 ms, was 550. Pipeline: 48/48 tokens equal
  the reference and the previous run's, logits within 3.0e-3, prefill of
  24 tokens 0.65 s (was 1.20), decode unchanged at 148 ms. Contributors
  (8,192-token prompt as above): prefill 26.0–28.1 s per stage, was
  75.6–77.8 (2.8–2.9x; a 64-token chunk 0.24 s including the wire
  transfer); stage D's 128 predictions identical to the previous run's;
  memory unchanged. Runs `omarchy-20261106-145714-stageA-gemm`,
  `…-145818-pipeline-gemm`, `…-150131-contributors-gemm`.
- 2026-11-06: Gate frozen by the user: the only gate is running with WebGPU
  contributors (see Gate). Built the contributor path
  (`experiments/qwen35_session.py`, `webapp/static/qwen35_contributor.mjs`):
  1. Live handoff: a coordinator (stdlib HTTP, optional TLS, session key in
     the join URL) assigns admitted browsers to stages A–D in arrival order
     and relays FP16 hidden states A → B → C → D and D's token back to A
     through per-stage long-poll mailboxes; each stage forwards a prefill
     chunk as soon as it is done, so prefill is pipelined across machines.
  2. Join page and weights: the page serves people and headless launchers
     alike; each stage downloads only its byte ranges of the GGUF, checks
     each file's SHA-256 (computed once by the coordinator, cached beside
     the GGUF) and keeps it in the Cache API.
  3. Self-test before admission: 9 kernel cases on slices of the model's
     own weights (all four matmul kernels, Q4_K/Q5_K/Q6_K/Q8_0, unaligned
     Q6_K blocks, odd row counts) must be within 1e-5 of the largest
     output (the 680M: 5.0e-7), and the cache's and the wire's float16
     conversions bit-identical to NumPy (the page falls back to an integer
     conversion where Float16Array is missing; bit-identical on 262,144
     values in Node). `tests/test_qwen35_session.py`: wrong kernel output,
     truncating float16 and missing outputs are rejected; a fifth browser is
     told the stages are taken.
  4. Dry runs on this machine (four headless Chromiums, one GPU; not the
     gate): the chat prompt, 48 tokens identical to the single-browser
     pipeline's, 155–157 ms per token (per-stage compute 32/32/31/52 ms,
     relay overhead about 3 ms per token on localhost), first token 0.7 s
     after go; cold load 6–15 s per stage (1.0–1.9 GB downloaded), from the
     cache 1.2–2.0 s. 8,192-token prompt + 128 tokens: completed, first
     token 92 s after go (all four stages share this GPU, so the pipelining
     cannot help here: about 83 s of prefill compute per stage when shared),
     median 160 ms per token (p95 166), 68 MB per hop; GPU memory per
     browser (driver) 1.83 / 1.23 / 1.22 / 2.10 GB. Same with `serve` and
     four separate `browser` launchers over https (self-signed, 16 tokens).
     Runs `omarchy-20261106-1708*`–`…-1715*`.
- 2026-11-06: Rented rehearsal approved by the user: budget $10, every
  machine deleted within 3 hours of the first one starting. Plan: one Vast.ai
  GPU container as a WebGPU probe; then four Vast.ai GPU containers on four
  different hosts (Europe) as contributors and a DigitalOcean CPU Droplet
  (ams3) as the coordinator over https; the chat prompt with 128 tokens, a
  1,024-token prompt, the 8,192-token prompt with 128 tokens. DigitalOcean
  offers this account no GPU Droplets; Vultr is paused. Setup modes in
  `scripts/remote.sh` (`qwen35-*`).
- 2026-11-06: Probe (Vast.ai RTX 3060 12 GB, Poland, $0.075/h, from 16:30
  UTC): headless Chrome got WebGPU only on SwiftShader. Two causes: the
  container lacked GLVND's libEGL, which the NVIDIA Vulkan driver loads
  (`vkCreateInstance` missing; fixed by installing libegl1), and the driver
  has no `VK_EXT_headless_surface`, so headless Chrome's Vulkan start fails
  (−7) and Chrome falls back to software GL, which exposes only SwiftShader.
  Fix: a regular Chrome on Xvfb with ANGLE on Vulkan (`YGARATY_CHROME_X11=1`,
  set by remote.sh's qwen35 modes): adapter nvidia/ampere; float16 check
  bit-identical to NumPy (here WGSL's `pack2x16float` also rounds to
  nearest even, unlike RADV's).
- 2026-11-06: **Gate met: Qwen3.5-9B Q4_K_M ran on four separate physical
  WebGPU contributors.** Vast.ai containers (Chrome 157 on Xvfb, ANGLE on
  Vulkan): RTX 3060 12 GB (Poland, machine 21698), RTX 3060 12 GB (Poland,
  machine 17972, same site and public address as the first), Quadro RTX
  4000 8 GB (Spain), RTX 3060 12 GB (Sweden); coordinator on a DigitalOcean
  s-4vcpu-8gb in ams3, https with a self-signed certificate; every hidden
  state and token relayed through it. Every device passed the self-test
  (worst kernel 5.3e-7; the Quadro's WGSL `pack2x16float` truncates like
  RADV, the page's own conversion is exact). Three sessions, all completed:
  | Prompt | First token after go | Per token median / p95 | Bytes per hop |
  |---|---|---|---|
  | chat, 24 tokens | 3.7 s | 568 / 635 ms | 1.2 MB |
  | 1,024 tokens (048's report) | 8.1 s | 577 / 681 ms | 9.4 MB |
  | 8,192 tokens (048's report) | 37.3 s | 580 / 698 ms | 68 MB |
  Tokens: the chat session's first 48 equal the single-browser pipeline's
  (which equal the reference); the 8,192-token session's 128 equal the local
  four-browser run's on the 680M (AMD) exactly. Decode: stage compute per
  step 38–94 ms (the RTX 3060 stages 38–65 ms, the Quadro 89–93 ms), about
  300 ms per token in all; the rest (about 270 ms) is the relay: five
  internet hops through Amsterdam per token. Loads: first download 55–232 s
  per stage (1.0–1.9 GB from the coordinator), from the browser cache
  2.0–3.0 s. Application GPU allocations 1.07–2.04 GB per contributor; the
  NVIDIA driver does not report per-client memory in fdinfo, so
  driver-attributed GPU memory was not recorded there; PSS of each host's
  Chrome 0.83–1.69 GB. Reports `runs/experiments/DO-36/rented-20261106-session/`.
- 2026-11-06: Chat on the coordinator (`serve --chat`, page `/chat`): the
  contributors load once and keep one conversation in their KV caches and
  DeltaNet states; each turn prefills only its new tokens (the previous
  answer's last token, the chat template and the message), streams the
  answer, and stops at `<|im_end|>` or `--generate`; past `--capacity`
  positions, or on "New chat", every stage resets. Local check (680M):
  answers coherent across turns, 155 ms per token. On the rented machines
  (cached weights, ready 33 s after start): a 40-token answer, first token
  2.1 s, 615 ms per token. Also checked: commit 577825d (native kernels,
  CFFI): its 13 tests pass built with zig and cc; the 9 suite failures
  outside DO-36 predate today's work (they fail at 2a78f47 too).
- 2026-11-06 18:40 UTC: Chat session `chat3` with three rented contributors
  (stages A–C: RTX 3060 and RTX 3060 in Poland, Quadro RTX 4000 in Spain) and
  stage D left open for the user's own browser; all three ready within 40 s
  from cached weights. 19:23 UTC: the watchdogs deleted all five machines;
  `pods ls` shows 0. Cost about $1.08 at listed hourly prices (Vast.ai
  storage and bandwidth not included), within the $10 budget.

## Results

All three sessions ran on the same four contributors, in that order, with
code 611f007 (`qwen35_session.py serve` on the coordinator, `browser` on each
GPU host). The first session downloaded the weights; the later two loaded
them from the browser cache.

| Session | Prompt | First token after go | Per token, median / p95 | Tokens/s | Bytes per hop |
|---|---:|---:|---|---:|---:|
| chat | 24 tokens | 3.71 s | 568 / 635 ms | 1.76 | 1.2 MB |
| k1 | 1,024 tokens (experiment 48's report) | 8.07 s | 577 / 681 ms | 1.73 | 9.4 MB |
| k8 | 8,192 tokens (experiment 48's report) | 37.25 s | 580 / 698 ms | 1.69 | 68 MB |

Per stage in the 8,192-token session (A, B: RTX 3060, Poland; C: Quadro RTX
4000, Spain; D: RTX 3060, Sweden; the stage order follows arrival):

| Stage | Layers | Prefill compute | Decode step compute, median | Load (cached) | Application GPU memory, peak |
|---|---|---:|---:|---:|---:|
| A | embedding, 0–7 | 10.6 s | 64 ms | 3.0 s | 1.77 GB |
| B | 8–15 | 23.4 s | 63 ms | 2.9 s | 1.17 GB |
| C | 16–23 | 14.9 s | 93 ms | 2.4 s | 1.15 GB |
| D | 24–31, output | 18.0 s | 42 ms | 2.0 s | 2.04 GB |

- **Correctness.** Every device passed the self-test: 9 kernel cases on the
  model's own weights, worst 5.3e-7 of the largest output against a 1e-5
  limit, and the float16 conversions bit-identical to NumPy. The chat
  session's first 48 tokens equal the single-browser pipeline's, which equal
  the transformers reference's greedy choice (Log, "Pipeline under the
  policy"). The 8,192-token session's 128 tokens equal those of the four-browser
  run on the 680M (`omarchy-20261106-171043-session-local-8k`), so NVIDIA
  Ampere, NVIDIA Turing and AMD RDNA 2 chose the same tokens.
- **Latency.** About 300 ms of each token is compute across the four stages;
  the other 270 ms is the relay: five internet hops through Amsterdam per
  token (A → B → C → D through the coordinator, then the token back to A).
  Stage A waited a median 491 ms per step for the rest of the pipeline.
- **Startup.** First download 55–232 s per stage (1.0–1.9 GB from the
  coordinator); from the browser cache 2.0–3.0 s.
- **Memory.** Application GPU allocations 1.07–2.04 GB per contributor
  (the analytic estimate was 1.42 GB average before state and scratch);
  PSS of each host's Chrome 0.83–1.69 GB. On the 680M, the driver's own
  count was 60–70 MB above the application's (Log, item 3).

## What it taught

- A 9B model fits four 4 GB browser contributors with room to spare: the
  largest stage (D, with the 0.8 GB output projection) uses 2.04 GB, so a
  device with 3 GB free could serve any stage.
- Decode is bound by the relay, not the GPUs: 270 of 580 ms is hops through
  the coordinator. Direct peer channels, or a coordinator close to the
  contributors, are the next lever for time per token; faster kernels would
  save at most the 300 ms of compute.
- Prefill pipelines across machines: 8,192 tokens took 37 s to the first
  token, although the stages' prefill compute adds up to 67 s.
- Bugs and portability findings: WGSL's `pack2x16float` truncates on RADV
  and on the Quadro RTX 4000's driver but rounds to nearest even on the RTX
  3060, so the kernels convert to float16 in integer arithmetic; NVIDIA's
  Vulkan driver lacks `VK_EXT_headless_surface`, so headless Chrome falls
  back to SwiftShader (fixed with Chrome on Xvfb, ANGLE on Vulkan, and
  libEGL in the container); one 2.2 s submission reset the 680M's GPU ring,
  so submissions hold at most 64 token-layers; summing each 256-column block
  separately brought the register-tiled prefill kernel's error back to
  5.5e-7.

## Limits

- Rented data-center GPUs (8–12 GB) with a 4 GB budget by design, not
  friends' devices on home links; the 4 GB cap was not enforced by the
  driver. Two of the four contributors share a site and a public address.
- One session per prompt length, not the five repeats and three coding
  fixtures of the original protocol; the latency figures have no spread
  across sessions.
- All traffic went through the coordinator; direct peer routes, RTT and
  upload bandwidth were not measured.
- Tokens beyond the 48 checked against transformers were compared with
  another WebGPU run, not with the reference; the 8,192-token prompt has no
  transformers reference.
- NVIDIA's driver reports no per-client GPU memory in fdinfo, so on the
  rented hosts only the application's allocations and system PSS were
  recorded.
- Cost from listed hourly prices: the pool about $1.08, the probe at most
  $0.06; Vast.ai storage and bandwidth charges are not included.

## Reproduce

Workstation checks (AMD or any WebGPU GPU; the GGUF at the pinned revision):

```bash
.venv/bin/python -m unittest tests.test_qwen35_gguf tests.test_qwen35_session
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu kernels
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu pipeline
.venv/bin/python -m ygaraty.experiments.qwen35_session local --text public/research/048-factorization-protein-map-webgpu.md --prompt-tokens 8192 --generate 128
```

Rented contributors (about $0.40/h for the pool; watchdogs delete it):

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do36gpu --provider vast \
  --location SI,CZ,DE,PL,NL,FR,SE,AT,HU,RO,BG,ES,IT,BE,DK,FI,NO,GB --server any-8gb --workers 3 --worker-size any-8gb \
  --minutes 165 --max-hourly 0.8 --no-collect
python -m ygaraty.experiments.pods up do36coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 165 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do36gpu && python -m ygaraty.experiments.pods ship do36coord
# coordinator: bash /root/remote.sh qwen35-coordinator-setup
# each GPU host: bash /root/remote.sh qwen35-contributor-setup
# coordinator: bash /root/remote.sh qwen35-serve k8 --text public/research/048-factorization-protein-map-webgpu.md --prompt-tokens 8192 --generate 128 --timeout 2400
# each GPU host, with the join URL from /workspace/qwen35-k8/join_url.txt on the coordinator:
#   bash /root/remote.sh qwen35-browser k8-<host> '<join URL>'
# chat instead: qwen35-serve chat3 --chat --generate 512 --capacity 8320, then open /chat
python -m ygaraty.experiments.pods down do36gpu && python -m ygaraty.experiments.pods down do36coord && python -m ygaraty.experiments.pods ls
```
