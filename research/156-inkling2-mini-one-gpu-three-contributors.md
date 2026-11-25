# Inkling 2 Mini runs on Ygaraty with the reference's tokens, on one rented A100 and as three WebGPU contributors split 0–5 | 6–17 | 18–23

| | |
|---|---|
| **Conclusion** | Works: Inkling 2 Mini gives the reference's greedy tokens on one A100 and on three contributors (measured, Vast.ai) |
| **Status** | executed |
| **ID** | DO-95 |
| **Category** | browser |
| **Where** | remote |
| **Priority** | med |
| **Date** | Added 2026-11-21; started 2026-11-21; finished 2026-11-22 |
| **Path** | New `inkling2` model family from ggml-org's `Inkling-2-Mini-IQ4_NL.gguf` (17.6 GB), text only. Arm A: one WebGPU contributor on one rented GPU. Arm B: three contributors, layers 0–5 + embedding, 6–17 (the looped core, all three passes local), 18–23 + output head |
| **Result** | Arm A (one A100 40 GB): math check within 3.1e-5 of NumPy on 5 prompts (bound 2e-3); 64 greedy tokens identical on all 5; 160 ms per token at 1k positions, 210 ms at 8k. Arm B (A4000, 4060 Ti, 3090; relay; `f32` wire): greedy tokens identical to arm A on the 3 prompts the chat API can express; 4.3, 8.1 and 6.3 GB allocated of 16 GB each; about 750–790 ms per token through the relay. Cost $0.96 + $0.81 |
| **Evidence** | `runs/experiments/DO-95/` (workstation): phase 0 `omarchy-20261122-003049`, phase 1 `omarchy-20261122-phase1-ref`, arm A `vast-do95a-20261122`, arm B `vast-do95b-20261122` |

## Question

Thinking Machines released Inkling 2 Mini on 2026-11-20 ([blog](https://thinkingmachines.ai/news/inkling-2-mini/), [model card](https://thinkingmachines.ai/model-card/inkling-2-mini/), Apache 2.0). It is a 26.4B-parameter mixture of experts with 3.4B active parameters per token. Can Ygaraty serve it with the reference's tokens, first on one GPU and then split over three contributors?

The model is unlike every family we support (see `ygaraty/backends/webgpu/checkpoints/families.py`). It also breaks two pipeline assumptions: `_stage_layout` in `distribution.py` requires contiguous layers visited once, and only one hidden vector crosses between stages.

Architecture, read from the GGUF header (`inkling2_loop`), not only from the blog:

- **Looped decoder.** There are 24 stored layers and 48 visits per token. Layers 0–5 run once, 6–17 run three times with shared weights, and 18–23 run once. Each visit keeps its own KV cache and convolution state. Core layers mix in the previous pass through rank-64 gain and forget controllers for Q, K, V and the FFN (`loop_*_ctrl_{down,gain,forget}`, `loop_msg_query`).
- **Attention residuals** ([arXiv 2603.15031](https://arxiv.org/abs/2603.15031)). In blocks of 6 layers, each layer's input is a learned mix of earlier outputs (`attn_res_query`, `ffn_res_query`, `output_res_query`), not a single running residual.
- **Attention.** It has 24 query heads, 8 KV heads and head size 128, with no RoPE. It uses learned relative-position logits (`attn_rel` from 2304 to 24×16, `attn_rel_bias` [16, 1024]), a per-token sink (`attn_sink_proj`), a sigmoid output gate (`attn_gate`), and learned Q/K norm clips. Queries are scaled by log length past 8192. Layers 5, 11, 17 and 23 attend globally, and the other 20 use a 1024-token window.
- **Short causal convolutions** with 4 taps, on K and V and on the attention and FFN outputs (`*_conv`, `*_post_conv`). The 4 global layers predict their taps per token (`attn_{k,v}_conv_proj`, from 2304 to 32) and normalize their Q/K/V inputs separately.
- **Mixture of experts in every layer.** There are 128 experts, top 8, with a biased router (`exp_probs_b`), weight scale 8 and fused gate and up weights. The activation is "SiTU" (`situ_beta 4`, `situ_linear_beta 25`). Final logits are softcapped at 30, and the vocabulary has 201,024 entries.
- **Weights in the IQ4_NL file.** Experts in layers 0–19 are `IQ4_NL`, a 32-weight, 18-byte block that is `Q4_0` with a 16-level lookup. Experts in layers 20–23 and all other matrices are `Q8_0`, and the rest is F32. Our matrix kernels do not decode `IQ4_NL` yet (`PACKED` in `families.py`).
- **Tokenizer and chat format.** The tokenizer is gpt2-style BPE with its own pre-tokenizer (`inkling2`) and 424k merges. The chat template uses role and content tokens, an effort system message (`Thinking effort level: 0.9` by default), and the end token `<|content_model_end_sampling|>` (200006).

## Prediction

Arm A works with greedy tokens identical to the reference, once a NumPy reference and the new kernels exist.

Arm B should keep the same tokens, because the split changes only where work runs, not its order. Decoding is about 48 visits × (8 experts ≈ 36 MB + attention ≈ 15 MB) ≈ 2.4 GB of weights read per token, similar to a 5B dense model at 4 bits. On one GPU that means tens of ms per token.

In arm B, the hand-off between stages grows from one hidden vector to the saved block outputs. That is still tens of KB per token when decoding, and more during prefill. Keeping the loop inside contributor 2 keeps decoding at two hops per token instead of six.

## Gate (frozen 2026-11-22, before any WebGPU run)

The proposed gate was revised after phase 0 showed that MoE routing near-ties make any two implementations disagree at some positions (see the Log, 2026-11-22). Frozen with the user's agreement:

- **Reference.** The NumPy reference `inkling2_gguf.py`, checked against llama.cpp `afe77e1` in phase 0: the 48 visit outputs agree to about 1% up to the first routing near-tie, and the last-position token agrees on all 5 prompts.
- **Math, both arms.** On the 5 prompts in `runs/experiments/DO-95/prompts/` (chat, code, tool, effort0, and the 2,761-token long prompt that crosses the 1024-token window), with the WebGPU path forced to use the reference's expert choices at every visit, last-position logits are within 2e-3 max absolute error of NumPy after the softcap.
- **End to end, both arms.** With the WebGPU path routing on its own, the first 64 greedy tokens on the same 5 prompts are identical to the NumPy reference's. A divergence is a failure unless it starts at a routing choice whose 8th/9th score gap is under 2e-3 in the reference, in which case it is logged and inspected by hand, and counts as a pass only if the routing math check above passes on that position.
- **Arm B is the same as arm A.** Its greedy tokens are identical to arm A's, with no contributor over its memory commitment.
- **Report, not gated.** Time per token and time to first token at 1k and 8k positions for both arms, and bytes per stage crossing in arm B.

## Protocol

**Phase 0, on the workstation (CPU, no GPU run).**

1. Read llama.cpp's `inkling2_loop` implementation to pin down the formulas the blog does not give:
   - router scoring (sigmoid or softmax);
   - SiTU;
   - the attention-residual mix, and why `attn_res_query` has shape [2304, 3];
   - the loop controllers;
   - the relative logits;
   - the dynamic convolution;
   - log scaling.
2. Write an `inkling2` family with `detect`, `check` and a geometry with 48 visits, plus a NumPy reference like `qwen3next_gguf.py`, the tokenizer, and the chat template.

**Phase 1, kernels.** Add:

- `IQ4_NL` decode;
- the attention variant (relative logits, sink, gate, clips, no RoPE);
- the convolutions with per-visit state;
- the residual mix;
- the loop controllers;
- SiTU.

The MoE path can reuse `kolibri_gpu`'s expert chunks and GPU-side routing.

**Arm A, one contributor.** One rented GPU with at least 24 GB, the same class experiment 113 used for Kolibri-1 (an A100 80 GB leaves headroom for long KV caches). The weights are 17.6 GB, and the KV cache is small because only 8 of the 48 visits grow past 1024 tokens.

**Arm B, three contributors.** Weights per stage, computed from the tensor list:

| Contributor | Layers | Weights |
|---|---|---|
| 1 | 0–5 + embedding | 4.1 GB |
| 2 | 6–17, all three passes | 7.4 GB |
| 3 | 18–23 + output head | 6.1 GB |

This needs plan and contract changes:

- allow a stage whose layers are visited repeatedly, with state kept per visit;
- carry the block outputs across the 5|6 and 17|18 boundaries (the core starts a new 6-layer block at 6, and 18 starts another);
- check that `layer_bytes` and the memory admission count per-visit KV and convolution state.

Run three rented GPUs with at least 12 GB each, ideally in two locations, so the hand-off size is felt. Hand-offs use the `f32` wire (decided 2026-11-22, see the Log): `f16` rounding of the block streams can flip routing near-ties, and arm B must give arm A's tokens.

**Out of scope here:** the DSpark draft model (`mtp.safetensors`, a later speculative-decoding experiment), image input (`mmproj`), the 1M-token context, and fine-tuning.

**Inputs:** `Inkling-2-Mini-IQ4_NL.gguf` (17.6 GB) and the llama.cpp commit used as the reference. Record both hashes. Rental budget and teardown deadline are still TBD; confirm them before provisioning.

## Reference formulas (phase 0)

These are read from llama.cpp afe77e1: `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `build_moe_ffn` in `src/llama-graph.cpp`, and `set_input_relative_pos` in `src/llama-kv-cache.cpp`. They replace the guesses in the Question.

- **Notation.** `rms(x)` is RMSNorm without a weight, eps 1e-6. `norm_w(x)` is `rms(x) * w`. `clip(x, s, T)` multiplies x by s (if given), then scales each row to an L2 norm of at most T: `x * min(1, T / (|x| + 1e-6))`.
- **Input.** `h = norm_w(embed[token], token_embd_norm)`. There is no BOS.
- **Depth mix.** `mix(blocks, cur, Q)`: stack `[blocks..., cur]` (n states) and apply `rms` to each. Then `scores = softmax_n(Q^T · states / sqrt(2304))` for each query column of Q. The output is the scores-weighted sum of the **normalized** states, with one output per column of Q.
- **Layer `il` at block position `ip`.** `ip == il` except in looped passes, where `ip` is the physical index.
  1. **Attention inputs.** Layer 0 uses `cur`. Other layers use `mix(blocks, cur, attn_res_query)`, with 1 query in sliding layers and 3 queries (Q, K, V) in full layers. In passes 2 and 3, each of Q/K/V goes through `control` with its own message (below).
  2. **Norms.** A sliding layer applies `attn_norm` to its input (or inputs, if messages made them differ). A full layer applies `attn_{q,k,v}_inp_norm` separately.
  3. **Projections.**
     - `q = Wq·xq`
     - `k = conv(Wk·xk, attn_k_conv, dyn = attn_k_conv_proj·xk if full)`
     - `v = conv(Wv·xv, attn_v_conv, dyn …)`
     - `r = attn_rel·xq` (24 heads × 16)
  4. **Per head.**
     - `q = clip(q, attn_q_clip[:, h], 22.627)`
     - `k = clip(k, -, 22.627)`
     - `v = rms(v)`
     - `r = clip(r, attn_rel_clip[:, h], 8)`
     - Full layers only: q and r are multiplied by `tau = 1 + 0.1·ln(max(1, (pos + 1)/8192))`.
  5. **Attention logit.** `q·k / sqrt(128) + r·rel_bias[:, d] − t_h`. Only `q·k` is scaled: llama.cpp passes the bias as the softmax mask, which `soft_max_ext` adds after scaling. The terms are:
     - `d` is the distance `pos_q − pos_k`, clamped to `extent`, with `extent = 1024` in both kinds of layer;
     - column `extent` of `rel_bias` is zero padding;
     - `t = attn_sink_proj·xq` is a per-head sink.
     
     The softmax includes one extra logit of 0 (the sink). Sliding layers mask keys 1024 or more positions back. Grouped-query attention uses 3 query heads per KV head.
  6. **Output.** `o = attn · sigmoid(attn_gate·xq)`, then `a = conv(attn_output·o, attn_post_conv)`.
  7. **Residual.** If `ip % 6 == 0`, push `cur` (the layer input) onto `blocks` and set `cur = a`: a new stream starts. Otherwise `cur = cur + a`.
  8. **FFN.** `f = mix(blocks, cur, ffn_res_query)`. In passes 2 and 3, apply `control` with the FFN message. Then `ffn_norm`.
  9. **Router.** `p = sigmoid(ffn_gate_inp·f)`. The top 8 are chosen on `p + exp_probs_b`, and their weights are `8 · p / (Σp_top + 1e-20)`, unbiased.
  10. **Experts.** Each expert computes `gate_up` (rows `[0, 1152)` are the gate, `[1152, 2304)` the up), then `y = 4·tanh(g/4)·sigmoid(g) · 25·tanh(u/25)`, then `ffn_down_exps`.
  11. **Final.** `cur = cur + conv(Σ w_e·y_e, ffn_post_conv)`.
- **conv.** Causal and depthwise, with 4 taps: `out[t] = Σ_k w[k]·x[t − 3 + k]`. With `dyn` set, it adds `Σ_lag dyn[lag, head]·x[t − lag]` over 8 KV heads × 4 taps, shared across each head's 128 channels. The result is `(x + out) / sqrt(5)`, or `/ sqrt(9)` with `dyn`. Each layer visit keeps the last 3 inputs at these channel offsets:

  | Channels | Holds |
  |---|---|
  | 0–1023 | K |
  | 1024–2047 | V |
  | 2048–4351 | attention output |
  | 4352–6655 | FFN output |

- **Loop.**
  - Pass 1 runs layers 6–17 normally.
  - Passes 2 and 3 restart from the output of layers 0–5, with `blocks` reset to their state after layer 5. That state is `[embedding stream]`.
  - The previous pass's bank is its block outputs after the first core block (its layer-11 stream) plus its output after layer 17.
  - Each core layer computes 4 messages: `mix(bank, bank_out, loop_msg_query)`.
  - `control(x, m) = x − forget·x + gain·m`, where `g = ctrl_down·[rms(x); rms(m)]` (rank 64), `gain = tanh(ctrl_gain·g)` and `forget = tanh(ctrl_forget·g)`.
  - The KV cache and convolution state are kept per visit: 48 slots.
- **Output.** `norm_w(mix(blocks, cur, output_res_query), output_norm)`, where `blocks` = `[embedding, prelude out, last pass's layer-11 stream, layer-17 out]`. Then `output·`, then `30·tanh(z/30)`, keeping the first 200,058 logits.
- **Tokenizer.** BPE with GPT-4o's pre-tokenizer regex (llama.cpp maps `inkling2` to `GPT4O`) and no BOS.

**Consequence for arm B.** The hand-off at the 5|6 boundary is `cur` plus `blocks = [embedding]`, so 2 vectors per token. At 17|18 it is `cur` plus `blocks = [embedding, prelude out, layer-11 stream]`, so 4 vectors (36 KB per token in f32). Contributor 2 must also keep the prelude output, so it can restart each pass.

## Log

- 2026-11-21: Added. Tensor layout and stage sizes come from parsing the first 32 MB of ggml-org's `Q8_0` and `IQ4_NL` GGUFs.
- 2026-11-21: Started phase 0 on the workstation (Omarchy, 16 cores, 28 GB RAM, CPU only) at ygaraty 12f33ea+dirty. Reference source: llama.cpp afe77e1 (`src/models/inkling2-loop.cpp`). Downloading `Inkling-2-Mini-IQ4_NL.gguf` to `~/Projects/ygaraty/models/inkling-2-mini/`, hash checked against Hugging Face's LFS sha256.
- 2026-11-22: GGUF downloaded, sha256 `cbdf3fa9…81af4` matches Hugging Face's LFS hash. NumPy reference `inkling2_gguf.py`, the admission-only `inkling2` family and `tests/test_inkling2.py` (6 synthetic checks) added. Full suite: 218 passed, 2 skipped. llama.cpp built for CPU; `runs/experiments/DO-95/tools/dump_logits` saves its logits for every position.
- 2026-11-22: The first phase 0 run failed at once: `/usr/bin/time` is not installed (`omarchy-20261122-001134`). The script was fixed and rerun under watch.py (stall 600 s, busy stall 3600 s, max 7200 s), with output in `omarchy-20261122-001219`. The run computes llama.cpp logits for the 5 prompts, then the NumPy reference in 3 processes.
- 2026-11-22: The phase 0 run finished (`omarchy-20261122-001219`), but the reference **failed**: top-1 agreement was 0–75% and the largest logit difference was 46. Layer-by-layer comparison against `llama-eval-callback` found the bug: the reference scaled the relative bias and sink by 1/sqrt(128), but llama.cpp passes them as the softmax mask, which is not scaled. After the fix, every one of the 48 visit outputs and the final normalized state match llama.cpp to within its 8-bit activation rounding on a 3-token prompt. Rerunning the 5 prompts with the reference only; the llama.cpp logits are reused.
- 2026-11-22: Phase 0 rerun with the fix (`omarchy-20261122-003049`, reusing the llama.cpp logits from `omarchy-20261122-001219`). **The last-position prediction matches on all 5 prompts** (`<|message_model|>`, 200001).

  | Prompt | Tokens | Top-1 agreement, all positions | Top-5 overlap | Largest logit difference |
  |---|---:|---:|---:|---:|
  | chat | 27 | 92.6% | 97.0% | 2.0 |
  | code | 30 | 90.0% | 94.7% | 9.8 |
  | tool | 109 | 94.5% | 90.3% | 9.0 |
  | effort0 | 18 | 94.4% | 96.7% | 2.0 |
  | long | 2,761 | 95.4% | 72.9% | 26.5 |

  The remaining differences come from routing near-ties, not a bug. On a 16-token prefix of the code prompt, all 48 visits pick the same experts as llama.cpp at the last token until visit 17. There the 8th and 9th experts' scores differ by only 0.0013, and llama.cpp's 8-bit activation rounding flips the choice. The next flips are also near-ties (gaps 0.0008, 0.0017, 0.0000). Large gaps never flip before the states have drifted. Before the first flip, every visit output matches to about 1%. The loop amplifies this: 48 routing decisions per token, and on long prompts the drift spreads through attention, which explains the lower top-5 overlap there.

  Consequence for the gate: no two implementations of this model will agree at every position. "Same top token everywhere against llama.cpp" cannot hold, and a fixed 2e-3 logit bound for WebGPU against NumPy is fragile wherever a near-tie flips. Proposed fix: compare WebGPU against NumPy with routing forced to the reference's choices (teacher-forced experts) under the 2e-3 bound, and separately require the same greedy tokens on the 5 prompts. To be frozen before arms A and B.
- 2026-11-22: Gate frozen as above, with the user's agreement. Phase 1 (WebGPU kernels) started.
- 2026-11-22: Phase 1 kernels written and checked on the workstation's Radeon 680M (Dawn `webgpu@0.6.2`, about 17 GB GPU-visible memory, so the experts of layers 0–5 and 18–23 are uploaded before each of their visits and those of the looped core stay resident):
  - `IQ4_NL` decoding in `webgpu_kernels.mts`, and `IQ4_NL` added to `PACKED`;
  - new `browser/inkling2_gpu.mts` (`Inkling2Stage`), one stage for the whole model, following the NumPy reference step by step. It has kernels for the depth mix, causal convolutions with per-visit history, Q/K/R clips with log scaling, attention with relative bias and sink (online softmax, window ring of 1024 + chunk rows), the loop controllers, the sigmoid router with top-k (or forced ids), routed expert products, and SiTU. Each visit keeps its own KV cache and convolution history, so prompts are prefilled in chunks and decoded a token at a time;
  - tools in `runs/experiments/DO-95/tools/`: `synth.py` (a tiny random model), `real_ref.py` (the GGUF's manifest plus NumPy's per-visit outputs, expert ids and 8th/9th score gaps), `gpu_check.mjs`, `greedy.mjs` and `greedy_check.py`; plus `tests/webgpu/browser/inkling2.test.mjs`. The browser suite passes (127 tests).

  Results (`omarchy-20261122-phase1-ref`), all on the real `IQ4_NL` GGUF except the first row:

  | Check | Result |
  |---|---|
  | Tiny random model, 2 seeds, prefill chunks of 1–13 tokens and token-by-token decode | Visits within 1e-5 relative, logits within 1.3e-5 |
  | effort0 (18 tokens) and chat (27), prefill | Last-position logits within 1.9e-5 and 1.6e-5; top token agrees at 18/18 and 27/27 positions; 0 routing differences; every one of the 48 visits within 1.6e-6 relative |
  | effort0, forced routing (the gate's math check) | Last-position logits within 1.9e-5 (bound 2e-3) |
  | effort0, prefill 10 then decode 8 | Same as prefill: logits within 3e-5, 0 routing differences |
  | Greedy, 64 tokens, effort0 and chat | Identical: NumPy's top token at every answer position, scoring prompt + answer in one pass, is the GPU's token |

  This is a local pre-check, not the gate: the gate runs on arm A's GPU, on all 5 prompts. The long (2,761-token) prompt, code and tool were not run here. With naive kernels on the iGPU, a 30-token prefill takes about 7 s and decoding 3.8 s per token, mostly uploading the streamed experts. Prefill of the long prompt would take hours with the current expert kernel, which does one workgroup per (row, token, expert) and no grouping by expert.
- 2026-11-22: **Prompt processing groups tokens by expert.** For runs of 8 or more tokens, a bucket kernel sorts the token slots by chosen expert on the GPU. A tiled kernel then takes 32 rows of one expert against 32 of its slots, so each weight row is decoded once per 32 slots instead of once per slot. Runs of 16 or more tokens are submitted once per visit, because a 256-token run as one submission tripped amdgpu's ~10 s job watchdog (device lost). Checks: the tiny model passes (prefill chunks of 4, 9 and 13, and decode); real-weight chat prefill gives logits within 2.2e-5, 27/27 top tokens, 0 routing differences, and visits within 2e-6. GPU time on the 680M (`tools/bench.mjs`, first tokens of the long prompt):

  | Prefill | Expert kernels, per slot | Expert kernels, grouped | Whole run, grouped |
  |---|---:|---:|---:|
  | 64 tokens | 4.95 s | 2.84 s | 3.9 s |
  | 256 tokens | not run (about 20 s at linear scaling) | 5.05 s | 9.8 s |

  The gain grows with run length, since each expert then gets more slots per tile. At 256 tokens the remaining time is the shared `Q8_0` matrix kernel (3.1 s) and attention (1.2 s).
- 2026-11-22: **Placement plan.** `families.loop_placement` (and `Inkling2.placement` on a GGUF header) chooses contiguous stages with the loop core 6–17 on one worker. Only boundaries outside 7–17 are allowed. It minimizes the fullest worker's memory, or its share of a per-worker budget, then the next fullest, then the vectors crossing between stages. Memory per stage = weights + embedding (first stage) or head (last stage) + per-visit KV and convolution state (core layers counted 3 times; sliding visits hold window + chunk rows) + one chunk of activations. A cut before layer c carries 1 + (blocks started before c) vectors per token. `Inkling2.check_splits` rejects boundaries inside the core, and `launch.py` uses the plan as the default split for local looped checkpoints. Output for the IQ4_NL file (4096-token context, 512-token chunks):

  | Workers | Splits | Memory per worker (GB) | Crossings (layer: vectors) |
  |---|---|---|---|
  | 1 | 0–23 | 18.7 | none |
  | 2 | 0–17, 18–23 | 12.5, 6.6 | 18: 4 |
  | 3 | 0–5, 6–17, 18–23 | 4.5, 8.3, 6.6 | 6: 2, 18: 4 |
  | 4 | 0–5, 6–17, 18–21, 22–23 | 4.5, 8.3, 3.8, 3.1 | 6: 2, 18: 4, 22: 4 |

  Three workers give arm B's split. Past three, the core's 8.3 GB worker sets the minimum memory per worker. Still missing for arm B: `Inkling2Stage` running a layer range from incoming streams, and multi-vector hand-offs in the plan, transport and `compile_distribution`, which still accepts only the four plan families.
- 2026-11-22: **Arm B pieces.**
  - `Inkling2Stage` runs a layer range [first, last). A later stage takes `[tokens][vectors][hidden]`: the block streams in order, then the current stream. It sends the same at its end (`boundaryVectors`: 2 before layer 6, 4 before 18). A stage holds the whole loop core or none of it.
  - `Inkling2ContributorStage` gives it the interface `qwen35_contributor.mjs` drives (`load(fetch)`, `run`, `x`, `hidden`, `next`, `plan`). `hidden` is vectors × 2304, so the existing transport and its size limits carry the wider hand-off unchanged. The contributor now selects this class for `inkling2` manifests.
  - Python: `inkling2_gguf.stage_manifest` writes a manifest and per-tensor files for each stage (4.11, 7.35 and 6.15 GB for 0–5, 6–17, 18–23), and `boundary_vectors`. `compile_distribution` has an `inkling2` branch: stage plans state the layers, the input and output streams, and their vectors, and transfers carry `vectors`. It rejects a split core and gaps.
  - Tests: `tests/test_inkling2.py` (13), browser suite (127).

  Checks (`tools/split_check.mjs`, stages chained in one process with the wire's rounding; `tools/adapter_check.mjs`, the adapter driven like the contributor):

  | Check | f32 wire | f16 wire |
  |---|---|---|
  | Tiny model, 3 and 5 stages, prefill and decode | Same as whole model (logits within 2e-5 of NumPy) | Logits within 0.018, every top token the same |
  | Tiny model, contributor adapter, token by token | 13/13 tokens | 13/13 tokens |
  | Real chat prefill, 0–5, 6–17, 18–23 | Same as arm A: logits within 6.1e-5 of NumPy, 27/27 top tokens | Logits off by up to 0.25; 26/27 top tokens |
  | Real greedy, 64 tokens, effort0 and chat | not run | Identical to arm A |

  f16 rounding of the block streams can flip routing near-ties, so it puts "arm B's tokens identical to arm A's" at risk on longer outputs. The f32 wire makes arm B exactly arm A at 36.9 KB per decoded token, against 18.4 KB with f16. Recommendation for arm B: wire `f32`. Not yet wired: bundle preparation (`prepare_session`, `selftest_fixture` and bundle export still build stages from model objects of the older families), and admission still reports "no WebGPU stage" (`INKLING2_STAGE`).
- 2026-11-22: Decided with the user: arm B uses the `f32` wire. Inkling 2 bundle preparation must set `wire="f32"` in its stage fixtures.
- 2026-11-22: **Inkling 2 runs through the normal bundle and session path.**
  - Checkpoint files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` and `special_tokens_map.json` from `thinkingmachines/Inkling-2-Mini`, saved in `models/inkling-2-mini/hf/` with sha256 in `hf.sha256` (`config.json` a7a34960…, `tokenizer.json` 7f332727…).
  - `prepare_session` builds Inkling 2 stages from GGUF headers (`prepare_inkling2_session`): per-tensor files, splits checked against the core, `wire="f32"`. The device self-test runs the generic matrix kernel on the first 256 rows of the first `Q8_0` and `IQ4_NL` tensors, at 1, 3, 21 and 64 tokens. `families.INKLING2_STAGE = "inkling2_gpu.mjs"`, so admission accepts the checkpoint (`ygaraty model inspect`: compatible, no findings). The stored-plan check accepts plans without task lists.
  - The contributor honors the fixture's wire (`f32` sends the float32 rows; `f16` is unchanged).
  - Chat: `inference_support.chat` has a `tagged` style, `{role token}<|content_text|>{content}<|end_message|>`, with `<|content_model_end_sampling|>` after assistant turns, the effort message before the first non-system message, and an empty generation prompt. `chat_template.chat_format` derives it from Inkling's own template and checks it against renderings. The serial chat path closes assistant turns with that token. It reproduces phase 0's hand-rendered chat prompt exactly (27 tokens).
  - Real bundle: `ygaraty model prepare --splits 0,6,18,24` on the IQ4_NL file → `models/bundles/inkling2-3w` (17.6 GB, 3 stages, validated, sha256 9e3557df…). It cannot be served here: the three stages need about 18 GB of GPU memory, and the 680M exposes about 17 GB.
  - End-to-end on this machine with a tiny random `inkling2_loop` GGUF that has the real tokenizer and template (`tools/tiny_gguf.py`; 6 layers, loop 2–3, `Q8_0` experts because gguf-py cannot quantize `IQ4_NL`): `ygaraty model prepare` with 3 stages, `ygaraty serve`, `ygaraty contribute --runtime node --num-workers 3` (Dawn), then `/v1/chat/completions`. All three passed the self-test and went ready. Answers, compared as decoded text with NumPy greedy on the same file:

    | Request | Prompt | Answer | Same as NumPy |
    |---|---:|---:|---|
    | One user message | 27 tokens | 12 tokens | Yes |
    | System + 3 turns (two 64-token prefill chunks) | 110 tokens | 80 tokens | Yes |

  Known gaps: the answer text includes Inkling's thinking block (special tokens are dropped when decoding, the thinking text is not), and pool-mode histories keep that text.
- 2026-11-22: **Arm A started.** User's budget: $10 ("try to be economic"); teardown: a 90-minute watchdog deletes the machine (about $0.75 at most). Everything was prepared on the workstation first, so the GPU time is only setup and runs:
  - NumPy references for code (30 tokens, 866 s), tool (109, 1,249 s) and long (2,761, 1,637 s) in `omarchy-20261122-phase1-ref`;
  - compact gate inputs per prompt (`tools/pack_gate.py`: last-row logits, every row's argmax, routing ids and gaps), and timing prompts of 1,024 and 8,192 tokens (the long prompt, repeated);
  - a 4.8 MB kit (`tools/make_kit.sh`: the three browser modules, `gpu_check.mjs`, `greedy.mjs`, `armA.sh`), copied over SSH, not cloned, because the changes are uncommitted. The machine downloads the GGUF from Hugging Face and checks its sha256.
  
  `armA.sh` runs per prompt: the math check (forced routing, last-row logits ≤ 2e-3), natural routing (flips reported), and 64 greedy tokens. Then decode timing after 1k and 8k prompts. The greedy answers are checked afterwards on the workstation (`greedy_check.py`, teacher-forced NumPy). It was rehearsed from the unpacked kit on the 680M (chat, forced: 2.2e-5).
- 2026-11-22: **Arm A ran** on Vast.ai instance 61977176 (`do95a`): A100 SXM4 40 GB, US, $0.515/h, driver 595.91.07, Dawn adapter `nvidia | ampere | nvidia-a100-sxm4-40gb`. 1.86 h, about **$0.96** of the $10 budget. The watchdog deadline went from 90 to 150 minutes during setup, still within budget. The machine was deleted by its recorded ID, and Vast confirms it is gone. Results in `runs/experiments/DO-95/vast-do95a-20261122/`.
  - Setup took most of the time. This host's network was poor: apt about 8 minutes, a single Hugging Face connection about 0.2–1 MB/s, about 23 MB/s with 16 parallel ranges. The first parallel download had 10 of 32 ranges corrupt; they were found by per-part sha256 against the workstation's copy, re-fetched, and the file's sha256 then matched. NVIDIA's Vulkan driver needed `libxext6` and `libx11-6`: without them Dawn fell back to `llvmpipe`. `armA.sh` now installs them and sets `XDG_RUNTIME_DIR`. Natural-routing runs were skipped to save time (not gated).
  - **Math check** (forced routing, last-position logits against NumPy, bound 2e-3):

    | Prompt | Tokens | Last-row max abs error | Run time (s) |
    |---|---:|---:|---:|
    | effort0 | 18 | 3.0e-5 | 1.7 |
    | chat | 27 | 2.4e-5 | 1.4 |
    | code | 30 | 2.6e-5 | 1.3 |
    | tool | 109 | 2.9e-5 | 1.7 |
    | long | 2,761 | 3.1e-5 | 15.2 (6 chunks of 512; the last row of each chunk agreed on its top token, 6/6) |

  - **Timing** (report, not gated; naive decode kernels, everything resident; load 37–39 s per process from page cache):

    | Prompt | Prefill | Decode per token |
    |---|---:|---:|
    | 1,024 tokens | 5.0 s | 160 ms |
    | 8,192 tokens | 43.8 s | 210 ms |

  - **Greedy** (64 tokens): effort0 and chat are identical to the 680M's answers, which were already checked against NumPy. code, tool and long are being checked by teacher-forced NumPy on the workstation.
- 2026-11-22: **Arm A passes the frozen gate.** End-to-end check: the first 64 greedy tokens are identical to the NumPy reference on all 5 prompts. effort0 and chat match the 680M answers already checked against NumPy; code (30-token prompt), tool (109) and long (2,761) were scored by teacher-forced NumPy over prompt + answer (`greedy_check.py`: identical, so no tie inspection was needed). The math check passes on all 5 (largest error 3.1e-5 against the 2e-3 bound). DO-95 stays ongoing until arm B (three contributors, 0–5 | 6–17 | 18–23, `f32` wire).
- 2026-11-22: **Arm B started.** User's budget: $8. Three Vast.ai machines picked for measured bandwidth (arm A's host lost most of its time to a slow network): coordinator plus one contributor on an RTX A4000 16 GB (Delaware, $0.109/h); contributors on an RTX 4060 Ti 16 GB and an RTX 3090 24 GB (Utah, $0.189 and $0.203/h). Each has its own 150-minute watchdog (about $1.25 at most in all). The coordinator machine installs the uncommitted working tree from a tarball and downloads the GGUF in 32 parts, each checked against the workstation's per-part sha256. It prepares the 0,6,18,24 bundle (context 8,448, `f32` wire), then serves it. The workers run `native_contributor.mjs` (Node + Dawn) on its link. Requests go through `/v1/chat/completions` (`armB-kit/client.py`).
  - Prompts: the chat API renders chat, code and long exactly as arm A's token files (checked: 27, 30 and 2,761 tokens). effort0 (effort 0) and tool (tool declarations) cannot be expressed through the chat API, so arm B's comparison with arm A covers chat, code and long. This narrows the "arm B = arm A" check to 3 of the 5 prompts; recorded here before the run.
  - Timing at 1k and 8k positions: user messages of 1,013 and 8,182 prompt tokens. The time to first token is a request with `max_tokens` 1; ms per token is (33-token request - 1-token request) / 32.
- 2026-11-22: **Arm B ran** on three Vast.ai machines: `do95b1` (61999915, RTX A4000 16 GB, Delaware, $0.126/h: coordinator plus one contributor), `do95b2` (61999914, RTX 4060 Ti 16 GB, Utah, $0.195/h) and `do95b3` (61999916, RTX 3090 24 GB, Utah, $0.211/h). 1.52 h each, about **$0.81** of the $8 budget. All three were deleted by their recorded IDs, and Vast confirms they are gone. Results in `runs/experiments/DO-95/vast-do95b-20261122/`.
  - Setup was fast on these hosts. The coordinator downloaded the GGUF in 5 minutes, all 32 parts passing their hash on the first try, and prepared the bundle in 3.5 minutes. Contributors joined over plain HTTP links with the relay and passed the self-test (worst kernel 1.8e-7, `IQ4_NL` and `Q8_0` cases). Loading took 53 s for stage A, on the coordinator's machine, and 190–340 s for stages B and C, which download 6.2–7.4 GB from the coordinator across the internet.
  - Memory: allocated 4.27, 8.08 and 6.31 GB for stages A, B and C, against a 16 GB commitment each (`--memory-gb 16`). No contributor over its commitment.
  - **Arm B = arm A**: the 64-token greedy answers to chat (27-token prompt), code (30) and long (2,761) are identical to arm A's, compared as decoded text. The answers read as expected: "The capital of Portugal is Lisbon", an iterative Fibonacci plan, and "7 marbles, shelf 1" for the long fact-retrieval prompt.
  - Two coordinator problems found and handled:
    1. **Relay mailbox overflow.** The long prompt's prefill dropped stage A with `429 Stage mailbox capacity reached`. The relay mailbox was a fixed 32 MiB. Inkling's `f32` hand-offs are 1.2–2.4 MB per 64-token call, and stage A finishes its prefill before stage B reads. Fixed in `_legacy/qwen_protocol.py` (`mailbox_bytes`): each mailbox now holds a whole-context prefill of its stage's largest inputs (158 and 315 MiB here, never below 32 MiB). The relay still has no backpressure; flow control would be the real fix.
    2. **The 120 s request deadline** (`session.py`, `request_timeout`). An 8,182-token prompt plus 33 decoded tokens exceeds it through three relayed stages: 503 "Request deadline exceeded". Not changed (serving policy). The 8k decode speed was measured from a 1-token and an 8-token request instead. A history-mode follow-up does not reuse the cached prefix, so it re-prefills and also times out.
  - Operations noise, mine: a restart with the old state directory resumed the old session (contributors got `reset`, not `go`), and one stale coordinator kept port 8000. Each was fixed with a clean restart. No effect on results.
  - **Timing** (report, not gated; relay through the coordinator in Delaware, two stages in Utah; simple decode kernels):

    | Prompt | Time to first token | Per generated token |
    |---|---:|---:|
    | 27–30 tokens (64 generated) | — | about 775 ms (49.6 s per 64-token answer) |
    | 1,013 tokens | 14.2 s | 752 ms |
    | 2,761 tokens (64 generated) | — | 105.6 s for the request |
    | 8,182 tokens | 113.7 s | about 790 ms (1- and 8-token requests) |

    Arm A decoded at 160 ms per token on one A100. Arm B's extra ~600 ms per token is the relay: three hops through the coordinator per token, two of them Delaware–Utah, by HTTP polling.
  - Bytes per stage crossing (`f32` wire): 2 × 2,304 × 4 = 18.4 KB per token before layer 6, and 4 × 2,304 × 4 = 36.9 KB per token before layer 18. That is 55 KB per decoded token across both, and 152 MB for an 8k-token prefill.

## Results against the frozen gate

| Gate item | Result |
|---|---|
| Math, arm A (forced routing, last-position logits within 2e-3 of NumPy, 5 prompts) | **Pass**: 2.4e-5 to 3.1e-5 |
| End to end, arm A (64 greedy tokens identical to NumPy, 5 prompts) | **Pass**: 5 of 5 identical; no near-tie inspection needed |
| Arm B = arm A (greedy tokens identical) | **Pass on 3 prompts**: chat, code, long identical. effort0 and tool cannot be expressed through the chat API (recorded before the run) |
| Arm B memory (no contributor over its commitment) | **Pass**: 4.27, 8.08, 6.31 GB of 16 GB |
| Report: time per token, time to first token at 1k and 8k, bytes per crossing | Arm A 160 / 210 ms per token, 5.0 / 43.8 s prefill. Arm B 752 / ~790 ms per token, 14.2 / 113.7 s first token. Crossings 18.4 and 36.9 KB per token |

The math check on arm B was not repeated: arm B runs the same stage code. Its split was checked against NumPy in one process (logits within 6.1e-5, `split_check.mjs`) and against arm A's tokens here.

## Limits

- Prompts: 5 short-to-medium prompts on arm A (up to 2,761 tokens); 3 on arm B through the chat API. 64 tokens per answer. One run per arm, no repeats.
- One GPU model per arm position (A100 40 GB; A4000, RTX 4060 Ti, RTX 3090). Arm B mixed three NVIDIA architectures and still matched arm A, but other vendors (AMD, Apple) were only exercised on the workstation's 680M (phase 1).
- Speed is not optimized: decode uses one workgroup per row and expert (about 160 ms per token on an A100, against a bandwidth floor of a few ms). Arm B's time is dominated by the relay (three HTTP-polled hops per token).
- The relay's mailbox fix bounds memory by context, not by flow control. The coordinator's 120 s request deadline rejects 8k-token prompts with more than a few generated tokens on this topology.
- Chat answers include Inkling's thinking text: special tokens are dropped when decoding, the thinking block is not.
- Out of scope: the DSpark draft model (`mtp.safetensors`), images, contexts past 8,448 positions, `Q8_0` and BF16 files.

## Reproduce

Workstation (CPU, NumPy reference; the GGUF and `hf/` files as in the Log):

```bash
python runs/experiments/DO-95/tools/real_ref.py MODEL.gguf runs/experiments/DO-95/omarchy-20261122-003049/<prompt> OUT/<prompt>
python runs/experiments/DO-95/tools/pack_gate.py OUT/<prompt> runs/experiments/DO-95/armA-kit/<prompt>
runs/experiments/DO-95/tools/make_kit.sh kit-armA.tgz          # arm A kit
```

Arm A machine (one GPU with at least 24 GB, NVIDIA_DRIVER_CAPABILITIES=all): unpack the kit in `/workspace/kit`, then `bash runs/experiments/DO-95/tools/armA.sh setup` and `bash runs/experiments/DO-95/tools/armA.sh`. Back on the workstation, `real_ref.py` on each `<prompt>.greedy.tokens` and `greedy_check.py`.

Arm B: `runs/experiments/DO-95/armB-kit/armB.sh` (`common` on all machines, `coord` and `serve` on the coordinator, `contribute URL` on each contributor), then `client.py URL OWNER_KEY OUT`. Compare each answer with arm A's decoded tokens.

Local, no GPU rental: `tools/synth.py` + `tools/gpu_check.mjs` (tiny random model), `tools/split_check.mjs` and `tools/adapter_check.mjs` (stages and the contributor adapter), `tools/tiny_gguf.py` + `ygaraty model prepare/serve/contribute` (the whole pipeline on a tiny GGUF).
