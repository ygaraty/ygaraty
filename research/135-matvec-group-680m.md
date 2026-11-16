# Products that read the same input now share one dispatch: 29% fewer launches per token, tokens identical, no loss on the 680M

| | |
|---|---|
| **Conclusion** | Works: grouped products cut Qwen3.5-9B launches 518 → 366 per token, tokens identical, step unchanged (measured, 680M) |
| **Status** | executed |
| **ID** | WS-36 (experiment 135) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | med |
| **Date** | Added 2026-11-12; run 2026-11-13 |
| **Path** | Plan-level fusions in `executePlan` and kernel epilogues: RoPE + KV cache store; gate·up product with SwiGLU in its epilogue; the output projection and down projection adding the residual in their epilogue; router matmul + softmax + top-k in one kernel; arms fused vs `--no-fuse`; Qwen3.5-9B, tiny Kolibri, 680M |
| **Result** | Built narrower than planned: consecutive one-token products of the same input in one dispatch (DeltaNet qkv + gate + alpha + beta; attention q + k + v; ffn gate + up with SwiGLU fused), opt-in `--group-matmul`. Radeon 680M, B G G B B G, resident loop: tokens identical in every variant at 64 and 1,024 positions; dispatches per token 518 → 366 (−29%); resident step 89.5 → 89.25 ms at 64 positions (−0.3%) and 90.9 → 90.6 at 1,024. Gate: 1 and 2 pass, 3 within 2%: **Works**. Stays opt-in until an A100 run, where launches cost more (experiment 134). Free. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/matvec-group-680m-2026-11-13.json`; raw `runs/experiments/WS-36/omarchy-20261113-114351/`; code in this commit (`matvecGroupShader` and `Gpu.matmulGroup` in `webgpu_kernels.mts`, the peephole in `inference_plan.mts`); unit test in `tests/webgpu/inference-plan.test.mjs` |

## Question

Experiment 117 removed 150 of 1,388 dispatches and saved 10%, mostly host time. Each remaining small operation costs launch time on the GPU and encode time on the host. How much do the next fusions give once host time is hidden (WS-32)?

## Prediction

Dispatches −40%; GPU time −10–15%.

## Gate (frozen 2026-11-13, before the measured runs)

Scope as built (narrower than the Path): the plan executor's peephole (beside experiment 117's norm fusions) puts consecutive one-token matrix products of the same input in one dispatch (`matmulGroup`, opt-in `groupMatmul` / `--group-matmul`): per DeltaNet layer attn_qkv + attn_gate + ssm_alpha + ssm_beta, per attention layer attn_q + attn_k + attn_v, and per layer ffn_gate + ffn_up with silu_mul fused (each thread computes both rows' sums). Every row's sum is matvecC's, in the same order. RoPE with the cache store and the router top-k are not in this round.

Arms B (today's production flags: `--matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8`) and G (the same plus `--group-matmul`), `bench --contexts 64,1024 --resident`, order B G G B B G, Radeon 680M, Qwen3.5-9B Q4_K_M, 32 tokens per variant.

1. **Correctness:** every G run's generated tokens identical to B's at both lengths, in every variant (bitwise arithmetic, so no tolerance).
2. **Launches:** dispatches per token at least 25% fewer in G.
3. **Speed (680M, GPU-bound):** resident step = the resident variants' wall ms per token at 64 positions, median over each arm's 3 runs. **Worth it** if G is at least 8% shorter; **Works** (fewer launches, no loss on this GPU; stays opt-in until an A100 run measures the launch-bound case) if G is within 2% of B or better; **Rejected** if G is more than 2% longer. Failing 1 or 2: **Failed gate**.

## Protocol

Driver `runs/experiments/WS-36/<host>-<time>/drive.sh`; logs and reports per run there. The development runs before the freeze (`omarchy-*-try`: a first GLU layout 9% slower, then the current one on par, tokens identical) are not part of the gate.

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-13: Started at user request ("yes", after experiment 134 pointed at fewer, larger dispatches). Code: grouped matvec shader and executor peephole, unit test (46/46 in inference-plan). Development runs: 518 → 366 dispatches per token, tokens identical; first GLU layout (half a workgroup per product) 96 vs 88 GPU ms, rewritten as two sums per thread: 87.4 vs 88.0.
- 2026-11-13 11:43–12:01: measured runs B1 G1 G2 B2 B3 G3 under the watchdog (`--stall 600 --max-runtime 9000`), all exit 0, state finished. Commit 4bdbb3d plus this change, uncommitted at run time. The resident mode records no dispatch count (0 in the reports), so gate 2 was counted afterwards by one non-resident run per arm on the same code (`count-B`, `count-G`): a deterministic count, not a timing.

- 2026-11-13: measured on an A100 in experiment [136](136-matvec-group-a100.md) (DO-75): step −11.5%, tokens identical; now the Qwen3.5 default.

## Results

| Positions | B runs (resident variants) | G runs | B median | G median | G against B |
|---|---|---|---|---|---|
| 64 | 89.7 89.4 / 89.7 89.5 / 89.5 89.5 | 89.8 89.2 / 89.3 89.2 / 89.3 89.1 | 89.5 | 89.25 | −0.3% |
| 1,024 | 91.0 90.9 / 90.8 90.9 / 90.9 90.9 | 91.0 90.4 / 90.6 90.6 / 90.6 90.4 | 90.9 | 90.6 | −0.3% |

Dispatches per token (count runs): B 518, G 366 (−29.3%). GPU time per kernel at 64 positions (detail profile): B's Q4_K products 45.6 ms become G's GLU kernel 30.4 ms (gate, up, SwiGLU) plus 8.9 ms of remaining Q4_K products; the DeltaNet group takes 13.5 ms. GPU step 88.3 (B) and 87.5 ms (G).

Gate (computed by script): 1 passes (tokens identical in every variant, both lengths, and in the count runs); 2 passes (−29%); 3: −0.3%, within 2%: **Works**.

## What it taught

- Fewer launches change nothing on a GPU-bound 680M, as expected: the GPU time is the matrix bytes. The point is the A100, where experiment 134 found small, short dispatches that cannot fill the GPU, and about 6 ms per token outside the large products.
- A first GLU layout (half a workgroup per product, the other half idle on the other decoder branch) was 9% slower; one thread computing both rows' sums (each in matvecC's order) is on par and reads the input once. Divergent decoder branches inside a workgroup are costly.
- The WGSL helpers read one global weight buffer, so a kernel over several matrices needs a renamed copy of the decoder helpers per weight binding (`w0`, `w1`, …). It is generated, not hand-written, so each segment keeps the exact arithmetic.

## Limits

- 680M only; the A100 (launch-bound) gain is unmeasured and is the reason to keep it.
- Gate 2's dispatch count came from separate non-resident runs after the timed runs (the resident bench records none).
- Not done from the original plan: RoPE with the cache store, the router top-k (Kolibri), batches (the grouping applies to one token through the coalesced kernel only; batches and prefill keep separate dispatches).
- Needs a device with at least 8 storage buffers per stage for the 4-product DeltaNet group (it falls back to smaller groups otherwise).

## Reproduce

```bash
M=runs/models/qwen3.5-9b
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu bench --gguf $M/Qwen3.5-9B-Q4_K_M.gguf --config $M/config.json --tokenizer $M/tokenizer.json \
  --contexts 64,1024 --resident --matvec 2,4,8 --matvec-c 16,4 --q8-fast-up-to 8 [--group-matmul] --out OUT
# driver: runs/experiments/WS-36/omarchy-20261113-114351/drive.sh
```
